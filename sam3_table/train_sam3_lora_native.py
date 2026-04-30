
#!/usr/bin/env python3
"""
SAM3 LoRA Training Script

Validation Strategy (Following SAM3):
  - During training: Only compute validation LOSS (fast, no metrics)
  - After training: Run validate_sam3_lora.py for full metrics (mAP, cgF1) with NMS

This approach significantly speeds up training by avoiding expensive metric computation
during each epoch, while still monitoring overfitting via validation loss.

Multi-GPU Training:
  Single GPU:
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu

  Multi-GPU with specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu
"""

import os
import argparse
import json
import signal
import urllib.request
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import contextlib

# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from sam3_table.lora_layers import LoRAConfig as LoRALayerConfig, LoRALayer, apply_lora_to_model, save_lora_weights, count_parameters
from sam3_table.training_config import SAM3LoRAConfig, DatasetSplit
from sam3_table.coco_schema import COCODataset, RLESegmentation

from torchvision.transforms import v2
import pycocotools.mask as mask_utils  # Required for RLE mask decoding in COCO dataset
from sam3.train.masks_ops import rle_encode  # For encoding masks to RLE format

# Note: Evaluation modules (mAP, cgF1, NMS) are in validate_sam3_lora.py
# Training only computes validation loss, following SAM3's approach


BPE_VOCAB_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
BPE_CACHE_PATH = Path("/tmp/bpe_simple_vocab_16e6.txt.gz")

CHECKPOINT_NAMES = ("checkpoint_epoch.pt", "checkpoint_best.pt", "checkpoint_signal.pt")
CKPT_PHASE_TRAIN_IN_EPOCH = "train_in_epoch"
CKPT_PHASE_PRE_END_EPOCH_VALIDATION = "pre_end_epoch_validation"
CKPT_PHASE_EPOCH_COMPLETE = "epoch_complete"


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """Get the number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def resolve_bpe_vocab_path() -> str:
    """Return a valid local path to the CLIP BPE vocab required by SAM3."""
    env_path = os.environ.get("SAM3_BPE_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    if BPE_CACHE_PATH.exists():
        return str(BPE_CACHE_PATH)

    BPE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print_rank0(f"Downloading SAM3 BPE vocab to {BPE_CACHE_PATH}...")
    urllib.request.urlretrieve(BPE_VOCAB_URL, BPE_CACHE_PATH)
    return str(BPE_CACHE_PATH)


class COCOSegmentDataset(Dataset):
    """Dataset class for COCO format segmentation data."""

    def __init__(
        self,
        coco_dataset: COCODataset,
        image_dir: Path | None = None,
        sample_percent: float = 100.0,
        seed: int = 42,
    ):
        self.coco_data = coco_dataset
        self.image_dir = image_dir

        # Build index: image_id -> image info
        self.images = {img.id: img for img in self.coco_data.images}
        self.image_ids = sorted(self.images.keys())

        if sample_percent < 100.0:
            import random
            rng = random.Random(seed)
            k = max(1, int(len(self.image_ids) * sample_percent / 100.0))
            # Use a stable rank ordering so nested ``sample_percent`` values
            # produce nested subsets: the top-k images for k=10 are a strict
            # subset of the top-k for k=30 when ``seed`` is unchanged. This
            # matters for staged sweeps where each rung trains/validates on a
            # progressively larger fraction of the dataset; without nesting,
            # consecutive stages mostly throw away the previous stage's
            # samples (``random.Random(seed).sample`` is independent in k).
            ranked_ids = sorted(self.image_ids, key=lambda image_id: rng.random())
            self.image_ids = sorted(ranked_ids[:k])

        # Build index: image_id -> list of annotations
        self.img_to_anns: dict[int, list] = {}
        for ann in self.coco_data.annotations:
            if ann.image_id not in self.img_to_anns:
                self.img_to_anns[ann.image_id] = []
            self.img_to_anns[ann.image_id].append(ann)

        # Load categories
        self.categories = {cat.id: cat.name for cat in self.coco_data.categories}
        if self.image_dir is not None:
            print(f"Loaded COCO dataset from {self.image_dir}")
        else:
            print("Loaded COCO dataset from passed object")
        total_images = len(self.images)
        used_images = len(self.image_ids)
        print(f"  Images: {used_images}/{total_images} ({sample_percent:.1f}%)")
        print(f"  Annotations: {len(self.coco_data.annotations)}")
        print(f"  Categories: {self.categories}")

        self.resolution = 1008
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_ids)

    @classmethod
    def from_split_config(
        cls,
        split_config: DatasetSplit,
        sample_percent: float = 100.0,
        seed: int = 42,
    ) -> "COCOSegmentDataset":
        ann_file = split_config.annotation_file
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")
        return cls(
            COCODataset.from_json(ann_file),
            image_dir=split_config.image_dir,
            sample_percent=sample_percent,
            seed=seed,
        )

    def _resolve_image_path(self, file_name: str) -> Path:
        image_path = Path(file_name)
        if image_path.is_absolute():
            return image_path
        if self.image_dir is not None:
            return self.image_dir / image_path
        return Path("/data") / image_path

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self._resolve_image_path(img_info.file_name)
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Resize image
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)

        # Transform to tensor
        image_tensor = self.transform(pil_image)

        # Get annotations for this image
        annotations = self.img_to_anns.get(img_id, [])

        objects = []
        object_class_names = []

        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        for i, ann in enumerate(annotations):
            # Get class name from category_id
            class_name = self.categories.get(ann.category_id, "object")
            object_class_names.append(class_name)

            # Convert from COCO [x, y, w, h] to normalized [cx, cy, w, h] (CxCyWH)
            # SAM3 internally expects boxes in CxCyWH format normalized to [0, 1]
            x, y, w, h = ann.bbox
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Scale to resolution and normalize to [0, 1]
            box_tensor = torch.tensor([
                cx * scale_w / self.resolution,
                cy * scale_h / self.resolution,
                w * scale_w / self.resolution,
                h * scale_h / self.resolution,
            ], dtype=torch.float32)

            # Handle segmentation mask (polygon or RLE format)
            segment = None
            segmentation = ann.segmentation

            if segmentation is not None:
                try:
                    if isinstance(segmentation, RLESegmentation):
                        rle_dict = {"counts": segmentation.counts, "size": segmentation.size}
                        if isinstance(segmentation.counts, list):
                            rle_dict = mask_utils.frPyObjects(rle_dict, segmentation.size[0], segmentation.size[1])
                        mask_np = mask_utils.decode(rle_dict)
                    else:
                        # Polygon format: [[x1, y1, x2, y2, ...], ...]
                        # Convert polygon to RLE, then decode
                        rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                        rle = mask_utils.merge(rles)
                        mask_np = mask_utils.decode(rle)

                    # Resize mask to model resolution
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_t = torch.nn.functional.interpolate(
                        mask_t,
                        size=(self.resolution, self.resolution),
                        mode="nearest"
                    )
                    segment = mask_t.squeeze() > 0.5  # [1008, 1008] boolean tensor

                except Exception as e:
                    print(f"Warning: Error processing mask for image {img_id}, ann {i}: {e}")
                    segment = None

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2] * box_tensor[3]).item(),
                object_id=i,
                segment=segment
            )
            objects.append(obj)

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )

        # Construct Queries - one per unique category
        # Each query maps to only the objects of that category
        from collections import defaultdict

        # Group object IDs by their class name
        class_to_object_ids = defaultdict(list)
        for obj, class_name in zip(objects, object_class_names):
            class_to_object_ids[class_name.lower()].append(obj.object_id)

        # Create one query per category
        queries = []
        if len(class_to_object_ids) > 0:
            for query_text, obj_ids in class_to_object_ids.items():
                query = FindQueryLoaded(
                    query_text=query_text,
                    image_id=0,
                    object_ids_output=obj_ids,
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=0,
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1
                    )
                )
                queries.append(query)
        else:
            # No annotations: create a single generic query
            query = FindQueryLoaded(
                query_text="object",
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=img_id,
                    original_image_id=img_id,
                    original_category_id=0,
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=-1
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_image]
        )


def merge_overlapping_masks(binary_masks, scores, boxes, iou_threshold=0.3):
    """
    Merge overlapping masks that likely represent the same object.

    Args:
        binary_masks: Binary masks [N, H, W]
        scores: Confidence scores [N]
        boxes: Bounding boxes [N, 4]
        iou_threshold: IoU threshold for merging (default: 0.3)

    Returns:
        Tuple of (merged_masks, merged_scores, merged_boxes)
    """
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    # Sort by score (highest first)
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    boxes = boxes[sorted_indices]
    binary_masks = binary_masks.to(dtype=torch.bool).cpu()
    encoded_masks = [
        mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))
        for mask in binary_masks
    ]

    merged_masks = []
    merged_scores = []
    merged_boxes = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_score = scores[i].item()
        current_box = boxes[i]
        current_rle = encoded_masks[i]
        used[i] = True

        # Find overlapping masks and merge them
        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            # Compute mask IoU using pycocotools implementation.
            iou = float(mask_utils.iou([current_rle], [encoded_masks[j]], [0])[0][0])

            # If overlaps significantly, merge it
            if iou > iou_threshold:
                current_mask = current_mask | binary_masks[j]
                current_rle = mask_utils.encode(np.asfortranarray(current_mask.numpy().astype(np.uint8)))
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device, dtype=scores.dtype)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


def convert_predictions_to_coco_format(predictions_list, image_ids, resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format for evaluation.

    OPTIMIZATION: Keep masks at native model output resolution (288×288)
    GT is downsampled to match, so no upsampling needed!

    Args:
        predictions_list: List of prediction dictionaries from the model
        image_ids: List of image IDs corresponding to predictions
        resolution: Mask resolution for evaluation (default: 288, model's native output)
        score_threshold: Minimum score threshold for predictions
        merge_overlaps: Whether to merge overlapping predictions (default: True)
        iou_threshold: IoU threshold for merging overlaps (default: 0.3)
        debug: Print debug information

    Returns:
        List of prediction dictionaries in COCO format
    """
    coco_predictions = []
    pred_id = 0

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Extract predictions
        logits = preds['pred_logits']  # [num_queries, 1]
        boxes = preds['pred_boxes']    # [num_queries, 4]
        masks = preds['pred_masks']    # [num_queries, H, W]

        scores = torch.sigmoid(logits).squeeze(-1)  # [num_queries]

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:  # Debug first image only
            print(f"  Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")

        # Convert masks to binary (apply sigmoid first, then threshold)
        binary_masks = (torch.sigmoid(masks) > 0.5).cpu()

        # Merge overlapping predictions to avoid over-segmentation penalty
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"  Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        # Encode masks to RLE (at native resolution - much faster!)
        if len(binary_masks) > 0:
            # Check if masks have content
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"  Mask shape: {binary_masks.shape}")
                print(f"  Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [cx, cy, w, h] to [x, y, w, h] in pixel coordinates
                cx, cy, w, h = box
                x = (cx - w/2) * resolution
                y = (cy - h/2) * resolution
                w = w * resolution
                h = h * resolution

                coco_predictions.append({
                    'image_id': int(img_id),
                    'category_id': 1,  # Single category for instance segmentation
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                })
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset(dataset, image_ids=None, mask_resolution=288):
    """
    Create COCO ground truth dictionary from SimpleSAM3Dataset.

    OPTIMIZATION: Downsample GT masks to match prediction resolution (288×288)
    instead of upsampling predictions to 1008×1008. Much faster!

    Args:
        dataset: SimpleSAM3Dataset instance
        image_ids: Optional list of specific image IDs to include
        mask_resolution: Resolution to downsample masks to (default: 288 to match model output)

    Returns:
        Dictionary in COCO format
    """
    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    # Scale factor for boxes (masks will be at mask_resolution, boxes scaled accordingly)
    scale_factor = mask_resolution / dataset.resolution

    for idx in indices:
        # Add image entry at mask resolution
        coco_gt['images'].append({
            'id': int(idx),
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True  # Required for cgF1 evaluation
        })

        # Get datapoint
        datapoint = dataset[idx]

        # Add annotations
        for obj in datapoint.images[0].objects:
            # Convert normalized CxCyWH box to COCO [x, y, w, h] at mask_resolution
            cx, cy, bw, bh = (obj.bbox * mask_resolution).tolist()
            x, y, w, h = cx - bw / 2, cy - bh / 2, bw, bh

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            # Add segmentation if available - downsample to mask_resolution
            if obj.segment is not None:
                # Downsample mask from 1008×1008 to mask_resolution×mask_resolution
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    return coco_gt


def convert_predictions_to_coco_format_original_res(predictions_list, image_ids, dataset, model_resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format at ORIGINAL image resolution.

    This matches the inference approach (infer_sam.py) where:
    1. Masks are upsampled from 288x288 to original image size
    2. Boxes are scaled to original image size
    3. Evaluation happens at original resolution

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs (indices into dataset)
        dataset: Dataset to get original image sizes
        model_resolution: Model output resolution (default: 288)
        score_threshold: Confidence threshold
        merge_overlaps: Whether to merge overlapping predictions
        iou_threshold: IoU threshold for merging
        debug: Print debug info
    """
    coco_predictions = []
    pred_id = 0

    if debug:
        print(f"\n[DEBUG] Converting {len(predictions_list)} predictions to COCO format (ORIGINAL RESOLUTION)...")
        if merge_overlaps:
            print(f"[DEBUG] Overlapping segment merging ENABLED (IoU threshold={iou_threshold})")

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Get original image size from dataset
        datapoint = dataset[img_id]
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        logits = preds['pred_logits']
        boxes = preds['pred_boxes']
        masks = preds['pred_masks']  # [N, 288, 288]

        scores = torch.sigmoid(logits).squeeze(-1)

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:
            print(f"[DEBUG] Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")
            if len(scores) > 0:
                print(f"[DEBUG]   Original size: {orig_w}x{orig_h}")
                print(f"[DEBUG]   Filtered scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        if len(masks) == 0:
            continue

        # Upsample masks from 288x288 to original resolution (like infer_sam.py)
        # Process on GPU then immediately move to CPU to save memory
        masks_sigmoid = torch.sigmoid(masks)  # [N, 288, 288]
        masks_upsampled = torch.nn.functional.interpolate(
            masks_sigmoid.unsqueeze(1).float(),  # [N, 1, 288, 288]
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, orig_h, orig_w]

        binary_masks = (masks_upsampled > 0.5).cpu()

        # Free temporary tensors after upsampling.
        del masks_sigmoid, masks_upsampled

        # Merge overlapping predictions
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        if len(binary_masks) > 0:
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Upsampled mask shape: {binary_masks.shape}")
                print(f"[DEBUG]   Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [0,1] to original image coordinates
                cx, cy, w_norm, h_norm = box
                x = (cx - w_norm/2) * orig_w
                y = (cy - h_norm/2) * orig_h
                w = w_norm * orig_w
                h = h_norm * orig_h

                # Clamp coordinates to image bounds
                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = max(0, min(w, orig_w - x))
                h = max(0, min(h, orig_h - y))

                # Skip if box is too small after clamping
                if w < 1 or h < 1:
                    continue

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                if debug and img_id == image_ids[0] and idx == 0:
                    print(f"[DEBUG]   First prediction bbox (at {orig_w}x{orig_h}): {pred_dict['bbox']}")

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset_original_res(dataset, image_ids=None, debug=False):
    """
    Create COCO ground truth dictionary from dataset at ORIGINAL resolution.

    This matches the inference approach (infer_sam.py) where GT is kept
    at original image size for evaluation.

    Args:
        dataset: Dataset with images and annotations
        image_ids: List of image IDs to include (None = all)
        debug: Print debug info
    """
    if debug:
        print(f"\n[DEBUG] Creating COCO ground truth (ORIGINAL RESOLUTION)...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in indices:
        datapoint = dataset[idx]

        # Get original image size
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        coco_gt['images'].append({
            'id': int(idx),
            'width': orig_w,
            'height': orig_h,
            'is_instance_exhaustive': True
        })

        for obj in datapoint.images[0].objects:
            # Convert normalized CxCyWH box to COCO [x, y, w, h] at original size
            cx, cy, bw, bh = obj.bbox.tolist()
            w = bw * orig_w
            h = bh * orig_h
            x = cx * orig_w - w / 2
            y = cy * orig_h - h / 2

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Upsample mask from 1008x1008 to original size
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                upsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = upsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    if debug:
        print(f"[DEBUG] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
        if len(coco_gt['annotations']) > 0:
            sample_gt = coco_gt['annotations'][0]
            sample_img = coco_gt['images'][0]
            print(f"[DEBUG] Sample GT: image_id={sample_gt['image_id']}, bbox={sample_gt['bbox']}, image_size={sample_img['width']}x{sample_img['height']}")

    return coco_gt


class SAM3TrainerNative:
    def __init__(
        self,
        config: "SAM3LoRAConfig | str | Path",
        train_coco_dataset: COCODataset | None = None,
        val_coco_dataset: COCODataset | None = None,
        test_coco_dataset: COCODataset | None = None,
        multi_gpu=False,
        on_checkpoint=None,
    ):
        if isinstance(config, (str, Path)):
            self.config = SAM3LoRAConfig.from_yaml(config)
        else:
            self.config = config

        self.train_coco_dataset = train_coco_dataset
        self.val_coco_dataset = val_coco_dataset
        self.test_coco_dataset = test_coco_dataset

        self._on_checkpoint = on_checkpoint
        self._shutdown_requested = False

        # Multi-GPU setup
        self.multi_gpu = multi_gpu
        self.local_rank = 0
        self.world_size = 1

        if self.multi_gpu:
            self.local_rank = setup_distributed()
            self.world_size = get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            print_rank0(f"Multi-GPU training enabled with {self.world_size} GPUs")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            hw_cfg = self.config.hardware
            torch.backends.cudnn.benchmark = hw_cfg.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = hw_cfg.allow_tf32
            torch.backends.cudnn.allow_tf32 = hw_cfg.allow_tf32
            torch.set_float32_matmul_precision(hw_cfg.float32_matmul_precision)
            print_rank0(
                "CUDA perf settings: "
                f"cudnn_benchmark={torch.backends.cudnn.benchmark}, "
                f"allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
                f"float32_matmul_precision={hw_cfg.float32_matmul_precision}"
            )

        # Build Model
        print_rank0("Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=self.config.hardware.use_compile,
            load_from_HF=True,
            bpe_path=resolve_bpe_vocab_path(),
            eval_mode=False
        )

        # Apply LoRA
        print_rank0("Applying LoRA...")
        lora_cfg = self.config.lora
        lora_config = LoRALayerConfig(
            rank=lora_cfg.rank,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            apply_to_vision_encoder=lora_cfg.apply_to_vision_encoder,
            apply_to_text_encoder=lora_cfg.apply_to_text_encoder,
            apply_to_geometry_encoder=lora_cfg.apply_to_geometry_encoder,
            apply_to_detr_encoder=lora_cfg.apply_to_detr_encoder,
            apply_to_detr_decoder=lora_cfg.apply_to_detr_decoder,
            apply_to_mask_decoder=lora_cfg.apply_to_mask_decoder,
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        stats = count_parameters(self.model)
        print_rank0(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        self.model.to(self.device)

        # Wrap model with DDP if multi-GPU
        if self.multi_gpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            print_rank0(f"Model wrapped with DistributedDataParallel")

        # Store reference to unwrapped model for accessing custom methods
        self._unwrapped_model = self.model.module if self.multi_gpu else self.model

        # Mixed-precision autocast context
        mp = self.config.training.mixed_precision
        if mp.value == "bf16" and torch.cuda.is_bf16_supported():
            self._autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            print_rank0("Mixed precision: BF16 enabled")
        elif mp.value == "fp16":
            self._autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
            print_rank0("Mixed precision: FP16 enabled")
        else:
            self._autocast_ctx = contextlib.nullcontext
            print_rank0("Mixed precision: disabled (FP32)")

        # Optimizer
        train_cfg = self.config.training
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
            eps=train_cfg.adam_epsilon,
        )
        self.scheduler: LambdaLR | None = None
        
        # Matcher & Loss
        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )

        # Create loss functions with correct weights (from original SAM3 training config)
        # Note: These weights are for mask-based training
        loss_fns = [
            Boxes(weight_dict={
                "loss_bbox": 5.0,
                "loss_giou": 2.0
            }),
            IABCEMdetr(
                pos_weight=10.0,
                weight_dict={
                    "loss_ce": 20.0,
                    "presence_loss": 20.0
                },
                pos_focal=False,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pad_n_queries=200,
            ),
            Masks(
                weight_dict={
                    "loss_mask": 200.0,  # Much higher weight for mask loss!
                    "loss_dice": 10.0
                },
                focal_alpha=0.25,
                focal_gamma=2.0,
                compute_aux=False
            )
        ]

        # Create one-to-many matcher for auxiliary outputs
        o2m_matcher = BinaryOneToManyMatcher(
            alpha=0.3,
            threshold=0.4,
            topk=4
        )

        # Use Sam3LossWrapper for proper loss computation
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",  # Use local normalization (no distributed training)
            normalize_by_valid_object_num=False,
        )
        
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        global_step: int,
        step_in_epoch: int,
        best_val_loss: float,
        checkpoint_phase: str = CKPT_PHASE_TRAIN_IN_EPOCH,
    ):
        """Persist full training state so a run can resume after interruption.

        Args:
            path: File to write.
            epoch: Current epoch (0-indexed).
            global_step: Total batches processed across all epochs.
            step_in_epoch: Batches processed in the current epoch.
                           0 is only unambiguous when paired with checkpoint_phase.
            best_val_loss: Best validation loss observed so far.
            checkpoint_phase:
                - train_in_epoch: regular training state
                - pre_end_epoch_validation: captured right before validation
                - epoch_complete: epoch train+validation complete
        """
        model_to_save = self.model.module if self.multi_gpu else self.model
        lora_state = {}
        for name, module in model_to_save.named_modules():
            if isinstance(module, LoRALayer):
                lora_state[f"{name}.lora_A"] = module.lora_A.cpu()
                lora_state[f"{name}.lora_B"] = module.lora_B.cpu()
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "step_in_epoch": step_in_epoch,
            "best_val_loss": best_val_loss,
            "checkpoint_phase": checkpoint_phase,
            "lora_state_dict": lora_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(checkpoint, path)
        print_rank0(
            f"Checkpoint saved to {path} (epoch {epoch + 1}, global_step {global_step}, "
            f"phase={checkpoint_phase})"
        )
        if self._on_checkpoint is not None:
            try:
                self._on_checkpoint()
            except Exception as e:
                print_rank0(f"Warning: on_checkpoint callback failed: {e}")

    def load_checkpoint(self, path: Path) -> dict:
        """Load training state from a checkpoint and return metadata."""
        checkpoint = torch.load(path, map_location=self.device)
        model_to_load = self.model.module if self.multi_gpu else self.model
        model_to_load.load_state_dict(checkpoint["lora_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        gs = checkpoint.get("global_step", 0)
        sie = checkpoint.get("step_in_epoch", 0)
        phase = checkpoint.get("checkpoint_phase")
        if phase is None:
            # Backward-compatibility for older checkpoints.
            phase = CKPT_PHASE_EPOCH_COMPLETE if sie == 0 else CKPT_PHASE_TRAIN_IN_EPOCH
            checkpoint["checkpoint_phase"] = phase
        print_rank0(
            f"Resumed from {path.name} "
            f"(epoch {checkpoint['epoch'] + 1}, global_step={gs}, "
            f"step_in_epoch={sie}, phase={phase}, "
            f"best_val_loss={checkpoint['best_val_loss']:.6f})"
        )
        return checkpoint

    @staticmethod
    def find_latest_checkpoint(out_dir: Path) -> Path | None:
        """Return the checkpoint file with the highest global_step, or None."""
        best_step = -1
        best_path = None
        for name in CHECKPOINT_NAMES:
            path = out_dir / name
            if not path.exists():
                continue
            try:
                ckpt = torch.load(path, map_location="cpu")
                gs = ckpt.get("global_step", ckpt.get("epoch", -1))
                if gs > best_step:
                    best_step = gs
                    best_path = path
            except Exception:
                continue
        return best_path

    def _handle_shutdown(self, signum, frame):
        """Signal handler that requests a graceful checkpoint-and-exit."""
        if self._shutdown_requested:
            # Ignore repeated signals so we can finish the in-flight batch
            # and persist a checkpoint instead of exiting immediately.
            print_rank0("\nShutdown already requested; waiting to save checkpoint...")
            return
        self._shutdown_requested = True
        sig_name = signal.Signals(signum).name
        print_rank0(f"\n{sig_name} received. Will save checkpoint after current batch...")
        signal_out_dir = getattr(self, "_signal_output_dir", None)
        if signal_out_dir is not None:
            print_rank0(f"Signal artifacts directory: {Path(signal_out_dir).resolve()}")

    def _build_lr_scheduler(self, total_optimizer_steps: int) -> LambdaLR | None:
        train_cfg = self.config.training
        scheduler_name = train_cfg.lr_scheduler.value
        warmup_steps = max(0, int(train_cfg.warmup_steps))

        if total_optimizer_steps <= 0:
            return None

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            if scheduler_name in {"constant", "constant_with_warmup"}:
                return 1.0

            decay_steps = max(1, total_optimizer_steps - warmup_steps)
            progress = min(1.0, float(current_step - warmup_steps) / float(decay_steps))

            if scheduler_name == "linear":
                return max(0.0, 1.0 - progress)

            if scheduler_name == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            return 1.0

        print_rank0(
            f"Scheduler: {scheduler_name} (warmup_steps={warmup_steps}, "
            f"total_optimizer_steps={total_optimizer_steps})"
        )
        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def train(self):
        train_cfg = self.config.training
        sample_pct = train_cfg.data.sample_percent
        val_sample_pct = train_cfg.data.valid_sample_percent
        val_sample_seed = train_cfg.data.valid_sample_seed
        seed = train_cfg.seed

        print_rank0(
            f"Data sampling: train={sample_pct:.1f}% "
            f"| valid={val_sample_pct:.1f}% (seed={val_sample_seed})"
        )

        if self.train_coco_dataset is not None:
            print_rank0("\nLoading training data from passed train_coco_dataset...")
            train_image_dir = train_cfg.data.train.image_dir
            train_ds = COCOSegmentDataset(
                self.train_coco_dataset,
                image_dir=train_image_dir,
                sample_percent=sample_pct,
                seed=seed,
            )
        else:
            print_rank0("\nLoading training data from config paths...")
            train_ds = COCOSegmentDataset.from_split_config(
                train_cfg.data.train, sample_percent=sample_pct, seed=seed,
            )

        has_validation = False
        val_ds = None

        if self.val_coco_dataset is not None:
            print_rank0("\nLoading validation data from passed val_coco_dataset...")
            val_image_dir = train_cfg.data.valid.image_dir if train_cfg.data.valid else None
            val_ds = COCOSegmentDataset(
                self.val_coco_dataset,
                image_dir=val_image_dir,
                sample_percent=val_sample_pct,
                seed=val_sample_seed,
            )
            if len(val_ds) > 0:
                has_validation = True
                print_rank0(f"Found validation data: {len(val_ds)} images")
            else:
                print_rank0("Validation dataset is empty.")
                val_ds = None
        elif train_cfg.data.valid is not None:
            print_rank0("\nLoading validation data from config paths...")
            try:
                val_ds = COCOSegmentDataset.from_split_config(
                    train_cfg.data.valid, sample_percent=val_sample_pct, seed=val_sample_seed,
                )
                if len(val_ds) > 0:
                    has_validation = True
                    print_rank0(f"Found validation data: {len(val_ds)} images")
            except FileNotFoundError as e:
                print_rank0(f"Validation data not found: {e}")
                val_ds = None

        def collate_fn(batch):
            return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None

        if self.multi_gpu:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=get_rank(),
                shuffle=True
            )
            if has_validation:
                val_sampler = DistributedSampler(
                    val_ds,
                    num_replicas=self.world_size,
                    rank=get_rank(),
                    shuffle=False
                )

        persistent = train_cfg.num_workers > 0
        train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=train_cfg.num_workers,
            pin_memory=self.config.hardware.dataloader_pin_memory,
            persistent_workers=persistent,
            prefetch_factor=4 if train_cfg.num_workers > 0 else None,
        )

        if has_validation:
            val_batch_size = train_cfg.val_batch_size or train_cfg.batch_size
            val_loader = DataLoader(
                val_ds,
                batch_size=val_batch_size,
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=train_cfg.num_workers,
                pin_memory=self.config.hardware.dataloader_pin_memory,
                persistent_workers=persistent,
                prefetch_factor=2 if train_cfg.num_workers > 0 else None,
            )
        else:
            val_loader = None

        self.model.train()

        # Weights from a standard SAM config roughly
        weight_dict = {
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0
        }

        epochs = train_cfg.num_epochs
        best_val_loss = float('inf')
        start_epoch = 0
        resume_step_in_epoch = 0
        global_step = 0
        resume_checkpoint_phase = CKPT_PHASE_TRAIN_IN_EPOCH
        pending_resume_validation_epoch = None
        pending_resume_validation_step = None

        # Helper to move BatchedDatapoint to device
        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device, non_blocking=True)
            elif isinstance(obj, list):
                return [move_to_device(x, device) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(move_to_device(x, device) for x in obj)
            elif isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            elif hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    val = getattr(obj, field)
                    setattr(obj, field, move_to_device(val, device))
                return obj
            return obj

        out_dir = Path(self.config.output.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._signal_output_dir = str(out_dir.resolve())

        steps_per_epoch = len(train_loader)
        print_rank0(f"Steps per epoch: {steps_per_epoch}")
        if has_validation:
            print_rank0(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        else:
            print_rank0(f"Training samples: {len(train_ds)}")
            print_rank0("No validation data found - training without validation")

        accum_steps = train_cfg.gradient_accumulation_steps
        optimizer_steps_per_epoch = (steps_per_epoch + accum_steps - 1) // accum_steps
        stage_total_optimizer_steps = max(1, epochs * optimizer_steps_per_epoch)
        # When the caller (e.g. a staged sweep) provides a lifetime
        # multiplier, build the LR schedule against the cumulative step
        # budget instead of just the current stage's ``num_epochs``. This
        # keeps the cosine/linear curve consistent across stages, so
        # resuming scheduler state from a previous stage's checkpoint is
        # meaningful (otherwise the ``current_step`` carried over from the
        # previous schedule lands on a brand-new curve and produces the
        # wrong LR for several hundred steps after each stage transition).
        lifetime_multiplier = getattr(
            train_cfg, "lr_scheduler_lifetime_multiplier", None
        )
        if (
            lifetime_multiplier is not None
            and float(lifetime_multiplier) > 0
            and not math.isclose(float(lifetime_multiplier), 1.0)
        ):
            total_optimizer_steps = max(
                stage_total_optimizer_steps,
                int(round(stage_total_optimizer_steps * float(lifetime_multiplier))),
            )
            print_rank0(
                f"Using cumulative LR schedule budget: stage_steps="
                f"{stage_total_optimizer_steps}, lifetime_multiplier="
                f"{float(lifetime_multiplier):.4f}, lifetime_steps="
                f"{total_optimizer_steps}"
            )
        else:
            total_optimizer_steps = stage_total_optimizer_steps
        self.scheduler = self._build_lr_scheduler(total_optimizer_steps)

        latest_ckpt = self.find_latest_checkpoint(out_dir)
        if latest_ckpt is not None:
            ckpt = self.load_checkpoint(latest_ckpt)
            best_val_loss = ckpt["best_val_loss"]
            global_step = ckpt.get("global_step", 0)
            resume_step_in_epoch = ckpt.get("step_in_epoch", 0)
            resume_checkpoint_phase = ckpt.get("checkpoint_phase", CKPT_PHASE_TRAIN_IN_EPOCH)
            if resume_checkpoint_phase == CKPT_PHASE_PRE_END_EPOCH_VALIDATION:
                pending_resume_validation_epoch = ckpt["epoch"]
                pending_resume_validation_step = resume_step_in_epoch
                start_epoch = ckpt["epoch"]
                print_rank0(
                    "Detected checkpoint captured before validation; "
                    f"will replay validation for epoch {pending_resume_validation_epoch + 1}."
                )
            elif resume_step_in_epoch == 0:
                start_epoch = ckpt["epoch"] + 1
            else:
                start_epoch = ckpt["epoch"]
            print_rank0(
                f"Resuming from epoch {start_epoch + 1}/{epochs}"
                + (f", step {resume_step_in_epoch}" if resume_step_in_epoch else "")
            )
        else:
            print_rank0(f"Starting training for {epochs} epochs...")

        gpu_multiplier = self.world_size if self.multi_gpu else 1
        effective_bs = train_cfg.batch_size * accum_steps * gpu_multiplier
        print_rank0(
            f"Effective batch size: {train_cfg.batch_size} x {accum_steps}"
            + (f" x {self.world_size} GPUs" if self.multi_gpu else "")
            + f" = {effective_bs}"
        )

        # Install signal handlers for graceful shutdown
        self._shutdown_requested = False
        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        try:
            signal.signal(signal.SIGTERM, self._handle_shutdown)
        except OSError:
            pass

        last_eval_step = -1
        log_every_steps = 20

        def save_signal_checkpoint(
            current_epoch: int,
            step_in_epoch: int,
            checkpoint_phase: str = CKPT_PHASE_TRAIN_IN_EPOCH,
        ) -> None:
            if not is_main_process():
                return
            signal_ckpt_path = out_dir / "checkpoint_signal.pt"
            print_rank0(f"Saving signal checkpoint before exit: {signal_ckpt_path}")
            self.save_checkpoint(
                signal_ckpt_path,
                current_epoch,
                global_step,
                step_in_epoch,
                best_val_loss,
                checkpoint_phase=checkpoint_phase,
            )
            print_rank0(
                f"Signal checkpoint saved at: {signal_ckpt_path} "
                f"(artifacts dir: {out_dir.resolve()}). Exiting training loop."
            )

        def save_pre_validation_checkpoint(current_epoch: int, step_in_epoch: int) -> None:
            """Persist state before long validation so abrupt app stops remain resumable."""
            if not is_main_process():
                return
            self.save_checkpoint(
                out_dir / "checkpoint_epoch.pt",
                current_epoch,
                global_step,
                step_in_epoch,
                best_val_loss,
                checkpoint_phase=CKPT_PHASE_PRE_END_EPOCH_VALIDATION,
            )

        def run_validation(current_epoch: int, step_in_epoch: int, avg_train_loss: float) -> tuple[float, bool]:
            nonlocal best_val_loss
            self.model.eval()
            val_loss_sum = torch.zeros(1, device=self.device)
            val_batch_count = 0
            val_start_time = time.perf_counter()
            interval_data_wait_sec = 0.0
            interval_compute_sec = 0.0
            interval_steps = 0
            last_val_iter_end_time = time.perf_counter()
            val_batch_size = train_cfg.val_batch_size or train_cfg.batch_size

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validation", disable=not is_main_process())

                for val_batch_idx, batch_dict in enumerate(val_pbar):
                    if self._shutdown_requested:
                        save_signal_checkpoint(
                            current_epoch,
                            step_in_epoch,
                            checkpoint_phase=CKPT_PHASE_PRE_END_EPOCH_VALIDATION,
                        )
                        return float((val_loss_sum / max(1, val_batch_count)).item()), True

                    val_iter_entry_time = time.perf_counter()
                    interval_data_wait_sec += val_iter_entry_time - last_val_iter_end_time
                    input_batch = batch_dict["input"]
                    input_batch = move_to_device(input_batch, self.device)

                    with self._autocast_ctx():
                        outputs_list = self.model(input_batch)
                        find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                        for targets in find_targets:
                            for k, v in targets.items():
                                if isinstance(v, torch.Tensor):
                                    targets[k] = v.to(self.device, non_blocking=True)

                        with SAM3Output.iteration_mode(
                            outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                        ) as outputs_iter:
                            for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                                stage_targets_list = [stage_targets] * len(stage_outputs)
                                for outputs, targets in zip(stage_outputs, stage_targets_list):
                                    outputs["indices"] = self.matcher(outputs, targets)
                                    if "aux_outputs" in outputs:
                                        for aux_out in outputs["aux_outputs"]:
                                            aux_out["indices"] = self.matcher(aux_out, targets)

                        loss_dict = self.loss_wrapper(outputs_list, find_targets)
                        total_loss = loss_dict[CORE_LOSS_KEY]
                    val_loss_sum += total_loss.detach()
                    val_batch_count += 1
                    interval_compute_sec += time.perf_counter() - val_iter_entry_time
                    interval_steps += 1

                    if is_main_process() and (val_batch_idx + 1) % log_every_steps == 0:
                        avg_data_wait = interval_data_wait_sec / max(1, interval_steps)
                        avg_compute = interval_compute_sec / max(1, interval_steps)
                        interval_imgs_per_sec = (
                            interval_steps * val_batch_size * gpu_multiplier
                        ) / max(interval_data_wait_sec + interval_compute_sec, 1e-6)
                        val_pbar.set_postfix(
                            {
                                "val_loss": float((val_loss_sum / val_batch_count).item()),
                                "val_step": val_batch_idx + 1,
                                "data_wait_s": f"{avg_data_wait:.2f}",
                                "compute_s": f"{avg_compute:.2f}",
                                "val_imgs_s": f"{interval_imgs_per_sec:.2f}",
                            }
                        )
                        interval_data_wait_sec = 0.0
                        interval_compute_sec = 0.0
                        interval_steps = 0

                    last_val_iter_end_time = time.perf_counter()

            avg_val_loss = float((val_loss_sum / max(1, val_batch_count)).item())

            if self.multi_gpu:
                val_loss_tensor = torch.tensor([avg_val_loss], device=self.device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                avg_val_loss = val_loss_tensor.item()
            val_elapsed_sec = time.perf_counter() - val_start_time
            val_batch_size = train_cfg.val_batch_size or train_cfg.batch_size
            val_imgs_per_sec = (val_batch_count * val_batch_size * gpu_multiplier) / max(val_elapsed_sec, 1e-6)

            print_rank0(
                f"\nEpoch {current_epoch+1}/{epochs} (global_step={global_step}) "
                f"- Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                f"Val Time: {val_elapsed_sec:.1f}s, Val Throughput: {val_imgs_per_sec:.2f} imgs/s"
            )

            if is_main_process():
                model_to_save = self.model.module if self.multi_gpu else self.model
                save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_lora_weights(model_to_save, str(out_dir / "best_lora_weights.pt"))
                    self.save_checkpoint(
                        out_dir / "checkpoint_best.pt",
                        current_epoch,
                        global_step,
                        step_in_epoch,
                        best_val_loss,
                        checkpoint_phase=CKPT_PHASE_TRAIN_IN_EPOCH,
                    )
                    print_rank0(f"New best model saved (val_loss: {avg_val_loss:.6f})")

                with open(out_dir / "val_stats.json", "a") as f:
                    f.write(json.dumps({
                        "epoch": current_epoch + 1,
                        "step_in_epoch": step_in_epoch,
                        "global_step": global_step,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                    }) + "\n")

            self.model.train()
            return avg_val_loss, False

        # If the latest checkpoint was captured before a validation pass, replay
        # that validation before proceeding so no validation event is lost.
        if (
            has_validation
            and val_loader is not None
            and pending_resume_validation_epoch is not None
            and pending_resume_validation_step is not None
        ):
            print_rank0(
                "Replaying interrupted validation pass from checkpoint metadata."
            )
            _, interrupted = run_validation(
                current_epoch=pending_resume_validation_epoch,
                step_in_epoch=pending_resume_validation_step,
                avg_train_loss=float("nan"),
            )
            if interrupted:
                return
            last_eval_step = global_step

            if pending_resume_validation_step == 0:
                # End-of-epoch validation has now completed; continue with next epoch.
                start_epoch = pending_resume_validation_epoch + 1
                resume_step_in_epoch = 0

        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.perf_counter()
            # Set epoch for distributed sampler (required for proper shuffling)
            if self.multi_gpu and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Track training losses for this epoch without per-step host sync.
            train_loss_sum = torch.zeros(1, device=self.device)
            train_loss_count = 0
            # Lightweight runtime breakdown to identify bottlenecks.
            interval_data_wait_sec = 0.0
            interval_compute_sec = 0.0
            interval_steps = 0
            last_iter_end_time = time.perf_counter()

            skip_to = resume_step_in_epoch if epoch == start_epoch else 0
            if skip_to > 0:
                print_rank0(f"Skipping first {skip_to} batches (already processed)...")

            self.optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
            for batch_idx, batch_dict in enumerate(pbar):
                iter_entry_time = time.perf_counter()
                interval_data_wait_sec += iter_entry_time - last_iter_end_time

                if batch_idx < skip_to:
                    last_iter_end_time = time.perf_counter()
                    continue

                input_batch = batch_dict["input"]
                input_batch = move_to_device(input_batch, self.device)

                with self._autocast_ctx():
                    outputs_list = self.model(input_batch)
                find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                for targets in find_targets:
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device, non_blocking=True)

                with SAM3Output.iteration_mode(
                    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                ) as outputs_iter:
                    for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                        stage_targets_list = [stage_targets] * len(stage_outputs)
                        for outputs, targets in zip(stage_outputs, stage_targets_list):
                            outputs["indices"] = self.matcher(outputs, targets)
                            if "aux_outputs" in outputs:
                                for aux_out in outputs["aux_outputs"]:
                                    aux_out["indices"] = self.matcher(aux_out, targets)

                loss_dict = self.loss_wrapper(outputs_list, find_targets)
                total_loss = loss_dict[CORE_LOSS_KEY] / accum_steps

                total_loss.backward()

                train_loss_sum += (total_loss.detach() * accum_steps)
                train_loss_count += 1
                interval_compute_sec += time.perf_counter() - iter_entry_time
                interval_steps += 1

                is_accum_boundary = (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader)
                if is_accum_boundary:
                    if train_cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.max_grad_norm)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if is_main_process() and global_step % log_every_steps == 0:
                        avg_data_wait = interval_data_wait_sec / max(1, interval_steps)
                        avg_compute = interval_compute_sec / max(1, interval_steps)
                        pbar.set_postfix(
                            {
                                "loss": float((train_loss_sum / max(1, train_loss_count)).item()),
                                "step": global_step,
                                "data_wait_s": f"{avg_data_wait:.2f}",
                                "compute_s": f"{avg_compute:.2f}",
                            }
                        )
                        interval_data_wait_sec = 0.0
                        interval_compute_sec = 0.0
                        interval_steps = 0

                    if is_main_process() and train_cfg.save_steps > 0 and global_step % train_cfg.save_steps == 0:
                        self.save_checkpoint(
                            out_dir / "checkpoint_epoch.pt",
                            epoch,
                            global_step,
                            batch_idx + 1,
                            best_val_loss,
                            checkpoint_phase=CKPT_PHASE_TRAIN_IN_EPOCH,
                        )

                    if (
                        has_validation
                        and val_loader is not None
                        and train_cfg.eval_steps > 0
                        and global_step % train_cfg.eval_steps == 0
                        and global_step != last_eval_step
                    ):
                        rolling_train_loss = float((train_loss_sum / max(1, train_loss_count)).item())
                        save_pre_validation_checkpoint(epoch, batch_idx + 1)
                        _, interrupted = run_validation(
                            current_epoch=epoch,
                            step_in_epoch=batch_idx + 1,
                            avg_train_loss=rolling_train_loss,
                        )
                        if interrupted:
                            return
                        last_eval_step = global_step

                # Graceful shutdown on SIGINT / SIGTERM: save mid-epoch checkpoint
                if self._shutdown_requested and is_main_process():
                    save_signal_checkpoint(epoch, batch_idx + 1)
                    return

                last_iter_end_time = time.perf_counter()

            # Calculate average training loss for this epoch
            avg_train_loss = float((train_loss_sum / max(1, train_loss_count)).item())
            epoch_elapsed_sec = time.perf_counter() - epoch_start_time
            approx_samples = train_loss_count * train_cfg.batch_size * gpu_multiplier
            samples_per_sec = approx_samples / max(epoch_elapsed_sec, 1e-6)

            # Validation (only compute loss - no metrics, like SAM3)
            if has_validation and val_loader is not None:
                should_run_epoch_eval = (
                    ((epoch + 1) % train_cfg.eval_every_n_epochs == 0) or ((epoch + 1) == epochs)
                )
                if should_run_epoch_eval and last_eval_step != global_step:
                    save_pre_validation_checkpoint(epoch, 0)
                    _, interrupted = run_validation(
                        current_epoch=epoch,
                        step_in_epoch=0,
                        avg_train_loss=avg_train_loss,
                    )
                    if interrupted:
                        return
                    last_eval_step = global_step
            else:
                if is_main_process():
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))

            # End-of-epoch checkpoint (always saved)
            if is_main_process():
                self.save_checkpoint(
                    out_dir / "checkpoint_epoch.pt",
                    epoch,
                    global_step,
                    0,
                    best_val_loss,
                    checkpoint_phase=CKPT_PHASE_EPOCH_COMPLETE,
                )
                print_rank0(
                    f"Epoch {epoch+1} timing: {epoch_elapsed_sec:.1f}s, "
                    f"approx throughput: {samples_per_sec:.2f} samples/s"
                )

        # Restore original signal handlers
        signal.signal(signal.SIGINT, prev_sigint)
        try:
            signal.signal(signal.SIGTERM, prev_sigterm)
        except OSError:
            pass

        if self.multi_gpu:
            dist.barrier()

        if is_main_process():
            if has_validation:
                print(f"\n{'='*80}")
                print(f"Training complete!")
                print(f"{'='*80}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (best validation loss)")
                print(f"  - last_lora_weights.pt (last epoch)")
                print(f"\nTo compute full metrics (mAP, cgF1) with NMS:")
                print(f"   python validate_sam3_lora.py \\")
                print(f"     --config <config_path> \\")
                print(f"     --weights {out_dir}/best_lora_weights.pt \\")
                print(f"     --val_data_dir <data_dir>/valid")
                print(f"{'='*80}")
            else:
                import shutil
                last_path = out_dir / "last_lora_weights.pt"
                best_path = out_dir / "best_lora_weights.pt"
                if last_path.exists():
                    shutil.copy(last_path, best_path)

                print(f"\n{'='*80}")
                print(f"Training complete!")
                print(f"{'='*80}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (copy of last epoch)")
                print(f"  - last_lora_weights.pt (last epoch)")
                print(f"\nNo validation data - consider adding data/valid/ for better model selection")
                print(f"{'='*80}")

            # Keep checkpoints so staged sweeps can resume across successive runs.
            # Artifact retention can be handled by a separate cleanup policy/job.
            print_rank0("Keeping checkpoint files for potential resume/chained stages.")

        if self.multi_gpu:
            cleanup_distributed()

def launch_distributed_training(args):
    """Launch training with multiple GPUs using torchrun subprocess."""
    import subprocess
    import sys

    devices = args.device
    num_gpus = len(devices)
    device_str = ",".join(map(str, devices))

    print(f"Launching distributed training on GPUs: {devices}")
    print(f"Number of processes: {num_gpus}")

    # Build the command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_port", str(args.master_port),
        sys.argv[0],  # This script
        "--config", args.config,
        "--device", *map(str, devices),
        "--_launched_by_torchrun"  # Internal flag to indicate we're in subprocess
    ]

    # Set environment variable for visible devices
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_str

    # Run the subprocess
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAM3 with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single GPU (default GPU 0):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Single GPU (specific GPU):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 1

  Multi-GPU (GPUs 0 and 1):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1

  Multi-GPU (GPUs 0, 2, 3):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 2 3

  Multi-GPU (all 4 GPUs):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1 2 3
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_lora_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ID(s) to use. Single value for single GPU, multiple values for multi-GPU. "
             "Example: --device 0 (single GPU), --device 0 1 2 (3 GPUs)"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for distributed training (default: 29500)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set automatically by torchrun)"
    )
    parser.add_argument(
        "--_launched_by_torchrun",
        action="store_true",
        help=argparse.SUPPRESS  # Hidden argument for internal use
    )
    args = parser.parse_args()

    # Determine if multi-GPU training is requested
    num_devices = len(args.device)
    is_torchrun_subprocess = args._launched_by_torchrun or "LOCAL_RANK" in os.environ

    if num_devices > 1 and not is_torchrun_subprocess:
        # Multi-GPU requested but not yet in torchrun - launch it
        launch_distributed_training(args)
    else:
        # Single GPU or already in torchrun subprocess
        multi_gpu = num_devices > 1 and is_torchrun_subprocess

        if not multi_gpu and num_devices == 1:
            # Single GPU mode - set the device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[0])
            print(f"Using single GPU: {args.device[0]}")

        trainer = SAM3TrainerNative(args.config, multi_gpu=multi_gpu)
        trainer.train()
