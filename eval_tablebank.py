#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from sam3_table.coco_schema import COCODataset
from sam3_table.lora_layers import LoRAConfig as LoRALayerConfig
from sam3_table.lora_layers import apply_lora_to_model
from sam3_table.postprocess import postprocess_sam3_predictions
from sam3_table.training_config import SAM3LoRAConfig
from voc_to_coco import convert_voc_to_coco

TABLEBANK_IMAGE = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("triton", "pycocotools")
    .add_local_python_source("sam3_table")
    .add_local_file("voc_to_coco.py", remote_path="/root/voc_to_coco.py")
)

app = modal.App(name="tablebank-eval", image=TABLEBANK_IMAGE)

tablebank_vol = modal.Volume.from_name("tablebank-vol")
artifacts_vol = modal.Volume.from_name("artifacts-vol", create_if_missing=True)

MODAL_DATA_DIR = "/data"
MODAL_ARTIFACTS_DIR = "/artifacts"
BPE_VOCAB_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
BPE_CACHE_PATH = Path("/tmp/bpe_simple_vocab_16e6.txt.gz")
DEFAULT_STUDY_NAME = "sam3-final-optuna-asha"
DEFAULT_NUM_RUNG_STAGES = 5


def ensure_deployed(
    *,
    environment_name: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Deploy the ``tablebank-eval`` app if it isn't already deployed.

    This is intended to be called from local entrypoints (e.g. the Optuna
    sweeps) before any remote work tries to resolve
    ``modal.Function.from_name("tablebank-eval", ...)``. Deploying is
    idempotent on Modal's side, but we still gate on a cheap app lookup so
    we don't pay for a redeploy on every run.
    """
    try:
        modal.App.lookup(app.name, environment_name=environment_name)
        if verbose:
            print(f"[eval_tablebank] App '{app.name}' already deployed; skipping deploy.")
        return {"already_deployed": True, "app_name": app.name}
    except modal.exception.NotFoundError:
        if verbose:
            print(
                f"[eval_tablebank] App '{app.name}' not deployed in environment "
                f"{environment_name or '<default>'}; deploying now so leader "
                "extraction can resolve it..."
            )
        from modal.runner import deploy_app

        result = deploy_app(app, environment_name=environment_name)
        app_id = getattr(result, "app_id", None)
        app_page_url = getattr(result, "app_page_url", None)
        if verbose:
            print(
                f"[eval_tablebank] Deployed '{app.name}' (app_id={app_id}). "
                f"View: {app_page_url}"
            )
        return {
            "already_deployed": False,
            "app_name": app.name,
            "app_id": app_id,
            "app_page_url": app_page_url,
        }


@dataclass(frozen=True)
class DetectionImage:
    image_id: int
    file_name: str
    width: int
    height: int
    image_path: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _score_annotation_candidate(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    score = 0
    if "annotation" in name:
        score += 4
    if "coco" in name:
        score += 3
    if "detect" in name or "table" in name:
        score += 1
    return (-score, len(path.parts), str(path))


def _resolve_annotations_source(
    annotations_path: Path,
    dataset_root_path: Path,
) -> tuple[Path, str]:
    if annotations_path.is_file():
        return annotations_path, "coco"

    search_roots: list[Path] = []
    if annotations_path.exists():
        search_roots.append(annotations_path)
    if dataset_root_path not in search_roots:
        search_roots.append(dataset_root_path)

    json_candidates: list[Path] = []
    xml_candidates: list[Path] = []
    seen_paths: set[Path] = set()
    for root in search_roots:
        if not root.exists() or root in seen_paths:
            continue
        seen_paths.add(root)
        if root.is_file():
            if root.suffix.lower() == ".json":
                json_candidates.append(root)
            elif root.suffix.lower() == ".xml":
                xml_candidates.append(root)
            continue
        json_candidates.extend(path for path in root.rglob("*.json") if path.is_file())
        xml_candidates.extend(path for path in root.rglob("*.xml") if path.is_file())

    if json_candidates:
        return sorted(set(json_candidates), key=_score_annotation_candidate)[0], "coco"

    if xml_candidates:
        xml_root = annotations_path if annotations_path.exists() else dataset_root_path
        return xml_root, "voc"

    raise FileNotFoundError(
        f"Could not find COCO JSON or VOC XML annotations under "
        f"{annotations_path} or {dataset_root_path}"
    )


def _resolve_device(device: str | Any | None) -> Any:
    import torch

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _move_to_device(obj: Any, device: Any) -> Any:
    import torch

    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, list):
        return [_move_to_device(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(item, device) for item in obj)
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            setattr(obj, field, _move_to_device(getattr(obj, field), device))
        return obj
    return obj


def _resolve_config_path(
    weights_path: Path,
    config_path: str | Path | None,
) -> Path:
    if config_path is not None:
        resolved_config_path = Path(config_path)
        if not resolved_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")
        return resolved_config_path

    sibling_config_path = weights_path.with_name("run_config.json")
    if sibling_config_path.exists():
        return sibling_config_path

    raise FileNotFoundError(
        "Could not find a config for the LoRA weights. "
        "Expected a sibling run_config.json next to the weights file, "
        "or pass config_path explicitly."
    )


def _prepare_output_dir(
    output_dir_path: Path,
    score_threshold: float,
    unique_output_dir: bool,
) -> Path:
    if not unique_output_dir:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        return output_dir_path

    threshold_tag = f"t{int(round(score_threshold * 100)):03d}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_output_dir = output_dir_path / f"{timestamp}_{threshold_tag}"
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return run_output_dir


def _select_detection_subset(
    items: list[DetectionImage],
    dataset_fraction: float,
    sample_seed: int | None,
    holdout_train_fraction: float = 0.0,
    holdout_train_seed: int | None = None,
) -> list[DetectionImage]:
    """Pick the subset of detection images to evaluate on.

    When ``holdout_train_fraction > 0``, this first reproduces the trainer's
    split logic (``random.Random(holdout_train_seed).sample`` over the COCO
    image ids sorted ascending) to recover the exact set of training images,
    then EXCLUDES them so eval runs only on the held-out complement. After
    that, ``dataset_fraction`` / ``sample_seed`` further subsamples the
    remaining items if requested.
    """
    working_items = items

    if holdout_train_fraction > 0.0:
        if holdout_train_fraction >= 1.0:
            raise ValueError(
                "holdout_train_fraction must be < 1.0 (otherwise the held-out "
                "complement is empty)."
            )
        if holdout_train_seed is None:
            raise ValueError(
                "holdout_train_seed is required when holdout_train_fraction > 0."
            )
        sorted_ids = sorted(item.image_id for item in working_items)
        k = max(1, int(len(sorted_ids) * holdout_train_fraction))
        train_ids = set(random.Random(holdout_train_seed).sample(sorted_ids, k))
        working_items = [item for item in working_items if item.image_id not in train_ids]
        if not working_items:
            raise ValueError(
                "Holdout filter removed every image; nothing left to evaluate."
            )

    if dataset_fraction >= 1.0 or not working_items:
        return working_items

    subset_size = max(1, int(len(working_items) * dataset_fraction))
    if sample_seed is None:
        return working_items[:subset_size]

    rng = random.Random(sample_seed)
    selected_indices = sorted(rng.sample(range(len(working_items)), k=subset_size))
    return [working_items[idx] for idx in selected_indices]


def _load_config(
    weights_path: Path,
    config_path: str | Path | None,
) -> SAM3LoRAConfig:
    resolved_config_path = _resolve_config_path(weights_path, config_path)
    if resolved_config_path.suffix.lower() == ".json":
        return SAM3LoRAConfig.model_validate(
            json.loads(resolved_config_path.read_text())
        )
    return SAM3LoRAConfig.from_yaml(resolved_config_path)


def _latest_stage_attr(user_attrs: dict[str, Any], suffix: str) -> Any | None:
    stage_keys = sorted(
        key
        for key in user_attrs.keys()
        if key.startswith("stage_") and key.endswith(f"_{suffix}")
    )
    if not stage_keys:
        return None
    return user_attrs[stage_keys[-1]]


def resolve_bpe_vocab_path() -> str:
    env_path = os.environ.get("SAM3_BPE_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    if BPE_CACHE_PATH.exists():
        return str(BPE_CACHE_PATH)

    BPE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SAM3 BPE vocab to {BPE_CACHE_PATH}...")
    urllib.request.urlretrieve(BPE_VOCAB_URL, BPE_CACHE_PATH)
    return str(BPE_CACHE_PATH)


def _bbox_iou_xywh(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _match_detections_at_iou50(
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, float]:
    preds_by_image: dict[int, list[dict[str, Any]]] = {}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}

    for pred in predictions:
        preds_by_image.setdefault(int(pred["image_id"]), []).append(pred)
    for ann in annotations:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    all_image_ids = set(preds_by_image) | set(anns_by_image)
    for image_id in all_image_ids:
        preds = sorted(
            preds_by_image.get(image_id, []),
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        gts = anns_by_image.get(image_id, [])
        matched_gt_indices: set[int] = set()

        for pred in preds:
            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt in enumerate(gts):
                if idx in matched_gt_indices:
                    continue
                iou = _bbox_iou_xywh(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= 0.5 and best_gt_idx >= 0:
                true_positives += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                false_positives += 1

        false_negatives += len(gts) - len(matched_gt_indices)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision_iou50": precision,
        "recall_iou50": recall,
        "f1_iou50": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _render_eval_visualizations(
    detection_images: list[DetectionImage],
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    output_dir_path: Path,
    max_images: int,
) -> dict[str, Any]:
    """Save a small sample of evaluation images with GT and predicted boxes."""
    from PIL import Image as PILImage
    from PIL import ImageDraw
    from PIL import ImageFont

    if max_images <= 0:
        return {
            "visualizations_dir": None,
            "num_visualizations": 0,
            "files": [],
        }

    preds_by_image: dict[int, list[dict[str, Any]]] = {}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for pred in predictions:
        preds_by_image.setdefault(int(pred["image_id"]), []).append(pred)
    for ann in annotations:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    vis_dir = output_dir_path / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    saved_files: list[str] = []
    for item in detection_images[:max_images]:
        image = PILImage.open(item.image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        gt_boxes = anns_by_image.get(item.image_id, [])
        pred_boxes = sorted(
            preds_by_image.get(item.image_id, []),
            key=lambda pred: float(pred.get("score", 0.0)),
            reverse=True,
        )

        for ann in gt_boxes:
            x, y, w, h = ann["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline="#00ff00", width=3)

        for pred in pred_boxes:
            x, y, w, h = pred["bbox"]
            score = float(pred.get("score", 0.0))
            draw.rectangle((x, y, x + w, y + h), outline="#ff3b30", width=2)
            label = f"pred {score:.2f}"
            text_y = max(0.0, y - 12.0)
            draw.text((x, text_y), label, fill="#ff3b30", font=font)

        legend = (
            f"GT={len(gt_boxes)} green | "
            f"Pred={len(pred_boxes)} red | "
            f"image_id={item.image_id}"
        )
        draw.text((8, 8), legend, fill="#ffffff", font=font)

        safe_stem = f"{item.image_id}_{Path(item.file_name).stem}"
        out_path = vis_dir / f"{safe_stem}.png"
        image.save(out_path)
        saved_files.append(str(out_path))

    return {
        "visualizations_dir": str(vis_dir),
        "num_visualizations": len(saved_files),
        "files": saved_files,
    }


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 10,
)
def get_current_optuna_leader(
    study_name: str = DEFAULT_STUDY_NAME,
    include_running_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    num_rung_stages: int = DEFAULT_NUM_RUNG_STAGES,
) -> dict[str, Any]:
    import importlib

    optuna = importlib.import_module("optuna")

    artifacts_vol.reload()
    sqlite_lock_timeout_sec = max(1, int(sqlite_lock_timeout_sec))
    num_rung_stages = max(1, int(num_rung_stages))

    sqlite_path = Path(MODAL_ARTIFACTS_DIR) / "optuna" / f"{study_name}.db"
    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"Optuna study DB was not found at {sqlite_path}. "
            "Make sure the ASHA sweep has started and wrote the study database."
        )

    with sqlite3.connect(str(sqlite_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.commit()

    storage_url = f"sqlite:///{sqlite_path.as_posix()}?timeout={sqlite_lock_timeout_sec}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    best_row: dict[str, Any] | None = None
    for trial in study.trials:
        score: float | None = None
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            score = float(trial.value)
        elif include_running_trials and trial.intermediate_values:
            latest_step = max(trial.intermediate_values.keys())
            score = float(trial.intermediate_values[latest_step])

        if score is None:
            continue

        stage_output_dir = trial.user_attrs.get(f"stage_{num_rung_stages}_output_dir")
        if not isinstance(stage_output_dir, str):
            stage_output_dir = _latest_stage_attr(trial.user_attrs, "output_dir")
        if not isinstance(stage_output_dir, str):
            continue

        weights_candidates = [
            Path(stage_output_dir) / "best_lora_weights.pt",
            Path(stage_output_dir) / "last_lora_weights.pt",
        ]
        weights_path = next((p for p in weights_candidates if p.exists()), None)
        if weights_path is None:
            continue

        row = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "score": score,
            "output_dir": stage_output_dir,
            "weights_path": str(weights_path),
            "run_config_path": str(Path(stage_output_dir) / "run_config.json"),
            "final_stage_val_loss": trial.user_attrs.get("final_stage_val_loss"),
            "objective_mode": trial.user_attrs.get("objective_mode"),
            "hyperparameters": dict(trial.params),
        }
        if best_row is None or row["score"] < best_row["score"]:
            best_row = row

    if best_row is None:
        raise RuntimeError(
            f"No leader candidate found for study '{study_name}'. "
            "Need at least one trial with an output_dir and available LoRA weights."
        )

    return best_row


@app.function(
    image=TABLEBANK_IMAGE,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODAL_DATA_DIR: tablebank_vol, MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=3600 * 24,
)
def run_tablebank_eval_on_current_leader(
    study_name: str = DEFAULT_STUDY_NAME,
    dataset_root: str = "/data/tablebank/extracted/TableBank/TableBank/Detection",
    annotations_path: str = "/data/tablebank/extracted/TableBank/TableBank/Detection/annotations",
    output_root_dir: str = "/artifacts/tablebank_eval_leaders",
    score_threshold: float = 0.25,
    query_text: str = "table",
    batch_size: int = 8,
    visualize_max_images: int = 20,
    dataset_fraction: float = 1.0,
    sample_seed: int | None = None,
    duplicate_iou_threshold: float = 0.5,
    min_box_area: float = 16.0,
    include_running_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    num_rung_stages: int = DEFAULT_NUM_RUNG_STAGES,
) -> dict[str, Any]:
    leader = get_current_optuna_leader.local(
        study_name=study_name,
        include_running_trials=include_running_trials,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
        num_rung_stages=num_rung_stages,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    study_slug = "".join(char if (char.isalnum() or char in "-_.") else "_" for char in study_name)
    leader_output_dir = (
        f"{output_root_dir.rstrip('/')}"
        f"/{study_slug}/trial_{int(leader['trial_number']):04d}_{timestamp}"
    )
    # Keep benchmark artifacts outside trial output dirs so sweep ranking is unchanged.
    result = run_tablebank_eval.remote(
        weights_path=str(leader["weights_path"]),
        dataset_root=dataset_root,
        annotations_path=annotations_path,
        output_dir=leader_output_dir,
        score_threshold=score_threshold,
        query_text=query_text,
        batch_size=batch_size,
        visualize_max_images=visualize_max_images,
        dataset_fraction=dataset_fraction,
        sample_seed=sample_seed,
        duplicate_iou_threshold=duplicate_iou_threshold,
        min_box_area=min_box_area,
    )
    result["leader"] = leader
    result["benchmark_output_dir"] = leader_output_dir
    return result


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 10,
)
def describe_current_leader(
    study_name: str = DEFAULT_STUDY_NAME,
    include_running_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    num_rung_stages: int = DEFAULT_NUM_RUNG_STAGES,
) -> dict[str, Any]:
    leader = get_current_optuna_leader.local(
        study_name=study_name,
        include_running_trials=include_running_trials,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
        num_rung_stages=num_rung_stages,
    )
    return {
        "study_name": study_name,
        "leader_trial_number": leader["trial_number"],
        "leader_state": leader["state"],
        "leader_score": leader["score"],
        "objective_mode": leader.get("objective_mode"),
        "weights_path": leader["weights_path"],
        "output_dir": leader["output_dir"],
        "hyperparameters": leader.get("hyperparameters", {}),
    }


@app.function(
    gpu="H200",
    image=TABLEBANK_IMAGE,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODAL_DATA_DIR: tablebank_vol, MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=3600 * 24,
)
def run_tablebank_eval(
    weights_path: str,
    dataset_root: str,
    annotations_path: str,
    output_dir: str,
    score_threshold: float = 0.25,
    unique_output_dir: bool = False,
    dataset_fraction: float = 1.0,
    sample_seed: int | None = None,
    holdout_train_fraction: float = 0.0,
    holdout_train_seed: int | None = None,
    duplicate_iou_threshold: float = 0.5,
    min_box_area: float = 16.0,
    query_text: str = "table",
    batch_size: int = 8,
    visualize_max_images: int = 20,
) -> dict[str, Any]:
    import numpy as np
    import pycocotools.mask as mask_utils
    import torch
    from PIL import Image as PILImage
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from sam3.model.model_misc import SAM3Output
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image,
        InferenceMetadata,
    )
    from torch.utils.data import Dataset
    from torchvision.transforms import v2

    from sam3_table.model_builder import build_sam3_image_model

    eval_start_time = time.monotonic()

    def _log_step(message: str) -> None:
        elapsed = time.monotonic() - eval_start_time
        print(f"[eval +{elapsed:6.1f}s] {message}", flush=True)

    _log_step("reloading modal volumes")
    tablebank_vol.reload()
    artifacts_vol.reload()
    _log_step("volumes reloaded")

    dataset_root_path = Path(dataset_root)
    annotations_root_path = Path(annotations_path)
    output_dir_path = _prepare_output_dir(
        Path(output_dir),
        score_threshold=score_threshold,
        unique_output_dir=unique_output_dir,
    )

    resolved_weights_path = Path(weights_path)
    if not resolved_weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {resolved_weights_path}")

    resolved_annotations_input, annotations_format = _resolve_annotations_source(
        annotations_root_path,
        dataset_root_path,
    )
    if dataset_fraction <= 0.0 or dataset_fraction > 1.0:
        raise ValueError("dataset_fraction must be in the range (0.0, 1.0].")
    if annotations_format == "voc":
        resolved_annotations_path = output_dir_path / "tablebank_annotations.coco.json"
        convert_voc_to_coco(
            resolved_annotations_input,
            resolved_annotations_path,
            single_category_name="table",
        )
    else:
        resolved_annotations_path = resolved_annotations_input

    _log_step(f"loading COCO annotations from {resolved_annotations_path}")
    coco_dataset = COCODataset.from_json(resolved_annotations_path)
    _log_step(
        f"loaded COCO annotations: {len(coco_dataset.images)} images, "
        f"{len(coco_dataset.annotations)} annotations"
    )

    _log_step(f"indexing image files under {dataset_root_path}")
    image_lookup: dict[str, Path] = {}
    image_root = dataset_root_path / "images"
    if image_root.exists():
        for path in image_root.rglob("*"):
            if path.is_file():
                image_lookup[path.name] = path
    else:
        for path in dataset_root_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_lookup[path.name] = path
    _log_step(f"indexed {len(image_lookup)} image files")

    detection_images: list[DetectionImage] = []
    missing_files: list[str] = []
    for image_entry in coco_dataset.images:
        image_path = image_lookup.get(Path(image_entry.file_name).name)
        if image_path is None:
            missing_files.append(image_entry.file_name)
            continue
        detection_images.append(
            DetectionImage(
                image_id=image_entry.id,
                file_name=image_entry.file_name,
                width=image_entry.width,
                height=image_entry.height,
                image_path=image_path,
            )
        )

    if missing_files:
        preview = ", ".join(missing_files[:10])
        raise FileNotFoundError(
            f"Could not locate {len(missing_files)} image(s) under {dataset_root_path}. "
            f"First missing entries: {preview}"
        )

    detection_images = _select_detection_subset(
        detection_images,
        dataset_fraction=dataset_fraction,
        sample_seed=sample_seed,
        holdout_train_fraction=holdout_train_fraction,
        holdout_train_seed=holdout_train_seed,
    )
    if holdout_train_fraction > 0.0:
        _log_step(
            f"holdout filter applied: excluded train sample of "
            f"{holdout_train_fraction:.4f} (seed={holdout_train_seed}); "
            f"{len(detection_images)} images remain for eval"
        )

    selected_image_ids = {item.image_id for item in detection_images}
    selected_annotations = [
        annotation.model_dump(mode="json")
        for annotation in coco_dataset.annotations
        if annotation.image_id in selected_image_ids
    ]
    selected_images = [
        image.model_dump()
        for image in coco_dataset.images
        if image.id in selected_image_ids
    ]
    selected_categories = [category.model_dump() for category in coco_dataset.categories]

    class _TableBankInferenceDataset(Dataset):
        def __init__(self, items: list[DetectionImage], normalized_query: str):
            self.items = items
            self.query = normalized_query
            self.resolution = 1008
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> Datapoint:
            item = self.items[idx]
            image = PILImage.open(item.image_path).convert("RGB")
            resized = image.resize((self.resolution, self.resolution), PILImage.BILINEAR)
            image_tensor = self.transform(resized)
            image_obj = Image(data=image_tensor, objects=[], size=(self.resolution, self.resolution))
            query = FindQueryLoaded(
                query_text=self.query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=item.image_id,
                    original_image_id=item.image_id,
                    original_category_id=1,
                    original_size=(item.height, item.width),
                    object_id=-1,
                    frame_index=-1,
                ),
            )
            return Datapoint(find_queries=[query], images=[image_obj], raw_images=[resized])

    def _encode_predictions_for_items(
        items: list[DetectionImage],
        predictions_list: list[dict[str, torch.Tensor]],
        pred_id_start: int,
    ) -> tuple[list[dict[str, Any]], int]:
        encoded: list[dict[str, Any]] = []
        pred_id = pred_id_start
        for item, preds in zip(items, predictions_list):
            if preds is None or len(preds.get("pred_logits", [])) == 0:
                continue

            processed_predictions = postprocess_sam3_predictions(
                pred_logits=preds["pred_logits"],
                pred_masks=preds["pred_masks"],
                original_size=(item.height, item.width),
                score_threshold=score_threshold,
                duplicate_iou_threshold=duplicate_iou_threshold,
                min_box_area=min_box_area,
            )

            for processed_prediction in processed_predictions:
                x, y, w, h = processed_prediction.bbox_xywh
                mask_np = processed_prediction.binary_mask.numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle["counts"] = rle["counts"].decode("utf-8")
                encoded.append(
                    {
                        "id": pred_id,
                        "image_id": item.image_id,
                        "category_id": 1,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(processed_prediction.score),
                        "segmentation": rle,
                    }
                )
                pred_id += 1

        return encoded, pred_id

    device_obj = _resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    _log_step(f"resolved device: {device_obj}")
    config = _load_config(resolved_weights_path, None)
    _log_step(
        "building SAM3 image model "
        "(load_from_HF=True; first run downloads HF weights)"
    )
    model = build_sam3_image_model(
        device=device_obj.type,
        compile=False,
        load_from_HF=True,
        bpe_path=resolve_bpe_vocab_path(),
        eval_mode=True,
    )
    _log_step("SAM3 base model built")
    lora_cfg = config.lora
    model = apply_lora_to_model(
        model,
        LoRALayerConfig(
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
        ),
    )
    _log_step(f"loading LoRA weights from {resolved_weights_path}")
    lora_state_dict = torch.load(resolved_weights_path, map_location=device_obj)
    model.load_state_dict(lora_state_dict, strict=False)
    _log_step(f"moving model to {device_obj}")
    model.to(device_obj)
    model.eval()
    _log_step("model ready on device; starting inference loop")

    normalized_query = query_text.strip().lower() or "table"
    dataset = _TableBankInferenceDataset(detection_images, normalized_query)
    predictions: list[dict[str, Any]] = []
    next_prediction_id = 0
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    log_every_batches = max(1, total_batches // 20)
    inference_start_time = time.monotonic()

    for batch_idx, start in enumerate(range(0, len(dataset), batch_size)):
        end = min(start + batch_size, len(dataset))
        batch_items = detection_images[start:end]
        batch = collate_fn_api(
            [dataset[idx] for idx in range(start, end)],
            dict_key="input",
            with_seg_masks=True,
        )
        input_batch = _move_to_device(batch["input"], device_obj)
        with torch.inference_mode():
            outputs = model(input_batch)

        with SAM3Output.iteration_mode(outputs, SAM3Output.IterMode.LAST_STEP_PER_STAGE):
            final_stage = list(outputs)[-1]

        pred_logits = final_stage["pred_logits"]
        pred_boxes = final_stage["pred_boxes"]
        pred_masks = final_stage["pred_masks"]
        if pred_logits.dim() == 2:
            predictions_list = [
                {
                    "pred_logits": pred_logits.detach(),
                    "pred_boxes": pred_boxes.detach(),
                    "pred_masks": pred_masks.detach(),
                }
            ]
        else:
            predictions_list = [
                {
                    "pred_logits": pred_logits[idx].detach(),
                    "pred_boxes": pred_boxes[idx].detach(),
                    "pred_masks": pred_masks[idx].detach(),
                }
                for idx in range(pred_logits.shape[0])
            ]

        encoded_batch, next_prediction_id = _encode_predictions_for_items(
            batch_items,
            predictions_list,
            next_prediction_id,
        )
        predictions.extend(encoded_batch)

        completed = batch_idx + 1
        if completed == 1 or completed % log_every_batches == 0 or completed == total_batches:
            inference_elapsed = time.monotonic() - inference_start_time
            images_done = min(completed * batch_size, len(dataset))
            rate = images_done / max(inference_elapsed, 1e-6)
            _log_step(
                f"inference batch {completed}/{total_batches} "
                f"({images_done}/{len(dataset)} images, {rate:.2f} img/s)"
            )

    _log_step(f"inference complete; {len(predictions)} predictions collected")
    predictions_path = output_dir_path / "predictions.coco.json"
    predictions_path.write_text(json.dumps(predictions, indent=2, default=_json_default))

    gt_payload = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": selected_categories,
        "info": {"description": "TableBank evaluation ground truth"},
    }
    gt_path = output_dir_path / "ground_truth.coco.json"
    gt_path.write_text(json.dumps(gt_payload, indent=2))

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(predictions_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    prf_metrics = _match_detections_at_iou50(
        predictions=predictions,
        annotations=selected_annotations,
    )
    visualization_summary = _render_eval_visualizations(
        detection_images=detection_images,
        predictions=predictions,
        annotations=selected_annotations,
        output_dir_path=output_dir_path,
        max_images=visualize_max_images,
    )
    metrics = {
        "map": float(coco_eval.stats[0]),
        "map50": float(coco_eval.stats[1]),
        "map75": float(coco_eval.stats[2]),
        "precision_iou50": float(prf_metrics["precision_iou50"]),
        "recall_iou50": float(prf_metrics["recall_iou50"]),
        "f1_iou50": float(prf_metrics["f1_iou50"]),
        "true_positives": int(prf_metrics["true_positives"]),
        "false_positives": int(prf_metrics["false_positives"]),
        "false_negatives": int(prf_metrics["false_negatives"]),
    }
    summary = {
        "summary_version": "tablebank_eval.v1",
        "baseline_id": "sam3lora",
        "baseline_name": "SAM3 LoRA",
        "model_family": "sam3",
        "weights_path": str(resolved_weights_path),
        "config_path": str(resolved_weights_path.with_name("run_config.json")),
        "dataset_root": str(dataset_root_path),
        "annotations_path": str(resolved_annotations_path),
        "output_dir": str(output_dir_path),
        "num_images": len(detection_images),
        "num_predictions": len(predictions),
        "score_threshold": score_threshold,
        "duplicate_iou_threshold": duplicate_iou_threshold,
        "min_box_area": min_box_area,
        "unique_output_dir": unique_output_dir,
        "dataset_fraction": dataset_fraction,
        "sample_seed": sample_seed,
        "holdout_train_fraction": holdout_train_fraction,
        "holdout_train_seed": holdout_train_seed,
        "query_text": normalized_query,
        "model": {
            "family": "sam3",
            "variant": "sam3-lora",
            "weights_path": str(resolved_weights_path),
            "config_path": str(resolved_weights_path.with_name("run_config.json")),
            "device": device_obj.type,
        },
        "evaluation": {
            "task": "table_detection",
            "dataset_name": "TableBank",
            "dataset_root": str(dataset_root_path),
            "annotations_path": str(resolved_annotations_path),
            "output_dir": str(output_dir_path),
            "num_images": len(detection_images),
            "num_predictions": len(predictions),
            "score_threshold": score_threshold,
            "dataset_fraction": dataset_fraction,
            "sample_seed": sample_seed,
            "holdout_train_fraction": holdout_train_fraction,
            "holdout_train_seed": holdout_train_seed,
            "visualize_max_images": visualize_max_images,
        },
        "parameters": {
            "query_text": normalized_query,
            "batch_size": batch_size,
            "duplicate_iou_threshold": duplicate_iou_threshold,
            "min_box_area": min_box_area,
            "unique_output_dir": unique_output_dir,
        },
        "metrics": metrics,
        "artifacts": {
            "predictions_coco_json": str(predictions_path),
            "ground_truth_coco_json": str(gt_path),
            "visualizations_dir": visualization_summary["visualizations_dir"],
            "metrics_json": str(output_dir_path / "metrics.json"),
        },
        "visualizations": visualization_summary,
    }
    metrics_path = output_dir_path / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, default=_json_default))
    artifacts_vol.commit()
    return summary


@app.local_entrypoint()
def main(
    weights: str = "",
    dataset_root: str = "/data/tablebank/extracted/TableBank/TableBank/Detection",
    annotations: str = "/data/tablebank/extracted/TableBank/TableBank/Detection/annotations",
    output_dir: str = "/artifacts/tablebank_eval",
    score_threshold: float = 0.25,
    unique_output_dir: bool = False,
    dataset_fraction: float = 1.0,
    sample_seed: int | None = None,
    holdout_train_fraction: float = 0.0,
    holdout_train_seed: int | None = None,
    duplicate_iou_threshold: float = 0.5,
    min_box_area: float = 16.0,
    query_text: str = "table",
    batch_size: int = 8,
    visualize_max_images: int = 20,
    use_current_leader: bool = False,
    study_name: str = DEFAULT_STUDY_NAME,
    include_running_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    num_rung_stages: int = DEFAULT_NUM_RUNG_STAGES,
    show_current_leader_only: bool = False,
):
    if show_current_leader_only:
        result = describe_current_leader.remote(
            study_name=study_name,
            include_running_trials=include_running_trials,
            sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
            num_rung_stages=num_rung_stages,
        )
    elif use_current_leader:
        result = run_tablebank_eval_on_current_leader.remote(
            study_name=study_name,
            dataset_root=dataset_root,
            annotations_path=annotations,
            output_root_dir=output_dir,
            score_threshold=score_threshold,
            query_text=query_text,
            batch_size=batch_size,
            visualize_max_images=visualize_max_images,
            dataset_fraction=dataset_fraction,
            sample_seed=sample_seed,
            duplicate_iou_threshold=duplicate_iou_threshold,
            min_box_area=min_box_area,
            include_running_trials=include_running_trials,
            sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
            num_rung_stages=num_rung_stages,
        )
    else:
        if not weights:
            raise ValueError(
                "weights is required unless use_current_leader=True."
            )
        result = run_tablebank_eval.remote(
            weights_path=weights,
            dataset_root=dataset_root,
            annotations_path=annotations,
            output_dir=output_dir,
            score_threshold=score_threshold,
            unique_output_dir=unique_output_dir,
            dataset_fraction=dataset_fraction,
            sample_seed=sample_seed,
            holdout_train_fraction=holdout_train_fraction,
            holdout_train_seed=holdout_train_seed,
            duplicate_iou_threshold=duplicate_iou_threshold,
            min_box_area=min_box_area,
            query_text=query_text,
            batch_size=batch_size,
            visualize_max_images=visualize_max_images,
        )
    print(json.dumps(result, indent=2, default=_json_default))

