from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_serializer, field_validator


class LRScheduler(str, Enum):
    COSINE = "cosine"
    LINEAR = "linear"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class MixedPrecision(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    NO = "no"


class EvalMetric(str, Enum):
    IOU = "iou"
    DICE = "dice"
    ACCURACY = "accuracy"


class Device(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


class ModelConfig(BaseModel):
    name: str = "facebook/sam3"
    cache_dir: Optional[str] = None


class LoRAConfig(BaseModel):
    rank: int = Field(32, gt=0, description="LoRA rank (dimension of low-rank matrices)")
    alpha: int = Field(64, gt=0, description="LoRA scaling factor, typically 2x rank")
    dropout: float = Field(0.1, ge=0.0, le=1.0)

    target_modules: list[str] = Field(
        default=[
            "q_proj", "k_proj", "v_proj", "out_proj",
            "qkv", "proj", "fc1", "fc2",
            "c_fc", "c_proj",
            "linear1", "linear2",
        ],
    )

    apply_to_vision_encoder: bool = True
    apply_to_text_encoder: bool = True
    apply_to_geometry_encoder: bool = True
    apply_to_detr_encoder: bool = True
    apply_to_detr_decoder: bool = True
    apply_to_mask_decoder: bool = True


class DatasetSplit(BaseModel):
    """Paths for a single dataset split (train or valid)."""

    image_dir: Path = Field(..., description="Directory containing the images")
    annotation_file: Path = Field(
        ..., description="Path to the COCO-format annotation JSON"
    )

    @field_serializer("image_dir", "annotation_file")
    @classmethod
    def _serialize_as_posix(cls, v: Path) -> str:
        """Always serialize with forward slashes so configs created on
        Windows work inside Linux containers."""
        return v.as_posix()

    @field_validator("annotation_file")
    @classmethod
    def annotation_must_be_json(cls, v: Path) -> Path:
        if v.suffix.lower() != ".json":
            raise ValueError(f"annotation_file must be a .json file, got '{v.suffix}'")
        return v


class DataConfig(BaseModel):
    """Dataset configuration with explicit paths per split."""

    train: DatasetSplit
    valid: Optional[DatasetSplit] = None
    sample_percent: float = Field(
        100.0, gt=0.0, le=100.0,
        description="Percentage of TRAIN split to use (e.g. 10.0 = 10%)",
    )
    valid_sample_percent: float = Field(
        100.0, gt=0.0, le=100.0,
        description="Percentage of VALID split to use (fixed holdout subset if <100)",
    )
    valid_sample_seed: int = Field(
        42,
        description="Seed used when subsampling the VALID split for deterministic holdouts",
    )


class TrainingConfig(BaseModel):
    data: DataConfig
    batch_size: int = Field(4, gt=0)
    val_batch_size: Optional[int] = Field(
        None,
        gt=0,
        description="Validation batch size override. Uses batch_size when unset.",
    )
    num_workers: int = Field(2, ge=0)

    learning_rate: float = Field(5e-5, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    adam_beta1: float = Field(0.9, ge=0, lt=1)
    adam_beta2: float = Field(0.999, ge=0, lt=1)
    adam_epsilon: float = Field(1e-8, gt=0)
    max_grad_norm: float = Field(1.0, gt=0)

    num_epochs: int = Field(100, gt=0)
    warmup_steps: int = Field(200, ge=0)
    lr_scheduler: LRScheduler = LRScheduler.COSINE
    lr_scheduler_lifetime_multiplier: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Multiplier applied to the current stage's ``total_optimizer_steps`` "
            "when building the LR schedule. Set this when training is invoked "
            "in stages so the scheduler curve is computed against the lifetime "
            "step budget across all stages, instead of just the current "
            "stage's ``num_epochs``. The sweep computes this from the "
            "(num_epochs * sample_percent) ratios of the full schedule, so the "
            "trainer does not need to know dataset size. Resumed scheduler "
            "state from a previous stage's checkpoint then lines up with the "
            "same underlying curve."
        ),
    )

    logging_steps: int = Field(10, gt=0)
    eval_steps: int = Field(100, gt=0)
    eval_every_n_epochs: int = Field(
        1,
        gt=0,
        description="Run end-of-epoch validation every N epochs (always runs on final epoch).",
    )
    save_steps: int = Field(100, gt=0)
    save_total_limit: int = Field(5, gt=0)

    mixed_precision: MixedPrecision = MixedPrecision.BF16
    seed: int = 42
    gradient_accumulation_steps: int = Field(8, gt=0)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


class OutputConfig(BaseModel):
    output_dir: str = "outputs/sam3_lora_full"
    logging_dir: str = "logs"
    save_lora_only: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    @field_validator("hub_model_id")
    @classmethod
    def hub_id_required_when_pushing(cls, v: Optional[str], info) -> Optional[str]:
        if info.data.get("push_to_hub") and not v:
            raise ValueError("hub_model_id is required when push_to_hub is True")
        return v


class EvaluationConfig(BaseModel):
    metric: EvalMetric = EvalMetric.IOU
    save_predictions: bool = False
    compute_metrics_during_training: bool = True


class HardwareConfig(BaseModel):
    device: Device = Device.CUDA
    dataloader_pin_memory: bool = True
    use_compile: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = True
    float32_matmul_precision: str = Field(
        "high",
        description="torch.set_float32_matmul_precision value: highest, high, or medium",
    )

    @field_validator("float32_matmul_precision")
    @classmethod
    def matmul_precision_must_be_valid(cls, v: str) -> str:
        allowed = {"highest", "high", "medium"}
        if v not in allowed:
            raise ValueError(f"float32_matmul_precision must be one of {sorted(allowed)}")
        return v


class SAM3LoRAConfig(BaseModel):
    """Top-level configuration for SAM3 LoRA training."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SAM3LoRAConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
            )


if __name__ == "__main__":
    config = SAM3LoRAConfig.from_yaml("full_lora_config.yaml")
    print(config.model_dump_json(indent=2))
