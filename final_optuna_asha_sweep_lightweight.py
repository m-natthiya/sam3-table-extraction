"""Lightweight clone of final_optuna_asha_sweep.py for smoke testing.

Goal:
- Run an extremely small Optuna + ASHA sweep that still exercises the same
  core functions and flow.
- Optionally call the in-progress model extraction path from eval_tablebank
  by invoking `describe_current_leader` after the sweep.

Usage:
    modal run final_optuna_asha_sweep_lightweight.py
"""

from __future__ import annotations

import copy
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import modal
import optuna

from sam3_table.cstone_train_sam3 import (
    MODAL_ARTIFACTS_DIR,
    app,
    artifacts_vol,
    train_sam3,
)
from sam3_table.training_config import SAM3LoRAConfig


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base (mutates and returns base)."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _commit_artifacts_volume_safely(context: str) -> None:
    """Commit the artifacts volume so other containers (e.g. a live leader
    monitor / benchmark) see the latest Optuna sqlite + checkpoint state.

    Never raises: a transient commit failure should not kill the sweep.
    """
    try:
        artifacts_vol.commit()
    except Exception as exc:  # pragma: no cover - best-effort sync
        print(f"[sweep-lightweight] artifacts_vol.commit() failed ({context}): {exc}")


def _close_optuna_storage(study: optuna.study.Study | None) -> None:
    """Dispose Optuna's SQLAlchemy engine so the underlying sqlite file
    is no longer held open.

    Modal's ``Volume.reload()`` refuses to operate while any file in the
    volume is open. After ``study.optimize()`` returns, Optuna's engine
    still has ``/artifacts/optuna/<study>.db`` open, which triggers
    ``ConflictError("there are open files preventing the operation: path
    optuna/<study>.db is open")``. ``engine.dispose()`` closes all pooled
    connections; later reads from the same ``study`` object will silently
    open fresh connections (which is fine -- by then the reload has
    completed).
    """
    if study is None:
        return
    import gc

    storage = getattr(study, "_storage", None)
    if storage is None:
        return

    engine = None
    for attr_chain in (
        ("engine",),
        ("_backend", "engine"),
        ("scoped_session", "bind"),
    ):
        obj: Any = storage
        try:
            for attr in attr_chain:
                obj = getattr(obj, attr)
            engine = obj
            break
        except AttributeError:
            continue
    if engine is not None and hasattr(engine, "dispose"):
        try:
            engine.dispose()
        except Exception as exc:  # pragma: no cover - best-effort sync
            print(f"[sweep-lightweight] optuna engine.dispose() failed: {exc}")
    if hasattr(storage, "remove_session"):
        try:
            storage.remove_session()
        except Exception:  # pragma: no cover - best-effort sync
            pass
    gc.collect()


def _commit_and_reload_artifacts(
    context: str,
    study: optuna.study.Study | None = None,
) -> None:
    """Commit pending writes, dispose any open Optuna sqlite handles, then
    reload the volume. Reload failures are logged and swallowed so a rare
    open-file edge case never crashes the sweep -- the remote
    leader-extraction container does its own reload at startup, and the
    canonical_leader fallback in this file already advertises the
    conventional weights path when local visibility lags.
    """
    _commit_artifacts_volume_safely(context)
    _close_optuna_storage(study)
    try:
        artifacts_vol.reload()
    except Exception as exc:  # pragma: no cover - best-effort sync
        print(f"[sweep-lightweight] artifacts_vol.reload() skipped ({context}): {exc}")


_RESUMABLE_CHECKPOINT_FILES = (
    "checkpoint_epoch.pt",
    "checkpoint_best.pt",
    "checkpoint_signal.pt",
)


def _hashable_params(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Stable, hashable representation of an Optuna ``trial.params`` dict."""
    return tuple(sorted(params.items(), key=lambda kv: kv[0]))


def _salvage_orphaned_running_trials(
    study: optuna.study.Study,
    num_rung_stages: int,
) -> dict[tuple[tuple[str, Any], ...], dict[str, Any]]:
    """Recover trials stuck in ``RUNNING`` from a previous aborted sweep.

    Mirrors the helper in ``final_optuna_asha_sweep.py``: marks orphaned
    RUNNING trials as FAIL, re-enqueues a fresh trial with identical
    hyperparameters, and returns ``params_key -> resume_hint`` so the
    objective can pass ``resume_output_dir`` to ``train_sam3.remote()``
    and pick up from the last committed checkpoint instead of restarting
    training from epoch 0.
    """
    if num_rung_stages < 1:
        return {}

    salvage_hints: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    salvaged_count = 0
    for trial in list(study.trials):
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue

        last_recorded_stage = 0
        last_output_dir: str | None = None
        for s in range(num_rung_stages, 0, -1):
            recorded = trial.user_attrs.get(f"stage_{s}_output_dir")
            if isinstance(recorded, str) and recorded:
                last_recorded_stage = s
                last_output_dir = recorded
                break

        if last_output_dir is not None:
            checkpoint_present = any(
                (Path(last_output_dir) / name).exists()
                for name in _RESUMABLE_CHECKPOINT_FILES
            )
            if not checkpoint_present:
                last_output_dir = None
                last_recorded_stage = 0

        try:
            study._storage.set_trial_state_values(
                trial._trial_id,
                state=optuna.trial.TrialState.FAIL,
            )
        except Exception as exc:
            print(
                f"[sweep-lightweight][salvage] failed to mark trial "
                f"{trial.number} FAIL: {exc}"
            )

        try:
            study.enqueue_trial(trial.params, skip_if_exists=False)
        except Exception as exc:
            print(
                f"[sweep-lightweight][salvage] failed to enqueue salvage "
                f"trial for orphan {trial.number}: {exc}"
            )
            continue

        params_key = _hashable_params(trial.params)
        salvage_hints[params_key] = {
            "original_trial_number": trial.number,
            "last_recorded_stage": last_recorded_stage,
            "resume_output_dir": last_output_dir,
        }
        salvaged_count += 1
        print(
            f"[sweep-lightweight][salvage] orphan trial {trial.number}: "
            f"FAIL+re-enqueued with same params, resume_dir="
            f"{last_output_dir or '<none>'} (last completed stage="
            f"{last_recorded_stage})"
        )

    if salvaged_count:
        print(
            f"[sweep-lightweight][salvage] re-enqueued {salvaged_count} "
            "orphaned trial(s) from a previous aborted run."
        )
    else:
        print("[sweep-lightweight][salvage] no orphaned RUNNING trials detected.")
    return salvage_hints


# Lightweight smoke schedule: very small subsets, 2 rungs.
PROMOTION_FRACTION = 0.5
REDUCTION_FACTOR = int(round(1 / PROMOTION_FRACTION))
SAMPLE_SCHEDULE = [0.1, 0.2]
VALID_SAMPLE_SCHEDULE = [0.5, 1.0]
EPOCH_RATIO_SCHEDULE = [0.5, 1.0]  # with max_epochs=2 -> epochs 1, 2
EVAL_EVERY_N_EPOCHS_SCHEDULE = [1, 1]
NUM_RUNG_STAGES = len(SAMPLE_SCHEDULE)
SWEEP_EVAL_STEPS = 10_000_000


def _build_stage_schedule(max_epochs: int) -> list[dict[str, float | int]]:
    if not (
        len(SAMPLE_SCHEDULE)
        == len(VALID_SAMPLE_SCHEDULE)
        == len(EPOCH_RATIO_SCHEDULE)
        == len(EVAL_EVERY_N_EPOCHS_SCHEDULE)
    ):
        raise ValueError("Stage schedules must have matching lengths.")

    stages: list[dict[str, float | int]] = []
    prev_epochs = 0
    for sample_percent, valid_sample_percent, eval_every_n_epochs, ratio in zip(
        SAMPLE_SCHEDULE,
        VALID_SAMPLE_SCHEDULE,
        EVAL_EVERY_N_EPOCHS_SCHEDULE,
        EPOCH_RATIO_SCHEDULE,
    ):
        epoch_budget = max(1, int(round(max_epochs * ratio)))
        epoch_budget = max(epoch_budget, prev_epochs + 1)
        stages.append(
            {
                "sample_percent": sample_percent,
                "valid_sample_percent": valid_sample_percent,
                "eval_every_n_epochs": eval_every_n_epochs,
                "num_epochs": epoch_budget,
            }
        )
        prev_epochs = epoch_budget

    stage_weights = [
        max(1e-9, stage["num_epochs"] * stage["sample_percent"])
        for stage in stages
    ]
    total_weight = sum(stage_weights)
    for stage, weight in zip(stages, stage_weights):
        stage["lr_scheduler_lifetime_multiplier"] = (
            float(total_weight / weight) if weight > 0 else 1.0
        )
    return stages


def _suggest_trial_overrides(trial: optuna.trial.Trial) -> tuple[dict[str, Any], int]:
    """Tiny search space for fast smoke tests."""
    rank = trial.suggest_categorical("lora_rank", [8, 16])
    overrides: dict[str, Any] = {
        "lora": {
            "rank": rank,
            "alpha": rank * 2,
            "dropout": trial.suggest_float("lora_dropout", 0.0, 0.1),
            "apply_to_vision_encoder": True,
            "apply_to_text_encoder": False,
            "apply_to_geometry_encoder": False,
            "apply_to_detr_encoder": False,
            "apply_to_detr_decoder": False,
            "apply_to_mask_decoder": False,
        },
        "training": {
            "learning_rate": trial.suggest_float("learning_rate", 2e-5, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True),
            "batch_size": 8,
            "num_workers": 4,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "bf16",
            "seed": 42,
        },
        "output": {"output_dir": "outputs/final_optuna_asha_lightweight"},
    }
    return overrides, 2


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 10,
)
def read_latest_val_loss(output_dir: str) -> float:
    import torch

    artifacts_vol.reload()
    stats_path = Path(output_dir) / "val_stats.json"
    last_val_loss = None
    if stats_path.exists():
        for raw_line in stats_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "val_loss" in record:
                last_val_loss = float(record["val_loss"])
        if last_val_loss is not None:
            return last_val_loss

    # Fallback for runs where val_stats.json wasn't emitted: read checkpoint loss.
    ckpt_candidates = [
        Path(output_dir) / "checkpoint_best.pt",
        Path(output_dir) / "checkpoint_epoch.pt",
    ]
    for ckpt_path in ckpt_candidates:
        if not ckpt_path.exists():
            continue
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        best_val_loss = checkpoint.get("best_val_loss")
        if isinstance(best_val_loss, (int, float)):
            return float(best_val_loss)

    raise FileNotFoundError(
        f"Missing validation stats and checkpoint val loss under: {output_dir}"
    )


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 10,
)
def read_stage_quality_metrics(output_dir: str) -> dict[str, float | None]:
    output_path = Path(output_dir)
    metrics_path = output_path / "metrics.json"
    if not metrics_path.exists():
        return {"iou": None, "f1": None, "composite": None}

    payload = json.loads(metrics_path.read_text())
    if not isinstance(payload, dict):
        return {"iou": None, "f1": None, "composite": None}

    iou = payload.get("iou")
    f1 = payload.get("f1")
    iou_val = float(iou) if isinstance(iou, (int, float)) else None
    f1_val = float(f1) if isinstance(f1, (int, float)) else None
    composite = None
    if iou_val is not None and f1_val is not None:
        composite = 0.5 * iou_val + 0.5 * f1_val
    return {"iou": iou_val, "f1": f1_val, "composite": composite}


def _build_stage_config(
    base_config: SAM3LoRAConfig,
    trial_overrides: dict[str, Any],
    stage_cfg: dict[str, float | int],
    trial_number: int,
) -> dict[str, Any]:
    config_dict = copy.deepcopy(base_config.model_dump(mode="json"))
    _deep_merge(config_dict, trial_overrides)
    stage_overrides: dict[str, Any] = {
        "training": {
            "data": {
                "sample_percent": stage_cfg["sample_percent"],
                "valid_sample_percent": stage_cfg["valid_sample_percent"],
                "valid_sample_seed": 42,
            },
            "num_epochs": stage_cfg["num_epochs"],
            "eval_steps": SWEEP_EVAL_STEPS,
            "eval_every_n_epochs": stage_cfg["eval_every_n_epochs"],
        },
        "output": {"output_dir": f"outputs/final_optuna_asha_lightweight/trial_{trial_number:04d}"},
    }
    if "lr_scheduler_lifetime_multiplier" in stage_cfg:
        stage_overrides["training"]["lr_scheduler_lifetime_multiplier"] = float(
            stage_cfg["lr_scheduler_lifetime_multiplier"]
        )
    _deep_merge(config_dict, stage_overrides)
    return config_dict


def _objective_factory(
    base_config: SAM3LoRAConfig,
    salvage_hints: dict[tuple[tuple[str, Any], ...], dict[str, Any]] | None = None,
):
    salvage_hints = dict(salvage_hints or {})

    def objective(trial: optuna.trial.Trial) -> float:
        trial_overrides, max_epochs = _suggest_trial_overrides(trial)
        stage_schedule = _build_stage_schedule(max_epochs=max_epochs)
        final_stage_val_loss = None

        # If this trial's params match a salvage hint from a previously
        # aborted run, point stage 1 at the last committed checkpoint dir.
        params_key = _hashable_params(trial.params)
        salvage = salvage_hints.pop(params_key, None)
        if salvage is not None:
            trial.set_user_attr("salvage_origin_trial_number", salvage["original_trial_number"])
            trial.set_user_attr("salvage_last_recorded_stage", salvage["last_recorded_stage"])
            trial.set_user_attr("salvage_resume_output_dir", salvage["resume_output_dir"])
            print(
                f"[sweep-lightweight][salvage] trial {trial.number} is a salvage "
                f"of {salvage['original_trial_number']}; resume_dir="
                f"{salvage['resume_output_dir'] or '<none>'}"
            )

        resume_output_dir = (
            salvage["resume_output_dir"] if salvage is not None else None
        )

        for stage_idx, stage_cfg in enumerate(stage_schedule, start=1):
            stage_config = _build_stage_config(
                base_config=base_config,
                trial_overrides=trial_overrides,
                stage_cfg=stage_cfg,
                trial_number=trial.number,
            )
            run_result = train_sam3.remote(
                stage_config,
                fresh_run=(stage_idx == 1) and (resume_output_dir is None),
                resume_output_dir=resume_output_dir,
            )
            resume_output_dir = run_result["output_dir"]
            stage_val_loss = read_latest_val_loss.remote(run_result["output_dir"])
            final_stage_val_loss = stage_val_loss

            trial.set_user_attr(f"stage_{stage_idx}_output_dir", run_result["output_dir"])
            trial.set_user_attr(f"stage_{stage_idx}_val_loss", stage_val_loss)
            trial.report(stage_val_loss, step=stage_idx)

            # Flush per-stage state to the shared volume so a live
            # describe-current-leader / benchmark call sees this stage
            # as soon as it finishes. Consistent with the full sweep.
            _commit_artifacts_volume_safely(
                f"trial={trial.number}, end of stage {stage_idx}"
            )

            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at stage {stage_idx}: val_loss={stage_val_loss:.6f}")

        if final_stage_val_loss is None:
            raise RuntimeError("Trial completed without a stage loss.")
        return float(final_stage_val_loss)

    return objective


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 20,
)
def check_in_progress_model_extraction(
    study_name: str,
    include_running_trials: bool = True,
    include_non_complete_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    num_rung_stages: int = NUM_RUNG_STAGES,
    prefer_complete_retry_seconds: int = 3,
) -> dict[str, Any]:
    """Exercise eval_tablebank's in-progress model extraction path.

    Primary path: resolve the deployed ``tablebank-eval`` Modal app and call
    its ``describe_current_leader`` function. Only a *lookup* failure (the
    app hasn't been deployed in this environment) triggers the local
    fallback below -- other errors are allowed to propagate so real bugs
    surface in ``model_extraction_error``.
    """
    artifacts_vol.reload()
    remote_exc: BaseException | None = None
    try:
        describe_fn = modal.Function.from_name("tablebank-eval", "describe_current_leader")
        result = describe_fn.remote(
            study_name=study_name,
            include_running_trials=include_running_trials,
            sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
            num_rung_stages=num_rung_stages,
        )
        if isinstance(result, dict):
            result.setdefault("fallback_used", False)
            result.setdefault(
                "fallback_reason",
                "Resolved via deployed tablebank-eval.describe_current_leader.",
            )
        return result
    except modal.exception.NotFoundError as lookup_exc:
        remote_exc = lookup_exc

    # Fallback only runs when the deployed tablebank-eval app can't be found.
    # Future users should not hit this because `ensure_deployed()` is called
    # from the local entrypoint, but we keep the fallback as a safety net
    # for direct `run_lightweight_optuna_test.remote(...)` callers.
    assert remote_exc is not None
    sqlite_lock_timeout_sec = max(1, int(sqlite_lock_timeout_sec))
    num_rung_stages = max(1, int(num_rung_stages))

    sqlite_path = Path(MODAL_ARTIFACTS_DIR) / "optuna" / f"{study_name}.db"
    if not sqlite_path.exists():
        return {
            "study_name": study_name,
            "leader_trial_number": None,
            "leader_state": None,
            "leader_score": None,
            "objective_mode": None,
            "weights_path": None,
            "output_dir": None,
            "hyperparameters": {},
            "fallback_used": True,
            "fallback_reason": (
                f"{type(remote_exc).__name__}: {remote_exc}. "
                f"Also could not find SQLite study DB at {sqlite_path}."
            ),
        }

    with sqlite3.connect(str(sqlite_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.commit()

    storage_url = f"sqlite:///{sqlite_path.as_posix()}?timeout={sqlite_lock_timeout_sec}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    best_row: dict[str, Any] | None = None

    def _collect_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        complete_rows: list[dict[str, Any]] = []
        fallback_rows: list[dict[str, Any]] = []
        for trial in study.trials:
            score: float | None = None
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
                score = float(trial.value)
            elif (
                include_running_trials
                and trial.state == optuna.trial.TrialState.RUNNING
                and trial.intermediate_values
            ):
                latest_step = max(trial.intermediate_values.keys())
                score = float(trial.intermediate_values[latest_step])
            elif include_non_complete_trials:
                stage_loss_keys = sorted(
                    key
                    for key in trial.user_attrs.keys()
                    if key.startswith("stage_") and key.endswith("_val_loss")
                )
                if stage_loss_keys:
                    maybe_score = trial.user_attrs.get(stage_loss_keys[-1])
                    if isinstance(maybe_score, (int, float)):
                        score = float(maybe_score)

            if score is None:
                continue

            stage_output_dir = trial.user_attrs.get(f"stage_{num_rung_stages}_output_dir")
            if not isinstance(stage_output_dir, str):
                stage_keys = sorted(
                    key
                    for key in trial.user_attrs.keys()
                    if key.startswith("stage_") and key.endswith("_output_dir")
                )
                if stage_keys:
                    stage_output_dir = trial.user_attrs[stage_keys[-1]]

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
                "objective_mode": trial.user_attrs.get("objective_mode"),
                "hyperparameters": dict(trial.params),
            }
            if trial.state == optuna.trial.TrialState.COMPLETE:
                complete_rows.append(row)
            else:
                fallback_rows.append(row)
        return complete_rows, fallback_rows

    complete_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    for attempt in range(max(1, int(prefer_complete_retry_seconds))):
        complete_rows, fallback_rows = _collect_rows()
        if complete_rows:
            break
        # Right after optimize(), sqlite readers can observe stale RUNNING rows briefly.
        if attempt < max(1, int(prefer_complete_retry_seconds)) - 1:
            time.sleep(1.0)
            study = optuna.load_study(study_name=study_name, storage=storage_url)

    if complete_rows:
        best_row = min(complete_rows, key=lambda row: row["score"])
    elif fallback_rows:
        best_row = min(fallback_rows, key=lambda row: row["score"])

    if best_row is None:
        return {
            "study_name": study_name,
            "leader_trial_number": None,
            "leader_state": None,
            "leader_score": None,
            "objective_mode": None,
            "weights_path": None,
            "output_dir": None,
            "hyperparameters": {},
            "fallback_used": True,
            "fallback_reason": (
                f"{type(remote_exc).__name__}: {remote_exc}. "
                "No leader candidate with available LoRA weights found."
            ),
        }

    return {
        "study_name": study_name,
        "leader_trial_number": best_row["trial_number"],
        "leader_state": best_row["state"],
        "leader_score": best_row["score"],
        "objective_mode": best_row.get("objective_mode"),
        "weights_path": best_row["weights_path"],
        "output_dir": best_row["output_dir"],
        "hyperparameters": best_row.get("hyperparameters", {}),
        "fallback_used": True,
        "fallback_reason": f"{type(remote_exc).__name__}: {remote_exc}",
    }


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 60 * 2,
)
def run_lightweight_optuna_test(
    n_trials: int = 1,
    timeout_hours: float = 2.0,
    study_name: str = "sam3-final-optuna-asha-lightweight",
    parallel_workers: int = 1,
    sqlite_lock_timeout_sec: int = 60,
    test_model_extraction: bool = True,
    pruner_type: str = "none",
    reset_study: bool = False,
    include_non_complete_leader_trials: bool = True,
    base_config_dict: dict[str, Any] | None = None,
    salvage_orphaned_trials: bool = True,
) -> dict[str, Any]:
    """Run a tiny Optuna sweep and optionally test leader extraction."""
    if base_config_dict is not None:
        base_config = SAM3LoRAConfig.model_validate(base_config_dict)
    else:
        config_path = (
            Path(__file__).resolve().parent / "sam3_table" / "testSamples" / "full_lora_config.yaml"
        )
        base_config = SAM3LoRAConfig.from_yaml(config_path)

    parallel_workers = max(1, int(parallel_workers))
    sqlite_lock_timeout_sec = max(1, int(sqlite_lock_timeout_sec))
    optuna_dir = Path(MODAL_ARTIFACTS_DIR) / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = optuna_dir / f"{study_name}.db"
    if reset_study and sqlite_path.exists():
        sqlite_path.unlink()

    with sqlite3.connect(str(sqlite_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.commit()

    storage_url = f"sqlite:///{sqlite_path.as_posix()}?timeout={sqlite_lock_timeout_sec}"
    existing_study = None
    if sqlite_path.exists():
        try:
            existing_study = optuna.load_study(study_name=study_name, storage=storage_url)
        except Exception:
            existing_study = None
    baseline_total_trials = len(existing_study.trials) if existing_study is not None else 0

    sampler = optuna.samplers.TPESampler(n_startup_trials=1, multivariate=False, seed=42)
    normalized_pruner = pruner_type.strip().lower()
    if normalized_pruner == "sha":
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=REDUCTION_FACTOR,
            min_early_stopping_rate=0,
        )
    elif normalized_pruner in {"none", "nop"}:
        pruner = optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unsupported pruner_type='{pruner_type}'. Use 'none' or 'sha'.")

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )

    # Recover any trials stuck in RUNNING from a previously aborted run so
    # this run can pick up where the kill happened (Modal budget cap, etc).
    salvage_hints: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    if salvage_orphaned_trials:
        salvage_hints = _salvage_orphaned_running_trials(
            study, num_rung_stages=NUM_RUNG_STAGES
        )
        _commit_artifacts_volume_safely("post-salvage state")
    else:
        print("[sweep-lightweight][salvage] disabled by salvage_orphaned_trials=False")

    objective = _objective_factory(base_config, salvage_hints=salvage_hints)

    def _commit_after_trial(
        _study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        """Optuna callback: flush the artifacts volume after each finished
        trial so a live leader-monitor / benchmark call can see the
        latest finalized result.
        """
        _commit_artifacts_volume_safely(
            f"trial={trial.number} state={trial.state.name}"
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=int(timeout_hours * 3600),
        gc_after_trial=True,
        show_progress_bar=True,
        n_jobs=parallel_workers,
        callbacks=[_commit_after_trial],
    )

    # Flush sqlite writes so other containers see this run's trials, then
    # dispose Optuna's open sqlite handle and reload so this container can
    # see weight files committed by train_sam3 in *its* containers. The
    # close-before-reload step is critical: Modal's Volume.reload() throws
    # ConflictError("there are open files preventing the operation: path
    # optuna/<study>.db is open") otherwise.
    _commit_and_reload_artifacts(
        "end of run_lightweight_optuna_test", study=study
    )

    extraction_result: dict[str, Any] | None = None
    extraction_error: str | None = None
    if test_model_extraction:
        try:
            extraction_result = check_in_progress_model_extraction.remote(
                study_name=study_name,
                include_running_trials=False,
                include_non_complete_trials=include_non_complete_leader_trials,
                sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
                num_rung_stages=NUM_RUNG_STAGES,
            )
        except Exception as exc:  # pragma: no cover - best-effort smoke path
            extraction_error = str(exc)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    new_trials = study.trials[baseline_total_trials:]
    state_counts: dict[str, int] = {}
    new_state_counts: dict[str, int] = {}
    for trial in study.trials:
        state_counts[trial.state.name] = state_counts.get(trial.state.name, 0) + 1
    for trial in new_trials:
        new_state_counts[trial.state.name] = new_state_counts.get(trial.state.name, 0) + 1

    best_value = None
    best_trial_number = None
    if completed_trials and study.best_trial.value is not None:
        best_value = float(study.best_trial.value)
        best_trial_number = study.best_trial.number

    # Canonicalize leader from the finalized in-process study. The
    # in-memory `study.best_trial` is the source of truth for *this* run, so
    # we use it to override any stale/wrong leader returned by the remote
    # extraction. We still prefer a locally-visible weights file (the volume
    # has been reloaded above), but fall back to advertising the conventional
    # path so a missing/delayed file doesn't silently disable the override.
    canonical_leader = None
    if best_trial_number is not None:
        stage_output_dir = study.best_trial.user_attrs.get(f"stage_{NUM_RUNG_STAGES}_output_dir")
        if not isinstance(stage_output_dir, str):
            stage_keys = sorted(
                key
                for key in study.best_trial.user_attrs.keys()
                if key.startswith("stage_") and key.endswith("_output_dir")
            )
            if stage_keys:
                stage_output_dir = study.best_trial.user_attrs[stage_keys[-1]]

        if isinstance(stage_output_dir, str):
            weights_candidates = [
                Path(stage_output_dir) / "best_lora_weights.pt",
                Path(stage_output_dir) / "last_lora_weights.pt",
            ]
            visible_weights = next((p for p in weights_candidates if p.exists()), None)
            weights_path_str = (
                str(visible_weights)
                if visible_weights is not None
                else str(weights_candidates[0])
            )
            canonical_leader = {
                "study_name": study_name,
                "leader_trial_number": best_trial_number,
                "leader_state": "COMPLETE",
                "leader_score": best_value,
                "objective_mode": study.best_trial.user_attrs.get("objective_mode"),
                "weights_path": weights_path_str,
                "output_dir": stage_output_dir,
                "hyperparameters": dict(study.best_trial.params),
                "fallback_used": False,
                "fallback_reason": (
                    "Canonicalized from finalized local study state."
                    if visible_weights is not None
                    else (
                        "Canonicalized from finalized local study state. "
                        "Weights file not yet visible locally after reload; "
                        "advertising the conventional best_lora_weights.pt path."
                    )
                ),
            }

    if canonical_leader is not None:
        extracted_trial = (
            extraction_result.get("leader_trial_number")
            if isinstance(extraction_result, dict)
            else None
        )
        if extracted_trial != canonical_leader["leader_trial_number"]:
            if isinstance(extraction_result, dict):
                prior_reason = extraction_result.get("fallback_reason")
                if isinstance(prior_reason, str) and prior_reason:
                    canonical_leader["fallback_reason"] = (
                        f"{canonical_leader['fallback_reason']} Previous extraction: {prior_reason}"
                    )
            extraction_result = canonical_leader

    return {
        "study_name": study_name,
        "sqlite_path": str(sqlite_path),
        "pruner_type": normalized_pruner,
        "reset_study": reset_study,
        "num_trials_total": len(study.trials),
        "num_trials_in_run": len(new_trials),
        "num_trials_complete": len(completed_trials),
        "trial_state_counts": state_counts,
        "new_trial_state_counts": new_state_counts,
        "best_trial_number": best_trial_number,
        "best_value": best_value,
        "model_extraction_result": extraction_result,
        "model_extraction_error": extraction_error,
    }


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 10,
)
def describe_leader_from_study(
    study_name: str = "sam3-final-optuna-asha-lightweight",
    include_running_trials: bool = False,
    include_pruned_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    num_rung_stages: int = NUM_RUNG_STAGES,
    require_weights_on_disk: bool = True,
) -> dict[str, Any]:
    """Self-contained leader extraction (no ``tablebank-eval`` required).

    Mirrors :func:`describe_leader_from_study` in
    ``final_optuna_asha_sweep.py``. See that file for the full doc; this
    is just the lightweight sweep's version, so the default
    ``study_name`` and ``num_rung_stages`` differ.
    """
    artifacts_vol.reload()
    sqlite_lock_timeout_sec = max(1, int(sqlite_lock_timeout_sec))
    num_rung_stages = max(1, int(num_rung_stages))

    sqlite_path = Path(MODAL_ARTIFACTS_DIR) / "optuna" / f"{study_name}.db"
    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"Optuna study DB not found at {sqlite_path}. "
            "Make sure a sweep with this study_name has started and committed."
        )

    with sqlite3.connect(str(sqlite_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.commit()

    storage_url = f"sqlite:///{sqlite_path.as_posix()}?timeout={sqlite_lock_timeout_sec}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    candidates: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    for trial in study.trials:
        state_counts[trial.state.name] = state_counts.get(trial.state.name, 0) + 1

        score: float | None = None
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
        ):
            score = float(trial.value)
        elif (
            include_pruned_trials
            and trial.state == optuna.trial.TrialState.PRUNED
            and trial.intermediate_values
        ):
            latest_step = max(trial.intermediate_values.keys())
            score = float(trial.intermediate_values[latest_step])
        elif (
            include_running_trials
            and trial.state == optuna.trial.TrialState.RUNNING
            and trial.intermediate_values
        ):
            latest_step = max(trial.intermediate_values.keys())
            score = float(trial.intermediate_values[latest_step])

        if score is None:
            continue

        stage_output_dir = trial.user_attrs.get(f"stage_{num_rung_stages}_output_dir")
        if not isinstance(stage_output_dir, str):
            stage_keys = sorted(
                k
                for k in trial.user_attrs.keys()
                if k.startswith("stage_") and k.endswith("_output_dir")
            )
            if stage_keys:
                stage_output_dir = trial.user_attrs[stage_keys[-1]]
        if not isinstance(stage_output_dir, str):
            continue

        weights_candidates = [
            Path(stage_output_dir) / "best_lora_weights.pt",
            Path(stage_output_dir) / "last_lora_weights.pt",
        ]
        visible_weights = next(
            (p for p in weights_candidates if p.exists()), None
        )
        if visible_weights is None and require_weights_on_disk:
            continue
        weights_path_str = (
            str(visible_weights)
            if visible_weights is not None
            else str(weights_candidates[0])
        )

        candidates.append(
            {
                "trial_number": trial.number,
                "state": trial.state.name,
                "score": score,
                "output_dir": stage_output_dir,
                "weights_path": weights_path_str,
                "weights_visible_on_disk": visible_weights is not None,
                "objective_mode": trial.user_attrs.get("objective_mode"),
                "salvage_origin_trial_number": trial.user_attrs.get(
                    "salvage_origin_trial_number"
                ),
                "hyperparameters": dict(trial.params),
            }
        )

    if not candidates:
        return {
            "study_name": study_name,
            "sqlite_path": str(sqlite_path),
            "total_trials": len(study.trials),
            "trial_state_counts": state_counts,
            "leader_trial_number": None,
            "leader_state": None,
            "leader_score": None,
            "weights_path": None,
            "output_dir": None,
            "hyperparameters": {},
            "reason": (
                "No trial with a usable score and weights path was found."
            ),
        }

    best = min(candidates, key=lambda c: c["score"])
    return {
        "study_name": study_name,
        "sqlite_path": str(sqlite_path),
        "total_trials": len(study.trials),
        "trial_state_counts": state_counts,
        "leader_trial_number": best["trial_number"],
        "leader_state": best["state"],
        "leader_score": best["score"],
        "weights_path": best["weights_path"],
        "weights_visible_on_disk": best["weights_visible_on_disk"],
        "output_dir": best["output_dir"],
        "objective_mode": best["objective_mode"],
        "salvage_origin_trial_number": best["salvage_origin_trial_number"],
        "hyperparameters": best["hyperparameters"],
    }


@app.function(
    volumes={MODAL_ARTIFACTS_DIR: artifacts_vol},
    timeout=60 * 10,
)
def _read_leader_weights_bytes(
    study_name: str = "sam3-final-optuna-asha-lightweight",
    include_running_trials: bool = False,
    include_pruned_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
) -> dict[str, Any]:
    """Resolve the current leader and return its weights file as bytes."""
    leader = describe_leader_from_study.local(
        study_name=study_name,
        include_running_trials=include_running_trials,
        include_pruned_trials=include_pruned_trials,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
        num_rung_stages=NUM_RUNG_STAGES,
        require_weights_on_disk=True,
    )
    weights_path = leader.get("weights_path")
    if not isinstance(weights_path, str) or not Path(weights_path).exists():
        raise FileNotFoundError(
            f"No leader weights file found for study '{study_name}'. "
            f"Got leader info: {leader}"
        )
    data = Path(weights_path).read_bytes()
    return {
        "leader_trial_number": leader["leader_trial_number"],
        "leader_score": leader["leader_score"],
        "remote_weights_path": weights_path,
        "weights_bytes": data,
        "hyperparameters": leader["hyperparameters"],
    }


@app.local_entrypoint()
def show_current_leader(
    study_name: str = "sam3-final-optuna-asha-lightweight",
    include_running_trials: bool = False,
    include_pruned_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
    require_weights_on_disk: bool = True,
):
    """Print the current best trial in this study as JSON.

    Self-contained -- reads the Optuna sqlite DB on ``artifacts-vol``
    directly and does not require ``tablebank-eval`` to be deployed.

    Examples:
        modal run final_optuna_asha_sweep_lightweight.py::show_current_leader
        modal run final_optuna_asha_sweep_lightweight.py::show_current_leader \\
            --study-name sam3-final-optuna-asha
    """
    result = describe_leader_from_study.remote(
        study_name=study_name,
        include_running_trials=include_running_trials,
        include_pruned_trials=include_pruned_trials,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
        num_rung_stages=NUM_RUNG_STAGES,
        require_weights_on_disk=require_weights_on_disk,
    )
    print(json.dumps(result, indent=2, default=str))


@app.local_entrypoint()
def download_leader_weights(
    study_name: str = "sam3-final-optuna-asha-lightweight",
    output_path: str = "leader_weights.pt",
    include_running_trials: bool = False,
    include_pruned_trials: bool = True,
    sqlite_lock_timeout_sec: int = 60,
):
    """Download the current leader's LoRA weights file to a local path.

    Self-contained -- does not require ``tablebank-eval`` to be deployed.
    """
    payload = _read_leader_weights_bytes.remote(
        study_name=study_name,
        include_running_trials=include_running_trials,
        include_pruned_trials=include_pruned_trials,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
    )
    local_target = Path(output_path).resolve()
    local_target.parent.mkdir(parents=True, exist_ok=True)
    local_target.write_bytes(payload["weights_bytes"])
    summary = {
        "study_name": study_name,
        "leader_trial_number": payload["leader_trial_number"],
        "leader_score": payload["leader_score"],
        "remote_weights_path": payload["remote_weights_path"],
        "local_weights_path": str(local_target),
        "bytes_written": len(payload["weights_bytes"]),
        "hyperparameters": payload["hyperparameters"],
    }
    print(json.dumps(summary, indent=2, default=str))


@app.local_entrypoint()
def main(
    n_trials: int = 2,
    timeout_hours: float = 2.0,
    study_name: str = "sam3-final-optuna-asha-lightweight",
    parallel_workers: int = 1,
    sqlite_lock_timeout_sec: int = 60,
    test_model_extraction: bool = True,
    pruner_type: str = "none",
    reset_study: bool = False,
    include_non_complete_leader_trials: bool = True,
    auto_deploy_tablebank_eval: bool = True,
    salvage_orphaned_trials: bool = True,
):
    config_path = (
        Path(__file__).resolve().parent / "sam3_table" / "testSamples" / "full_lora_config.yaml"
    )
    base_config = SAM3LoRAConfig.from_yaml(config_path)

    # Ensure the `tablebank-eval` app is deployed BEFORE launching the remote
    # sweep so that `modal.Function.from_name("tablebank-eval", ...)` can
    # resolve inside the sweep container. Without this, leader extraction
    # silently falls through to a local reimplementation that reads a stale
    # sqlite snapshot and returns the wrong trial.
    if test_model_extraction and auto_deploy_tablebank_eval:
        from eval_tablebank import ensure_deployed as _ensure_tablebank_eval_deployed

        _ensure_tablebank_eval_deployed()

    result = run_lightweight_optuna_test.remote(
        n_trials=n_trials,
        timeout_hours=timeout_hours,
        study_name=study_name,
        parallel_workers=parallel_workers,
        sqlite_lock_timeout_sec=sqlite_lock_timeout_sec,
        test_model_extraction=test_model_extraction,
        pruner_type=pruner_type,
        reset_study=reset_study,
        include_non_complete_leader_trials=include_non_complete_leader_trials,
        salvage_orphaned_trials=salvage_orphaned_trials,
        base_config_dict=base_config.model_dump(mode="json"),
    )
    print(json.dumps(result, indent=2))
