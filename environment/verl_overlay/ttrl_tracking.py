"""Experiment-local training metrics for the batch TTRL overlay.

The stock VERL trainer already sends optimizer, reward, timing, throughput,
and memory metrics to every configured logger.  The batch-TTRL reward audit is
written just before that logger call, so this module joins the matching audit
record into the same step without changing the stock VERL checkout.

Only gold-blind quantities are exposed here.  Any comparison with official
MATH answers belongs in a separately named post-hoc evaluation run.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from verl.utils.tracking import Tracking as VerlTracking


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def audit_metrics(
    records: Iterable[dict[str, Any]], support_threshold: int = 9
) -> dict[str, int | float]:
    """Aggregate one or more same-step reward-audit records for logging."""

    records = list(records)
    group_records = [group for record in records for group in record.get("group_records", [])]
    groups = sum(int(record.get("groups", len(record.get("group_records", [])))) for record in records)
    usable_groups = sum(int(record.get("usable_groups", 0)) for record in records)
    zero_variance_groups = sum(
        int(record.get("zero_variance_groups", 0)) for record in records
    )
    rollout_sequences = sum(int(record.get("rollout_sequences", 0)) for record in records)
    response_tokens = sum(int(record.get("rollout_response_tokens", 0)) for record in records)
    reward_wall_seconds = sum(float(record.get("reward_wall_seconds", 0.0)) for record in records)

    supports = [int(group.get("majority_count", 0)) for group in group_records]
    valid_counts = [int(group.get("valid_count", 0)) for group in group_records]
    reward_variances = [float(group.get("reward_variance", 0.0)) for group in group_records]
    usable_records = [group for group in group_records if group.get("status") == "usable"]
    gated_usable = [
        group
        for group in usable_records
        if int(group.get("majority_count", 0)) >= support_threshold
    ]
    vote_slots = sum(
        int(record.get("vote_n", 0))
        * int(record.get("groups", len(record.get("group_records", []))))
        for record in records
    )

    # The support-threshold fields are counterfactual observability metrics.
    # They do not alter rewards or decide which groups update the actor.
    return {
        "ttrl/groups": groups,
        "ttrl/usable_groups": usable_groups,
        "ttrl/usable_ratio": _safe_ratio(usable_groups, groups),
        "ttrl/zero_variance_groups": zero_variance_groups,
        "ttrl/zero_variance_ratio": _safe_ratio(zero_variance_groups, groups),
        "ttrl/valid_answer_ratio": _safe_ratio(sum(valid_counts), vote_slots),
        "ttrl/modal_support_mean": mean(supports) if supports else 0.0,
        "ttrl/modal_support_min": min(supports, default=0),
        "ttrl/modal_support_max": max(supports, default=0),
        "ttrl/reward_variance_mean": mean(reward_variances) if reward_variances else 0.0,
        f"ttrl/usable_support_ge{support_threshold}_groups": len(gated_usable),
        f"ttrl/usable_support_ge{support_threshold}_ratio_all": _safe_ratio(
            len(gated_usable), groups
        ),
        f"ttrl/usable_support_ge{support_threshold}_ratio_usable": _safe_ratio(
            len(gated_usable), usable_groups
        ),
        "ttrl/rollout_sequences": rollout_sequences,
        "ttrl/rollout_response_tokens": response_tokens,
        "ttrl/response_tokens_per_group": _safe_ratio(response_tokens, groups),
        "ttrl/response_tokens_per_usable_group": _safe_ratio(response_tokens, usable_groups),
        "ttrl/reward_wall_seconds": reward_wall_seconds,
    }


class TTRLTracking(VerlTracking):
    """VERL Tracking adapter that attaches the current TTRL audit record."""

    def __init__(self, *args, **kwargs):
        resolved_config = kwargs.get("config")
        super().__init__(*args, **kwargs)
        self._ttrl_audit_path = Path(
            os.environ.get(
                "TTRL_AUDIT_PATH",
                "runs/ttrl_group_audit.jsonl",
            )
        )
        self._ttrl_audit_offset = 0
        self._ttrl_audits_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._ttrl_support_threshold = int(os.environ.get("TTRL_SUPPORT_THRESHOLD", "9"))
        self._ttrl_cumulative_groups = 0
        self._ttrl_cumulative_usable_groups = 0
        self._ttrl_cumulative_response_tokens = 0
        self._ttrl_cumulative_step_seconds = 0.0
        visible_devices = [
            value.strip()
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value.strip()
        ]
        configured_gpu_count = 0
        if isinstance(resolved_config, dict):
            trainer_config = resolved_config.get("trainer", {})
            configured_gpu_count = int(trainer_config.get("n_gpus_per_node", 0)) * int(
                trainer_config.get("nnodes", 1)
            )
        explicit_gpu_count = int(os.environ.get("TTRL_GPU_COUNT", "0"))
        self._ttrl_gpu_count = explicit_gpu_count or configured_gpu_count or len(visible_devices)
        trainer_config = resolved_config.get("trainer", {}) if isinstance(resolved_config, dict) else {}
        self._ttrl_total_training_steps = int(trainer_config.get("total_training_steps", 0))
        self._ttrl_wandb_finish_marker = Path(
            os.environ.get(
                "TTRL_WANDB_FINISH_MARKER",
                str(self._ttrl_audit_path.with_name("wandb_finish.json")),
            )
        )
        self._ttrl_wandb_finished = False
        self._ttrl_progress_path = Path(os.environ.get(
            "TTRL_PROGRESS_PATH", str(self._ttrl_audit_path.with_name("training_progress.json"))
        ))

    def _write_progress(self, step: int) -> None:
        self._ttrl_progress_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": "ttrl_training_progress_v1", "last_step": int(step),
                   "requested_steps": self._ttrl_total_training_steps}
        temporary = self._ttrl_progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self._ttrl_progress_path)

    def _read_new_audits(self) -> None:
        if not self._ttrl_audit_path.is_file():
            return
        with self._ttrl_audit_path.open(encoding="utf-8", errors="replace") as handle:
            handle.seek(self._ttrl_audit_offset)
            while True:
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                self._ttrl_audits_by_step[int(record["global_step"])].append(record)
            self._ttrl_audit_offset = handle.tell()

    def log(self, data, step, backend=None):
        self._read_new_audits()
        merged = dict(data)
        records = self._ttrl_audits_by_step.pop(int(step), [])
        if records:
            current = audit_metrics(records, self._ttrl_support_threshold)
            merged.update(current)
            self._ttrl_cumulative_groups += int(current["ttrl/groups"])
            self._ttrl_cumulative_usable_groups += int(current["ttrl/usable_groups"])
            self._ttrl_cumulative_response_tokens += int(current["ttrl/rollout_response_tokens"])
            self._ttrl_cumulative_step_seconds += float(data.get("timing_s/step", 0.0))
            merged.update(
                {
                    "ttrl/cumulative_groups": self._ttrl_cumulative_groups,
                    "ttrl/cumulative_usable_groups": self._ttrl_cumulative_usable_groups,
                    "ttrl/cumulative_rollout_response_tokens": self._ttrl_cumulative_response_tokens,
                    "cost/cumulative_step_seconds": self._ttrl_cumulative_step_seconds,
                    "cost/accounted_gpu_count": self._ttrl_gpu_count,
                    "cost/cumulative_step_gpu_hours": (
                        self._ttrl_cumulative_step_seconds * self._ttrl_gpu_count / 3600
                    ),
                }
            )
        result = super().log(data=merged, step=step, backend=backend)
        # Initial validation must not masquerade as a completed update.
        if "training/global_step" in data:
            self._write_progress(int(data["training/global_step"]))

        # Stock VERL relies on Tracking.__del__ to finish W&B. In the Ray
        # TaskRunner that destructor can run after Ray has already closed the
        # W&B service transport, leaving the online run stuck in "running"
        # with its final metric row missing. Finish synchronously after the
        # final training metric instead, while the transport is still alive.
        if (
            not self._ttrl_wandb_finished
            and "wandb" in self.logger
            and self._ttrl_total_training_steps > 0
            and int(step) >= self._ttrl_total_training_steps
        ):
            self.logger["wandb"].finish(exit_code=0)
            self.logger.pop("wandb", None)
            self._ttrl_wandb_finished = True
            self._ttrl_wandb_finish_marker.parent.mkdir(parents=True, exist_ok=True)
            self._ttrl_wandb_finish_marker.write_text(
                json.dumps(
                    {
                        "schema": "ttrl_wandb_finish_v1",
                        "step": int(step),
                        "finished_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        return result


def install_tracking_overlay() -> None:
    """Patch only the current TaskRunner process's Tracking class."""

    import verl.utils.tracking as tracking_module

    tracking_module.Tracking = TTRLTracking
