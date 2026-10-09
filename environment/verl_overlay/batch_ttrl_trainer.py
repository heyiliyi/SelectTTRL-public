"""Batch-aware TTRL reward integration for the installed RayPPOTrainer.

VERL streams a per-item placeholder reward during rollout. Immediately after
the complete response batch is joined, RayPPOTrainer calls extract_reward(batch)
before old/ref log-prob and advantage calculation. This subclass replaces that
process-local function with a group-aware TTRL majority reward.

For the matched TTRL control, ``rollout.n`` remains the number of responses
that enter the optimizer (16).  A process-local rollout wrapper expands those
requests to 32, freezes the pseudo label from all 32 responses, writes every
raw response to disk, and returns the official first 16 responses to stock
VERL.  Consequently old/ref log-prob, GRPO advantage, and actor update all see
16 responses, while pseudo-label construction sees 32.  Neither VERL tree is
modified.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from group_reward import plan_group  # noqa: E402
from evaluation_metrics import accuracy_metrics  # noqa: E402
from gpqa_choice import (  # noqa: E402
    GPQA_CHOICE_SHA256,
    GPQA_CHOICE_SOURCE,
    GPQA_NORMALIZER_NAME,
    GPQA_REWARD_GRADER_NAME,
    extract_gpqa_option,
    score_response_against_pseudo as score_gpqa_response_against_pseudo,
)
from mmlu_choice import (  # noqa: E402
    MMLU_CHOICE_SHA256,
    MMLU_CHOICE_SOURCE,
    MMLU_NORMALIZER_NAME,
    MMLU_REWARD_GRADER_NAME,
    extract_mmlu_option,
    score_response_against_pseudo as score_mmlu_response_against_pseudo,
)
from ttrl_official_normalizer import (  # noqa: E402
    TTRL_MATH_INIT,
    TTRL_MATH_SHA256,
    TTRL_NORMALIZER_NAME,
    TTRL_REWARD_GRADER_NAME,
    extract_and_simplify,
    score_response_against_pseudo,
)


def _population_variance(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    value_mean = sum(values) / len(values)
    return sum((value - value_mean) ** 2 for value in values) / len(values)


def _normalized_answer_entropy(values: list[str]) -> float:
    """Entropy of the rollout answer consensus, normalized to approximately [0, 1]."""

    valid = [str(value) for value in values if str(value) not in {"", "None", "null"}]
    if len(valid) <= 1:
        return 0.0
    counts = Counter(valid)
    n = float(len(valid))
    entropy = -sum((count / n) * math.log(count / n) for count in counts.values())
    return float(entropy / math.log(max(2, len(valid))))


class BatchTTRLRayPPOTrainer(RayPPOTrainer):
    """Use complete rollout groups to construct binary majority rewards."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use the actual dataloader after dataset filtering, not just the
        # launcher's optional pre-filter estimate. Do not silently under-train.
        available_steps = len(self.train_dataloader) * int(self.config.trainer.total_epochs)
        if available_steps < self.total_training_steps:
            raise ValueError(
                f"filtered training data permits {available_steps} steps, requested "
                f"{self.total_training_steps}; increase trainer.total_epochs"
            )
        self.ttrl_vote_n = int(os.environ.get("TTRL_VOTE_N", self.config.actor_rollout_ref.rollout.n))
        self.ttrl_train_n = int(os.environ.get("TTRL_TRAIN_N", self.ttrl_vote_n))
        self.ttrl_task = os.environ.get("TTRL_TASK", "math").strip().lower()
        configured_train_n = int(self.config.actor_rollout_ref.rollout.n)
        if self.ttrl_train_n != configured_train_n:
            raise ValueError(
                "TTRL_TRAIN_N must equal actor_rollout_ref.rollout.n so old/ref/adv/update "
                f"all use the training response count ({self.ttrl_train_n} != {configured_train_n})"
            )
        if not 0 < self.ttrl_train_n <= self.ttrl_vote_n:
            raise ValueError("require 0 < TTRL_TRAIN_N <= TTRL_VOTE_N")
        if self.ttrl_vote_n % self.ttrl_train_n != 0:
            raise ValueError("TTRL_VOTE_N must be an integer multiple of TTRL_TRAIN_N")
        if self.ttrl_task == "math":
            self.ttrl_extractor = extract_and_simplify
            self.ttrl_scorer = score_response_against_pseudo
            self.ttrl_normalizer_name = TTRL_NORMALIZER_NAME
            self.ttrl_normalizer_source = TTRL_MATH_INIT
            self.ttrl_normalizer_sha256 = TTRL_MATH_SHA256
            self.ttrl_reward_grader_name = TTRL_REWARD_GRADER_NAME
        elif self.ttrl_task == "gpqa":
            self.ttrl_extractor = extract_gpqa_option
            self.ttrl_scorer = score_gpqa_response_against_pseudo
            self.ttrl_normalizer_name = GPQA_NORMALIZER_NAME
            self.ttrl_normalizer_source = GPQA_CHOICE_SOURCE
            self.ttrl_normalizer_sha256 = GPQA_CHOICE_SHA256
            self.ttrl_reward_grader_name = GPQA_REWARD_GRADER_NAME
        elif self.ttrl_task == "mmlu":
            self.ttrl_extractor = extract_mmlu_option
            self.ttrl_scorer = score_mmlu_response_against_pseudo
            self.ttrl_normalizer_name = MMLU_NORMALIZER_NAME
            self.ttrl_normalizer_source = MMLU_CHOICE_SOURCE
            self.ttrl_normalizer_sha256 = MMLU_CHOICE_SHA256
            self.ttrl_reward_grader_name = MMLU_REWARD_GRADER_NAME
        else:
            raise ValueError(f"unsupported TTRL_TASK: {self.ttrl_task!r}")
        self.ttrl_audit_path = Path(
            os.environ.get(
                "TTRL_AUDIT_PATH",
                "runs/ttrl_group_audit.jsonl",
            )
        )
        self.ttrl_audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttrl_full_rollout_dir = Path(
            os.environ.get(
                "TTRL_FULL_ROLLOUT_DIR",
                "runs/full_vote_rollouts",
            )
        )
        self.ttrl_full_rollout_dir.mkdir(parents=True, exist_ok=True)

        # fit() resolves extract_reward in the ray_trainer module at runtime.
        # The TaskRunner is its own Ray process, so this patch is process-local.
        import verl.trainer.ppo.ray_trainer as ray_trainer_module

        ray_trainer_module.extract_reward = self._extract_batch_ttrl_reward

    def init_workers(self):
        """Install vote expansion only after stock VERL creates its manager."""

        result = super().init_workers()
        if self.ttrl_vote_n > self.ttrl_train_n:
            original_generate = self.async_rollout_manager.generate_sequences

            def generate_vote_then_select(prompts: DataProto) -> DataProto:
                return self._generate_vote_then_select(prompts, original_generate)

            self.async_rollout_manager.generate_sequences = generate_vote_then_select
        return result

    @staticmethod
    def _group_positions(batch: DataProto) -> dict[str, list[int]]:
        if "uid" not in batch.non_tensor_batch:
            raise KeyError("batch lacks uid; cannot form GRPO/TTRL groups")
        grouped: dict[str, list[int]] = defaultdict(list)
        for position, raw_uid in enumerate(batch.non_tensor_batch["uid"].tolist()):
            grouped[str(raw_uid)].append(position)
        return grouped

    @staticmethod
    def _single_group_value(values, positions: list[int], name: str):
        if values is None:
            return None
        distinct = {
            str(values[position])
            for position in positions
            if values[position] is not None
        }
        if len(distinct) > 1:
            raise ValueError(f"one TTRL uid mixes {name}: {sorted(distinct)}")
        return next(iter(distinct), None)

    def _write_full_vote_rollouts(
        self,
        batch: DataProto,
        texts: list[str],
        valid_lengths: list[int],
        plans: dict[str, object],
        rewards: dict[str, tuple[float, ...]],
        grouped: dict[str, list[int]],
    ) -> None:
        """Persist all vote responses before any first-k downsampling."""

        output_path = self.ttrl_full_rollout_dir / f"{int(self.global_steps)}.jsonl"
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite full vote rollout audit: {output_path}")
        source_question_ids = batch.non_tensor_batch.get("source_question_id")
        source_rows = batch.non_tensor_batch.get("source_row")
        route_arms = batch.non_tensor_batch.get("route_arm")
        request_ids = batch.non_tensor_batch.get("request_id")
        prompts = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        with output_path.open("x", encoding="utf-8") as handle:
            for uid, positions in grouped.items():
                plan = plans[uid]
                group_rewards = rewards[uid]
                records = []
                for local_position, batch_position in enumerate(positions):
                    records.append(
                        {
                            "rollout_index": local_position,
                            "selected_for_update": local_position < self.ttrl_train_n,
                            "response_tokens": valid_lengths[batch_position],
                            "normalized_answer": plan.answers[local_position],
                            "reward": group_rewards[local_position],
                            "request_id": (
                                str(request_ids[batch_position]) if request_ids is not None else None
                            ),
                            "response": texts[batch_position],
                        }
                    )
                row = {
                    "schema": "matched_ttrl_full_vote_rollouts_v1",
                    "global_step": int(self.global_steps),
                    "task": self.ttrl_task,
                    "uid": uid,
                    "source_question_id": self._single_group_value(
                        source_question_ids, positions, "source_question_id"
                    ),
                    "source_row": self._single_group_value(source_rows, positions, "source_row"),
                    "route_arm": self._single_group_value(route_arms, positions, "route_arm"),
                    "vote_n": self.ttrl_vote_n,
                    "train_n": self.ttrl_train_n,
                    "selection": "first_k_per_prompt_official_ttrl_semantics",
                    "pseudo_label": plan.pseudo_label,
                    "majority_count": plan.majority_count,
                    "valid_count": plan.valid_count,
                    "prompt": prompts[positions[0]],
                    "responses": records,
                }
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def _generate_vote_then_select(self, prompts: DataProto, generate) -> DataProto:
        """Generate vote_n responses, freeze plurality, return first train_n."""

        if prompts.meta_info.get("validate", False):
            return generate(prompts)

        grouped_train = self._group_positions(prompts)
        for uid, positions in grouped_train.items():
            if len(positions) != self.ttrl_train_n:
                raise ValueError(
                    f"uid {uid!r} has {len(positions)} optimizer requests; expected {self.ttrl_train_n}"
                )

        multiplier = self.ttrl_vote_n // self.ttrl_train_n
        vote_prompts = prompts.repeat(repeat_times=multiplier, interleave=True)
        full_output = generate(vote_prompts)
        if len(full_output) != len(vote_prompts):
            raise ValueError(
                f"rollout manager returned {len(full_output)} responses for {len(vote_prompts)} vote requests"
            )
        # With VERL's async reward loop enabled, generation output intentionally
        # omits most input metadata.  Output order is stable, so restore only the
        # frozen grouping/identity fields needed before the stock trainer unions
        # the selected output with its repeated prompt batch.
        for key in ("uid", "source_question_id", "source_row", "route_arm"):
            source_values = vote_prompts.non_tensor_batch.get(key)
            output_values = full_output.non_tensor_batch.get(key)
            if source_values is None:
                continue
            if output_values is None:
                full_output.non_tensor_batch[key] = source_values.copy()
            elif [str(value) for value in output_values] != [str(value) for value in source_values]:
                raise ValueError(f"rollout manager changed frozen TTRL metadata field {key!r}")
        reward_started = time.perf_counter()
        grouped = self._group_positions(full_output)
        texts, valid_lengths = self._decode_responses(full_output)
        plans = {}
        rewards = {}
        selected_positions: list[int] = []
        extra = {
            "ttrl_rollout_index": np.zeros(len(full_output), dtype=np.int64),
            "ttrl_selected_for_update": np.zeros(len(full_output), dtype=np.int64),
            "ttrl_answer": np.empty(len(full_output), dtype=object),
            "ttrl_pseudo_label": np.empty(len(full_output), dtype=object),
            "ttrl_majority_count": np.zeros(len(full_output), dtype=np.int64),
            "ttrl_valid_count": np.zeros(len(full_output), dtype=np.int64),
            "ttrl_majority_ratio": np.zeros(len(full_output), dtype=np.float32),
            "ttrl_full_reward_variance": np.zeros(len(full_output), dtype=np.float32),
            "ttrl_update_reward_variance": np.zeros(len(full_output), dtype=np.float32),
            "ttrl_group_usable": np.zeros(len(full_output), dtype=np.int64),
            "ttrl_consensus_entropy": np.zeros(len(full_output), dtype=np.float32),
            "ttrl_update_mean_response_tokens": np.zeros(len(full_output), dtype=np.float32),
            "ttrl_update_length_clip_ratio": np.zeros(len(full_output), dtype=np.float32),
        }
        audit_groups = []
        source_question_ids = full_output.non_tensor_batch.get("source_question_id")
        source_rows = full_output.non_tensor_batch.get("source_row")
        route_arms = full_output.non_tensor_batch.get("route_arm")
        for group_number, (uid, positions) in enumerate(grouped.items()):
            if len(positions) != self.ttrl_vote_n:
                raise ValueError(
                    f"uid {uid!r} has {len(positions)} vote responses; expected {self.ttrl_vote_n}"
                )
            plan = plan_group(
                uid,
                [texts[position] for position in positions],
                vote_n=self.ttrl_vote_n,
                train_n=self.ttrl_train_n,
                seed=8253216 + int(self.global_steps) * 10000 + group_number,
                extractor=self.ttrl_extractor,
                equivalence=None,
                min_valid=1,
                skip_zero_variance=False,
                cosine_scores=None,
            )
            group_rewards = tuple(
                self.ttrl_scorer(texts[position], plan.pseudo_label) for position in positions
            )
            full_variance = _population_variance(group_rewards)
            update_rewards = group_rewards[: self.ttrl_train_n]
            update_variance = _population_variance(update_rewards)
            usable = int(update_variance > 0.0)
            majority_ratio = plan.majority_count / self.ttrl_vote_n
            consensus_entropy = _normalized_answer_entropy(plan.answers)
            update_lengths = [valid_lengths[position] for position in positions[: self.ttrl_train_n]]
            update_mean_length = sum(update_lengths) / max(1, len(update_lengths))
            update_clip_ratio = sum(length >= 4096 for length in update_lengths) / max(1, len(update_lengths))
            source_question_id = self._single_group_value(
                source_question_ids, positions, "source_question_id"
            )
            source_row = self._single_group_value(source_rows, positions, "source_row")
            route_arm = self._single_group_value(route_arms, positions, "route_arm")
            plans[uid] = plan
            rewards[uid] = group_rewards
            for local_position, batch_position in enumerate(positions):
                selected = local_position < self.ttrl_train_n
                extra["ttrl_rollout_index"][batch_position] = local_position
                extra["ttrl_selected_for_update"][batch_position] = int(selected)
                extra["ttrl_answer"][batch_position] = plan.answers[local_position]
                extra["ttrl_pseudo_label"][batch_position] = plan.pseudo_label
                extra["ttrl_majority_count"][batch_position] = plan.majority_count
                extra["ttrl_valid_count"][batch_position] = plan.valid_count
                extra["ttrl_majority_ratio"][batch_position] = majority_ratio
                extra["ttrl_full_reward_variance"][batch_position] = full_variance
                extra["ttrl_update_reward_variance"][batch_position] = update_variance
                extra["ttrl_group_usable"][batch_position] = usable
                extra["ttrl_consensus_entropy"][batch_position] = consensus_entropy
                extra["ttrl_update_mean_response_tokens"][batch_position] = update_mean_length
                extra["ttrl_update_length_clip_ratio"][batch_position] = update_clip_ratio
                if selected:
                    selected_positions.append(batch_position)
            audit_groups.append(
                {
                    "uid": uid,
                    "source_question_id": source_question_id,
                    "source_row": source_row,
                    "route_arm": route_arm,
                    "status": "usable" if usable else "zero_update_reward_variance",
                    "pseudo_label": plan.pseudo_label,
                    "majority_count": plan.majority_count,
                    "valid_count": plan.valid_count,
                    "majority_ratio": majority_ratio,
                    "reward_positive_count": sum(value > 0.0 for value in group_rewards),
                    "update_reward_positive_count": sum(value > 0.0 for value in update_rewards),
                    "reward_variance": update_variance,
                    "full_reward_variance": full_variance,
                    "reward_mean": sum(update_rewards) / len(update_rewards),
                    "full_reward_mean": sum(group_rewards) / len(group_rewards),
                    "response_tokens": sum(valid_lengths[position] for position in positions),
                    "update_response_tokens": sum(
                        valid_lengths[position] for position in positions[: self.ttrl_train_n]
                    ),
                    "selected_rollout_indices": list(range(self.ttrl_train_n)),
                }
            )

        for key, value in extra.items():
            full_output.non_tensor_batch[key] = value
        self._write_full_vote_rollouts(full_output, texts, valid_lengths, plans, rewards, grouped)
        audit_record = {
            "schema": "batch_ttrl_reward_audit_v5_matched_first_k",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "global_step": int(self.global_steps),
            "task": self.ttrl_task,
            "vote_n": self.ttrl_vote_n,
            "train_n": self.ttrl_train_n,
            "selection": "first_k_per_prompt_official_ttrl_semantics",
            "temperature": float(self.config.actor_rollout_ref.rollout.temperature),
            "answer_normalizer": self.ttrl_normalizer_name,
            "answer_normalizer_source": str(self.ttrl_normalizer_source),
            "answer_normalizer_sha256": self.ttrl_normalizer_sha256,
            "reward_grader": self.ttrl_reward_grader_name,
            "zero_variance_policy": os.environ.get(
                "TTRL_ZERO_VARIANCE_POLICY", "keep_zero_advantage_ttrl_official_reward"
            ),
            "groups": len(audit_groups),
            "usable_groups": sum(row["status"] == "usable" for row in audit_groups),
            "zero_variance_groups": sum(row["status"] != "usable" for row in audit_groups),
            "rollout_sequences": len(full_output),
            "update_sequences": len(selected_positions),
            "rollout_response_tokens": sum(valid_lengths),
            "update_response_tokens": sum(valid_lengths[position] for position in selected_positions),
            "reward_wall_seconds": time.perf_counter() - reward_started,
            "group_records": audit_groups,
        }
        with self.ttrl_audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(audit_record, ensure_ascii=False, allow_nan=False) + "\n")

        selected = full_output.select_idxs(selected_positions)
        if len(selected) != len(grouped) * self.ttrl_train_n:
            raise AssertionError("matched TTRL first-k selection produced the wrong update batch size")
        for key in ("uid", "source_question_id", "source_row", "route_arm"):
            expected = prompts.non_tensor_batch.get(key)
            actual = selected.non_tensor_batch.get(key)
            if expected is not None and (
                actual is None
                or [str(value) for value in actual] != [str(value) for value in expected]
            ):
                raise ValueError(
                    f"first-k output order no longer aligns with stock VERL prompt batch for {key!r}"
                )
        return selected

    def _decode_responses(self, batch: DataProto) -> tuple[list[str], list[int]]:
        responses = batch.batch["responses"]
        prompt_length = batch.batch["prompts"].size(1)
        valid_lengths = batch.batch["attention_mask"][:, prompt_length:].sum(dim=1).tolist()
        texts = []
        lengths = []
        for row, raw_length in zip(responses, valid_lengths, strict=True):
            length = int(raw_length)
            if length <= 0:
                raise RuntimeError("rollout produced an empty response")
            lengths.append(length)
            texts.append(self.tokenizer.decode(row[:length], skip_special_tokens=True))
        return texts, lengths

    def _extract_batch_ttrl_reward(self, batch: DataProto):
        if batch.meta_info.get("validate", False):
            return self._extract_validation_reward(batch)
        started = time.perf_counter()
        if self.ttrl_vote_n > self.ttrl_train_n:
            return self._extract_precomputed_first_k_reward(batch)
        if "uid" not in batch.non_tensor_batch:
            raise KeyError("batch lacks uid; cannot form GRPO/TTRL groups")
        uids = [str(value) for value in batch.non_tensor_batch["uid"].tolist()]
        texts, valid_lengths = self._decode_responses(batch)
        source_question_ids = batch.non_tensor_batch.get("source_question_id")
        source_rows = batch.non_tensor_batch.get("source_row")
        route_arms = batch.non_tensor_batch.get("route_arm")
        grouped: dict[str, list[int]] = defaultdict(list)
        for position, uid in enumerate(uids):
            grouped[uid].append(position)

        rm_scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        extra = {
            "ttrl_uid": np.empty(len(batch), dtype=object),
            "ttrl_answer": np.empty(len(batch), dtype=object),
            "ttrl_pseudo_label": np.empty(len(batch), dtype=object),
            "ttrl_majority_count": np.zeros(len(batch), dtype=np.int64),
            "ttrl_valid_count": np.zeros(len(batch), dtype=np.int64),
            "ttrl_majority_ratio": np.zeros(len(batch), dtype=np.float32),
            "ttrl_reward_variance": np.zeros(len(batch), dtype=np.float32),
            "ttrl_group_usable": np.zeros(len(batch), dtype=np.int64),
            "ttrl_source_question_id": np.empty(len(batch), dtype=object),
            "ttrl_source_row": np.empty(len(batch), dtype=object),
            "ttrl_route_arm": np.empty(len(batch), dtype=object),
            "ttrl_consensus_entropy": np.zeros(len(batch), dtype=np.float32),
            "ttrl_update_mean_response_tokens": np.zeros(len(batch), dtype=np.float32),
            "ttrl_update_length_clip_ratio": np.zeros(len(batch), dtype=np.float32),
        }
        audit_groups = []
        for group_number, (uid, positions) in enumerate(sorted(grouped.items())):
            if len(positions) != self.ttrl_vote_n:
                raise ValueError(
                    f"uid {uid!r} has {len(positions)} responses; expected {self.ttrl_vote_n}"
                )
            plan = plan_group(
                uid,
                [texts[position] for position in positions],
                vote_n=self.ttrl_vote_n,
                train_n=self.ttrl_train_n,
                seed=8251625 + int(self.global_steps) * 10000 + group_number,
                extractor=self.ttrl_extractor,
                equivalence=None,
                min_valid=1,
                skip_zero_variance=False,
                cosine_scores=None,
            )
            group_texts = tuple(texts[position] for position in positions)
            official_rewards = tuple(
                self.ttrl_scorer(response, plan.pseudo_label)
                for response in group_texts
            )
            reward_variance = _population_variance(official_rewards)
            reward_positive_count = sum(reward > 0.0 for reward in official_rewards)
            usable = int(reward_variance > 0.0)
            ttrl_majority_ratio = plan.majority_count / self.ttrl_vote_n
            consensus_entropy = _normalized_answer_entropy(plan.answers)
            group_mean_length = sum(valid_lengths[position] for position in positions) / max(1, len(positions))
            group_clip_ratio = sum(valid_lengths[position] >= 4096 for position in positions) / max(1, len(positions))
            group_source_ids = {
                str(source_question_ids[position])
                for position in positions
                if source_question_ids is not None and source_question_ids[position] is not None
            }
            if len(group_source_ids) > 1:
                raise ValueError(f"uid {uid!r} mixes source question IDs: {sorted(group_source_ids)}")
            source_question_id = next(iter(group_source_ids), None)
            group_source_rows = {
                str(source_rows[position])
                for position in positions
                if source_rows is not None and source_rows[position] is not None
            }
            if len(group_source_rows) > 1:
                raise ValueError(f"uid {uid!r} mixes source rows: {sorted(group_source_rows)}")
            source_row = next(iter(group_source_rows), None)
            group_route_arms = {
                str(route_arms[position])
                for position in positions
                if route_arms is not None and route_arms[position] is not None
            }
            if len(group_route_arms) > 1:
                raise ValueError(f"uid {uid!r} mixes route arms: {sorted(group_route_arms)}")
            route_arm = next(iter(group_route_arms), None)
            for local_position, batch_position in enumerate(positions):
                reward = official_rewards[local_position]
                rm_scores[batch_position, valid_lengths[batch_position] - 1] = reward
                extra["ttrl_uid"][batch_position] = uid
                extra["ttrl_answer"][batch_position] = plan.answers[local_position]
                extra["ttrl_pseudo_label"][batch_position] = plan.pseudo_label
                extra["ttrl_majority_count"][batch_position] = plan.majority_count
                extra["ttrl_valid_count"][batch_position] = plan.valid_count
                extra["ttrl_majority_ratio"][batch_position] = ttrl_majority_ratio
                extra["ttrl_reward_variance"][batch_position] = reward_variance
                extra["ttrl_group_usable"][batch_position] = usable
                extra["ttrl_source_question_id"][batch_position] = source_question_id
                extra["ttrl_source_row"][batch_position] = source_row
                extra["ttrl_route_arm"][batch_position] = route_arm
                extra["ttrl_consensus_entropy"][batch_position] = consensus_entropy
                extra["ttrl_update_mean_response_tokens"][batch_position] = group_mean_length
                extra["ttrl_update_length_clip_ratio"][batch_position] = group_clip_ratio
            audit_groups.append(
                {
                    "uid": uid,
                    "source_question_id": source_question_id,
                    "source_row": source_row,
                    "route_arm": route_arm,
                    "status": "usable" if usable else "zero_reward_variance",
                    "pseudo_label": plan.pseudo_label,
                    "majority_count": plan.majority_count,
                    "valid_count": plan.valid_count,
                    "majority_ratio": ttrl_majority_ratio,
                    "reward_positive_count": reward_positive_count,
                    "reward_variance": reward_variance,
                    "reward_mean": sum(official_rewards) / len(official_rewards),
                    "response_tokens": sum(valid_lengths[position] for position in positions),
                }
            )

        record = {
            "schema": "batch_ttrl_reward_audit_v4",
            "global_step": int(self.global_steps),
            "task": self.ttrl_task,
            "vote_n": self.ttrl_vote_n,
            "train_n": self.ttrl_train_n,
            "temperature": float(self.config.actor_rollout_ref.rollout.temperature),
            "answer_normalizer": self.ttrl_normalizer_name,
            "answer_normalizer_source": str(self.ttrl_normalizer_source),
            "answer_normalizer_sha256": self.ttrl_normalizer_sha256,
            "reward_grader": self.ttrl_reward_grader_name,
            "zero_variance_policy": os.environ.get(
                "TTRL_ZERO_VARIANCE_POLICY", "keep_zero_advantage_for_plumbing"
            ),
            "groups": len(audit_groups),
            "usable_groups": sum(row["status"] == "usable" for row in audit_groups),
            "zero_variance_groups": sum(row["status"] != "usable" for row in audit_groups),
            "rollout_sequences": len(batch),
            "rollout_response_tokens": sum(valid_lengths),
            "reward_wall_seconds": time.perf_counter() - started,
            "group_records": audit_groups,
        }
        with self.ttrl_audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

        return (
            rm_scores,
            {key: value.tolist() for key, value in extra.items()},
        )

    def _extract_validation_reward(self, batch: DataProto):
        """Grade against gold answers only on the validation path."""
        texts, lengths = self._decode_responses(batch)
        labels = batch.non_tensor_batch.get("reward_model")
        if labels is None or len(labels) != len(texts):
            raise ValueError("validation requires reward_model.ground_truth for every response")
        scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        accuracies, predictions = [], []
        for i, (text, length, label) in enumerate(zip(texts, lengths, labels, strict=True)):
            gold = label.get("ground_truth") if isinstance(label, dict) else None
            if gold is None or str(gold).strip() == "":
                raise ValueError("validation requires a nonempty ground_truth")
            # Reuse the task's answer grader, but supply gold, never a vote label.
            value = float(self.ttrl_scorer(text, str(gold)))
            scores[i, length - 1] = value
            accuracies.append(value)
            predictions.append(self.ttrl_extractor(text))
        return scores, {"acc": accuracies, "pred": predictions}

    def _validate(self, merged: bool = False):
        """Evaluate greedy and sampled accuracy without training vote expansion."""
        if merged:
            return super()._validate(merged=True)
        sampling = self.config.actor_rollout_ref.rollout.val_kwargs
        saved = {key: getattr(sampling, key) for key in ("n", "temperature", "do_sample")}
        original_dir = self.config.trainer.get("validation_data_dir", None)
        metrics = {}
        try:
            for mode, n, temperature, do_sample in (
                ("greedy", 1, 0.0, False), ("sampled", 16, 0.6, True)
            ):
                sampling.n, sampling.temperature, sampling.do_sample = n, temperature, do_sample
                if original_dir:
                    self.config.trainer.validation_data_dir = str(Path(original_dir) / mode)
                result = super()._validate(merged=True)
                metrics.update(accuracy_metrics(result, mode=mode))
        finally:
            for key, value in saved.items():
                setattr(sampling, key, value)
            if original_dir:
                self.config.trainer.validation_data_dir = original_dir
        return metrics

    def _extract_precomputed_first_k_reward(self, batch: DataProto):
        """Score the first-k batch against the pseudo label frozen from all votes."""

        required = {
            "ttrl_rollout_index",
            "ttrl_selected_for_update",
            "ttrl_answer",
            "ttrl_pseudo_label",
            "ttrl_majority_count",
            "ttrl_valid_count",
            "ttrl_majority_ratio",
            "ttrl_full_reward_variance",
            "ttrl_update_reward_variance",
            "ttrl_group_usable",
        }
        missing = sorted(required - set(batch.non_tensor_batch))
        if missing:
            raise KeyError(f"matched TTRL batch lacks precomputed vote fields: {missing}")
        grouped = self._group_positions(batch)
        texts, valid_lengths = self._decode_responses(batch)
        rm_scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        for uid, positions in grouped.items():
            if len(positions) != self.ttrl_train_n:
                raise ValueError(
                    f"uid {uid!r} has {len(positions)} update responses; expected {self.ttrl_train_n}"
                )
            rollout_indices = sorted(
                int(batch.non_tensor_batch["ttrl_rollout_index"][position]) for position in positions
            )
            if rollout_indices != list(range(self.ttrl_train_n)):
                raise ValueError(
                    f"uid {uid!r} did not retain official first-k responses: {rollout_indices}"
                )
            for position in positions:
                if int(batch.non_tensor_batch["ttrl_selected_for_update"][position]) != 1:
                    raise ValueError("downsampled actor batch contains an unselected vote response")
                pseudo_label = batch.non_tensor_batch["ttrl_pseudo_label"][position]
                expected_answer = batch.non_tensor_batch["ttrl_answer"][position]
                if self.ttrl_extractor(texts[position]) != expected_answer:
                    raise ValueError("normalized answer changed between vote and reward stages")
                reward = self.ttrl_scorer(texts[position], pseudo_label)
                rm_scores[position, valid_lengths[position] - 1] = reward
        extra_keys = sorted(required | {"uid"})
        extra = {}
        for key in extra_keys:
            source_key = "uid" if key == "uid" else key
            output_key = "ttrl_uid" if key == "uid" else key
            extra[output_key] = batch.non_tensor_batch[source_key].tolist()
        return rm_scores, extra
