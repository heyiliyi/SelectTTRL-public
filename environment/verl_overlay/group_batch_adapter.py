"""DataProto adapter for a future post-rollout VERL hook.

The adapter is intentionally callable only by a batch-level hook. It is not
imported by stock RayPPO automatically and therefore cannot silently change an
existing run.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Sequence

from online_group_hook import OnlineGroupBatch, plan_online_group_batch


@dataclass(frozen=True)
class AppliedGroupBatch:
    batch: object
    plan: object
    kept_rewards: tuple[float, ...]


def filter_before_old_ref(
    batch,
    *,
    uids: Sequence[str],
    response_texts: Sequence[str],
    vote_rollouts: int,
    train_rollouts: int,
    step: int,
    extractor=None,
    equivalence=None,
) -> AppliedGroupBatch:
    """Filter a completed rollout batch and attach scalar outcome rewards.

    The caller must invoke this before old/ref log-prob computation. The input
    order must match `batch` exactly; a mismatch raises instead of silently
    assigning a reward to the wrong response.
    """

    if len(batch) != len(uids) or len(uids) != len(response_texts):
        raise ValueError("batch, uids, and response_texts must have equal length")
    plan = plan_online_group_batch(
        OnlineGroupBatch(uids, response_texts, vote_rollouts, train_rollouts, step),
        extractor=extractor or _default_extractor(),
        equivalence=equivalence,
    )
    if not plan.selected_positions:
        raise RuntimeError("all rollout groups were unusable; caller must skip this update step")
    selected = batch.select_idxs(list(plan.selected_positions))
    import torch

    responses = selected.batch["responses"]
    scores = torch.zeros_like(responses, dtype=torch.float32)
    response_mask = selected.batch.get("response_mask")
    if response_mask is None:
        attention_mask = selected.batch.get("attention_mask")
        if attention_mask is None:
            raise ValueError("response_mask or attention_mask required for terminal rewards")
        response_mask = attention_mask[:, -responses.shape[1]:]
    if response_mask.shape != responses.shape:
        raise ValueError("response mask shape must match responses")
    # Place the outcome reward on the last valid token, never on padding.
    terminal_positions = []
    for row_mask in response_mask.tolist():
        valid = [i for i, value in enumerate(row_mask) if value]
        if not valid:
            raise ValueError("cannot attach an outcome reward to an empty response")
        terminal_positions.append(valid[-1])
    rewards = []
    grouped_positions = defaultdict(list)
    for original_position, uid in enumerate(uids):
        grouped_positions[str(uid)].append(original_position)
    reward_by_position = {}
    for group in plan.groups:
        for local, position in enumerate(group.selected_indices):
            original_position = grouped_positions[group.uid][position]
            reward_by_position[original_position] = group.rewards[position]
    for original_position in plan.selected_positions:
        value = reward_by_position.get(original_position, 0.0)
        scores[len(rewards), terminal_positions[len(rewards)]] = float(value)
        rewards.append(float(value))
    selected.batch["rm_scores"] = scores
    return AppliedGroupBatch(selected, plan, tuple(rewards))


def _default_extractor():
    from pathlib import Path
    import sys

    scripts = Path(__file__).resolve().parent
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from group_reward import extract_boxed_or_answer

    return extract_boxed_or_answer
