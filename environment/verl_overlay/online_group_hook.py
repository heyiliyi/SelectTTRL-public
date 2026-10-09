"""Explicit batch-level interface for the real online TTRL integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class OnlineGroupBatch:
    uids: Sequence[str]
    responses: Sequence[str]
    vote_rollouts: int
    train_rollouts: int
    step: int


def plan_online_group_batch(batch: OnlineGroupBatch, **kwargs: Any):
    """Return a filtered plan; caller must apply it before old/ref log-prob.

    This function is deliberately independent of VERL DataProto. A future
    adapter can call it from a true post-rollout batch hook and then remove the
    dropped rows from the batch. Calling it from a per-item reward callback is
    unsupported because callback order is not a group boundary.
    """

    from pathlib import Path
    import sys

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from group_reward import plan_batch

    return plan_batch(batch.uids, batch.responses, vote_n=batch.vote_rollouts, train_n=batch.train_rollouts, seed=batch.step, **kwargs)
