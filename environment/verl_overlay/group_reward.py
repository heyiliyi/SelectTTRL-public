#!/usr/bin/env python3
"""Pure, batch-level TTRL planning primitives used by the 8.25 pilot.

The planner never reads gold labels. Gold can be supplied by an offline audit
caller only after a plan is frozen. A group with no reward variance is dropped;
it is never randomly padded back into the actor batch.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

AnswerExtractor = Callable[[str], str | None]
AnswerEquivalence = Callable[[str, str], bool]


def _last_balanced_boxed(text: str) -> str | None:
    starts = [text.rfind("\\boxed{"), text.rfind("\\fbox{")]
    start = max(starts)
    if start < 0:
        return None
    left = text.find("{", start)
    depth = 0
    for index in range(left, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[left + 1 : index].strip()
    return None


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    value = re.sub(r"^\\text\{(.*)\}$", r"\1", str(answer).strip())
    value = re.sub(r"\s+", "", value.strip("$."))
    return value.upper() or None


def extract_boxed_or_answer(text: str) -> str | None:
    boxed = _last_balanced_boxed(text)
    if boxed is not None:
        return normalize_answer(boxed)
    patterns = (
        r"(?is)(?:final\s+answer|answer)\s*[:：]\s*\(?([A-J])\)?\b",
        r"(?is)(?:final\s+answer|answer)\s*[:：]\s*([^\n]+)$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return normalize_answer(matches[-1])
    return None


def math_verify_equivalence(left: str, right: str) -> bool:
    """Use the already-installed math_verify package; never install it here."""

    if normalize_answer(left) == normalize_answer(right):
        return True
    from math_verify import parse, verify

    try:
        parsed_left = parse(r"\boxed{" + str(left) + "}")
        parsed_right = parse(r"\boxed{" + str(right) + "}")
        if parsed_left is None or parsed_right is None:
            return False
        return bool(verify(parsed_left, parsed_right) or verify(parsed_right, parsed_left))
    except Exception:
        return False


@dataclass(frozen=True)
class GroupPlan:
    uid: str
    answers: tuple[str | None, ...]
    pseudo_label: str | None
    rewards: tuple[float, ...]
    selected_indices: tuple[int, ...]
    majority_count: int
    valid_count: int
    majority_ratio: float
    reward_variance: float
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BatchPlan:
    selected_positions: tuple[int, ...]
    dropped_uids: tuple[str, ...]
    groups: tuple[GroupPlan, ...]

    def to_dict(self) -> dict:
        return {
            "selected_positions": list(self.selected_positions),
            "dropped_uids": list(self.dropped_uids),
            "groups": [group.to_dict() for group in self.groups],
        }


def _population_variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def plan_group(
    uid: str,
    outputs: Sequence[str],
    *,
    vote_n: int,
    train_n: int,
    seed: int,
    extractor: AnswerExtractor = extract_boxed_or_answer,
    equivalence: AnswerEquivalence | None = None,
    min_valid: int = 2,
    skip_zero_variance: bool = True,
    cosine_scores: Sequence[float] | None = None,
    cosine_reward_max: float = 0.1,
) -> GroupPlan:
    if len(outputs) != vote_n:
        raise ValueError(f"uid {uid!r} has {len(outputs)} outputs, expected {vote_n}")
    if not 0 < train_n <= vote_n:
        raise ValueError("require 0 < train_n <= vote_n")
    if cosine_scores is not None and len(cosine_scores) != vote_n:
        raise ValueError("cosine_scores must have one value per output")

    answers = tuple(extractor(output) for output in outputs)
    valid = [i for i, answer in enumerate(answers) if answer is not None]
    if len(valid) < min_valid:
        return GroupPlan(uid, answers, None, tuple(0.0 for _ in outputs), (), 0, len(valid), 0.0, 0.0, "insufficient_valid_answers")

    def same(left: str, right: str) -> bool:
        return left == right if equivalence is None else bool(equivalence(left, right))

    clusters: list[list[int]] = []
    for position in valid:
        answer = answers[position]
        assert answer is not None
        for cluster in clusters:
            representative = answers[cluster[0]]
            assert representative is not None
            if same(answer, representative):
                cluster.append(position)
                break
        else:
            clusters.append([position])
    winning = max(clusters, key=lambda cluster: (len(cluster), -cluster[0]))
    winning_set = set(winning)
    pseudo = answers[winning[0]]
    assert pseudo is not None
    rewards = [float(index in winning_set) for index in range(vote_n)]
    if cosine_scores is not None and cosine_reward_max > 0:
        # Optional shaping is bounded and remains disabled by the contract.
        lo, hi = min(cosine_scores), max(cosine_scores)
        denom = hi - lo if hi > lo else 1.0
        for index in range(vote_n):
            rewards[index] += cosine_reward_max * (float(cosine_scores[index]) - lo) / denom
    variance = _population_variance(rewards)
    if skip_zero_variance and variance == 0.0:
        return GroupPlan(uid, answers, pseudo, tuple(rewards), (), len(winning), len(valid), len(winning) / len(valid), variance, "zero_reward_variance")

    chosen = tuple(sorted(random.Random(seed).sample(range(vote_n), train_n))) if train_n < vote_n else tuple(range(vote_n))
    selected_variance = _population_variance([rewards[index] for index in chosen])
    if skip_zero_variance and selected_variance == 0.0:
        return GroupPlan(uid, answers, pseudo, tuple(rewards), (), len(winning), len(valid), len(winning) / len(valid), variance, "downsample_lost_reward_variance")
    return GroupPlan(uid, answers, pseudo, tuple(rewards), chosen, len(winning), len(valid), len(winning) / len(valid), variance, "usable")


def plan_batch(
    uids: Sequence[str],
    outputs: Sequence[str],
    *,
    vote_n: int,
    train_n: int,
    seed: int,
    extractor: AnswerExtractor = extract_boxed_or_answer,
    equivalence: AnswerEquivalence | None = None,
    skip_zero_variance: bool = True,
) -> BatchPlan:
    if len(uids) != len(outputs):
        raise ValueError("uids and outputs must have the same length")
    grouped: dict[str, list[int]] = defaultdict(list)
    for position, uid in enumerate(uids):
        grouped[str(uid)].append(position)
    plans: list[GroupPlan] = []
    selected: list[int] = []
    dropped: list[str] = []
    for group_number, (uid, positions) in enumerate(sorted(grouped.items())):
        plan = plan_group(uid, [outputs[index] for index in positions], vote_n=vote_n, train_n=train_n, seed=seed + group_number, extractor=extractor, equivalence=equivalence, skip_zero_variance=skip_zero_variance)
        plans.append(plan)
        if plan.status != "usable":
            dropped.append(uid)
            continue
        selected.extend(positions[index] for index in plan.selected_indices)
    return BatchPlan(tuple(sorted(selected)), tuple(dropped), tuple(plans))


def validate_synthetic_contract() -> None:
    good = plan_group("good", [r"\\boxed{1}", r"\\boxed{1}", r"\\boxed{2}", r"\\boxed{2}"], vote_n=4, train_n=4, seed=1)
    assert good.status == "usable" and sum(good.rewards) == 2
    same = plan_group("same", [r"\\boxed{1}"] * 4, vote_n=4, train_n=4, seed=1)
    assert same.status == "zero_reward_variance" and not same.selected_indices
    invalid = plan_group("invalid", ["no answer"] * 4, vote_n=4, train_n=4, seed=1)
    assert invalid.status == "insufficient_valid_answers"


if __name__ == "__main__":
    validate_synthetic_contract()
    print("group_reward synthetic contract: OK")
