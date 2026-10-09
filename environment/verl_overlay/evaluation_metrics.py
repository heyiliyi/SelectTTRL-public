"""Exact question-averaged accuracy metrics for held-out scoring."""
from collections import Counter, defaultdict


def accuracy_metrics(result, *, mode):
    """Use the same 16 sampled responses for Mean1 and Maj.@16 (no bootstrap)."""
    if mode not in {"greedy", "sampled"}:
        raise ValueError("mode must be greedy or sampled")
    sources = [str(source) for batch in result["data_sources"] for source in batch]
    uids = result["sample_uids"]
    info = result["reward_extra_infos_dict"]
    grouped = defaultdict(list)
    for source, uid, acc, pred in zip(sources, uids, info["acc"], info["pred"], strict=True):
        grouped[(source, str(uid))].append((float(acc), pred))
    expected = 1 if mode == "greedy" else 16
    per_source = defaultdict(list)
    for (source, uid), rows in grouped.items():
        if len(rows) != expected:
            raise ValueError(f"{source}/{uid}: expected {expected} evaluation responses, got {len(rows)}")
        mean = sum(acc for acc, _ in rows) / expected
        # Invalid answers receive zero accuracy and do not cast a vote.
        # Counter breaks ties by the first observed answer, deterministically.
        votes = Counter(pred for _, pred in rows if pred is not None and pred != "")
        majority = votes.most_common(1)[0][0] if votes else None
        maj_acc = next((acc for acc, pred in rows if pred == majority), 0.0) if votes else 0.0
        per_source[source].append((mean, maj_acc))
    metrics = {}
    for source, values in per_source.items():
        mean = sum(x[0] for x in values) / len(values)
        if mode == "greedy":
            metrics[f"val/{source}/Greedy"] = mean
        else:
            metrics[f"val/{source}/Mean1"] = mean
            metrics[f"val/{source}/Maj@16"] = sum(x[1] for x in values) / len(values)
    return metrics
