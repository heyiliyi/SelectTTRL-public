"""Dependency-light helpers for new cache and summary entry points only.

These helpers do not change the original v1 steering or SAE ablation readouts.
"""
from collections import defaultdict
import math


def nonthinking_prompt(tokenizer, row, input_format="raw"):
    if input_format == "preformatted":
        text = row.get("prompt", row.get("text"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("preformatted input needs a nonempty prompt/text")
        return text
    if input_format == "chat":
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("chat input needs messages")
    elif input_format == "raw":
        text = row.get("prompt", row.get("text"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("raw input needs a nonempty prompt/text")
        messages = [{"role": "user", "content": text}]
    else:
        raise ValueError("unknown input format")
    return tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)


def last_box_end(text):
    marker = "\\boxed{"
    cursor, last = 0, None
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            return last
        pos, depth = start + len(marker), 1
        while pos < len(text) and depth:
            depth += (text[pos] == "{") - (text[pos] == "}")
            pos += 1
        if depth == 0:
            last = pos
        cursor = start + len(marker)


def answer_position(tokenizer, token_ids, prompt_width, mode, index=None, eos_ids=()):
    if not 0 < prompt_width < len(token_ids):
        raise ValueError("a nonempty generated response is required")
    if mode == "index":
        if index is None or not prompt_width <= index < len(token_ids):
            raise ValueError("index must identify a generated response token")
        return index, "explicit_index"
    if mode == "sequence_last":
        return len(token_ids) - 1, "sequence_last"
    if mode not in {"boxed_or_response_last", "response_last"}:
        raise ValueError("unknown readout mode")
    end = len(token_ids) - 1
    while end >= prompt_width and token_ids[end] in eos_ids:
        end -= 1
    if end < prompt_width:
        raise ValueError("response contains only EOS tokens")
    if mode == "boxed_or_response_last":
        response = token_ids[prompt_width:end + 1]
        text = tokenizer.decode(response, skip_special_tokens=False)
        target = last_box_end(text)
        if target is not None:
            for count in range(1, len(response) + 1):
                prefix = tokenizer.decode(response[:count], skip_special_tokens=False)
                if last_box_end(prefix) == target:
                    return prompt_width + count - 1, "last_complete_box"
            raise ValueError("could not map boxed answer to a token")
    return end, "response_last"


def ranking_components(rows, k=5):
    # No missing metadata silently treated as a complete response.
    if any(not isinstance(row.get("truncated"), bool) for row in rows):
        return {"status": "unavailable", "reason": "boolean truncated metadata required for every row"}
    usable = sorted((r for r in rows if not r["truncated"]), key=lambda r: float(r["cos"]))
    if len(usable) < 2 * k:
        return {"status": "unavailable", "reason": f"need at least {2*k} complete responses"}
    low, high = usable[:k], usable[-k:]
    components = ("feature_sum", "bias", "error")
    low_means = {key: sum(float(r[key]) for r in low) / k for key in components}
    high_means = {key: sum(float(r[key]) for r in high) / k for key in components}
    deltas = {key: high_means[key] - low_means[key] for key in components}
    cos_gap = sum(float(r["cos"]) for r in high) / k - sum(float(r["cos"]) for r in low) / k
    if not all(math.isfinite(x) for x in [cos_gap, *deltas.values()]):
        raise ValueError("nonfinite decomposition data")
    abs_total = sum(abs(x) for x in deltas.values())
    return {"status": "computed", "k": k, "complete_rows": len(usable),
        "high_question_ids": [str(r["question_id"]) for r in high],
        "low_question_ids": [str(r["question_id"]) for r in low],
        "low_means": low_means, "high_means": high_means,
        "component_deltas": deltas, "cos_gap": cos_gap,
        "gap_closure_error": abs(sum(deltas.values()) - cos_gap),
        "signed_delta_share": {key: deltas[key] / cos_gap if cos_gap else None for key in components},
        "absolute_delta_share": {key: abs(deltas[key]) / abs_total if abs_total else None for key in components}}


def question_aggregates(rows, scores, labels, id_field="question_id"):
    groups = defaultdict(list)
    for row, score, label in zip(rows, scores, labels, strict=True):
        if id_field not in row or row[id_field] is None or str(row[id_field]) == "":
            raise ValueError(f"question aggregation requires {id_field}")
        if not math.isfinite(float(score)) or not math.isfinite(float(label)) or not 0 <= float(label) <= 1:
            raise ValueError("finite COS and correctness in [0,1] required")
        groups[str(row[id_field])].append((float(score), float(label)))
    return ([sum(x for x, y in group) / len(group) for group in groups.values()],
            [sum(y for x, y in group) / len(group) for group in groups.values()],
            [len(group) for group in groups.values()])
