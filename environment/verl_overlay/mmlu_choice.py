"""Gold-free MMLU-Pro option extraction and pseudo-label scoring.

MMLU-Pro STEM answers use labels A--J.  This module only sees generated
responses and a pseudo-label built from those responses; it never reads the
dataset answer column or any gold field.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


MMLU_CHOICE_SOURCE = Path(__file__).resolve()
MMLU_CHOICE_SHA256 = hashlib.sha256(MMLU_CHOICE_SOURCE.read_bytes()).hexdigest()
MMLU_NORMALIZER_NAME = "mmlu_a_j_last_balanced_box_or_explicit_final_v1"
MMLU_REWARD_GRADER_NAME = "mmlu_exact_a_j_against_plurality_pseudo_v1"
VALID = set("ABCDEFGHIJ")


def _last_balanced_boxed(text: str) -> str | None:
    starts = [text.rfind(r"\boxed{"), text.rfind(r"\fbox{")]
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


def _unwrap_latex(value: str) -> str:
    value = value.strip().strip("$").strip()
    for _ in range(4):
        unwrapped = re.fullmatch(
            r"\\(?:text|textbf|mathrm|mathbf|operatorname)\s*\{(.*)\}",
            value,
            flags=re.S,
        )
        if unwrapped is None:
            break
        value = unwrapped.group(1).strip()
    return value


def _normalize_surface(surface: str | None) -> str | None:
    if surface is None:
        return None
    value = _unwrap_latex(str(surface)).upper()
    # Prefer explicit option markers so letters in explanations/units do not
    # win, e.g. ``\text{(C) ...}``.
    for pattern in (r"\(\s*([A-J])\s*\)", r"\bOPTION\s*([A-J])\b"):
        match = re.search(pattern, value)
        if match and match.group(1) in VALID:
            return match.group(1)
    value = re.sub(r"\\(?:text|textbf|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", value)
    value = value.strip()
    if value in VALID:
        return value
    exact = re.fullmatch(r"[\s$\\()\[\].,:;*#_\-]*([A-J])[\s$\\()\[\].,:;*#_\-]*", value)
    if exact and exact.group(1) in VALID:
        return exact.group(1)
    match = re.match(r"^\s*([A-J])(?:\s|$|[).,:-])", value)
    return match.group(1) if match and match.group(1) in VALID else None


def extract_mmlu_option(text: str) -> str | None:
    """Extract the final A--J answer option from one response."""

    boxed = _last_balanced_boxed(text)
    option = _normalize_surface(boxed)
    if option is not None:
        return option
    tail = text[-3000:]
    patterns = (
        r"(?is)(?:final\s+)?(?:answer|option|choice)\s*(?:is|:|=)?\s*(?:\\boxed\s*\{\s*)?"
        r"(?:\\(?:text|textbf)\s*\{\s*)?\(?\s*([A-J])\s*\)?(?:\s*[).}:*]|\s|$)",
        r"(?im)^\s*\(?([A-J])\)?\s*[.!]?\s*$",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, tail))
        if matches:
            return matches[-1].group(1).upper()
    return None


def score_response_against_pseudo(response: str, pseudo_label: str | None) -> float:
    """Return an exact binary A--J reward against a generated pseudo-label."""

    if pseudo_label not in VALID:
        return 0.0
    return float(extract_mmlu_option(response) == pseudo_label)
