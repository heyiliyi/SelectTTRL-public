"""Gold-free GPQA A-D answer extraction and pseudo-label scoring.

This module is experiment-local and intentionally has no dataset loader.  It
only receives generated response text and a plurality pseudo-label produced
from other generated responses.  Therefore it cannot access GPQA gold labels.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


GPQA_CHOICE_SOURCE = Path(__file__).resolve()
GPQA_CHOICE_SHA256 = hashlib.sha256(GPQA_CHOICE_SOURCE.read_bytes()).hexdigest()
GPQA_NORMALIZER_NAME = "gpqa_a_d_last_balanced_box_or_explicit_final_v1"
GPQA_REWARD_GRADER_NAME = "gpqa_exact_a_d_against_plurality_pseudo_v1"


def _last_balanced_boxed(text: str) -> str | None:
    spans: list[str] = []
    cursor = text.find(r"\boxed{")
    while cursor >= 0:
        start = cursor + len(r"\boxed{")
        depth = 1
        index = start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            spans.append(text[start : index - 1].strip())
        cursor = text.find(r"\boxed{", cursor + 1)
    return spans[-1] if spans else None


def _normalize_option_surface(surface: str) -> str | None:
    value = surface.strip().strip("$").strip()
    wrapped_label = re.match(
        r"^\\(?:text|textbf|mathrm|mathbf|operatorname)\s*\{\s*\(?\s*([A-D])\s*\)?(?:\s*[.:-]|\s*\})",
        value,
    )
    if wrapped_label:
        return wrapped_label.group(1).upper()
    for _ in range(3):
        wrapper = re.fullmatch(
            r"\\(?:text|textbf|mathrm|mathbf|operatorname)\s*\{(.*)\}",
            value,
            flags=re.S,
        )
        if wrapper is None:
            break
        value = wrapper.group(1).strip()
    explicit = re.fullmatch(
        r"(?is)(?:the\s+)?(?:final\s+)?(?:answer|option|choice)?\s*[:=]?\s*\(?\s*([A-D])\s*\)?[.\s]*",
        value,
    )
    if explicit:
        return explicit.group(1).upper()
    # The local GPQA prompts sometimes retain the original lower-case choice
    # surface after the shuffled upper-case label, for example ``D. c``.
    labeled_choice = re.match(
        r"(?s)^(?:final\s+)?(?:answer|option|choice)?\s*[:=]?\s*\(?\s*([A-D])\s*\)?(?:\s*[.:-]|\s*$)",
        value,
    )
    return labeled_choice.group(1).upper() if labeled_choice else None


def extract_gpqa_option(text: str) -> str | None:
    """Extract the final A-D label using the frozen evaluation parser."""

    boxed = _last_balanced_boxed(text)
    if boxed is not None:
        option = _normalize_option_surface(boxed)
        if option is not None:
            return option
    tail = text[-2000:]
    patterns = (
        r"(?is:(?:final|correct)\s+(?:answer|option|choice)\s*(?:is|:|=)?\s*(?:\\boxed\s*\{\s*)?(?:\\(?:text|textbf)\s*\{\s*)?[\s$*_#✅-]*)\(?\s*([A-D])(?:\s*[).}:*]|\s|$)",
        r"(?im)^\s*(?:[#>*✅-]+\s*)?answer\s*(?:is|:|=)\s*[\s$*_]*(?:\\boxed\s*\{\s*)?(?:\\(?:text|textbf)\s*\{\s*)?\(?\s*([A-D])(?:\s*[).}:*]|\s|$)",
        r"(?im)^\s*\(?([A-D])\)?\s*[.!]?\s*$",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, tail))
        if matches:
            return matches[-1].group(1).upper()
    return None


def score_response_against_pseudo(response: str, pseudo_label: str | None) -> float:
    """Return an exact binary A-D reward against a generated pseudo-label."""

    if pseudo_label not in {"A", "B", "C", "D"}:
        return 0.0
    return float(extract_gpqa_option(response) == pseudo_label)

