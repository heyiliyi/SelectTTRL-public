"""Load the bundled TTRL math normalizer under an isolated package name."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


TTRL_MATH_DIR = Path(__file__).resolve().parents[1] / "ttrl_math"
TTRL_MATH_INIT = TTRL_MATH_DIR / "__init__.py"
TTRL_NORMALIZER_NAME = "TTRL extract_answer + simplify_expression_string"
TTRL_REWARD_GRADER_NAME = "TTRL compute_score(response, pseudo_label, fast=False)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


TTRL_MATH_SHA256 = _sha256(TTRL_MATH_INIT)


@lru_cache(maxsize=1)
def _module() -> ModuleType:
    if not TTRL_MATH_INIT.is_file():
        raise FileNotFoundError(TTRL_MATH_INIT)
    package_name = "_dataset_level_grpo_ttrl_math"
    existing = sys.modules.get(package_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        package_name,
        TTRL_MATH_INIT,
        submodule_search_locations=[str(TTRL_MATH_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load TTRL math normalizer from {TTRL_MATH_INIT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def extract_and_simplify(response: str) -> str | None:
    """Apply exactly the answer extraction/standardization used for TTRL vote."""

    module = _module()
    answer = module.extract_answer(str(response))
    if answer is None:
        return None
    return str(module.simplify_expression_string(answer))


def score_response_against_pseudo(response: str, pseudo_label: str | None) -> float:
    """Apply the exact reward path used by TTRL after plurality voting."""

    if pseudo_label is None:
        return 0.0
    result = _module().compute_score(str(response), str(pseudo_label), fast=False)
    if isinstance(result, dict):
        return float(result["score"])
    if isinstance(result, (int, float, bool)):
        return float(result)
    return float(result[0])


def grade_answer(model_answer: str, ground_truth: str) -> bool:
    """Use TTRL's bundled fast math grader for offline audit only."""

    return bool(_module().grade(str(model_answer), str(ground_truth), fast=True))
