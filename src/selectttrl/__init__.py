"""Core, model-agnostic primitives for the anonymous SelectTTRL release."""
from .cos import cosine_scores, mean_difference, normalize, select_percentile_interval
from .sae import clamp_feature, decompose, encode_topk, load_sae

__all__ = ["cosine_scores", "mean_difference", "normalize", "select_percentile_interval",
           "clamp_feature", "decompose", "encode_topk", "load_sae"]
