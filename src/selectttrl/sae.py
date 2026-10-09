"""SAE encoding, COS decomposition, and local feature clamping."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class SAEWeights:
    w_enc: torch.Tensor
    w_dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor


def load_sae(path: str, *, device="cpu") -> SAEWeights:
    raw = torch.load(path, map_location=device, weights_only=True)
    aliases = {"W_enc": "w_enc", "W_dec": "w_dec", "b_enc": "b_enc", "b_dec": "b_dec"}
    if not all(key in raw for key in aliases):
        raise ValueError("SAE checkpoint must contain W_enc, W_dec, b_enc, and b_dec")
    return SAEWeights(*(raw[key].float() for key in aliases))


def encode_topk(hidden: torch.Tensor, sae: SAEWeights, k: int = 50):
    pre = hidden.float() @ sae.w_enc.T + sae.b_enc
    k = min(int(k), pre.shape[-1])
    return pre.relu().topk(k, dim=-1)


def decompose(hidden: torch.Tensor, direction: torch.Tensor, sae: SAEWeights, k: int = 50) -> dict:
    """Return exact COS component terms for one hidden vector."""
    hidden, direction = hidden.float().reshape(-1), direction.float().reshape(-1)
    if hidden.numel() != direction.numel() or hidden.numel() != sae.w_dec.shape[0]:
        raise ValueError("hidden, direction, and SAE dimensions do not match")
    direction = direction / direction.norm()
    values, indices = encode_topk(hidden, sae, k)
    decoder = sae.w_dec[:, indices]
    reconstruction = decoder @ values + sae.b_dec
    residual = hidden - reconstruction
    denominator = hidden.norm()
    feature_terms = values * (direction @ decoder) / denominator
    bias_term = sae.b_dec @ direction / denominator
    residual_term = residual @ direction / denominator
    cos = hidden @ direction / denominator
    return {"cos": float(cos), "feature_sum": float(feature_terms.sum()),
            "bias": float(bias_term), "error": float(residual_term),
            "active_ids": indices.tolist(), "active_values": values.tolist(),
            "active_contributions": feature_terms.tolist(),
            "reconstruction_cos": float(torch.nn.functional.cosine_similarity(hidden[None], reconstruction[None])),
            "closure_error": float(abs(cos - feature_terms.sum() - bias_term - residual_term))}


def clamp_feature(hidden: torch.Tensor, feature_id: int, sae: SAEWeights, k: int = 50) -> torch.Tensor:
    """Set one feature to zero, retaining all other local terms and residual."""
    values, indices = encode_topk(hidden, sae, k)
    activation = (values * (indices == int(feature_id)).to(values.dtype)).sum(dim=-1, keepdim=True)
    return hidden - activation * sae.w_dec[:, int(feature_id)]
