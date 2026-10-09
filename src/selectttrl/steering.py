"""Inference-only residual-stream CAA steering for Qwen decoder models."""

from __future__ import annotations

import math
import torch


class ResidualSteering:
    """Add ``alpha * direction`` at a decoder block output.

    ``hidden_index=21`` means the output of Qwen3-8B block 20 (zero-based),
    i.e. decoder layer 21 in one-based numbering.
    """
    def __init__(self, model, direction, *, hidden_index=21, alpha=0.0,
                 prompt_width: int, token_scope="boundary_and_response", normalize_direction=True):
        model_type = getattr(model.config, "model_type", "")
        if model_type not in {"qwen3", "qwen2"}:
            raise ValueError("only Qwen2/Qwen3 decoder structure is supported")
        if token_scope not in {"response_only", "boundary_and_response", "all"}:
            raise ValueError("invalid token_scope")
        if not isinstance(prompt_width, int) or prompt_width < 1:
            raise ValueError("prompt_width must be positive")
        if not 1 <= hidden_index <= len(model.model.layers):
            raise ValueError("hidden_index must identify an intermediate block output")
        vector = torch.as_tensor(direction).detach().float().cpu()
        if vector.shape != (model.config.hidden_size,) or not torch.isfinite(vector).all():
            raise ValueError("direction shape or values are invalid")
        if vector.norm() == 0 or not math.isfinite(float(alpha)):
            raise ValueError("direction must be nonzero and alpha finite")
        self.model, self.layer = model, model.model.layers[hidden_index - 1]
        self.vector = vector / vector.norm() if normalize_direction else vector
        self.alpha, self.scope = float(alpha), token_scope
        self.normalize_direction = bool(normalize_direction)
        self.start = prompt_width - int(token_scope == "boundary_and_response")
        self.handle = self.delta = None
        self._fallback_position = 0

    def _hook(self, module, args, kwargs, output):
        if self.model.training or torch.is_grad_enabled():
            raise RuntimeError("steering is inference-only")
        hidden = output[0] if isinstance(output, tuple) else output
        if self.delta is None or self.delta.device != hidden.device or self.delta.dtype != hidden.dtype:
            self.delta = (self.alpha * self.vector).to(device=hidden.device, dtype=hidden.dtype)
        if self.scope == "all":
            edited = hidden + self.delta
        else:
            positions = kwargs.get("cache_position")
            if positions is None:
                positions = torch.arange(self._fallback_position, self._fallback_position + hidden.shape[1], device=hidden.device)
                self._fallback_position += hidden.shape[1]
            else:
                positions = positions.reshape(-1)
            if positions.numel() != hidden.shape[1]:
                raise RuntimeError("the model must expose one position per block token")
            mask = (positions.to(hidden.device) >= self.start).to(hidden.dtype)[None, :, None]
            edited = hidden + mask * self.delta
        return (edited,) + output[1:] if isinstance(output, tuple) else edited

    def __enter__(self):
        if self.model.training:
            raise RuntimeError("call model.eval() first")
        self.handle = self.layer.register_forward_hook(self._hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        self.handle = self.delta = None
        self._fallback_position = 0
