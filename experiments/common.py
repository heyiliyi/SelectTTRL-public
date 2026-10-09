"""Small file-format helpers shared by release scripts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_array(path, preferred=None):
    """Load a tensor/array from .pt, .npy, or .npz without assuming a private layout."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        raw = np.load(path)
        key = preferred or ("hidden" if "hidden" in raw else raw.files[0])
        return raw[key]
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        keys = [preferred] if preferred else []
        keys += ["hidden", "activations", "last_hidden_state", "features", "tensor", "direction", "vectors"]
        for key in keys:
            if key and key in raw:
                raw = raw[key]; break
        else:
            tensor_values = [v for v in raw.values() if torch.is_tensor(v) or isinstance(v, np.ndarray)]
            if len(tensor_values) != 1:
                raise ValueError(f"cannot identify an array in {path}")
            raw = tensor_values[0]
    if torch.is_tensor(raw):
        return raw.detach().cpu().numpy()
    return np.asarray(raw)


def save_torch(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
