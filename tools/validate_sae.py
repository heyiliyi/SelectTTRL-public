#!/usr/bin/env python3
"""Validate the frozen SAE artifact; never download weights or run a model."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hidden-index", type=int, default=21)
    args = ap.parse_args()
    meta = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if meta["layer"] + 1 != args.hidden_index or meta["hidden_index"] != args.hidden_index:
        raise ValueError("SAE layer does not match the requested residual hidden index")
    if (meta["release_encoder"], meta["centered_input"], meta["input_normalization"], meta["topk"]) != ("linear_relu_topk", False, "none", 50):
        raise ValueError("manifest does not describe the unchanged v1 encoding")
    digest = hashlib.sha256()
    with Path(args.sae).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != meta["checkpoint_sha256"]:
        raise ValueError("SAE checkpoint SHA256 mismatch")
    import torch
    raw = torch.load(args.sae, map_location="cpu", weights_only=True)
    d, f = int(meta["hidden_size"]), int(meta["feature_count"])
    shapes = {"W_enc": (f, d), "W_dec": (d, f), "b_enc": (f,), "b_dec": (d,)}
    if sorted(meta["state_keys"]) != sorted(shapes):
        raise ValueError("manifest state key mismatch")
    for key, shape in shapes.items():
        if key not in raw or tuple(raw[key].shape) != shape:
            raise ValueError(f"SAE tensor {key} has an unexpected shape")
    print(json.dumps({"status": "validated", "sha256": digest.hexdigest(),
        "layer": meta["layer"], "hidden_index": args.hidden_index}))


if __name__ == "__main__":
    main()
