#!/usr/bin/env python3
"""Generate a greedy answer-position cache without changing v1 interventions."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from reproduction_helpers import nonthinking_prompt, answer_position


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "input", "output-hidden", "output-ids"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--layer-index", type=int, default=21)
    ap.add_argument("--readout-mode", choices=["boxed_or_response_last", "response_last", "sequence_last", "index"], required=True)
    ap.add_argument("--readout-index", type=int)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--max-context-tokens", type=int, default=8192)
    ap.add_argument("--input-format", choices=["raw", "chat", "preformatted"], default="raw")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or args.max_new_tokens <= 0:
        raise ValueError("nonempty input and positive generation budget required")
    out_hidden, out_ids = Path(args.output_hidden), Path(args.output_ids)
    metadata_path = out_hidden.with_suffix(".metadata.json")
    for path in (out_hidden, out_ids, metadata_path):
        if path.exists():
            raise FileExistsError(path)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map=args.device_map,
        trust_remote_code=args.trust_remote_code).eval()
    device = model.get_input_embeddings().weight.device
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
    if tok.eos_token_id is not None:
        eos_ids.add(tok.eos_token_id)
    hidden, ids, seen = [], [], set()
    for n, row in enumerate(rows):
        qid = str(row.get("question_id", row.get("id", n)))
        if qid in seen:
            raise ValueError(f"duplicate question ID: {qid}")
        seen.add(qid)
        prompt = nonthinking_prompt(tok, row, args.input_format)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {key: value.to(device) for key, value in enc.items()}
        width = int(enc["input_ids"].shape[1])
        budget = min(args.max_new_tokens, args.max_context_tokens - width)
        if budget <= 0:
            raise ValueError(f"prompt exceeds context budget for {qid}")
        with torch.inference_mode():
            generated = model.generate(**enc, max_new_tokens=budget, do_sample=False,
                return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
            seq = generated.sequences[0]
            forward = model(input_ids=seq[None], use_cache=False, output_hidden_states=True)
        tokens = seq.tolist()
        position, reason = answer_position(tok, tokens, width, args.readout_mode, args.readout_index, eos_ids)
        hidden.append(forward.hidden_states[args.layer_index][0, position].float().cpu())
        count = len(tokens) - width
        ids.append({"id": qid, "question_id": qid, "readout_position": position,
            "readout_mode": args.readout_mode, "readout_reason": reason,
            "prompt_width": width, "response_tokens": count,
            "truncated": bool(count >= budget and tokens[-1] not in eos_ids),
            "response": tok.decode(tokens[width:], skip_special_tokens=True)})
        del forward, generated
    for path in (out_hidden, out_ids, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack(hidden), out_hidden)
    out_ids.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in ids) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps({"schema": "endpoint_hidden_cache_v1", "model": args.model,
        "layer_index": args.layer_index, "readout_mode": args.readout_mode,
        "input_format": args.input_format, "enable_thinking": False,
        "max_context_tokens": args.max_context_tokens, "max_new_tokens": args.max_new_tokens,
        "n": len(ids)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(ids), "hidden": str(out_hidden), "ids": str(out_ids)}))


if __name__ == "__main__":
    main()
