#!/usr/bin/env python3
"""Decompose endpoint hidden states into SAE, bias, and residual COS terms."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"experiments"))
from common import load_array, read_jsonl, write_jsonl
from selectttrl.sae import decompose, load_sae
from reproduction_helpers import ranking_components

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--hidden",required=True); ap.add_argument("--sae",required=True); ap.add_argument("--direction",required=True); ap.add_argument("--output",required=True); ap.add_argument("--ids"); ap.add_argument("--topk",type=int,default=50)
    args=ap.parse_args(); hidden=load_array(args.hidden); direction=load_array(args.direction).reshape(-1); sae=load_sae(args.sae)
    if hidden.ndim!=2: raise ValueError("hidden must have shape [n, hidden]")
    if len(hidden) == 0: raise ValueError("hidden cache must not be empty")
    ids=read_jsonl(args.ids) if args.ids else [{} for _ in range(len(hidden))]
    if len(ids)!=len(hidden): raise ValueError("ids and hidden row counts differ")
    rows=[]
    for i,vector in enumerate(hidden):
        record=decompose(__import__('torch').as_tensor(vector),__import__('torch').as_tensor(direction),sae,args.topk)
        item=dict(ids[i]); item.setdefault("question_id",str(item.get("id",i))); item.update(record); rows.append(item)
    write_jsonl(args.output,rows)
    summary={"n":len(rows),"mean_cos":sum(r["cos"] for r in rows)/len(rows),"mean_feature_sum":sum(r["feature_sum"] for r in rows)/len(rows),"mean_bias":sum(r["bias"] for r in rows)/len(rows),"mean_error":sum(r["error"] for r in rows)/len(rows),"max_closure_error":max(r["closure_error"] for r in rows)}
    summary["high_low"] = ranking_components(rows)
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary))
if __name__=="__main__": main()
