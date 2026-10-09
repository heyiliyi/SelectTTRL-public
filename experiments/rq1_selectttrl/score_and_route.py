#!/usr/bin/env python3
"""Compute COS and select the paper's ascending 25--60 percentile interval."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"experiments"))
from selectttrl.cos import cosine_scores, select_percentile_interval
from common import load_array, read_jsonl, write_jsonl


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hidden", required=True, help="[n,d] endpoint representations (.pt/.npy/.npz)")
    ap.add_argument("--direction", required=True, help="[d] CAA direction (.pt/.npy)")
    ap.add_argument("--ids", required=True, help="JSONL with id/question_id, one row per hidden vector")
    ap.add_argument("--output", required=True)
    ap.add_argument("--low", type=float, default=.25); ap.add_argument("--high", type=float, default=.60)
    args=ap.parse_args()
    hidden=load_array(args.hidden); direction=load_array(args.direction).reshape(-1)
    rows=read_jsonl(args.ids)
    if len(rows)!=len(hidden): raise ValueError("ids and hidden row counts differ")
    ids=[str(r.get("id", r.get("question_id", i))) for i,r in enumerate(rows)]
    scores=cosine_scores(hidden, direction); route=select_percentile_interval(scores, lower=args.low, upper=args.high, question_ids=ids)
    selected=set(route.selected_indices); order=sorted(range(len(rows)), key=lambda i:(float(scores[i]), ids[i]))
    ranks={idx:rank for rank,idx in enumerate(order, start=1)}
    out=[]
    for i,row in enumerate(rows):
        item=dict(row); item.update({"cos":float(scores[i]), "ascending_rank":ranks[i], "selected":i in selected,
                                     "selection_low":args.low, "selection_high":args.high})
        out.append(item)
    write_jsonl(args.output,out)
    print({"n":len(out),"selected":len(selected),"fraction":route.selected_fraction,"output":args.output})

if __name__=="__main__": main()
