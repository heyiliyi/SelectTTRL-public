#!/usr/bin/env python3
"""Select the 35% subset used by Random or answer-PPL controls.

Input JSONL must contain a stable ``id`` (or ``question_id``). A-PPL sorts
by ``perplexity`` (or ``mean_nll``) descending; Random samples without
replacement using the supplied seed. By default, both controls use the
same finite-set count as the paper's COS interval. Only selection is
implemented here.
"""
from __future__ import annotations
import argparse, math, random, sys
from pathlib import Path
HERE=Path(__file__).resolve(); ROOT=HERE.parents[1]; sys.path.insert(0,str(ROOT/"experiments"))
from common import read_jsonl, write_jsonl

def selection_count(size, *, lower=.25, upper=.60, fraction=None):
    """Match COS endpoint rounding, or use an explicit non-paper fraction."""
    if size < 1:
        raise ValueError("nonempty input required")
    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError("require 0<fraction<=1")
        return max(1, math.floor(size * fraction))
    if not 0 <= lower < upper <= 1:
        raise ValueError("require 0 <= lower < upper <= 1")
    return math.floor(upper * size) - math.floor(lower * size)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--mode",choices=["random","appl"],required=True); ap.add_argument("--fraction",type=float,default=None,help="explicit fraction override for non-paper runs; default matches the COS interval count"); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--low",type=float,default=.25); ap.add_argument("--high",type=float,default=.60)
    ap.add_argument("--field",default="perplexity",help="A-PPL score field; mean_nll is also accepted")
    args=ap.parse_args(); rows=read_jsonl(args.input)
    n=selection_count(len(rows),lower=args.low,upper=args.high,fraction=args.fraction)
    ids=lambda i:str(i.get("id",i.get("question_id","")))
    if args.mode=="random":
        rng=random.Random(args.seed); chosen=set(rng.sample(range(len(rows)),n))
    else:
        def score(row):
            value=row.get(args.field,row.get("mean_nll"))
            if value is None: raise ValueError(f"missing {args.field}/mean_nll for {ids(row)}")
            return float(value)
        chosen=set(sorted(range(len(rows)),key=lambda i:(-score(rows[i]),ids(rows[i])))[:n])
    out=[]
    for i,row in enumerate(rows):
        item=dict(row); item.update({"selected":i in chosen,"selection_mode":args.mode,"selection_fraction":n/len(rows)})
        if i in chosen: out.append(item)
    write_jsonl(args.output,out); print({"mode":args.mode,"n_input":len(rows),"n_selected":len(out),"output":args.output})
if __name__=="__main__": main()
