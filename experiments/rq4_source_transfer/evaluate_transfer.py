#!/usr/bin/env python3
"""Evaluate COS/correctness correlations for one frozen source direction."""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; sys.path.insert(0,str(ROOT/"experiments")); sys.path.insert(0,str(ROOT/"src"))
from common import load_array, read_jsonl
from selectttrl.cos import cosine_scores
from reproduction_helpers import question_aggregates

def ranks(values):
    order=sorted(range(len(values)),key=lambda i:float(values[i])); out=[0.0]*len(values); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and float(values[order[j]])==float(values[order[i]]): j+=1
        rank=(i+j-1)/2+1
        for k in range(i,j): out[order[k]]=rank
        i=j
    return out

def corr(x,y):
    if len(x)<2:return None
    mx=sum(x)/len(x); my=sum(y)/len(y); dx=[a-mx for a in x]; dy=[b-my for b in y]; den=math.sqrt(sum(a*a for a in dx)*sum(b*b for b in dy))
    return None if den==0 else sum(a*b for a,b in zip(dx,dy))/den

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--direction",required=True); ap.add_argument("--targets",required=True,help="JSONL rows with hidden, correct, dataset and question_id"); ap.add_argument("--output",required=True)
    ap.add_argument("--aggregation", choices=["question", "rollout"], default="question")
    ap.add_argument("--question-id-field", default="question_id")
    args=ap.parse_args()
    direction=load_array(args.direction).reshape(-1); rows=read_jsonl(args.targets); hidden=[]; labels=[]
    if not rows: raise ValueError("target rows must not be empty")
    for r in rows:
        if "hidden" not in r or "correct" not in r: raise ValueError("each target row needs hidden and correct")
        hidden.append(r["hidden"]); labels.append(float(r["correct"]))
    scores=cosine_scores(hidden,direction); groups=defaultdict(list)
    for i,r in enumerate(rows):groups[str(r.get("dataset","all"))].append(i)
    report={"n":len(rows),"aggregation":args.aggregation,"groups":{}}
    for name,idx in groups.items():
        x=[float(scores[i]) for i in idx]; y=[labels[i] for i in idx]
        counts = [1] * len(idx)
        if args.aggregation == "question":
            x, y, counts = question_aggregates([rows[i] for i in idx], x, y, args.question_id_field)
        report["groups"][name]={"n":len(x),"rows":len(idx),"pearson":corr(x,y),"spearman":corr(ranks(x),ranks(y)),"mean_cos":sum(x)/len(x),"accuracy":sum(y)/len(y),"responses_per_question":counts}
    report["macro_average"] = {}
    for metric in ("pearson", "spearman"):
        valid = [item[metric] for item in report["groups"].values() if item[metric] is not None]
        report["macro_average"][metric] = sum(valid) / len(valid) if valid else None
        report["macro_average"][metric + "_dataset_count"] = len(valid)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report))
if __name__=="__main__": main()
