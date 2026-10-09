#!/usr/bin/env python3
"""Print the strongest token-level SAE activations from a JSONL dump.

The input is intentionally generic: each row contains ``question_id`` and a
list of ``tokens``; each token has ``text`` and either ``active_ids`` with
``active_values`` or a ``features`` mapping. This is an appendix diagnostic,
not part of the main protocol.
"""
from __future__ import annotations
import argparse, json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--feature",type=int,required=True); ap.add_argument("--top",type=int,default=20); args=ap.parse_args(); hits=[]
    with open(args.input,encoding="utf-8") as f:
        for row in f:
            if not row.strip():continue
            item=json.loads(row)
            for pos,tok in enumerate(item.get("tokens",[])):
                if "features" in tok:
                    value=float(tok["features"].get(str(args.feature),0.0))
                else:
                    value=0.0
                    for fid,val in zip(tok.get("active_ids",[]),tok.get("active_values",[])):
                        if int(fid)==args.feature:value=float(val)
                if value>0:hits.append({"question_id":item.get("question_id",item.get("id")),"position":pos,"text":tok.get("text",""),"activation":value})
    for hit in sorted(hits,key=lambda x:(-x["activation"],str(x["question_id"])))[:args.top]:print(json.dumps(hit,ensure_ascii=False))
if __name__=="__main__":main()
