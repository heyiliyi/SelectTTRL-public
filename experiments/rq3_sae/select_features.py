#!/usr/bin/env python3
"""Select COS-contributing SAE features under the paper protocol.

The paper-facing candidate pool is defined from complete endpoint rows: a
feature's activity score is the mean of its *nonzero* activations when it is
present in the sparse Top-K code. This is a conditional activation mean; zero
entries for rows where the feature is not in the code are not averaged in.
The five candidates are then ranked by the absolute signed high-COS minus
low-COS contribution gap.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; sys.path.insert(0,str(ROOT/"experiments"))
from common import read_jsonl

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--decomposition",required=True); ap.add_argument("--output",required=True); ap.add_argument("--top-features",type=int,default=50); ap.add_argument("--select-k",type=int,default=5); ap.add_argument("--high-low-k",type=int,default=5)
    args=ap.parse_args(); rows=read_jsonl(args.decomposition)
    if min(args.top_features,args.select_k,args.high_low_k) <= 0:
        raise ValueError("feature counts and high-low-k must be positive")
    if any(not isinstance(r.get("truncated"),bool) for r in rows):
        raise ValueError("boolean truncated metadata required for every row")
    usable=[r for r in rows if not bool(r.get("truncated",False))]
    if len(usable)<2*args.high_low_k: raise ValueError("not enough complete rows for high/low groups")
    usable.sort(key=lambda r:float(r["cos"])); low=usable[:args.high_low_k]; high=usable[-args.high_low_k:]
    act=defaultdict(list); contrib=defaultdict(list); counts=defaultdict(int)
    # Candidate activity is measured on the same complete endpoint rows used
    # for the high/low COS comparison. Including truncated/fallback rows can
    # change the candidate pool and is not the paper-facing protocol.
    activity_rows = usable
    for r in activity_rows:
        seen=set()
        for fid,val,term in zip(r.get("active_ids",[]),r.get("active_values",[]),r.get("active_contributions",[])):
            fid=int(fid); numeric=float(val)
            if numeric > 0:
                act[fid].append(numeric)
                contrib[fid].append((str(r.get("question_id",r.get("id",""))),float(term)))
                seen.add(fid)
        for fid in seen: counts[fid]+=1
    # Do not divide by the number of questions: that would treat sparse
    # non-activation as an observed zero and would exclude rare but strongly
    # activated features such as AIME24 feature 52642.
    mean_activation={fid:sum(vals)/len(vals) for fid,vals in act.items()}
    candidates=sorted(mean_activation,key=lambda f:(-mean_activation[f],f))[:args.top_features]
    def group_mean(group,fid):
        values=[]
        for r in group:
            values.append(sum(float(term) for x,term in zip(r.get("active_ids",[]),r.get("active_contributions",[])) if int(x)==fid))
        return sum(values)/len(values)
    table=[]
    for fid in candidates:
        hi,lo=group_mean(high,fid),group_mean(low,fid); table.append({"feature_id":fid,"mean_activation":mean_activation[fid],"active_questions":counts[fid],"high_contribution":hi,"low_contribution":lo,"signed_gap":hi-lo,"abs_gap":abs(hi-lo)})
    table.sort(key=lambda r:(-r["abs_gap"],r["feature_id"])); selected=[r["feature_id"] for r in table[:args.select_k]]
    result={"protocol":"paper","n_rows":len(rows),"complete_rows":len(usable),"high_question_ids":[str(r.get("question_id",r.get("id",""))) for r in high],"low_question_ids":[str(r.get("question_id",r.get("id",""))) for r in low],"candidate_pool_size":len(candidates),"candidate_activity_stat":"mean nonzero activation over complete endpoint rows","selected_features":selected,"ranking":table}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps({"protocol":"paper","selected_features":selected}))
if __name__=="__main__": main()
