#!/usr/bin/env python3
"""Extract a normalized CAA direction from paired source answers."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; sys.path.insert(0,str(ROOT/"experiments")); sys.path.insert(0,str(ROOT/"src"))
from common import load_array, read_jsonl
from selectttrl.cos import mean_difference, normalize

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",required=True); ap.add_argument("--positive-hidden"); ap.add_argument("--negative-hidden"); ap.add_argument("--pairs"); ap.add_argument("--model"); ap.add_argument("--layer",type=int,default=21); ap.add_argument("--device-map",default="auto"); ap.add_argument("--trust-remote-code",action="store_true")
    args=ap.parse_args()
    if args.positive_hidden or args.negative_hidden:
        if not (args.positive_hidden and args.negative_hidden): raise ValueError("provide both hidden arrays")
        positive,negative=load_array(args.positive_hidden),load_array(args.negative_hidden)
    else:
        if not (args.pairs and args.model): raise ValueError("--pairs and --model are required when hidden arrays are absent")
        import torch; from transformers import AutoModelForCausalLM,AutoTokenizer
        rows=read_jsonl(args.pairs); tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=args.trust_remote_code); model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype="auto",device_map=args.device_map,trust_remote_code=args.trust_remote_code); model.eval(); device=next(model.parameters()).device
        pos=[]; neg=[]
        for row in rows:
            prompt=str(row.get("prompt","")); ptxt=prompt+str(row["positive"]); ntxt=prompt+str(row["negative"])
            def endpoint(text):
                enc=tok(text,return_tensors="pt"); enc={k:v.to(device) for k,v in enc.items()}
                with torch.inference_mode(): out=model(**enc,output_hidden_states=True,use_cache=False)
                return out.hidden_states[args.layer][0,-1].float().cpu().numpy()
            pos.append(endpoint(ptxt)); neg.append(endpoint(ntxt))
        positive,negative=pos,neg
    raw_direction=mean_difference(positive,negative)
    direction=normalize(raw_direction)
    import torch
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); torch.save({"direction":torch.as_tensor(direction),"raw_direction":torch.as_tensor(raw_direction),"raw_norm":float((raw_direction**2).sum()**0.5),"n_positive":len(positive),"n_negative":len(negative),"layer":args.layer},args.output)
    print(json.dumps({"n":len(positive),"dimension":int(direction.size),"output":args.output}))
if __name__=="__main__": main()
