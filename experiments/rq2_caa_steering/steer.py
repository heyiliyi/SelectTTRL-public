#!/usr/bin/env python3
"""Run the RQ2 CAA steering sweep on prompt JSONL."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"experiments"))
from common import load_array, read_jsonl, write_jsonl
from selectttrl.steering import ResidualSteering


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True); ap.add_argument("--direction",required=True); ap.add_argument("--input",required=True); ap.add_argument("--output-dir",required=True)
    ap.add_argument("--alphas",type=float,nargs="+",default=[-1,-.5,.5,1]); ap.add_argument("--hidden-index",type=int,default=21)
    ap.add_argument("--token-scope",choices=["response_only","boundary_and_response","all"],default="boundary_and_response")
    ap.add_argument("--max-new-tokens",type=int,default=4096); ap.add_argument("--do-sample",action="store_true"); ap.add_argument("--temperature",type=float,default=.6)
    ap.add_argument("--device-map",default="auto"); ap.add_argument("--unit-direction",action="store_true",help="opt in to a normalized direction; default is raw CAA mean difference"); ap.add_argument("--trust-remote-code",action="store_true")
    args=ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows=read_jsonl(args.input); direction=load_array(args.direction,preferred="direction" if args.unit_direction else "raw_direction").reshape(-1)
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=args.trust_remote_code)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype="auto",device_map=args.device_map,trust_remote_code=args.trust_remote_code)
    model.eval(); device=next(model.parameters()).device
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    for alpha in args.alphas:
        records=[]
        for n,row in enumerate(rows):
            prompt=str(row.get("prompt",row.get("text","")))
            if not prompt: raise ValueError(f"row {n} has no prompt/text")
            encoded=tokenizer(prompt,return_tensors="pt"); encoded={k:v.to(device) for k,v in encoded.items()}
            width=int(encoded["input_ids"].shape[1])
            with ResidualSteering(model,direction,hidden_index=args.hidden_index,alpha=alpha,prompt_width=width,token_scope=args.token_scope,normalize_direction=args.unit_direction):
                generate_kwargs=dict(max_new_tokens=args.max_new_tokens,do_sample=args.do_sample,
                                     return_dict_in_generate=True,output_hidden_states=False,
                                     pad_token_id=tokenizer.eos_token_id)
                if args.do_sample: generate_kwargs["temperature"]=args.temperature
                with torch.inference_mode(): generated=model.generate(**encoded,**generate_kwargs)
            sequence=generated.sequences[0]; text=tokenizer.decode(sequence[width:],skip_special_tokens=True)
            # Re-read COS with a clean forward pass, as required by RQ2;
            # the intervention affects the response tokens but not this readout.
            cos=None
            with torch.inference_mode():
                clean=model(input_ids=sequence[None],use_cache=False,output_hidden_states=True)
            hidden=clean.hidden_states[args.hidden_index][0,-1].float()
            vec=torch.as_tensor(direction,device=hidden.device,dtype=hidden.dtype)
            cos=float(torch.dot(hidden,vec)/(hidden.norm()*vec.norm()))
            records.append({**row,"alpha":float(alpha),"response":text,"response_tokens":int(sequence.numel()-width),"cos":cos})
        write_jsonl(outdir/f"alpha_{alpha:g}.jsonl",records)
        print(json.dumps({"alpha":alpha,"n":len(records)},ensure_ascii=False))

if __name__=="__main__": main()
