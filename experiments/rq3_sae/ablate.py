#!/usr/bin/env python3
"""Autoregressively clamp one SAE feature at a time during generation."""
from __future__ import annotations
import argparse, sys
from dataclasses import dataclass
from pathlib import Path
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"experiments"))
from common import load_array, read_jsonl, write_jsonl
from selectttrl.cos import cosine_scores
from selectttrl.sae import load_sae

@dataclass
class _Clamp:
    model: object; sae: object; selected: tuple[int,...]; prompt_width: int; layer_index: int=21; token_scope: str="response_only"
    def __post_init__(self): self.handle=None; self.last_hidden=None; self.next_position=0; self.sae_cache={}
    def _hook(self,module,args,kwargs,output):
        import torch
        hidden=output[0] if isinstance(output,tuple) else output
        device=hidden.device; dtype=hidden.dtype
        key=(device,dtype)
        if key not in self.sae_cache:
            self.sae_cache[key]=tuple(x.to(device=device,dtype=torch.float32) for x in (self.sae.w_enc,self.sae.w_dec,self.sae.b_enc,self.sae.b_dec))
        wenc,wdec,benc,_=self.sae_cache[key]; pre=hidden.float()@wenc.T+benc; k=min(50,pre.shape[-1]); values,indices=pre.relu().topk(k,dim=-1)
        delta=torch.zeros_like(hidden.float())
        for fid in self.selected:
            mask=(indices==int(fid)).to(values.dtype); activation=(values*mask).unsqueeze(-1); delta=delta+(activation*wdec[:,int(fid)].reshape(1,1,1,-1)).sum(dim=-2)
        positions=kwargs.get("cache_position")
        if positions is None:
            positions=torch.arange(self.next_position,self.next_position+hidden.shape[1],device=device); self.next_position+=hidden.shape[1]
        else: positions=positions.reshape(-1).to(device)
        start = self.prompt_width - (1 if self.token_scope == "boundary_and_response" else 0)
        mask=torch.ones(hidden.shape[1],device=device,dtype=hidden.dtype) if self.token_scope == "all" else (positions>=start).to(hidden.dtype)
        edited=hidden-delta.to(dtype)*mask[None,:,None]
        self.last_hidden=edited[:,-1].detach()
        return (edited,)+output[1:] if isinstance(output,tuple) else edited
    def __enter__(self): self.handle=self.model.model.layers[self.layer_index-1].register_forward_hook(self._hook,with_kwargs=True); return self
    def __exit__(self,*exc):
        if self.handle is not None:self.handle.remove()
        self.handle=None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--sae",required=True); ap.add_argument("--features",type=int,nargs="+",required=True); ap.add_argument("--input",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--cos-direction"); ap.add_argument("--max-new-tokens",type=int,default=4096); ap.add_argument("--device-map",default="auto"); ap.add_argument("--layer-index",type=int,default=21); ap.add_argument("--token-scope",choices=["response_only","boundary_and_response","all"],default="response_only"); ap.add_argument("--trust-remote-code",action="store_true")
    args=ap.parse_args(); import torch; from transformers import AutoModelForCausalLM,AutoTokenizer
    rows=read_jsonl(args.input); tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=args.trust_remote_code); model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype="auto",device_map=args.device_map,trust_remote_code=args.trust_remote_code); model.eval(); device=next(model.parameters()).device; sae=load_sae(args.sae); direction=load_array(args.cos_direction).reshape(-1) if args.cos_direction else None
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    arms=[None]+args.features
    for feature in arms:
        records=[]
        for n,row in enumerate(rows):
            prompt=str(row.get("prompt",row.get("text",""))); enc=tok(prompt,return_tensors="pt"); enc={k:v.to(device) for k,v in enc.items()}; width=int(enc["input_ids"].shape[1]); ctx=_Clamp(model,sae,() if feature is None else (feature,),width,args.layer_index,args.token_scope)
            with ctx:
                with torch.inference_mode(): out=model.generate(**enc,max_new_tokens=args.max_new_tokens,do_sample=False,return_dict_in_generate=True,pad_token_id=tok.eos_token_id)
            seq=out.sequences[0]; response=tok.decode(seq[width:],skip_special_tokens=True); cos=None
            if direction is not None and ctx.last_hidden is not None:
                hidden=ctx.last_hidden.float().cpu().numpy()
                cos=float(cosine_scores(hidden,direction)[0])
            records.append({**row,"ablated_feature":feature,"response":response,"response_tokens":int(seq.numel()-width),"cos":cos})
        name="baseline" if feature is None else f"feature_{feature}"; write_jsonl(outdir/f"{name}.jsonl",records); print({"arm":name,"n":len(records)})
if __name__=="__main__": main()
