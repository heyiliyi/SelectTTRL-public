import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
import numpy as np
import torch
from selectttrl.cos import cosine_scores, select_percentile_interval
from selectttrl.sae import SAEWeights, decompose, clamp_feature

def test_cos_and_paper_interval():
    scores=cosine_scores(np.eye(4,dtype=np.float32),np.ones(4,dtype=np.float32))
    route=select_percentile_interval(scores,lower=.25,upper=.75,question_ids=["a","b","c","d"])
    assert len(route.selected_indices)==2

def test_sae_decomposition_closes():
    torch.manual_seed(0); d=4; m=6
    sae=SAEWeights(torch.randn(m,d),torch.randn(d,m),torch.randn(m),torch.randn(d))
    h=torch.randn(d); v=torch.randn(d); out=decompose(h,v,sae,k=m)
    assert out["closure_error"]<1e-5

def test_clamp_nonactive_is_identity():
    sae=SAEWeights(torch.ones(3,2),torch.eye(2,3),torch.zeros(3),torch.zeros(2)); h=torch.tensor([1.,-1.])
    assert torch.equal(clamp_feature(h,2,sae,k=3),h)


def test_steering_changes_only_boundary_and_response():
    from selectttrl.steering import ResidualSteering
    class Layer(torch.nn.Module):
        def forward(self, hidden, cache_position=None): return hidden
    class Core: pass
    class Config: model_type="qwen3"; hidden_size=2
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.config=Config(); self.model=Core(); self.model.layers=torch.nn.ModuleList([Layer()])
    model=Model().eval(); x=torch.zeros(1,3,2); v=torch.tensor([1.,0.])
    with torch.inference_mode():
        with ResidualSteering(model,v,hidden_index=1,alpha=2.,prompt_width=2,token_scope="boundary_and_response"):
            y=model.model.layers[0](x,cache_position=torch.tensor([0,1,2]))
    assert torch.equal(y[0,0],torch.zeros(2)); assert torch.equal(y[0,1],torch.tensor([2.,0.]))
