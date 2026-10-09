# VERL environment contract

The release uses a clean checkout of stock VERL at the commit recorded in
`verl_commit.txt`, plus the process-local files in `verl_overlay/`. The
overlay is loaded through `PYTHONPATH`; it does not modify the stock checkout.

Install `requirements-verl.txt`, `requirements-cuda.txt`, and the
known-good pins in `requirements-runtime.txt` in the same Python environment. The original run was validated with Transformers 4.56.1,
Torch 2.8.0, vLLM 0.11.0, Ray 2.54.0, TensorDict 0.10.0, and FlashAttention
2.8.1. These are compatibility observations rather than bundled binaries;
CUDA and GPU driver versions remain host responsibilities.

`run_batch_ttrl.py` sets the paper TTRL contract through `TTRL_VOTE_N=32`,
`TTRL_TRAIN_N=16`, and `TTRL_TASK`. Dataset preparation, model checkpoints,
and generated artifacts stay outside the release tree.
