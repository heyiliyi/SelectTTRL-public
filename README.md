# SelectTTRL

SelectTTRL is a representation-guided data selection method for efficient
test-time reinforcement learning (TTRL). TTRL adapts a model to unlabeled
questions using majority-vote pseudo-labels, but sampling and updating on
every question can be costly. Our work asks which questions are worth training
on before this expensive stage.

We introduce COS, the cosine similarity between a response's residual-stream
representation and a correctness-related direction extracted from an independent
QA dataset using CAA. Motivated by the relationship between COS and question-level
accuracy, SelectTTRL ranks target questions and trains on the ascending
25th–60th percentile interval, without using target ground-truth answers.
The standard TTRL sampling, pseudo-labeling, and optimization procedures remain
unchanged for the selected questions.

This repository includes the selection and training pipeline, CAA steering,
SAE feature analysis, and source-direction transfer experiments.

## Setup

Use a clean VERL checkout at the commit in `environment/verl_commit.txt`.
Run the following commands from this repository's root in a compatible CUDA
environment:

```bash
pip install -r environment/requirements-verl.txt \
  -r environment/requirements-cuda.txt \
  -r environment/requirements-runtime.txt
export VERL_ROOT=/path/to/verl
git -C "$VERL_ROOT" checkout "$(cat environment/verl_commit.txt)"
pip install --no-deps -e "$VERL_ROOT"
```

Prepare model checkpoints and datasets separately. Parameter defaults are
listed in `configs/experiment_contract.yaml`; SAE weights are specified in
`configs/sae_manifest.json`.

## Usage

Generate the hidden cache, compute COS, and select questions:

```bash
python experiments/rq1_selectttrl/extract_endpoint_hidden.py \
  --model /path/to/model --input prompts.jsonl --input-format raw \
  --readout-mode boxed_or_response_last \
  --output-hidden target_hidden.pt --output-ids target_ids.jsonl

python experiments/rq1_selectttrl/score_and_route.py \
  --hidden target_hidden.pt --direction truthfulqa_layer21.pt \
  --ids target_ids.jsonl --output route.jsonl
```

Input rows contain `question_id` and `prompt`. Build `selected.parquet` from
rows with `selected=true`, and use the complete target set for validation:

```bash
python experiments/rq1_selectttrl/run_ttrl.py \
  --model /path/to/model --train-files selected.parquet \
  --val-files full_target.parquet --dataset math500 --output-dir runs/rq1 \
  --steps 120 --train-question-count 175 --override trainer.save_freq=120
```

Use the actual selected question count and a new output directory for each run.
The launcher checks epoch capacity and completed steps. Training uses 32 votes
and 16 update responses; validation reports Greedy, Mean1, and Maj.@16.

## SAE and transfer analysis

Validate the SAE checkpoint and produce the decomposition summary:

```bash
python tools/validate_sae.py --sae /path/to/layer20.sae.pt \
  --manifest configs/sae_manifest.json --hidden-index 21

python experiments/rq3_sae/decompose.py --hidden target_hidden.pt \
  --ids target_ids.jsonl --sae /path/to/layer20.sae.pt \
  --direction truthfulqa_layer21.pt --output decomposition.jsonl
```

Keep the cache metadata: RQ3 excludes truncated responses and summarizes
component contributions using the highest- and lowest-COS five questions.

Evaluate transfer using JSONL rows containing `dataset`, `question_id`,
`hidden`, and `correct`:

```bash
python experiments/rq4_source_transfer/evaluate_transfer.py \
  --direction source_direction.pt --targets target_hidden_and_labels.jsonl \
  --aggregation question --output transfer.json
```

The output includes question-level Pearson/Spearman correlations and dataset
macro averages. The four targets are MATH-500, AIME24, AIME25, and AMC 2022–2023.

## Experiment entry points

- **RQ1:** `experiments/rq1_selectttrl/`; Random/A-PPL: `baselines/select_35_percent.py`.
- **RQ2:** `experiments/rq2_caa_steering/steer.py`.
- **RQ3:** `experiments/rq3_sae/` — decomposition, feature selection, and ablation.
- **RQ4:** `experiments/rq4_source_transfer/` — direction extraction and transfer evaluation.

Run each script with `--help` for arguments. Data preparation, A-PPL scores,
and generated experiment artifacts are supplied separately.

Training uses `BatchTTRLRayPPOTrainer`; `group_batch_adapter.py` is an unused
alternative interface and is not called by the current training entry point.

## Checks

```bash
python -m pytest tests
python tools/audit_anonymity.py .
```
