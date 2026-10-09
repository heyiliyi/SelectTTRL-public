#!/usr/bin/env python3
"""Launch the process-local VERL batch-TTRL overlay with paper defaults."""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]

DATASET_SETTINGS = {
    "math500": ("math", 16), "gpqa": ("gpqa", 16),
    "mmlu_pro_stem": ("mmlu", 8), "aime24": ("math", 8),
    "aime25": ("math", 8), "amc": ("math", 8),
}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True); ap.add_argument("--train-files", required=True)
    ap.add_argument("--val-files", required=True)
    ap.add_argument("--dataset", required=True, choices=DATASET_SETTINGS)
    ap.add_argument("--task", choices=["math","mmlu","gpqa"], help="optional check; inferred from --dataset")
    ap.add_argument("--steps", type=int, default=120); ap.add_argument("--gpus", type=int, default=4); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--verl-root", default=os.environ.get("VERL_ROOT"))
    ap.add_argument("--train-question-count", type=int, help="selected rows after filtering; optional capacity estimate")
    ap.add_argument("--total-epochs", type=int, help="explicit epoch capacity; must be sufficient for --steps")
    ap.add_argument("--override", action="append", default=[], help="additional stock VERL Hydra key=value override")
    args=ap.parse_args()
    task, train_batch_size = DATASET_SETTINGS[args.dataset]
    if args.task is not None and args.task != task:
        ap.error("--task does not match --dataset")
    args.task = task
    if not args.verl_root: ap.error("--verl-root or VERL_ROOT is required")
    if args.steps <= 0 or args.gpus <= 0: ap.error("steps and gpus must be positive")
    if args.train_question_count is not None:
        batches = args.train_question_count // train_batch_size
        if batches < 1: ap.error("train-question-count must provide at least one full batch")
        minimum_epochs = math.ceil(args.steps / batches)
    else:
        # At least one batch/epoch is required by VERL; this capacity is safe
        # without guessing parquet row counts before runtime filtering.
        minimum_epochs = args.steps
    total_epochs = args.total_epochs if args.total_epochs is not None else minimum_epochs
    if total_epochs <= 0: ap.error("total-epochs must be positive")
    if args.train_question_count is not None and total_epochs < minimum_epochs:
        ap.error(f"epoch capacity cannot reach {args.steps} steps; need at least {minimum_epochs}")
    protected = {"trainer.total_training_steps", "trainer.total_epochs", "trainer.default_local_dir",
                 "data.train_batch_size", "data.gen_batch_size", "actor_rollout_ref.rollout.n"}
    if any(value.split("=", 1)[0].lstrip("+~") in protected for value in args.override):
        ap.error("use launcher arguments for step, epoch, directory and batch contract; do not override them")
    run_dir = Path(args.output_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        ap.error("output-dir must be new or empty; existing runs are never overwritten")
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "training_progress.json"
    overlay=ROOT/"environment"/"verl_overlay"; entry=overlay/"run_batch_ttrl.py"
    env=os.environ.copy(); env["VERL_ROOT"]=str(Path(args.verl_root).resolve()); env["VERL_CONFIG_DIR"]=str(Path(args.verl_root)/"verl"/"trainer"/"config")
    env["TTRL_VOTE_N"]="32"; env["TTRL_TRAIN_N"]="16"; env["TTRL_TASK"]=args.task
    env["TTRL_AUDIT_PATH"] = str(run_dir / "ttrl_group_audit.jsonl")
    env["TTRL_FULL_ROLLOUT_DIR"] = str(run_dir / "full_vote_rollouts")
    env["TTRL_PROGRESS_PATH"] = str(progress_path)
    env["PYTHONPATH"]=os.pathsep.join([str(overlay), str(ROOT/"environment"), env.get("PYTHONPATH","")])
    overrides=["actor_rollout_ref.model.path="+args.model, "data.train_files="+args.train_files,
               "data.val_files="+args.val_files, "trainer.total_training_steps="+str(args.steps),
               "data.train_batch_size="+str(train_batch_size),
               "trainer.total_epochs="+str(total_epochs),
               "actor_rollout_ref.actor.ppo_mini_batch_size=1",
               "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
               "trainer.test_freq="+str(args.steps),
               "trainer.default_local_dir="+str(run_dir), "actor_rollout_ref.rollout.name=vllm",
               "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
               "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
               "actor_rollout_ref.rollout.n=16",
               "actor_rollout_ref.rollout.temperature=0.6", "actor_rollout_ref.rollout.top_p=1.0",
               "actor_rollout_ref.rollout.top_k=-1", "actor_rollout_ref.rollout.max_model_len=8192",
               "data.max_prompt_length=8192", "data.max_response_length=4096",
               "data.filter_overlong_prompts=true", "data.truncation=error",
               "+data.apply_chat_template_kwargs.enable_thinking=false",
               "algorithm.adv_estimator=grpo", "actor_rollout_ref.actor.optim.lr=5e-7",
               "actor_rollout_ref.actor.optim.lr_scheduler_type=cosine", "trainer.n_gpus_per_node="+str(args.gpus),
               "trainer.nnodes=1", "trainer.logger=[console]",
               "reward.custom_reward_function.path="+str(overlay/"placeholder_reward.py"),
               "reward.custom_reward_function.name=compute_score"] + list(args.override)
    cmd=[sys.executable,str(entry),*overrides]
    print("Launching:", " ".join(cmd)); subprocess.run(cmd, env=env, check=True)
    if not progress_path.is_file():
        raise RuntimeError(f"training produced no completed-step marker: {progress_path}")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if int(progress.get("requested_steps", -1)) != args.steps or int(progress.get("last_step", -1)) != args.steps:
        raise RuntimeError(f"actual training progress does not match requested {args.steps} steps: {progress}")

if __name__=="__main__": main()
