"""Check dataset-specific launcher settings without starting VERL."""
import runpy
import sys
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

ENTRY = Path(__file__).resolve().parents[1] / "experiments/rq1_selectttrl/run_ttrl.py"


class LauncherTests(unittest.TestCase):
    def test_paper_dataset_settings(self):
        for dataset, task, batch in (("math500", "math", 16), ("gpqa", "gpqa", 16),
                                     ("mmlu_pro_stem", "mmlu", 8), ("aime24", "math", 8),
                                     ("aime25", "math", 8), ("amc", "math", 8)):
            with self.subTest(dataset=dataset):
                argv = [str(ENTRY), "--model", "model", "--train-files", "train.parquet",
                        "--val-files", "val.parquet", "--dataset", dataset,
                        "--output-dir", "runs/test", "--verl-root", "/path/to/verl"]
                with tempfile.TemporaryDirectory() as directory:
                    argv[argv.index("--output-dir") + 1] = directory
                    def completed(cmd, *, env, check):
                        Path(env["TTRL_PROGRESS_PATH"]).write_text(json.dumps(
                            {"last_step": 120, "requested_steps": 120}), encoding="utf-8")
                    with patch.object(sys, "argv", argv), patch("subprocess.run", side_effect=completed) as launch, patch("builtins.print"):
                        runpy.run_path(str(ENTRY), run_name="__main__")
                cmd = launch.call_args.args[0]
                self.assertIn(f"data.train_batch_size={batch}", cmd)
                self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=1", cmd)
                self.assertIn("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1", cmd)
                self.assertIn("actor_rollout_ref.rollout.name=vllm", cmd)
                self.assertIn("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1", cmd)
                self.assertIn("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1", cmd)
                self.assertIn("trainer.total_epochs=120", cmd)
                self.assertEqual(launch.call_args.kwargs["env"]["TTRL_TASK"], task)


if __name__ == "__main__":
    unittest.main()
