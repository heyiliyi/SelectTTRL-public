"""CPU regression checks for release documentation and boundary fixes."""
import ast
import __future__
import json
import runpy
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class FeatureSelectionTests(unittest.TestCase):
    def run_selection(self, rows):
        def read_jsonl(_):
            return rows
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "features.json"
            argv = ["select_features.py", "--decomposition", "unused.jsonl",
                    "--output", str(output), "--top-features", "1", "--select-k", "1"]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules,
                    {"common": SimpleNamespace(read_jsonl=read_jsonl)}), patch("builtins.print"):
                runpy.run_path(str(ROOT / "experiments/rq3_sae/select_features.py"), run_name="__main__")
            return json.loads(output.read_text())

    def rows(self):
        # Feature 1 has one activation of 10 and nine zero-valued TopK entries.
        # Its nonzero mean must be 10, not 1; feature 2 has mean 2.
        return [{"question_id": str(i), "cos": i / 10, "truncated": False,
                 "active_ids": [1, 2, 3], "active_values": [10 if i == 9 else 0, 2, 0],
                 "active_contributions": [1 if i == 9 else 0, .1, 0]} for i in range(10)]

    def test_conditional_mean_excludes_zero_entries(self):
        result = self.run_selection(self.rows())
        self.assertEqual(result["selected_features"], [1])
        self.assertEqual(result["ranking"][0]["mean_activation"], 10)
        self.assertEqual(result["ranking"][0]["active_questions"], 1)

    def test_truncated_rows_do_not_affect_candidates(self):
        rows = self.rows() + [{"question_id": "truncated", "cos": 1e3,
            "truncated": True, "active_ids": [9], "active_values": [1e6],
            "active_contributions": [1e3]}]
        result = self.run_selection(rows)
        self.assertEqual(result["complete_rows"], 10)
        self.assertEqual(result["selected_features"], [1])

    def test_missing_truncation_is_rejected(self):
        rows = self.rows()
        del rows[0]["truncated"]
        with self.assertRaisesRegex(ValueError, "truncated metadata"):
            self.run_selection(rows)


class AdapterTests(unittest.TestCase):
    def invoke(self, response_mask=None, attention_mask=None):
        path = ROOT / "environment/verl_overlay/group_batch_adapter.py"
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "filter_before_old_ref")
        plan = SimpleNamespace(selected_positions=(0, 1), groups=[
            SimpleNamespace(uid="q", selected_indices=(0, 1), rewards=(1., 0.))])
        namespace = {"defaultdict": defaultdict,
            "OnlineGroupBatch": lambda *args: args,
            "plan_online_group_batch": lambda *args, **kwargs: plan,
            "AppliedGroupBatch": lambda batch, plan, rewards: SimpleNamespace(batch=batch, kept_rewards=rewards)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)

        class Batch:
            def __init__(self):
                self.batch = {"responses": np.zeros((2, 4), dtype=int)}
                if response_mask is not None:
                    self.batch["response_mask"] = np.array(response_mask)
                if attention_mask is not None:
                    self.batch["attention_mask"] = np.array(attention_mask)
            def __len__(self):
                return 2
            def select_idxs(self, indices):
                assert indices == [0, 1]
                return self

        with patch.dict(sys.modules, {"torch": SimpleNamespace(zeros_like=np.zeros_like, float32=np.float32)}):
            return namespace[fn.name](Batch(), uids=["q", "q"], response_texts=["A", "B"],
                vote_rollouts=2, train_rollouts=2, step=1, extractor=lambda x: x)

    def test_reward_is_not_written_on_padding(self):
        result = self.invoke(response_mask=[[1, 1, 0, 0], [1, 1, 1, 0]])
        np.testing.assert_array_equal(result.batch.batch["rm_scores"], [[0, 1, 0, 0], [0, 0, 0, 0]])

    def test_attention_mask_fallback(self):
        result = self.invoke(attention_mask=[[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]])
        self.assertEqual(result.batch.batch["rm_scores"][0, 1], 1.)

    def test_empty_response_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty response"):
            self.invoke(response_mask=[[0, 0, 0, 0], [1, 1, 0, 0]])

    def test_missing_mask_rejected(self):
        with self.assertRaisesRegex(ValueError, "mask required"):
            self.invoke()


class ReleaseLayoutTests(unittest.TestCase):
    def test_v2_is_not_part_of_release(self):
        self.assertIn("v2/", (ROOT / ".gitignore").read_text().splitlines())


if __name__ == "__main__":
    unittest.main()
