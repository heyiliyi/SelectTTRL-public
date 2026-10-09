"""CPU-only regression checks; no model or training dependencies."""
import ast
import importlib.util
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from reproduction_helpers import answer_position, nonthinking_prompt, question_aggregates, ranking_components


class Tokens:
    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return messages[0]["content"] + "[assistant]"


class SupplementTests(unittest.TestCase):
    def test_nonthinking_template(self):
        tok = Tokens()
        self.assertEqual(nonthinking_prompt(tok, {"prompt": "Q"}), "Q[assistant]")
        self.assertIs(tok.kwargs["enable_thinking"], False)

    def test_boxed_position_and_incomplete_tail(self):
        text = "P" + r"a \boxed{1} b \boxed{\frac{2}{3}} c \boxed{"
        pos, reason = answer_position(Tokens(), list(map(ord, text)), 1, "boxed_or_response_last")
        self.assertEqual(pos, text.rfind("}"))
        self.assertEqual(reason, "last_complete_box")

    def test_response_fallback_excludes_eos(self):
        pos, reason = answer_position(Tokens(), list(map(ord, "Panswer~~")), 1, "boxed_or_response_last", eos_ids={ord("~")})
        self.assertEqual(pos, 6)
        self.assertEqual(reason, "response_last")

    def test_question_aggregation_not_hidden_averaging(self):
        rows = [{"question_id": "a"}, {"question_id": "a"}, {"question_id": "b"}]
        x, y, n = question_aggregates(rows, [.2, .8, .9], [0, 1, 1])
        self.assertEqual((x, y, n), ([.5, .9], [.5, 1.], [2, 1]))

    def test_missing_question_id_rejected(self):
        with self.assertRaises(ValueError):
            question_aggregates([{}], [.5], [1])

    def test_signed_ranking_components(self):
        rows = [{"question_id": str(i), "cos": float(i), "feature_sum": .48*i,
            "bias": -.05*i, "error": .57*i, "truncated": False} for i in range(10)]
        result = ranking_components(rows)
        for key, value in (("feature_sum", .48), ("bias", -.05), ("error", .57)):
            self.assertAlmostEqual(result["signed_delta_share"][key], value)
        self.assertLess(result["gap_closure_error"], 1e-12)

    def test_missing_truncation_is_not_silently_complete(self):
        self.assertEqual(ranking_components([{"cos": 1}])["status"], "unavailable")

    def test_capacity_check_uses_actual_dataloader(self):
        tree = ast.parse((ROOT / "environment/verl_overlay/batch_ttrl_trainer.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        check = compile(ast.Module(body=init.body[1:3], type_ignores=[]), "actual_capacity_check", "exec")
        obj = SimpleNamespace(train_dataloader=[1], config=SimpleNamespace(trainer=SimpleNamespace(total_epochs=30)), total_training_steps=120)
        with self.assertRaises(ValueError):
            exec(check, {"self": obj})
        obj.config.trainer.total_epochs = 120
        exec(check, {"self": obj})

    def test_v1_interval_retained(self):
        spec = importlib.util.spec_from_file_location("test_v1_cos", ROOT / "src/selectttrl/cos.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.assertEqual(module.select_percentile_interval(range(30)).selected_indices, tuple(range(7, 18)))

    def test_atomic_progress_marker(self):
        tree = ast.parse((ROOT / "environment/verl_overlay/ttrl_tracking.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TTRLTracking")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_write_progress")
        namespace = {"json": json}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "progress_writer", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_progress.json"
            tracker = SimpleNamespace(_ttrl_progress_path=path, _ttrl_total_training_steps=120)
            namespace["_write_progress"](tracker, 120)
            self.assertEqual(json.loads(path.read_text())["last_step"], 120)
            self.assertEqual(json.loads(path.read_text())["requested_steps"], 120)
            self.assertFalse(path.with_suffix(".tmp").exists())


class LauncherGuardTests(unittest.TestCase):
    def launch(self, directory, extra=(), marker_step=120):
        entry = ROOT / "experiments/rq1_selectttrl/run_ttrl.py"
        argv = [str(entry), "--model", "model", "--train-files", "selected.parquet",
                "--val-files", "full_target.parquet", "--dataset", "aime24",
                "--output-dir", str(directory), "--verl-root", "/path/to/verl", *extra]
        def completed(cmd, *, env, check):
            if marker_step is not None:
                Path(env["TTRL_PROGRESS_PATH"]).write_text(json.dumps(
                    {"last_step": marker_step, "requested_steps": 120}), encoding="utf-8")
        with patch.object(sys, "argv", argv), patch("subprocess.run", side_effect=completed) as launch, patch("builtins.print"):
            runpy.run_path(str(entry), run_name="__main__")
        return launch.call_args

    def test_epoch_capacity_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = self.launch(Path(directory) / "selected", ["--train-question-count", "11"])
            full = self.launch(Path(directory) / "full", ["--train-question-count", "30"])
            self.assertIn("trainer.total_epochs=120", selected.args[0])
            self.assertIn("trainer.total_epochs=40", full.args[0])
            self.assertIn("data.val_files=full_target.parquet", full.args[0])
            for key in ("TTRL_AUDIT_PATH", "TTRL_FULL_ROLLOUT_DIR", "TTRL_PROGRESS_PATH"):
                self.assertNotEqual(selected.kwargs["env"][key], full.kwargs["env"][key])

    def test_insufficient_epochs_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(SystemExit):
            self.launch(directory, ["--train-question-count", "11", "--total-epochs", "30"])

    def test_missing_marker_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "no completed-step marker"):
            self.launch(directory, marker_step=None)

    def test_short_run_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "does not match"):
            self.launch(directory, marker_step=30)

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.launch(directory)
            with self.assertRaises(SystemExit):
                self.launch(directory)

    def test_contract_override_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(SystemExit):
            self.launch(directory, ["--override", "trainer.total_epochs=1"])


if __name__ == "__main__":
    unittest.main()
