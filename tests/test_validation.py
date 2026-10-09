"""CPU-only regression tests for the training/evaluation boundary."""
import ast
import __future__
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "verl_overlay"))
from evaluation_metrics import accuracy_metrics


def make_harness(base=object):
    tree = ast.parse((ROOT / "environment/verl_overlay/batch_ttrl_trainer.py").read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
    names = {"_generate_vote_then_select", "_extract_batch_ttrl_reward", "_extract_validation_reward", "_validate"}
    cls.body = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in names]
    ns = {"RayPPOTrainer": base, "Path": Path, "time": __import__("time"),
          "torch": SimpleNamespace(zeros_like=np.zeros_like, float32=np.float32),
          "accuracy_metrics": accuracy_metrics}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "trainer_methods", "exec",
                 flags=__future__.annotations.compiler_flag), ns)
    return ns[cls.name]()


class ValidationTests(unittest.TestCase):
    def test_validation_bypasses_vote_expansion(self):
        trainer = make_harness()
        batch = SimpleNamespace(meta_info={"validate": True})
        self.assertIs(trainer._generate_vote_then_select(batch, lambda x: x), batch)

    def test_training_keeps_pseudo_reward_path(self):
        trainer = make_harness()
        trainer.ttrl_vote_n, trainer.ttrl_train_n = 32, 16
        trainer._extract_precomputed_first_k_reward = lambda _: "pseudo"
        batch = SimpleNamespace(meta_info={})
        self.assertEqual(trainer._extract_batch_ttrl_reward(batch), "pseudo")

    def test_validation_uses_gold_not_consensus(self):
        trainer = make_harness()
        trainer._decode_responses = lambda _: (["B", "B"], [1, 1])
        trainer.ttrl_scorer = lambda text, gold: float(text == gold)
        trainer.ttrl_extractor = lambda text: text
        batch = SimpleNamespace(meta_info={"validate": True}, batch={"responses": np.zeros((2, 1))},
                                non_tensor_batch={"reward_model": [{"ground_truth": "A"}] * 2})
        scores, info = trainer._extract_batch_ttrl_reward(batch)
        self.assertEqual(info["acc"], [0.0, 0.0])
        self.assertEqual(scores.sum(), 0.0)
        batch.non_tensor_batch["reward_model"] = [{}, {}]
        with self.assertRaises(ValueError):
            trainer._extract_batch_ttrl_reward(batch)

    def test_exact_sampled_metrics(self):
        result = {"data_sources": [["math"] * 32], "sample_uids": ["q1"] * 16 + ["q2"] * 16,
                  "reward_extra_infos_dict": {"acc": [0] * 9 + [1] * 23,
                                               "pred": ["B"] * 9 + ["A"] * 23}}
        metrics = accuracy_metrics(result, mode="sampled")
        self.assertEqual(metrics["val/math/Mean1"], 23 / 32)
        self.assertEqual(metrics["val/math/Maj@16"], .5)

    def test_two_passes_restore_configuration(self):
        calls = []
        class Base:
            def _validate(self, merged=False):
                s = self.config.actor_rollout_ref.rollout.val_kwargs
                calls.append((s.n, s.temperature, s.do_sample))
                return {"data_sources": [["math"] * s.n], "sample_uids": ["q"] * s.n,
                        "reward_extra_infos_dict": {"acc": [1] * s.n, "pred": ["A"] * s.n}}
        trainer = make_harness(Base)
        sampling = SimpleNamespace(n=3, temperature=.2, do_sample=False)
        trainer.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(val_kwargs=sampling)), trainer={})
        metrics = trainer._validate()
        self.assertEqual(calls, [(1, 0., False), (16, .6, True)])
        self.assertEqual((sampling.n, sampling.temperature, sampling.do_sample), (3, .2, False))
        self.assertEqual(set(metrics), {"val/math/Greedy", "val/math/Mean1", "val/math/Maj@16"})


if __name__ == "__main__":
    unittest.main()
