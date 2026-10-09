#!/usr/bin/env python3
"""Compose stock VERL config and run a process-local batch-TTRL trainer."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import ray
from hydra import compose, initialize_config_dir

from batch_ttrl_trainer import BatchTTRLRayPPOTrainer
from ttrl_tracking import install_tracking_overlay
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device


class BatchTTRLTaskRunner(TaskRunner):
    def run(self, config):
        # TaskRunner.run and RayPPOTrainer.fit resolve these module globals at
        # runtime.  Both replacements are local to this Ray TaskRunner process.
        import verl.trainer.main_ppo as main_ppo_module

        main_ppo_module.RayPPOTrainer = BatchTTRLRayPPOTrainer
        install_tracking_overlay()
        return super().run(config)


def main() -> None:
    config_value = os.environ.get("VERL_CONFIG_DIR")
    if config_value:
        config_dir = Path(config_value)
    else:
        verl_value = os.environ.get("VERL_ROOT")
        if not verl_value:
            raise RuntimeError("Set VERL_ROOT to the installed stock VERL checkout")
        config_dir = Path(verl_value) / "verl" / "trainer" / "config"
    overrides = list(os.sys.argv[1:])
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    runner_class = ray.remote(num_cpus=1)(BatchTTRLTaskRunner)
    run_ppo(config, task_runner_class=runner_class)


if __name__ == "__main__":
    main()
