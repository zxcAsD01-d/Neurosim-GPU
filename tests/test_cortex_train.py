from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from neurosim_gpu.cortex.config import (
    BackboneConfig,
    BrainConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from neurosim_gpu.cortex.train import (
    build_optimizer,
    build_scheduler,
    dry_run,
)


def tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="train-test",
        backbone=BackboneConfig(
            kind="tiny",
            encoder_layers=1,
            decoder_layers=1,
            tiny_num_layers=2,
            tiny_hidden_size=32,
            tiny_intermediate_size=64,
            tiny_num_heads=4,
            tiny_num_kv_heads=4,
            tiny_max_position_embeddings=64,
            freeze_encoder=False,
            freeze_decoder=False,
        ),
        brain=BrainConfig(
            n_neurons=64,
            edges_per_neuron=8,
            regions=2,
            ticks_per_token=2,
            detach_interval=4,
            seed=1,
        ),
        model=ModelConfig(),
        data=DataConfig(tokenizer="byte", block_size=8),
        train=TrainConfig(
            max_steps=20,
            warmup_steps=2,
            min_lr_ratio=0.1,
            device="cpu",
            amp="none",
        ),
    )


class TrainWiringTests(unittest.TestCase):
    def test_optimizer_groups_and_lrs(self) -> None:
        from neurosim_gpu.cortex.model import CortexSandwichLM
        from neurosim_gpu.cortex.tokenization import ByteTokenizer

        config = tiny_config()
        model = CortexSandwichLM(config, ByteTokenizer().vocab_size)
        optimizer = build_optimizer(model, config)
        names = [group.get("name") for group in optimizer.param_groups]
        self.assertIn("brain", names)
        self.assertIn("other", names)
        brain_group = next(
            group for group in optimizer.param_groups if group.get("name") == "brain"
        )
        self.assertEqual(brain_group["lr"], config.train.brain_lr)
        self.assertEqual(brain_group["weight_decay"], 0.0)

        scheduler = build_scheduler(optimizer, config)
        initial = scheduler.get_last_lr()
        self.assertAlmostEqual(
            initial[0], config.train.brain_lr * 0.5, places=8
        )
        for _ in range(config.train.max_steps):
            optimizer.step()
            scheduler.step()
        final = scheduler.get_last_lr()
        self.assertAlmostEqual(
            final[0], config.train.brain_lr * config.train.min_lr_ratio, places=8
        )

    def test_dry_run_reports_groups(self) -> None:
        import argparse

        config = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            args = argparse.Namespace(report=str(report_path))
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = dry_run(config, args)
            self.assertEqual(code, 0)
            self.assertTrue(report_path.exists())
            text = report_path.read_text(encoding="utf-8")
        self.assertIn('"brain"', text)
        self.assertIn('"grad_norm"', text)
        self.assertIn('"optimizer"', text)


if __name__ == "__main__":
    unittest.main()
