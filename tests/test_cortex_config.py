from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurosim_gpu.cortex.config import (
    BackboneConfig,
    BrainConfig,
    ConfigError,
    ExperimentConfig,
    TrainConfig,
    apply_overrides,
    load_config,
    save_config,
)


class ConfigTests(unittest.TestCase):
    def test_defaults_construct(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(config.name, "cortex")
        self.assertEqual(config.brain.n_edges, 2048 * 96)
        self.assertEqual(config.backbone.kind, "tiny")

    def test_json_roundtrip(self) -> None:
        config = ExperimentConfig(
            name="roundtrip",
            brain=BrainConfig(n_neurons=64, edges_per_neuron=4, regions=2),
            backbone=BackboneConfig(
                kind="tiny",
                encoder_layers=1,
                decoder_layers=1,
                tiny_num_layers=2,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_config(config, path)
            loaded = load_config(path)
        self.assertEqual(config, loaded)

    def test_unknown_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown BrainConfig"):
            ExperimentConfig.from_dict({"brain": {"n_neuron": 4}})

    def test_invalid_ranges_are_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "ticks_per_token"):
            BrainConfig(ticks_per_token=0)
        with self.assertRaisesRegex(ConfigError, "local_fraction"):
            BrainConfig(local_fraction=1.5)
        with self.assertRaisesRegex(ConfigError, "alpha range"):
            BrainConfig(alpha_min=0.99, alpha_max=0.9)
        with self.assertRaisesRegex(ConfigError, "amp"):
            TrainConfig(amp="float32")

    def test_lora_requires_hf(self) -> None:
        with self.assertRaisesRegex(ConfigError, "kind=hf"):
            BackboneConfig(kind="tiny", lora_rank=4)

    def test_tiny_split_must_cover_all_layers(self) -> None:
        with self.assertRaisesRegex(ConfigError, "must equal tiny_num_layers"):
            BackboneConfig(
                encoder_layers=1,
                decoder_layers=1,
                tiny_num_layers=4,
            )

    def test_overrides_parse_json_values(self) -> None:
        raw = ExperimentConfig().to_dict()
        updated = apply_overrides(
            raw,
            [
                "brain.ticks_per_token=4",
                "train.amp=none",
                "data.corpus=[\"a.txt\"]",
            ],
        )
        self.assertEqual(updated["brain"]["ticks_per_token"], 4)
        self.assertEqual(updated["train"]["amp"], "none")
        self.assertEqual(updated["data"]["corpus"], ["a.txt"])
        config = ExperimentConfig.from_dict(updated)
        self.assertEqual(config.brain.ticks_per_token, 4)

    def test_override_path_must_exist(self) -> None:
        with self.assertRaisesRegex(ConfigError, "does not exist"):
            apply_overrides(ExperimentConfig().to_dict(), ["brain.nope=1"])

    def test_dry_run_config_file_is_valid(self) -> None:
        path = Path(__file__).resolve().parents[1] / "llm" / "configs" / "cortex_tiny.json"
        config = load_config(path)
        self.assertEqual(config.backbone.kind, "tiny")
        self.assertFalse(config.data.corpus)

    def test_qwen_config_is_valid(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "llm"
            / "configs"
            / "cortex_qwen3_3p2_minimind.json"
        )
        config = load_config(path)
        self.assertEqual(config.backbone.kind, "hf")
        self.assertEqual(config.backbone.model_id, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.backbone.encoder_layers, 3)
        self.assertEqual(config.backbone.decoder_layers, 2)
        self.assertEqual(config.backbone.lora_rank, 0)
        self.assertEqual(config.brain.weight_mode, "dense")
        self.assertTrue(config.brain.init_from_backbone)
        self.assertEqual(config.brain.n_neurons, 1024)


if __name__ == "__main__":
    unittest.main()
