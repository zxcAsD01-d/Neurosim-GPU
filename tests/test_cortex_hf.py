from __future__ import annotations

import unittest

import torch

from neurosim_gpu.cortex.backbone import (
    HFBackbone,
    LoRALinear,
    build_backbone,
    inject_lora,
)
from neurosim_gpu.cortex.config import (
    BackboneConfig,
    BrainConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from neurosim_gpu.cortex.model import CortexSandwichLM

try:  # optional dependency
    from transformers import Qwen3Config, Qwen3ForCausalLM

    HAVE_TRANSFORMERS = True
except Exception:  # pragma: no cover - CI without the lm extra
    HAVE_TRANSFORMERS = False


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class HFBackboneTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
        )
        self.hf_model = Qwen3ForCausalLM(config)
        self.hf_model.eval()
        self.config = BackboneConfig(
            kind="hf",
            encoder_layers=1,
            decoder_layers=1,
            freeze_encoder=True,
            freeze_decoder=True,
            lora_rank=4,
            lora_alpha=8.0,
            lora_targets=("q_proj", "v_proj"),
        )
        self.backbone = HFBackbone(self.hf_model, self.config)

    def logits(self, model: HFBackbone, ids: torch.Tensor) -> torch.Tensor:
        hidden, context = model.encode(ids)
        return model.logits(model.decode(hidden, context))

    def test_split_matches_full_model(self) -> None:
        ids = torch.randint(0, 128, (2, 8))
        with torch.no_grad():
            expected = self.hf_model(input_ids=ids).logits
            actual = self.logits(self.backbone, ids)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_causality_survives_split(self) -> None:
        ids = torch.randint(0, 128, (1, 8))
        changed = ids.clone()
        changed[0, -1] = (changed[0, -1] + 1) % 128
        with torch.no_grad():
            first = self.logits(self.backbone, ids)
            second = self.logits(self.backbone, changed)
        torch.testing.assert_close(
            first[0, :-1], second[0, :-1], atol=1e-6, rtol=1e-5
        )
        self.assertGreater(
            float((first[0, -1] - second[0, -1]).abs().max()), 0.0
        )

    def test_lora_zero_update_then_gradients(self) -> None:
        ids = torch.randint(0, 128, (2, 6))
        with torch.no_grad():
            before = self.logits(self.backbone, ids).clone()

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        replaced = inject_lora(
            self.backbone,
            self.config.lora_targets,
            self.config.lora_rank,
            self.config.lora_alpha,
            0.0,
        )
        self.assertEqual(replaced, 4)
        self.assertIsInstance(
            self.backbone.model.model.layers[0].self_attn.q_proj,
            LoRALinear,
        )

        with torch.no_grad():
            after = self.logits(self.backbone, ids)
        torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-5)

        output = self.logits(self.backbone, ids)
        output.float().logsumexp(dim=-1).mean().backward()
        trainable = [
            parameter
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        ]
        self.assertEqual(len(trainable), 8)
        self.assertTrue(all(p.grad is not None for p in trainable))
        self.assertIsNone(
            self.backbone.model.model.layers[0].self_attn.q_proj.base.weight.grad
        )

        lora_b = self.backbone.model.model.layers[0].self_attn.q_proj.lora_b
        with torch.no_grad():
            lora_b.weight.add_(0.1)
        changed = self.logits(self.backbone, ids)
        self.assertGreater(float((changed - before).detach().abs().max()), 1e-4)

    def test_build_backbone_lora_only_decoder(self) -> None:
        backbone = HFBackbone(self.hf_model, self.config)
        built = build_backbone(self.config, 128, backbone=backbone)
        encoder_q = built.encoder_layers[0].self_attn.q_proj
        decoder_q = built.decoder_layers[0].self_attn.q_proj
        self.assertNotIsInstance(encoder_q, LoRALinear)
        self.assertIsInstance(decoder_q, LoRALinear)
        self.assertFalse(
            any(parameter.requires_grad for parameter in encoder_q.parameters())
        )
        trainable = [
            name
            for name, parameter in built.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all("layers.1." in name for name in trainable))

    def test_middle_layers_are_dropped(self) -> None:
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
        )
        model = Qwen3ForCausalLM(config)
        backbone_config = BackboneConfig(
            kind="hf",
            encoder_layers=1,
            decoder_layers=1,
            freeze_encoder=True,
            freeze_decoder=True,
        )
        backbone = HFBackbone(model, backbone_config)
        self.assertEqual(backbone.encoder_count, 1)
        self.assertEqual(backbone.decoder_count, 1)
        self.assertEqual(backbone.middle_count, 2)
        # The retained blocks are the first and the last, not the first two.
        self.assertEqual(len(model.model.layers), 2)
        self.assertIs(model.model.layers[-1], backbone.decoder_layers[0])
        ids = torch.randint(0, 128, (2, 6))
        hidden, context = backbone.encode(ids)
        logits = backbone.logits(backbone.decode(hidden, context))
        self.assertEqual(logits.shape, (2, 6, 128))

    def test_model_transplants_middle_layers_into_dense_brain(self) -> None:
        model_config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
        )
        hf_model = Qwen3ForCausalLM(model_config)
        backbone_config = BackboneConfig(
            kind="hf", encoder_layers=1, decoder_layers=1
        )
        backbone = HFBackbone(hf_model, backbone_config)
        config = ExperimentConfig(
            name="transplant-test",
            backbone=backbone_config,
            brain=BrainConfig(
                n_neurons=64,
                edges_per_neuron=0,
                weight_mode="dense",
                init_from_backbone=True,
                init_radius=0.9,
                ticks_per_token=1,
                detach_interval=4,
                seed=1,
            ),
            model=ModelConfig(brain_gate_init=1.0),
            data=DataConfig(tokenizer="byte", block_size=6),
            train=TrainConfig(max_steps=1, device="cpu", amp="none"),
        )
        model = CortexSandwichLM(config, 128, backbone=backbone)
        self.assertTrue(torch.isfinite(model.brain.recurrent).all())
        self.assertFalse(backbone.middle_layers)
        ids = torch.randint(0, 128, (2, 6))
        output = model(ids, labels=ids)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.brain.recurrent.grad)
        self.assertGreater(
            float(model.brain.recurrent.grad.abs().sum()), 0.0
        )

    def test_bad_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be in|exceeds"):
            HFBackbone(
                self.hf_model,
                BackboneConfig(
                    kind="hf", encoder_layers=2, decoder_layers=2
                ),
            )

    def test_model_with_hf_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        inject_lora(
            self.backbone,
            self.config.lora_targets,
            self.config.lora_rank,
            self.config.lora_alpha,
            0.0,
        )
        config = ExperimentConfig(
            name="hf-test",
            backbone=self.config,
            brain=BrainConfig(
                n_neurons=64,
                edges_per_neuron=8,
                regions=2,
                ticks_per_token=2,
                detach_interval=4,
                seed=1,
            ),
            model=ModelConfig(),
            data=DataConfig(tokenizer="byte", block_size=6),
            train=TrainConfig(max_steps=1, device="cpu", amp="none"),
        )
        model = CortexSandwichLM(config, 128, backbone=self.backbone)
        ids = torch.randint(0, 128, (2, 6))
        output = model(ids, labels=ids)
        self.assertEqual(output.logits.shape, (2, 6, 128))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        lora = [
            parameter
            for name, parameter in model.named_parameters()
            if ".lora_" in name
        ]
        self.assertTrue(lora)
        self.assertTrue(all(p.grad is not None for p in lora))


if __name__ == "__main__":
    unittest.main()
