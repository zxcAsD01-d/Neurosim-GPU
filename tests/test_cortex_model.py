from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from neurosim_gpu.cortex.checkpoint import load_checkpoint, save_checkpoint
from neurosim_gpu.cortex.config import (
    BackboneConfig,
    BrainConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from neurosim_gpu.cortex.model import CortexSandwichLM, language_model_loss
from neurosim_gpu.cortex.tokenization import ByteTokenizer


def make_config(
    *,
    use_brain: bool = True,
    freeze_backbone: bool = False,
) -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
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
            freeze_encoder=freeze_backbone,
            freeze_decoder=freeze_backbone,
        ),
        brain=BrainConfig(
            n_neurons=64,
            edges_per_neuron=8,
            regions=2,
            ticks_per_token=2,
            detach_interval=4,
            seed=2,
        ),
        model=ModelConfig(use_brain=use_brain),
        data=DataConfig(tokenizer="byte", block_size=8),
        train=TrainConfig(max_steps=1, device="cpu", amp="none"),
    )


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.tokenizer = ByteTokenizer()
        self.config = make_config()
        self.model = CortexSandwichLM(
            self.config, self.tokenizer.vocab_size
        )

    def test_forward_backward(self) -> None:
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (2, 8))
        output = self.model(input_ids, labels=input_ids)
        self.assertEqual(output.logits.shape, (2, 8, 257))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(self.model.brain.drive_proj.weight.grad)
        self.assertIsNotNone(self.model.brain.readout_proj.weight.grad)
        self.assertIsNotNone(self.model.brain_gate.grad)
        self.assertGreater(
            float(self.model.brain.drive_proj.weight.grad.abs().sum()), 0.0
        )

    def test_encoder_decoder_freeze(self) -> None:
        config = make_config(freeze_backbone=True)
        model = CortexSandwichLM(config, self.tokenizer.vocab_size)
        description = model.describe()
        self.assertGreater(int(description["groups"]["frozen"]), 0)
        self.assertGreater(int(description["groups"]["brain"]), 0)
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(
            all(name.startswith("brain") for name in trainable), trainable[:5]
        )

    def test_no_brain_path(self) -> None:
        config = make_config(use_brain=False)
        model = CortexSandwichLM(config, self.tokenizer.vocab_size)
        self.assertIsNone(model.brain)
        self.assertFalse(model.brain_gate.requires_grad)
        ids = torch.randint(0, 257, (2, 8))
        output = model(ids, labels=ids)
        self.assertEqual(output.logits.shape, (2, 8, 257))
        self.assertIsNotNone(output.loss)

    def test_padding_mask(self) -> None:
        ids = torch.randint(0, 257, (2, 10))
        mask = torch.ones(2, 10, dtype=torch.long)
        mask[1, 6:] = 0
        labels = ids.clone()
        labels[mask == 0] = -100
        output = self.model(ids, attention_mask=mask, labels=labels)
        self.assertTrue(torch.isfinite(output.loss))

    def test_loss_z_loss(self) -> None:
        logits = torch.randn(2, 4, 10)
        labels = torch.randint(0, 10, (2, 4))
        plain = language_model_loss(logits, labels)
        regularised = language_model_loss(logits, labels, z_loss=0.01)
        self.assertGreater(float(regularised), float(plain))

    def test_start_end_token_weights_change_loss(self) -> None:
        logits = torch.randn(2, 6, 10)
        labels = torch.tensor([[0, 9, 1, 2, 9, 3], [4, 5, 9, 6, 7, 9]])
        plain = language_model_loss(logits, labels)
        weighted = language_model_loss(
            logits,
            labels,
            eos_token_id=9,
            bos_token_id=0,
            eos_weight=4.0,
            bos_weight=2.0,
        )
        self.assertNotAlmostEqual(float(plain), float(weighted), places=5)
        self.assertTrue(torch.isfinite(weighted))

    def test_describe(self) -> None:
        description = self.model.describe()
        self.assertEqual(description["hidden_size"], 32)
        self.assertEqual(description["vocab_size"], 257)
        self.assertGreater(int(description["trainable_parameters"]), 0)
        self.assertEqual(description["brain"]["n_neurons"], 64)

    def test_generation(self) -> None:
        prompt = torch.tensor([[10, 11, 12]])
        generated = self.model.generate(
            prompt, max_new_tokens=3, temperature=0.0
        )
        self.assertEqual(generated.shape, (1, 6))
        self.assertTrue(bool((generated >= 0).all()))
        self.assertTrue(bool((generated < 257).all()))

    def test_open_loop_stream_generation(self) -> None:
        prompt = torch.tensor([[10, 11, 12]])
        generated = self.model.stream_generate(
            prompt,
            max_new_tokens=4,
            temperature=0.0,
            top_p=0.95,
        )
        self.assertEqual(generated.shape, (1, 7))
        self.assertTrue(bool((generated >= 0).all()))
        self.assertTrue(bool((generated < 257).all()))
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            self.model.stream_generate(torch.zeros((2, 3), dtype=torch.long))

    def test_checkpoint_roundtrip(self) -> None:
        state = self.model.trainable_state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path, self.model, None, None, 3, self.config
            )
            restored = CortexSandwichLM(
                self.config, self.tokenizer.vocab_size
            )
            payload = load_checkpoint(path, restored)
            self.assertEqual(payload["step"], 3)
            for name, tensor in state.items():
                torch.testing.assert_close(
                    restored.trainable_state_dict()[name], tensor
                )

    def test_empty_sequence_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one token"):
            self.model.brain(torch.empty((2, 0, 32)))


if __name__ == "__main__":
    unittest.main()
