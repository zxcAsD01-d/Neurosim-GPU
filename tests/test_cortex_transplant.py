from __future__ import annotations

import unittest

import torch

from neurosim_gpu.cortex.transplant import (
    linearized_block_matrix,
    spectral_radius,
    transplant_recurrence,
)

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    HAVE_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAVE_TRANSFORMERS = False


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TransplantTests(unittest.TestCase):
    def setUp(self) -> None:
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
        )
        self.model = Qwen3ForCausalLM(config)
        self.layers = list(self.model.model.layers)

    def test_block_matrix_handles_gqa(self) -> None:
        block = linearized_block_matrix(self.layers[0])
        self.assertEqual(tuple(block.shape), (32, 32))
        self.assertTrue(torch.isfinite(block).all())

    def test_composition_and_radius(self) -> None:
        recurrence = transplant_recurrence(self.layers, target_radius=0.9)
        self.assertEqual(tuple(recurrence.shape), (32, 32))
        self.assertTrue(torch.isfinite(recurrence).all())
        self.assertAlmostEqual(
            spectral_radius(recurrence, iterations=40), 0.9, delta=0.03
        )

    def test_empty_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no middle layers"):
            transplant_recurrence([])


if __name__ == "__main__":
    unittest.main()
