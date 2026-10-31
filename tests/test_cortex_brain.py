from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from neurosim_gpu.cortex.brain import CorticalCore
from neurosim_gpu.cortex.config import BrainConfig


class BrainTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = BrainConfig(
            n_neurons=96,
            edges_per_neuron=16,
            regions=3,
            ticks_per_token=3,
            detach_interval=0,
            seed=5,
        )
        self.core = CorticalCore(self.config, d_model=16)
        self.drive = torch.randn(2, 5, 16)

    def test_forward_shapes(self) -> None:
        output, state, stats = self.core(self.drive)
        self.assertEqual(output.shape, (2, 5, 16))
        self.assertEqual(state.shape, (2, 96))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(state).all())
        self.assertEqual(stats["activity"].ndim, 0)
        self.assertTrue(0.0 <= float(stats["activity"]) <= 1.0)

    def test_biological_memory_consolidates_trace(self) -> None:
        config = replace(
            self.config,
            weight_mode="dense",
            n_neurons=32,
            edges_per_neuron=0,
            eligibility_decay=0.9,
            consolidation_rate=0.01,
            short_term_tau=2.0,
            short_term_utilization=0.05,
            homeostatic_strength=0.1,
        )
        core = CorticalCore(config, d_model=8)
        core.train()
        drive = torch.randn(2, 4, 8)
        before = core.recurrent.detach().clone()
        core(drive)
        self.assertGreater(float(core.eligibility.abs().sum()), 0.0)
        stats = core.consolidate(1.0)
        self.assertIn("consolidated", stats)
        self.assertFalse(torch.equal(before, core.recurrent.detach()))

    def test_state_is_recurrent(self) -> None:
        full, full_state, _ = self.core(self.drive)
        first, state, _ = self.core(self.drive[:, :1])
        second, state, _ = self.core(self.drive[:, 1:2], state=state)
        torch.testing.assert_close(
            torch.cat([first, second], dim=1), full[:, :2], atol=1e-6, rtol=1e-6
        )
        self.assertEqual(state.shape, (2, 96))

    def test_gradients_reach_interface_and_gains(self) -> None:
        drive = self.drive.clone().requires_grad_(True)
        output, _, _ = self.core(drive)
        output.square().mean().backward()
        self.assertIsNotNone(drive.grad)
        self.assertGreater(float(drive.grad.abs().sum()), 0.0)
        self.assertIsNotNone(self.core.drive_proj.weight.grad)
        self.assertIsNotNone(self.core.readout_proj.weight.grad)
        self.assertIsNotNone(self.core.init_state.grad)
        self.assertIsNotNone(self.core.gain_delta.grad)
        self.assertIsNotNone(self.core.region_log_gain.grad)
        for parameter in (
            self.core.drive_proj.weight,
            self.core.readout_proj.weight,
            self.core.gain_delta,
        ):
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_detach_interval_stops_old_token_gradient(self) -> None:
        drive = self.drive.clone().requires_grad_(True)
        output, _, _ = self.core(drive, detach_interval=2)
        output[:, -1].square().mean().backward()
        self.assertEqual(float(drive.grad[:, 0].abs().sum()), 0.0)
        self.assertGreater(float(drive.grad[:, -1].abs().sum()), 0.0)

        drive_full = self.drive.clone().requires_grad_(True)
        output_full, _, _ = self.core(drive_full, detach_interval=0)
        output_full[:, -1].square().mean().backward()
        self.assertGreater(float(drive_full.grad[:, 0].abs().sum()), 0.0)

    def test_masked_token_has_no_output(self) -> None:
        mask = torch.ones(2, 5, dtype=torch.bool)
        mask[:, 2] = False
        output, _, _ = self.core(self.drive, attention_mask=mask)
        self.assertEqual(float(output[:, 2].detach().abs().sum()), 0.0)
        self.assertGreater(float(output[:, 3].detach().abs().sum()), 0.0)

    def test_bad_shapes_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "drive must have shape"):
            self.core(torch.randn(5, 16))
        with self.assertRaisesRegex(ValueError, "state must have shape"):
            self.core(self.drive, state=torch.zeros(1, 96))

    def test_dense_mode_trains_recurrent_matrix(self) -> None:
        config = BrainConfig(
            n_neurons=32,
            edges_per_neuron=0,
            weight_mode="dense",
            ticks_per_token=1,
            detach_interval=4,
            regions=4,
            seed=3,
        )
        initial = torch.randn(32, 32) * 0.05
        core = CorticalCore(config, d_model=32, init_weight=initial)
        torch.testing.assert_close(core.recurrent, initial)
        drive = torch.randn(2, 5, 32, requires_grad=True)
        output, state, _ = core(drive)
        self.assertEqual(output.shape, (2, 5, 32))
        self.assertEqual(state.shape, (2, 32))
        output.square().mean().backward()
        self.assertIsNotNone(core.recurrent.grad)
        self.assertGreater(float(core.recurrent.grad.abs().sum()), 0.0)
        self.assertGreater(float(drive.grad.abs().sum()), 0.0)

    def test_dense_initial_weight_shape_is_checked(self) -> None:
        config = BrainConfig(
            n_neurons=16,
            edges_per_neuron=0,
            weight_mode="dense",
            regions=2,
        )
        with self.assertRaisesRegex(ValueError, "init_weight must have shape"):
            CorticalCore(config, d_model=16, init_weight=torch.zeros(8, 8))

    def test_dense_random_init_has_target_radius(self) -> None:
        from neurosim_gpu.cortex.transplant import spectral_radius

        config = BrainConfig(
            n_neurons=64,
            edges_per_neuron=0,
            weight_mode="dense",
            init_radius=0.9,
            regions=4,
            seed=9,
        )
        core = CorticalCore(config, d_model=64)
        radius = spectral_radius(core.recurrent.detach(), iterations=30)
        self.assertAlmostEqual(radius, 0.9, delta=0.03)

    def test_readout_last(self) -> None:
        config = BrainConfig(
            n_neurons=64,
            edges_per_neuron=8,
            regions=2,
            ticks_per_token=2,
            readout="last",
            seed=1,
        )
        core = CorticalCore(config, d_model=8)
        output, _, _ = core(torch.randn(1, 3, 8))
        self.assertEqual(output.shape, (1, 3, 8))


if __name__ == "__main__":
    unittest.main()
