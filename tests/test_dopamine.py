from __future__ import annotations

import unittest

import torch

from neurosim_gpu.dopamine import (
    DopamineReceptors,
    TemporalDifferenceRPE,
    dopamine_gated_weight_update,
    dopamine_gated_batch_weight_update,
    modulate_neuron_dynamics,
    update_eligibility_trace,
)


class DopamineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

    def tensor(self, values: list[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=self.dtype, device=self.device)

    def test_burst_and_dip_reverse_d1_d2_pathway_signs(self) -> None:
        receptors = DopamineReceptors(
            d1_density=self.tensor([1.0, 0.0]),
            d2_density=self.tensor([0.0, 1.0]),
            tonic_concentration=1.0,
            d1_half_saturation=1.0,
            d2_half_saturation=1.0,
        )
        burst = receptors.response(2.0)
        dip = receptors.response(0.5)

        self.assertGreater(float(burst.d1_pathway[0]), 0.0)
        self.assertLess(float(burst.d2_pathway[1]), 0.0)
        self.assertLess(float(dip.d1_pathway[0]), 0.0)
        self.assertGreater(float(dip.d2_pathway[1]), 0.0)

    def test_tonic_is_exact_zero_and_concentration_is_nonnegative(self) -> None:
        receptors = DopamineReceptors(
            d1_density=self.tensor([0.2, 0.8, 1.0]),
            d2_density=self.tensor([1.0, 0.4, 0.1]),
            tonic_concentration=self.tensor([0.5, 1.0, 2.0]),
        )
        tonic = receptors.response(self.tensor([0.5, 1.0, 2.0]))
        torch.testing.assert_close(
            tonic.net_phasic,
            torch.zeros_like(tonic.net_phasic),
            rtol=0.0,
            atol=0.0,
        )

        clamped = receptors.response(self.tensor([-2.0, -1.0, 0.0]))
        self.assertTrue(bool(torch.all(clamped.concentration >= 0)))
        self.assertTrue(bool(torch.all(clamped.d1_occupancy >= 0)))
        self.assertTrue(bool(torch.all(clamped.d2_occupancy >= 0)))

    def test_receptor_occupancy_saturates(self) -> None:
        receptors = DopamineReceptors(
            d1_density=self.tensor([1.0, 1.0]),
            d2_density=self.tensor([1.0, 1.0]),
            tonic_concentration=0.1,
            d1_half_saturation=self.tensor([0.5, 2.0]),
            d2_half_saturation=self.tensor([0.25, 1.0]),
            d1_hill_coefficient=2.0,
            d2_hill_coefficient=2.0,
        )
        low = receptors.response(1.0)
        saturated = receptors.response(1e12)
        self.assertTrue(bool(torch.all(saturated.d1_occupancy > low.d1_occupancy)))
        self.assertTrue(bool(torch.all(saturated.d2_occupancy > low.d2_occupancy)))
        torch.testing.assert_close(
            saturated.d1_occupancy,
            torch.ones_like(saturated.d1_occupancy),
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertTrue(bool(torch.all(saturated.d2_occupancy <= 1.0)))

    def test_mixed_receptors_modulate_drive_and_gain(self) -> None:
        receptors = DopamineReceptors(
            d1_density=self.tensor([1.0, 0.25]),
            d2_density=self.tensor([0.25, 1.0]),
            tonic_concentration=1.0,
            d1_half_saturation=1.0,
            d2_half_saturation=1.0,
        )
        signals = receptors.response(2.0)
        modulation = modulate_neuron_dynamics(
            base_drive=self.tensor([0.1, 0.1]),
            base_gain=self.tensor([1.0, 1.0]),
            dopamine=signals,
            drive_scale=2.0,
            gain_scale=1.0,
        )
        self.assertGreater(float(signals.net_phasic[0]), 0.0)
        self.assertLess(float(signals.net_phasic[1]), 0.0)
        self.assertGreater(float(modulation.drive[0]), 0.1)
        self.assertLess(float(modulation.drive[1]), 0.1)
        self.assertGreater(float(modulation.gain[0]), 1.0)
        self.assertLess(float(modulation.gain[1]), 1.0)
        self.assertTrue(bool(torch.all(modulation.gain >= 0)))

    def test_eligibility_decay_without_new_coactivity(self) -> None:
        src = torch.tensor([0, 0, 1], dtype=torch.int32, device=self.device)
        dst = torch.tensor([1, 2, 2], dtype=torch.int32, device=self.device)
        trace = self.tensor([2.0, -4.0, 6.0])
        zeros = torch.zeros((2, 3), dtype=self.dtype, device=self.device)
        updated = update_eligibility_trace(
            trace,
            zeros,
            zeros,
            src,
            dst,
            decay=0.5,
        )
        torch.testing.assert_close(updated, self.tensor([1.0, -2.0, 3.0]))

    def test_eligibility_uses_mean_batch_aggregation(self) -> None:
        src = torch.tensor([0, 0, 1], dtype=torch.int64, device=self.device)
        dst = torch.tensor([1, 2, 2], dtype=torch.int64, device=self.device)
        pre = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=self.dtype,
            device=self.device,
        )
        post = torch.tensor(
            [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]],
            dtype=self.dtype,
            device=self.device,
        )
        actual = update_eligibility_trace(
            torch.zeros(3, dtype=self.dtype, device=self.device),
            pre,
            post,
            src,
            dst,
            decay=0.9,
            batch_reduction="mean",
        )
        expected = (pre[:, src] * post[:, dst]).mean(dim=0)
        torch.testing.assert_close(actual, expected)

    def test_pathway_signs_gate_local_weight_updates(self) -> None:
        receptors = DopamineReceptors(
            d1_density=self.tensor([1.0, 0.0]),
            d2_density=self.tensor([0.0, 1.0]),
            tonic_concentration=1.0,
            d1_half_saturation=1.0,
            d2_half_saturation=1.0,
        )
        dst = torch.tensor([0, 1], dtype=torch.int64, device=self.device)
        weight = self.tensor([0.2, 0.2])
        eligibility = self.tensor([1.0, 1.0])

        burst = dopamine_gated_weight_update(
            weight,
            eligibility,
            receptors.response(2.0),
            dst=dst,
            learning_rate=0.1,
        )
        dip = dopamine_gated_weight_update(
            weight,
            eligibility,
            receptors.response(0.5),
            dst=dst,
            learning_rate=0.1,
        )
        tonic = dopamine_gated_weight_update(
            weight,
            eligibility,
            receptors.response(1.0),
            dst=dst,
            learning_rate=0.1,
        )
        self.assertGreater(float(burst.delta[0]), 0.0)
        self.assertLess(float(burst.delta[1]), 0.0)
        self.assertLess(float(dip.delta[0]), 0.0)
        self.assertGreater(float(dip.delta[1]), 0.0)
        torch.testing.assert_close(tonic.delta, torch.zeros_like(tonic.delta))

    def test_weight_decay_bounds_and_sign_preservation(self) -> None:
        weight = self.tensor([0.9, -0.9, 0.0])
        bounded = dopamine_gated_weight_update(
            weight,
            self.tensor([10.0, -10.0, 10.0]),
            self.tensor([1.0, 1.0, 1.0]),
            learning_rate=1.0,
            minimum_weight=-1.0,
            maximum_weight=1.0,
            preserve_sign=True,
        )
        torch.testing.assert_close(bounded.weight, self.tensor([1.0, -1.0, 0.0]))

        crossing = dopamine_gated_weight_update(
            self.tensor([0.2, -0.2]),
            self.tensor([-1.0, 1.0]),
            self.tensor([1.0, 1.0]),
            preserve_sign=True,
        )
        torch.testing.assert_close(crossing.weight, self.tensor([0.0, 0.0]))

        decayed = dopamine_gated_weight_update(
            self.tensor([0.5, -0.5]),
            self.tensor([1.0, 1.0]),
            self.tensor([0.0, 0.0]),
            weight_decay=0.2,
        )
        torch.testing.assert_close(decayed.weight, self.tensor([0.4, -0.4]))

    def test_temporal_difference_positive_negative_and_terminal(self) -> None:
        critic = TemporalDifferenceRPE(discount=0.9, maximum_absolute=2.0)
        reward = self.tensor([1.0, -1.0, 0.5])
        value = self.tensor([0.2, 0.5, 0.1])
        next_value = self.tensor([0.5, 0.1, 10.0])
        terminal = self.tensor([0.0, 0.0, 1.0])
        signal = critic.compute(reward, value, next_value, terminal)
        expected = self.tensor([1.25, -1.41, 0.4])
        torch.testing.assert_close(signal, expected)

    def test_zero_concentration_has_finite_gradient(self) -> None:
        receptors = DopamineReceptors(
            d1_density=self.tensor([1.0]),
            d2_density=self.tensor([1.0]),
            tonic_concentration=0.5,
        )
        concentration = self.tensor([0.0]).requires_grad_(True)
        signal = receptors.response(concentration).net_phasic.sum()
        gradient = torch.autograd.grad(signal, concentration)[0]
        self.assertTrue(bool(torch.all(torch.isfinite(gradient))))

    def test_magnitude_update_weakens_inhibitory_d2_edge(self) -> None:
        update = dopamine_gated_weight_update(
            self.tensor([-0.5]),
            self.tensor([1.0]),
            self.tensor([-0.2]),
            learning_rate=1.0,
            preserve_sign=True,
            update_magnitude=True,
        )
        self.assertLess(float(update.weight.abs()), 0.5)

    def test_td_terminal_range_is_validated(self) -> None:
        critic = TemporalDifferenceRPE()
        with self.assertRaisesRegex(ValueError, "terminal"):
            critic.compute(
                self.tensor([0.0]),
                self.tensor([0.0]),
                self.tensor([1.0]),
                self.tensor([2.0]),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            critic.compute(
                self.tensor([0.0]),
                self.tensor([0.0]),
                self.tensor([1.0]),
                self.tensor([float("nan")]),
            )

    def test_joint_batch_update_preserves_sample_correlation(self) -> None:
        weight = self.tensor([0.2])
        eligibility = torch.tensor(
            [[1.0], [-1.0]], dtype=self.dtype, device=self.device
        )
        dopamine = torch.tensor(
            [[1.0], [-1.0]], dtype=self.dtype, device=self.device
        )
        update = dopamine_gated_batch_weight_update(
            weight,
            eligibility,
            dopamine,
            dst=torch.tensor([0], device=self.device),
            learning_rate=0.1,
        )
        torch.testing.assert_close(update.delta, self.tensor([0.1]))


if __name__ == "__main__":
    unittest.main()
