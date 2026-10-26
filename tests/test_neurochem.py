from __future__ import annotations

import unittest

import torch

from neurosim_gpu.chemistry import RegionalDopamineField
from neurosim_gpu.dopamine import DopamineReceptors
from neurosim_gpu.dynamics import Dynamics
from neurosim_gpu.neurochem import (
    DopamineMatrixRuntime,
    DopaminePersistentRuntime,
    DopaminePlasticityConfig,
)
from neurosim_gpu.topology import SparseTopology


class DopamineRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

    def build_runtime(
        self,
        *,
        learning_rate: float = 0.0,
        d1_only: bool = False,
        d2_only: bool = False,
        chemistry_interval: int = 1,
        dopamine_delay_steps: int = 0,
    ) -> DopamineMatrixRuntime:
        src = torch.tensor([0, 0, 1, 2], device=self.device, dtype=torch.int64)
        dst = torch.tensor([1, 2, 3, 3], device=self.device, dtype=torch.int64)
        weight = torch.tensor(
            [0.12, 0.10, -0.08, 0.09],
            device=self.device,
            dtype=self.dtype,
        )
        topology = SparseTopology.from_edges(4, src, dst, weight)
        dynamics = Dynamics(
            alpha=torch.full((4,), 0.8, device=self.device, dtype=self.dtype),
            gain=torch.full((4,), 0.7, device=self.device, dtype=self.dtype),
        )
        chemistry = RegionalDopamineField(
            tonic_baseline=torch.full(
                (2,), 0.5, device=self.device, dtype=self.dtype
            ),
            uptake_rate=torch.full(
                (2,), 0.2, device=self.device, dtype=self.dtype
            ),
            decay_rate=torch.zeros(2, device=self.device, dtype=self.dtype),
            release_projection=torch.tensor(
                [[1.0, 1.0]], device=self.device, dtype=self.dtype
            ),
            edge_src=torch.tensor([0], device=self.device),
            edge_dst=torch.tensor([1], device=self.device),
            diffusion_rate=torch.tensor(
                [0.1], device=self.device, dtype=self.dtype
            ),
            dt=0.1,
            burst_gain=0.8,
            dip_gain=1.0,
        )
        if d1_only:
            d1 = torch.ones(4, device=self.device, dtype=self.dtype)
            d2 = torch.zeros_like(d1)
        elif d2_only:
            d2 = torch.ones(4, device=self.device, dtype=self.dtype)
            d1 = torch.zeros_like(d2)
        else:
            d1 = torch.tensor(
                [1.0, 0.0, 1.0, 0.0], device=self.device, dtype=self.dtype
            )
            d2 = 1.0 - d1
        receptors = DopamineReceptors(
            d1_density=d1,
            d2_density=d2,
            tonic_concentration=0.5,
            d1_half_saturation=0.5,
            d2_half_saturation=0.5,
        )
        return DopamineMatrixRuntime(
            topology,
            dynamics,
            torch.full((1, 4), 0.4, device=self.device, dtype=self.dtype),
            chemistry=chemistry,
            receptors=receptors,
            neuron_region=torch.tensor(
                [0, 0, 1, 1], device=self.device, dtype=torch.int64
            ),
            config=DopaminePlasticityConfig(
                drive_scale=1.5,
                log_gain_scale=0.0,
                pre_trace_decay=0.0,
                post_trace_decay=0.0,
                eligibility_decay=0.0,
                potentiation_scale=1.0,
                depression_scale=0.0,
                learning_rate=learning_rate,
                minimum_weight=-1.0,
                maximum_weight=1.0,
                preserve_weight_sign=True,
                chemistry_interval=chemistry_interval,
                dopamine_delay_steps=dopamine_delay_steps,
            ),
        )

    def build_persistent_runtime(
        self,
        *,
        theta: float,
        learning_rate: float = 0.0,
    ) -> DopaminePersistentRuntime:
        matrix = self.build_runtime(learning_rate=learning_rate)
        return DopaminePersistentRuntime(
            matrix.topology,
            matrix.dynamics,
            matrix.state.clone(),
            chemistry=matrix.chemistry,
            receptors=matrix.receptors,
            neuron_region=matrix.neuron_region,
            theta=theta,
            config=matrix.config,
            initial_concentration=matrix.concentration.clone(),
        )

    def rpe(self, value: float) -> torch.Tensor:
        return torch.tensor([[value]], device=self.device, dtype=self.dtype)

    def test_reward_cannot_change_already_computed_transition(self) -> None:
        burst = self.build_runtime()
        dip = self.build_runtime()
        burst_state = burst.step(rpe=self.rpe(1.0))
        dip_state = dip.step(rpe=self.rpe(-1.0))
        torch.testing.assert_close(burst_state, dip_state, rtol=0.0, atol=0.0)
        self.assertTrue(bool(torch.all(burst.concentration > dip.concentration)))

    def test_burst_and_dip_affect_following_d1_transition_oppositely(self) -> None:
        burst = self.build_runtime(d1_only=True)
        neutral = self.build_runtime(d1_only=True)
        dip = self.build_runtime(d1_only=True)
        burst.step(rpe=self.rpe(1.0))
        neutral.step(rpe=self.rpe(0.0))
        dip.step(rpe=self.rpe(-1.0))
        burst_next = burst.step(rpe=self.rpe(0.0))
        neutral_next = neutral.step(rpe=self.rpe(0.0))
        dip_next = dip.step(rpe=self.rpe(0.0))
        self.assertTrue(bool(torch.all(burst_next > neutral_next)))
        self.assertTrue(bool(torch.all(dip_next < neutral_next)))

    def test_d2_pathway_reverses_d1_drive_direction(self) -> None:
        d1 = self.build_runtime(d1_only=True)
        d2 = self.build_runtime(d2_only=True)
        d1.step(rpe=self.rpe(1.0))
        d2.step(rpe=self.rpe(1.0))
        d1_next = d1.step(rpe=self.rpe(0.0))
        d2_next = d2.step(rpe=self.rpe(0.0))
        self.assertTrue(bool(torch.all(d1_next > d2_next)))

    def test_tonic_state_has_no_dopamine_weight_drift(self) -> None:
        runtime = self.build_runtime(learning_rate=0.5)
        initial_weight = runtime.topology.weight.clone()
        _, stats = runtime.step(rpe=self.rpe(0.0), collect_stats=True)
        torch.testing.assert_close(runtime.topology.weight, initial_weight)
        self.assertEqual(stats.weight_delta_maximum, 0.0)

    def test_receptor_refresh_can_be_disabled_for_neutral_realtime_profile(self) -> None:
        enabled = self.build_runtime(chemistry_interval=2)
        disabled = self.build_runtime(chemistry_interval=2)
        enabled.config = DopaminePlasticityConfig(
            chemistry_interval=2,
            drive_scale=0.0,
            log_gain_scale=0.0,
            track_eligibility=False,
            refresh_receptors=True,
        )
        disabled.config = DopaminePlasticityConfig(
            chemistry_interval=2,
            drive_scale=0.0,
            log_gain_scale=0.0,
            track_eligibility=False,
            refresh_receptors=False,
        )
        disabled.current_signals = disabled._signals(disabled.concentration)
        disabled._drive_shift, disabled._gain_scale = disabled._modulation(
            disabled.current_signals
        )
        for value in (0.2, 0.0, -0.1, 0.0):
            rpe = self.rpe(value)
            expected = enabled.step(rpe=rpe)
            actual = disabled.step(rpe=rpe)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_burst_updates_d1_and_d2_targets_with_opposite_signs(self) -> None:
        runtime = self.build_runtime(learning_rate=0.5)
        initial = runtime.topology.weight.clone()
        runtime.step(rpe=self.rpe(1.0))
        delta = runtime.topology.weight - initial
        targets = runtime.topology.dst
        d1_edges = (targets == 0) | (targets == 2)
        d2_edges = (targets == 1) | (targets == 3)
        magnitude_delta = runtime.topology.weight.abs() - initial.abs()
        if bool(torch.any(d1_edges)):
            self.assertTrue(bool(torch.all(magnitude_delta[d1_edges] >= 0)))
        self.assertTrue(bool(torch.all(magnitude_delta[d2_edges] < 0)))
        reconstructed = runtime.topology.in_weight[
            runtime.topology.out_weight_index.to(torch.int64)
        ]
        torch.testing.assert_close(reconstructed, runtime.topology.weight)

    def test_theta_zero_persistent_matches_matrix_chemistry(self) -> None:
        matrix = self.build_runtime(learning_rate=0.1)
        persistent = self.build_persistent_runtime(theta=0.0, learning_rate=0.1)
        for value in (1.0, 0.0, -0.5, 0.25, 0.0):
            expected = matrix.step(rpe=self.rpe(value))
            actual = persistent.step(rpe=self.rpe(value))
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(
            persistent.concentration,
            matrix.concentration,
            rtol=1e-12,
            atol=1e-13,
        )
        torch.testing.assert_close(
            persistent.topology.weight,
            matrix.topology.weight,
            rtol=1e-12,
            atol=1e-13,
        )

    def test_plasticity_resynchronizes_persistent_current(self) -> None:
        runtime = self.build_persistent_runtime(theta=0.01, learning_rate=0.5)
        runtime.step(rpe=self.rpe(1.0))
        recomputed = runtime._propagate(runtime.emitted_state).contiguous()
        torch.testing.assert_close(
            runtime.persistent_current,
            recomputed,
            rtol=1e-12,
            atol=1e-13,
        )
        self.assertLess(
            float((runtime.state - runtime.emitted_state).abs().max()),
            0.01 + 1e-12,
        )

    def test_multirate_chemistry_holds_between_boundaries(self) -> None:
        runtime = self.build_runtime(
            d1_only=True,
            chemistry_interval=2,
        )
        baseline = runtime.concentration.clone()
        _, first = runtime.step(rpe=self.rpe(1.0), collect_stats=True)
        torch.testing.assert_close(runtime.concentration, baseline)
        self.assertFalse(first.chemistry_applied)
        _, second = runtime.step(rpe=self.rpe(1.0), collect_stats=True)
        self.assertTrue(second.chemistry_applied)
        self.assertTrue(bool(torch.all(runtime.concentration > baseline)))

    def test_rpe_delay_bucket_defers_chemical_response(self) -> None:
        runtime = self.build_runtime(
            d1_only=True,
            dopamine_delay_steps=2,
        )
        baseline = runtime.concentration.clone()
        runtime.step(rpe=self.rpe(1.0))
        runtime.step(rpe=self.rpe(0.0))
        torch.testing.assert_close(runtime.concentration, baseline)
        runtime.step(rpe=self.rpe(0.0))
        self.assertTrue(bool(torch.all(runtime.concentration > baseline)))

    def test_invalid_release_is_transactional(self) -> None:
        runtime = self.build_runtime(learning_rate=0.2)
        state = runtime.state.clone()
        concentration = runtime.concentration.clone()
        eligibility = runtime.eligibility.clone()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            runtime.step(
                rpe=self.rpe(0.0),
                source_release=self.rpe(-1.0),
            )
        torch.testing.assert_close(runtime.state, state)
        torch.testing.assert_close(runtime.concentration, concentration)
        torch.testing.assert_close(runtime.eligibility, eligibility)
        self.assertEqual(runtime._tick, 0)

    def test_shared_online_plasticity_rejects_batching_approximation(self) -> None:
        runtime = self.build_runtime(learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "batch=1"):
            DopamineMatrixRuntime(
                runtime.topology,
                runtime.dynamics,
                runtime.state.expand(2, -1).clone(),
                chemistry=runtime.chemistry,
                receptors=runtime.receptors,
                neuron_region=runtime.neuron_region,
                config=DopaminePlasticityConfig(learning_rate=0.1),
            )
        with self.assertRaisesRegex(ValueError, "batch=1"):
            DopamineMatrixRuntime(
                runtime.topology,
                runtime.dynamics,
                runtime.state.expand(2, -1).clone(),
                chemistry=runtime.chemistry,
                receptors=runtime.receptors,
                neuron_region=runtime.neuron_region,
                config=DopaminePlasticityConfig(weight_decay=0.1),
            )


if __name__ == "__main__":
    unittest.main()
