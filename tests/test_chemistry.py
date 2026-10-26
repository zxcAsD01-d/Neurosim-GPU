from __future__ import annotations

import unittest

import torch

from neurosim_gpu.chemistry import RegionalDopamineField


class RegionalDopamineFieldTests(unittest.TestCase):
    @staticmethod
    def devices() -> list[torch.device]:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        return devices

    def make_field(
        self,
        device: torch.device,
        *,
        baseline: float = 0.4,
        uptake: float = 0.3,
        decay: float = 0.1,
        diffusion: float = 0.2,
        dt: float = 0.1,
    ) -> RegionalDopamineField:
        dtype = torch.float64
        return RegionalDopamineField(
            tonic_baseline=torch.full((4,), baseline, device=device, dtype=dtype),
            uptake_rate=torch.full((4,), uptake, device=device, dtype=dtype),
            decay_rate=torch.full((4,), decay, device=device, dtype=dtype),
            release_projection=torch.tensor(
                [[1.0, 0.5, 0.25, 0.0], [0.0, 0.25, 0.5, 1.0]],
                device=device,
                dtype=dtype,
            ),
            edge_src=torch.tensor([0, 1, 2], device=device),
            edge_dst=torch.tensor([1, 2, 3], device=device),
            diffusion_rate=torch.full(
                (3,), diffusion, device=device, dtype=dtype
            ),
            dt=dt,
            burst_gain=0.6,
            dip_gain=1.0,
        )

    def test_tonic_baseline_is_equilibrium(self) -> None:
        for device in self.devices():
            with self.subTest(device=device):
                field = self.make_field(device)
                concentration = field.tonic_baseline.expand(3, -1).clone()
                rpe = torch.zeros(
                    (3, field.n_sources), device=device, dtype=torch.float64
                )
                for _ in range(100):
                    concentration = field.advance(concentration, rpe)
                torch.testing.assert_close(
                    concentration,
                    field.tonic_baseline.expand_as(concentration),
                    rtol=0.0,
                    atol=2e-14,
                )

    def test_positive_rpe_bursts_and_negative_rpe_dips(self) -> None:
        for device in self.devices():
            with self.subTest(device=device):
                field = self.make_field(device)
                baseline = field.tonic_baseline.unsqueeze(0)
                neutral = field.advance(
                    baseline,
                    torch.zeros((1, 2), device=device, dtype=torch.float64),
                )
                burst = field.advance(
                    baseline,
                    torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float64),
                )
                dip = field.advance(
                    baseline,
                    torch.tensor([[-1.0, 0.0]], device=device, dtype=torch.float64),
                )
                projected = field.release_projection[0] > 0
                self.assertTrue(torch.all(burst[0, projected] > neutral[0, projected]))
                self.assertTrue(torch.all(dip[0, projected] < neutral[0, projected]))
                self.assertEqual(float(burst[0, 3]), float(neutral[0, 3]))
                self.assertEqual(float(dip[0, 3]), float(neutral[0, 3]))

    def test_negative_rpe_never_creates_negative_concentration(self) -> None:
        for device in self.devices():
            with self.subTest(device=device):
                field = self.make_field(device)
                concentration = torch.full(
                    (5, 4), 1e-9, device=device, dtype=torch.float64
                )
                rpe = torch.full(
                    (5, 2), -1e6, device=device, dtype=torch.float64
                )
                for _ in range(200):
                    concentration = field.advance(concentration, rpe)
                    self.assertGreaterEqual(float(concentration.min()), 0.0)

    def test_diffusion_conserves_mass_without_release_or_clearance(self) -> None:
        for device in self.devices():
            with self.subTest(device=device):
                field = self.make_field(
                    device,
                    baseline=0.0,
                    uptake=0.0,
                    decay=0.0,
                    diffusion=0.4,
                    dt=0.2,
                )
                concentration = torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]],
                    device=device,
                    dtype=torch.float64,
                )
                initial_mass = concentration.sum(dim=1)
                zero_rpe = torch.zeros((2, 2), device=device, dtype=torch.float64)
                for _ in range(50):
                    concentration = field.advance(concentration, zero_rpe)
                torch.testing.assert_close(
                    concentration.sum(dim=1),
                    initial_mass,
                    rtol=1e-13,
                    atol=1e-13,
                )

    def test_diffusion_conserves_volume_weighted_mass(self) -> None:
        device = self.device_for_volume_test()
        dtype = torch.float64
        volume = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        field = RegionalDopamineField(
            tonic_baseline=torch.zeros(2, device=device, dtype=dtype),
            uptake_rate=torch.zeros(2, device=device, dtype=dtype),
            decay_rate=torch.zeros(2, device=device, dtype=dtype),
            release_projection=torch.zeros(1, 2, device=device, dtype=dtype),
            edge_src=torch.tensor([0], device=device),
            edge_dst=torch.tensor([1], device=device),
            diffusion_rate=torch.tensor([0.4], device=device, dtype=dtype),
            dt=0.2,
            region_volume=volume,
        )
        concentration = torch.tensor(
            [[1.0, 0.0]], device=device, dtype=dtype
        )
        initial_mass = (concentration * volume).sum()
        for _ in range(30):
            concentration = field.advance(
                concentration,
                torch.zeros(1, 1, device=device, dtype=dtype),
            )
        torch.testing.assert_close(
            (concentration * volume).sum(),
            initial_mass,
            rtol=1e-13,
            atol=1e-13,
        )

    @staticmethod
    def device_for_volume_test() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_batches_are_isolated(self) -> None:
        for device in self.devices():
            with self.subTest(device=device):
                field = self.make_field(device)
                concentration = torch.tensor(
                    [[0.2, 0.3, 0.4, 0.5], [0.8, 0.6, 0.4, 0.2]],
                    device=device,
                    dtype=torch.float64,
                )
                rpe = torch.tensor(
                    [[0.7, -0.2], [-0.5, 1.1]],
                    device=device,
                    dtype=torch.float64,
                )
                together = field.advance(concentration, rpe)
                separately = torch.cat(
                    [
                        field.advance(concentration[index : index + 1], rpe[index])
                        for index in range(2)
                    ],
                    dim=0,
                )
                torch.testing.assert_close(together, separately, rtol=0.0, atol=0.0)

    def test_unstable_timestep_is_rejected(self) -> None:
        device = torch.device("cpu")
        with self.assertRaisesRegex(ValueError, "stability"):
            self.make_field(device, uptake=2.0, decay=1.0, diffusion=1.0, dt=1.0)

    def test_autoreceptor_reduces_burst_above_tonic(self) -> None:
        device = self.device_for_volume_test()
        dtype = torch.float64
        field = RegionalDopamineField(
            tonic_baseline=torch.tensor([0.5], device=device, dtype=dtype),
            uptake_rate=torch.tensor([0.1], device=device, dtype=dtype),
            decay_rate=torch.zeros(1, device=device, dtype=dtype),
            release_projection=torch.ones(1, 1, device=device, dtype=dtype),
            edge_src=torch.empty(0, device=device, dtype=torch.int64),
            edge_dst=torch.empty(0, device=device, dtype=torch.int64),
            diffusion_rate=torch.empty(0, device=device, dtype=dtype),
            dt=0.1,
            burst_gain=1.0,
            autoreceptor_gain=5.0,
            autoreceptor_half_saturation=0.25,
        )
        rpe = torch.ones(1, 1, device=device, dtype=dtype)
        tonic = field.advance(torch.tensor([[0.5]], device=device, dtype=dtype), rpe)
        high = field.advance(torch.tensor([[2.0]], device=device, dtype=dtype), rpe)
        tonic_increment = tonic - 0.5
        high_without_passive = high - (
            2.0 + field.dt * (field.tonic_release_rate - 0.2)
        )
        self.assertLess(float(high_without_passive), float(tonic_increment))

    def test_source_specific_dip_preserves_other_tonic_pool(self) -> None:
        device = self.device_for_volume_test()
        dtype = torch.float64
        field = RegionalDopamineField(
            tonic_baseline=torch.tensor([1.0], device=device, dtype=dtype),
            uptake_rate=torch.tensor([0.2], device=device, dtype=dtype),
            decay_rate=torch.zeros(1, device=device, dtype=dtype),
            release_projection=torch.ones(2, 1, device=device, dtype=dtype),
            edge_src=torch.empty(0, device=device, dtype=torch.int64),
            edge_dst=torch.empty(0, device=device, dtype=torch.int64),
            diffusion_rate=torch.empty(0, device=device, dtype=dtype),
            dt=0.1,
            dip_gain=1.0,
        )
        baseline = torch.tensor([[1.0]], device=device, dtype=dtype)
        one_dip = field.advance(
            baseline,
            torch.tensor([[-1.0, 0.0]], device=device, dtype=dtype),
        )
        both_dip = field.advance(
            baseline,
            torch.tensor([[-1.0, -1.0]], device=device, dtype=dtype),
        )
        neutral = field.advance(
            baseline,
            torch.zeros(1, 2, device=device, dtype=dtype),
        )
        self.assertLess(float(one_dip), float(neutral))
        self.assertGreater(float(one_dip), float(both_dip))


if __name__ == "__main__":
    unittest.main()
