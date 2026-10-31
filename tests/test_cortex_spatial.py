from __future__ import annotations

import unittest

import torch

from neurosim_gpu.cortex.spatial import (
    PlateGeometry,
    loss_driven_rewire,
    migrate,
    rewire,
    sparsify_weights,
)


class SpatialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = PlateGeometry(
            n_neurons=128,
            mean_neuron_distance=1.0,
            plate_distance=50.0,
        )
        generator = torch.Generator().manual_seed(5)
        self.positions = self.geometry.sample_positions(seed=5)
        self.weights = torch.randn(128, 128, generator=generator) * 0.05
        self.activity = torch.rand(128, generator=generator)

    def test_geometry(self) -> None:
        self.assertAlmostEqual(self.geometry.plate_distance, 50.0)
        self.assertAlmostEqual(
            self.geometry.side,
            (128 * 1.0**3 / 50.0) ** 0.5,
            places=6,
        )
        self.assertTrue(
            bool((self.positions >= self.geometry.domain_min - 1e-6).all())
        )
        self.assertTrue(
            bool((self.positions <= self.geometry.domain_max + 1e-6).all())
        )

    def test_sparsify_keeps_fixed_degree(self) -> None:
        mask = sparsify_weights(
            self.weights,
            self.positions,
            edges_per_neuron=8,
            distance_decay=5.0,
        )
        self.assertEqual(tuple(mask.shape), (128, 128))
        self.assertTrue(bool((mask.sum(dim=1) == 8).all()))
        self.assertEqual(int(mask.sum()), 128 * 8)

    def test_migration_stays_in_domain(self) -> None:
        moved, displacement = migrate(
            self.positions,
            self.activity,
            self.geometry,
            maximum_speed=0.5,
        )
        self.assertEqual(displacement.shape, (128,))
        self.assertGreater(float(displacement.mean()), 0.0)
        self.assertTrue(
            bool((moved >= self.geometry.domain_min - 1e-5).all())
        )
        self.assertTrue(
            bool((moved <= self.geometry.domain_max + 1e-5).all())
        )

    def test_loss_driven_rewire_keeps_edge_count(self) -> None:
        mask = sparsify_weights(
            self.weights,
            self.positions,
            edges_per_neuron=8,
            distance_decay=5.0,
        )
        weights = self.weights * mask
        gradient = torch.randn_like(weights) * 0.01
        before = int(mask.sum())
        new_weights, new_mask, stats = loss_driven_rewire(
            weights,
            gradient,
            mask,
            self.positions,
            fraction=0.05,
            candidates_per_source=4,
        )
        self.assertEqual(int(new_mask.sum()), before)
        self.assertEqual(stats["loss_driven"], 1.0)
        self.assertGreater(stats["formed"], 0)
        self.assertGreater(stats["pruned"], 0)
        self.assertTrue(bool((new_weights[~new_mask] == 0).all()))

    def test_rewire_forms_and_prunes_without_changing_edge_count(self) -> None:
        mask = sparsify_weights(
            self.weights,
            self.positions,
            edges_per_neuron=8,
            distance_decay=5.0,
        )
        weights = self.weights * mask
        before = int(mask.sum())
        new_weights, new_mask, stats = rewire(
            weights,
            mask,
            self.positions,
            self.activity,
            fraction=0.05,
            candidates_per_source=4,
        )
        self.assertEqual(int(new_mask.sum()), before)
        self.assertGreater(stats["formed"], 0)
        self.assertGreater(stats["pruned"], 0)
        self.assertGreater(stats["rewire_ratio"], 0)
        self.assertTrue(bool((new_weights[~new_mask] == 0).all()))
        self.assertTrue(torch.isfinite(new_weights).all())


if __name__ == "__main__":
    unittest.main()
