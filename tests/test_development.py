from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from neurosim_gpu.development import (
    MigrationConfig,
    StructuralPlasticityConfig,
    migrate_neurons,
    score_candidate_edges,
    update_structural_connectome,
)
from neurosim_gpu.topology import SparseTopology


class MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

    def tensor(self, value: object) -> torch.Tensor:
        return torch.tensor(value, dtype=self.dtype, device=self.device)

    def test_zero_force_is_stationary(self) -> None:
        positions = self.tensor([[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]])
        actual = migrate_neurons(
            positions,
            positions.clone(),
            torch.zeros_like(positions),
            motility=self.tensor([0.5, 2.0]),
            domain_minimum=0.0,
            domain_maximum=1.0,
            config=MigrationConfig(dt=0.25, maximum_speed=10.0),
        )
        torch.testing.assert_close(actual, positions, rtol=0.0, atol=0.0)

    def test_speed_and_domain_bound_migration(self) -> None:
        positions = self.tensor([[0.95, 0.5, 0.5], [0.1, 0.1, 0.1]])
        target = self.tensor([[10.0, 10.0, 10.0], [-5.0, -5.0, -5.0]])
        gradient = self.tensor([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
        noise = self.tensor([[100.0, 100.0, 0.0], [-100.0, 0.0, 0.0]])
        config = MigrationConfig(
            dt=0.5,
            attraction_radius=0.2,
            maximum_gradient_norm=0.3,
            noise_scale=2.0,
            maximum_speed=0.2,
        )
        actual = migrate_neurons(
            positions,
            target,
            gradient,
            motility=4.0,
            domain_minimum=0.0,
            domain_maximum=1.0,
            config=config,
            noise=noise,
        )
        displacement = torch.linalg.vector_norm(actual - positions, dim=1)
        self.assertTrue(bool(torch.all(displacement <= 0.1 + 1e-12)))
        self.assertTrue(bool(torch.all((actual >= 0.0) & (actual <= 1.0))))

    def test_target_and_chemotaxis_have_expected_direction(self) -> None:
        positions = self.tensor([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]])
        target = self.tensor([[0.9, 0.5, 0.5], [0.9, 0.9, 0.9]])
        gradient = self.tensor([[0.0, 0.3, 0.0], [1.0, 1.0, 1.0]])
        actual = migrate_neurons(
            positions,
            target,
            gradient,
            motility=self.tensor([1.0, 0.0]),
            domain_minimum=0.0,
            domain_maximum=1.0,
            config=MigrationConfig(dt=0.1, maximum_speed=10.0),
        )
        self.assertGreater(float(actual[0, 0]), float(positions[0, 0]))
        self.assertGreater(float(actual[0, 1]), float(positions[0, 1]))
        self.assertEqual(float(actual[0, 2]), float(positions[0, 2]))
        torch.testing.assert_close(actual[1], positions[1], rtol=0.0, atol=0.0)

    def test_supplied_noise_is_deterministic(self) -> None:
        positions = self.tensor([[0.5, 0.5, 0.5]])
        noise = self.tensor([[0.3, -0.1, 0.2]])
        kwargs = dict(
            domain_minimum=0.0,
            domain_maximum=1.0,
            config=MigrationConfig(dt=0.1, noise_scale=0.5),
            noise=noise,
        )
        first = migrate_neurons(
            positions,
            positions,
            torch.zeros_like(positions),
            1.0,
            **kwargs,
        )
        second = migrate_neurons(
            positions,
            positions,
            torch.zeros_like(positions),
            1.0,
            **kwargs,
        )
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


class StructuralPlasticityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        self.positions = torch.stack(
            (
                torch.arange(6, device=self.device, dtype=self.dtype) * 0.1,
                torch.zeros(6, device=self.device, dtype=self.dtype),
                torch.zeros(6, device=self.device, dtype=self.dtype),
            ),
            dim=1,
        )
        self.activity = torch.ones(6, device=self.device, dtype=self.dtype)
        self.types = torch.tensor(
            [0, 1, 0, 1, 0, 1], dtype=torch.int64, device=self.device
        )
        self.sign = torch.tensor(
            [1, -1, 1, -1, 1, -1], dtype=self.dtype, device=self.device
        )
        self.all_types = torch.ones(
            (2, 2), dtype=torch.bool, device=self.device
        )
        self.chemical_only = StructuralPlasticityConfig(
            distance_weight=0.0,
            activity_weight=0.0,
            chemical_weight=1.0,
            formation_threshold=0.8,
            prune_threshold=0.3,
            maximum_new_edges_per_source=8,
            initial_weight_magnitude=0.05,
        )

    def indices(self, value: object) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.int64, device=self.device)

    def values(self, value: object) -> torch.Tensor:
        return torch.tensor(value, dtype=self.dtype, device=self.device)

    def update(
        self,
        old_src: torch.Tensor,
        old_dst: torch.Tensor,
        old_weight: torch.Tensor,
        candidate_src: torch.Tensor,
        candidate_dst: torch.Tensor,
        affinity: torch.Tensor,
        *,
        config: StructuralPlasticityConfig | None = None,
        compatibility: torch.Tensor | None = None,
    ):
        return update_structural_connectome(
            self.positions,
            self.activity,
            old_src,
            old_dst,
            old_weight,
            candidate_src,
            candidate_dst,
            affinity,
            self.types,
            self.sign,
            self.all_types if compatibility is None else compatibility,
            config=self.chemical_only if config is None else config,
        )

    def test_only_explicit_candidates_are_scored(self) -> None:
        large_positions = torch.zeros(
            (10_000, 3), dtype=self.dtype, device=self.device
        )
        activity = torch.ones(10_000, dtype=self.dtype, device=self.device)
        src = self.indices([0, 10, 9999])
        dst = self.indices([1, 20, 0])
        affinity = self.values([0.1, 0.5, 0.9])
        with patch("torch.cdist", side_effect=AssertionError("all-pairs used")):
            scores = score_candidate_edges(
                large_positions,
                activity,
                src,
                dst,
                affinity,
                config=self.chemical_only,
            )
        self.assertEqual(scores.total.shape, (3,))

    def test_formation_and_pruning_with_canonical_rebuild(self) -> None:
        result = self.update(
            self.indices([0, 1]),
            self.indices([1, 2]),
            self.values([0.2, -0.3]),
            self.indices([0, 1, 2]),
            self.indices([1, 2, 3]),
            self.values([0.1, 0.5, 0.9]),
        )
        self.assertEqual(result.n_pruned, 1)
        self.assertEqual(result.n_formed, 1)
        torch.testing.assert_close(result.src, self.indices([1, 2]))
        torch.testing.assert_close(result.dst, self.indices([2, 3]))
        torch.testing.assert_close(result.weight, self.values([-0.3, 0.05]))

        rebuilt = SparseTopology.from_edges(
            6,
            result.src,
            result.dst,
            result.weight,
        )
        torch.testing.assert_close(rebuilt.src.to(torch.int64), result.src)
        torch.testing.assert_close(rebuilt.dst.to(torch.int64), result.dst)
        torch.testing.assert_close(rebuilt.weight, result.weight)

    def test_new_edge_budget_is_per_source_and_score_ranked(self) -> None:
        config = StructuralPlasticityConfig(
            distance_weight=0.0,
            activity_weight=0.0,
            chemical_weight=1.0,
            formation_threshold=0.8,
            prune_threshold=0.2,
            maximum_new_edges_per_source=2,
            initial_weight_magnitude=0.05,
        )
        result = self.update(
            self.indices([]),
            self.indices([]),
            self.values([]),
            self.indices([0, 0, 0, 0, 1, 1]),
            self.indices([1, 2, 3, 4, 2, 3]),
            self.values([0.90, 0.95, 0.95, 0.99, 0.91, 0.92]),
            config=config,
        )
        expected_pairs = {(0, 2), (0, 4), (1, 2), (1, 3)}
        actual_pairs = set(zip(result.src.tolist(), result.dst.tolist()))
        self.assertEqual(actual_pairs, expected_pairs)
        self.assertEqual(int((result.src == 0).sum()), 2)
        self.assertEqual(int((result.src == 1).sum()), 2)

    def test_type_compatibility_and_presynaptic_sign(self) -> None:
        compatibility = torch.tensor(
            [[True, False], [True, True]],
            dtype=torch.bool,
            device=self.device,
        )
        result = self.update(
            self.indices([]),
            self.indices([]),
            self.values([]),
            self.indices([0, 1]),
            self.indices([1, 2]),
            self.values([1.0, 1.0]),
            compatibility=compatibility,
        )
        torch.testing.assert_close(result.src, self.indices([1]))
        torch.testing.assert_close(result.dst, self.indices([2]))
        torch.testing.assert_close(result.weight, self.values([-0.05]))
        self.assertFalse(bool(result.formed_mask[0]))
        self.assertTrue(bool(result.formed_mask[1]))

    def test_candidate_order_does_not_change_deterministic_output(self) -> None:
        src = self.indices([0, 0, 1, 2])
        dst = self.indices([2, 1, 3, 4])
        affinity = self.values([0.9, 0.9, 0.95, 1.0])
        empty_index = self.indices([])
        empty_weight = self.values([])
        first = self.update(
            empty_index,
            empty_index,
            empty_weight,
            src,
            dst,
            affinity,
        )
        permutation = self.indices([3, 1, 0, 2])
        second = self.update(
            empty_index,
            empty_index,
            empty_weight,
            src[permutation],
            dst[permutation],
            affinity[permutation],
        )
        torch.testing.assert_close(first.src, second.src, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first.dst, second.dst, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first.weight, second.weight, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
