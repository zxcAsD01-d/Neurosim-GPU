from __future__ import annotations

import unittest

from neurosim_gpu.cost_model import CostModel


class CostModelTests(unittest.TestCase):
    def test_event_work_drops_with_activity(self) -> None:
        full = CostModel(1000, 10000, active_edge_fraction=1.0)
        sparse = CostModel(1000, 10000, active_edge_fraction=0.01)
        self.assertLess(sparse.event_work, full.event_work)
        self.assertEqual(full.base_work, sparse.base_work)

    def test_multirate_terms_are_amortized(self) -> None:
        fast = CostModel(1000, 10000, chemistry_interval=1)
        slow = CostModel(1000, 10000, chemistry_interval=10)
        self.assertGreater(fast.chemistry_work_per_neural_tick, slow.chemistry_work_per_neural_tick)

    def test_memory_scales_linearly(self) -> None:
        one = CostModel(1000, 10000)
        two = CostModel(1000, 20000)
        self.assertGreater(two.slim_inference_bytes, one.slim_inference_bytes)
        self.assertEqual(two.mutable_online_bytes - one.mutable_online_bytes, 24 * 10000)

    def test_optional_full_system_terms_are_explicit(self) -> None:
        core = CostModel(1000, 10000)
        embodied = CostModel(
            1000,
            10000,
            include_communication=True,
            include_eligibility=True,
            include_plasticity=True,
            include_development=True,
            migration_interval=100,
            structural_interval=1000,
            candidate_edges=1000,
        )
        self.assertGreater(embodied.full_matrix_work, core.full_matrix_work)


if __name__ == "__main__":
    unittest.main()
