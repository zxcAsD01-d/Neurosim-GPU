from __future__ import annotations

import unittest

import torch

from neurosim_gpu.cortex.config import BrainConfig
from neurosim_gpu.cortex.topology import build_cortical_topology


class TopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BrainConfig(
            n_neurons=1024,
            edges_per_neuron=16,
            regions=8,
            local_fraction=0.6,
            inhibitory_fraction=0.2,
            incoming_norm=2.0,
            seed=11,
        )
        self.topology = build_cortical_topology(self.config)

    def test_edge_count_and_range(self) -> None:
        graph = self.topology.graph
        self.assertGreater(graph.n_edges, 0)
        # SparseTopology coalesces parallel edges, so the canonical count is at
        # most the requested count. Most of the loss comes from local edges in
        # small regions, where the source/destination pair space is finite.
        self.assertLessEqual(graph.n_edges, self.config.n_edges)
        self.assertGreater(graph.n_edges, 0.95 * self.config.n_edges)
        self.assertTrue(bool((graph.src >= 0).all()))
        self.assertTrue(bool((graph.dst >= 0).all()))
        self.assertTrue(bool((graph.src < self.config.n_neurons).all()))
        self.assertTrue(bool((graph.dst < self.config.n_neurons).all()))

    def test_dales_law_is_per_neuron(self) -> None:
        graph = self.topology.graph
        sign = torch.sign(graph.weight)
        for neuron in range(self.config.n_neurons):
            outgoing = sign[graph.src == neuron]
            if outgoing.numel():
                self.assertTrue(bool((outgoing == outgoing[0]).all()))
        observed = float((self.topology.sign < 0).float().mean())
        self.assertAlmostEqual(
            observed, self.config.inhibitory_fraction, delta=0.03
        )

    def test_incoming_weights_are_normalised(self) -> None:
        graph = self.topology.graph
        incoming = torch.zeros(self.config.n_neurons).index_add_(
            0, graph.dst, graph.weight.abs()
        )
        torch.testing.assert_close(
            incoming,
            torch.full_like(incoming, self.config.incoming_norm),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_local_fraction_is_respected(self) -> None:
        graph = self.topology.graph
        same_region = (
            self.topology.region[graph.src]
            == self.topology.region[graph.dst]
        ).float().mean()
        self.assertGreater(float(same_region), self.config.local_fraction)

    def test_regions_are_contiguous(self) -> None:
        region = self.topology.region
        for value in range(self.config.regions):
            positions = torch.nonzero(region == value).flatten()
            self.assertGreater(positions.numel(), 0)
            self.assertEqual(
                int(positions.max() - positions.min() + 1),
                positions.numel(),
            )

    def test_build_is_deterministic(self) -> None:
        again = build_cortical_topology(self.config)
        torch.testing.assert_close(
            again.graph.weight, self.topology.graph.weight
        )
        torch.testing.assert_close(
            again.graph.src, self.topology.graph.src
        )
        torch.testing.assert_close(again.alpha, self.topology.alpha)

    def test_different_seed_changes_graph(self) -> None:
        other = build_cortical_topology(
            BrainConfig(
                n_neurons=1024,
                edges_per_neuron=16,
                regions=8,
                seed=12,
            )
        )
        self.assertFalse(
            bool(torch.equal(other.graph.src, self.topology.graph.src))
        )

    def test_alpha_range_and_gain(self) -> None:
        self.assertTrue(
            bool(
                (self.topology.alpha >= self.config.alpha_min).all()
                and (self.topology.alpha < self.config.alpha_max).all()
            )
        )
        self.assertTrue(bool((self.topology.base_gain > 0).all()))

    def test_describe(self) -> None:
        description = self.topology.describe()
        self.assertEqual(description["n_neurons"], 1024)
        self.assertEqual(description["n_regions"], 8)
        self.assertGreater(float(description["inhibitory_fraction"]), 0.0)


if __name__ == "__main__":
    unittest.main()
