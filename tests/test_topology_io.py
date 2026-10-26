from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import torch

from neurosim_gpu.topology import SparseTopology
from neurosim_gpu.topology_io import (
    SingleWeightInferenceView,
    load_topology,
    memory_accounting,
    save_topology,
    single_weight_inference_view,
    topology_from_dict,
    topology_to_dict,
)


class TopologyIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        # Deliberately include parallel edges; SparseTopology canonicalizes and
        # coalesces them before this module sees the graph.
        src = torch.tensor([2, 0, 1, 2, 0, 3], dtype=torch.int64, device=self.device)
        dst = torch.tensor([1, 2, 0, 1, 2, 0], dtype=torch.int64, device=self.device)
        weight = torch.tensor(
            [0.3, 0.4, -0.2, 0.7, 0.1, 0.05],
            dtype=self.dtype,
            device=self.device,
        )
        self.topology = SparseTopology.from_edges(4, src, dst, weight)

    def test_roundtrip_cpu_resets_weight_version(self) -> None:
        self.topology.mark_weights_mutated_()
        self.topology.mark_weights_mutated_()
        payload = topology_to_dict(self.topology)
        self.assertEqual(payload["src"].device.type, "cpu")
        self.assertEqual(payload["src"].dtype, torch.int32)
        self.assertNotIn("in_weight", payload)

        loaded = topology_from_dict(payload)
        self.assertIsInstance(loaded, SparseTopology)
        self.assertEqual(loaded.weight_version, 0)
        self.assertEqual(loaded.n_neurons, self.topology.n_neurons)
        for name in (
            "src",
            "dst",
            "weight",
            "out_rowptr",
            "in_rowptr",
            "in_col",
            "out_weight_index",
            "in_weight",
        ):
            torch.testing.assert_close(getattr(loaded, name).cpu(), getattr(self.topology, name).cpu())

    def test_path_and_buffer_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology.pt"
            save_topology(self.topology, path, include_destination_weights=True)
            loaded = load_topology(path)
            torch.testing.assert_close(
                loaded.dense_matrix(), self.topology.dense_matrix().cpu()
            )

        buffer = io.BytesIO()
        save_topology(self.topology, buffer)
        buffer.seek(0)
        loaded = load_topology(buffer)
        torch.testing.assert_close(loaded.in_weight, self.topology.in_weight.cpu())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_load_can_transfer_all_views_to_cuda(self) -> None:
        cpu_topology = SparseTopology.from_edges(
            self.topology.n_neurons,
            self.topology.src.cpu(),
            self.topology.dst.cpu(),
            self.topology.weight.cpu(),
        )
        buffer = io.BytesIO()
        save_topology(cpu_topology, buffer)
        buffer.seek(0)
        loaded = load_topology(buffer, device="cuda")
        self.assertEqual(loaded.src.device.type, "cuda")
        self.assertEqual(loaded.in_rowptr.device.type, "cuda")
        torch.testing.assert_close(loaded.dense_matrix().cpu(), cpu_topology.dense_matrix())

    def test_deterministic_order_and_coalescing(self) -> None:
        self.assertEqual(self.topology.n_edges, 4)
        first = topology_to_dict(self.topology)
        second = topology_to_dict(self.topology)
        for key in first:
            if isinstance(first[key], torch.Tensor):
                torch.testing.assert_close(first[key], second[key])
            else:
                self.assertEqual(first[key], second[key])
        keys = first["src"].to(torch.int64) * self.topology.n_neurons + first["dst"].to(torch.int64)
        self.assertTrue(bool(torch.all(keys[1:] > keys[:-1])))
        # (0, 2) and (2, 1) were each supplied twice and are summed.
        pairs = {
            (int(src), int(dst)): float(weight)
            for src, dst, weight in zip(
                self.topology.src.tolist(),
                self.topology.dst.tolist(),
                self.topology.weight.tolist(),
            )
        }
        self.assertAlmostEqual(pairs[(0, 2)], 0.5)
        self.assertAlmostEqual(pairs[(2, 1)], 1.0)

    def test_malformed_metadata_is_rejected(self) -> None:
        payload = topology_to_dict(self.topology)

        bad_dtype = dict(payload)
        bad_dtype["src"] = payload["src"].to(torch.int64)
        with self.assertRaisesRegex(ValueError, "int32"):
            topology_from_dict(bad_dtype)

        bad_duplicate = dict(payload)
        bad_duplicate["src"] = payload["src"].clone()
        bad_duplicate["dst"] = payload["dst"].clone()
        bad_duplicate["src"][1] = bad_duplicate["src"][0]
        bad_duplicate["dst"][1] = bad_duplicate["dst"][0]
        with self.assertRaisesRegex(ValueError, "duplicate|ordered"):
            topology_from_dict(bad_duplicate)

        bad_range = dict(payload)
        bad_range["dst"] = payload["dst"].clone()
        bad_range["dst"][0] = self.topology.n_neurons
        with self.assertRaisesRegex(ValueError, "range"):
            topology_from_dict(bad_range)

        bad_permutation = dict(payload)
        bad_permutation["out_weight_index"] = payload["out_weight_index"].clone()
        bad_permutation["out_weight_index"][0] = bad_permutation["out_weight_index"][1]
        with self.assertRaisesRegex(ValueError, "permutation|destination"):
            topology_from_dict(bad_permutation)

    def test_memory_accounting_and_single_weight_view(self) -> None:
        accounting = memory_accounting(self.topology)
        self.assertEqual(accounting.total_bytes, self.topology.storage_bytes())
        self.assertLess(accounting.compact_serialized_bytes, accounting.canonical_bytes)
        view = single_weight_inference_view(self.topology)
        self.assertIsInstance(view, SingleWeightInferenceView)
        self.assertEqual(view.storage_bytes(), self.topology.native_storage_bytes())
        torch.testing.assert_close(view.sparse_matrix().to_dense(), self.topology.dense_matrix())
        self.assertFalse(hasattr(view, "apply_source_weights_"))

    def test_single_weight_view_can_be_loaded_directly(self) -> None:
        view = topology_from_dict(topology_to_dict(self.topology), single_weight=True)
        self.assertIsInstance(view, SingleWeightInferenceView)
        torch.testing.assert_close(
            view.sparse_matrix().to_dense(), self.topology.dense_matrix().cpu()
        )
        self.assertEqual(view.weight.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
