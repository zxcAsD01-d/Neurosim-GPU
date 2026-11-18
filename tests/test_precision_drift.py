from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from neurosim_gpu import DendriticDynamics, DendriticMatrixRuntime, Dynamics, random_topology


_SPEC = importlib.util.spec_from_file_location(
    "precision_drift_benchmark", Path(__file__).resolve().parents[1] / "benchmarks/precision_drift.py",
)
assert _SPEC is not None and _SPEC.loader is not None
drift = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drift)


class DriftSamplingTests(unittest.TestCase):
    def test_ticks_include_endpoints_regular_log_and_input_boundaries(self):
        ticks = drift.sample_ticks(100000, 256)
        self.assertEqual(ticks, sorted(set(ticks)))
        self.assertEqual((ticks[0], ticks[-1]), (0, 100000))
        self.assertTrue(set(range(256, 100001, 256)).issubset(ticks))
        self.assertTrue({1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,
                         5000, 10000, 20000, 50000, 100000}.issubset(ticks))
        for boundary in (8000, 24000, 32000, 48000, 56000, 72000, 80000, 96000):
            self.assertTrue({boundary - 1, boundary, boundary + 1}.issubset(ticks))

    def test_tiny_run_has_no_out_of_range_samples(self):
        self.assertEqual(drift.sample_ticks(1, 256), [0, 1])
        self.assertTrue(all(0 <= tick <= 13 for tick in drift.sample_ticks(13, 7)))
        for steps, interval in ((0, 1), (1, 0), (-1, 2)):
            with self.assertRaises(ValueError):
                drift.sample_ticks(steps, interval)

    def test_weighted_rms_constant_and_partial_windows(self):
        self.assertEqual(drift.time_weighted_rms([0, 1, 10], [2, 2, 2], 0, 10), 2)
        self.assertEqual(drift.time_weighted_rms([0, 1, 10], [0, 0, 0], 2, 8), 0)
        # Squared sampled errors lie on y=t, so integral[2,6]/4 = 4.
        self.assertAlmostEqual(
            drift.time_weighted_rms([0, 1, 10], [0, 1, math.sqrt(10)], 2, 6), 2,
        )
        self.assertAlmostEqual(
            drift.time_weighted_rms([0, 1, 10], [0, 1, math.sqrt(10)], 0, 10), math.sqrt(5),
        )

    def test_summary_reports_sampled_thresholds_and_time_weighted_windows(self):
        ticks, errors = [0, 1, 2, 5, 10], [0, 1e-4, 1e-3, 0.02, 0.2]
        samples = [dict(tick=tick, precisions={"fp8": {
            field: dict(rmse=error, max_abs=3 * error, bias=-error)
            for field in drift.FIELDS
        }}) for tick, error in zip(ticks, errors)]
        summary = drift.summarize(samples, 10)["fp8"]
        for field in drift.FIELDS:
            result = summary[field]
            self.assertEqual(result['final'], samples[-1]['precisions']['fp8'][field])
            self.assertEqual(result['peak_sampled_rmse'], 0.2)
            self.assertAlmostEqual(result['peak_sampled_max_abs'], 0.6)
            self.assertEqual(result['first_sampled_rmse_threshold_tick'],
                             {'0.0001': 1, '0.001': 2, '0.01': 5, '0.1': 10})
            early, late = math.sqrt(2.55e-7), math.sqrt(0.03208)
            self.assertAlmostEqual(result['early_20_percent_time_weighted_rmse'], early)
            self.assertAlmostEqual(result['late_20_percent_time_weighted_rmse'], late)
            self.assertAlmostEqual(result['late_to_early_rmse_ratio'], late / early)
        for sample in samples:
            for field in drift.FIELDS:
                sample['precisions']['fp8'][field].update(rmse=0, max_abs=0)
        zero = drift.summarize(samples, 10)['fp8']['state']
        self.assertIsNone(zero['late_to_early_rmse_ratio'])
        self.assertTrue(all(tick is None for tick in zero['first_sampled_rmse_threshold_tick'].values()))


class DriftSnapshotTests(unittest.TestCase):
    def runners(self):
        base = torch.tensor([[0, 0.2, 1], [0.4, 0.6, 0.8]], dtype=torch.float32)
        values = (base, base - 0.4, base * 0.5)
        result = {}
        for backend, offset in (('reference', 0), ('native', 0.03)):
            for index, mode in enumerate(drift.MODES):
                result[backend, mode] = SimpleNamespace(
                    buffers=[value + offset + index * 0.01 for value in values],
                )
        return result

    def test_batched_snapshot_matches_independent_tensor_oracle(self):
        runners = self.runners()
        for backend in ('reference', 'native'):
            other = 'native' if backend == 'reference' else 'reference'
            actual = drift.snapshot(runners, backend, 23)
            self.assertEqual(actual['tick'], 23)
            for field, base in zip(drift.FIELDS, runners[backend, 'fp32'].buffers):
                expected_stats = drift.tensor_stats(base)
                for key, value in expected_stats.items():
                    self.assertAlmostEqual(actual['reference'][field][key], value, places=7)
            for mode in (*drift.MODES[1:], 'fp32_other_backend'):
                candidate = runners[other, 'fp32'] if mode == 'fp32_other_backend' else runners[backend, mode]
                rows = actual[mode] if mode == 'fp32_other_backend' else actual['precisions'][mode]
                for index, field in enumerate(drift.FIELDS):
                    expected = drift.error_stats(candidate.buffers[index], runners[backend, 'fp32'].buffers[index])
                    for key, value in expected.items():
                        self.assertAlmostEqual(rows[field][key], value, places=7)

    def test_snapshot_rejects_nonfinite_candidate_and_baseline(self):
        for key in (('reference', 'fp32'), ('reference', 'nvfp4'), ('native', 'fp32')):
            runners = self.runners()
            runners[key].buffers[1][0, 0] = float('nan')
            with self.assertRaisesRegex(RuntimeError, 'Non-finite reference state at tick 9'):
                drift.snapshot(runners, 'reference', 9)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for graph replay')
class DriftGraphTests(unittest.TestCase):
    @torch.inference_mode()
    def test_captured_full_chunks_preserve_state_across_eager_tails_and_reset(self):
        neurons, batch, chunk = 17, 2, 4
        graph = random_topology(neurons, 80, device='cuda', seed=11)
        model = DendriticDynamics.from_time_constants(
            Dynamics(torch.full((neurons,), 0.72, device='cuda'),
                     torch.full((neurons,), 0.8, device='cuda')),
            dt=1, tau_dendrite=4, tau_adaptation=20,
        )
        initial = torch.full((batch, neurons), 0.25, device='cuda')
        inputs = drift.drives('stable_drive', 0, chunk, neurons, batch, 'cuda')
        for mode in drift.MODES:
            factory = lambda: DendriticMatrixRuntime(graph, model, initial.clone(), precision=mode)
            runner = drift.GraphRollout(factory, inputs)
            oracle = factory()
            tick = 0
            for count in (chunk, 1, chunk, chunk - 1, 2, chunk):
                values = drift.drives('stable_drive', tick, chunk, neurons, batch, 'cuda')
                for dest, value in zip(inputs, values):
                    dest.copy_(value)
                runner.advance(count)
                for index in range(count):
                    oracle.step(inputs[0][index], dendritic_drive=inputs[1][index])
                for actual, field in zip(runner.buffers, drift.FIELDS):
                    torch.testing.assert_close(actual, getattr(oracle, field), atol=0, rtol=0)
                tick += count
            runner.reset()
            for actual, expected in zip(runner.buffers, runner.initial):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            report = runner.validate('stable_drive', neurons, batch)
            self.assertEqual(report['graph_eager_max_abs_error'], 0)


if __name__ == '__main__':
    unittest.main()
