"""Long-horizon fixed-weight precision drift; time is measured in model ticks.

CUDA graphs replay the existing runtime operations. Every path is checked against
eager execution before a rollout; no model equations or precision are substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch

from neurosim_gpu import DendriticDynamics, DendriticMatrixRuntime, Dynamics, random_topology
from neurosim_gpu.native import load_native_extension
from neurosim_gpu.native_compartmental import NativeDendriticMatrixRuntime

FIELDS = ('state', 'dendritic_state', 'adaptation_state')
MODES = ('fp32', 'fp16', 'fp8', 'nvfp4')
SCENARIOS = ('stable_drive', 'slow_adaptation', 'recurrent_stress')
OPTIONS = {'fp32': {}, 'fp16': {'weight_dtype': torch.float16},
           'fp8': {'weight_format': 'fp8'}, 'nvfp4': {'weight_format': 'nvfp4'}}


def scenario_parameters(name):
    if name not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {name}')
    result = dict(weight_scale=0.15, alpha=0.72, gain=0.8, tau_dendrite=4.0,
                  tau_adaptation=20.0, coupling=0.45, adaptation_strength=0.15)
    if name == 'slow_adaptation':
        result.update(tau_adaptation=5000.0, adaptation_strength=0.6)
    if name == 'recurrent_stress':
        result.update(weight_scale=1.7, alpha=0.9, gain=1.3, tau_dendrite=8.0,
                      tau_adaptation=500.0, coupling=0.3, adaptation_strength=0.6)
    return result


def sample_ticks(steps, interval):
    if steps < 1 or interval < 1:
        raise ValueError('steps and sample interval must be positive')
    ticks = {0, steps, *range(interval, steps + 1, interval)}
    for power in range(int(math.log10(steps)) + 1):
        ticks.update(multiplier * 10 ** power for multiplier in (1, 2, 5)
                     if multiplier * 10 ** power <= steps)
    # Explicit input boundaries, including the tick on either side.
    for start in range(0, steps + 1, 24000):
        ticks.update(t for boundary in (start, start + 8000)
                     for t in (boundary - 1, boundary, boundary + 1) if 0 <= t <= steps)
    return sorted(ticks)


def drives(name, start, count, neurons, batch, device):
    ticks = torch.arange(start, start + count, device=device, dtype=torch.float32)[:, None, None]
    neurons_axis = torch.arange(neurons, device=device, dtype=torch.float32)[None, None, :]
    batch_axis = torch.arange(batch, device=device, dtype=torch.float32)[None, :, None]
    spatial = 0.039 * neurons_axis + 0.7 * batch_axis
    if name == 'recurrent_stress':
        soma = (0.13 + 0.08 * torch.sin(spatial)).expand(count, batch, neurons)
        dendritic = torch.zeros_like(soma)
    else:
        soma = 0.20 + 0.05 * torch.sin(ticks * (2 * math.pi / 257) + spatial)
        dendritic = (0.025 + 0.015 * torch.sin(ticks * (2 * math.pi / 503) + spatial))
        if name == 'slow_adaptation':
            gate = ((ticks % 24000) < 8000).float()
            soma = gate * (soma + 0.15)
            dendritic = gate * dendritic
    return soma.contiguous(), dendritic.contiguous()


class GraphRollout:
    """Closed state feedback around a graph of existing eager runtime calls."""

    def __init__(self, factory, inputs):
        self.factory, self.inputs = factory, inputs
        self.runtime = factory()
        self.buffers = [getattr(self.runtime, field) for field in FIELDS]
        self.initial = [value.clone() for value in self.buffers]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.runtime.step(inputs[0][0], dendritic_drive=inputs[1][0])
        torch.cuda.current_stream().wait_stream(stream)
        self.reset()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for index in range(inputs[0].shape[0]):
                self.runtime.step(inputs[0][index], dendritic_drive=inputs[1][index])
            self._collect()
        self.reset()

    def _attach(self):
        for field, value in zip(FIELDS, self.buffers):
            setattr(self.runtime, field, value)

    def _collect(self):
        for field, value in zip(FIELDS, self.buffers):
            value.copy_(getattr(self.runtime, field))

    def reset(self):
        for buffer, value in zip(self.buffers, self.initial):
            buffer.copy_(value)
        self._attach()

    def advance(self, count):
        if count == self.inputs[0].shape[0]:
            self.graph.replay()
        else:
            for index in range(count):
                self.runtime.step(self.inputs[0][index], dendritic_drive=self.inputs[1][index])
            self._collect()
            self._attach()

    def validate(self, name, neurons, batch, chunks=3):
        oracle = self.factory()
        errors = []
        chunk = self.inputs[0].shape[0]
        for index in range(chunks):
            values = drives(name, index * chunk, chunk, neurons, batch, self.buffers[0].device)
            for dest, value in zip(self.inputs, values):
                dest.copy_(value)
            self.advance(chunk)
            for tick in range(chunk):
                oracle.step(self.inputs[0][tick], dendritic_drive=self.inputs[1][tick])
            errors.extend((actual - getattr(oracle, field)).abs().max()
                          for actual, field in zip(self.buffers, FIELDS))
        maximum = float(torch.stack(errors).max())
        if maximum != 0:
            raise RuntimeError(f'CUDA graph does not match eager execution: {maximum}')
        # Exercise eager tails followed by captured replay as well.
        self.advance(1)
        oracle.step(self.inputs[0][0], dendritic_drive=self.inputs[1][0])
        self.advance(chunk)
        for tick in range(chunk):
            oracle.step(self.inputs[0][tick], dendritic_drive=self.inputs[1][tick])
        for actual, field in zip(self.buffers, FIELDS):
            torch.testing.assert_close(actual, getattr(oracle, field), rtol=0, atol=0)
        self.reset()
        return {'graph_eager_max_abs_error': maximum, 'verified_ticks': chunks * chunk + 1 + chunk}


def tensor_stats(values):
    return dict(mean=float(values.mean()), std=float(values.std(unbiased=False)),
                min=float(values.min()), max=float(values.max()),
                at_zero_fraction=float((values == 0).float().mean()),
                at_one_fraction=float((values == 1).float().mean()))


def error_stats(actual, expected):
    error = actual - expected
    return dict(mae=float(error.abs().mean()), rmse=float(error.square().mean().sqrt()),
                max_abs=float(error.abs().max()), bias=float(error.mean()),
                population_std_difference=float(actual.std(unbiased=False) - expected.std(unbiased=False)),
                **{'candidate_' + key: value for key, value in tensor_stats(actual).items()})


def snapshot(runners, backend, tick):
    other = 'reference' if backend == 'native' else 'native'
    values = torch.stack([torch.stack(runners[backend, mode].buffers) for mode in MODES]
                         + [torch.stack(runners[other, 'fp32'].buffers)])
    dims = (-2, -1)
    means, stds = values.mean(dim=dims), values.std(dim=dims, unbiased=False)
    statistics = torch.stack((means, stds, values.amin(dim=dims), values.amax(dim=dims),
        (values == 0).float().mean(dim=dims), (values == 1).float().mean(dim=dims),
        torch.isfinite(values).all(dim=-1).all(dim=-1).float()), dim=-1)
    errors = values[1:] - values[0]
    diagnostics = torch.stack((errors.abs().mean(dim=dims),
        errors.square().mean(dim=dims).sqrt(), errors.abs().amax(dim=dims),
        errors.mean(dim=dims), stds[1:] - stds[0]), dim=-1)
    # One host transfer for all diagnostics, rather than one sync per scalar.
    payload = torch.cat((statistics.flatten(), diagnostics.flatten())).cpu().tolist()
    statistics = torch.tensor(payload[:5 * 3 * 7]).reshape(5, 3, 7).tolist()
    diagnostics = torch.tensor(payload[5 * 3 * 7:]).reshape(4, 3, 5).tolist()
    if not all(row[-1] == 1 for mode in statistics for row in mode):
        raise RuntimeError(f'Non-finite {backend} state at tick {tick}')
    stat_names = ('mean', 'std', 'min', 'max', 'at_zero_fraction', 'at_one_fraction')
    error_names = ('mae', 'rmse', 'max_abs', 'bias', 'population_std_difference')
    result = dict(tick=tick, reference={field: dict(zip(stat_names, row))
        for field, row in zip(FIELDS, statistics[0])}, precisions={})
    for index, mode in enumerate((*MODES[1:], 'fp32_other_backend')):
        mode_stats = {field: {**dict(zip(error_names, diagnostics[index][j])),
                              **dict(zip(('candidate_' + key for key in stat_names), statistics[index+1][j]))}
                      for j, field in enumerate(FIELDS)}
        if mode == 'fp32_other_backend':
            result[mode] = mode_stats
        else:
            result['precisions'][mode] = mode_stats
    return result


def time_weighted_rms(ticks, errors, start, end):
    total = 0.0
    for i in range(1, len(ticks)):
        lo, hi = max(start, ticks[i-1]), min(end, ticks[i])
        if hi <= lo:
            continue
        duration = ticks[i] - ticks[i-1]
        y0, y1 = errors[i-1] ** 2, errors[i] ** 2
        left = y0 + (y1 - y0) * ((lo - ticks[i-1]) / duration)
        right = y0 + (y1 - y0) * ((hi - ticks[i-1]) / duration)
        total += (left + right) * 0.5 * (hi - lo)
    return math.sqrt(total / (end - start))


def summarize(samples, steps):
    ticks = [sample['tick'] for sample in samples]
    output = {}
    for mode in samples[0]['precisions']:
        output[mode] = {}
        for field in FIELDS:
            rows = [sample['precisions'][mode][field] for sample in samples]
            errors = [row['rmse'] for row in rows]
            early = time_weighted_rms(ticks, errors, 0, steps * 0.2)
            late = time_weighted_rms(ticks, errors, steps * 0.8, steps)
            output[mode][field] = dict(
                final=rows[-1], peak_sampled_rmse=max(errors),
                peak_sampled_max_abs=max(row['max_abs'] for row in rows),
                early_20_percent_time_weighted_rmse=early, late_20_percent_time_weighted_rmse=late,
                late_to_early_rmse_ratio=late / early if early else None,
                first_sampled_rmse_threshold_tick={str(threshold): next(
                    (tick for tick, error in zip(ticks, errors) if error >= threshold), None)
                    for threshold in (1e-4, 1e-3, 1e-2, 1e-1)},
            )
    return output


@torch.inference_mode()
def run_case(args, scenario, seed, extension):
    params = scenario_parameters(scenario)
    graph = random_topology(args.neurons, args.edges, device='cuda', seed=seed,
                            weight_scale=params['weight_scale'])
    model = DendriticDynamics.from_time_constants(
        Dynamics(torch.full((args.neurons,), params['alpha'], device='cuda'),
                 torch.full((args.neurons,), params['gain'], device='cuda')),
        dt=1, **{key: params[key] for key in ('tau_dendrite', 'tau_adaptation',
                                            'coupling', 'adaptation_strength')})
    initial = torch.full((args.batch, args.neurons), 0.25, device='cuda')
    inputs = drives(scenario, 0, args.chunk, args.neurons, args.batch, 'cuda')
    runners, cases = {}, []
    for backend in ('native', 'reference'):
        validation, execution = {}, {}
        for mode in MODES:
            def factory(backend=backend, mode=mode):
                if backend == 'native':
                    return NativeDendriticMatrixRuntime(graph, model, initial.clone(),
                                                        extension=extension, **OPTIONS[mode])
                return DendriticMatrixRuntime(graph, model, initial.clone(), precision=mode)
            runner = GraphRollout(factory, inputs)
            validation[mode] = runner.validate(scenario, args.neurons, args.batch)
            execution[mode] = runner.runtime.precision_execution
            runners[backend, mode] = runner
        cases.append(dict(scenario=scenario, backend=backend, seed=seed, parameters=params,
                          neurons=args.neurons, candidate_edges=args.edges, edges=graph.n_edges,
                          batch=args.batch, execution=execution, replay_validation=validation,
                          samples=[]))
    tick = 0
    started = time.perf_counter()
    for target in sample_ticks(args.steps, args.sample_interval):
        while tick < target:
            count = min(args.chunk, target - tick)
            values = drives(scenario, tick, args.chunk, args.neurons, args.batch, 'cuda')
            for dest, value in zip(inputs, values):
                dest.copy_(value)
            for runner in runners.values():
                runner.advance(count)
            tick += count
        for case in cases:
            backend = case['backend']
            case['samples'].append(snapshot(runners, backend, tick))
        if tick and tick % 10000 == 0:
            print(f'{scenario} seed={seed} tick={tick}/{args.steps}', flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    for case in cases:
        case['summary'] = summarize(case['samples'], args.steps)
        case['wall_seconds_including_both_backends_and_diagnostics'] = elapsed
    print(f'completed {scenario} seed={seed} in {elapsed:.1f}s', flush=True)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--neurons', type=int, default=1024)
    parser.add_argument('--edges', type=int, default=16384)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--sample-interval', type=int, default=256)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 47])
    parser.add_argument('--scenarios', choices=SCENARIOS, nargs='+', default=list(SCENARIOS))
    parser.add_argument('--output', type=Path, default=Path('results/precision_drift.json'))
    args = parser.parse_args()
    if min(args.steps, args.neurons, args.batch, args.chunk, args.sample_interval) < 1 or args.edges < 0:
        parser.error('sizes must be positive and edges non-negative')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the two-backend drift comparison')
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    tracked = ['benchmarks/precision_drift.py', 'src/neurosim_gpu/precision.py',
               'src/neurosim_gpu/compartmental.py', 'src/neurosim_gpu/native_compartmental.py',
               'src/neurosim_gpu/topology.py', 'src/neurosim_gpu/dynamics.py',
               'csrc/compartmental.cu', 'csrc/bindings.cpp']
    output = dict(environment=dict(python=platform.python_version(), torch=torch.__version__,
                  cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  compute_capability=list(torch.cuda.get_device_capability())),
                  protocol=dict(steps=args.steps, seeds=args.seeds, scenarios=args.scenarios,
                    sample_interval=args.sample_interval, chunk=args.chunk,
                    unit='model tick; no biological millisecond calibration',
                    weights='frozen; inference drift, not optimizer/training drift',
                    comparison='same backend FP32 baseline, graph, initial state and identical inputs',
                    states='FP32 for all modes',
                    peaks='sampled peaks only; no all-tick error bound',
                    finite_checks='all states at sample checkpoints',
                    timing='includes both backends and diagnostics, not a speed benchmark',
                    execution='CUDA graph replay verified bitwise against eager runtime before each case'),
                  source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in tracked},
                  cases=[])
    extension = load_native_extension()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for scenario in args.scenarios:
        for seed in args.seeds:
            output['cases'].extend(run_case(args, scenario, seed, extension))
            args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
