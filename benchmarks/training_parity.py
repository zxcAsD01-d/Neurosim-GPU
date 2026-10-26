from __future__ import annotations

import argparse
import gc
import statistics
import time

import torch

from neurosim_gpu import Dynamics, random_topology


def sparse_weight(topology, values: torch.Tensor) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        topology.in_rowptr,
        topology.in_col,
        values,
        size=(topology.n_neurons, topology.n_neurons),
        device=values.device,
        dtype=values.dtype,
        check_invariants=False,
    )


def propagate(weight: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(weight, state.T).T


def standard_rnn(
    weight: torch.Tensor,
    dynamics: Dynamics,
    initial: torch.Tensor,
    steps: int,
    drive: float,
) -> torch.Tensor:
    del dynamics
    state = initial
    for _ in range(steps):
        state = torch.tanh(propagate(weight, state) + drive)
    return state


def matrix_dynamics(
    weight: torch.Tensor,
    dynamics: Dynamics,
    initial: torch.Tensor,
    steps: int,
    drive: float,
) -> torch.Tensor:
    state = initial
    for _ in range(steps):
        state = dynamics.advance(state, propagate(weight, state), drive)
    return state


def persistent_dynamics(
    weight: torch.Tensor,
    dynamics: Dynamics,
    initial: torch.Tensor,
    steps: int,
    drive: float,
) -> torch.Tensor:
    state = initial
    emitted = initial
    current = propagate(weight, emitted)
    for _ in range(steps):
        next_state = dynamics.advance(state, current, drive)
        emitted_delta = next_state - emitted
        current = current + propagate(weight, emitted_delta)
        emitted = next_state
        state = next_state
    return state


def native_matrix_dynamics(
    weight: torch.Tensor,
    dynamics: Dynamics,
    initial: torch.Tensor,
    steps: int,
    drive: float,
) -> torch.Tensor:
    from neurosim_gpu.native import native_advance_training

    state = initial
    for _ in range(steps):
        state = native_advance_training(
            state,
            propagate(weight, state),
            dynamics,
            drive,
        )
    return state


def measure(
    rollout,
    topology,
    dynamics: Dynamics,
    base_values: torch.Tensor,
    initial: torch.Tensor,
    steps: int,
    drive: float,
    samples: int,
) -> tuple[float, float]:
    timings = []
    peak_deltas = []
    for _ in range(samples):
        values = base_values.clone().requires_grad_(True)
        weight = sparse_weight(topology, values)
        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        final = rollout(weight, dynamics, initial, steps, drive)
        loss = final.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
        peak_deltas.append(
            (torch.cuda.max_memory_allocated() - start_memory) / 2**20
        )
        del final, loss, weight, values
        gc.collect()
        torch.cuda.empty_cache()
    return statistics.median(timings), statistics.median(peak_deltas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, default=4096)
    parser.add_argument("--edges", type=int, default=100_000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--drive", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    topology = random_topology(
        args.neurons,
        args.edges,
        device=device,
        seed=args.seed,
        weight_scale=0.12,
    )
    base_values = topology.in_weight.detach()
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    initial = torch.rand(
        args.batch,
        args.neurons,
        device=device,
        generator=generator,
    )
    dynamics = Dynamics(
        alpha=torch.full((args.neurons,), 0.95, device=device),
        gain=torch.full((args.neurons,), 0.8, device=device),
    )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    rollouts = {
        "same-param-rnn": standard_rnn,
        "matrix-dynamics": matrix_dynamics,
        "persistent-t0": persistent_dynamics,
    }
    if args.native:
        from neurosim_gpu.native import load_native_extension

        load_native_extension(verbose=False)
        rollouts["native-pointwise"] = native_matrix_dynamics
    results: dict[str, tuple[float, float]] = {}
    print(
        f"device={torch.cuda.get_device_name(device)} N={args.neurons:,} "
        f"M={topology.n_edges:,} B={args.batch} T={args.steps}"
    )
    print("training            forward_backward_ms   peak_graph_MiB")
    for name, rollout in rollouts.items():
        result = measure(
            rollout,
            topology,
            dynamics,
            base_values,
            initial,
            args.steps,
            args.drive,
            args.samples,
        )
        results[name] = result
        print(f"{name:19} {result[0]:19.3f} {result[1]:16.1f}")
    baseline = results["same-param-rnn"][0]
    print(
        "parity              "
        f"matrix/rnn={results['matrix-dynamics'][0] / baseline:.4f} "
        f"persistent/rnn={results['persistent-t0'][0] / baseline:.4f}"
        + (
            f" native/rnn={results['native-pointwise'][0] / baseline:.4f}"
            if args.native
            else ""
        )
    )


if __name__ == "__main__":
    main()
