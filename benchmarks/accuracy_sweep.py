from __future__ import annotations

import argparse

import torch

from neurosim_gpu import DenseOracle, Dynamics, HybridPELSRuntime, SparseTopology, random_topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, default=128)
    parser.add_argument("--edges", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    base = random_topology(
        args.neurons,
        args.edges,
        device=device,
        dtype=dtype,
        seed=args.seed,
        weight_scale=0.12,
    )
    source = base.src.detach().clone()
    destination = base.dst.detach().clone()
    base_weight = base.weight.detach().clone()
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    initial = torch.rand(
        args.batch,
        args.neurons,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    dynamics = Dynamics(
        alpha=torch.full((args.neurons,), 0.95, dtype=dtype, device=device),
        gain=torch.full((args.neurons,), 0.75, dtype=dtype, device=device),
    )

    print(
        "theta       max_state_error  grad_cosine  grad_rel_error  "
        "active_edge_fraction"
    )
    for theta in args.thresholds:
        edge_parameter = base_weight.clone().requires_grad_(True)
        topology = SparseTopology.from_edges(
            args.neurons,
            source,
            destination,
            edge_parameter,
        )
        dense = DenseOracle(topology, dynamics, initial.clone())
        pels = HybridPELSRuntime(
            topology,
            dynamics,
            initial.clone(),
            theta=theta,
            propagation="matrix",
        )
        max_error = 0.0
        active_edges = 0
        for _ in range(args.steps):
            reference = dense.step(0.15)
            approximation, stats = pels.step(0.15, collect_stats=True)
            max_error = max(
                max_error,
                float((approximation - reference).detach().abs().max()),
            )
            active_edges += stats.active_edges

        dense_loss = dense.state.square().mean()
        pels_loss = pels.state.square().mean()
        dense_gradient = torch.autograd.grad(
            dense_loss,
            edge_parameter,
            retain_graph=True,
        )[0]
        pels_gradient = torch.autograd.grad(pels_loss, edge_parameter)[0]
        cosine = torch.nn.functional.cosine_similarity(
            dense_gradient,
            pels_gradient,
            dim=0,
            eps=1e-30,
        )
        relative_error = (
            (pels_gradient - dense_gradient).norm()
            / dense_gradient.norm().clamp_min(1e-30)
        )
        possible_edges = topology.n_edges * args.batch * args.steps
        active_fraction = active_edges / max(possible_edges, 1)
        print(
            f"{theta:<11.4g} {max_error:<16.7g} "
            f"{float(cosine):<12.7f} {float(relative_error):<15.7g} "
            f"{active_fraction:.7f}"
        )


if __name__ == "__main__":
    main()

