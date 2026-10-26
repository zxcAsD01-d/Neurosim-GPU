from __future__ import annotations

from dataclasses import dataclass

import torch

from .dynamics import Dynamics
from .topology import SparseTopology


@dataclass(frozen=True)
class UniformErrorBudget:
    theta: torch.Tensor
    contraction_row_sum: torch.Tensor
    current_tolerance: torch.Tensor


@torch.no_grad()
def uniform_error_budget(
    topology: SparseTopology,
    dynamics: Dynamics,
    target_state_error: float,
) -> UniformErrorBudget:
    """Construct a conservative per-source threshold for a uniform state bound.

    This certificate uses the infinity norm of
    diag(alpha) + diag((1-alpha)*gain) @ abs(W). It is valid only when every
    row is contractive. Signed connections are handled through abs(W).
    """

    if target_state_error <= 0:
        raise ValueError("target_state_error must be positive")
    if topology.n_neurons != dynamics.n_neurons:
        raise ValueError("topology and dynamics neuron counts differ")

    device = topology.weight.device
    dtype = topology.weight.dtype
    src = topology.src.to(torch.int64)
    dst = topology.dst.to(torch.int64)
    absolute_weight = topology.weight.detach().abs()
    row_weight_sum = torch.zeros(
        topology.n_neurons,
        dtype=dtype,
        device=device,
    ).index_add_(0, dst, absolute_weight)
    beta_gain = dynamics.beta.detach() * dynamics.gain.detach()
    contraction = dynamics.alpha.detach() + beta_gain * row_weight_sum
    if bool(torch.any(contraction >= 1.0)):
        maximum = float(contraction.max())
        raise ValueError(
            "no infinite-horizon infinity-norm certificate: "
            f"maximum contraction row sum is {maximum:.6g}"
        )

    numerator = target_state_error * (1.0 - contraction)
    current_tolerance = torch.where(
        beta_gain > 0,
        numerator / beta_gain,
        torch.full_like(beta_gain, torch.inf),
    )
    in_degree = torch.bincount(dst, minlength=topology.n_neurons).clamp_min(1)
    denominator = in_degree[dst] * absolute_weight
    edge_threshold = torch.where(
        denominator > 0,
        current_tolerance[dst] / denominator,
        torch.full_like(absolute_weight, torch.inf),
    )
    theta = torch.full(
        (topology.n_neurons,),
        torch.inf,
        dtype=dtype,
        device=device,
    )
    theta.scatter_reduce_(0, src, edge_threshold, reduce="amin", include_self=True)
    return UniformErrorBudget(
        theta=theta,
        contraction_row_sum=contraction,
        current_tolerance=current_tolerance,
    )

