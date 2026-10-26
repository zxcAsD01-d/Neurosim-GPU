"""Cart-pole control probes: LQR expert, behavior cloning, sparse brain, plasticity.

The experiment asks three questions about the project's sparse rate model:

1. *Can it act?*  A behavior-cloned sparse recurrent core and a dense MLP are
   cloned from an LQR expert and evaluated in closed loop on the classic
   cart-pole swing-up-free balancing task.
2. *Is it robust?*  One-factor-at-a-time sweeps over sensor noise, action
   delay, pole and mass scale, and (in hard mode) an impulse kick.
3. *Does local three-factor plasticity help after an unannounced plant change?*
   A brain is cloned, the plant degrades mid-run, and frozen and online
   variants are compared under a reward-modulated eligibility trace on the
   recurrent edges only.  Eligibility is episode-scoped: the trace decays by
   0.9 per recurrent tick, the weight update runs once at each episode
   boundary and the trace is then cleared; the dopamine signal is the mean
   reward proxy minus a 0.9-decay running baseline.  The readout stays frozen.

Everything is pure PyTorch on CPU.  Each invocation prints exactly one
``RESULT {json}`` line on stdout; progress goes to stderr.

Physics: gym-style cart-pole (masscart=1.0, total pole mass=0.1, half-length
0.5, force 10 N, gravity 9.8, dt 0.02, +/-12 deg termination), integrated with
RK4 in float64.  The continuous-time model is the standard uniform-rod
cart-pole, so the same ``4/3`` denominator used by gym appears below.

``hard=true`` switches to a harder protocol: theta is drawn from
U(-0.15, 0.15), the other states from U(-0.1, 0.1), the actuator is limited to
6 N, and a seeded +-3 N kick is applied for three steps at a per-episode step
in [60, 140].  ``eval=online`` always uses the hard protocol and degrades the
actuator from 6 N to 1 N at episode 10.  That shift was chosen empirically:
pole-length and pole-mass shifts (x3..x10) and force >= 1.5 N are absorbed by
the bang-bang policy (frozen survival stays near 200), while 1 N produces a
real frozen drop (~200 -> ~125 mean over seeds 0-2).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

# The venv has no numpy; torch emits a one-off warning about it at import time.
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    RESULTS,
    Timer,
    emit_result,
    ensure_dir,
    measure_latency,
    percentile,
    seed_everything,
    trainable_parameters,
)
from neurosim_gpu import (
    Dynamics,
    dopamine_gated_weight_update,
    random_topology,
    update_eligibility_trace,
)

DEVICE = "cpu"
ACTIONS = 3
CHECKPOINT_FORMAT = 1
CHECKPOINTS = RESULTS / "checkpoints"
THETA_DEGREES = 12.0
SUCCESS_STEPS = 195
EDGE_BOUND = 1.5

DEFAULTS: dict[str, Any] = {
    # Public config surface (every key optional).
    "policy": "mlp",  # lqr | mlp | brain
    "eval": "closed_loop",  # closed_loop | robustness | online
    "neurons": 128,
    "degree": 8,
    "steps": 2,
    "hidden": 64,
    "episodes": 100,
    "bc_samples": 20000,
    "variant": "online",  # frozen | online
    "noise": 0.0,
    "delay": 0,
    "length_scale": 1.0,
    "mass_scale": 1.0,
    "hard": False,  # larger init, weaker actuator, seeded impulse
    # Hard protocol parameters.
    "hard_force_mag": 6.0,
    "hard_theta_range": 0.15,
    "hard_other_range": 0.1,
    "impulse_strength": 3.0,
    "impulse_first": 60,
    "impulse_last": 140,
    "impulse_steps": 3,
    # Knobs that are part of the reproducible recipe, not the public sweep.
    "alpha": 0.72,
    "gain": 1.05,
    "weight_scale": 0.15,
    "sensory_fanout": 8,
    "sensory_scale": 1.5,
    "initial_state": "sigmoid",
    "epochs": 12,
    "batch": 128,
    "lr": 3e-3,
    "noise_prob": 0.1,
    "deadzone": 0.05,
    "change_episode": 10,
    "online_change": "force",  # length | force
    "change_length_scale": 3.0,
    "change_force_mag": 1.0,
    "eligibility_decay": 0.9,
    "online_lr": 1e-3,
    "baseline_decay": 0.9,
    "latency_iterations": 25,
}

CHECKPOINT_KEYS = (
    "policy",
    "neurons",
    "degree",
    "steps",
    "hidden",
    "bc_samples",
    "epochs",
    "batch",
    "lr",
    "noise_prob",
    "alpha",
    "gain",
    "weight_scale",
    "sensory_fanout",
    "sensory_scale",
    "initial_state",
    "hard",
    "hard_force_mag",
    "hard_theta_range",
    "hard_other_range",
    "impulse_strength",
    "impulse_first",
    "impulse_last",
    "impulse_steps",
)

_INT_KEYS = (
    "neurons",
    "degree",
    "steps",
    "hidden",
    "episodes",
    "bc_samples",
    "delay",
    "epochs",
    "batch",
    "change_episode",
    "latency_iterations",
    "sensory_fanout",
    "impulse_first",
    "impulse_last",
    "impulse_steps",
)
_FLOAT_KEYS = (
    "noise",
    "length_scale",
    "mass_scale",
    "alpha",
    "gain",
    "weight_scale",
    "sensory_scale",
    "lr",
    "noise_prob",
    "deadzone",
    "change_length_scale",
    "change_force_mag",
    "hard_force_mag",
    "hard_theta_range",
    "hard_other_range",
    "impulse_strength",
    "eligibility_decay",
    "online_lr",
    "baseline_decay",
)
_BOOL_KEYS = ("hard",)


def log(message: str) -> None:
    print(f"[cartpole] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Physics: gym-style cart-pole, RK4 in float64
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plant:
    """Cart-pole parameters and termination bounds.

    ``length`` is the *half* length and ``masspole`` the *total* pole mass, as
    in gym's CartPole.  ``length_scale``/``mass_scale`` perturb the plant
    without touching the task definition.
    """

    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_mag: float = 10.0
    gravity: float = 9.8
    dt: float = 0.02
    x_threshold: float = 2.4
    theta_threshold: float = THETA_DEGREES * math.pi / 180.0
    max_steps: int = 200
    length_scale: float = 1.0
    mass_scale: float = 1.0

    @property
    def pole_length(self) -> float:
        return self.length * self.length_scale

    @property
    def pole_mass(self) -> float:
        return self.masspole * self.mass_scale

    @property
    def total_mass(self) -> float:
        return self.masscart + self.pole_mass

    @property
    def polemass_length(self) -> float:
        return self.pole_mass * self.pole_length


@dataclass(frozen=True)
class Protocol:
    """Episode initial-state distribution and the optional disturbance kick.

    The nominal protocol samples all four state variables from U(-0.05, 0.05).
    The hard protocol widens theta to U(-0.15, 0.15) and the rest to
    U(-0.1, 0.1), and can add a seeded +-``impulse_strength`` N kick applied
    for ``impulse_steps`` steps starting at a per-episode step in
    ``[impulse_first, impulse_last]``.
    """

    theta_range: float = 0.05
    other_range: float = 0.05
    impulse: bool = False
    impulse_strength: float = 3.0
    impulse_first: int = 60
    impulse_last: int = 140
    impulse_steps: int = 3

    @property
    def scale(self) -> torch.Tensor:
        return torch.tensor(
            [self.other_range, self.other_range, self.theta_range, self.other_range],
            dtype=torch.float64,
        )


NOMINAL_PROTOCOL = Protocol()


def make_protocol(
    config: dict[str, Any],
    *,
    impulse: bool,
    hard: bool | None = None,
) -> Protocol:
    """Build the protocol for a run; the impulse only exists in hard mode."""

    is_hard = bool(config["hard"]) if hard is None else bool(hard)
    if not is_hard:
        return NOMINAL_PROTOCOL
    if is_hard and impulse:
        return Protocol(
            theta_range=float(config["hard_theta_range"]),
            other_range=float(config["hard_other_range"]),
            impulse=True,
            impulse_strength=float(config["impulse_strength"]),
            impulse_first=int(config["impulse_first"]),
            impulse_last=int(config["impulse_last"]),
            impulse_steps=int(config["impulse_steps"]),
        )
    return Protocol(
        theta_range=float(config["hard_theta_range"]),
        other_range=float(config["hard_other_range"]),
        impulse=False,
    )


def make_plant(
    config: dict[str, Any],
    *,
    hard: bool,
    force_mag: float | None = None,
    length_scale: float | None = None,
    mass_scale: float | None = None,
) -> Plant:
    resolved_force = (
        float(force_mag)
        if force_mag is not None
        else (float(config["hard_force_mag"]) if hard else 10.0)
    )
    return Plant(
        force_mag=resolved_force,
        length_scale=float(config["length_scale"]) if length_scale is None else float(length_scale),
        mass_scale=float(config["mass_scale"]) if mass_scale is None else float(mass_scale),
    )


def sample_initial_states(
    episodes: int,
    generator: torch.Generator,
    protocol: Protocol,
) -> torch.Tensor:
    uniform = torch.rand(episodes, 4, generator=generator, dtype=torch.float64) * 2.0 - 1.0
    return uniform * protocol.scale


def sample_impulse_schedule(
    episodes: int,
    generator: torch.Generator,
    protocol: Protocol,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-episode kick start step and sign (deterministic from the generator)."""

    first = int(protocol.impulse_first)
    last = max(first, int(protocol.impulse_last))
    start = torch.randint(first, last + 1, (episodes,), generator=generator)
    draw = torch.rand(episodes, generator=generator, dtype=torch.float64)
    sign = torch.where(
        draw < 0.5,
        torch.full((episodes,), -1.0, dtype=torch.float64),
        torch.ones(episodes, dtype=torch.float64),
    )
    return start, sign


def _derivative(
    state: torch.Tensor,
    force: torch.Tensor,
    plant: Plant,
) -> torch.Tensor:
    """d/dt [x, x_dot, theta, theta_dot] for a uniform-rod pole on a cart."""

    _, x_dot, theta, theta_dot = state.unbind(-1)
    total = plant.total_mass
    length = plant.pole_length
    ml = plant.polemass_length
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    denominator = length * (4.0 / 3.0 - plant.pole_mass * cos_t * cos_t / total)
    temp = (force + ml * theta_dot * theta_dot * sin_t) / total
    theta_acc = (plant.gravity * sin_t - cos_t * temp) / denominator
    x_acc = temp - ml * theta_acc * cos_t / total
    return torch.stack((x_dot, x_acc, theta_dot, theta_acc), dim=-1)


def rk4_step(
    state: torch.Tensor,
    force: torch.Tensor,
    plant: Plant,
) -> torch.Tensor:
    """One classic RK4 step of the cart-pole ODE."""

    dt = plant.dt
    k1 = _derivative(state, force, plant)
    k2 = _derivative(state + 0.5 * dt * k1, force, plant)
    k3 = _derivative(state + 0.5 * dt * k2, force, plant)
    k4 = _derivative(state + dt * k3, force, plant)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def action_force(actions: torch.Tensor, plant: Plant) -> torch.Tensor:
    """Map discrete action indices {0,1,2} to forces {-10, 0, +10} N."""

    return (actions.to(torch.float64) - 1.0) * plant.force_mag


def total_energy(state: torch.Tensor, plant: Plant) -> torch.Tensor:
    """Mechanical energy (J) of the uniform-rod cart-pole; conserved at F=0."""

    _, x_dot, theta, theta_dot = state.unbind(-1)
    mass = plant.pole_mass
    length = plant.pole_length
    cart = 0.5 * plant.masscart * x_dot * x_dot
    pole = 0.5 * mass * (
        x_dot * x_dot
        + 2.0 * length * x_dot * theta_dot * torch.cos(theta)
        + (4.0 / 3.0) * length * length * theta_dot * theta_dot
    )
    potential = mass * plant.gravity * length * torch.cos(theta)
    return cart + pole + potential


# ---------------------------------------------------------------------------
# Expert: discrete LQR (DARE by iteration), with a PD fallback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateFeedbackPolicy:
    """``u = -K s`` mapped to the three discrete forces with a dead zone."""

    gain: torch.Tensor  # [4], float64
    force_mag: float
    deadzone: float = 0.05
    kind: str = "lqr"

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        state = observation.to(torch.float64)
        control = -(state @ self.gain)
        action = torch.zeros(control.shape, dtype=torch.long)
        action[control > 0.0] = 2
        action[control.abs() < self.deadzone] = 1
        return action


def linearize_upright(plant: Plant) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler-discretized (A, B) of the cart-pole about x=0, theta=0."""

    total = plant.total_mass
    length = plant.pole_length
    ml = plant.polemass_length
    denominator = length * (4.0 / 3.0 - plant.pole_mass / total)
    A = torch.zeros(4, 4, dtype=torch.float64)
    B = torch.zeros(4, 1, dtype=torch.float64)
    A[0, 1] = 1.0
    A[1, 2] = -(ml * plant.gravity) / (total * denominator)
    B[1, 0] = 1.0 / total + ml / (total * total * denominator)
    A[2, 3] = 1.0
    A[3, 2] = plant.gravity / denominator
    B[3, 0] = -1.0 / (total * denominator)
    return A, B


def discretize_upright(plant: Plant) -> tuple[torch.Tensor, torch.Tensor]:
    A, B = linearize_upright(plant)
    return torch.eye(4, dtype=torch.float64) + A * plant.dt, B * plant.dt


def solve_dare(
    plant: Plant,
    *,
    q_diag: tuple[float, float, float, float] = (1.0, 1.0, 10.0, 1.0),
    r: float = 0.1,
    max_iterations: int = 10000,
    tolerance: float = 1e-13,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Iterate ``P <- Q + A'PA - A'PB(R+B'PB)^-1 B'PA`` to convergence."""

    A, B = discretize_upright(plant)
    Q = torch.diag(torch.tensor(q_diag, dtype=torch.float64))
    R = torch.tensor([[r]], dtype=torch.float64)
    P = Q.clone()
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        BtP = B.T @ P
        gain = torch.linalg.solve(R + BtP @ B, BtP @ A)
        updated = Q + A.T @ P @ A - A.T @ P @ B @ gain
        if torch.allclose(updated, P, atol=tolerance, rtol=0.0):
            P = updated
            break
        P = updated
    K = torch.linalg.solve(R + B.T @ P @ B, B.T @ P @ A).reshape(-1)
    return K, P, iterations


_EXPERT_CACHE: dict[tuple[Any, ...], tuple[StateFeedbackPolicy, dict[str, Any], str]] = {}


def build_expert(plant: Plant, seed: int = 0) -> tuple[StateFeedbackPolicy, dict[str, Any], str]:
    """Return the LQR expert, its acceptance stats, and a note.

    Acceptance: mean survival >= 190/200 and success >= 0.95 over 100 seeded
    episodes.  If the Riccati-based controller misses that bar the hand-tuned
    PD gains are used instead (and the note says so).
    """

    key = (
        plant.masscart,
        plant.masspole,
        plant.length,
        plant.force_mag,
        plant.gravity,
        plant.dt,
        plant.x_threshold,
        plant.theta_threshold,
        plant.max_steps,
        seed,
    )
    if key in _EXPERT_CACHE:
        return _EXPERT_CACHE[key]

    gain, _, iterations = solve_dare(plant)
    expert = StateFeedbackPolicy(gain=gain, force_mag=plant.force_mag)
    stats = summarize(run_batch(expert, plant, episodes=100, seed=seed))
    if stats["survival_mean"] >= 190.0 and stats["success_rate"] >= 0.95:
        note = (
            f"LQR expert (DARE, {iterations} iterations, "
            f"K=[{', '.join(f'{float(v):.2f}' for v in gain)}]) balanced "
            f"{stats['survival_mean']:.1f}/200 over 100 seeded episodes."
        )
    else:
        pd_gain = torch.tensor([2.0, 3.0, 42.0, 9.0], dtype=torch.float64)
        pd = StateFeedbackPolicy(gain=-pd_gain, force_mag=plant.force_mag, kind="pd")
        pd_stats = summarize(run_batch(pd, plant, episodes=100, seed=seed))
        note = (
            "LQR gain was unstable "
            f"({stats['survival_mean']:.1f}/200, success {stats['success_rate']:.2f}); "
            f"fell back to hand-tuned PD ({pd_stats['survival_mean']:.1f}/200)."
        )
        expert, stats = pd, pd_stats
    _EXPERT_CACHE[key] = (expert, stats, note)
    return expert, stats, note


# ---------------------------------------------------------------------------
# Batched closed-loop evaluation
# ---------------------------------------------------------------------------


def _empty_batch(episodes: int) -> dict[str, Any]:
    return {
        "survival": torch.zeros(episodes, dtype=torch.long),
        "angle_sq": torch.zeros(episodes, dtype=torch.float64),
        "cart_sq": torch.zeros(episodes, dtype=torch.float64),
        "switch_sum": 0.0,
        "episodes": episodes,
    }


@torch.no_grad()
def run_batch(
    policy: Any,
    plant: Plant,
    *,
    episodes: int,
    seed: int,
    noise: float = 0.0,
    delay: int = 0,
    protocol: Protocol | None = None,
) -> dict[str, Any]:
    """Run independent episodes in parallel; all randomness is seeded.

    ``protocol`` controls the initial-state distribution and the optional
    seeded impulse kick (the kick is added on top of the policy force).
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    protocol = NOMINAL_PROTOCOL if protocol is None else protocol
    generator = torch.Generator().manual_seed(int(seed))
    state = sample_initial_states(episodes, generator, protocol)
    kick_start, kick_sign = (
        sample_impulse_schedule(episodes, generator, protocol)
        if protocol.impulse
        else (None, None)
    )
    alive = torch.ones(episodes, dtype=torch.bool)
    batch = _empty_batch(episodes)
    survival = batch["survival"]
    previous = torch.ones(episodes, dtype=torch.long)
    queued = (
        torch.ones(episodes, delay, dtype=torch.long) if delay > 0 else None
    )
    for step in range(plant.max_steps):
        observation = state.to(torch.float32)
        if noise > 0.0:
            observation = observation + torch.randn(
                observation.shape, generator=generator, dtype=torch.float32
            ) * noise
        action = policy.act(observation)
        if delay > 0:
            assert queued is not None
            applied = queued[:, 0].clone()
            queued = torch.cat((queued[:, 1:], action.unsqueeze(1)), dim=1)
        else:
            applied = action
        force = action_force(applied, plant)
        if protocol.impulse:
            assert kick_start is not None and kick_sign is not None
            kicking = (step >= kick_start) & (step < kick_start + int(protocol.impulse_steps))
            force = force + torch.where(
                kicking,
                kick_sign * float(protocol.impulse_strength),
                torch.zeros_like(force),
            )
        following = rk4_step(state, force, plant)
        survived = (
            alive
            & (following[:, 0].abs() <= plant.x_threshold)
            & (following[:, 2].abs() <= plant.theta_threshold)
        )
        survival[survived] = step + 1
        batch["angle_sq"][alive] += following[alive, 2].square()
        batch["cart_sq"][alive] += following[alive, 0].square()
        if step > 0 and bool(alive.any()):
            batch["switch_sum"] += float((applied - previous).abs()[alive].sum())
        state = torch.where(survived.unsqueeze(-1), following, state)
        previous = applied
        alive = survived
        if not bool(alive.any()):
            break
    return batch


def summarize(batch: dict[str, Any]) -> dict[str, Any]:
    survival = [int(value) for value in batch["survival"].tolist()]
    total_steps = max(sum(survival), 1)
    return {
        "episodes": len(survival),
        "survival_mean": sum(survival) / len(survival),
        "survival_median": percentile(survival, 0.5),
        "survival_min": min(survival),
        "survival_max": max(survival),
        "success_rate": sum(1 for value in survival if value >= SUCCESS_STEPS) / len(survival),
        "angle_rms": math.sqrt(float(batch["angle_sq"].sum()) / total_steps) * 180.0 / math.pi,
        "cart_rms": math.sqrt(float(batch["cart_sq"].sum()) / total_steps),
        "switch_rate": float(batch["switch_sum"]) / total_steps,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MLPPolicyModel(nn.Module):
    """Dense baseline: Linear(4, hidden) -> ReLU -> Linear(hidden, 3)."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.register_buffer("obs_mean", torch.zeros(4))
        self.register_buffer("obs_std", torch.ones(4))
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(4, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, ACTIONS),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net((observation - self.obs_mean) / self.obs_std)


class SparseBrainPolicy(nn.Module):
    """Fixed sparse recurrent core with a cheap linear readout.

    The recurrent graph (``neurons * degree`` requested edges, coalesced to
    unique source/destination pairs), the sensory projection and the neural
    time constants are fixed; only the recurrent edge amplitudes and the
    readout are trained.

    The sensory projection is a fixed-fanout sparse map: every one of the four
    observation features drives exactly ``sensory_fanout`` neurons (the same
    per-feature fanout convention as the MNIST/Gomoku sparse projections).
    """

    def __init__(
        self,
        *,
        neurons: int = 128,
        degree: int = 8,
        steps: int = 2,
        seed: int = 0,
        sensory_fanout: int = 8,
        sensory_scale: float = 1.5,
        weight_scale: float = 0.15,
        alpha: float = 0.72,
        gain: float = 1.05,
        initial_state: str = "sigmoid",
    ) -> None:
        super().__init__()
        if neurons <= 0 or degree <= 0 or steps <= 0 or sensory_fanout <= 0:
            raise ValueError("neurons, degree, steps and sensory_fanout must be positive")
        if initial_state not in ("sigmoid", "zero"):
            raise ValueError("initial_state must be sigmoid or zero")
        self.neurons = int(neurons)
        self.steps = int(steps)
        self.sensory_fanout = int(sensory_fanout)
        self.initial_state = initial_state

        topology = random_topology(
            self.neurons,
            self.neurons * int(degree),
            device=DEVICE,
            dtype=torch.float32,
            seed=int(seed),
            weight_scale=float(weight_scale),
        )
        self.register_buffer("in_rowptr", topology.in_rowptr.detach().clone())
        self.register_buffer("in_col", topology.in_col.detach().clone())
        self.register_buffer("src", topology.src.detach().clone())
        self.register_buffer("dst", topology.dst.detach().clone())
        # Canonical source-major edge id -> destination-CSR position, inverted
        # so the CSR values can be gathered from the trainable edge vector.
        self.register_buffer(
            "in_weight_index",
            torch.argsort(topology.out_weight_index.to(torch.int64)),
        )
        self.edge_weight = nn.Parameter(topology.weight.detach().clone())

        generator = torch.Generator(device=DEVICE).manual_seed(int(seed) + 17)
        sensory_src = torch.arange(4, dtype=torch.int64).repeat_interleave(self.sensory_fanout)
        sensory_dst = torch.randint(
            0, self.neurons, (sensory_src.numel(),), generator=generator
        )
        self.register_buffer("sensory_src", sensory_src)
        self.register_buffer("sensory_dst", sensory_dst)
        self.register_buffer(
            "sensory_weight",
            torch.randn(sensory_src.numel(), generator=generator) * float(sensory_scale),
        )

        self.register_buffer("obs_mean", torch.zeros(4))
        self.register_buffer("obs_std", torch.ones(4))
        self.register_buffer("alpha", torch.full((self.neurons,), float(alpha)))
        self.register_buffer("gain", torch.full((self.neurons,), float(gain)))
        self.dynamics = Dynamics(self.alpha, self.gain)
        self.readout = nn.Linear(self.neurons, ACTIONS)

    @property
    def edge_count(self) -> int:
        return int(self.edge_weight.numel())

    @property
    def sensory_edge_count(self) -> int:
        return int(self.sensory_weight.numel())

    def _csr(self) -> torch.Tensor:
        values = self.edge_weight[self.in_weight_index]
        return torch.sparse_csr_tensor(
            self.in_rowptr,
            self.in_col,
            values,
            size=(self.neurons, self.neurons),
            check_invariants=False,
        )

    def forward(
        self,
        observation: torch.Tensor,
        *,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        normalized = (observation - self.obs_mean) / self.obs_std
        batch = normalized.shape[0]
        sensory = normalized.new_zeros((batch, self.neurons))
        sensory.scatter_add_(
            1,
            self.sensory_dst.expand(batch, -1),
            normalized.index_select(1, self.sensory_src) * self.sensory_weight,
        )
        drive = torch.tanh(sensory)
        if self.initial_state == "sigmoid":
            state = torch.sigmoid(sensory)
        else:
            state = torch.zeros_like(sensory)
        history = [state]
        weight = self._csr()
        for _ in range(self.steps):
            current = torch.sparse.mm(weight, state.T).T
            state = self.dynamics.advance(state, current, drive)
            if return_states:
                history.append(state)
        logits = self.readout(state)
        if return_states:
            return logits, history
        return logits


class TorchPolicy:
    """Adapts an ``nn.Module`` to the ``act(observation) -> indices`` protocol."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation).argmax(dim=1)


def build_model(config: dict[str, Any], seed: int) -> nn.Module:
    kind = config["policy"]
    if kind == "mlp":
        return MLPPolicyModel(hidden=int(config["hidden"]))
    if kind == "brain":
        return SparseBrainPolicy(
            neurons=int(config["neurons"]),
            degree=int(config["degree"]),
            steps=int(config["steps"]),
            seed=int(seed),
            sensory_fanout=int(config["sensory_fanout"]),
            sensory_scale=float(config["sensory_scale"]),
            weight_scale=float(config["weight_scale"]),
            alpha=float(config["alpha"]),
            gain=float(config["gain"]),
            initial_state=str(config["initial_state"]),
        )
    raise ValueError(f"policy {kind!r} has no trainable model")


# ---------------------------------------------------------------------------
# Behavior cloning
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_bc_data(
    expert: StateFeedbackPolicy,
    plant: Plant,
    *,
    samples: int,
    seed: int,
    noise_prob: float = 0.1,
    batch: int = 64,
    protocol: Protocol | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded randomized-initial-state rollouts labeled by the expert.

    The executed action is replaced by a uniform random action with
    probability ``noise_prob`` (state coverage); the *label* is always the
    expert action for the observed state.  ``protocol`` matches the evaluation
    distribution, including the per-episode impulse kick in hard mode.
    """

    protocol = NOMINAL_PROTOCOL if protocol is None else protocol
    generator = torch.Generator().manual_seed(int(seed) + 7001)
    state = sample_initial_states(batch, generator, protocol)
    age = torch.zeros(batch, dtype=torch.long)
    kick_start, kick_sign = (
        sample_impulse_schedule(batch, generator, protocol)
        if protocol.impulse
        else (None, None)
    )
    states: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    collected = 0
    while collected < samples:
        label = expert.act(state)
        executed = label
        if noise_prob > 0.0:
            draw = torch.rand(batch, generator=generator)
            random_action = torch.randint(0, ACTIONS, (batch,), generator=generator)
            executed = torch.where(draw < noise_prob, random_action, label)
        states.append(state.to(torch.float32).clone())
        labels.append(label.clone())
        collected += batch
        force = action_force(executed, plant)
        if protocol.impulse:
            assert kick_start is not None and kick_sign is not None
            kicking = (age >= kick_start) & (age < kick_start + int(protocol.impulse_steps))
            force = force + torch.where(
                kicking,
                kick_sign * float(protocol.impulse_strength),
                torch.zeros_like(force),
            )
        following = rk4_step(state, force, plant)
        done = (
            (following[:, 0].abs() > plant.x_threshold)
            | (following[:, 2].abs() > plant.theta_threshold)
        )
        fresh = sample_initial_states(batch, generator, protocol)
        state = torch.where(done.unsqueeze(-1), fresh, following)
        age = torch.where(done, torch.zeros_like(age), age + 1)
        if protocol.impulse:
            assert kick_start is not None and kick_sign is not None
            new_start, new_sign = sample_impulse_schedule(batch, generator, protocol)
            kick_start = torch.where(done, new_start, kick_start)
            kick_sign = torch.where(done, new_sign, kick_sign)
    return torch.cat(states)[:samples], torch.cat(labels)[:samples]


@torch.no_grad()
def _accuracy(model: nn.Module, states: torch.Tensor, labels: torch.Tensor, batch: int = 1024) -> float:
    correct = 0
    for start in range(0, states.shape[0], batch):
        logits = model(states[start : start + batch])
        correct += int((logits.argmax(dim=1) == labels[start : start + batch]).sum())
    return correct / max(int(states.shape[0]), 1)


def train_bc(
    model: nn.Module,
    states: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    batch: int,
    lr: float,
) -> dict[str, Any]:
    """AdamW cross-entropy clone with a deterministic 80/20 split."""

    timer = Timer()
    total = int(states.shape[0])
    train_size = int(total * 0.8)
    if train_size <= 0 or total - train_size <= 0:
        raise ValueError("bc_samples is too small for an 80/20 split")
    generator = torch.Generator().manual_seed(int(seed) + 99)
    permutation = torch.randperm(total, generator=generator)
    train_index = permutation[:train_size]
    test_index = permutation[train_size:]
    if isinstance(model, (MLPPolicyModel, SparseBrainPolicy)):
        with torch.no_grad():
            model.obs_mean.copy_(states[train_index].mean(dim=0))
            model.obs_std.copy_(states[train_index].std(dim=0).clamp_min(1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        order = train_index[torch.randperm(train_size, generator=generator)]
        loss_sum = 0.0
        seen = 0
        for start in range(0, train_size, int(batch)):
            index = order[start : start + int(batch)]
            logits = model(states.index_select(0, index))
            loss = F.cross_entropy(logits, labels.index_select(0, index))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * int(index.numel())
            seen += int(index.numel())
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": loss_sum / max(seen, 1),
                "seconds": timer.seconds,
            }
        )
    model.eval()
    return {
        "train_accuracy": _accuracy(model, states.index_select(0, train_index), labels.index_select(0, train_index)),
        "test_accuracy": _accuracy(model, states.index_select(0, test_index), labels.index_select(0, test_index)),
        "train_size": train_size,
        "test_size": total - train_size,
        "history": history,
        "seconds": timer.seconds,
        "final_loss": history[-1]["loss"] if history else float("nan"),
    }


def checkpoint_path(config: dict[str, Any], seed: int) -> Path:
    subset = {key: config[key] for key in CHECKPOINT_KEYS}
    digest = hashlib.sha1(
        json.dumps(subset, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return (
        CHECKPOINTS
        / f"cartpole_bc_v{CHECKPOINT_FORMAT}_{config['policy']}_{digest}_seed={int(seed)}.pt"
    )


def prepare_policy(
    config: dict[str, Any],
    plant: Plant,
    expert: StateFeedbackPolicy,
    seed: int,
    *,
    protocol: Protocol | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build and (unless a compatible checkpoint exists) train the BC policy."""

    model = build_model(config, seed)
    path = checkpoint_path(config, seed)
    if path.exists():
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
            if (
                isinstance(blob, dict)
                and blob.get("format") == CHECKPOINT_FORMAT
                and blob.get("policy") == config["policy"]
            ):
                model.load_state_dict(blob["state_dict"])
                model.eval()
                log(f"reused BC checkpoint {path.name}")
                return model, {
                    "collect_seconds": 0.0,
                    "train_seconds": 0.0,
                    "bc_accuracy": float(blob["test_accuracy"]),
                    "bc_train_accuracy": float(blob["train_accuracy"]),
                    "bc_train_size": int(blob["train_size"]),
                    "bc_test_size": int(blob["test_size"]),
                    "checkpoint": str(path),
                    "cached": True,
                }
        except Exception as exc:  # pragma: no cover - corrupt cache only
            log(f"could not read {path.name} ({exc}); retraining")
    timer = Timer()
    states, labels = collect_bc_data(
        expert,
        plant,
        samples=int(config["bc_samples"]),
        seed=int(seed),
        noise_prob=float(config["noise_prob"]),
        protocol=protocol,
    )
    collect_seconds = timer.seconds
    training = train_bc(
        model,
        states,
        labels,
        seed=int(seed),
        epochs=int(config["epochs"]),
        batch=int(config["batch"]),
        lr=float(config["lr"]),
    )
    log(
        f"trained {config['policy']} on {states.shape[0]} transitions: "
        f"test acc {training['test_accuracy']:.3f} in {training['seconds']:.2f}s"
    )
    ensure_dir(path.parent)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "policy": config["policy"],
            "seed": int(seed),
            "config": {key: config[key] for key in CHECKPOINT_KEYS},
            "state_dict": model.state_dict(),
            "train_accuracy": training["train_accuracy"],
            "test_accuracy": training["test_accuracy"],
            "train_size": training["train_size"],
            "test_size": training["test_size"],
        },
        temporary,
    )
    temporary.replace(path)
    return model, {
        "collect_seconds": collect_seconds,
        "train_seconds": training["seconds"],
        "bc_accuracy": training["test_accuracy"],
        "bc_train_accuracy": training["train_accuracy"],
        "bc_train_size": training["train_size"],
        "bc_test_size": training["test_size"],
        "final_loss": training["final_loss"],
        "checkpoint": str(path),
        "cached": False,
    }


# ---------------------------------------------------------------------------
# Online three-factor plasticity
# ---------------------------------------------------------------------------


def _reward_proxy(plant: Plant, theta: float, x: float) -> float:
    value = 1.0 - abs(theta) / plant.theta_threshold - 0.5 * abs(x) / plant.x_threshold
    return max(-1.0, min(1.0, value))


@torch.no_grad()
def run_online_variant(
    brain: SparseBrainPolicy,
    variant: str,
    config: dict[str, Any],
    *,
    seed: int,
    change_episode: int,
) -> dict[str, Any]:
    """One hard-protocol run; only the recurrent edges can change.

    Episodes 0..change_episode-1 use the hard nominal plant; the remaining
    episodes use the shifted plant selected by ``online_change``.
    """

    if variant not in ("frozen", "online"):
        raise ValueError("variant must be frozen or online")
    model = copy.deepcopy(brain)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    episodes = int(config["episodes"])
    protocol = make_protocol(config, impulse=True, hard=True)
    pre_plant = make_plant(config, hard=True)
    if config["online_change"] == "length":
        post_plant = replace(
            pre_plant,
            length_scale=float(config["change_length_scale"]),
        )
    else:
        post_plant = replace(
            pre_plant,
            force_mag=float(config["change_force_mag"]),
        )
    decay = float(config["eligibility_decay"])
    learning_rate = float(config["online_lr"])
    baseline_decay = float(config["baseline_decay"])

    generator = torch.Generator().manual_seed(int(seed) + 4242)
    weights_before = model.edge_weight.detach().clone()
    trace = torch.zeros_like(weights_before)
    baseline = 0.0
    survival: list[int] = []
    per_episode: list[dict[str, Any]] = []
    timer = Timer()
    for episode in range(episodes):
        shifted = episode >= change_episode
        plant = post_plant if shifted else pre_plant
        state = sample_initial_states(1, generator, protocol)
        kick_start, kick_sign = (
            sample_impulse_schedule(1, generator, protocol)
            if protocol.impulse
            else (None, None)
        )
        steps = 0
        rewards: list[float] = []
        for step in range(plant.max_steps):
            observation = state.to(torch.float32)
            logits, states = model(observation, return_states=True)
            action = logits.argmax(dim=1)
            force = action_force(action, plant)
            if protocol.impulse:
                assert kick_start is not None and kick_sign is not None
                kicking = (
                    (step >= kick_start) & (step < kick_start + int(protocol.impulse_steps))
                )
                force = force + torch.where(
                    kicking,
                    kick_sign * float(protocol.impulse_strength),
                    torch.zeros_like(force),
                )
            following = rk4_step(state, force, plant)
            steps += 1
            theta = float(following[0, 2])
            x = float(following[0, 0])
            rewards.append(_reward_proxy(plant, theta, x))
            for index in range(len(states) - 1):
                trace = update_eligibility_trace(
                    trace,
                    states[index],
                    states[index + 1],
                    model.src,
                    model.dst,
                    decay=decay,
                )
            alive = abs(x) <= plant.x_threshold and abs(theta) <= plant.theta_threshold
            if not alive:
                break
            state = following
        survival.append(steps)
        mean_reward = sum(rewards) / max(len(rewards), 1)
        if variant == "online":
            dopamine = mean_reward - baseline
            update = dopamine_gated_weight_update(
                model.edge_weight.detach(),
                trace,
                float(dopamine),
                learning_rate=learning_rate,
                preserve_sign=True,
                minimum_weight=-EDGE_BOUND,
                maximum_weight=EDGE_BOUND,
            )
            model.edge_weight.copy_(update.weight)
            trace = torch.zeros_like(trace)
        baseline = baseline_decay * baseline + (1.0 - baseline_decay) * mean_reward
        per_episode.append(
            {
                "episode": episode,
                "hard": True,
                "shifted": shifted,
                "length_scale": plant.length_scale,
                "force_mag": plant.force_mag,
                "steps": steps,
                "mean_reward": mean_reward,
            }
        )

    pre = survival[:change_episode]
    post = survival[change_episode:]
    survival_pre = sum(pre) / len(pre) if pre else None
    survival_post = sum(post) / len(post) if post else None
    recovery = next(
        (
            index
            for index, value in enumerate(survival)
            if index >= change_episode and value >= 150
        ),
        None,
    )
    delta = model.edge_weight.detach() - weights_before
    return {
        "variant": variant,
        "survival_pre": survival_pre,
        "survival_post": survival_post,
        "recovery_episode": recovery,
        "adaptation_gain": (
            survival_post - survival_pre
            if survival_post is not None and survival_pre is not None
            else None
        ),
        "weight_change_l2": float(delta.norm()),
        "weight_change_max": float(delta.abs().max()) if delta.numel() else 0.0,
        "initial_weight_l2": float(weights_before.norm()),
        "episodes": episodes,
        "survival": survival,
        "per_episode": per_episode,
        "seconds": timer.seconds,
    }


def ablate_recurrent(model: SparseBrainPolicy) -> SparseBrainPolicy:
    """Clone with all recurrent edge weights zeroed (sensory/readout intact)."""

    clone = copy.deepcopy(model)
    with torch.no_grad():
        clone.edge_weight.zero_()
    for parameter in clone.parameters():
        parameter.requires_grad_(False)
    clone.eval()
    return clone


def ablation_metrics(
    model: SparseBrainPolicy,
    plant: Plant,
    protocol: Protocol,
    *,
    episodes: int,
    seed: int,
    noise: float = 0.0,
    delay: int = 0,
) -> dict[str, float]:
    """Closed-loop contribution of the learned recurrent core."""

    full = summarize(
        run_batch(
            TorchPolicy(model),
            plant,
            episodes=episodes,
            seed=seed,
            noise=noise,
            delay=delay,
            protocol=protocol,
        )
    )
    ablated = summarize(
        run_batch(
            TorchPolicy(ablate_recurrent(model)),
            plant,
            episodes=episodes,
            seed=seed,
            noise=noise,
            delay=delay,
            protocol=protocol,
        )
    )
    return {
        "recurrent_ablation_survival": ablated["survival_mean"],
        "recurrent_ablation_success_rate": ablated["success_rate"],
        "recurrent_full_survival": full["survival_mean"],
        "recurrent_contribution": full["survival_mean"] - ablated["survival_mean"],
    }


def online_experiment(
    brain: SparseBrainPolicy,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Frozen versus online from the same BC checkpoint and episode seeds."""

    change_episode = min(int(config["change_episode"]), int(config["episodes"]))
    frozen = run_online_variant(brain, "frozen", config, seed=seed, change_episode=change_episode)
    online = run_online_variant(brain, "online", config, seed=seed, change_episode=change_episode)
    requested = online if config["variant"] == "online" else frozen
    if frozen["survival_pre"] is not None and online["survival_pre"] is not None:
        if abs(frozen["survival_pre"] - online["survival_pre"]) > 1e-9:
            log("warning: frozen and online pre-change phases diverged")

    hard_plant = make_plant(config, hard=True)
    hard_protocol = make_protocol(config, impulse=True, hard=True)
    ablation = ablation_metrics(
        brain,
        hard_plant,
        hard_protocol,
        episodes=int(config["episodes"]),
        seed=int(seed),
    )
    if config["online_change"] == "length":
        shift = {
            "kind": "length",
            "value": float(config["change_length_scale"]),
            "description": (
                f"pole half-length x{float(config['change_length_scale']):g} "
                f"({0.5 * float(config['change_length_scale']):g} m)"
            ),
        }
    else:
        shift = {
            "kind": "force",
            "value": float(config["change_force_mag"]),
            "description": (
                f"actuator authority {float(config['hard_force_mag']):g} -> "
                f"{float(config['change_force_mag']):g} N"
            ),
        }
    if (
        frozen["survival_post"] is None
        or online["survival_post"] is None
    ):
        verdict = "inconclusive"
        sentence = (
            "Online plasticity: no post-change episodes were run, so the "
            "frozen/online comparison is inconclusive."
        )
        post_delta = None
    else:
        delta = online["survival_post"] - frozen["survival_post"]
        post_delta = delta
        if delta >= 3.0:
            verdict = "helped"
        elif delta <= -3.0:
            verdict = "hurt"
        else:
            verdict = "neutral"
        initial_l2 = max(online["initial_weight_l2"], 1e-12)
        ratio = 100.0 * online["weight_change_l2"] / initial_l2
        sentence = (
            "Online plasticity "
            f"{verdict} under the hard protocol ({shift['description']}): "
            f"pre {online['survival_pre']:.1f} -> post online "
            f"{online['survival_post']:.1f} vs frozen {frozen['survival_post']:.1f} "
            f"(delta {delta:+.1f} steps); online recurrent weight change L2 "
            f"{online['weight_change_l2']:.2e} ({ratio:.0f}% of the initial L2 "
            f"{online['initial_weight_l2']:.2e}, max |dw| "
            f"{online['weight_change_max']:.2e}), readout frozen."
        )
        if verdict == "neutral" and min(online["survival_post"], frozen["survival_post"]) >= 199.0:
            sentence += (
                " Both variants sit at the 200-step ceiling, so this comparison "
                "has no headroom to detect an adaptation benefit."
            )
        if abs(delta) < 10.0:
            sentence += (
                " The difference is small relative to the 200-step horizon, so "
                "read it as no measurable benefit rather than a robust effect."
            )
    contribution = ablation["recurrent_contribution"]
    ablation_sentence = (
        f" Recurrent ablation: zeroing the learned edges leaves survival "
        f"{ablation['recurrent_ablation_survival']:.1f} vs "
        f"{ablation['recurrent_full_survival']:.1f} full "
        f"(contribution {contribution:+.1f} steps)."
    )
    if abs(contribution) < 5.0:
        ablation_sentence += (
            " The learned recurrence is nearly dispensable, so the plasticity "
            "rule has little behavioural leverage: the policy is carried by "
            "the sensory projection and the frozen readout."
        )
    sentence += ablation_sentence
    summary_keys = (
        "survival_pre",
        "survival_post",
        "recovery_episode",
        "adaptation_gain",
        "weight_change_l2",
        "weight_change_max",
        "initial_weight_l2",
        "episodes",
    )
    requested_summary = {key: requested[key] for key in summary_keys}
    requested_summary["post_change_delta"] = post_delta
    return {
        "requested": requested_summary,
        "requested_name": requested["variant"],
        "per_episode": requested["per_episode"],
        "variants": {
            "frozen": {key: frozen[key] for key in summary_keys},
            "online": {key: online[key] for key in summary_keys},
        },
        "ablation": ablation,
        "shift": shift,
        "verdict": verdict,
        "post_change_delta": post_delta,
        "note": sentence,
        "seconds": frozen["seconds"] + online["seconds"],
    }


# ---------------------------------------------------------------------------
# Robustness sweep
# ---------------------------------------------------------------------------


def robustness_conditions(hard: bool = False) -> list[dict[str, Any]]:
    """One factor at a time away from the nominal condition.

    The non-hard grid is unchanged.  The hard grid widens every factor and
    adds an explicit ``impulse`` row; its reference row is the hard plant
    *without* the impulse so that each row differs by exactly one factor
    (hard closed-loop and online runs include the impulse).
    """

    rows = [
        {
            "condition": "nominal",
            "factor": "nominal",
            "noise": 0.0,
            "delay": 0,
            "length_scale": 1.0,
            "mass_scale": 1.0,
            "impulse": False,
        },
        {
            "condition": "impulse",
            "factor": "impulse",
            "noise": 0.0,
            "delay": 0,
            "length_scale": 1.0,
            "mass_scale": 1.0,
            "impulse": True,
        },
    ]
    noises = (0.02, 0.05, 0.1, 0.2) if hard else (0.01, 0.05)
    delays = (1, 2, 4, 6) if hard else (1, 2)
    lengths = (0.5, 2.0, 4.0) if hard else (0.5, 2.0)
    masses = (0.3, 3.0) if hard else (0.5, 2.0)
    for noise in noises:
        rows.append(
            {
                "condition": f"noise={noise:g}",
                "factor": "sensor_noise",
                "noise": noise,
                "delay": 0,
                "length_scale": 1.0,
                "mass_scale": 1.0,
                "impulse": False,
            }
        )
    for delay in delays:
        rows.append(
            {
                "condition": f"delay={delay}",
                "factor": "action_delay",
                "noise": 0.0,
                "delay": delay,
                "length_scale": 1.0,
                "mass_scale": 1.0,
                "impulse": False,
            }
        )
    for length_scale in lengths:
        rows.append(
            {
                "condition": f"length_scale={length_scale:g}",
                "factor": "pole_length",
                "noise": 0.0,
                "delay": 0,
                "length_scale": length_scale,
                "mass_scale": 1.0,
                "impulse": False,
            }
        )
    for mass_scale in masses:
        rows.append(
            {
                "condition": f"mass_scale={mass_scale:g}",
                "factor": "pole_mass",
                "noise": 0.0,
                "delay": 0,
                "length_scale": 1.0,
                "mass_scale": mass_scale,
                "impulse": False,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def parse_config(raw: str | None) -> dict[str, Any]:
    config = dict(DEFAULTS)
    if raw:
        update = json.loads(raw)
        if not isinstance(update, dict):
            raise ValueError("--config-json must decode to an object")
        unknown = sorted(set(update) - set(DEFAULTS))
        if unknown:
            log(f"ignoring unknown config keys: {unknown}")
        config.update({key: value for key, value in update.items() if key in DEFAULTS})
    for key in _INT_KEYS:
        try:
            config[key] = int(config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"config key {key!r} must be an integer") from exc
    for key in _FLOAT_KEYS:
        try:
            config[key] = float(config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"config key {key!r} must be a number") from exc
    for key in _BOOL_KEYS:
        config[key] = bool(config[key])
    if config["policy"] not in ("lqr", "mlp", "brain"):
        raise ValueError("policy must be one of lqr, mlp, brain")
    if config["eval"] not in ("closed_loop", "robustness", "online"):
        raise ValueError("eval must be one of closed_loop, robustness, online")
    if config["online_change"] not in ("length", "force"):
        raise ValueError("online_change must be length or force")
    if config["variant"] not in ("frozen", "online"):
        raise ValueError("variant must be frozen or online")
    if config["episodes"] <= 0:
        raise ValueError("episodes must be positive")
    if config["bc_samples"] < 50:
        raise ValueError("bc_samples must be at least 50")
    if config["neurons"] <= 0 or config["degree"] <= 0 or config["steps"] <= 0:
        raise ValueError("neurons, degree and steps must be positive")
    if config["hidden"] <= 0:
        raise ValueError("hidden must be positive")
    if config["epochs"] <= 0 or config["batch"] <= 0:
        raise ValueError("epochs and batch must be positive")
    if config["delay"] < 0:
        raise ValueError("delay must be non-negative")
    if config["noise"] < 0.0:
        raise ValueError("noise must be non-negative")
    if not 0.0 <= config["alpha"] < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    if config["gain"] < 0.0:
        raise ValueError("gain must be non-negative")
    if not 0.0 <= config["eligibility_decay"] <= 1.0:
        raise ValueError("eligibility_decay must lie in [0, 1]")
    if config["hard_theta_range"] < 0.0 or config["hard_other_range"] < 0.0:
        raise ValueError("hard init ranges must be non-negative")
    if config["impulse_steps"] <= 0 or config["impulse_first"] < 0:
        raise ValueError("impulse_steps must be positive and impulse_first non-negative")
    if config["impulse_last"] < config["impulse_first"]:
        raise ValueError("impulse_last must be >= impulse_first")
    if config["change_length_scale"] <= 0.0 or config["change_force_mag"] <= 0.0:
        raise ValueError("online plant-shift magnitudes must be positive")
    if config["eval"] == "online" and config["policy"] != "brain":
        raise ValueError("eval='online' requires policy='brain'")
    return config


def run_experiment(config: dict[str, Any], seed: int) -> dict[str, Any]:
    total_timer = Timer()
    seed_everything(int(seed))
    hard = bool(config["hard"]) or config["eval"] == "online"
    plant = make_plant(config, hard=hard)
    protocol = make_protocol(config, impulse=True, hard=hard)
    nominal_plant = Plant()
    expert, expert_stats, expert_note = build_expert(nominal_plant, seed=0)
    notes = [expert_note]
    timings = {"collect_seconds": 0.0, "train_seconds": 0.0, "eval_seconds": 0.0}
    bc_accuracy: float | None = None
    bc_train_accuracy: float | None = None
    trainable = 0
    model: nn.Module | None = None

    if hard:
        notes.append(
            "Hard protocol: theta~U(-%.2f,%.2f), other states~U(-%.2f,%.2f), "
            "%.1f N actuator, seeded +-%.1f N kick for %d steps at a per-episode "
            "step in [%d,%d]."
            % (
                float(config["hard_theta_range"]),
                float(config["hard_theta_range"]),
                float(config["hard_other_range"]),
                float(config["hard_other_range"]),
                float(config["hard_force_mag"]),
                float(config["impulse_strength"]),
                int(config["impulse_steps"]),
                int(config["impulse_first"]),
                int(config["impulse_last"]),
            )
        )
    if config["eval"] == "online" and not bool(config["hard"]):
        notes.append("eval=online forces the hard protocol.")

    if config["policy"] == "lqr":
        policy: Any = expert
        notes.append("No behavior cloning for policy=lqr.")
    else:
        # ``eval=online`` forces the hard protocol, so key the BC checkpoint on
        # the *effective* protocol rather than the raw config flag.
        effective = dict(config)
        effective["hard"] = hard
        model, info = prepare_policy(
            effective, plant, expert, int(seed), protocol=protocol
        )
        policy = TorchPolicy(model)
        timings["collect_seconds"] = float(info["collect_seconds"])
        timings["train_seconds"] = float(info["train_seconds"])
        bc_accuracy = float(info["bc_accuracy"])
        bc_train_accuracy = float(info["bc_train_accuracy"])
        trainable = trainable_parameters(model)
        cache_note = "BC checkpoint reused." if info["cached"] else "BC checkpoint written."
        notes.append(
            f"BC on {info['bc_train_size'] + info['bc_test_size']} transitions "
            f"(80/20 split): train {info['bc_train_accuracy']:.3f}, "
            f"test {info['bc_accuracy']:.3f}; {cache_note}"
        )
        if config["policy"] == "brain":
            assert isinstance(model, SparseBrainPolicy)
            notes.append(
                f"brain has {model.edge_count} recurrent edges "
                f"(degree {config['degree']}) and "
                f"{model.sensory_edge_count} fixed sensory edges "
                f"(fanout {config['sensory_fanout']}/input, scale "
                f"{config['sensory_scale']}, {config['initial_state']} init); "
                "recurrent weights + readout trained."
            )

    eval_timer = Timer()
    metrics: dict[str, Any] = {}
    if config["eval"] == "closed_loop":
        batch = run_batch(
            policy,
            plant,
            episodes=int(config["episodes"]),
            seed=int(seed),
            noise=float(config["noise"]),
            delay=int(config["delay"]),
            protocol=protocol,
        )
        metrics.update(summarize(batch))
        notes.append(
            f"closed loop: {config['episodes']} episodes, noise {config['noise']}, "
            f"delay {config['delay']}, length_scale {config['length_scale']}, "
            f"mass_scale {config['mass_scale']}."
        )
        if config["policy"] == "brain":
            ablation = ablation_metrics(
                model,
                plant,
                protocol,
                episodes=int(config["episodes"]),
                seed=int(seed),
                noise=float(config["noise"]),
                delay=int(config["delay"]),
            )
            metrics.update(ablation)
    elif config["eval"] == "robustness":
        conditions: list[dict[str, Any]] = []
        for row in robustness_conditions(hard):
            condition_plant = make_plant(
                config,
                hard=hard,
                length_scale=float(row["length_scale"]),
                mass_scale=float(row["mass_scale"]),
            )
            condition_protocol = make_protocol(
                config, impulse=bool(row["impulse"]), hard=hard
            )
            batch = run_batch(
                policy,
                condition_plant,
                episodes=int(config["episodes"]),
                seed=int(seed),
                noise=float(row["noise"]),
                delay=int(row["delay"]),
                protocol=condition_protocol,
            )
            conditions.append({**row, **summarize(batch)})
            if row["condition"] == "nominal" and config["policy"] == "brain":
                metrics.update(
                    ablation_metrics(
                        model,
                        condition_plant,
                        condition_protocol,
                        episodes=int(config["episodes"]),
                        seed=int(seed),
                    )
                )
        metrics["conditions"] = conditions
        nominal_row = conditions[0]
        for key in (
            "episodes",
            "survival_mean",
            "survival_median",
            "survival_min",
            "survival_max",
            "success_rate",
            "angle_rms",
            "cart_rms",
            "switch_rate",
        ):
            metrics[key] = nominal_row[key]
        notes.append(
            f"robustness sweep: {len(conditions)} one-factor conditions x "
            f"{config['episodes']} episodes "
            f"({'hard' if hard else 'nominal'} reference condition)."
        )
    else:
        assert model is not None, "online eval requires a trained brain model"
        online = online_experiment(model, config, int(seed))
        metrics.update(online["requested"])
        metrics.update(online["ablation"])
        metrics["per_episode"] = online["per_episode"]
        metrics["variants"] = online["variants"]
        metrics["adaptation_verdict"] = online["verdict"]
        metrics["requested_variant"] = online["requested_name"]
        metrics["online_shift"] = online["shift"]
        notes.append(online["note"])
        notes.append(
            f"plant change at episode "
            f"{min(int(config['change_episode']), int(config['episodes']))}: "
            f"{online['shift']['description']}."
        )
    timings["eval_seconds"] = eval_timer.seconds

    single = torch.zeros(1, 4, dtype=torch.float32)
    latency = measure_latency(
        lambda: policy.act(single),
        warmup=5,
        iterations=int(config["latency_iterations"]),
    )
    metrics.update(
        {
            "bc_accuracy": bc_accuracy,
            "bc_test_accuracy": bc_accuracy,
            "bc_train_accuracy": bc_train_accuracy,
            "trainable_params": int(trainable),
            "p50_ms": latency["p50_ms"],
            "p95_ms": latency["p95_ms"],
            "p99_ms": latency["p99_ms"],
            "latency_mean_ms": latency["mean_ms"],
            "policy_kind": config["policy"],
            "eval_kind": config["eval"],
            "hard": hard,
            "expert_survival_mean": expert_stats["survival_mean"],
            "expert_success_rate": expert_stats["success_rate"],
        }
    )
    if config["policy"] == "brain":
        assert isinstance(model, SparseBrainPolicy)
        metrics["recurrent_edges"] = model.edge_count
        metrics["sensory_edges"] = model.sensory_edge_count
        metrics["neurons"] = model.neurons
        metrics["recurrent_steps"] = model.steps
    timings["total_seconds"] = total_timer.seconds
    return {
        "experiment": "cartpole",
        "config": config,
        "seed": int(seed),
        "device": DEVICE,
        "metrics": metrics,
        "timing": timings,
        "notes": " ".join(notes),
    }


# ---------------------------------------------------------------------------
# Self-test and smoke modes
# ---------------------------------------------------------------------------


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_self_test(seed: int = 0) -> int:
    checks: list[dict[str, Any]] = []

    def record(name: str, fn: Callable[[], str]) -> None:
        try:
            detail = fn()
        except AssertionError as exc:
            checks.append({"name": name, "passed": False, "detail": str(exc)})
            log(f"FAIL {name}: {exc}")
        except Exception as exc:  # pragma: no cover - unexpected test error
            checks.append({"name": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
            log(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            checks.append({"name": name, "passed": True, "detail": detail})
            log(f"ok {name}: {detail}")

    def physics_free_motion() -> str:
        plant = replace(Plant(), gravity=0.0)
        state = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
        following = rk4_step(state, torch.zeros(1, dtype=torch.float64), plant)
        _assert(abs(float(following[0, 0]) - 0.02) < 1e-12, "constant-velocity cart is not exact")
        _assert(abs(float(following[0, 2])) < 1e-15, "theta drifted without torque")
        return "F=0, g=0, x_dot=1 gives x=0.02 m and theta=0"

    def physics_energy() -> str:
        plant = Plant()
        state = torch.tensor([[0.0, 0.0, 0.5, 0.0]], dtype=torch.float64)
        initial = float(total_energy(state, plant))
        drift = 0.0
        for _ in range(plant.max_steps):
            state = rk4_step(state, torch.zeros(1, dtype=torch.float64), plant)
            drift = max(drift, abs(float(total_energy(state, plant)) - initial) / abs(initial))
        _assert(drift < 1e-4, f"energy drift {drift:.2e} exceeds 1e-4")
        return f"relative energy drift {drift:.2e} over {plant.max_steps} RK4 steps"

    def lqr_acceptance() -> str:
        plant = Plant()
        gain, _, iterations = solve_dare(plant)
        _assert(bool(torch.isfinite(gain).all()), "LQR gain is not finite")
        _assert(float(gain[2]) < 0.0, "theta gain sign is wrong for u=-Kx")
        stats = summarize(run_batch(build_expert(plant)[0], plant, episodes=100, seed=0))
        _assert(stats["survival_mean"] >= 190.0, f"mean survival {stats['survival_mean']:.1f} < 190")
        _assert(stats["success_rate"] >= 0.95, f"success rate {stats['success_rate']:.2f} < 0.95")
        return (
            f"K={[round(float(v), 3) for v in gain]} ({iterations} DARE iterations), "
            f"survival {stats['survival_mean']:.1f}/200, success {stats['success_rate']:.2f}"
        )

    def action_shapes() -> str:
        states = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.3, -0.2, 0.05, 0.4],
                [-1.0, 0.5, -0.12, -0.9],
                [2.0, 1.0, 0.18, 2.0],
                [-2.2, -1.5, -0.19, -1.5],
                [0.01, 0.02, 0.003, 0.004],
                [1.5, -2.0, 0.15, 0.1],
            ],
            dtype=torch.float32,
        )
        expert = build_expert(Plant())[0]
        mlp = TorchPolicy(MLPPolicyModel(hidden=16))
        brain = TorchPolicy(
            SparseBrainPolicy(neurons=16, degree=4, steps=2, seed=seed)
        )
        for name, policy in (("lqr", expert), ("mlp", mlp), ("brain", brain)):
            logits = policy.act(states)
            _assert(logits.shape == (7,), f"{name} returned shape {tuple(logits.shape)}")
            _assert(bool(((logits >= 0) & (logits < ACTIONS)).all()), f"{name} produced invalid actions")
        with torch.no_grad():
            mlp_logits = mlp.model(states)
            brain_logits, history = brain.model(states, return_states=True)
        _assert(mlp_logits.shape == (7, ACTIONS), "MLP logits shape is wrong")
        _assert(brain_logits.shape == (7, ACTIONS), "brain logits shape is wrong")
        _assert(bool(torch.isfinite(brain_logits).all()), "brain logits are not finite")
        _assert(len(history) == 3 and history[0].shape == (7, 16), "brain history shape is wrong")
        _assert(all(bool(torch.isfinite(item).all()) for item in history), "brain state is not finite")
        return "lqr/mlp/brain actions in {0,1,2}; brain logits [7,3]; history [3 x (7,16)]"

    def determinism() -> str:
        plant = Plant()
        expert = build_expert(plant)[0]
        first = run_batch(expert, plant, episodes=8, seed=seed)
        second = run_batch(expert, plant, episodes=8, seed=seed)
        _assert(torch.equal(first["survival"], second["survival"]), "LQR episodes are not repeatable")
        data = collect_bc_data(expert, plant, samples=256, seed=seed, noise_prob=0.1)
        states, labels = data
        clone_states, clone_labels = collect_bc_data(expert, plant, samples=256, seed=seed, noise_prob=0.1)
        _assert(torch.equal(states, clone_states), "BC data collection is not repeatable")
        _assert(torch.equal(labels, clone_labels), "BC labels are not repeatable")
        models = []
        for _ in range(2):
            seed_everything(seed)
            model = MLPPolicyModel(hidden=16)
            model.obs_mean.copy_(states.mean(0))
            model.obs_std.copy_(states.std(0).clamp_min(1e-6))
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(states[:128]), labels[:128])
                loss.backward()
                optimizer.step()
            models.append(model)
        for left, right in zip(models[0].parameters(), models[1].parameters()):
            _assert(torch.equal(left, right), "MLP training is not bitwise repeatable")
        return "repeated LQR episodes, data collection and MLP training are bitwise identical"

    def online_semantics() -> str:
        seed_everything(seed)
        config = dict(DEFAULTS)
        config.update(
            {
                "policy": "brain",
                "neurons": 16,
                "degree": 4,
                "steps": 2,
                "episodes": 4,
                "change_episode": 2,
                "variant": "online",
            }
        )
        brain = build_model(config, seed)
        assert isinstance(brain, SparseBrainPolicy)
        online = run_online_variant(brain, "online", config, seed=seed, change_episode=2)
        frozen = run_online_variant(brain, "frozen", config, seed=seed, change_episode=2)
        _assert(online["survival_pre"] == frozen["survival_pre"], "pre-change phases differ")
        _assert(online["weight_change_l2"] > 0.0, "online variant changed no weights")
        _assert(frozen["weight_change_l2"] == 0.0, "frozen variant changed weights")
        _assert(len(online["per_episode"]) == 4, "per-episode trace is incomplete")
        return (
            f"online weight delta L2 {online['weight_change_l2']:.2e}; "
            f"frozen pre/post {frozen['survival_pre']}/{frozen['survival_post']}"
        )

    def hard_protocol() -> str:
        config = dict(DEFAULTS)
        config.update({"hard": True})
        plant = make_plant(config, hard=True)
        protocol = make_protocol(config, impulse=True, hard=True)
        _assert(plant.force_mag == 6.0, "hard actuator is not 6 N")
        _assert(protocol.impulse, "hard protocol has no impulse")
        generator = torch.Generator().manual_seed(seed)
        states = sample_initial_states(512, generator, protocol)
        _assert(float(states[:, 2].abs().max()) <= 0.15 + 1e-12, "theta init exceeds 0.15")
        _assert(float(states[:, 2].abs().max()) > 0.13, "theta init never approaches the bound")
        other = states[:, [0, 1, 3]]
        _assert(float(other.abs().max()) <= 0.1 + 1e-12, "non-theta init exceeds 0.1")
        again = sample_initial_states(512, torch.Generator().manual_seed(seed), protocol)
        _assert(torch.equal(states, again), "hard initial states are not repeatable")
        start, sign = sample_impulse_schedule(64, torch.Generator().manual_seed(seed), protocol)
        _assert(int(start.min()) >= 60 and int(start.max()) <= 140, "kick step outside [60,140]")
        _assert(set(sign.tolist()) == {-1.0, 1.0}, "kick signs are not +-1")
        expert = build_expert(Plant())[0]
        first = run_batch(expert, plant, episodes=20, seed=seed, protocol=protocol)
        second = run_batch(expert, plant, episodes=20, seed=seed, protocol=protocol)
        _assert(torch.equal(first["survival"], second["survival"]), "hard episodes are not repeatable")
        _assert(int(first["survival"].max()) <= plant.max_steps, "survival exceeds max_steps")
        return (
            f"hard plant 6 N, theta init max {float(states[:, 2].abs().max()):.3f} rad, "
            f"kick steps {int(start.min())}-{int(start.max())}, deterministic"
        )

    record("physics_free_motion", physics_free_motion)
    record("physics_energy", physics_energy)
    record("lqr_acceptance", lqr_acceptance)
    record("action_shapes", action_shapes)
    record("determinism", determinism)
    record("hard_protocol", hard_protocol)
    record("online_semantics", online_semantics)
    passed = all(check["passed"] for check in checks)
    emit_result(
        {
            "experiment": "cartpole",
            "mode": "self_test",
            "seed": int(seed),
            "device": DEVICE,
            "passed": passed,
            "checks": checks,
            "metrics": {
                "checks": len(checks),
                "failed": sum(1 for check in checks if not check["passed"]),
            },
        }
    )
    return 0 if passed else 1


def run_smoke(seed: int, total_timer: Timer) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    problems: list[str] = []
    plans = [
        ("lqr_closed_loop", {"policy": "lqr", "eval": "closed_loop", "episodes": 5}),
        ("mlp_bc", {"policy": "mlp", "eval": "closed_loop", "episodes": 5, "bc_samples": 500, "epochs": 1}),
        ("brain_bc", {"policy": "brain", "eval": "closed_loop", "episodes": 5, "bc_samples": 500, "epochs": 1}),
        (
            "online",
            {
                "policy": "brain",
                "eval": "online",
                "episodes": 4,
                "bc_samples": 500,
                "epochs": 1,
                "change_episode": 2,
                "variant": "online",
            },
        ),
    ]
    for name, overrides in plans:
        config = dict(DEFAULTS)
        config.update(overrides)
        timer = Timer()
        payload = run_experiment(config, seed)
        metrics = payload["metrics"]
        runs.append(
            {
                "name": name,
                "seconds": timer.seconds,
                "metrics": {
                    key: metrics.get(key)
                    for key in (
                        "survival_mean",
                        "success_rate",
                        "bc_accuracy",
                        "p50_ms",
                        "survival_pre",
                        "survival_post",
                        "adaptation_gain",
                        "weight_change_l2",
                    )
                },
                "timing": payload["timing"],
            }
        )
        if name == "lqr_closed_loop":
            if metrics.get("survival_mean", 0.0) < 190.0:
                problems.append(f"lqr survival {metrics.get('survival_mean')} < 190")
        elif name in ("mlp_bc", "brain_bc"):
            accuracy = metrics.get("bc_accuracy")
            if accuracy is None or not 0.0 <= float(accuracy) <= 1.0:
                problems.append(f"{name} bc_accuracy is invalid: {accuracy}")
        else:
            if metrics.get("survival_pre") is None or metrics.get("survival_post") is None:
                problems.append("online run did not produce pre/post survival")
        log(f"smoke {name} ok in {timer.seconds:.2f}s")
    return {
        "experiment": "cartpole",
        "mode": "smoke",
        "config": {"smoke": True, **{key: DEFAULTS[key] for key in ("neurons", "degree", "steps", "hidden")}},
        "seed": int(seed),
        "device": DEVICE,
        "metrics": {
            "runs": runs,
            "total_seconds": total_timer.seconds,
            "all_passed": not problems,
            "problems": problems,
        },
        "timing": {
            "total_seconds": total_timer.seconds,
            "per_run_seconds": [run["seconds"] for run in runs],
        },
        "notes": (
            "Smoke mode: LQR 5 episodes, MLP BC 500 samples/1 epoch, brain BC "
            "500 samples/1 epoch, online 4 episodes (plant change at episode 2)."
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", default=None, help="JSON overrides for the defaults")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="accepted for compatibility; RESULT is always emitted")
    parser.add_argument("--smoke", action="store_true", help="tiny fast end-to-end configuration")
    parser.add_argument("--self-test", action="store_true", dest="self_test", help="run the physics/LQR/determinism checks")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test(seed=args.seed)

    total_timer = Timer()
    if args.smoke:
        payload = run_smoke(args.seed, total_timer)
        emit_result(payload)
        return 0

    config = parse_config(args.config_json)
    log(
        f"policy={config['policy']} eval={config['eval']} episodes={config['episodes']} "
        f"seed={args.seed} device={DEVICE}"
    )
    payload = run_experiment(config, args.seed)
    metrics = payload["metrics"]
    log(
        f"done in {payload['timing']['total_seconds']:.2f}s "
        f"(survival {metrics.get('survival_mean')}, bc {metrics.get('bc_accuracy')})"
    )
    emit_result(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
