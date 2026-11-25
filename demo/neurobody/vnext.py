"""NeuroBody vNext: a minimal causal-control experiment.

The question is deliberately narrow:

    can the 5,000-neuron brain drive the body through a descending M1 -> motor
    projection, without a future-reference teacher?

The existing ``EmbodiedBrain`` had no brain -> motor edge, so the action loss
never reached the recurrent brain.  ``VNextBrain`` adds that projection while
keeping the neuron count, types and graph size otherwise frozen.  The tasks are
kinematic (no gravity yet), so they measure command tracking and disturbance
recovery, not balance.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embodied import EmbodiedBrain, EmbodiedConfig
from neurosim_gpu import DendriticDynamics, Dynamics


NeuralState = torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# --------------------------------------------------------------------------- brain
class VNextBrain(EmbodiedBrain):
    """EmbodiedBrain + optional M1 -> motor and a plain-RNN dynamics baseline."""

    def __init__(
        self,
        config: EmbodiedConfig,
        n_symbols: int = 0,
        seed: int = 7,
        *,
        dynamics: str = "neurosim",
        freeze_recurrent: bool = False,
        m1_to_motor: int = 8,
        error_to_brain: int = 0,
        error_to_motor: bool = False,
        motor_recurrence: float = 0.1,
        reflex_gain: float = 1.0,
        plan_horizon: int = 0,
        plan_hidden: int = 128,
        dendrite_tau: float = 4.0,
        adaptation_tau: float = 25.0,
        compartment_coupling: float = 0.5,
        adaptation_strength: float = 0.2,
        dale_constrained: bool | None = None,
    ) -> None:
        super().__init__(
            config,
            n_symbols=n_symbols,
            seed=seed,
            m1_to_motor=m1_to_motor,
            error_to_brain=error_to_brain,
            error_to_motor=error_to_motor,
            motor_recurrence=motor_recurrence,
            reflex_gain=reflex_gain,
        )
        if dynamics not in ("neurosim", "plain", "dendritic"):
            raise ValueError("dynamics must be 'neurosim', 'plain' or 'dendritic'")
        self.dynamics = dynamics
        self.seed = int(seed)
        self.dale_constrained = (
            dynamics == "dendritic" if dale_constrained is None else bool(dale_constrained)
        )
        self.compartment_parameters = {
            "dt": 1.0,
            "dendrite_tau": float(dendrite_tau),
            "adaptation_tau": float(adaptation_tau),
            "coupling": float(compartment_coupling),
            "adaptation_strength": float(adaptation_strength),
        }
        self._dendritic_dynamics: DendriticDynamics | None = None
        if dynamics == "dendritic":
            compartment = DendriticDynamics.from_time_constants(
                Dynamics(self.alpha, self.gain),
                dt=1.0,
                tau_dendrite=dendrite_tau,
                tau_adaptation=adaptation_tau,
                coupling=compartment_coupling,
                adaptation_strength=adaptation_strength,
            )
            self.register_buffer("dendritic_decay", compartment.dendrite_decay)
            self.register_buffer("adaptation_decay", compartment.adaptation_decay)
            # Scalar inputs are broadcast views; clone them so checkpoint
            # load_state_dict can safely copy into each per-neuron entry.
            self.register_buffer("compartment_coupling", compartment.coupling.clone())
            self.register_buffer("adaptation_strength", compartment.adaptation_strength.clone())
            self._dendritic_dynamics = self._make_dendritic_dynamics()
        # ``freeze_recurrent`` freezes only brain<->brain edges.  The descending
        # M1->motor projection and the motor readout stay trainable, so the
        # reservoir baseline has the same output interface as the full model.
        self.freeze_brain = bool(freeze_recurrent)
        if self.freeze_brain:
            self.dendrite.requires_grad_(False)
        brain = config.brain_neurons
        src = self.src.to(torch.int64)
        dst = self.dst.to(torch.int64)
        mask = (src < brain) & (dst < brain)
        self.register_buffer("brain_edge_mask", mask)
        self.register_buffer("frozen_brain_weight", self.edge_weight.detach().clone())
        self.plan_horizon = int(plan_horizon)
        if self.plan_horizon > 0:
            # Planning is a readout of the current brain state, not a separate
            # controller: predict H future absolute joint targets.
            self.plan_head = nn.Sequential(
                nn.Linear(self.config.n_neurons + 82, plan_hidden),
                nn.Tanh(),
                nn.Linear(plan_hidden, self.plan_horizon * 41),
            )
            self.counterfactual_head = nn.Sequential(
                nn.Linear(
                    self.config.n_neurons + 82 + self.plan_horizon * 41,
                    plan_hidden,
                ),
                nn.Tanh(),
                nn.Linear(plan_hidden, 1),
            )
            # Start as "repeat the current goal"; the head learns a residual.
            nn.init.zeros_(self.plan_head[-1].weight)
            nn.init.zeros_(self.plan_head[-1].bias)

    def _make_dendritic_dynamics(self) -> DendriticDynamics:
        return DendriticDynamics(
            Dynamics(self.alpha, self.gain),
            self.dendritic_decay,
            self.adaptation_decay,
            self.compartment_coupling,
            self.adaptation_strength,
        )

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        if self.dynamics == "dendritic":
            # nn.Module.to() may replace registered tensors; refresh the cached
            # pure update object so its parameters follow the module device/dtype.
            self._dendritic_dynamics = self._make_dendritic_dynamics()
        return result

    def _csr(self):
        values = self.edge_weight
        if self.dale_constrained:
            # Keep each source neuron's outgoing sign fixed (Dale constraint),
            # while allowing the learned edge magnitude to vary.
            source_type = self.type_ids[self.src.to(torch.int64)]
            inhibitory = (source_type == 1) | (source_type == 5)
            sign = torch.where(inhibitory, -torch.ones_like(values), torch.ones_like(values))
            values = sign * values.abs()
        values = values[self.in_weight_index.to(torch.int64)]
        return torch.sparse_csr_tensor(
            self.in_rowptr, self.in_col, values,
            size=(self.config.n_neurons, self.config.n_neurons), check_invariants=False,
        )

    def initial_neural_state(self, drive: torch.Tensor) -> NeuralState:
        soma = self.initial_state(drive)
        if self.dynamics != "dendritic":
            return soma
        return soma, torch.zeros_like(soma), torch.zeros_like(soma)

    @staticmethod
    def soma_activity(state: NeuralState) -> torch.Tensor:
        return state[0] if isinstance(state, tuple) else state

    @staticmethod
    def with_soma_activity(
        state: NeuralState, soma: torch.Tensor
    ) -> NeuralState:
        if isinstance(state, tuple):
            return soma, state[1], state[2]
        return soma

    @staticmethod
    def detach_neural_state(state: NeuralState) -> NeuralState:
        if isinstance(state, tuple):
            return tuple(value.detach() for value in state)  # type: ignore[return-value]
        return state.detach()

    def plan(
        self, state: torch.Tensor, goal: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        if self.plan_horizon <= 0:
            raise RuntimeError("plan_horizon is 0")
        out = self.plan_head(torch.cat([state, goal, q], dim=-1))
        residual = out.reshape(state.shape[0], self.plan_horizon, 41)
        return goal[:, None, :] + 0.1 * residual

    def counterfactual_error(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        q: torch.Tensor,
        imagined: torch.Tensor,
    ) -> torch.Tensor:
        if self.plan_horizon <= 0:
            raise RuntimeError("plan_horizon is 0")
        out = self.counterfactual_head(
            torch.cat([state, goal, q, imagined.reshape(state.shape[0], -1)], dim=-1)
        )
        return out[:, 0]

    def advance(self, state: NeuralState, drive: torch.Tensor) -> NeuralState:
        if self.dynamics == "dendritic":
            if not isinstance(state, tuple) or len(state) != 3:
                raise ValueError("dendritic dynamics require (soma, dendrite, adaptation) state")
            if self._dendritic_dynamics is None:
                raise RuntimeError("dendritic dynamics were not initialized")
            soma, dendrite, adaptation = state
            current = torch.sparse.mm(self._csr(), soma.T).T
            return self._dendritic_dynamics.advance(
                soma, dendrite, adaptation, current, drive
            )
        if self.dynamics == "neurosim":
            return super().advance(state, drive)
        # Plain tanh RNN on the same graph: same edges, same readout, no
        # per-type alpha/gain and no dendritic coincidence term.
        current = torch.sparse.mm(self._csr(), state.T).T
        return torch.tanh(current + drive)

    def trainable_parameters(self) -> list[nn.Parameter]:
        parameters = [p for p in self.parameters() if p.requires_grad]
        if self.dynamics == "dendritic":
            # The static coincidence coefficient belongs to the older point
            # update and is not consulted by the compartmental state equation.
            parameters = [p for p in parameters if p is not self.dendrite]
        return parameters


# --------------------------------------------------------------------------- task
def _profile_velocity(t: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
    """0 -> amplitude -> 0 piecewise command, shape (B, 2)."""
    phase = torch.where(
        t < 60,
        torch.zeros_like(t),
        torch.where(t < 180, torch.ones_like(t), torch.zeros_like(t)),
    )
    return torch.stack([phase * amplitude, torch.zeros_like(phase)], dim=-1)


class ServoTask:
    """Kinematic 41-DoF task on the existing ``JointBody`` servo.

    ``posture``: hold a constant posture after a random initial perturbation and
    a mid-episode push.  ``velocity``: track a root-velocity command profile.
    Both use sensor noise and an actuator delay; neither has gravity, so neither
    is a balance task.
    """

    def __init__(
        self,
        mode: str,
        *,
        batch: int,
        device: torch.device,
        seed: int = 7,
        dt: float = 1.0 / 30.0,
        max_step: float = 0.12,
        episode_ticks: int = 320,
        push_tick: int = 60,
        delay: int = 1,
        sensor_noise: float = 0.01,
        velocity_noise: float = 0.05,
        velocity_scale: float = 0.3,
        root_max_step: float = 0.02,
        joint_leak: float = 0.05,
        joint_load: float = 0.02,
        root_leak: float = 0.0,
        root_load: float = 0.0,
        target_a: torch.Tensor | None = None,
        target_b: torch.Tensor | None = None,
        switch_b_scale: float = 1.0,
        switch_cycles: int = 1,
        switch_loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        goal_mode: str = "cycle",
        goal_dwell: int = 150,
        goal_targets: tuple[torch.Tensor, ...] | None = None,
        action_lowpass: float = 0.0,
        max_accel: float = 0.0,
        smoothness_weight: float = 0.0,
    ) -> None:
        if mode not in ("posture", "velocity", "switch", "goals"):
            raise ValueError("mode must be 'posture', 'velocity', 'switch' or 'goals'")
        self.mode = mode
        self.batch = batch
        self.device = device
        self.dt = dt
        self.max_step = max_step
        self.root_max_step = root_max_step
        if mode in ("posture", "switch", "goals"):
            # Joint-space tasks: do not let the policy translate the root,
            # otherwise it can "solve" the task by walking away.
            self.root_max_step = 0.0
        self.joint_leak = joint_leak
        self.joint_load = joint_load
        self.root_leak = root_leak
        self.root_load = root_load
        self.episode_ticks = episode_ticks
        self.push_tick = push_tick
        self.delay = delay
        self.sensor_noise = sensor_noise
        self.velocity_noise = velocity_noise
        self.velocity_scale = velocity_scale
        self.n_joints = 41
        self.joint_count = 39
        self.switch_cycles = max(1, int(switch_cycles))
        self.switch_split = max(1, int(episode_ticks) // (3 * self.switch_cycles))
        self.switch_b_scale = float(switch_b_scale)
        self.switch_loss_weights = tuple(float(value) for value in switch_loss_weights)
        if len(self.switch_loss_weights) != 3:
            raise ValueError("switch_loss_weights must have three entries (A, B, A2)")
        if target_a is None:
            target_a = torch.zeros(41)
        if target_b is None:
            target_b = (
                self._default_switch_b() * float(switch_b_scale)
                if mode == "switch"
                else torch.zeros(41)
            )
        self.target_a = torch.as_tensor(target_a, dtype=torch.float32, device=device).clone()
        self.target_b = torch.as_tensor(target_b, dtype=torch.float32, device=device).clone()
        self.goal_mode = goal_mode
        self.goal_dwell = max(1, int(goal_dwell))
        if goal_targets is None:
            c = self._default_switch_b() * (-0.8 * float(switch_b_scale))
            goal_targets = (
                torch.zeros(41),
                self._default_switch_b() * float(switch_b_scale),
                c,
            )
        self.goal_targets = tuple(
            torch.as_tensor(item, dtype=torch.float32, device=device).clone()
            for item in goal_targets
        )
        if self.goal_mode not in ("cycle", "random", "continuous"):
            raise ValueError("goal_mode must be 'cycle', 'random' or 'continuous'")
        self.goal_index = 0
        self.action_lowpass = float(action_lowpass)
        if not 0.0 <= self.action_lowpass < 1.0:
            raise ValueError("action_lowpass must lie in [0, 1)")
        self.max_accel = max(0.0, float(max_accel))
        self.smoothness_weight = max(0.0, float(smoothness_weight))
        self.rng = np.random.default_rng(seed)
        self.q_goal = torch.zeros(41, device=device)
        self.t = torch.zeros(batch, dtype=torch.long, device=device)
        self.q = torch.zeros(batch, 41, device=device)
        self.qd = torch.zeros(batch, 41, device=device)
        self.prev_q = torch.zeros(batch, 41, device=device)
        self.v_cmd = torch.zeros(batch, 2, device=device)
        self.amplitude = torch.zeros(batch, device=device)
        self.push = torch.zeros(batch, 41, device=device)
        self.pushed = torch.zeros(batch, dtype=torch.bool, device=device)
        self.delay_buffer = torch.zeros(batch, delay + 1, 41, device=device)
        self.prev_action = torch.zeros(batch, 41, device=device)
        self.prev_action_before = torch.zeros(batch, 41, device=device)
        self.last_velocity = torch.zeros(batch, 41, device=device)
        self.reset()

    # ------------------------------------------------------------------ reset
    @staticmethod
    def _default_switch_b() -> torch.Tensor:
        """Asymmetric target B: left side up, right side down, plus spine."""

        target = torch.zeros(41)
        target[0] = 1.0    # leftUpperArm x
        target[1] = 0.4    # leftUpperArm y
        target[2] = 0.2    # leftUpperArm z
        target[3] = 0.5    # leftLowerArm x
        target[6] = -1.0   # rightUpperArm x
        target[7] = -0.4   # rightUpperArm y
        target[8] = -0.2   # rightUpperArm z
        target[9] = -0.5   # rightLowerArm x
        target[12] = 0.5   # leftUpperLeg x
        target[15] = 0.3   # leftLowerLeg x
        target[18] = -0.5  # rightUpperLeg x
        target[21] = -0.3  # rightLowerLeg x
        target[33] = 0.2   # spine x
        target[36] = 0.2   # chest x
        return target

    def _update_goal(self) -> None:
        count = len(self.goal_targets)
        tick = int(self.t[0]) if self.t.numel() else 0
        if self.goal_mode == "cycle":
            index = (tick // self.goal_dwell) % count
            self.goal_index = index
            self.q_goal = self.goal_targets[index]
        elif self.goal_mode == "random":
            if tick > 0 and tick % self.goal_dwell == 0:
                index = int(self.rng.integers(0, count))
                if index == self.goal_index:
                    index = (index + 1) % count
                self.goal_index = index
                self.q_goal = self.goal_targets[index]
        else:  # continuous
            index = (tick // self.goal_dwell) % count
            local = (tick % self.goal_dwell) / self.goal_dwell
            smooth = local * local * (3.0 - 2.0 * local)
            start = self.goal_targets[index]
            end = self.goal_targets[(index + 1) % count]
            self.goal_index = index
            self.q_goal = start + smooth * (end - start)

    def _switch_goal(self) -> torch.Tensor:
        tick = int(self.t[0]) if self.t.numel() else 0
        period = 3 * self.switch_split
        phase = tick % period
        if phase < self.switch_split or phase >= 2 * self.switch_split:
            return self.target_a
        return self.target_b

    def set_switch_b_scale(self, scale: float) -> None:
        """Curriculum hook: change target-B amplitude without changing A."""

        self.switch_b_scale = float(scale)
        self.target_b = torch.as_tensor(
            self._default_switch_b() * self.switch_b_scale,
            dtype=torch.float32,
            device=self.device,
        )
        if self.mode == "switch":
            self.q_goal = self._switch_goal()

    def reset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = self.batch
        device = self.device
        self.t = torch.zeros(b, dtype=torch.long, device=device)
        if self.mode == "goals":
            self.goal_index = 0
            self.q_goal = self.goal_targets[0].clone()
        elif self.mode in ("posture", "switch"):
            self.q_goal = self.target_a.clone()
        else:
            self.q_goal = torch.zeros(41, device=device)
        current = int(self.rng.integers(0, 2**31 - 1))
        gen = torch.Generator(device=device).manual_seed(current)
        if self.mode == "posture":
            # Random initial perturbation around the target posture.
            self.q = self.q_goal + 0.3 * torch.randn(
                b, 41, generator=gen, device=device
            )
            self.q[:, self.joint_count :] *= 0.2
            self.amplitude = torch.zeros(b, device=device)
        elif self.mode in ("switch", "goals"):
            self.q = self.q_goal + 0.15 * torch.randn(
                b, 41, generator=gen, device=device
            )
            self.q[:, self.joint_count :] *= 0.2
            self.amplitude = torch.zeros(b, device=device)
        else:
            self.q = 0.05 * torch.randn(b, 41, generator=gen, device=device)
            self.q[:, self.joint_count :] *= 0.2
            self.amplitude = 0.4 + 0.2 * torch.rand(b, generator=gen, device=device)
        self.q[:, : self.joint_count].clamp_(-2.6, 2.6)
        self.qd = torch.zeros_like(self.q)
        self.prev_q = self.q.clone()
        self.v_cmd = _profile_velocity(self.t, self.amplitude)
        self.push = torch.zeros(b, 41, device=device)
        if self.mode == "posture":
            # A sparse random push on five joints.
            joints = torch.randint(
                0, self.joint_count, (b, 5), generator=gen, device=device
            )
            values = 0.25 * (2.0 * torch.rand(b, 5, generator=gen, device=device) - 1.0)
            self.push.scatter_(1, joints, values)
        self.pushed = torch.zeros(b, dtype=torch.bool, device=device)
        self.delay_buffer = torch.zeros(b, self.delay + 1, 41, device=device)
        self.prev_action = torch.zeros(b, 41, device=device)
        self.prev_action_before = torch.zeros(b, 41, device=device)
        self.last_velocity = torch.zeros(b, 41, device=device)
        return self.observation()

    def future_targets(self, horizon: int) -> torch.Tensor:
        """(B, H, 41) goals for the next ``horizon`` ticks."""

        goals = [self._goal_at(int(self.t[0]) + step) for step in range(horizon)]
        return torch.stack(goals, dim=0)[None, :, :].expand(self.batch, -1, -1)

    def _goal_at(self, tick: int) -> torch.Tensor:
        if self.mode == "switch":
            period = 3 * self.switch_split
            phase = tick % period
            if phase < self.switch_split or phase >= 2 * self.switch_split:
                return self.target_a
            return self.target_b
        if self.mode == "goals":
            count = len(self.goal_targets)
            if self.goal_mode == "cycle":
                return self.goal_targets[(tick // self.goal_dwell) % count]
            if self.goal_mode == "continuous":
                index = (tick // self.goal_dwell) % count
                local = (tick % self.goal_dwell) / self.goal_dwell
                smooth = local * local * (3.0 - 2.0 * local)
                start = self.goal_targets[index]
                end = self.goal_targets[(index + 1) % count]
                return start + smooth * (end - start)
            return self.q_goal
        return self.q_goal

    def imagine(self, actions: torch.Tensor) -> torch.Tensor:
        """Run the plant forward under an imagined action sequence (no noise)."""

        q = self.q.clone()
        for step in range(actions.shape[1]):
            velocity = actions[:, step].clamp(-self.max_step, self.max_step)
            velocity = torch.cat(
                [
                    velocity[:, : self.joint_count],
                    velocity[:, self.joint_count :].clamp(
                        -self.root_max_step, self.root_max_step
                    ),
                ],
                dim=1,
            )
            q = q + velocity
            joints = (1.0 - self.joint_leak) * q[:, : self.joint_count] + self.joint_load
            root = (1.0 - self.root_leak) * q[:, self.joint_count :] + self.root_load
            q = torch.cat([joints.clamp(-2.6, 2.6), root], dim=1)
        return q

    def observation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device
        q_noise = self.sensor_noise * torch.randn(
            self.batch, 41, device=device
        )
        qd_noise = self.velocity_noise * torch.randn(
            self.batch, 41, device=device
        )
        q_obs = self.q + q_noise
        qd_obs = self.qd + qd_noise
        # 10-dim vestibular proxy: root orientation (unobserved -> 0), root
        # angular velocity (0), root linear velocity, height.
        vest = torch.zeros(self.batch, 10, device=device)
        # The first two channels carry the current task command (a goal signal),
        # not a real vestibular measurement.  Without it the velocity task is
        # only observable through the error neurons, which is a much harder
        # credit-assignment problem.
        vest[:, 0] = self.v_cmd[:, 0]
        vest[:, 1] = self.v_cmd[:, 1]
        vest[:, 6] = qd_obs[:, self.joint_count]
        vest[:, 8] = qd_obs[:, self.joint_count + 1]
        vest[:, 9] = 0.9
        target = self.target()
        return q_obs, qd_obs, vest, target

    def target(self) -> torch.Tensor:
        target = self.q_goal[None, :].expand(self.batch, -1).clone()
        if self.mode == "velocity":
            target[:, self.joint_count :] = (
                self.q[:, self.joint_count :]
                + self.v_cmd * self.velocity_scale
            )
        return target

    # ------------------------------------------------------------------- step
    def step(
        self, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = self.batch
        device = self.device
        delayed = self.delay_buffer[:, 0].clone()
        if self.delay > 0:
            self.delay_buffer = torch.cat(
                [self.delay_buffer[:, 1:], velocity[:, None, :]], dim=1
            )
        else:
            delayed = velocity
        self.prev_action_before = self.prev_action
        velocity = delayed
        if self.action_lowpass > 0.0:
            # First-order low-pass on the commanded velocity: alpha near 1
            # suppresses tick-to-tick chatter.
            velocity = (
                (1.0 - self.action_lowpass) * velocity
                + self.action_lowpass * self.prev_action
            )
        if self.max_accel > 0.0:
            delta = (velocity - self.prev_action).clamp(
                -self.max_accel, self.max_accel
            )
            velocity = self.prev_action + delta
        velocity = velocity.clamp(-self.max_step, self.max_step)
        velocity = torch.cat(
            [
                velocity[:, : self.joint_count],
                velocity[:, self.joint_count :].clamp(
                    -self.root_max_step, self.root_max_step
                ),
            ],
            dim=1,
        )
        self.prev_action = velocity.detach()
        self.last_velocity = velocity
        self.prev_q = self.q.clone()
        self.q = self.q + velocity
        # Stable, non-trivial plant: joint leak + constant load (a "virtual
        # gravity").  A pure integrator would let any constant motor bias drift
        # to the clamp, which makes the control experiment unidentifiable.
        joints = (1.0 - self.joint_leak) * self.q[:, : self.joint_count] + self.joint_load
        root = (1.0 - self.root_leak) * self.q[:, self.joint_count :] + self.root_load
        self.q = torch.cat([joints.clamp(-2.6, 2.6), root], dim=1)
        self.qd = (self.q - self.prev_q) / self.dt

        # One push per episode, at ``push_tick``.
        if self.mode == "posture":
            fire = (self.t == self.push_tick) & (~self.pushed)
            if bool(fire.any()):
                self.q = self.q + self.push * fire[:, None].to(self.q.dtype)
                self.pushed = self.pushed | fire

        self.t = self.t + 1
        if self.mode == "switch":
            self.q_goal = self._switch_goal()
        elif self.mode == "goals":
            self._update_goal()
        self.v_cmd = _profile_velocity(self.t, self.amplitude)
        return self.observation()

    # ------------------------------------------------------------------- loss
    def loss(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        velocity: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        joints = q[:, : self.joint_count]
        goal = self.q_goal[None, : self.joint_count].expand_as(joints)
        effort = velocity.square().mean()
        smooth = (
            (velocity - self.prev_action_before).square().mean()
            if self.smoothness_weight > 0.0
            else torch.zeros((), device=velocity.device)
        )
        if self.mode == "switch":
            period = 3 * self.switch_split
            phase = self.t % period
            weight = torch.where(
                phase < self.switch_split,
                torch.full_like(phase, self.switch_loss_weights[0], dtype=q.dtype),
                torch.where(
                    phase < 2 * self.switch_split,
                    torch.full_like(phase, self.switch_loss_weights[1], dtype=q.dtype),
                    torch.full_like(phase, self.switch_loss_weights[2], dtype=q.dtype),
                ),
            )
            posture = ((joints - goal).square().mean(dim=1) * weight).mean()
            root_goal = self.q_goal[None, self.joint_count :].expand(q.shape[0], -1)
            root_error = (
                (q[:, self.joint_count :] - root_goal).square().mean(dim=1) * weight
            ).mean()
            return (
                posture
                + 0.1 * root_error
                + 0.005 * effort
                + self.smoothness_weight * smooth
            )
        posture = F.mse_loss(joints, goal)
        if self.mode == "velocity":
            root = qd[:, self.joint_count :]
            track = F.mse_loss(root, self.v_cmd)
            return (
                track
                + 0.1 * posture
                + 0.005 * effort
                + self.smoothness_weight * smooth
            )
        root_goal = self.q_goal[None, self.joint_count :].expand(q.shape[0], -1)
        root_error = F.mse_loss(q[:, self.joint_count :], root_goal)
        return (
            posture
            + 0.1 * root_error
            + 0.005 * effort
            + self.smoothness_weight * smooth
        )

    # ---------------------------------------------------------------- metrics
    def detach_(self) -> None:
        """Detach tensors that can carry a graph across a TBPTT boundary."""

        for name in ("q", "qd", "prev_q", "v_cmd", "push", "delay_buffer"):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.detach())

    def episode_metrics(
        self, q: torch.Tensor, qd: torch.Tensor, velocity: torch.Tensor
    ) -> dict[str, float]:
        joints = q[:, : self.joint_count]
        goal = self.q_goal[None, : self.joint_count].expand_as(joints)
        error = (joints - goal).abs().mean().item()
        out = {
            "postureError": error,
            "effort": velocity.abs().mean().item(),
            "rootDrift": (q[:, self.joint_count :] - self.q_goal[self.joint_count :])
            .abs()
            .mean()
            .item(),
        }
        if self.mode == "velocity":
            root = qd[:, self.joint_count :]
            out["velocityMae"] = (root - self.v_cmd).abs().mean().item()
            out["velocityMean"] = root[:, 0].mean().item()
        return out


# --------------------------------------------------------------------------- policies
class PDPolicy:
    """A hand-written proportional-derivative controller (performance baseline)."""

    name = "pd"

    def __init__(self, *, kp: float = 0.4, kd: float = 0.5, kv: float = 1.0) -> None:
        self.kp = kp
        self.kd = kd
        self.kv = kv
        self.n_joints = 41
        self.joint_count = 39

    def reset(self) -> None:
        pass

    def act(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        target: torch.Tensor,
        v_cmd: torch.Tensor,
        *,
        max_step: float = 0.12,
    ) -> torch.Tensor:
        dt = 1.0 / 30.0
        velocity = self.kp * (target - q) - self.kd * qd * dt
        velocity[:, self.joint_count :] = (
            v_cmd * dt
            + self.kv * (v_cmd - qd[:, self.joint_count :]) * dt
        )
        return velocity.clamp(-max_step, max_step)


class ReflexPolicy:
    """Fixed random spinal reflex + PD; no training (local-loop baseline)."""

    name = "reflex"

    def __init__(self, seed: int = 7, gain: float = 0.05) -> None:
        self.gain = gain
        self.n_joints = 41
        self.joint_count = 39
        rng = np.random.default_rng(seed)
        self.matrix = torch.as_tensor(
            rng.normal(0.0, 0.1, size=(41, 82)), dtype=torch.float32
        )

    def reset(self) -> None:
        pass

    def act(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        target: torch.Tensor,
        v_cmd: torch.Tensor,
        *,
        max_step: float = 0.12,
    ) -> torch.Tensor:
        error = target - q
        x = torch.cat([error, qd], dim=-1)
        reflex = torch.tanh(x @ self.matrix.T.to(x.device))
        dt = 1.0 / 30.0
        velocity = 1.0 * error - 0.5 * qd * dt + self.gain * reflex
        velocity[:, self.joint_count :] = (
            v_cmd * dt
            + 1.0 * (v_cmd - qd[:, self.joint_count :]) * dt
            + self.gain * reflex[:, self.joint_count :]
        )
        return velocity.clamp(-max_step, max_step)


class BrainPolicy:
    """Wraps a ``VNextBrain`` into the task interface."""

    name = "brain"

    def __init__(self, brain: VNextBrain) -> None:
        self.brain = brain
        self.state: NeuralState | None = None
        self._vest: torch.Tensor | None = None

    def reset(self) -> None:
        self.state = None

    def act(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        target: torch.Tensor,
        v_cmd: torch.Tensor,
        *,
        max_step: float = 0.12,
    ) -> torch.Tensor:
        vest = self._vest
        if vest is None:
            vest = torch.zeros(q.shape[0], 10, device=q.device, dtype=q.dtype)
        drive = self.brain.encode(q, qd, vest, None, target)
        if self.state is None:
            self.state = self.brain.initial_neural_state(drive)
        self.state = self.brain.advance(self.state, drive)
        return self.brain.motor_velocity(self.brain.soma_activity(self.state))

    def set_vest(self, vest: torch.Tensor) -> None:
        self._vest = vest


# --------------------------------------------------------------------------- training
@dataclass
class TrainConfig:
    ticks: int = 1600
    batch: int = 8
    lr: float = 1e-3
    window: int = 8
    seed: int = 7
    episode_ticks: int = 320
    self_weight: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 50
    b_scale_schedule: tuple[tuple[float, float], ...] = ()
    plan_horizon: int = 0
    plan_weight: float = 0.0
    counterfactual_weight: float = 0.0


def train_brain(
    brain: VNextBrain,
    task: ServoTask,
    cfg: TrainConfig,
    *,
    device: torch.device,
    log_path: Path | None = None,
) -> dict[str, Any]:
    brain.to(device).train()
    optimizer = torch.optim.AdamW(brain.trainable_parameters(), lr=cfg.lr, weight_decay=1e-5)
    q, qd, vest, target = task.reset()
    state: NeuralState | None = None
    previous: NeuralState | None = None
    losses: list[torch.Tensor] = []
    self_losses: list[torch.Tensor] = []
    running = 0.0
    updates = 0
    last_loss = float("nan")
    started = time.perf_counter()
    records: list[dict[str, Any]] = []

    def _curriculum_scale(progress: float) -> float:
        scale = cfg.b_scale_schedule[0][1]
        for fraction, value in cfg.b_scale_schedule:
            if progress >= fraction:
                scale = value
        return scale

    if task.mode == "switch" and cfg.b_scale_schedule:
        task.set_switch_b_scale(_curriculum_scale(0.0))
    for tick in range(cfg.ticks):
        if tick > 0 and tick % task.episode_ticks == 0:
            if task.mode == "switch" and cfg.b_scale_schedule:
                task.set_switch_b_scale(_curriculum_scale(tick / max(cfg.ticks - 1, 1)))
            q, qd, vest, target = task.reset()
            state = None
            previous = None
            losses.clear()
            self_losses.clear()
        drive = brain.encode(q, qd, vest, None, target)
        if state is None:
            state = brain.initial_neural_state(drive)
        previous = state
        state = brain.advance(state, drive)
        soma = brain.soma_activity(state)
        velocity = brain.motor_velocity(soma)
        q, qd, vest, target = task.step(velocity)
        base_loss = task.loss(task.q, task.qd, velocity, target)
        if cfg.plan_horizon > 0 and getattr(brain, "plan_horizon", 0) > 0:
            future = task.future_targets(cfg.plan_horizon)
            planned = brain.plan(soma, target, task.q)
            plan_loss = F.mse_loss(planned, future)
            imagined = planned.detach() + 0.05 * torch.randn_like(planned)
            imagined_q = task.imagine(imagined)
            cf_target = (
                (imagined_q[:, :39] - future[:, -1, :39]).abs().mean(dim=1)
            )
            cf_pred = brain.counterfactual_error(
                soma, target, task.q, imagined
            )
            cf_loss = F.mse_loss(cf_pred, cf_target)
            base_loss = (
                base_loss
                + cfg.plan_weight * plan_loss
                + cfg.counterfactual_weight * cf_loss
            )
        losses.append(base_loss)
        if previous is not None:
            self_losses.append(
                F.mse_loss(
                    brain.predict_next(brain.soma_activity(previous)), soma
                )
            )
        if (tick + 1) % cfg.window == 0:
            total = torch.stack(losses).mean()
            if self_losses:
                total = total + cfg.self_weight * torch.stack(self_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if getattr(brain, "freeze_brain", False) and brain.edge_weight.grad is not None:
                brain.edge_weight.grad[brain.brain_edge_mask] = 0.0
            torch.nn.utils.clip_grad_norm_(brain.trainable_parameters(), cfg.grad_clip)
            optimizer.step()
            if getattr(brain, "freeze_brain", False):
                with torch.no_grad():
                    brain.edge_weight[brain.brain_edge_mask] = brain.frozen_brain_weight[
                        brain.brain_edge_mask
                    ]
            state = brain.detach_neural_state(state)
            q = q.detach()
            qd = qd.detach()
            vest = vest.detach()
            task.detach_()
            # ``target`` is a graph tensor for the velocity task; recompute it
            # from the detached state so the next window starts a fresh graph.
            target = task.target()
            previous = None
            losses.clear()
            self_losses.clear()
            updates += 1
            running += float(total.detach())
            last_loss = float(total.detach())
        if (tick + 1) % cfg.log_every == 0:
            log_loss = last_loss
            if math.isnan(log_loss) and losses:
                # Before the first TBPTT boundary there is no optimizer-step
                # loss yet; log the finite in-window mean rather than a NaN
                # placeholder so short smoke runs remain auditable.
                log_loss = float(torch.stack(losses).mean().detach())
            record = {
                "tick": tick + 1,
                "loss": log_loss,
                "updates": updates,
                "seconds": round(time.perf_counter() - started, 2),
            }
            records.append(record)
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            print(
                f"tick {tick + 1:5d}  loss {record['loss']:.5f}  "
                f"updates {updates}  {record['seconds']:6.1f}s",
                flush=True,
            )
    return {
        "updates": updates,
        "mean_loss": running / max(updates, 1),
        "seconds": round(time.perf_counter() - started, 2),
        "records": records,
    }


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(
    policy,
    task: ServoTask,
    *,
    episodes: int = 8,
    seed: int = 123,
    device: torch.device,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        torch.manual_seed(episode_seed)
        if task.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)
        task.rng = np.random.default_rng(seed + episode)
        q, qd, vest, target = task.reset()
        policy.reset()
        if isinstance(policy, BrainPolicy):
            policy.brain.eval()
        trajectory = []
        for tick in range(task.episode_ticks):
            if isinstance(policy, BrainPolicy):
                policy.set_vest(vest)
            velocity = policy.act(q, qd, target, task.v_cmd, max_step=task.max_step)
            q, qd, vest, target = task.step(velocity)
            trajectory.append(
                {
                    "q": task.q[0].cpu().numpy().copy(),
                    "qd": task.qd[0].cpu().numpy().copy(),
                    "velocity": velocity[0].cpu().numpy().copy(),
                    "v_cmd": task.v_cmd[0].cpu().numpy().copy(),
                    "goal": target[0].cpu().numpy().copy(),
                }
            )
        rows.append(_episode_summary(trajectory, task))
    keys = rows[0].keys()
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in keys
    } | {
        f"{key}Std": float(np.std([row[key] for row in rows]))
        for key in keys
    }


def _episode_summary(trajectory: list[dict[str, np.ndarray]], task: ServoTask) -> dict[str, float]:
    q = np.stack([row["q"] for row in trajectory])
    qd = np.stack([row["qd"] for row in trajectory])
    velocity = np.stack([row["velocity"] for row in trajectory])
    joints = q[:, :39]
    if task.mode == "switch":
        split = task.switch_split
        period = 3 * split
        a = task.target_a[:39].detach().cpu().numpy()
        b = task.target_b[:39].detach().cpu().numpy()
        target = np.empty_like(joints)
        for cycle in range(task.switch_cycles):
            base = cycle * period
            target[base : base + split] = a
            target[base + split : base + 2 * split] = b
            target[base + 2 * split : base + period] = a
        posture = np.abs(joints - target).mean(axis=1)
    elif task.mode == "goals":
        goal = np.stack([row["goal"] for row in trajectory])[:, :39]
        posture = np.abs(joints - goal).mean(axis=1)
    else:
        posture = np.abs(joints).mean(axis=1)
    effort = np.abs(velocity).mean()
    out = {
        "postureError": float(posture[-30:].mean()),
        "effort": float(effort),
        "rootDrift": float(np.abs(q[-1, 39:]).mean()),
    }
    if task.mode == "switch":
        split = task.switch_split
        period = 3 * split
        a_scores, b_scores, a2_scores = [], [], []
        for cycle in range(task.switch_cycles):
            base = cycle * period
            a_scores.append(float(posture[base : base + split].mean()))
            b_scores.append(float(posture[base + split : base + 2 * split].mean()))
            a2_scores.append(float(posture[base + 2 * split : base + period].mean()))
        out["switchA"] = float(np.mean(a_scores))
        out["switchB"] = float(np.mean(b_scores))
        out["switchA2"] = float(np.mean(a2_scores))
        out["switchPeak"] = float(posture.max())
        out["switchLastCycle"] = float(
            np.mean([a_scores[-1], b_scores[-1], a2_scores[-1]])
        )
        out["switchFirstCycle"] = float(
            np.mean([a_scores[0], b_scores[0], a2_scores[0]])
        )
        delta = b - a
        denom = float(np.linalg.norm(delta))
        ratios, onsets, settlings, overshoots = [], [], [], []
        if denom > 1e-9:
            for cycle in range(task.switch_cycles):
                base = cycle * period
                q_b = joints[base + split : base + 2 * split]
                q_b_end = q_b[-10:].mean(axis=0)
                ratios.append(float(np.linalg.norm(q_b_end - a) / denom))
                projection = ((q_b - a) @ delta) / (delta @ delta)
                reached = np.where(projection > 0.1)[0]
                onsets.append(float(reached[0]) if reached.size else float(split))
                error_b = np.abs(q_b - b).mean(axis=1)
                settled = np.where(error_b < 0.05)[0]
                settlings.append(float(settled[0]) if settled.size else float(split))
                overshoots.append(float(max(0.0, projection.max() - 1.0)))
        if ratios:
            out["bDisplacementRatio"] = float(np.mean(ratios))
            out["bOnsetTicks"] = float(np.mean(onsets))
            out["bSettlingTicks"] = float(np.mean(settlings))
            out["bOvershoot"] = float(np.mean(overshoots))
    if task.mode == "goals":
        window = min(task.goal_dwell, len(posture))
        out["goalFirst"] = float(posture[:window].mean())
        out["goalLast"] = float(posture[-window:].mean())
        out["goalPeak"] = float(posture.max())
        out["goalDrift"] = out["goalLast"] - out["goalFirst"]
    if task.mode == "posture":
        # Settling after the mid-episode push: first tick where the joint error
        # stays below 0.05 rad for 10 ticks.
        after = posture[task.push_tick :]
        settled = float("nan")
        for index in range(len(after) - 10):
            if np.all(after[index : index + 10] < 0.05):
                settled = float(index)
                break
        out["settlingTicks"] = float(len(after)) if math.isnan(settled) else settled
        out["overshoot"] = float(max(0.0, after.max() - posture[task.push_tick - 1]))
    elif task.mode == "velocity":
        command = np.stack([row["v_cmd"] for row in trajectory])[:, 0]
        root = qd[:, 39]
        mask = command > 0.1
        if mask.any():
            out["velocityMae"] = float(np.abs(root[mask] - command[mask]).mean())
            onset = int(np.argmax(mask))
            reached = np.where(root[onset:] > 0.9 * command[onset])[0]
            out["riseTicks"] = float(reached[0]) if reached.size else float(len(root) - onset)
        else:
            out["velocityMae"] = 0.0
            out["riseTicks"] = 0.0
    return out


# --------------------------------------------------------------------------- io
def save_checkpoint(path: Path, brain: VNextBrain, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": brain.state_dict(),
            "config": brain.config.__dict__,
            "dynamics": brain.dynamics,
            "dale_constrained": brain.dale_constrained,
            "m1_to_motor": brain.m1_to_motor,
            "error_to_brain": brain.error_to_brain,
            "error_to_motor": brain.error_to_motor,
            "motor_recurrence": brain.motor_recurrence,
            "reflex_gain": brain.reflex_gain,
            "freeze_brain": getattr(brain, "freeze_brain", False),
            "seed": brain.seed,
            "plan_horizon": getattr(brain, "plan_horizon", 0),
            "compartment_parameters": (
                dict(brain.compartment_parameters)
                if brain.dynamics == "dendritic"
                else None
            ),
            "meta": meta,
        },
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device, *, seed: int | None = None
) -> tuple[VNextBrain, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model_state = state["model"]
    config_values = dict(state["config"])
    stored_neurons = int(model_state["type_ids"].numel())
    # Early vNext checkpoints predate the extra touch/audio/memory pools. Their
    # serialized config therefore omits those fields, while current dataclass
    # defaults would silently add 584 neurons and change the graph shape.
    extra_fields = ("touch_neurons", "audio_neurons", "nocicept_neurons", "memory_neurons")
    if not any(name in config_values for name in extra_fields):
        zero_pool_values = config_values | {name: 0 for name in extra_fields}
        zero_pool_config = EmbodiedConfig(**zero_pool_values)
        default_config = EmbodiedConfig(**config_values)
        if stored_neurons == zero_pool_config.n_neurons:
            config_values.update({name: 0 for name in extra_fields})
        elif stored_neurons != default_config.n_neurons:
            raise ValueError(
                "legacy checkpoint neuron count cannot be inferred from its config; "
                "provide an explicit pool-size migration"
            )
    config = EmbodiedConfig(**config_values)
    if config.n_neurons != stored_neurons:
        raise ValueError(
            f"checkpoint has {stored_neurons} neurons but config reconstructs "
            f"{config.n_neurons}"
        )
    symbol_weight = model_state.get("symbol_head.weight")
    n_symbols = int(symbol_weight.shape[0]) if symbol_weight is not None else 0
    train_seed = int(state.get("seed", seed if seed is not None else 7))
    compartment_parameters = state.get("compartment_parameters") or {}
    brain = VNextBrain(
        config,
        n_symbols=n_symbols,
        seed=train_seed,
        dynamics=state.get("dynamics", "neurosim"),
        dale_constrained=state.get(
            "dale_constrained", state.get("dynamics", "neurosim") == "dendritic"
        ),
        freeze_recurrent=state.get("freeze_brain", False),
        m1_to_motor=state.get("m1_to_motor", 8),
        error_to_brain=state.get("error_to_brain", 0),
        error_to_motor=state.get("error_to_motor", False),
        motor_recurrence=state.get("motor_recurrence", 0.1),
        reflex_gain=state.get("reflex_gain", 1.0),
        plan_horizon=state.get("plan_horizon", 0),
        **{
            name: compartment_parameters[key]
            for name, key in (
                ("dendrite_tau", "dendrite_tau"),
                ("adaptation_tau", "adaptation_tau"),
                ("compartment_coupling", "coupling"),
                ("adaptation_strength", "adaptation_strength"),
            )
            if key in compartment_parameters
        },
    )

    # Topology construction samples a random graph and duplicate coalescing can
    # change edge count between seeds. Restore the checkpoint's exact sparse
    # structure before load_state_dict instead of assuming the regenerated graph
    # has matching shapes/order.
    for name in (
        "src", "dst", "in_rowptr", "in_col", "out_weight_index",
        "in_weight_index", "brain_edge_mask", "frozen_brain_weight",
    ):
        value = model_state.get(name)
        if value is not None and name in brain._buffers:
            brain._buffers[name] = value.detach().clone()
    saved_edge_weight = model_state.get("edge_weight")
    if saved_edge_weight is not None:
        brain.edge_weight = nn.Parameter(saved_edge_weight.detach().clone())
    brain.load_state_dict(model_state, strict=False)
    brain.eval().to(device)
    return brain, state.get("meta", {})
