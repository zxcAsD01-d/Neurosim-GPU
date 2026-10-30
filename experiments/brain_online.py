"""Faithful "brain" pipeline: retina + sparse core + local three-factor learning.

This is the route the project documents, not BPTT on the brain:

* a topographic **retina** (fixed positions, brightness only, no parameters) is
  wired into the *same* topology as the core, so retina->neuron edges are
  plastic like every other edge;
* learning is the engine's local online three-factor rule inside
  ``DopamineMatrixRuntime`` (``torch.no_grad``), batch=1, eligibility tracked,
  chemistry gated, and the reward injected one tick late;
* only the linear **readout** is trained by an external optimizer, which
  ``docs/gomoku-protocol.md`` Track B lists as an explicit hybrid variant.

Task: MNIST multi-digit strings (controlled, learnable), same as
``experiments/mnist_seq.py`` so results are comparable.

Usage::

    python experiments/brain_online.py --config-json '{"epochs":3,"learning_rate":0.0}'
    python experiments/brain_online.py --config-json '{"epochs":3,"learning_rate":0.002}'
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DATA, emit_result, load_mnist, seed_everything

from neurosim_gpu import (
    DopamineMatrixRuntime,
    DopaminePlasticityConfig,
    DopamineReceptors,
    Dynamics,
    RegionalDopamineField,
    SparseTopology,
)

from mnist_seq import build_sequences, shift_images
from str_brain import Charset, edit_distance_str

DEFAULTS: dict[str, Any] = {
    "img_h": 28,
    "img_w": 92,
    "scan_win": 10,
    "scan_stride": 4,
    "neurons": 512,
    "retina_fanout": 8,
    "degree": 8,
    "alpha": 0.6,
    "gain": 1.0,
    "weight_scale": 0.18,
    "learning_rate": 0.0,
    "eligibility_decay": 0.9,
    "chemistry_interval": 4,
    "dopamine_delay_steps": 0,
    "readout_lr": 3e-3,
    "epochs": 3,
    "train_n": 2000,
    "test_n": 500,
    "device": "auto",
    "seed": 0,
    "reward_mode": "sign",
    # "engine" uses neurosim_gpu.Dynamics unchanged. "soft_clamp_control" is an
    # explicitly labelled experiment-only arm used to attribute the dead-zone
    # effect; it never changes the engine.
    "cell_model": "engine",
    "smooth_beta": 10.0,
}


# ---------------------------------------------------------------------------
# System construction
# ---------------------------------------------------------------------------


class SoftClampDynamics:
    """EXPERIMENT-ONLY control arm -- **not** the engine cell.

    Identical to ``neurosim_gpu.Dynamics`` except that the hard
    ``clamp(x, 0, 1)`` is replaced by a smooth soft-clamp, so the dead zone
    (``x <= 0``) keeps a non-zero derivative.  ``src/neurosim_gpu`` is not
    modified; this object is only ever passed to the runtime from the
    experiment script, and every payload is labelled with ``CONTROL_ARM``.
    """

    CONTROL_ARM = "soft_clamp_control (experiment-only; engine unmodified)"

    def __init__(
        self,
        alpha: torch.Tensor,
        gain: torch.Tensor,
        *,
        smooth_beta: float = 10.0,
    ) -> None:
        self.alpha = alpha
        self.gain = gain
        self.smooth_beta = float(smooth_beta)

    @property
    def n_neurons(self) -> int:
        return self.alpha.numel()

    @property
    def beta(self) -> torch.Tensor:
        return 1.0 - self.alpha

    def advance(
        self,
        state: torch.Tensor,
        synaptic_current: torch.Tensor,
        drive: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        return self.advance_modulated(state, synaptic_current, drive)

    def advance_modulated(
        self,
        state: torch.Tensor,
        synaptic_current: torch.Tensor,
        drive: torch.Tensor | float = 0.0,
        *,
        drive_shift: torch.Tensor | float = 0.0,
        gain_scale: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        x = self.alpha * state + self.beta * self.gain * gain_scale * torch.tanh(
            synaptic_current + drive + drive_shift
        )
        b = self.smooth_beta
        return F.softplus(b * x) / b - F.softplus(b * (x - 1.0)) / b


def build_system(config: dict[str, Any], seed: int, device: torch.device):
    retina = int(config["img_h"]) * int(config["scan_win"])
    brain = int(config["neurons"])
    total = retina + brain
    dtype = torch.float32
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # retina -> brain (plastic), brain -> brain (plastic)
    ret_src = torch.arange(retina, dtype=torch.int64).repeat_interleave(int(config["retina_fanout"]))
    ret_dst = retina + torch.randint(
        brain, (ret_src.numel(),), generator=generator, dtype=torch.int64
    )
    brn_src = retina + torch.randint(brain, (brain * int(config["degree"]),), generator=generator, dtype=torch.int64)
    brn_dst = retina + torch.randint(brain, (brain * int(config["degree"]),), generator=generator, dtype=torch.int64)
    src = torch.cat([ret_src, brn_src])
    dst = torch.cat([ret_dst, brn_dst])
    scale = float(config["weight_scale"])
    weight = torch.randn(src.numel(), generator=generator) * scale / math.sqrt(
        max(1.0, float(config["degree"]))
    )

    topology = SparseTopology.from_edges(
        total, src.to(device), dst.to(device), weight.to(device=device, dtype=dtype)
    )
    alpha_t = torch.full((total,), float(config["alpha"]), device=device, dtype=dtype)
    gain_t = torch.full((total,), float(config["gain"]), device=device, dtype=dtype)
    if str(config.get("cell_model", "engine")) == "soft_clamp_control":
        dynamics = SoftClampDynamics(
            alpha_t, gain_t, smooth_beta=float(config.get("smooth_beta", 10.0))
        )
    else:
        dynamics = Dynamics(alpha=alpha_t, gain=gain_t)

    chemistry = RegionalDopamineField(
        tonic_baseline=torch.full((2,), 0.5, device=device, dtype=dtype),
        uptake_rate=torch.full((2,), 0.2, device=device, dtype=dtype),
        decay_rate=torch.zeros(2, device=device, dtype=dtype),
        # release_projection is [sources, regions]: one dopaminergic source
        # releasing into both the retinal and the core region.
        release_projection=torch.ones((1, 2), device=device, dtype=dtype),
        edge_src=torch.tensor([0], device=device),
        edge_dst=torch.tensor([1], device=device),
        diffusion_rate=torch.tensor([0.1], device=device, dtype=dtype),
        dt=0.1,
        burst_gain=0.8,
        dip_gain=1.0,
    )
    # D1-only coupling keeps a burst purely potentiating; tonic concentration
    # matches the regional baseline so the resting phasic signal is zero.
    receptors = DopamineReceptors(
        d1_density=torch.ones(total, device=device, dtype=dtype),
        d2_density=torch.zeros(total, device=device, dtype=dtype),
        tonic_concentration=0.5,
        d1_half_saturation=0.5,
        d2_half_saturation=0.5,
    )
    neuron_region = torch.cat(
        [
            torch.zeros(retina, dtype=torch.int64, device=device),
            torch.ones(brain, dtype=torch.int64, device=device),
        ]
    )

    runtime = DopamineMatrixRuntime(
        topology,
        dynamics,
        torch.zeros((1, total), device=device, dtype=dtype),
        chemistry=chemistry,
        receptors=receptors,
        neuron_region=neuron_region,
        config=DopaminePlasticityConfig(
            drive_scale=0.0,          # dopamine acts on weights only here
            log_gain_scale=0.0,
            learning_rate=float(config["learning_rate"]),
            track_eligibility=True,
            eligibility_decay=float(config["eligibility_decay"]),
            chemistry_interval=int(config["chemistry_interval"]),
            plasticity_interval=1,
            dopamine_delay_steps=int(config["dopamine_delay_steps"]),
            preserve_weight_sign=True,
            minimum_weight=-1.0,
            maximum_weight=1.0,
            validate_inputs=True,
        ),
    )
    return runtime, retina, brain, total


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


def reset_runtime(runtime: DopamineMatrixRuntime) -> None:
    runtime.state.zero_()
    runtime.pre_trace.zero_()
    runtime.post_trace.zero_()
    if runtime.eligibility.numel():
        runtime.eligibility.zero_()


def run_episode(
    runtime: DopamineMatrixRuntime,
    readout: nn.Module,
    image: torch.Tensor,
    *,
    retina: int,
    brain: int,
    steps: int,
    win: int,
    stride: int,
    pending_rpe: torch.Tensor,
    rpe_per_tick: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one image; RPE is injected on the first tick only (next-tick rule)."""

    logits = []
    states_cls = []
    drive = torch.zeros((1, retina + brain), device=image.device)
    first_rpe = pending_rpe
    for step in range(steps):
        start = step * stride
        drive.zero_()
        drive[0, :retina] = image[:, start : start + win].reshape(-1)
        rpe = first_rpe if step == 0 else rpe_per_tick
        state = runtime.step(drive, rpe=rpe)
        brain_state = state[:, retina:]
        states_cls.append(brain_state)
        logits.append(readout(brain_state))
    return torch.stack(logits, dim=1), torch.cat(states_cls, dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = dict(DEFAULTS)
    if args.config_json:
        config.update(json.loads(args.config_json))
    if args.smoke:
        config.update({"epochs": 1, "train_n": 128, "test_n": 64})

    requested = str(config.get("device", "auto"))
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested)
    seed = int(config["seed"])
    seed_everything(seed)

    data = load_mnist(DATA / "mnist")
    charset = Charset([str(d) for d in range(10)])
    train_images, train_texts = build_sequences(
        data["train_images"], data["train_labels"], int(config["train_n"]), seed=seed
    )
    test_images, test_texts = build_sequences(
        data["test_images"], data["test_labels"], int(config["test_n"]), seed=seed + 1
    )
    train_targets = [charset.encode(t) for t in train_texts]
    test_targets = [charset.encode(t) for t in test_texts]

    runtime, retina, brain, total = build_system(config, seed, device)
    readout = nn.Linear(brain, len(charset)).to(device)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=float(config["readout_lr"]))

    steps = (int(config["img_w"]) - int(config["scan_win"])) // int(config["scan_stride"]) + 1
    rpe_zero = torch.zeros((1, 1), device=device)
    print(
        f"[system] neurons={total} (retina={retina}, brain={brain}) "
        f"edges={runtime.topology.n_edges} steps={steps} "
        f"learning_rate={config['learning_rate']} device={device}",
        flush=True,
    )

    pending = rpe_zero.clone()
    history = []
    for epoch in range(int(config["epochs"])):
        total_loss = 0.0
        seen = 0
        for i in range(train_images.shape[0]):
            if not train_targets[i]:
                continue
            reset_runtime(runtime)
            image = train_images[i].float().div_(255.0).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = run_episode(
                runtime, readout, image,
                retina=retina, brain=brain, steps=steps,
                win=int(config["scan_win"]), stride=int(config["scan_stride"]),
                pending_rpe=pending, rpe_per_tick=rpe_zero,
            )
            target = torch.tensor([train_targets[i]], dtype=torch.long, device=device)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.ctc_loss(
                log_probs.transpose(0, 1),
                target.reshape(-1),
                torch.tensor([steps], device=device),
                torch.tensor([target.shape[1]], device=device),
                blank=0,
                zero_infinity=True,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            seen += 1

            with torch.no_grad():
                pred = ctc_greedy(logits, charset)
                gold = "".join(charset.chars[j] for j in train_targets[i])
                if config["reward_mode"] == "sign":
                    pending = torch.tensor(
                        [[1.0 if pred == gold else -1.0]], device=device
                    )
                else:
                    pending = rpe_zero.clone()
        history.append({"epoch": epoch + 1, "ctc_loss": total_loss / max(seen, 1)})
        print(f"  epoch {epoch+1}/{int(config['epochs'])} ctc_loss={history[-1]['ctc_loss']:.4f}", flush=True)

    metrics = evaluate(
        runtime, readout, test_images, test_targets, charset,
        retina=retina, brain=brain, steps=steps,
        win=int(config["scan_win"]), stride=int(config["scan_stride"]),
        rpe_zero=rpe_zero,
    )
    weight_norm = float(runtime.topology.weight.norm())
    diag = measure_dead_zone(
        runtime, test_images,
        retina=retina, brain=brain, steps=steps,
        win=int(config["scan_win"]), stride=int(config["scan_stride"]),
        alpha=float(config["alpha"]), gain=float(config["gain"]),
    )
    is_control = str(config.get("cell_model", "engine")) == "soft_clamp_control"
    payload = {
        "experiment": "brain_online_union_protocol",
        "cell_model": "soft_clamp_control" if is_control else "engine",
        "cell_model_label": (
            SoftClampDynamics.CONTROL_ARM if is_control
            else "engine (src/neurosim_gpu/dynamics.py, unmodified)"
        ),
        "config": config,
        "device": str(device),
        "neurons": total,
        "edges": runtime.topology.n_edges,
        "history": history,
        "metrics": metrics,
        "final_weight_norm": weight_norm,
        **diag,
    }
    if args.json:
        emit_result(payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


@torch.no_grad()
def measure_dead_zone(
    runtime, images, *, retina, brain, steps, win, stride, alpha, gain
) -> dict[str, float]:
    """Fraction of neuron-ticks where the engine clamp would kill the gradient."""

    device = runtime.state.device
    zero_rpe = torch.zeros((1, 1), device=device)
    dead = 0.0
    sat = 0.0
    n = 0
    for i in range(min(8, images.shape[0])):
        reset_runtime(runtime)
        img = images[i].float().div_(255.0).to(device)
        drive = torch.zeros((1, retina + brain), device=device)
        for t in range(steps):
            drive.zero_()
            drive[0, :retina] = img[:, t * stride : t * stride + win].reshape(-1)
            pre = alpha * runtime.state + (1.0 - alpha) * gain * torch.tanh(
                runtime._propagate(runtime.state) + drive
            )
            brain_pre = pre[:, retina:]
            dead += float((brain_pre <= 0).float().mean())
            sat += float((brain_pre >= 1).float().mean())
            n += 1
            runtime.step(drive, rpe=zero_rpe)
    return {"dead_fraction": dead / max(n, 1), "saturated_fraction": sat / max(n, 1)}


def ctc_greedy(logits: torch.Tensor, charset: Charset) -> str:
    best = logits[0].argmax(dim=-1).tolist()
    out = []
    prev = -1
    for idx in best:
        if idx != prev and idx != 0:
            out.append(charset.chars[idx])
        prev = idx
    return "".join(out)


@torch.no_grad()
def evaluate(
    runtime, readout, images, targets, charset, *,
    retina, brain, steps, win, stride, rpe_zero,
) -> dict[str, float]:
    words = 0
    errors = 0
    chars = 0
    n = 0
    for i in range(images.shape[0]):
        if not targets[i]:
            continue
        reset_runtime(runtime)
        image = images[i].float().div_(255.0).to(runtime.state.device)
        logits, _ = run_episode(
            runtime, readout, image,
            retina=retina, brain=brain, steps=steps, win=win, stride=stride,
            pending_rpe=rpe_zero, rpe_per_tick=rpe_zero,
        )
        pred = ctc_greedy(logits, charset)
        gold = "".join(charset.chars[j] for j in targets[i])
        n += 1
        words += int(pred == gold)
        errors += edit_distance_str(pred, gold)
        chars += max(len(gold), 1)
    return {
        "samples": n,
        "word_accuracy": words / max(n, 1),
        "char_accuracy": max(0.0, 1.0 - errors / max(chars, 1)),
        "cer": errors / max(chars, 1),
    }


if __name__ == "__main__":
    main()
