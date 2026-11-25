"""Run one NeuroBody vNext condition and write a reproducible record.

Examples::

    .venv-str\\Scripts\\python.exe demo\\neurobody\\vnext_run.py --task posture --policy vnext --ticks 1280
    .venv-str\\Scripts\\python.exe demo\\neurobody\\vnext_run.py --task velocity --policy pd

The script writes ``train_log.jsonl``, ``eval.json``, ``config.json`` and (for
learned policies) ``model.pt`` under ``demo/runs/vnext/<task>_<policy>_s<seed>``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

DEMO = Path(__file__).resolve().parents[1]
ROOT = DEMO.parent
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neurobody.device import resolve_device  # noqa: E402
from neurobody.embodied import EmbodiedConfig  # noqa: E402
from neurobody.vnext import (  # noqa: E402
    BrainPolicy,
    PDPolicy,
    ReflexPolicy,
    ServoTask,
    TrainConfig,
    VNextBrain,
    evaluate,
    save_checkpoint,
    train_brain,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["posture", "velocity", "switch", "goals"], default="posture")
    parser.add_argument(
        "--policy",
        choices=["pd", "reflex", "reservoir", "plain", "vnext", "vnext_dale", "dendritic"],
        default="dendritic",
    )
    parser.add_argument("--ticks", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--m1-to-motor", type=int, default=8)
    parser.add_argument("--motor-recurrence", type=float, default=0.1)
    parser.add_argument("--reflex-gain", type=float, default=1.0)
    parser.add_argument("--velocity-scale", type=float, default=0.3)
    parser.add_argument("--switch-b-scale", type=float, default=1.0)
    parser.add_argument("--switch-cycles", type=int, default=1)
    parser.add_argument("--goal-mode", choices=["cycle", "random", "continuous"], default="cycle")
    parser.add_argument("--goal-dwell", type=int, default=150)
    parser.add_argument("--action-lowpass", type=float, default=0.0)
    parser.add_argument("--max-accel", type=float, default=0.0)
    parser.add_argument("--smooth-weight", type=float, default=0.0)
    parser.add_argument("--plan-horizon", type=int, default=0)
    parser.add_argument("--plan-weight", type=float, default=0.5)
    parser.add_argument("--cf-weight", type=float, default=0.1)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument(
        "--loss-variant",
        choices=["baseline", "weighted_b", "balanced", "b_only"],
        default="baseline",
    )
    parser.add_argument("--eval-ticks", type=int, default=0)
    parser.add_argument("--eval-cycles", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    # The historical runner seeded task NumPy generators but left parameter
    # initialization and sensor noise on PyTorch's process-global RNG.
    # Seed both CPU and CUDA so repeated conditions with the same seed are
    # comparable and reproducible.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    brain_config = None
    if args.policy not in ("pd", "reflex"):
        brain_config = EmbodiedConfig()
        # All neural policies use the public 6,020-neuron layout until a
        # regenerated layout for the optional sensory/memory pools is served.
        brain_config.touch_neurons = 0
        brain_config.audio_neurons = 0
        brain_config.nocicept_neurons = 0
        brain_config.memory_neurons = 0

    device = resolve_device(args.device, require_cuda=not args.allow_cpu)
    episode_ticks = {
        "posture": 320,
        "velocity": 240,
        "switch": 450,
        "goals": 3 * args.goal_dwell,
    }[args.task]
    ticks = args.ticks or {
        "posture": 1280,
        "velocity": 2400,
        "switch": 6000 if args.curriculum else 2250,
        "goals": 6000,
    }[args.task]
    train_b_scale = 0.2 if (args.curriculum and args.task == "switch") else args.switch_b_scale
    b_schedule = (
        ((0.0, 0.2), (0.34, 0.5), (0.67, 1.0))
        if args.curriculum and args.task == "switch"
        else ()
    )
    loss_weights = {
        "baseline": (1.0, 1.0, 1.0),
        "weighted_b": (1.0, 3.0, 1.0),
        "balanced": (0.5, 1.0, 0.5),
        "b_only": (0.0, 1.0, 0.0),
    }[args.loss_variant]
    out = Path(args.out) if args.out else (
        DEMO / "runs" / "vnext" / f"{args.task}_{args.policy}_s{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "policy": args.policy,
                "dynamics": "dendritic" if args.policy == "dendritic" else ("plain" if args.policy == "plain" else "neurosim"),
                "dale_constrained": args.policy in ("dendritic", "vnext_dale"),
                "neuron_count": brain_config.n_neurons if brain_config is not None else None,
                "error_to_motor": False,
                "compartment_parameters": (
                    {
                        "dt": 1.0,
                        "dendrite_tau": 4.0,
                        "adaptation_tau": 25.0,
                        "coupling": 0.5,
                        "adaptation_strength": 0.2,
                        "status": "illustrative; not task-fitted",
                    }
                    if args.policy == "dendritic"
                    else None
                ),
                "ticks": ticks,
                "batch": args.batch,
                "lr": args.lr,
                "seed": args.seed,
                "torch_rng_seed": args.seed,
                "torch_rng_controlled": True,
                "episodes": args.episodes,
                "m1_to_motor": args.m1_to_motor,
                "motor_recurrence": args.motor_recurrence,
                "reflex_gain": args.reflex_gain,
                "velocity_scale": args.velocity_scale,
                "switch_b_scale": args.switch_b_scale,
                "switch_cycles": args.switch_cycles,
                "curriculum": args.curriculum,
                "loss_variant": args.loss_variant,
                "goal_mode": args.goal_mode,
                "goal_dwell": args.goal_dwell,
                "action_lowpass": args.action_lowpass,
                "max_accel": args.max_accel,
                "smooth_weight": args.smooth_weight,
                "plan_horizon": args.plan_horizon,
                "plan_weight": args.plan_weight,
                "cf_weight": args.cf_weight,
                "eval_ticks": args.eval_ticks,
                "eval_cycles": args.eval_cycles,
                "episode_ticks": episode_ticks,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_task = ServoTask(
        args.task,
        batch=args.batch,
        device=device,
        episode_ticks=episode_ticks,
        velocity_scale=args.velocity_scale,
        switch_b_scale=train_b_scale,
        switch_cycles=args.switch_cycles,
        switch_loss_weights=loss_weights,
        goal_mode=args.goal_mode,
        goal_dwell=args.goal_dwell,
        action_lowpass=args.action_lowpass,
        max_accel=args.max_accel,
        smoothness_weight=args.smooth_weight,
    )
    eval_task = ServoTask(
        args.task,
        batch=1,
        device=device,
        episode_ticks=episode_ticks,
        velocity_scale=args.velocity_scale,
        switch_b_scale=args.switch_b_scale,
        switch_cycles=args.switch_cycles,
        switch_loss_weights=loss_weights,
        goal_mode=args.goal_mode,
        goal_dwell=args.goal_dwell,
        action_lowpass=args.action_lowpass,
        max_accel=args.max_accel,
        smoothness_weight=args.smooth_weight,
    )
    started = time.perf_counter()
    train_summary: dict = {}

    if args.policy in ("pd", "reflex"):
        policy = PDPolicy() if args.policy == "pd" else ReflexPolicy(seed=args.seed)
        brain = None
    else:
        assert brain_config is not None
        brain = VNextBrain(
            brain_config,
            n_symbols=1,
            seed=args.seed,
            dynamics=(
                "plain" if args.policy == "plain"
                else "dendritic" if args.policy == "dendritic"
                else "neurosim"
            ),
            dale_constrained=args.policy in ("dendritic", "vnext_dale"),
            freeze_recurrent=args.policy == "reservoir",
            m1_to_motor=args.m1_to_motor,
            error_to_motor=False,
            motor_recurrence=args.motor_recurrence,
            reflex_gain=args.reflex_gain,
            plan_horizon=args.plan_horizon,
        ).to(device)
        policy = BrainPolicy(brain)
        train_summary = train_brain(
            brain,
            train_task,
            TrainConfig(
                ticks=ticks,
                batch=args.batch,
                lr=args.lr,
                seed=args.seed,
                episode_ticks=episode_ticks,
                self_weight=0.0,
                log_every=max(ticks // 8, 1),
                b_scale_schedule=b_schedule,
                plan_horizon=args.plan_horizon,
                plan_weight=args.plan_weight,
                counterfactual_weight=args.cf_weight,
            ),
            device=device,
            log_path=out / "train_log.jsonl",
        )
        save_checkpoint(out / "model.pt", brain, {"task": args.task, "policy": args.policy})

    metrics = evaluate(
        policy,
        eval_task,
        episodes=args.episodes,
        seed=args.seed + 1000,
        device=device,
    )
    long_metrics = None
    if args.eval_ticks:
        long_task = ServoTask(
            args.task,
            batch=1,
            device=device,
            episode_ticks=args.eval_ticks,
            velocity_scale=args.velocity_scale,
            switch_b_scale=args.switch_b_scale,
            switch_cycles=args.eval_cycles,
            switch_loss_weights=loss_weights,
            goal_mode=args.goal_mode,
            goal_dwell=args.goal_dwell,
            action_lowpass=args.action_lowpass,
            max_accel=args.max_accel,
            smoothness_weight=args.smooth_weight,
        )
        long_metrics = evaluate(
            policy,
            long_task,
            episodes=args.episodes,
            seed=args.seed + 2000,
            device=device,
        )
        (out / "eval_long.json").write_text(
            json.dumps(
                {
                    "task": args.task,
                    "policy": args.policy,
                    "seed": args.seed,
                    "ticks": args.eval_ticks,
                    "cycles": args.eval_cycles,
                    "metrics": long_metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    n_trainable = int(sum(p.numel() for p in brain.trainable_parameters())) if brain is not None else 0
    if brain is not None and getattr(brain, "freeze_brain", False):
        n_trainable -= int(brain.brain_edge_mask.sum())
    summary = {
        "task": args.task,
        "policy": args.policy,
        "seed": args.seed,
        "ticks": ticks,
        "m1_to_motor": args.m1_to_motor,
        "motor_recurrence": args.motor_recurrence,
        "reflex_gain": args.reflex_gain,
        "velocity_scale": args.velocity_scale,
        "switch_b_scale": args.switch_b_scale,
        "switch_cycles": args.switch_cycles,
        "curriculum": args.curriculum,
        "loss_variant": args.loss_variant,
        "goal_mode": args.goal_mode,
        "goal_dwell": args.goal_dwell,
        "plan_horizon": args.plan_horizon,
        "plan_weight": args.plan_weight,
        "cf_weight": args.cf_weight,
        "eval_ticks": args.eval_ticks,
        "eval_cycles": args.eval_cycles,
        "n_edges": int(brain.edge_weight.numel()) if brain is not None else None,
        "n_trainable": n_trainable,
        "train": {k: v for k, v in train_summary.items() if k != "records"},
        "metrics": metrics,
        "eval_long": long_metrics,
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    (out / "eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
