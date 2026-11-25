"""Export a vNext goal-conditioned rollout as a frontend clip.

    .venv-str\\Scripts\\python.exe demo\\neurobody\\rollout_vnext.py \
        --checkpoint demo/runs/vnext/goals_continuous_vnext_s0/model.pt \
        --config demo/config/local_avatar.json --goal-mode continuous --frames 600 \\
        --out demo/assets/clip_embodied.json

The vNext model was trained without vision, so the exported ``eyes`` are neutral
placeholders and the metadata says so explicitly.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEMO = Path(__file__).resolve().parents[1]
ROOT = DEMO.parent
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neurobody.amass import action_joints, quat_wxyz_to_three, retarget_sequence  # noqa: E402
from neurobody.body import load_config  # noqa: E402
from neurobody.device import resolve_device  # noqa: E402
from neurobody.vnext import ServoTask, load_checkpoint  # noqa: E402
from neurobody.vrm_rig import VrmRig  # noqa: E402


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True, help="Path to a trained NeuroBody checkpoint",
    )
    parser.add_argument("--config", required=True, help="Local JSON with avatar.vrm and optional avatar.height_m")
    parser.add_argument("--goal-mode", choices=["continuous", "cycle", "random"], default="continuous")
    parser.add_argument("--goal-dwell", type=int, default=150)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--action-lowpass", type=float, default=0.9)
    parser.add_argument("--max-accel", type=float, default=0.002)
    parser.add_argument("--arm-pose", type=float, default=1.0)
    parser.add_argument(
        "--out", default=str(DEMO / "assets" / "clip_embodied.json")
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device, require_cuda=not args.allow_cpu)
    match = re.search(r"_s(\d+)$", Path(args.checkpoint).parent.name)
    seed = int(match.group(1)) if match else 0
    brain, _ = load_checkpoint(Path(args.checkpoint), device, seed=seed)
    brain.eval()

    config = load_config(args.config)
    rig = VrmRig.load(
        config["avatar"]["vrm"], height_m=float(config["avatar"].get("height_m", 1.62))
    )
    joints = [joint.smpl_joint for joint in action_joints()]
    task = ServoTask(
        "goals",
        batch=1,
        device=device,
        episode_ticks=args.frames,
        goal_mode=args.goal_mode,
        goal_dwell=args.goal_dwell,
        action_lowpass=args.action_lowpass,
        max_accel=args.max_accel,
    )
    q, qd, vest, target = task.reset()
    state = None
    actions, activity, errors = [], [], []
    with torch.no_grad():
        for _ in range(args.frames):
            drive = brain.encode(q, qd, vest, None, target)
            state = (
                brain.initial_neural_state(drive)
                if state is None
                else brain.advance(state, drive)
            )
            soma = brain.soma_activity(state)
            velocity = brain.motor_velocity(soma)
            q, qd, vest, target = task.step(velocity)
            # Record the physical body state, not the noisy sensor observation
            # returned by ``step`` (that noise is only the model's input).
            actions.append(task.q[0].cpu().numpy().copy())
            activity.append(soma[0].cpu().numpy().astype(np.float32))
            errors.append(
                np.abs(
                    task.q[0, :39].cpu().numpy()
                    - target[0, :39].cpu().numpy()
                )
            )

    aa = np.zeros((len(actions), 22, 3), dtype=np.float64)
    trans = np.zeros((len(actions), 3), dtype=np.float64)
    for tick, action in enumerate(actions):
        for slot, joint in enumerate(joints):
            aa[tick, joint] = action[slot * 3 : slot * 3 + 3]
        trans[tick, 0] = 0.0
        trans[tick, 2] = 0.0

    # Display-only A-pose offset: the VRM rest pose is a T-pose and the SMPL
    # shoulder x-rotation is mostly a twist, so a pure model rollout looks like
    # a T-pose.  Lower the upper arms around the front axis for the viewer.
    if args.arm_pose:
        for joint in action_joints():
            if joint.vrm_bone == "leftUpperArm":
                aa[:, joint.smpl_joint, 2] -= args.arm_pose
            elif joint.vrm_bone == "rightUpperArm":
                aa[:, joint.smpl_joint, 2] += args.arm_pose
    bones, root = retarget_sequence(rig, aa, trans, root_scale=1.0)

    activity_array = np.stack(activity)
    activity_bytes = np.clip(activity_array * 255.0, 0, 255).astype(np.uint8)
    error_array = np.zeros_like(activity_bytes)

    velocity = np.diff(np.stack(actions)[:, :39], axis=0) * 30.0
    accel = np.diff(velocity, axis=0)
    sign_flip = float(
        (np.sign(velocity[1:]) * np.sign(velocity[:-1]) < 0).mean()
    )
    freqs = np.fft.rfftfreq(velocity.shape[0], d=1.0 / 30.0)
    power = np.abs(np.fft.rfft(velocity - velocity.mean(axis=0), axis=0)) ** 2
    total_power = power.sum(axis=0) + 1e-12
    hf_ratio = float(np.mean(power[freqs > 5.0].sum(axis=0) / total_power))

    placeholder = Image.new("RGB", (64, 64), (10, 10, 14))
    eye_url = _png_data_url(placeholder)
    eyes = [{"left": eye_url, "right": eye_url} for _ in range(len(actions))]

    payload = {
        "version": 1,
        "name": f"vnext-{brain.dynamics}-{args.goal_mode}",
        "mode": "embodied",
        "fps": 30,
        "frames": len(actions),
        "seedClip": "vnext",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpointBytes": Path(args.checkpoint).stat().st_size,
        "checkpointMtime": int(Path(args.checkpoint).stat().st_mtime),
        "symbol": 0,
        "symbolId": "vnext_goal",
        "symbolLabel": "vNext 目标条件",
        "source": f"vNext {brain.dynamics} goal-conditioned ({args.goal_mode})",
        "runtime": (
            "vnext-goals-dendritic"
            if brain.dynamics == "dendritic"
            else "vnext-goals"
        ),
        "neuronCore": brain.dynamics,
        "compartmentParameters": (
            dict(brain.compartment_parameters)
            if brain.dynamics == "dendritic"
            else None
        ),
        "vision": {
            "pipeline": "vnext_no_vision",
            "source": "none (trained without vision)",
            "eyeSize": 64,
            "eyeSeparation": 0.03,
            "panelDistance": 1.0,
            "fovDeg": 45,
            "gazeFrom": "none",
            "directGlyph": False,
            "frames": len(eyes),
            "note": "placeholder eyes; the vNext model used images=None",
        },
        "eyes": eyes,
        "gaze": [[0.0, 0.0] for _ in actions],
        "bones": {
            name: np.round(np.stack([quat_wxyz_to_three(q) for q in bones[name]]), 5).tolist()
            for name in rig.ordered()
            if name in bones
        },
        "root": np.round(root, 4).tolist(),
        "action": np.round(np.stack(actions), 5).tolist(),
        "activity": [base64.b64encode(row.tobytes()).decode("ascii") for row in activity_bytes],
        "error": [base64.b64encode(row.tobytes()).decode("ascii") for row in error_array],
        "metrics": {
            "activityMean": float(activity_array.mean()),
            "frames": int(aa.shape[0]),
            "goalErrorMean": float(np.stack(errors).mean()),
            "actionStd": float(np.stack(actions).std(axis=0).mean()),
            "motionMean": float(np.abs(np.stack(actions)).mean()),
            "jitter": float(np.abs(accel).mean()),
            "signFlipRate": sign_flip,
            "hf5Ratio": hf_ratio,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), {payload['frames']} frames")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
