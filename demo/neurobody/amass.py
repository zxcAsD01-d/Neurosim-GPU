"""AMASS (SMPL-H 52) loading, proprioception features and VRM retargeting.

The dataset on this machine is already converted to SMPL-H with 52 joints at a
uniform frame rate, one ``.npz`` per clip:

    poses          (T, 156)  axis-angle, 52 joints
    global_orient  (T, 3)
    body_pose      (T, 63)   joints 1..21
    trans          (T, 3)

Only the 22 body joints are used.  The neural pipeline consumes proprioceptive
features and predicts the next tick's 39 action components (13 joints x 3).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .body import LIMBS, SMPL_JOINT_NAMES
from .nervous import FEATURE_DIM, ActionJoint, action_joints
from .rotations import quat_from_axis_angle, quat_to_matrix
from .vision import parse_text
from .vrm_rig import SMPL_BODY_JOINTS, VrmRig, retarget_local

GAIT_PERIOD_FRAMES = 40.0


@dataclass
class Clip:
    name: str
    path: str
    aa: np.ndarray  # (T, 22, 3) axis-angle, joint 0 = pelvis
    trans: np.ndarray  # (T, 3)
    fps: float

    def __len__(self) -> int:
        return int(self.aa.shape[0])


def _resample(array: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if abs(source_fps - target_fps) < 1e-6:
        return array
    length = array.shape[0]
    new_length = max(int(round(length * target_fps / source_fps)), 2)
    old_t = np.arange(length) / source_fps
    new_t = np.arange(new_length) / target_fps
    flat = array.reshape(length, -1)
    out = np.stack([np.interp(new_t, old_t, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape((new_length,) + array.shape[1:])


def load_clip(path: str, *, fps: float = 30.0) -> Clip:
    with np.load(path) as data:
        global_orient = np.asarray(data["global_orient"], dtype=np.float64)
        body_pose = np.asarray(data["body_pose"], dtype=np.float64).reshape(-1, 21, 3)
        trans = np.asarray(data["trans"], dtype=np.float64)
        source_fps = float(data["mocap_framerate"]) if "mocap_framerate" in data else fps
    aa = np.concatenate([global_orient[:, None, :], body_pose], axis=1)
    aa = _resample(aa, source_fps, fps)
    trans = _resample(trans, source_fps, fps)
    return Clip(name=Path(path).stem, path=str(path), aa=aa, trans=trans, fps=fps)


class AmassDataset:
    """Lazy clip index with a deterministic train/val split."""

    def __init__(
        self,
        root: str | Path,
        *,
        fps: float = 30.0,
        max_clips: int | None = None,
        seed: int = 7,
        val_fraction: float = 0.1,
    ) -> None:
        self.root = Path(root)
        files = sorted(glob.glob(os.path.join(str(self.root), "**", "*.npz"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no AMASS .npz files under {self.root}")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(files))
        files = [files[i] for i in order]
        if max_clips:
            files = files[: int(max_clips)]
        n_val = max(1, int(round(len(files) * val_fraction))) if len(files) > 2 else 0
        self.val_files = sorted(files[:n_val])
        self.train_files = sorted(files[n_val:])
        self.fps = fps
        self._cache: dict[str, Clip] = {}

    def __len__(self) -> int:
        return len(self.train_files) + len(self.val_files)

    def clip(self, path: str) -> Clip:
        if path not in self._cache:
            self._cache[path] = load_clip(path, fps=self.fps)
        return self._cache[path]

    def split(self, which: str = "train") -> list[str]:
        return self.train_files if which == "train" else self.val_files


def proprioception(aa: np.ndarray, trans: np.ndarray, fps: float) -> np.ndarray:
    """(T, 22, 3) + (T, 3) -> (T, FEATURE_DIM) float32."""

    frames = aa.shape[0]
    flat = aa.reshape(frames, -1)
    velocity = np.zeros_like(flat)
    if frames > 1:
        velocity[1:] = (flat[1:] - flat[:-1]) * fps
        velocity[0] = velocity[1]
    root_velocity = np.zeros_like(trans)
    if frames > 1:
        root_velocity[1:] = (trans[1:] - trans[:-1]) * fps
        root_velocity[0] = root_velocity[1]
    features = np.concatenate(
        [
            flat,
            np.clip(velocity, -8.0, 8.0),
            np.clip(root_velocity, -4.0, 4.0),
            trans[:, 1:2] - trans[:, 1:2].mean(),
        ],
        axis=1,
    ).astype(np.float32)
    if features.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"feature width {features.shape[1]} != {FEATURE_DIM}")
    return features


def action_vector(aa: np.ndarray, joints: list[ActionJoint]) -> np.ndarray:
    return np.concatenate([aa[:, joint.smpl_joint, :] for joint in joints], axis=1)


def feature_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return mean.astype(np.float32), np.maximum(std, 1e-4).astype(np.float32)


def build_windows(
    dataset: AmassDataset,
    paths: Iterable[str],
    *,
    ticks: int,
    rng: np.random.Generator,
    batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``batch`` teacher-forced windows -> (features, targets)."""

    paths = list(paths)
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    joints = action_joints()
    for _ in range(batch):
        clip = dataset.clip(paths[int(rng.integers(len(paths)))])
        if len(clip) < ticks + 2:
            continue
        start = int(rng.integers(0, len(clip) - ticks - 1))
        aa = clip.aa[start : start + ticks + 1]
        trans = clip.trans[start : start + ticks + 1]
        features = proprioception(aa, trans, clip.fps)
        actions = action_vector(aa, joints)
        feature_rows.append(features[:ticks])
        target_rows.append(actions[1 : ticks + 1])
    if not feature_rows:
        raise RuntimeError("no clip long enough for the requested tick window")
    return (
        np.stack(feature_rows).astype(np.float32),
        np.stack(target_rows).astype(np.float32),
    )


def retarget_sequence(
    rig: VrmRig,
    aa: np.ndarray,
    trans: np.ndarray,
    *,
    root_scale: float = 0.85,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """SMPL (T,22,3) -> absolute VRM local quaternions + root offsets.

    Returns ``(bones, root)`` where ``bones[bone]`` is (T, 4) wxyz and ``root``
    is (T, 3) world-space translation applied to the avatar root.
    """

    from .vrm_rig import SMPL_BODY_JOINTS

    frames = aa.shape[0]
    quats = quat_from_axis_angle(aa.reshape(-1, 3)).reshape(frames, aa.shape[1], 4)
    matrices = {
        SMPL_BODY_JOINTS[index]: quat_to_matrix(quats[:, index])
        for index in range(len(SMPL_BODY_JOINTS))
    }
    bones = retarget_local(rig, matrices)
    yaw = rig.facing_yaw
    c, s = np.cos(yaw), np.sin(yaw)
    delta = trans - trans[:1]
    root = np.stack(
        [
            c * delta[:, 0] + s * delta[:, 2],
            np.zeros(frames),
            -s * delta[:, 0] + c * delta[:, 2],
        ],
        axis=1,
    ) * root_scale
    return bones, root


def quat_wxyz_to_three(q: np.ndarray) -> list[float]:
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


@dataclass
class SymbolSet:
    """Shape/colour concepts from ``demo/config/actions.json``."""

    symbols: list["Symbol"]
    clips: list[str]
    labels: list[str]
    root: Path

    @property
    def ids(self) -> list[str]:
        return [symbol.id for symbol in self.symbols]

    def resolve(self, key: str) -> int:
        key = key.strip()
        for index, (symbol, clip, label) in enumerate(zip(self.symbols, self.clips, self.labels)):
            stem = Path(clip).stem
            if key in (symbol.id, label, stem, f"{symbol.shape}_{symbol.color}"):
                return index
        parsed = parse_text(key)
        matches = [i for i, symbol in enumerate(self.symbols) if symbol.shape == parsed.shape]
        if matches:
            return matches[0]
        raise KeyError(f"no symbol matches {key!r}; known: {self.ids}")

    def clip_path(self, index: int) -> str:
        return str(self.root / self.clips[index])


def load_symbols(path: str | Path, dataset_root: str | Path) -> SymbolSet:
    from .vision import COLORS, Symbol

    payload = json.loads(Path(path).read_text("utf-8"))
    symbols: list[Symbol] = []
    clips: list[str] = []
    labels: list[str] = []
    by_rgb = {value: key for key, value in COLORS.items()}
    for entry in payload["symbols"]:
        hex_color = entry["color"].lstrip("#")
        rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        color = by_rgb.get(rgb, "white")
        symbols.append(Symbol(shape=entry["shape"], color=color, rgb=rgb))
        clips.append(entry["clip"])
        labels.append(entry.get("label", entry["action"]))
    return SymbolSet(symbols=symbols, clips=clips, labels=labels, root=Path(dataset_root))


def build_symbol_windows(
    dataset: AmassDataset,
    symbols: SymbolSet,
    *,
    ticks: int,
    rng: np.random.Generator,
    batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (glyph, clip) pairs and return windows plus symbol ids."""

    joints = action_joints()
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    symbol_ids: list[int] = []
    attempts = 0
    while len(feature_rows) < batch and attempts < batch * 8:
        attempts += 1
        index = int(rng.integers(len(symbols.symbols)))
        clip = dataset.clip(symbols.clip_path(index))
        if len(clip) < ticks + 2:
            continue
        start = int(rng.integers(0, len(clip) - ticks - 1))
        aa = clip.aa[start : start + ticks + 1]
        trans = clip.trans[start : start + ticks + 1]
        features = proprioception(aa, trans, clip.fps)
        actions = action_vector(aa, joints)
        feature_rows.append(features[:ticks])
        target_rows.append(actions[1 : ticks + 1])
        symbol_ids.append(index)
    if not feature_rows:
        raise RuntimeError("no symbol clip long enough for the requested window")
    return (
        np.stack(feature_rows).astype(np.float32),
        np.stack(target_rows).astype(np.float32),
        np.asarray(symbol_ids, dtype=np.int64),
    )
