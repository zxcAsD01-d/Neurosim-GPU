"""Predefined body plan: brain location, regions and the six limb pools.

All positions are in the avatar's rest-pose world space, so they can be sent to
the viewer unchanged.  The brain lives inside the skull around the VRM ``head``
bone; each limb pool is spread along that limb's bone segments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .rotations import quat_to_matrix
from .vrm_rig import VRM_TO_SMPL, VrmRig

SMPL_JOINT_NAMES = [
    "pelvis", "leftHip", "rightHip", "spine1", "leftKnee", "rightKnee",
    "spine2", "leftAnkle", "rightAnkle", "spine3", "leftFoot", "rightFoot",
    "neck", "leftCollar", "rightCollar", "head", "leftShoulder", "rightShoulder",
    "leftElbow", "rightElbow", "leftWrist", "rightWrist",
]


@dataclass(frozen=True)
class LimbSpec:
    id: str
    label: str
    color: str
    joints: tuple[str, ...]  # VRM bone names whose local rotation is predicted
    segments: tuple[tuple[str, str, int], ...]  # (from bone, to bone, n neurons)

    @property
    def action_dim(self) -> int:
        return 3 * len(self.joints)


LIMBS: tuple[LimbSpec, ...] = (
    LimbSpec(
        "left_arm", "左臂", "#4fc3f7",
        ("leftUpperArm", "leftLowerArm"),
        (("leftShoulder", "leftUpperArm", 4), ("leftUpperArm", "leftLowerArm", 6),
         ("leftLowerArm", "leftHand", 6)),
    ),
    LimbSpec(
        "right_arm", "右臂", "#ff8a65",
        ("rightUpperArm", "rightLowerArm"),
        (("rightShoulder", "rightUpperArm", 4), ("rightUpperArm", "rightLowerArm", 6),
         ("rightLowerArm", "rightHand", 6)),
    ),
    LimbSpec(
        "left_leg", "左腿", "#81c784",
        ("leftUpperLeg", "leftLowerLeg"),
        (("leftUpperLeg", "leftLowerLeg", 6), ("leftLowerLeg", "leftFoot", 6),
         ("leftFoot", "leftToes", 4)),
    ),
    LimbSpec(
        "right_leg", "右腿", "#ffd54f",
        ("rightUpperLeg", "rightLowerLeg"),
        (("rightUpperLeg", "rightLowerLeg", 6), ("rightLowerLeg", "rightFoot", 6),
         ("rightFoot", "rightToes", 4)),
    ),
    LimbSpec(
        "neck_head", "颈部", "#ce93d8",
        ("neck", "head"),
        (("chest", "neck", 5), ("neck", "head", 7), ("head", "head", 4)),
    ),
    LimbSpec(
        "torso", "躯干", "#90a4ae",
        ("hips", "spine", "chest"),
        (("hips", "spine", 5), ("spine", "chest", 6), ("chest", "neck", 5)),
    ),
)


@dataclass(frozen=True)
class RegionSpec:
    id: str
    label: str
    count: int
    centroid: tuple[float, float, float]  # direction in the skull ellipsoid
    spread: float
    bilateral: bool
    color: str


# 5000 neurons: 800 + 1440 + 1120 + 1120 + 520.
BRAIN_REGIONS: tuple[RegionSpec, ...] = (
    RegionSpec("pfc", "前额叶", 800, (0.05, 0.35, -0.85), 0.45, False, "#7e57c2"),
    RegionSpec("m1", "运动皮层", 1440, (0.62, 0.34, -0.28), 0.42, True, "#ef5350"),
    RegionSpec("s1", "躯体感觉", 1120, (0.62, 0.38, 0.42), 0.42, True, "#26a69a"),
    RegionSpec("cereb", "小脑", 1120, (0.0, -0.72, 0.55), 0.35, False, "#42a5f5"),
    RegionSpec("assoc", "联合区", 520, (0.0, -0.05, 0.05), 0.85, False, "#ffa726"),
)

BRAIN_ELLIPSOID = (0.062, 0.074, 0.078)  # radii around the skull centre
SKULL_OFFSET = (0.0, 0.082, -0.004)  # from the head bone origin

# Visual neurons sit in two small eye clusters, bound to the head bone.
VISUAL_PER_EYE = 256
EYE_OFFSETS = ((-0.030, 0.058, -0.058), (0.030, 0.058, -0.058))
EYE_RADIUS = 0.011
VISUAL_COLOR = "#e0f7fa"


@dataclass
class BodyLayout:
    positions: np.ndarray  # (N, 3) float32
    region: np.ndarray  # (N,) int32, region index or -1
    limb: np.ndarray  # (N,) int32, limb index or -1
    region_names: list[str]
    limb_names: list[str]
    brain_center: np.ndarray
    head_origin: np.ndarray
    rig: VrmRig
    bind_bones: list[str]
    bone_index: np.ndarray  # (N,) index into bind_bones
    local_positions: np.ndarray  # (N, 3) position in the bound bone's local space
    visual: np.ndarray  # (N,) bool, eye-cluster neurons
    private_features: np.ndarray  # (N, 6) body position + bone direction (limbs only)
    private_joint: np.ndarray  # (N,) SMPL joint index for the private channel, -1 elsewhere
    limb_action_dims: list[int] = field(default_factory=list)

    @property
    def n_neurons(self) -> int:
        return int(self.positions.shape[0])

    def to_json(self, *, max_tracts: int = 24) -> dict[str, Any]:
        positions = np.round(self.positions, 5)
        regions = []
        for index, spec in enumerate(BRAIN_REGIONS):
            count = int((self.region == index).sum())
            regions.append(
                {"id": spec.id, "label": spec.label, "color": spec.color, "count": count,
                 "centroid": _region_centroid(self, index)}
            )
        limbs = []
        for index, spec in enumerate(LIMBS):
            mask = self.limb == index
            limbs.append(
                {"id": spec.id, "label": spec.label, "color": spec.color,
                 "count": int(mask.sum()), "joints": list(spec.joints),
                 "centroid": np.round(self.positions[mask].mean(axis=0), 5).tolist()}
            )
        return {
            "version": 1,
            "nNeurons": self.n_neurons,
            "brainCenter": np.round(self.brain_center, 5).tolist(),
            "headOrigin": np.round(self.head_origin, 5).tolist(),
            "regions": regions,
            "limbs": limbs,
            "positions": positions.reshape(-1).tolist(),
            "regionIndex": self.region.tolist(),
            "limbIndex": self.limb.tolist(),
            "bindBones": self.bind_bones,
            "boneIndex": self.bone_index.tolist(),
            "localPositions": np.round(self.local_positions, 6).reshape(-1).tolist(),
            "visualIndex": np.where(self.visual)[0].tolist(),
            "visualColor": VISUAL_COLOR,
            "tracts": _tracts(self, max_tracts),
        }


def _region_centroid(layout: BodyLayout, index: int) -> list[float]:
    mask = layout.region == index
    if not mask.any():
        return layout.brain_center.tolist()
    return np.round(layout.positions[mask].mean(axis=0), 5).tolist()


def _tracts(layout: BodyLayout, max_tracts: int) -> list[dict[str, Any]]:
    tracts: list[dict[str, Any]] = []
    ids = [spec.id for spec in BRAIN_REGIONS]
    m1 = ids.index("m1")
    sources = np.where(layout.region == m1)[0]
    per_limb = max(1, max_tracts // len(LIMBS))
    for index, spec in enumerate(LIMBS):
        targets = np.where(layout.limb == index)[0]
        if targets.size == 0 or sources.size == 0:
            continue
        for k in range(per_limb):
            source = sources[(k * 37 + index * 11) % sources.size]
            target = targets[(k * 3 + index) % targets.size]
            tracts.append(
                {"limb": spec.id, "from": int(source), "to": int(target)}
            )
    return tracts


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text("utf-8"))


def _segment(rig: VrmRig, start: str, end: str, count: int) -> np.ndarray:
    a = rig.world_pos(start)
    b = rig.world_pos(end)
    if np.linalg.norm(b - a) < 1e-5:
        b = a + np.array([0.0, 0.02, 0.0])
    fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
    points = a[None, :] + fractions[:, None] * (b - a)[None, :]
    # Push points slightly outward from the bone axis so they do not sit inside
    # the mesh centre line.
    axis = (b - a) / np.linalg.norm(b - a)
    side = np.cross(axis, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(side) < 1e-5:
        side = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    side /= np.linalg.norm(side)
    offset = side * 0.012 + np.array([0.0, 0.008, 0.0])
    return points + offset


def _brain_points(rig: VrmRig, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    head = rig.world_pos("head")
    center = head + np.array(SKULL_OFFSET)
    radii = np.array(BRAIN_ELLIPSOID)
    chunks = []
    for spec in BRAIN_REGIONS:
        direction = np.array(spec.centroid, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        for half in ((0, 1) if spec.bilateral else (0,)):
            noise = rng.normal(0.0, spec.spread, size=(spec.count // (2 if spec.bilateral else 1), 3))
            dirs = direction[None, :] + noise
            dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
            if spec.bilateral and half == 1:
                dirs[:, 0] *= -1.0
            radius = np.cbrt(rng.random(dirs.shape[0]))[:, None]
            chunks.append(center + dirs * radius * radii)
    points = np.concatenate(chunks, axis=0)
    if points.shape[0] < sum(spec.count for spec in BRAIN_REGIONS):
        raise RuntimeError("brain sampling produced fewer neurons than requested")
    return points[: sum(spec.count for spec in BRAIN_REGIONS)]


def _eye_points(rig: VrmRig, per_eye: int, seed: int = 23) -> np.ndarray:
    head = rig.world_pos("head")
    rng = np.random.default_rng(seed)
    clusters = []
    for offset in EYE_OFFSETS:
        center = head + np.array(offset, dtype=np.float64)
        direction = rng.normal(size=(per_eye, 3))
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        radius = np.cbrt(rng.random(per_eye))[:, None] * EYE_RADIUS
        clusters.append(center + direction * radius)
    return np.concatenate(clusters, axis=0)


def build_layout(rig: VrmRig, nervous: dict[str, Any], *, seed: int = 7) -> BodyLayout:
    brain_total = int(nervous["brain_neurons"])
    configured = sum(spec.count for spec in BRAIN_REGIONS)
    if configured != brain_total:
        raise ValueError(
            f"region counts sum to {configured}, config asks for {brain_total}"
        )
    brain = _brain_points(rig, seed=seed)
    region = np.repeat(
        np.arange(len(BRAIN_REGIONS), dtype=np.int32),
        [spec.count for spec in BRAIN_REGIONS],
    )
    chunks = [brain]
    chunk_bones = ["head"]
    limb_neurons = int(nervous["limb_neurons"])
    limb_ids = []
    action_dims = []
    for index, spec in enumerate(LIMBS):
        segments = spec.segments
        base = limb_neurons // len(segments)
        remainder = limb_neurons - base * len(segments)
        counts = [base + (1 if i < remainder else 0) for i in range(len(segments))]
        for (start, end, _), count in zip(segments, counts):
            chunks.append(_segment(rig, start, end, count))
            chunk_bones.append(start)
        limb_ids.append(np.full(limb_neurons, index, dtype=np.int32))
        action_dims.append(spec.action_dim)

    visual_count = int(nervous.get("visual_neurons", 0))
    if visual_count:
        if visual_count % 2:
            raise ValueError("visual_neurons must be even (two eye clusters)")
        chunks.append(_eye_points(rig, visual_count // 2))
        chunk_bones.append("head")

    positions = np.concatenate(chunks, axis=0).astype(np.float32)
    limb = np.concatenate(
        [np.full(brain.shape[0], -1, dtype=np.int32)]
        + limb_ids
        + [np.full(visual_count, -1, dtype=np.int32)]
    )
    region = np.concatenate(
        [region, np.full(limb.shape[0] - region.shape[0], -1, dtype=np.int32)]
    )
    visual = np.zeros(positions.shape[0], dtype=bool)
    if visual_count:
        visual[-visual_count:] = True

    bind_bones = sorted(set(chunk_bones), key=chunk_bones.index)
    bone_lookup = {name: index for index, name in enumerate(bind_bones)}
    bone_index = np.repeat(
        np.array([bone_lookup[name] for name in chunk_bones], dtype=np.int32),
        [chunk.shape[0] for chunk in chunks],
    )
    local_positions = np.empty_like(positions)
    cursor = 0
    for chunk, bone_name in zip(chunks, chunk_bones):
        size = chunk.shape[0]
        bone = rig.bones[bone_name]
        world_rotation = quat_to_matrix(bone.rest_world)
        local_positions[cursor : cursor + size] = (chunk - bone.rest_world_pos) @ world_rotation
        cursor += size

    # Private per-neuron senses for limb/torso pools: body-relative position and
    # the direction of the bound bone.  Brain and visual neurons stay at zero.
    private_features = np.zeros((positions.shape[0], 6), dtype=np.float32)
    private_joint = np.full(positions.shape[0], -1, dtype=np.int32)
    hips = rig.world_pos("hips")
    for index in range(brain.shape[0], brain.shape[0] + limb_neurons * len(LIMBS)):
        bone = bind_bones[bone_index[index]]
        rotation = quat_to_matrix(rig.bones[bone].rest_world)
        direction = rotation[:, 1]
        private_features[index] = np.concatenate(
            [(positions[index] - hips) / 0.5, direction]
        )
        private_joint[index] = VRM_TO_SMPL.get(bone, -1)

    return BodyLayout(
        positions=positions,
        region=region,
        limb=limb,
        region_names=[spec.id for spec in BRAIN_REGIONS],
        limb_names=[spec.id for spec in LIMBS],
        brain_center=(rig.world_pos("head") + np.array(SKULL_OFFSET)).astype(np.float32),
        head_origin=rig.world_pos("head").astype(np.float32),
        rig=rig,
        bind_bones=bind_bones,
        bone_index=bone_index,
        local_positions=local_positions.astype(np.float32),
        visual=visual,
        private_features=private_features,
        private_joint=private_joint,
        limb_action_dims=action_dims,
    )
