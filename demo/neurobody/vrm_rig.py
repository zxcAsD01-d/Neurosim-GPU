"""Read a VRM (glTF 2.0 + VRM 0.x / VRMC_vrm) file without any 3D runtime.

Only the rig is needed: the humanoid bone hierarchy, the rest pose and the rest
local rotations.  The parser works on the GLB container directly, so it does not
depend on three.js, Unity or a browser, and the same numbers can be exported to
the viewer as a JSON rig description.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .rotations import matrix_to_quat, quat_conjugate, quat_mul, quat_normalize

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A

# VRM humanoid order used by the export and by every downstream consumer.
HUMANOID_ORDER = [
    "hips",
    "spine",
    "chest",
    "upperChest",
    "neck",
    "head",
    "leftShoulder",
    "leftUpperArm",
    "leftLowerArm",
    "leftHand",
    "rightShoulder",
    "rightUpperArm",
    "rightLowerArm",
    "rightHand",
    "leftUpperLeg",
    "leftLowerLeg",
    "leftFoot",
    "leftToes",
    "rightUpperLeg",
    "rightLowerLeg",
    "rightFoot",
    "rightToes",
]


def read_glb_json(path: str | Path) -> dict[str, Any]:
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError(f"{path} is not a GLB file")
    if version != 2:
        raise ValueError(f"{path}: unsupported glTF version {version}")
    offset = 12
    while offset < min(length, len(data)):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == CHUNK_JSON:
            return json.loads(chunk.decode("utf-8"))
    raise ValueError(f"{path}: no JSON chunk")


def _is_vrm1(document: dict[str, Any]) -> bool:
    return "VRMC_vrm" in document.get("extensions", {})


def _humanoid_bones(document: dict[str, Any]) -> tuple[dict[str, int], str, dict[str, Any]]:
    extensions = document.get("extensions", {})
    if "VRMC_vrm" in extensions:
        spec = extensions["VRMC_vrm"]
        table = spec.get("humanoid", {}).get("humanBones", {})
        bones = {name: int(entry["node"]) for name, entry in table.items()}
        return bones, "1.0", spec.get("meta", {})
    if "VRM" in extensions:
        spec = extensions["VRM"]
        table = spec.get("humanoid", {}).get("humanBones", {})
        bones = {}
        for entry in table:
            name = entry.get("bone") or entry.get("name")
            bones[name] = int(entry["node"])
        return bones, "0.x", spec.get("meta", {})
    raise ValueError("the file has no VRM humanoid extension")


def _local_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        values = np.asarray(node["matrix"], dtype=np.float64)
        return values.reshape(4, 4).T
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    rotation = np.asarray(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    q = quat_normalize(np.array([rotation[3], rotation[0], rotation[1], rotation[2]]))
    matrix = np.eye(4)
    r = np.array(
        [
            [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
            [2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] - q[0] * q[1])],
            [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)],
        ]
    )
    matrix[:3, :3] = r * scale[None, :]
    matrix[:3, 3] = translation
    return matrix


@dataclass
class Bone:
    name: str
    node: int
    parent: str | None
    rest_local: np.ndarray  # (4,) wxyz
    rest_world: np.ndarray  # (4,) wxyz
    rest_local_pos: np.ndarray  # (3,)
    rest_world_pos: np.ndarray  # (3,)


@dataclass
class VrmRig:
    path: str
    version: str
    meta: dict[str, Any]
    bones: dict[str, Bone] = field(default_factory=dict)
    facing_yaw: float = 0.0
    height_m: float = 1.62

    def ordered(self) -> list[str]:
        return [name for name in HUMANOID_ORDER if name in self.bones]

    def world_pos(self, name: str) -> np.ndarray:
        return self.bones[name].rest_world_pos

    def to_json(self) -> dict[str, Any]:
        bones = {}
        for name in self.ordered():
            bone = self.bones[name]
            bones[name] = {
                "node": bone.node,
                "parent": bone.parent,
                "restLocal": [round(float(v), 9) for v in bone.rest_local],
                "restLocalPos": [round(float(v), 6) for v in bone.rest_local_pos],
                "restWorld": [round(float(v), 9) for v in bone.rest_world],
                "restWorldPos": [round(float(v), 6) for v in bone.rest_world_pos],
            }
        return {
            "version": 1,
            "source": self.path,
            "vrmVersion": self.version,
            "facingYawRad": round(float(self.facing_yaw), 9),
            "heightM": self.height_m,
            "meta": self.meta,
            "bones": bones,
            "order": self.ordered(),
        }

    @classmethod
    def load(cls, path: str | Path, *, height_m: float = 1.62) -> "VrmRig":
        path = str(path)
        document = read_glb_json(path)
        nodes = document.get("nodes", [])
        bone_nodes, version, meta = _humanoid_bones(document)
        node_to_name = {node: name for name, node in bone_nodes.items()}

        parents = [-1] * len(nodes)
        for index, node in enumerate(nodes):
            for child in node.get("children", []):
                parents[child] = index

        world_matrix: dict[int, np.ndarray] = {}

        def resolve(index: int) -> np.ndarray:
            if index in world_matrix:
                return world_matrix[index]
            local = _local_matrix(nodes[index])
            parent = parents[index]
            world = local if parent < 0 else resolve(parent) @ local
            world_matrix[index] = world
            return world

        rig = cls(path=path, version=version, meta=meta, height_m=height_m)
        for name, node_index in bone_nodes.items():
            world = resolve(node_index)
            local = _local_matrix(nodes[node_index])
            scale = np.linalg.norm(world[:3, 0])
            rest_world_pos = world[:3, 3] / max(scale, 1e-12)
            rest_local_pos = local[:3, 3]
            rest_world = matrix_to_quat(world[:3, :3] / max(scale, 1e-12))
            rest_local = matrix_to_quat(local[:3, :3] / max(np.linalg.norm(local[:3, 0]), 1e-12))
            parent_node = parents[node_index]
            rig.bones[name] = Bone(
                name=name,
                node=node_index,
                parent=node_to_name.get(parent_node),
                rest_local=rest_local,
                rest_world=rest_world,
                rest_local_pos=rest_local_pos,
                rest_world_pos=rest_world_pos,
            )
        rig.facing_yaw = _facing_yaw(rig)
        return rig


def _facing_yaw(rig: VrmRig) -> float:
    """Detect whether the glTF model faces +Z (0) or -Z (pi).

    Toe-forward versus foot is the only reliable hint shared by both VRM 0.x
    (Unity export) and VRM 1.0 files.
    """

    for side in ("left", "right"):
        foot = rig.bones.get(f"{side}Foot")
        toes = rig.bones.get(f"{side}Toes")
        if foot is not None and toes is not None:
            if toes.rest_world_pos[2] > foot.rest_world_pos[2] + 1e-4:
                return 0.0
            if toes.rest_world_pos[2] < foot.rest_world_pos[2] - 1e-4:
                return float(np.pi)
    # Fall back to the elbow direction: a T-pose model's arms are along X.
    return 0.0


def smpl_local_rotations(global_orient: np.ndarray, body_pose: np.ndarray) -> dict[str, np.ndarray]:
    """SMPL-H axis-angle arrays -> per-SMPL-joint local rotation matrices.

    ``global_orient`` is (T, 3); ``body_pose`` is (T, 63) for joints 1..21.
    SMPL's rest pose has identity world orientation for every joint, so the
    returned local rotation is also the joint's world delta.
    """

    from .rotations import quat_to_matrix

    out: dict[str, np.ndarray] = {}
    names = SMPL_BODY_JOINTS
    aa = np.concatenate([global_orient[:, None, :], body_pose.reshape(-1, 21, 3)], axis=1)
    quats = _axis_angle_to_quat(aa.reshape(-1, 3)).reshape(aa.shape[0], aa.shape[1], 4)
    matrices = quat_to_matrix(quats)
    for index, name in enumerate(names):
        out[name] = matrices[:, index]
    return out


def _axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    from .rotations import quat_from_axis_angle

    return quat_from_axis_angle(aa)


# SMPL-H joint order (parents in brackets), matching AMASS ``body_pose``.
SMPL_BODY_JOINTS = [
    "pelvis",
    "leftHip",
    "rightHip",
    "spine1",
    "leftKnee",
    "rightKnee",
    "spine2",
    "leftAnkle",
    "rightAnkle",
    "spine3",
    "leftFoot",
    "rightFoot",
    "neck",
    "leftCollar",
    "rightCollar",
    "head",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
]

# VRM bone -> SMPL joint index (index into the 22-joint body table above).
VRM_TO_SMPL = {
    "hips": 0,
    "spine": 3,
    "chest": 6,
    "upperChest": 9,
    "neck": 12,
    "head": 15,
    "leftShoulder": 13,
    "leftUpperArm": 16,
    "leftLowerArm": 18,
    "leftHand": 20,
    "rightShoulder": 14,
    "rightUpperArm": 17,
    "rightLowerArm": 19,
    "rightHand": 21,
    "leftUpperLeg": 1,
    "leftLowerLeg": 4,
    "leftFoot": 7,
    "leftToes": 10,
    "rightUpperLeg": 2,
    "rightLowerLeg": 5,
    "rightFoot": 8,
    "rightToes": 11,
}


def retarget_local(
    rig: VrmRig,
    smpl_rotations: dict[str, np.ndarray],
    *,
    yaw_rad: float | None = None,
) -> dict[str, np.ndarray]:
    """Transfer SMPL joint rotations onto the VRM rig's absolute local quats.

    Source rest orientation is identity for every SMPL joint, so the target
    local delta is ``W_t^T R W_t`` with ``W_t`` the bone's rest world
    orientation.  A world yaw correction ``C`` (SMPL vs glTF facing) is folded
    in as ``C R C^T``.
    """

    from .rotations import quat_from_euler, quat_to_matrix, matrix_to_quat

    yaw = rig.facing_yaw if yaw_rad is None else float(yaw_rad)
    correction = quat_to_matrix(quat_from_euler(ry=yaw))
    out: dict[str, np.ndarray] = {}
    for name, bone in rig.bones.items():
        if name not in VRM_TO_SMPL:
            continue
        source = smpl_rotations.get(SMPL_BODY_JOINTS[VRM_TO_SMPL[name]])
        if source is None:
            continue
        world_rest = quat_to_matrix(bone.rest_world)
        delta = world_rest.T @ (correction @ source @ correction.T) @ world_rest
        anim = matrix_to_quat(delta)
        out[name] = quat_normalize(quat_mul(bone.rest_local, anim))
    return out
