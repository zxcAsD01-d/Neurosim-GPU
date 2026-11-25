"""Build the nervous system: 5000 brain neurons + 6 limb pools of 16.

The graph is sparse and fixed; only its weights are trainable.  Three kinds of
projection are predefined:

* local recurrent edges inside each brain region (plus a fraction of long-range
  edges),
* descending edges from the motor region to every limb pool,
* ascending edges from every limb pool to the somatosensory and cerebellar
  regions.

A separate fixed fanout carries the sensory feature vector into the graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .body import BRAIN_REGIONS, LIMBS, BodyLayout, build_layout
from .vrm_rig import VRM_TO_SMPL, VrmRig

# proprioception: 22 joint rotations, 22 angular velocities, root velocity,
# root height.  No phase: it would be window-relative during training.
FEATURE_DIM = 22 * 3 + 22 * 3 + 3 + 1


@dataclass(frozen=True)
class ActionJoint:
    limb: str
    smpl_joint: int
    vrm_bone: str


def action_joints() -> list[ActionJoint]:
    out: list[ActionJoint] = []
    for limb in LIMBS:
        for bone in limb.joints:
            out.append(ActionJoint(limb.id, VRM_TO_SMPL[bone], bone))
    return out


@dataclass
class NervousSystem:
    layout: BodyLayout
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    slices: dict[str, slice]
    sensory_src: np.ndarray
    sensory_dst: np.ndarray
    sensory_weight: np.ndarray
    action: list[ActionJoint]
    seed: int

    @property
    def n_neurons(self) -> int:
        return self.layout.n_neurons

    @property
    def n_edges(self) -> int:
        return int(self.src.size)

    def summary(self) -> dict[str, Any]:
        keys = self.src.astype(np.int64) * self.n_neurons + self.dst.astype(np.int64)
        canonical = int(np.unique(keys).size)
        return {
            "neurons": self.n_neurons,
            "edges": canonical,
            "rawEdges": self.n_edges,
            "brain": int((self.layout.region >= 0).sum()),
            "visual": int(self.layout.visual.sum()),
            "limbs": {spec.id: int((self.layout.limb == i).sum()) for i, spec in enumerate(LIMBS)},
            "features": FEATURE_DIM,
            "actionDim": 3 * len(self.action),
            "regions": {
                spec.id: int((self.layout.region == i).sum())
                for i, spec in enumerate(BRAIN_REGIONS)
            },
        }

    def to_json(self) -> dict[str, Any]:
        data = self.layout.to_json()
        data["sensory"] = {
            "features": FEATURE_DIM,
            "hidden": None,
        }
        data["action"] = [
            {"limb": joint.limb, "smplJoint": joint.smpl_joint, "vrmBone": joint.vrm_bone}
            for joint in self.action
        ]
        data["edges"] = self.n_edges
        return data


def build_nervous_system(
    rig: VrmRig,
    config: dict[str, Any],
    *,
    seed: int = 7,
) -> NervousSystem:
    nervous = config["nervous"]
    layout = build_layout(rig, nervous, seed=seed)
    brain_n = int(nervous["brain_neurons"])
    degree = int(nervous["degree"])
    scale = float(nervous["weight_scale"]) * 2.0
    rng = np.random.default_rng(seed)
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    weight: list[np.ndarray] = []

    region_offsets = np.cumsum([0] + [spec.count for spec in BRAIN_REGIONS])
    region_of = np.repeat(np.arange(len(BRAIN_REGIONS)), [spec.count for spec in BRAIN_REGIONS])
    m1_start, m1_end = int(region_offsets[1]), int(region_offsets[2])
    s1_start, s1_end = int(region_offsets[2]), int(region_offsets[3])
    cb_start, cb_end = int(region_offsets[3]), int(region_offsets[4])

    # --- brain recurrence ---------------------------------------------------
    same_region = rng.random((brain_n, degree)) < 0.72
    global_src = rng.integers(0, brain_n, size=(brain_n, degree))
    local_src = np.empty((brain_n, degree), dtype=np.int64)
    for region_index in range(len(BRAIN_REGIONS)):
        rows = np.where(region_of == region_index)[0]
        lo, hi = int(region_offsets[region_index]), int(region_offsets[region_index + 1])
        local_src[rows] = rng.integers(lo, hi, size=(rows.size, degree))
    sources = np.where(same_region, local_src, global_src)
    targets = np.repeat(np.arange(brain_n), degree).reshape(brain_n, degree)
    src.append(sources.reshape(-1))
    dst.append(targets.reshape(-1))
    weight.append(rng.normal(0.0, scale / np.sqrt(degree), size=brain_n * degree))

    # --- descending motor projection ---------------------------------------
    limb_total = int(nervous["limb_neurons"]) * len(LIMBS)
    descending = int(nervous.get("descending", 8))
    m1_sources = rng.integers(m1_start, m1_end, size=(limb_total, descending))
    limb_targets = np.arange(brain_n, brain_n + limb_total)[:, None]
    src.append(m1_sources.reshape(-1))
    dst.append(np.repeat(limb_targets, descending, axis=1).reshape(-1))
    weight.append(rng.normal(0.0, scale * 0.8, size=limb_total * descending))

    # --- ascending sensory projection --------------------------------------
    ascending_s1 = int(nervous.get("ascending_s1", 4))
    ascending_cb = int(nervous.get("ascending_cerebellum", 2))
    limb_sources = np.arange(brain_n, brain_n + limb_total)
    s1_targets = rng.integers(s1_start, s1_end, size=(limb_total, ascending_s1))
    cb_targets = rng.integers(cb_start, cb_end, size=(limb_total, ascending_cb))
    src.append(np.repeat(limb_sources, ascending_s1))
    dst.append(s1_targets.reshape(-1))
    weight.append(rng.normal(0.0, scale * 0.8, size=limb_total * ascending_s1))
    src.append(np.repeat(limb_sources, ascending_cb))
    dst.append(cb_targets.reshape(-1))
    weight.append(rng.normal(0.0, scale * 0.8, size=limb_total * ascending_cb))

    # --- local limb recurrence ---------------------------------------------
    local = int(nervous.get("limb_local", 2))
    pool_sources = []
    pool_targets = []
    for index in range(len(LIMBS)):
        base = brain_n + index * int(nervous["limb_neurons"])
        size = int(nervous["limb_neurons"])
        rows = np.arange(base, base + size)
        picks = rng.integers(0, size, size=(size, local))
        picks = picks + np.arange(size)[:, None]
        picks %= size
        pool_sources.append(base + picks)
        pool_targets.append(np.repeat(rows[:, None], local, axis=1))
    src.append(np.concatenate(pool_sources).reshape(-1))
    dst.append(np.concatenate(pool_targets).reshape(-1))
    weight.append(rng.normal(0.0, scale, size=limb_total * local))

    # --- visual neurons -> brain -------------------------------------------
    visual_count = int(nervous.get("visual_neurons", 0))
    if visual_count:
        visual_base = brain_n + limb_total
        visual_sources = np.arange(visual_base, visual_base + visual_count)
        fanout = int(nervous.get("visual_fanout", 8))
        targets = rng.integers(0, brain_n, size=(visual_count, fanout))
        src.append(np.repeat(visual_sources, fanout))
        dst.append(targets.reshape(-1))
        weight.append(rng.normal(0.0, scale * 1.2, size=visual_count * fanout))

    src_all = np.concatenate(src).astype(np.int64)
    dst_all = np.concatenate(dst).astype(np.int64)
    weight_all = np.concatenate(weight).astype(np.float32)

    # --- fixed sensory fanout (features -> hidden is learned separately) ----
    hidden = int(nervous["sensory_hidden"])
    fanout = int(nervous["sensory_fanout"])
    sensory_src = np.repeat(np.arange(hidden), fanout).astype(np.int64)
    sensory_dst = rng.integers(0, layout.n_neurons, size=hidden * fanout).astype(np.int64)
    sensory_weight = rng.uniform(0.25, 0.55, size=hidden * fanout).astype(np.float32)
    sensory_weight *= rng.choice([-1.0, 1.0], size=sensory_weight.size)

    slices = {"brain": slice(0, brain_n)}
    for index, spec in enumerate(BRAIN_REGIONS):
        slices[spec.id] = slice(int(region_offsets[index]), int(region_offsets[index + 1]))
    for index, spec in enumerate(LIMBS):
        base = brain_n + index * int(nervous["limb_neurons"])
        slices[spec.id] = slice(base, base + int(nervous["limb_neurons"]))
    if visual_count:
        visual_base = brain_n + limb_total
        slices["visual"] = slice(visual_base, visual_base + visual_count)

    return NervousSystem(
        layout=layout,
        src=src_all,
        dst=dst_all,
        weight=weight_all,
        slices=slices,
        sensory_src=sensory_src,
        sensory_dst=sensory_dst,
        sensory_weight=sensory_weight,
        action=action_joints(),
        seed=seed,
    )
