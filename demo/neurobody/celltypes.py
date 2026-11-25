"""Ten human cortical / subcortical cell classes and their parameters.

The engine's per-neuron dynamics are extended with a cell-type profile:

    x = slope * (current + drive - threshold) + dendrite * current * drive
    phi(x) = sign(x) * |x| ** power
    s <- clamp(alpha * s + (1 - alpha) * gain * phi(x), 0, 1)

``power = 1`` is the linear-rate regime; ``power > 1`` adds a nonlinear,
contrast-enhancing response.  ``differentiation`` raises slope/power globally,
which is the "switch to nonlinear differentiation when the error is large"
mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CellType:
    id: str
    label: str
    role: str  # excitatory / inhibitory / sensory / motor
    dale: int  # +1 excitatory, -1 inhibitory
    alpha: float  # membrane time constant (leak)
    gain: float
    threshold: float
    slope: float
    power: float  # nonlinearity exponent
    dendrite: float
    color: str


CELL_TYPES: tuple[CellType, ...] = (
    CellType("l23_pyramidal", "L2/3 锥体细胞", "excitatory", +1, 0.86, 1.05, 0.05, 1.0, 1.0, 0.05, "#ef5350"),
    CellType("l5_pyramidal", "L5 锥体细胞", "excitatory", +1, 0.74, 1.15, 0.10, 1.2, 1.5, 0.08, "#ff7043"),
    CellType("pv_basket", "PV+ 篮状细胞", "inhibitory", -1, 0.42, 1.30, 0.00, 1.6, 1.0, 0.02, "#5c6bc0"),
    CellType("sst_martinotti", "SST+ Martinotti", "inhibitory", -1, 0.90, 0.95, 0.15, 0.8, 1.0, 0.02, "#7e57c2"),
    CellType("vip", "VIP+ 中间神经元", "inhibitory", -1, 0.68, 1.10, 0.10, 1.0, 1.5, 0.04, "#ab47bc"),
    CellType("chandelier", "Chandelier 轴突-轴突细胞", "inhibitory", -1, 0.50, 1.20, 0.20, 2.0, 2.0, 0.01, "#8e24aa"),
    CellType("purkinje", "小脑浦肯野细胞", "inhibitory", -1, 0.34, 1.40, 0.00, 2.2, 2.0, 0.02, "#42a5f5"),
    CellType("msn", "纹状体中等棘神经元", "inhibitory", -1, 0.81, 1.00, 0.10, 0.9, 1.0, 0.03, "#26a69a"),
    CellType("thalamocortical", "丘脑-皮层中继细胞", "excitatory", +1, 0.55, 1.25, 0.05, 1.3, 1.5, 0.06, "#ffd54f"),
    CellType("alpha_motor", "α 运动神经元", "motor", +1, 0.70, 1.10, 0.05, 1.0, 1.0, 0.03, "#ff8a65"),
)

TYPE_INDEX = {cell.id: index for index, cell in enumerate(CELL_TYPES)}


def assign_cell_types(region_ids: np.ndarray, role_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Map functional role + brain region to one of the ten classes.

    ``role_ids``: 0/1 excitatory/inhibitory brain, 2 tonic, 3 phasic, 4/5 ON/OFF,
    6 motor, 7 error.
    """

    out = np.full(role_ids.shape[0], TYPE_INDEX["l23_pyramidal"], dtype=np.int32)
    for index in range(role_ids.shape[0]):
        role = int(role_ids[index])
        if role == 6:
            out[index] = TYPE_INDEX["alpha_motor"]
        elif role in (2, 3, 4, 5, 7):
            out[index] = TYPE_INDEX["thalamocortical"]
        else:
            region = int(region_ids[index]) if region_ids is not None else -1
            draw = rng.random()
            if region == 4:  # cerebellum
                out[index] = (
                    TYPE_INDEX["purkinje"] if draw < 0.6 else TYPE_INDEX["msn"] if draw < 0.8 else TYPE_INDEX["pv_basket"]
                )
            elif region == 1:  # motor cortex
                out[index] = (
                    TYPE_INDEX["l5_pyramidal"] if draw < 0.55 else TYPE_INDEX["l23_pyramidal"] if draw < 0.75
                    else TYPE_INDEX["pv_basket"] if draw < 0.9 else TYPE_INDEX["chandelier"]
                )
            elif region == 2:  # somatosensory
                out[index] = (
                    TYPE_INDEX["thalamocortical"] if draw < 0.4 else TYPE_INDEX["l23_pyramidal"] if draw < 0.7
                    else TYPE_INDEX["pv_basket"] if draw < 0.9 else TYPE_INDEX["sst_martinotti"]
                )
            elif region == 0:  # prefrontal
                out[index] = (
                    TYPE_INDEX["l23_pyramidal"] if draw < 0.6 else TYPE_INDEX["pv_basket"] if draw < 0.78
                    else TYPE_INDEX["sst_martinotti"] if draw < 0.9 else TYPE_INDEX["vip"]
                )
            else:  # association
                out[index] = (
                    TYPE_INDEX["l23_pyramidal"] if draw < 0.5 else TYPE_INDEX["vip"] if draw < 0.72
                    else TYPE_INDEX["sst_martinotti"] if draw < 0.9 else TYPE_INDEX["pv_basket"]
                )
    return out
