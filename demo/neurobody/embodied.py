"""Embodied pretraining loop without physics.

This is the corrected training logic from ``docs/embodied-loop.md`` with the
physics stage postponed:

* every control tick the brain receives **its own** sensors (per-joint angle
  and velocity, vestibular) plus two first-person eye images of a command card,
* motor neurons drive **their own** joint velocities through a fixed
  neuron->joint mapping, and a first-order servo integrates them,
* perception, action and learning run in one continuous stream (truncated BPTT
  over the last 8 ticks, update every 8 ticks, state carried forever).

Once this loop is stable, the servo body is swapped for MuJoCo (stage B+).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter

from .body import BRAIN_REGIONS
from .vision import GlyphEncoder

EYE_SIZE = 64
EYE_FOV_DEG = 70.0
CARD_DISTANCE = 1.2
EYE_BASELINE = 0.06
N_GAZE = 5


# --------------------------------------------------------------------------- eyes
def render_eye(
    glyph: torch.Tensor,
    yaw: float,
    pitch: float,
    side: float,
    size: int = EYE_SIZE,
) -> np.ndarray:
    """Approximate first-person view of a card at 1.2 m.

    ``glyph`` is a 32x32 RGB tensor; ``side`` is -1/0/+1 for left/centre/right
    eye.  This is a cheap projective approximation (no z-buffer), enough to make
    the visual input depend on gaze and eye.
    """

    base = Image.fromarray(
        (glyph.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ).resize((24, 24), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (8, 8, 10))
    focal = size / (2.0 * math.tan(math.radians(EYE_FOV_DEG) / 2.0))
    offset_x = focal * math.tan(yaw) + side * EYE_BASELINE * focal / (2.0 * CARD_DISTANCE)
    offset_y = focal * math.tan(pitch)
    # the card is ~0.34 m wide at 1.2 m
    card_px = int(round(focal * 0.34 / CARD_DISTANCE))
    card = base.resize((max(card_px, 6), max(card_px, 6)), Image.BILINEAR)
    left = int(round(size / 2 + offset_x - card.width / 2))
    top = int(round(size / 2 + offset_y - card.height / 2))
    canvas.paste(card, (left, top))
    canvas = canvas.filter(ImageFilter.GaussianBlur(0.9))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    noise = np.random.default_rng(0).normal(0.0, 0.012, array.shape).astype(np.float32)
    return np.clip(array + noise, 0.0, 1.0)


class EyeBank:
    """Pre-rendered binocular views for a few gaze bins and symbols."""

    def __init__(self, glyphs: list[torch.Tensor], size: int = EYE_SIZE) -> None:
        self.glyphs = glyphs
        self.yaws = np.linspace(-0.28, 0.28, N_GAZE)
        self.pitches = np.linspace(-0.18, 0.18, 3)
        self.images = torch.zeros(
            (len(glyphs), len(self.yaws), len(self.pitches), 2, 3, size, size)
        )
        for k, glyph in enumerate(glyphs):
            for y, yaw in enumerate(self.yaws):
                for p, pitch in enumerate(self.pitches):
                    for eye, side in enumerate((-1.0, 1.0)):
                        image = render_eye(glyph, float(yaw), float(pitch), side, size)
                        self.images[k, y, p, eye] = torch.from_numpy(image).permute(2, 0, 1)
        # Keep the bucket boundaries with the lookup table so ``lookup`` never
        # needs a CPU tensor when the gaze comes from the GPU.
        self.yaw_mid = torch.tensor((self.yaws[:-1] + self.yaws[1:]) / 2, dtype=torch.float32)
        self.pitch_mid = torch.tensor((self.pitches[:-1] + self.pitches[1:]) / 2, dtype=torch.float32)

    def to(self, device: torch.device | str) -> "EyeBank":
        self.images = self.images.to(device)
        self.yaw_mid = self.yaw_mid.to(device)
        self.pitch_mid = self.pitch_mid.to(device)
        return self

    def lookup(self, symbols: torch.Tensor, gaze_yaw: torch.Tensor, gaze_pitch: torch.Tensor) -> torch.Tensor:
        """(B,) symbol ids + (B,) gaze -> (B, 2, 3, S, S)."""

        y = torch.bucketize(gaze_yaw.contiguous(), self.yaw_mid.to(gaze_yaw.device))
        p = torch.bucketize(gaze_pitch.contiguous(), self.pitch_mid.to(gaze_pitch.device))
        return self.images[symbols, y, p]  # (B, 2, 3, S, S)


# --------------------------------------------------------------------------- body
@dataclass
class JointBody:
    """First-order servo: q <- q + clip(v, +-max_step)."""

    max_step: float = 0.12
    joints: int = 13
    q: torch.Tensor = field(init=False)

    def reset(self, batch: int, q0: torch.Tensor, device: torch.device) -> None:
        self.q = q0.to(device).clone()

    def step(self, velocity: torch.Tensor) -> torch.Tensor:
        self.q = self.q + torch.clamp(velocity, -self.max_step, self.max_step)
        return self.q


def sensor_rates(q: torch.Tensor, qd: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Per-joint sensors: (B, 13, 3) features -> (B, 13, 3) neuron rates."""

    features = torch.stack([torch.cos(q), torch.sin(q), torch.clamp(qd, -2.0, 2.0)], dim=-1)
    return torch.tanh(torch.einsum("bjk,jkc->bjc", features, weights) + bias)


# --------------------------------------------------------------------------- brain
@dataclass
class EmbodiedConfig:
    brain_neurons: int = 5000
    degree: int = 100
    alpha: float = 0.72
    gain: float = 1.05
    weight_scale: float = 0.14
    visual_per_eye: int = 256
    motor_per_joint: int = 8
    max_step: float = 0.12
    touch_neurons: int = 128
    audio_neurons: int = 64
    nocicept_neurons: int = 8
    memory_neurons: int = 384

    @property
    def n_joints(self) -> int:
        # 13 joints x 3 axis-angle components + 2 root translation DoF (x, z)
        return 41

    @property
    def n_sensory(self) -> int:
        # 3 per joint (cos, sin, velocity) + 16 vestibular + 1 error per DoF
        return self.n_joints * 3 + 16 + self.n_joints

    @property
    def n_visual(self) -> int:
        return 2 * self.visual_per_eye

    @property
    def n_motor(self) -> int:
        return self.n_joints * self.motor_per_joint

    @property
    def n_neurons(self) -> int:
        return (
            self.brain_neurons
            + self.n_sensory
            + self.n_visual
            + self.n_motor
            + self.touch_neurons
            + self.audio_neurons
            + self.nocicept_neurons
            + self.memory_neurons
        )


class EmbodiedBrain(nn.Module):
    def __init__(
        self,
        config: EmbodiedConfig,
        n_symbols: int,
        seed: int = 7,
        *,
        m1_to_motor: int = 0,
        error_to_brain: int = 0,
        error_to_motor: bool = True,
        motor_recurrence: float = 0.08,
        reflex_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_symbols = n_symbols
        self.m1_to_motor = int(m1_to_motor)
        self.error_to_brain = int(error_to_brain)
        self.error_to_motor = bool(error_to_motor)
        # The error->motor flag governs both explicit graph edges and the
        # historical direct motor readout.  Keeping the latter active when the
        # graph path is disabled silently created a teacher/error bypass.
        self.error_readout_gain = 12.0 if self.error_to_motor else 0.0
        self.motor_recurrence = float(motor_recurrence)
        self.reflex_gain = float(reflex_gain)
        rng = np.random.default_rng(seed)
        n = config.n_neurons
        cfg = config

        self.brain = slice(0, cfg.brain_neurons)
        self.sensory = slice(cfg.brain_neurons, cfg.brain_neurons + cfg.n_sensory)
        self.visual = slice(self.sensory.stop, self.sensory.stop + cfg.n_visual)
        self.motor = slice(self.visual.stop, self.visual.stop + cfg.n_motor)
        self.touch = slice(self.motor.stop, self.motor.stop + cfg.touch_neurons)
        self.audio = slice(self.touch.stop, self.touch.stop + cfg.audio_neurons)
        self.nocicept = slice(self.audio.stop, self.audio.stop + cfg.nocicept_neurons)
        self.memory = slice(self.nocicept.stop, self.nocicept.stop + cfg.memory_neurons)

        # Functional M1: ``body._brain_points`` lays neurons out in
        # ``BRAIN_REGIONS`` order, so the slice is available without moving a
        # single neuron.  The recurrent graph's local blocks are still five
        # equal partitions, so this is a functional (not anatomical) M1.
        offset = 0
        self.m1 = slice(0, 0)
        for spec in BRAIN_REGIONS:
            if spec.id == "m1":
                self.m1 = slice(offset, offset + spec.count)
                break
            offset += spec.count

        # --- neuron types -----------------------------------------------------
        # 0 excitatory, 1 inhibitory (brain); 2 tonic sensory, 3 phasic sensory,
        # 4 ON visual, 5 OFF visual, 6 motor, 7 error (climbing-fibre-like).
        type_ids = np.zeros(n, dtype=np.int32)
        brain = cfg.brain_neurons
        type_ids[:brain] = np.where(rng.random(brain) < 0.8, 0, 1)
        sensor_start = cfg.brain_neurons
        for j in range(cfg.n_joints):
            type_ids[sensor_start + j * 3 + 0] = 2
            type_ids[sensor_start + j * 3 + 1] = 2
            type_ids[sensor_start + j * 3 + 2] = 3
        vest_start = sensor_start + cfg.n_joints * 3
        type_ids[vest_start : vest_start + 16] = 2
        error_start = vest_start + 16
        type_ids[error_start : error_start + cfg.n_joints] = 7
        type_ids[self.visual] = np.where(rng.random(cfg.n_visual) < 0.5, 4, 5)
        type_ids[self.motor] = 6
        type_ids[self.touch] = 2
        type_ids[self.audio] = 2
        type_ids[self.nocicept] = 2
        type_ids[self.memory] = 0
        sign = np.ones(8)
        sign[1] = -1.0  # inhibitory brain neurons
        sign[5] = -1.0  # OFF visual neurons

        # --- ten human cell classes -------------------------------------------
        from .celltypes import CELL_TYPES, assign_cell_types

        region_ids = np.full(n, -1, dtype=np.int32)
        offset = 0
        for region_index, spec in enumerate(BRAIN_REGIONS):
            region_ids[offset : offset + spec.count] = region_index
            offset += spec.count
        cell_type = assign_cell_types(region_ids, type_ids, rng)
        self.register_buffer("cell_type", torch.as_tensor(cell_type))

        # --- sparse graph ------------------------------------------------------
        src, dst, weight = [], [], []

        def add(sources, targets, scale):
            sources = np.asarray(sources, dtype=np.int64).reshape(-1)
            targets = np.asarray(targets, dtype=np.int64).reshape(-1)
            scale = np.asarray(scale, dtype=np.float32).reshape(-1)
            dst.append(targets)
            src.append(sources)
            # Dale's law: the sign of an edge is fixed by the source type.
            weight.append((scale * sign[type_ids[sources]]).astype(np.float32))

        same = rng.random((brain, cfg.degree)) < 0.72
        glob = rng.integers(0, brain, size=(brain, cfg.degree))
        rows = np.repeat(np.arange(brain), cfg.degree).reshape(brain, cfg.degree)
        local = np.empty_like(rows)
        region_size = brain // 5
        for region in range(5):
            lo, hi = region * region_size, (region + 1) * region_size
            block = rows[lo:hi]
            local[lo:hi] = rng.integers(lo, hi, size=block.shape)
        add(
            np.where(same, local, glob),
            rows,
            rng.normal(0, cfg.weight_scale * 2 / math.sqrt(cfg.degree), size=brain * cfg.degree),
        )
        for count, start in ((cfg.n_sensory, sensor_start), (cfg.n_visual, self.visual.start)):
            sources = np.repeat(np.arange(start, start + count), cfg.degree)
            targets = rng.integers(0, brain, size=count * cfg.degree)
            add(sources, targets, rng.normal(0, cfg.weight_scale * 2.4, size=count * cfg.degree))
        motor_sources = rng.integers(self.motor.start, self.motor.stop, size=cfg.n_motor * cfg.degree)
        motor_targets = np.repeat(np.arange(self.motor.start, self.motor.stop), cfg.degree)
        add(
            motor_sources,
            motor_targets,
            rng.normal(
                0,
                cfg.weight_scale * 1.5 * self.motor_recurrence,
                size=cfg.n_motor * cfg.degree,
            ),
        )

        # --- reflex arcs: joint sensors -> that joint's motor pool -------------
        if self.reflex_gain > 0:
            reflex_src, reflex_dst = [], []
            for j in range(cfg.n_joints):
                for k in range(3):
                    for m in range(cfg.motor_per_joint):
                        reflex_src.append(sensor_start + j * 3 + k)
                        reflex_dst.append(self.motor.start + j * cfg.motor_per_joint + m)
            add(
                reflex_src,
                reflex_dst,
                rng.normal(
                    0.55 * self.reflex_gain,
                    0.05 * abs(self.reflex_gain),
                    size=len(reflex_src),
                ),
            )

        # --- error neurons -> motor pool (cerebellar climbing-fibre analogue) --
        self.error_slice = slice(error_start, error_start + cfg.n_joints)
        if error_to_motor:
            # Three fixed-sign excitatory paths from each joint's error neuron to
            # that joint's motor pool: positive error must mean positive velocity,
            # not a coin flip decided by eight random readout signs.
            err_src, err_dst, err_w = [], [], []
            for j in range(cfg.n_joints):
                for m in range(cfg.motor_per_joint):
                    for _ in range(3):
                        err_src.append(error_start + j)
                        err_dst.append(self.motor.start + j * cfg.motor_per_joint + m)
                        err_w.append(1.0)
            add(err_src, err_dst, np.asarray(err_w, dtype=np.float32))

        # --- descending M1 -> motor projection (vNext) -------------------------
        # Each motor neuron receives a fixed number of excitatory/inhibitory
        # edges from the functional M1 slice.  This is the missing link between
        # the 5,000-neuron brain and the body.
        if self.m1_to_motor:
            m1_sources = rng.integers(
                self.m1.start,
                self.m1.stop,
                size=(cfg.n_motor, self.m1_to_motor),
            )
            motor_targets = np.repeat(
                np.arange(self.motor.start, self.motor.stop)[:, None],
                self.m1_to_motor,
                axis=1,
            )
            add(
                m1_sources,
                motor_targets,
                rng.normal(
                    0,
                    cfg.weight_scale * 1.5,
                    size=cfg.n_motor * self.m1_to_motor,
                ),
            )

        # --- error neurons -> brain (vNext) ------------------------------------
        # The task error must be able to reach the recurrent brain, otherwise it
        # is only a PD input to the motor pool.
        if self.error_to_brain:
            err_brain_src = np.repeat(
                np.arange(error_start, error_start + cfg.n_joints),
                self.error_to_brain,
            )
            err_brain_dst = rng.integers(
                0, brain, size=cfg.n_joints * self.error_to_brain
            )
            add(
                err_brain_src,
                err_brain_dst,
                rng.normal(
                    0,
                    cfg.weight_scale * 1.8,
                    size=cfg.n_joints * self.error_to_brain,
                ),
            )

        for start, count in (
            (self.touch.start, cfg.touch_neurons),
            (self.audio.start, cfg.audio_neurons),
            (self.nocicept.start, cfg.nocicept_neurons),
            (self.memory.start, cfg.memory_neurons),
        ):
            sources = np.repeat(np.arange(start, start + count), cfg.degree)
            targets = rng.integers(0, brain, size=count * cfg.degree)
            add(sources, targets, rng.normal(0, cfg.weight_scale * 2.4, size=count * cfg.degree))
        mem_sources = rng.integers(0, brain, size=cfg.memory_neurons * 20)
        mem_targets = np.repeat(np.arange(self.memory.start, self.memory.stop), 20)
        add(mem_sources, mem_targets, rng.normal(0, cfg.weight_scale * 1.5, size=cfg.memory_neurons * 20))

        from neurosim_gpu import SparseTopology

        topology = SparseTopology.from_edges(
            n,
            torch.as_tensor(np.concatenate([block.reshape(-1) for block in src])),
            torch.as_tensor(np.concatenate([block.reshape(-1) for block in dst])),
            torch.as_tensor(np.concatenate([block.reshape(-1) for block in weight])),
        )
        self.register_buffer("src", topology.src)
        self.register_buffer("dst", topology.dst)
        self.register_buffer("in_rowptr", topology.in_rowptr)
        self.register_buffer("in_col", topology.in_col)
        self.register_buffer("out_weight_index", topology.out_weight_index)
        self.register_buffer(
            "in_weight_index", torch.argsort(topology.out_weight_index.to(torch.int64)).to(torch.int32)
        )
        self.edge_weight = nn.Parameter(topology.weight.clone())
        self.register_buffer("type_ids", torch.as_tensor(type_ids))

        # --- differentiated dynamics -------------------------------------------
        alpha_by_cell = np.array([cell.alpha for cell in CELL_TYPES], dtype=np.float32)
        gain_by_cell = np.array([cell.gain for cell in CELL_TYPES], dtype=np.float32)
        theta_by_cell = np.array([cell.threshold for cell in CELL_TYPES], dtype=np.float32)
        slope_by_cell = np.array([cell.slope for cell in CELL_TYPES], dtype=np.float32)
        power_by_cell = np.array([cell.power for cell in CELL_TYPES], dtype=np.float32)
        dendrite_by_cell = np.array([cell.dendrite for cell in CELL_TYPES], dtype=np.float32)
        self.register_buffer("alpha", torch.as_tensor(alpha_by_cell[cell_type]))
        self.register_buffer("gain", torch.as_tensor(gain_by_cell[cell_type]))
        self.register_buffer("theta", torch.as_tensor(theta_by_cell[cell_type]))
        self.register_buffer("base_slope", torch.as_tensor(slope_by_cell[cell_type]))
        self.register_buffer("base_power", torch.as_tensor(power_by_cell[cell_type]))
        self.register_buffer("slope", self.base_slope.clone())
        self.register_buffer("power", self.base_power.clone())
        self.dendrite = nn.Parameter(torch.as_tensor(dendrite_by_cell[cell_type]))

        # --- per-joint motor range --------------------------------------------
        limits = np.full(cfg.n_joints, 0.20, dtype=np.float32)
        limits[0:12] = 0.30    # arms
        limits[12:24] = 0.22   # legs
        limits[24:30] = 0.15   # neck / head
        limits[30:39] = 0.12   # torso
        limits[39:41] = 0.05   # root translation (m per tick)
        self.register_buffer("motor_limit", torch.as_tensor(limits))

        # --- per-neuron sensors ------------------------------------------------
        self.register_buffer(
            "sensor_w", torch.as_tensor(rng.normal(0, 1.0, size=(cfg.n_joints, 3, 3)), dtype=torch.float32)
        )
        self.register_buffer(
            "sensor_b", torch.as_tensor(rng.normal(0, 0.2, size=(cfg.n_joints, 3)), dtype=torch.float32)
        )
        self.register_buffer(
            "vest_w", torch.as_tensor(rng.normal(0, 0.6, size=(16, 10)), dtype=torch.float32)
        )
        self.register_buffer("vest_b", torch.zeros(16))

        # --- vision: shared encoder + one retina per eye -----------------------
        self.encoder = GlyphEncoder()
        self.retina_weight = nn.Parameter(torch.randn(2, 256, 32) / math.sqrt(32))
        self.retina_bias = nn.Parameter(torch.zeros(2, 256))
        cells = int(math.ceil(math.sqrt(256)))
        coords = []
        for index in range(256):
            row, col = divmod(index, cells)
            coords.append([(col + 0.5) / cells * 2.0 - 1.0, (row + 0.5) / cells * 2.0 - 1.0])
        self.register_buffer("retina_pos", torch.tensor(coords))
        self.visual_gain = nn.Parameter(torch.tensor(2.0))

        # --- motor: 8 neurons per joint, each with its own nonlinearity --------
        # Positive bias: a positive motor rate means a positive joint velocity, so
        # the error/reflex channel starts with the correct sign instead of a coin flip.
        self.motor_w = nn.Parameter(
            0.15 + torch.randn(cfg.n_joints, cfg.motor_per_joint) * 0.02
        )
        self.motor_gain = nn.Parameter(torch.tensor(1.0))
        self.motor_a = nn.Parameter(torch.full((cfg.n_motor,), 4.0))
        self.motor_b = nn.Parameter(torch.full((cfg.n_motor,), -0.5))

        # --- sensory organs ----------------------------------------------------
        from . import sensory

        touch_w, touch_b = sensory.make_touch_bank()
        self.register_buffer("touch_w", touch_w)
        self.register_buffer("touch_b", touch_b)
        audio_freq, audio_params = sensory.make_audio_bank()
        self.register_buffer("audio_freq", audio_freq)
        self.register_buffer("audio_params", audio_params)
        self.register_buffer("place_centers", sensory.make_place_bank())

        # --- predictive coding: the brain predicts its own next sensations ----
        self.pred_head = nn.Linear(
            42, 2 * cfg.n_joints + cfg.touch_neurons + cfg.audio_neurons + cfg.nocicept_neurons
        )

        # --- symbol readout ----------------------------------------------------
        self.symbol_head = nn.Linear(41, n_symbols)

        # --- self prediction ---------------------------------------------------
        self.self_w = nn.Parameter(torch.randn(n) * 0.15)
        self.self_b = nn.Parameter(torch.zeros(n))

    # ------------------------------------------------------------------ helpers
    def _csr(self):
        values = self.edge_weight[self.in_weight_index.to(torch.int64)]
        return torch.sparse_csr_tensor(
            self.in_rowptr, self.in_col, values,
            size=(self.config.n_neurons, self.config.n_neurons), check_invariants=False,
        )

    def encode(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        vestibular: torch.Tensor,
        images: torch.Tensor,
        target: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = q.shape[0]
        drive = q.new_zeros((batch, self.config.n_neurons))
        sensors = sensor_rates(q, qd, self.sensor_w, self.sensor_b)  # (B, J, 3)
        drive[:, self.sensory.start : self.sensory.start + self.config.n_joints * 3] = sensors.reshape(batch, -1)
        vest = torch.tanh(vestibular @ self.vest_w.T + self.vest_b)
        vest_start = self.sensory.start + self.config.n_joints * 3
        drive[:, vest_start : vest_start + 16] = vest
        # Error neurons: q_ref - q.  A training-time teacher channel; the brain
        # reads it like a climbing-fibre / proprioceptive mismatch signal.
        if target is None:
            error = torch.full_like(q, 0.5)
        else:
            # Rate neurons clamp at 0, so encode the signed error around a
            # 0.5 baseline: 0.5 means zero error, 1 means positive, 0 negative.
            error = 0.5 + 0.5 * torch.tanh(5.0 * (target - q))
        drive[:, vest_start + 16 : self.sensory.stop] = error

        # binocular: encode each eye, sample at the retina's own positions
        if self.config.n_visual and images is not None:
            b, e, c, h, w = images.shape
            flat = images.reshape(b * e, c, h, w)
            encoded = self.encoder(flat)
            # 32x32 glyphs encode to 8x8; the 64x64 eye images encode to 16x16.
            # Derive the grid from the tensor instead of hard-coding the glyph size.
            feature = encoded.reshape(b * e, encoded.shape[1], encoded.shape[2], encoded.shape[3])
            # One grid_sample call per eye image; 256 retinal positions, each with
            # its own 32-channel weight vector.
            grid = self.retina_pos[None, :, None, :].expand(b * e, -1, -1, -1)
            sample = F.grid_sample(
                feature, grid, mode="bilinear", padding_mode="border", align_corners=True
            ).squeeze(-1)  # (B*E, 32, 256)
            sample = sample.transpose(1, 2)  # (B*E, 256, 32)
            eye = torch.arange(b * e, device=images.device) % e
            response = torch.einsum(
                "bnc,bnc->bn", sample, self.retina_weight[eye]
            ) + self.retina_bias[eye]
            response = torch.tanh(response).reshape(b, -1)
            drive[:, self.visual] += response * self.visual_gain

        if phase is not None:
            from . import sensory

            drive[:, self.touch] = sensory.encode_touch(q, qd, self.touch_w, self.touch_b)
            drive[:, self.audio] = sensory.encode_audio(phase, self.audio_freq, self.audio_params)
            drive[:, self.nocicept] = sensory.encode_nocicept(qd)
            if pos is not None:
                drive[:, self.memory] = sensory.encode_memory(pos, phase, self.place_centers)

        return torch.tanh(drive)

    def advance(self, state: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        current = torch.sparse.mm(self._csr(), state.T).T
        # cell-type nonlinearity: threshold, slope, dendritic coincidence, power
        x = self.slope[None] * (current + drive - self.theta[None])
        x = x + self.dendrite[None] * current * drive
        phi = torch.sign(x) * torch.abs(x).clamp(max=6.0) ** self.power[None]
        equilibrium = self.gain * phi
        return torch.clamp(self.alpha * state + (1 - self.alpha) * equilibrium, 0.0, 1.0)

    def set_differentiation(self, level: float) -> None:
        """Raise slope/power of every cell class (adaptive nonlinear switch)."""

        with torch.no_grad():
            level = max(0.0, float(level))
            self.slope.copy_(self.base_slope * (1.0 + 0.6 * level))
            self.power.copy_(1.0 + (self.base_power - 1.0) * (1.0 + level))

    def initial_state(self, drive: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(drive) * 0.6

    def motor_velocity(self, state: torch.Tensor) -> torch.Tensor:
        rates = state[:, self.motor]
        batch = rates.shape[0]
        # each motor neuron applies its own sigmoid-like tuning curve
        tuned = torch.tanh(self.motor_a * rates + self.motor_b)
        tuned = tuned.reshape(batch, self.config.n_joints, self.config.motor_per_joint)
        raw = (tuned * self.motor_w[None]).sum(-1) * self.motor_gain
        # Fixed-sign reflex: each joint's error neuron adds directly to its own
        # velocity command.  Multi-path (3 synapses) above, one readout here.
        raw = raw + self.error_readout_gain * (state[:, self.error_slice] - 0.55)
        return self.motor_limit[None] * torch.tanh(raw)

    def symbol_logits(self, state: torch.Tensor) -> torch.Tensor:
        brain = state[:, : self.config.brain_neurons]
        pooled = brain.reshape(brain.shape[0], 40, -1).mean(-1)
        return self.symbol_head(torch.cat([brain.mean(-1, keepdim=True), pooled], dim=-1))

    @torch.no_grad()
    def rewire_(self, generator: torch.Generator, prune_frac: float = 0.02) -> tuple[int, int]:
        """Prune the weakest brain<->brain edges and grow new random ones.

        Only brain-to-brain edges are rewired for now; sensory/visual/motor
        projections keep their anatomical targets.  Dale's law is preserved
        because the sign is derived from the (new) source type.
        """

        brain = self.config.brain_neurons
        src = self.src.to(torch.int64).clone()
        dst = self.dst.to(torch.int64).clone()
        weight = self.edge_weight.detach().clone()
        brain_edges = (src < brain) & (dst < brain)
        candidates = torch.where(brain_edges)[0]
        if candidates.numel() == 0:
            return 0, 0
        count = max(int(candidates.numel() * prune_frac), 1)
        order = torch.argsort(weight[candidates].abs())[:count]
        victims = candidates[order]
        new_src = torch.randint(0, brain, (count,), generator=generator, dtype=torch.int64)
        new_dst = torch.randint(0, brain, (count,), generator=generator, dtype=torch.int64)
        src[victims] = new_src.to(src.device)
        dst[victims] = new_dst.to(dst.device)
        # new synapses start weak and take the sign of their source type
        source_types = self.type_ids.cpu()[new_src]
        sign = torch.where(
            (source_types == 1) | (source_types == 5),
            -torch.ones(count),
            torch.ones(count),
        )
        weight[victims] = (0.02 * sign).to(weight.device)

        from neurosim_gpu import SparseTopology

        topology = SparseTopology.from_edges(
            self.config.n_neurons, src, dst, weight
        )
        self.src = topology.src
        self.dst = topology.dst
        self.in_rowptr = topology.in_rowptr
        self.in_col = topology.in_col
        self.out_weight_index = topology.out_weight_index
        self.in_weight_index = torch.argsort(
            topology.out_weight_index.to(torch.int64)
        ).to(torch.int32)
        self.edge_weight = nn.Parameter(topology.weight.clone())
        return int(count), int(topology.n_edges)

    def predict_senses(self, state: torch.Tensor) -> torch.Tensor:
        brain = state[:, : self.config.brain_neurons]
        pooled = brain.reshape(brain.shape[0], 40, -1).mean(-1)
        summary = torch.cat(
            [brain.mean(-1, keepdim=True), pooled, state[:, self.visual].mean(-1, keepdim=True)],
            dim=-1,
        )
        return self.pred_head(summary)

    def predict_next(self, state: torch.Tensor) -> torch.Tensor:
        return state + 0.2 * torch.tanh(self.self_w * state + self.self_b)
