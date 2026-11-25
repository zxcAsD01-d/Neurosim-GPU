"""Small, dependency-light rotation helpers (numpy, float64).

Quaternions are stored ``[w, x, y, z]`` and rotations are right-handed, Y-up,
following both glTF and three.js.  Everything here is batched over leading
axes so a clip of ``T`` frames is one call.
"""
from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return normalize(q, axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., 1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = q[..., 1:]
    w = q[..., :1]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 4) quaternion."""

    aa = np.asarray(axis_angle, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    half = 0.5 * angle
    # sin(half)/angle with the limit 0.5 at angle -> 0.
    scale = np.where(angle > 1e-9, np.sin(half) / np.maximum(angle, 1e-12), 0.5)
    q = np.concatenate([np.cos(half), aa * scale], axis=-1)
    return quat_normalize(q)


def quat_from_euler(rx: float = 0.0, ry: float = 0.0, rz: float = 0.0) -> np.ndarray:
    qx = np.array([np.cos(rx / 2), np.sin(rx / 2), 0.0, 0.0])
    qy = np.array([np.cos(ry / 2), 0.0, np.sin(ry / 2), 0.0])
    qz = np.array([np.cos(rz / 2), 0.0, 0.0, np.sin(rz / 2)])
    return quat_normalize(quat_mul(quat_mul(qy, qx), qz))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(np.asarray(q, dtype=np.float64))
    w, x, y, z = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    out = np.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        axis=-1,
    )
    return out.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrix -> (..., 4) quaternion (wxyz)."""

    m = np.asarray(m, dtype=np.float64)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    out = np.empty((m.shape[0], 4), dtype=np.float64)
    for i, r in enumerate(m):
        trace = r[0, 0] + r[1, 1] + r[2, 2]
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
        out[i] = (w, x, y, z)
    return quat_normalize(out).reshape(shape + (4,))


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    return quat_to_matrix(quat_from_axis_angle(aa))


def matrix_to_axis_angle(m: np.ndarray) -> np.ndarray:
    q = quat_normalize(matrix_to_quat(m))
    # Keep the scalar part non-negative so the angle stays in [0, pi].
    sign = np.where(q[..., :1] < 0.0, -1.0, 1.0)
    q = q * sign
    vec = q[..., 1:]
    sin_half = np.linalg.norm(vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, q[..., :1])
    scale = np.where(sin_half > 1e-9, angle / np.maximum(sin_half, 1e-12), 2.0)
    return vec * scale


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    if float(np.max(sin_theta)) < 1e-6:
        return quat_normalize(a * (1.0 - t) + b * t)
    w1 = np.sin((1.0 - t) * theta) / np.maximum(sin_theta, 1e-12)
    w2 = np.sin(t * theta) / np.maximum(sin_theta, 1e-12)
    return quat_normalize(w1 * a + w2 * b)


def look_rotation(forward: np.ndarray, up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Quaternion whose local +Z points along ``forward`` (three.js convention)."""

    z = normalize(np.asarray(forward, dtype=np.float64))
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
    x = normalize(x)
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=-1)
    return matrix_to_quat(m)
