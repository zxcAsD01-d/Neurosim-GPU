"""Explicit reduced-precision paths for the three-state sparse core."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, cast

import torch


Precision = Literal["fp32", "fp16", "fp8", "nvfp4"]
PRECISIONS = ("fp32", "fp16", "fp8", "nvfp4")
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@lru_cache(maxsize=None)
def _e2m1_constants(
    device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse immutable device tables; initialize during CUDA graph warmup.

    Normal tensors allow an inference-mode warmup to precede training. Keeping
    the tables on device also avoids a host-to-device copy on every NVFP4 tick.
    """
    with torch.inference_mode(False), torch.no_grad():
        return (
            torch.tensor(_E2M1_MIDPOINTS, device=device, dtype=dtype),
            torch.tensor(_E2M1_LEVELS, device=device, dtype=dtype),
        )


def validate_precision(value: str) -> Precision:
    if value not in PRECISIONS:
        raise ValueError("precision must be 'fp32', 'fp16', 'fp8', or 'nvfp4'")
    return cast(Precision, value)


def _ste(value: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and value.requires_grad:
        return quantized + (value - value.detach())
    return quantized


def _encode_e4m3(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = value.abs().amax() if value.numel() else value.new_zeros(())
    scale = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    codes = (value / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return codes, scale


def _encode_e2m1(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode E2M1 blocks along the last dimension, padding on the right."""
    block_size = 16
    width = value.shape[-1]
    pad = (-width) % block_size
    if pad:
        value = torch.nn.functional.pad(value, (0, pad))
    blocks = value.reshape(*value.shape[:-1], (width + pad) // block_size, block_size)
    abs_blocks = blocks.abs()
    block_amax = abs_blocks.amax(dim=-1, keepdim=True)
    amax = block_amax.amax() if value.numel() else value.new_zeros(())
    global_scale = (amax / (6.0 * 448.0)).clamp_min(torch.finfo(torch.float32).tiny)
    block_scale = (block_amax / (6.0 * global_scale)).clamp_max(448.0).to(torch.float8_e4m3fn)
    denominator = global_scale * block_scale.float()
    normalized = abs_blocks / torch.where(denominator > 0, denominator, 1.0)

    # Bucketize in O(numel) memory. Exact ties round to the even E2M1 code.
    midpoints, _ = _e2m1_constants(value.device, value.dtype)
    nearest = torch.bucketize(normalized, midpoints)
    on_odd_boundary = (nearest % 2 == 1) & (normalized == midpoints[nearest.clamp_max(6)])
    code = (nearest + on_odd_boundary).to(torch.uint8) | (blocks.signbit().to(torch.uint8) << 3)
    return code, block_scale.squeeze(-1), global_scale


def encode_sparse_weights(
    value: torch.Tensor, precision: str
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Return compressed CSR value bytes, optional block-scale bytes and FP32 scale."""
    if value.ndim != 1 or value.dtype != torch.float32:
        raise ValueError("encoding requires a vector of FP32 CSR weights")
    if precision == "fp8":
        codes, scale = _encode_e4m3(value.detach())
        return codes.view(torch.uint8), None, scale
    if precision != "nvfp4":
        raise ValueError("compressed CSR encoding requires fp8 or nvfp4")
    codes, scales, scale = _encode_e2m1(value.detach())
    codes = codes.flatten()
    packed = codes[::2] | (codes[1::2] << 4)
    return packed[: (value.numel() + 1) // 2].contiguous(), scales.view(torch.uint8), scale


def quantize_dequantize(value: torch.Tensor, precision: Precision) -> torch.Tensor:
    """Round-trip a tensor through the named format, with STE gradients."""
    validate_precision(precision)
    if precision == "fp32" or value.numel() == 0:
        return value
    if value.dtype != torch.float32:
        raise ValueError("reduced-precision quantization requires an FP32 master tensor")
    if precision == "fp16":
        return _ste(value, value.detach().to(torch.float16).to(value.dtype))
    if precision == "fp8":
        codes, scale = _encode_e4m3(value.detach())
        decoded = codes.float() * scale
    else:
        vector = value.detach() if value.ndim else value.detach().reshape(1)
        codes, block_scales, scale = _encode_e2m1(vector)
        _, levels = _e2m1_constants(value.device, value.dtype)
        magnitude = levels[(codes & 7).long()] * (block_scales.float() * scale).unsqueeze(-1)
        decoded = torch.where(codes >= 8, -magnitude, magnitude).flatten(-2)
        decoded = decoded[..., :vector.shape[-1]].reshape(value.shape)
    return _ste(value, decoded)
