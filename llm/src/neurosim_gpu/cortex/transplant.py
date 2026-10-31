"""Weight transplant from removed transformer blocks into a dense neuron sheet.

The transformer stack is not literally a recurrent matrix, so "direct
replacement" is defined here as the **linearised residual transform** of the
removed blocks:

    h_{l+1} = h_l + Attn(h_l) + MLP(h_l)

For a block we take the value/output path ``o_proj @ v_proj`` and the MLP path
``down_proj @ up_proj`` (the gated/SiLU nonlinearity and the attention softmax
are dropped). Composing the residual maps of the removed blocks gives a
``[d, d]`` matrix ``A``; the neuron sheet's recurrent matrix is initialised to
``A`` rescaled to a target spectral radius.

This is a functional initialisation, not an exact equivalence. It gives the
sheet a pretrained, non-random starting point while keeping the computation
trainable and sparse-free (``d=1024`` is small enough for a dense matrix).
"""

from __future__ import annotations

from typing import Iterable

import torch


def linearized_block_matrix(layer: torch.nn.Module) -> torch.Tensor:
    """Return ``o @ repeat(v) + down @ up`` for one Qwen-style block."""

    attention = layer.self_attn
    mlp = layer.mlp
    with torch.no_grad():
        value = attention.v_proj.weight.detach().float()
        output = attention.o_proj.weight.detach().float()
        query_width = int(attention.q_proj.weight.shape[0])
        value_width = int(value.shape[0])
        if query_width % value_width:
            raise ValueError(
                "q_proj and v_proj widths are not GQA-compatible: "
                f"{query_width} vs {value_width}"
            )
        repeat = query_width // value_width
        if repeat > 1:
            # GQA: repeat each KV head so the value path has the query width
            # before the output projection.
            head_dim = int(getattr(attention, "head_dim", 0))
            if head_dim <= 0:
                raise ValueError("attention head_dim is missing")
            kv_heads = value_width // head_dim
            value = (
                value.view(kv_heads, head_dim, value.shape[1])
                .repeat_interleave(repeat, dim=0)
                .reshape(query_width, value.shape[1])
            )
        attention_map = output @ value

        up = mlp.up_proj.weight.detach().float()
        down = mlp.down_proj.weight.detach().float()
        mlp_map = down @ up

    if attention_map.shape != mlp_map.shape:
        raise ValueError(
            "attention and MLP linearisations have different shapes: "
            f"{tuple(attention_map.shape)} vs {tuple(mlp_map.shape)}"
        )
    return attention_map + mlp_map


def spectral_radius(matrix: torch.Tensor, iterations: int = 20) -> float:
    """Estimate the spectral norm with power iteration on ``A^T A``."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("spectral_radius expects a square matrix")
    vector = torch.randn(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    vector = vector / vector.norm().clamp_min(1e-12)
    for _ in range(iterations):
        vector = matrix.t() @ (matrix @ vector)
        vector = vector / vector.norm().clamp_min(1e-12)
    return float((matrix @ vector).norm())


def transplant_recurrence(
    layers: Iterable[torch.nn.Module],
    *,
    target_radius: float = 0.95,
) -> torch.Tensor:
    """Compose removed blocks into a recurrent matrix with a bounded radius."""

    layers = list(layers)
    if not layers:
        raise ValueError("no middle layers to transplant")
    dimension = int(layers[0].self_attn.o_proj.out_features)
    identity = torch.eye(dimension, dtype=torch.float32)
    recurrence = identity
    for layer in layers:
        block = linearized_block_matrix(layer)
        if block.shape != (dimension, dimension):
            raise ValueError(
                f"expected a {dimension}x{dimension} block, got {tuple(block.shape)}"
            )
        recurrence = (identity + block) @ recurrence

    radius = spectral_radius(recurrence)
    if not torch.isfinite(recurrence).all() or radius <= 0:
        raise RuntimeError(
            f"transplant produced an invalid recurrence (radius={radius})"
        )
    return recurrence * (float(target_radius) / radius)
