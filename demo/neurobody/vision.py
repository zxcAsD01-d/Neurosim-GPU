"""Text -> glyph image -> single-neuron retinal code.

The visual channel is deliberately image-based:

    text ("红色三角形")  ->  (shape, colour)  ->  32x32 RGB icon
      ->  small conv encoder (shared)  ->  8x8x32 feature map
      ->  512 retinal neurons, each with its own position and weight vector
          (one neuron = one scalar, "single-neuron encoding")
      ->  visual neurons in the body

Keeping the icon as the interface means new symbols are added by rendering a
new image, not by hand-designing feature columns for the network.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

IMAGE_SIZE = 32
FEATURE_CHANNELS = 32
FEATURE_GRID = 8

COLORS = {
    "red": (239, 83, 80),
    "green": (102, 187, 106),
    "blue": (66, 165, 245),
    "yellow": (255, 213, 79),
    "cyan": (79, 195, 247),
    "purple": (171, 71, 188),
    "orange": (255, 138, 101),
    "white": (240, 244, 250),
    "black": (20, 20, 24),
}

COLOR_ALIASES = {
    "红": "red", "红色": "red", "red": "red",
    "绿": "green", "绿色": "green", "green": "green",
    "蓝": "blue", "蓝色": "blue", "blue": "blue",
    "黄": "yellow", "黄色": "yellow", "yellow": "yellow",
    "青": "cyan", "青色": "cyan", "cyan": "cyan",
    "紫": "purple", "紫色": "purple", "purple": "purple",
    "橙": "orange", "橙色": "orange", "orange": "orange",
    "白": "white", "白色": "white", "white": "white",
    "黑": "black", "黑色": "black", "black": "black",
}

SHAPE_ALIASES = {
    "圆": "circle", "圆形": "circle", "circle": "circle",
    "方": "square", "方形": "square", "正方形": "square", "square": "square",
    "三角": "triangle", "三角形": "triangle", "triangle": "triangle",
    "星": "star", "星形": "star", "五角星": "star", "star": "star",
    "六边形": "hexagon", "六角": "hexagon", "hexagon": "hexagon",
    "菱形": "diamond", "diamond": "diamond",
}

SHAPES = ("circle", "square", "triangle", "star", "hexagon", "diamond")


@dataclass(frozen=True)
class Symbol:
    shape: str
    color: str
    rgb: tuple[int, int, int]

    @property
    def id(self) -> str:
        return f"{self.shape}_{self.color}"


def parse_text(text: str) -> Symbol:
    """Very small tokenizer: find a colour and a shape substring.

    Falls back to a white square when either part is missing, which keeps the
    demo usable for arbitrary input.
    """

    lowered = text.strip().lower()
    color = "white"
    shape = "square"
    # longest aliases first so 三角形 is not captured by 三角 plus noise
    for alias in sorted(COLOR_ALIASES, key=len, reverse=True):
        if alias in lowered:
            color = COLOR_ALIASES[alias]
            break
    for alias in sorted(SHAPE_ALIASES, key=len, reverse=True):
        if alias in lowered:
            shape = SHAPE_ALIASES[alias]
            break
    return Symbol(shape=shape, color=color, rgb=COLORS[color])


def _polygon(shape: str) -> list[tuple[float, float]]:
    """Unit-circle polygon vertices, y down."""

    def point(angle: float, radius: float = 1.0) -> tuple[float, float]:
        return (math.sin(angle) * radius, -math.cos(angle) * radius)

    if shape == "triangle":
        return [point(math.tau * i / 3) for i in range(3)]
    if shape == "diamond":
        return [point(math.tau * i / 4, 1.0) for i in range(4)]
    if shape == "hexagon":
        return [point(math.tau * i / 6) for i in range(6)]
    if shape == "star":
        verts = []
        for i in range(10):
            radius = 1.0 if i % 2 == 0 else 0.45
            verts.append(point(math.tau * i / 10, radius))
        return verts
    return []


def render(symbol: Symbol, size: int = IMAGE_SIZE, margin: float = 0.16) -> Image.Image:
    image = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    cx = cy = size / 2.0
    radius = size * (0.5 - margin)
    if symbol.shape == "circle":
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=symbol.rgb)
    elif symbol.shape == "square":
        draw.rectangle((cx - radius, cy - radius, cx + radius, cy + radius), fill=symbol.rgb)
    else:
        verts = [(cx + x * radius, cy + y * radius) for x, y in _polygon(symbol.shape)]
        draw.polygon(verts, fill=symbol.rgb)
    return image


def render_tensor(symbol: Symbol, size: int = IMAGE_SIZE) -> torch.Tensor:
    array = np.asarray(render(symbol, size), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class GlyphEncoder(nn.Module):
    """3x conv, 32x32 RGB -> 8x8x32 feature map."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, FEATURE_CHANNELS, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class Retina(nn.Module):
    """512 visual neurons, each sampling its own position with its own weights."""

    def __init__(self, neurons: int = 512, channels: int = FEATURE_CHANNELS, seed: int = 11) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        # stratified positions in [-1, 1]^2 (grid_sample coordinates)
        cells = int(math.ceil(math.sqrt(neurons)))
        coords = []
        for index in range(neurons):
            row, col = divmod(index, cells)
            jitter = torch.rand(2, generator=generator) * 0.9 - 0.45
            coords.append(
                torch.tensor(
                    [
                        (col + 0.5 + jitter[0]) / cells * 2.0 - 1.0,
                        (row + 0.5 + jitter[1]) / cells * 2.0 - 1.0,
                    ]
                )
            )
        self.register_buffer("positions", torch.stack(coords))
        self.weight = nn.Parameter(torch.randn(neurons, channels, generator=generator) / math.sqrt(channels))
        self.bias = nn.Parameter(torch.zeros(neurons))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch = feature.shape[0]
        grid = self.positions[None, :, None, :].expand(batch, -1, -1, -1)
        sampled = F.grid_sample(
            feature, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # (B, C, N, 1)
        sampled = sampled.squeeze(-1).transpose(1, 2)  # (B, N, C)
        return torch.tanh(torch.einsum("bnc,nc->bn", sampled, self.weight) + self.bias)


class VisualFrontend(nn.Module):
    """Image -> encoder -> retinal code, shared by every visual neuron."""

    def __init__(self, neurons: int = 512) -> None:
        super().__init__()
        self.encoder = GlyphEncoder()
        self.retina = Retina(neurons=neurons)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.retina(self.encoder(images))
