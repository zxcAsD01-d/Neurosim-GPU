"""Controlled sequence task on MNIST: multi-digit strings.

Purpose: give the sparse "brain" a *learnable* sequence task before probing the
dopamine rule on Union14M.  Digits are pasted left-to-right onto a wide canvas;
the retina scans columns with a sliding window and CTC decodes the digit string.

Everything after the retina is plastic; ``src/neurosim_gpu`` is only called.

Usage::

    python experiments/mnist_seq.py --config-json '{"epochs":30}'
    python experiments/mnist_seq.py --config-json '{"epochs":30}' --shift noise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import DATA, emit_result, load_mnist, seed_everything, trainable_parameters

from str_brain import Charset, RetinaBrain, evaluate, train_model

CANVAS_H = 28
DIGIT_W = 28
GAP = 4
K_MAX = 3
CANVAS_W = K_MAX * DIGIT_W + (K_MAX - 1) * GAP  # 92

DEFAULTS = {
    "img_h": CANVAS_H,
    "img_w": CANVAS_W,
    "scan_win": 10,
    "scan_stride": 4,
    "neurons": 512,
    "fanout": 32,
    "degree": 8,
    "inner_steps": 1,
    "alpha": 0.6,
    "gain": 1.0,
    "weight_scale": 0.18,
    "sensory_gain": 2.0,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 30,
    "batch": 32,
    "train_n": 20000,
    "test_n": 2000,
    "warmup_frac": 0.05,
    "max_text_len": 3,
    "blank_bias": -6.0,
    "device": "auto",
    "seed": 0,
}


def build_sequences(
    images: torch.Tensor,
    labels: torch.Tensor,
    n: int,
    *,
    seed: int,
    min_digits: int = 2,
) -> tuple[torch.Tensor, list[str]]:
    """Paste ``min_digits``..3 random digits left-aligned on a white canvas."""

    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(images.shape[0], generator=generator)[:n]
    canvas = torch.full((n, CANVAS_H, CANVAS_W), 1.0)
    texts: list[str] = []
    for row, src in enumerate(index.tolist()):
        k = int(torch.randint(min_digits, K_MAX + 1, (1,), generator=generator).item())
        picks = torch.randint(images.shape[0], (k,), generator=generator).tolist()
        digits = []
        for slot, pi in enumerate(picks):
            digit = images[pi].view(CANVAS_H, DIGIT_W)
            left = slot * (DIGIT_W + GAP)
            canvas[row, :, left : left + DIGIT_W] = torch.minimum(
                canvas[row, :, left : left + DIGIT_W], digit
            )
            digits.append(str(int(labels[pi].item())))
        texts.append("".join(digits))
    return (canvas * 255.0).clamp(0, 255).to(torch.uint8), texts


def shift_images(images: torch.Tensor, mode: str, seed: int) -> torch.Tensor:
    """Distribution shift applied only at evaluation time."""

    generator = torch.Generator().manual_seed(seed)
    x = images.float().div(255.0)
    if mode == "noise":
        x = (x + torch.randn(x.shape, generator=generator) * 0.30).clamp(0.0, 1.0)
    elif mode == "noise15":
        x = (x + torch.randn(x.shape, generator=generator) * 0.15).clamp(0.0, 1.0)
    elif mode == "invert":
        x = 1.0 - x
    elif mode == "blur":
        kernel = torch.tensor([[[[1.0, 2.0, 1.0]]]]) / 4.0
        x = torch.nn.functional.conv2d(x.unsqueeze(1), kernel, padding=(0, 1)).squeeze(1)
    elif mode == "shift":
        x = torch.roll(x, shifts=2, dims=2)
    elif mode == "none":
        return images
    else:
        raise ValueError(f"unknown shift {mode!r}")
    return (x * 255.0).clamp(0, 255).to(torch.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--shift", default="none", choices=["none", "noise", "invert", "blur", "shift"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = dict(DEFAULTS)
    if args.config_json:
        config.update(json.loads(args.config_json))
    if args.smoke:
        config.update({"epochs": 1, "train_n": 1024, "test_n": 256, "batch": 16})

    requested = str(config.get("device", "auto"))
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested)
    seed = int(config["seed"])
    seed_everything(seed)

    data = load_mnist(DATA / "mnist")
    charset = Charset([str(d) for d in range(10)])

    train_images, train_texts = build_sequences(
        data["train_images"], data["train_labels"], int(config["train_n"]), seed=seed
    )
    test_images, test_texts = build_sequences(
        data["test_images"], data["test_labels"], int(config["test_n"]), seed=seed + 1
    )
    train_targets = [charset.encode(t) for t in train_texts]
    test_targets = [charset.encode(t) for t in test_texts]

    model = RetinaBrain(config, seed, len(charset)).to(device)
    with torch.no_grad():
        model.readout.bias[0] = float(config["blank_bias"])

    print(
        f"[data] train={len(train_targets)} test={len(test_targets)} "
        f"charset={len(charset)} steps={(CANVAS_W-config['scan_win'])//config['scan_stride']+1} "
        f"device={device}",
        flush=True,
    )

    result = train_model(
        model, train_images, train_targets, charset, config, device, seed
    )
    clean = evaluate(model, test_images, test_targets, charset, device)
    print(f"[clean] word_acc={clean['word_accuracy']:.4f} char_acc={clean['char_accuracy']:.4f}", flush=True)

    shifted = {}
    for mode in ("noise", "invert", "blur", "shift"):
        if args.shift not in ("none", mode, "all"):
            continue
        view = shift_images(test_images, mode, seed + 2)
        shifted[mode] = evaluate(model, view, test_targets, charset, device)
        print(
            f"[{mode}] word_acc={shifted[mode]['word_accuracy']:.4f} "
            f"char_acc={shifted[mode]['char_accuracy']:.4f}",
            flush=True,
        )

    payload = {
        "experiment": "mnist_digit_strings",
        "config": config,
        "device": str(device),
        "charset_size": len(charset),
        "trainable_params": trainable_parameters(model),
        "edges": model.edge_count,
        "train_seconds": result["seconds"],
        "history": result["history"],
        "clean": clean,
        "shifted": shifted,
    }
    if args.json:
        emit_result(payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
