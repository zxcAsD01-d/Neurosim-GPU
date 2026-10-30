"""Dopamine (three-factor) probe on the MNIST digit-string task.

Question: under what conditions does the engine's dopamine rule actually have
behavioural leverage?

Protocol
--------
1. Train the retina+brain on clean digit strings.
2. Freeze the base model.
3. Adapt on a *shifted* distribution (Gaussian noise / contrast inversion) with
   one of several plasticity settings:
       frozen            no weight change (control)
       reward=0          three-factor rule with a zero dopamine signal (control)
       rec               plasticity only on recurrent edges
       sensory           plasticity only on the retinal projection
       both              plasticity on both
4. Compare word/char accuracy on a held-out shifted split before vs after.

Only engine primitives are used for plasticity:
``update_eligibility_trace`` and ``dopamine_gated_weight_update``.
``src/neurosim_gpu`` is not modified.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch

from common import DATA, emit_result, load_mnist, seed_everything

from neurosim_gpu import dopamine_gated_weight_update, update_eligibility_trace

from mnist_seq import DEFAULTS as SEQ_DEFAULTS
from mnist_seq import build_sequences, shift_images
from str_brain import Charset, RetinaBrain, evaluate, greedy_decode, train_model


def run_with_traces(model, images, trace_rec, trace_sens, decay):
    """One forward pass that also updates eligibility traces (no gradients)."""

    batch = images.shape[0]
    state = None
    outputs = []
    with torch.no_grad():
        for step in range(model.steps):
            start = step * model.stride
            patch = images[:, :, start : start + model.win].reshape(batch, -1)
            sensory = model.sensory(patch)
            if state is None:
                state = torch.sigmoid(sensory)
            prev = state
            drive = 0.75 * torch.tanh(sensory)
            state = model.dynamics.advance(state, model._recurrent(state), drive)
            outputs.append(model.readout(state))
            if trace_rec is not None:
                trace_rec = update_eligibility_trace(
                    trace_rec, prev, state, model.rec_src, model.rec_dst, decay=decay
                )
            if trace_sens is not None:
                trace_sens = update_eligibility_trace(
                    trace_sens, patch, state, model.sensory.src, model.sensory.dst, decay=decay
                )
        log_probs = torch.log_softmax(torch.stack(outputs, dim=1), dim=-1)
    return log_probs, trace_rec, trace_sens


def adapt(
    model,
    images,
    targets,
    charset,
    device,
    *,
    which: str,
    eta: float,
    decay: float,
    epochs: int,
    batch: int,
    reward_mode: str,
) -> dict[str, float]:
    """Run the three-factor rule for ``epochs`` passes over the shifted split."""

    use_rec = which in ("rec", "both")
    use_sens = which in ("sensory", "both")
    trace_rec = torch.zeros_like(model.edge_weight, device=device) if use_rec else None
    trace_sens = torch.zeros_like(model.sensory.weight, device=device) if use_sens else None

    valid = [i for i, t in enumerate(targets) if t]
    updates = 0
    reward_total = 0.0
    n_batches = 0
    for _ in range(epochs):
        for start in range(0, len(valid), batch):
            idx = valid[start : start + batch]
            x = images.index_select(0, torch.tensor(idx)).float().div_(255.0).to(device)
            gold = ["".join(charset.chars[i] for i in targets[j]) for j in idx]
            log_probs, trace_rec, trace_sens = run_with_traces(
                model, x, trace_rec, trace_sens, decay
            )
            preds = greedy_decode(log_probs, charset)
            correct = sum(p == g for p, g in zip(preds, gold))
            if reward_mode == "sign":
                dopamine = torch.tensor(
                    2.0 * correct / len(gold) - 1.0, device=device
                )
            elif reward_mode == "zero":
                dopamine = torch.tensor(0.0, device=device)
            elif reward_mode == "pos":
                # Reward only successes; never push on failures.
                dopamine = torch.tensor(correct / len(gold), device=device)
            else:
                raise ValueError(reward_mode)
            reward_total += float(dopamine)
            n_batches += 1
            if which == "frozen":
                continue
            with torch.no_grad():
                if use_rec:
                    step = dopamine_gated_weight_update(
                        model.edge_weight, trace_rec, dopamine, learning_rate=eta
                    )
                    model.edge_weight.copy_(step.weight)
                if use_sens:
                    step = dopamine_gated_weight_update(
                        model.sensory.weight, trace_sens, dopamine, learning_rate=eta
                    )
                    model.sensory.weight.copy_(step.weight)
            updates += 1
    return {
        "updates": updates,
        "mean_reward": reward_total / max(n_batches, 1),
        "trace_rec_norm": float(trace_rec.norm()) if trace_rec is not None else 0.0,
        "trace_sens_norm": float(trace_sens.norm()) if trace_sens is not None else 0.0,
    }


CONDITIONS = [
    ("frozen", "frozen", 0.0, "sign"),
    ("both_eta0.001", "both", 0.001, "sign"),
    ("both_eta0.002", "both", 0.002, "sign"),
    ("sensory_eta0.001", "sensory", 0.001, "sign"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--adapt-epochs", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = dict(SEQ_DEFAULTS)
    if args.config_json:
        config.update(json.loads(args.config_json))
    config["epochs"] = args.epochs
    if args.smoke:
        config.update({"epochs": 2, "train_n": 1024, "test_n": 512, "batch": 16})
        args.adapt_epochs = 1

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

    base = RetinaBrain(config, seed, len(charset)).to(device)
    with torch.no_grad():
        base.readout.bias[0] = float(config["blank_bias"])
    print("[stage] training base model", flush=True)
    train_model(base, train_images, train_targets, charset, config, device, seed)
    clean = evaluate(base, test_images, test_targets, charset, device)
    print(f"[base] clean word_acc={clean['word_accuracy']:.4f} char_acc={clean['char_accuracy']:.4f}", flush=True)

    results: dict[str, dict] = {}
    for shift in ("noise15", "shift"):
        shifted_train = shift_images(train_images, shift, seed + 3)
        shifted_test = shift_images(test_images, shift, seed + 4)
        before = evaluate(base, shifted_test, test_targets, charset, device)
        print(f"\n[shift={shift}] before adaptation word_acc={before['word_accuracy']:.4f}", flush=True)
        rows = {}
        for name, which, eta, reward in CONDITIONS:
            model = copy.deepcopy(base)
            stats = adapt(
                model,
                shifted_train,
                train_targets,
                charset,
                device,
                which=which,
                eta=eta,
                decay=0.9,
                epochs=args.adapt_epochs,
                batch=int(config["batch"]),
                reward_mode=reward,
            )
            after = evaluate(model, shifted_test, test_targets, charset, device)
            rows[name] = {
                "which": which,
                "eta": eta,
                "reward": reward,
                "word_accuracy": after["word_accuracy"],
                "char_accuracy": after["char_accuracy"],
                "delta_word": after["word_accuracy"] - before["word_accuracy"],
                **stats,
            }
            print(
                f"  {name:16s} word_acc={after['word_accuracy']:.4f} "
                f"d={rows[name]['delta_word']:+.4f} char_acc={after['char_accuracy']:.4f} "
                f"updates={stats['updates']} reward={stats['mean_reward']:+.3f}",
                flush=True,
            )
        results[shift] = {"before": before, "conditions": rows}

    payload = {
        "experiment": "mnist_seq_dopamine_probe",
        "config": config,
        "device": str(device),
        "clean": clean,
        "results": results,
    }
    if args.json:
        emit_result(payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
