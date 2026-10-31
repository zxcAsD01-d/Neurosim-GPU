"""The sandwich model: pretrained encoder -> sparse cortical sheet -> decoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import build_backbone
from .brain import CorticalCore
from .config import ExperimentConfig


@dataclass
class CortexOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    state: torch.Tensor | None = None
    stats: dict[str, torch.Tensor] = field(default_factory=dict)


class CortexSandwichLM(nn.Module):
    """Encoder / brain / decoder language model.

    The brain output is added to the encoder residual stream through a small
    learnable gate. With the gate near zero and a frozen decoder the model
    starts close to the pretrained LM, so training measures what the brain adds
    rather than how quickly a random decoder forgets language.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        vocab_size: int,
        backbone: nn.Module | None = None,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.eos_token_id = int(
            getattr(tokenizer, "eos_token_id", 0) or 0
        )
        self.bos_token_id = int(
            getattr(tokenizer, "bos_token_id", 0) or 0
        )
        if backbone is None:
            backbone = build_backbone(config.backbone, vocab_size)
        self.backbone = backbone
        hidden_size = int(self.backbone.hidden_size)
        self.hidden_size = hidden_size
        self.vocab_size = int(self.backbone.vocab_size)
        if self.vocab_size < vocab_size:
            raise ValueError(
                f"backbone vocabulary ({self.vocab_size}) is smaller than the "
                f"tokenizer vocabulary ({vocab_size})"
            )

        self.brain: CorticalCore | None
        if config.model.use_brain:
            init_weight = None
            if config.brain.init_from_backbone:
                middle = getattr(self.backbone, "middle_layers", None)
                if not middle:
                    raise ValueError(
                        "brain.init_from_backbone=true but the backbone has no "
                        "removed middle layers to transplant"
                    )
                from .transplant import transplant_recurrence

                init_weight = transplant_recurrence(
                    middle, target_radius=config.brain.init_radius
                )
            release = getattr(self.backbone, "release_middle_layers", None)
            if release is not None:
                release()
            self.brain = CorticalCore(
                config.brain, hidden_size, init_weight=init_weight
            )
        else:
            self.brain = None
        self.brain_gate = nn.Parameter(
            torch.tensor(float(config.model.brain_gate_init))
        )
        if self.brain is None:
            self.brain_gate.requires_grad_(False)
        self.dropout = nn.Dropout(config.model.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        detach_interval: int | None = None,
    ) -> CortexOutput:
        hidden, context = self.backbone.encode(input_ids, attention_mask)
        encoder_hidden = hidden
        stats: dict[str, torch.Tensor] = {}
        brain_out = None
        if self.brain is not None:
            # Sparse CSR matmul does not support half precision reliably; the
            # sheet stays fp32 and only its interface crosses the autocast
            # boundary.
            dtype = hidden.dtype
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                brain_out, state, stats = self.brain(
                    hidden.float(),
                    state=state,
                    attention_mask=attention_mask,
                    detach_interval=detach_interval,
                )
            hidden = hidden + self.brain_gate.to(dtype) * brain_out.to(dtype)

        differentiation = None
        if (
            self.training
            and self.config.model.differentiation_weight > 0
            and brain_out is not None
            and brain_out.shape[1] > 1
        ):
            normalized = F.normalize(brain_out.float(), dim=-1)
            gram = normalized @ normalized.transpose(1, 2)
            eye = torch.eye(
                gram.shape[1], device=gram.device, dtype=gram.dtype
            )
            differentiation = ((gram - eye) ** 2).mean()
            stats["differentiation"] = differentiation.detach()

        hidden = self.dropout(hidden)
        hidden = self.backbone.decode(hidden, context)
        logits = self.backbone.logits(hidden)

        loss = None
        if labels is not None:
            loss = language_model_loss(
                logits,
                labels,
                label_smoothing=self.config.model.label_smoothing,
                z_loss=self.config.model.z_loss,
                eos_token_id=self.eos_token_id,
                bos_token_id=self.bos_token_id,
                eos_weight=self.config.model.eos_weight,
                bos_weight=self.config.model.bos_weight,
            )
            if (
                self.training
                and self.brain is not None
                and self.config.model.open_loop_weight > 0
                and self.config.model.open_loop_probability > 0
                and torch.rand((), device=labels.device)
                < self.config.model.open_loop_probability
            ):
                auxiliary = self._open_loop_loss(encoder_hidden, labels)
                if auxiliary is not None:
                    stats["open_loop_loss"] = auxiliary.detach()
                    loss = (
                        loss
                        + self.config.model.open_loop_weight * auxiliary
                    )
            if differentiation is not None:
                loss = (
                    loss
                    + self.config.model.differentiation_weight
                    * differentiation
                )
        return CortexOutput(
            logits=logits, loss=loss, state=state, stats=stats
        )

    def _open_loop_loss(
        self,
        encoder_hidden: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor | None:
        """Short zero-drive rollout, decoded without token feedback."""

        if self.brain is None:
            return None
        tokens = encoder_hidden.shape[1]
        steps = min(
            self.config.model.open_loop_tokens,
            max(1, tokens // 2 - 1),
        )
        prefix_length = tokens - steps - 1
        if prefix_length < 1:
            return None
        dtype = encoder_hidden.dtype
        with torch.autocast(
            device_type=encoder_hidden.device.type, enabled=False
        ):
            prefix = encoder_hidden[:, :prefix_length].float()
            brain_out, state, _ = self.brain(prefix)
            hidden = prefix.to(dtype) + self.brain_gate.to(dtype) * (
                brain_out.to(dtype)
            )
            for _ in range(steps):
                zero = torch.zeros_like(prefix[:, :1])
                brain_out, state, _ = self.brain(zero, state=state)
                hidden = torch.cat(
                    [
                        hidden,
                        hidden[:, -1:]
                        + self.brain_gate.to(dtype)
                        * brain_out.to(dtype),
                    ],
                    dim=1,
                )
            decoded = self.backbone.decode_hidden(
                hidden.to(dtype)
            )
            logits = self.backbone.logits(decoded)
        predictions = logits[
            :, prefix_length : prefix_length + steps
        ].float()
        targets = labels[
            :, prefix_length + 1 : prefix_length + steps + 1
        ]
        if targets.numel() == 0:
            return None
        return F.cross_entropy(
            predictions.reshape(-1, predictions.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy or nucleus sampling.

        This is the reference generator: it re-runs the prefix for every new
        token. A KV-cache path is deliberately deferred until the training
        pipeline has produced a checkpoint worth serving.
        """

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        self.eval()
        generated = input_ids
        batch = input_ids.shape[0]
        finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            logits = self(generated).logits[:, -1].float()
            if repetition_penalty != 1.0:
                for row in range(batch):
                    tokens = torch.unique(generated[row])
                    selected = logits[row, tokens]
                    logits[row, tokens] = torch.where(
                        selected < 0,
                        selected * repetition_penalty,
                        selected / repetition_penalty,
                    )

            if temperature <= 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    cutoff = torch.topk(
                        logits, min(top_k, logits.shape[-1]), dim=-1
                    ).values[..., -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_index = torch.sort(
                        logits, descending=True, dim=-1
                    )
                    cumulative = torch.softmax(
                        sorted_logits, dim=-1
                    ).cumsum(dim=-1)
                    remove = cumulative > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(
                        remove, float("-inf")
                    )
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        -1, sorted_index, sorted_logits
                    )
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            next_token = next_token.clone()
            for row in range(batch):
                if finished[row]:
                    next_token[row] = 0
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if bool(finished.all()):
                    break
        return generated

    @torch.no_grad()
    def stream_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 24,
        *,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Open-loop readout: prompt once, then let the brain run.

        The front plate is only used for the prompt prefill. After that the
        brain is stepped with zero drive, and the back plate decodes each new
        brain state. Generated tokens are **not** fed back into the brain.
        """

        if self.brain is None:
            raise RuntimeError("stream_generate requires a brain")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if input_ids.shape[0] != 1:
            raise ValueError("open-loop stream_generate supports batch size 1")
        self.eval()
        generated = input_ids
        hidden, _ = self.backbone.encode(input_ids)
        brain_out, state, _ = self.brain(hidden.float())
        hidden = hidden + self.brain_gate.to(hidden.dtype) * brain_out.to(
            hidden.dtype
        )
        hidden = self.dropout(hidden)

        for step in range(max_new_tokens):
            logits = self.backbone.logits(
                self.backbone.decode_hidden(hidden)
            )[:, -1].float()
            if repetition_penalty != 1.0:
                tokens = torch.unique(generated)
                selected = logits[0, tokens]
                logits[0, tokens] = torch.where(
                    selected < 0,
                    selected * repetition_penalty,
                    selected / repetition_penalty,
                )
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_index = torch.sort(
                        logits, descending=True, dim=-1
                    )
                    cumulative = torch.softmax(
                        sorted_logits, dim=-1
                    ).cumsum(dim=-1)
                    remove = cumulative > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(
                        remove, float("-inf")
                    )
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        -1, sorted_index, sorted_logits
                    )
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), num_samples=1
                )
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and int(next_token) == eos_token_id:
                break

            # No token feedback: only the brain's own next state enters the
            # back plate. The last hidden vector is kept as a residual anchor.
            zero_drive = torch.zeros_like(hidden[:, :1])
            brain_out, state, _ = self.brain(zero_drive, state=state)
            hidden = torch.cat(
                [
                    hidden,
                    hidden[:, -1:]
                    + self.brain_gate.to(hidden.dtype)
                    * brain_out.to(hidden.dtype),
                ],
                dim=1,
            )
        return generated

    def trainable_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                yield name, parameter

    def describe(self) -> dict[str, object]:
        groups = {"brain": 0, "lora": 0, "other": 0, "frozen": 0}
        total = 0
        for name, parameter in self.named_parameters():
            count = parameter.numel()
            total += count
            if not parameter.requires_grad:
                groups["frozen"] += count
            elif name.startswith("brain.") or name == "brain_gate":
                groups["brain"] += count
            elif ".lora_" in name:
                groups["lora"] += count
            else:
                groups["other"] += count
        trainable = total - groups["frozen"]
        report: dict[str, object] = {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "groups": groups,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "backbone": self.config.backbone.kind,
            "encoder_layers": getattr(self.backbone, "encoder_count", None),
            "decoder_layers": getattr(self.backbone, "decoder_count", None),
        }
        if self.brain is not None:
            report["brain"] = {
                "n_neurons": self.brain.n_neurons,
                "n_edges": self.brain.n_edges,
                "weight_mode": self.brain.weight_mode,
                "ticks_per_token": self.config.brain.ticks_per_token,
            }
            if self.brain.weight_mode == "dense":
                from .transplant import spectral_radius

                report["brain"]["recurrent_radius"] = spectral_radius(
                    self.brain.recurrent.detach().float()
                )
        return report

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.trainable_parameters()
        }

    def load_trainable_state_dict(
        self,
        state: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        expected = {name for name, _ in self.trainable_parameters()}
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "trainable checkpoint mismatch: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        result = self.load_state_dict(state, strict=False)
        if strict and result.unexpected_keys:
            raise RuntimeError(
                f"unexpected checkpoint keys: {result.unexpected_keys[:5]}"
            )


def language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    eos_token_id: int = 0,
    bos_token_id: int = 0,
    eos_weight: float = 1.0,
    bos_weight: float = 1.0,
) -> torch.Tensor:
    """Token-level cross entropy with optional start/end weighting."""

    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits {tuple(logits.shape[:2])} and labels "
            f"{tuple(labels.shape)} do not align"
        )
    logits = logits.float()
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
        reduction="none",
    ).view_as(labels)
    weights = torch.ones_like(token_loss)
    if eos_token_id and eos_weight != 1.0:
        weights = torch.where(
            labels == eos_token_id,
            torch.full_like(weights, eos_weight),
            weights,
        )
    if bos_token_id and bos_weight != 1.0:
        weights = torch.where(
            labels == bos_token_id,
            torch.full_like(weights, bos_weight),
            weights,
        )
    weights = torch.where(labels == -100, torch.zeros_like(weights), weights)
    loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)
    if z_loss:
        log_normalizer = torch.logsumexp(logits, dim=-1)
        loss = loss + z_loss * log_normalizer.pow(2).mean()
    return loss
