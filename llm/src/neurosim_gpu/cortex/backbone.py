"""Text encoder and decoder halves.

``CortexSandwichLM`` does not re-implement a transformer. It loads a pretrained
causal LM and splits it at a layer boundary: embedding plus the first
``encoder_layers`` blocks encode tokens, the remaining blocks plus the LM head
decode the brain's output. ``kind="tiny"`` provides the same interface from
scratch so the pipeline can be tested without downloading a checkpoint.

Hugging Face is an optional dependency. Importing this module only requires
PyTorch; the transformers import happens inside :meth:`HFBackbone.from_pretrained`.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import BackboneConfig


@dataclass(frozen=True)
class BackboneContext:
    """Per-sequence data shared by the encoder and decoder halves."""

    attention_mask: Any
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None
    position_ids: torch.Tensor


class LoRALinear(nn.Module):
    """A frozen linear layer plus a low-rank trainable update."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.scale = float(alpha) / float(rank)
        nn.init.normal_(self.lora_a.weight, std=0.02)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        update = self.lora_b(self.lora_a(self.dropout(hidden_states)))
        return self.base(hidden_states) + update * self.scale


def inject_lora(
    module: nn.Module,
    targets: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    """Replace matching ``nn.Linear`` children with :class:`LoRALinear`."""

    targets = set(targets)
    replaced = 0
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear) or not name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, leaf, LoRALinear(child, rank, alpha, dropout))
        replaced += 1
    return replaced


class _TinyBlock(nn.Module):
    """Pre-norm causal transformer block used by the offline stand-in."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        query = self.q_proj(hidden_states).view(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch, tokens, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, tokens, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        if attention_mask is None:
            attended = F.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )
        else:
            causal = torch.ones(
                tokens, tokens, dtype=torch.bool, device=hidden_states.device
            ).tril()
            keep = causal.unsqueeze(0).unsqueeze(0) & attention_mask.to(
                torch.bool
            ).unsqueeze(1).unsqueeze(2)
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=keep
            )
        attended = attended.transpose(1, 2).reshape(batch, tokens, -1)
        hidden_states = residual + self.o_proj(attended)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.down_proj(
            F.gelu(self.up_proj(hidden_states))
        )
        return residual + hidden_states


class TinyBackbone(nn.Module):
    """A small causal LM with the same encoder/decoder split as the HF path."""

    def __init__(self, config: BackboneConfig, vocab_size: int) -> None:
        super().__init__()
        total = config.tiny_num_layers
        encoder = config.encoder_layers
        decoder = config.decoder_layers or total - encoder
        if encoder + decoder != total:
            raise ValueError(
                "tiny encoder_layers + decoder_layers must equal tiny_num_layers"
            )
        if decoder <= 0:
            raise ValueError("tiny backbone needs at least one decoder layer")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.config = config
        self.hidden_size = config.tiny_hidden_size
        self.vocab_size = int(vocab_size)
        self.encoder_count = encoder
        self.decoder_count = decoder

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.tiny_max_position_embeddings, self.hidden_size
        )
        make_block = lambda: _TinyBlock(  # noqa: E731
            self.hidden_size,
            config.tiny_num_heads,
            config.tiny_num_kv_heads,
            config.tiny_intermediate_size,
        )
        self.encoder_layers = nn.ModuleList(
            [make_block() for _ in range(encoder)]
        )
        self.decoder_layers = nn.ModuleList(
            [make_block() for _ in range(decoder)]
        )
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > self.position_embeddings.num_embeddings:
            raise ValueError("sequence is longer than tiny_max_position_embeddings")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return self.embed_tokens(input_ids) + self.position_embeddings(positions)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, BackboneContext]:
        hidden = self.embed(input_ids)
        for block in self.encoder_layers:
            hidden = block(hidden, attention_mask)
        context = BackboneContext(
            attention_mask=attention_mask,
            position_embeddings=None,
            position_ids=torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0),
        )
        return hidden, context

    def decode(
        self,
        hidden: torch.Tensor,
        context: BackboneContext,
    ) -> torch.Tensor:
        for block in self.decoder_layers:
            hidden = block(hidden, context.attention_mask)
        return hidden

    def decode_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode a hidden-state sequence that has no token ids attached."""

        context = BackboneContext(
            attention_mask=attention_mask,
            position_embeddings=None,
            position_ids=torch.arange(
                hidden.shape[1], device=hidden.device
            ).unsqueeze(0),
        )
        return self.decode(hidden, context)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(hidden))

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.embed_tokens.parameters()
        yield from self.position_embeddings.parameters()
        yield from self.encoder_layers.parameters()

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.decoder_layers.parameters()
        yield from self.final_norm.parameters()


class HFBackbone(nn.Module):
    """A pretrained causal LM split into encoder and decoder layer stacks.

    The encoder is the first ``encoder_layers`` blocks; the decoder is the last
    ``decoder_layers`` blocks. Any blocks in between are the part replaced by
    the cortical sheet and are removed from the module tree, so the frozen
    backbone only pays for the retained blocks.
    """

    def __init__(self, model: nn.Module, config: BackboneConfig) -> None:
        super().__init__()
        self.model = model
        base = self.base
        if base is None or not hasattr(base, "layers"):
            raise ValueError(
                "HFBackbone expects a model with a transformer `.layers` stack"
            )
        self.config = config
        self.hidden_size = int(model.config.hidden_size)
        self.vocab_size = int(model.config.vocab_size)

        total = len(base.layers)
        encoder = config.encoder_layers
        decoder = config.decoder_layers or total - encoder
        if encoder < 0 or encoder >= total:
            raise ValueError(
                f"encoder_layers ({encoder}) must be in [0, {total})"
            )
        if decoder <= 0:
            raise ValueError("the decoder half needs at least one layer")
        if encoder + decoder > total:
            raise ValueError(
                f"encoder_layers ({encoder}) + decoder_layers ({decoder}) "
                f"exceeds the backbone's {total} layers"
            )
        self._encoder_layers = list(base.layers[:encoder])
        # Explicit decoder_layers means "the last N blocks"; the layers in
        # between are the part replaced by the cortical sheet.
        self._decoder_layers = list(base.layers[total - decoder :])
        self._middle_layers = list(base.layers[encoder : total - decoder])
        self.encoder_count = encoder
        self.decoder_count = decoder
        self.middle_count = total - encoder - decoder
        self._layer_types = list(
            getattr(model.config, "layer_types", None)
            or ["full_attention"] * total
        )
        self._encoder_types = self._layer_types[:encoder]
        self._decoder_types = self._layer_types[total - decoder :]
        self._has_sliding = any(
            layer_type != "full_attention" for layer_type in self._layer_types
        )
        self.gradient_checkpointing = bool(config.gradient_checkpointing)
        self._passes = self._layer_signature(self._decoder_layers[-1])

        # Retain only the encoder and decoder blocks. The unused middle
        # transformer layers are dropped from the module tree, so they do not
        # occupy GPU memory and cannot leak into the optimizer or checkpoint.
        if self.middle_count:
            base.layers = nn.ModuleList(
                self._encoder_layers + self._decoder_layers
            )

    @property
    def base(self) -> nn.Module | None:
        base = getattr(self.model, "model", None)
        if base is not None:
            return base
        prefix = getattr(self.model, "base_model_prefix", "")
        return getattr(self.model, prefix, None) if prefix else None

    @property
    def encoder_layers(self) -> list[nn.Module]:
        return self._encoder_layers

    @property
    def decoder_layers(self) -> list[nn.Module]:
        return self._decoder_layers

    @property
    def middle_layers(self) -> list[nn.Module]:
        return self._middle_layers

    def release_middle_layers(self) -> None:
        """Drop references after the weight transplant has consumed them."""

        self._middle_layers = []

    @property
    def embed_tokens(self) -> nn.Module:
        return self.base.embed_tokens

    @property
    def final_norm(self) -> nn.Module:
        return self.base.norm

    @property
    def lm_head(self) -> nn.Module:
        return self.model.lm_head

    @staticmethod
    def _layer_signature(layer: nn.Module) -> dict[str, bool]:
        parameters = inspect.signature(layer.forward).parameters
        return {
            "position_embeddings": "position_embeddings" in parameters,
            "position_ids": "position_ids" in parameters,
        }

    @classmethod
    def from_pretrained(cls, config: BackboneConfig) -> "HFBackbone":
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:  # pragma: no cover - exercised on CI
            raise ImportError(
                "loading kind='hf' requires the optional 'transformers' "
                "dependency; install `pip install \"transformers>=4.51\" safetensors`"
            ) from error

        dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[config.dtype]
        kwargs: dict[str, Any] = {
            "local_files_only": config.local_files_only,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.revision:
            kwargs["revision"] = config.revision
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id, dtype=dtype, **kwargs
            )
        except TypeError:  # transformers < 4.56 calls the argument torch_dtype
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id, torch_dtype=dtype, **kwargs
            )
        return cls(model, config)

    def _build_context(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> BackboneContext:
        position_ids = torch.arange(
            hidden.shape[1], device=hidden.device
        ).unsqueeze(0)
        try:
            from transformers.masking_utils import (  # type: ignore[import-not-found]
                create_causal_mask,
            )
        except ImportError:  # transformers 4.x
            from transformers.modeling_attn_mask_utils import (  # type: ignore[import-not-found]
                create_causal_mask,
            )

            mask = create_causal_mask(
                attention_mask=attention_mask,
                input_shape=(hidden.shape[0], hidden.shape[1]),
                inputs_embeds=hidden,
                past_key_values_length=0,
            )
            return BackboneContext(mask, None, position_ids)

        mask_kwargs = {
            "config": self.model.config,
            "inputs_embeds": hidden,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        full_mask = create_causal_mask(**mask_kwargs)
        mask: Any = full_mask
        if self._has_sliding:
            try:
                from transformers.masking_utils import (
                    create_sliding_window_causal_mask,
                )
            except ImportError:  # pragma: no cover - version dependent
                create_sliding_window_causal_mask = None
            if create_sliding_window_causal_mask is not None:
                mask = {
                    "full_attention": full_mask,
                    "sliding_attention": create_sliding_window_causal_mask(
                        **mask_kwargs
                    ),
                }

        position_embeddings = None
        rotary = getattr(self.base, "rotary_emb", None)
        if rotary is not None:
            position_embeddings = rotary(hidden, position_ids)
        else:
            raise NotImplementedError(
                "HFBackbone currently requires a rotary-embedding backbone "
                "(Qwen/Llama-style). Learned absolute positions need a separate "
                "split adapter."
            )
        return BackboneContext(mask, position_embeddings, position_ids)

    def _run_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        context: BackboneContext,
        layer_type: str,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {}
        if context.attention_mask is not None:
            if isinstance(context.attention_mask, dict):
                kwargs["attention_mask"] = context.attention_mask[layer_type]
            else:
                kwargs["attention_mask"] = context.attention_mask
        if self._passes["position_ids"]:
            kwargs["position_ids"] = context.position_ids
        if (
            self._passes["position_embeddings"]
            and context.position_embeddings is not None
        ):
            kwargs["position_embeddings"] = context.position_embeddings
        if (
            self.gradient_checkpointing
            and self.training
            and hidden.requires_grad
        ):
            return checkpoint(layer, hidden, use_reentrant=False, **kwargs)
        return layer(hidden, **kwargs)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, BackboneContext]:
        hidden = self.embed_tokens(input_ids)
        context = self._build_context(hidden, attention_mask)
        for index, layer in enumerate(self.encoder_layers):
            hidden = self._run_layer(
                layer, hidden, context, self._encoder_types[index]
            )
        return hidden, context

    def decode(
        self,
        hidden: torch.Tensor,
        context: BackboneContext,
    ) -> torch.Tensor:
        for index, layer in enumerate(self.decoder_layers):
            hidden = self._run_layer(
                layer, hidden, context, self._decoder_types[index]
            )
        return hidden

    def decode_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode arbitrary hidden states (open-loop stream, no token ids)."""

        context = self._build_context(hidden, attention_mask)
        return self.decode(hidden, context)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(hidden))

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.embed_tokens.parameters()
        for layer in self.encoder_layers:
            yield from layer.parameters()

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        for layer in self.decoder_layers:
            yield from layer.parameters()
        yield from self.final_norm.parameters()


def build_backbone(
    config: BackboneConfig,
    vocab_size: int,
    backbone: nn.Module | None = None,
) -> nn.Module:
    """Build, freeze and (optionally) LoRA-wrap one of the two backbones.

    ``backbone`` is an injection point for tests: pass an already-split
    backbone to exercise the freeze/LoRA policy without a network download.
    """

    if config.lora_rank and not (config.freeze_encoder and config.freeze_decoder):
        raise ValueError(
            "LoRA keeps the base weights frozen; set freeze_encoder and "
            "freeze_decoder to true, or set lora_rank to 0 for full tuning"
        )

    if backbone is None:
        if config.kind == "hf":
            backbone = HFBackbone.from_pretrained(config)
        else:
            tiny_vocab = config.tiny_vocab_size or vocab_size
            if tiny_vocab < vocab_size:
                raise ValueError(
                    f"tiny_vocab_size ({tiny_vocab}) is smaller than the tokenizer "
                    f"vocabulary ({vocab_size})"
                )
            backbone = TinyBackbone(config, tiny_vocab)
    elif config.kind == "tiny" and not isinstance(backbone, TinyBackbone):
        raise TypeError("kind='tiny' expects a TinyBackbone instance")
    elif config.kind == "hf" and not isinstance(backbone, HFBackbone):
        raise TypeError("kind='hf' expects an HFBackbone instance")

    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    if not config.freeze_encoder:
        for parameter in backbone.encoder_parameters():
            parameter.requires_grad_(True)
    if not config.freeze_decoder:
        for parameter in backbone.decoder_parameters():
            parameter.requires_grad_(True)

    if config.lora_rank:
        replaced = 0
        for layer in backbone.decoder_layers:
            replaced += inject_lora(
                layer,
                config.lora_targets,
                config.lora_rank,
                config.lora_alpha,
                config.lora_dropout,
            )
        if replaced == 0:
            raise ValueError(
                "no linear modules matched backbone.lora_targets="
                f"{list(config.lora_targets)}"
            )

    return backbone
