"""Tokenizers for the Cortex-LM pipeline.

The real configuration reuses the pretrained model's tokenizer. ``ByteTokenizer``
is a dependency-free stand-in so the pipeline and tests run offline; it is not
meant for a serious Chinese/English language-model run.
"""

from __future__ import annotations

from typing import Protocol

from .config import BackboneConfig, DataConfig


class Tokenizer(Protocol):
    name: str
    vocab_size: int
    eos_token_id: int
    pad_token_id: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


class ByteTokenizer:
    """UTF-8 bytes plus one end-of-text id. Vocabulary size is 257."""

    name = "byte"
    vocab_size = 257
    eos_token_id = 256
    bos_token_id = 256
    pad_token_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [
            list(text.encode("utf-8", errors="replace")) for text in texts
        ]

    def decode(self, token_ids: list[int]) -> str:
        data = bytes(int(token) for token in token_ids if 0 <= int(token) < 256)
        return data.decode("utf-8", errors="replace")


class HFTokenizer:
    """Thin adapter around ``transformers``' tokenizer API."""

    def __init__(self, tokenizer, name: str) -> None:
        self.tokenizer = tokenizer
        self.name = name
        self.vocab_size = len(tokenizer)
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError(f"tokenizer {name!r} has no eos_token_id")
        self.eos_token_id = int(eos)
        bos = tokenizer.bos_token_id
        self.bos_token_id = int(bos) if bos is not None else self.eos_token_id
        pad = tokenizer.pad_token_id
        self.pad_token_id = int(pad) if pad is not None else self.eos_token_id

    def encode(self, text: str) -> list[int]:
        return list(
            self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        )

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        return [list(ids) for ids in encoded]

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def _byte_encode_batch(texts: list[str]) -> list[list[int]]:
    return [list(text.encode("utf-8", errors="replace")) for text in texts]


def build_tokenizer(
    data: DataConfig,
    backbone: BackboneConfig,
) -> Tokenizer | HFTokenizer:
    """Build the tokenizer named by ``data.tokenizer``."""

    if data.tokenizer == "byte":
        return ByteTokenizer()

    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - CI path
        raise ImportError(
            "data.tokenizer='hf' requires the optional 'transformers' "
            "dependency; install `pip install \"transformers>=4.51\" safetensors`"
        ) from error

    tokenizer_id = backbone.tokenizer_id or backbone.model_id
    kwargs = {
        "local_files_only": backbone.local_files_only,
        "trust_remote_code": backbone.trust_remote_code,
    }
    if backbone.revision:
        kwargs["revision"] = backbone.revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
    return HFTokenizer(tokenizer, name=tokenizer_id)
