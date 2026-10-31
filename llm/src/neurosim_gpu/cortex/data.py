"""Corpus loading, tokenisation, packing and dataloaders.

Training samples are fixed-length blocks of a token stream with an end-of-text
token between documents. Every block starts from a fresh brain state; within a
block the state is recurrent and gradients are truncated at
``brain.detach_interval`` tokens. The cache stores the packed blocks, not the
raw text, so a second run does not re-tokenise a corpus.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig
from .tokenization import Tokenizer


@dataclass(frozen=True)
class CorpusMeta:
    n_documents: int
    n_train_tokens: int
    n_val_tokens: int
    n_train_blocks: int
    n_val_blocks: int
    block_size: int
    files: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["files"] = list(self.files)
        return payload


def collect_text_files(paths: tuple[str, ...], extensions: tuple[str, ...]) -> list[Path]:
    """Expand files and directories into a sorted, unique list."""

    if not paths:
        raise ValueError(
            "data.corpus is empty; pass --corpus or set data.corpus to a file "
            "or directory"
        )
    found: dict[str, Path] = {}
    suffixes = {extension.lower() for extension in extensions}
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"corpus path does not exist: {path}")
        if path.is_dir():
            candidates = [
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in suffixes
            ]
        else:
            candidates = [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            found[str(resolved)] = resolved
    return [found[key] for key in sorted(found)]


def _read_text_prefix(path: Path, max_bytes: int) -> str:
    if max_bytes <= 0:
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        with path.open("rb") as handle:
            text = handle.read(max_bytes).decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").strip()


def _read_jsonl_documents(
    path: Path,
    *,
    text_field: str,
    max_bytes: int,
    budget: int,
    limit: int,
) -> tuple[list[str], int]:
    """Read documents from a JSONL file without loading the whole file.

    ``budget`` is the remaining byte budget (0 means unlimited); returned bytes
    are counted against it. ``limit`` is the remaining document count.
    """

    documents: list[str] = []
    consumed = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            if max_bytes and consumed + len(raw_line) > budget:
                break
            if limit and len(documents) >= limit:
                break
            consumed += len(raw_line)
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get(text_field) if isinstance(record, dict) else None
            if isinstance(text, str) and text.strip():
                documents.append(text.strip())
    return documents, consumed


def read_documents(
    files: list[Path],
    *,
    text_field: str = "text",
    max_documents: int = 0,
    max_bytes: int = 0,
) -> list[str]:
    """Read text or JSONL documents, respecting optional byte/doc caps."""

    documents: list[str] = []
    consumed = 0
    for path in files:
        remaining_docs = (
            max_documents - len(documents) if max_documents else 0
        )
        if max_documents and remaining_docs <= 0:
            break
        remaining_bytes = (
            max_bytes - consumed if max_bytes else 0
        )
        if max_bytes and remaining_bytes <= 0:
            break

        if path.suffix.lower() == ".jsonl":
            found, used = _read_jsonl_documents(
                path,
                text_field=text_field,
                max_bytes=max_bytes,
                budget=remaining_bytes,
                limit=remaining_docs,
            )
            documents.extend(found)
            consumed += used
        else:
            text = _read_text_prefix(path, remaining_bytes)
            consumed += len(text.encode("utf-8"))
            if text:
                documents.append(text)
                if max_documents and len(documents) >= max_documents:
                    break

    if not documents:
        raise ValueError("the corpus contains no non-empty text documents")
    return documents


def split_documents(
    documents: list[str],
    train_split: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split by document, falling back to a character split for one document."""

    if len(documents) == 1:
        text = documents[0]
        cut = int(len(text) * train_split)
        cut = min(max(cut, 1), len(text) - 1)
        return [text[:cut]], [text[cut:]]

    order = list(range(len(documents)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(round(len(documents) * (1.0 - train_split))))
    n_val = min(n_val, len(documents) - 1)
    val_indices = set(order[:n_val])
    train = [doc for index, doc in enumerate(documents) if index not in val_indices]
    val = [doc for index, doc in enumerate(documents) if index in val_indices]
    return train, val


def _encode_documents(
    tokenizer: Tokenizer,
    documents: list[str],
    *,
    batch_size: int = 64,
    add_bos: bool = False,
) -> torch.Tensor:
    eos = tokenizer.eos_token_id
    bos = getattr(tokenizer, "bos_token_id", eos)
    ids: list[int] = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        if hasattr(tokenizer, "encode_batch"):
            encoded = tokenizer.encode_batch(batch)
        else:
            encoded = [tokenizer.encode(text) for text in batch]
        for sequence in encoded:
            if add_bos:
                ids.append(int(bos))
            ids.extend(int(token) for token in sequence)
            ids.append(eos)
    # int32 halves the packing/cache footprint; vocabularies stay below 2**31.
    return torch.tensor(ids, dtype=torch.int32)


def pack_blocks(token_ids: torch.Tensor, block_size: int) -> torch.Tensor:
    """Reshape a token stream into ``[blocks, block_size + 1]`` windows."""

    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    width = block_size + 1
    n_blocks = token_ids.numel() // width
    if n_blocks == 0:
        return torch.empty((0, width), dtype=torch.int64)
    return token_ids[: n_blocks * width].view(n_blocks, width).contiguous()


class PackedBlocks(Dataset):
    """A block yields inputs and the same stream shifted by one token."""

    def __init__(self, blocks: torch.Tensor) -> None:
        if blocks.ndim != 2:
            raise ValueError("blocks must be two-dimensional")
        self.blocks = blocks

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        block = self.blocks[index]
        return {
            "input_ids": block[:-1].long(),
            "labels": block[1:].long(),
        }


def _cache_signature(
    files: list[Path],
    tokenizer: Tokenizer,
    config: DataConfig,
) -> tuple[str, dict[str, object]]:
    payload = {
        "files": [
            [str(path), path.stat().st_size, path.stat().st_mtime_ns]
            for path in files
        ],
        "tokenizer": tokenizer.name,
        "vocab_size": int(tokenizer.vocab_size),
        "block_size": config.block_size,
        "train_split": config.train_split,
        "add_bos": config.add_bos,
        "max_tokens": config.max_tokens,
        "max_bytes": config.max_bytes,
        "max_documents": config.max_documents,
        "text_field": config.text_field,
        "seed": config.seed,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    return digest, payload


def build_datasets(
    config: DataConfig,
    tokenizer: Tokenizer,
    *,
    verbose: bool = False,
) -> tuple[PackedBlocks, PackedBlocks, CorpusMeta]:
    """Load, tokenise and pack the corpus described by ``config``."""

    files = collect_text_files(config.corpus, config.extensions)
    digest, payload = _cache_signature(files, tokenizer, config)
    cache_path = Path(config.cache_dir) / f"blocks-{digest}.pt"

    if cache_path.exists():
        try:
            cached = torch.load(
                cache_path, map_location="cpu", weights_only=True
            )
            if cached.get("cache_version") == 1:
                meta = CorpusMeta(**cached["meta"])
                if verbose:
                    print(f"corpus cache hit: {cache_path}")
                return (
                    PackedBlocks(cached["train"]),
                    PackedBlocks(cached["val"]),
                    meta,
                )
        except Exception:
            if verbose:
                print(f"corpus cache unreadable, rebuilding: {cache_path}")

    documents = read_documents(
        files,
        text_field=config.text_field,
        max_documents=config.max_documents,
        max_bytes=config.max_bytes,
    )
    train_docs, val_docs = split_documents(
        documents, config.train_split, config.seed
    )
    train_ids = _encode_documents(
        tokenizer, train_docs, add_bos=config.add_bos
    )
    val_ids = _encode_documents(
        tokenizer, val_docs, add_bos=config.add_bos
    )

    if config.max_tokens:
        train_ids = train_ids[: config.max_tokens]
        val_cap = max(
            config.block_size + 1,
            int(config.max_tokens * (1.0 - config.train_split) / config.train_split),
        )
        val_ids = val_ids[:val_cap]

    train_blocks = pack_blocks(train_ids, config.block_size)
    val_blocks = pack_blocks(val_ids, config.block_size)
    if len(train_blocks) == 0 or len(val_blocks) == 0:
        raise ValueError(
            "corpus is too small for the requested block size: "
            f"{len(train_blocks)} train blocks, {len(val_blocks)} val blocks; "
            "add text or lower data.block_size"
        )

    meta = CorpusMeta(
        n_documents=len(documents),
        n_train_tokens=int(train_ids.numel()),
        n_val_tokens=int(val_ids.numel()),
        n_train_blocks=len(train_blocks),
        n_val_blocks=len(val_blocks),
        block_size=config.block_size,
        files=tuple(str(path) for path in files),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_version": 1,
            "signature": digest,
            "payload": payload,
            "train": train_blocks,
            "val": val_blocks,
            "meta": meta.to_dict(),
        },
        cache_path,
    )
    if verbose:
        print(
            f"corpus: {meta.n_documents} docs, "
            f"{meta.n_train_tokens:,} train tokens, "
            f"{meta.n_val_tokens:,} val tokens, "
            f"{meta.n_train_blocks:,} train blocks"
        )
    return PackedBlocks(train_blocks), PackedBlocks(val_blocks), meta


def build_dataloaders(
    config: DataConfig,
    tokenizer: Tokenizer,
    *,
    verbose: bool = False,
) -> tuple[DataLoader, DataLoader, CorpusMeta]:
    train_set, val_set, meta = build_datasets(
        config, tokenizer, verbose=verbose
    )
    pin_memory = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader, meta
