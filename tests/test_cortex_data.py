from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from neurosim_gpu.cortex.config import DataConfig
from neurosim_gpu.cortex.data import (
    PackedBlocks,
    build_datasets,
    collect_text_files,
    pack_blocks,
    read_documents,
    split_documents,
)
from neurosim_gpu.cortex.tokenization import ByteTokenizer


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ByteTokenizer()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for index in range(6):
            text = (
                f"document {index}\n"
                + "the quick brown fox jumps over the lazy dog. " * 8
            )
            (self.root / f"doc{index}.txt").write_text(text, encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self, **overrides) -> DataConfig:
        values = {
            "corpus": (str(self.root),),
            "tokenizer": "byte",
            "block_size": 16,
            "batch_size": 2,
            "eval_batch_size": 1,
            "cache_dir": str(self.root / "cache"),
            "train_split": 0.8,
            "extensions": (".txt",),
        }
        values.update(overrides)
        return DataConfig(**values)

    def test_collect_and_read(self) -> None:
        files = collect_text_files((str(self.root),), (".txt",))
        self.assertEqual(len(files), 6)
        documents = read_documents(files)
        self.assertEqual(len(documents), 6)
        with self.assertRaises(FileNotFoundError):
            collect_text_files((str(self.root / "missing"),), (".txt",))

    def test_split_single_document(self) -> None:
        train, val = split_documents(["a" * 100], 0.8, seed=0)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        self.assertGreater(len(train[0]), len(val[0]))
        self.assertEqual(train[0] + val[0], "a" * 100)

    def test_pack_and_shift(self) -> None:
        ids = torch.arange(10, dtype=torch.int64)
        blocks = pack_blocks(ids, 3)
        self.assertEqual(tuple(blocks.shape), (2, 4))
        dataset = PackedBlocks(blocks)
        sample = dataset[0]
        torch.testing.assert_close(sample["input_ids"], blocks[0, :-1])
        torch.testing.assert_close(sample["labels"], blocks[0, 1:])
        self.assertEqual(sample["input_ids"].dtype, torch.int64)

    def test_build_datasets_and_cache(self) -> None:
        config = self.config()
        train, val, meta = build_datasets(config, self.tokenizer)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertEqual(meta.n_documents, 6)
        self.assertEqual(train.blocks.shape[1], config.block_size + 1)
        self.assertTrue(
            bool((train.blocks >= 0).all() and (train.blocks < 257).all())
        )
        cache_files = list((self.root / "cache").glob("*.pt"))
        self.assertEqual(len(cache_files), 1)

        train_again, val_again, meta_again = build_datasets(
            config, self.tokenizer
        )
        torch.testing.assert_close(train_again.blocks, train.blocks)
        torch.testing.assert_close(val_again.blocks, val.blocks)
        self.assertEqual(meta, meta_again)

    def test_max_tokens_cap(self) -> None:
        config = self.config(max_tokens=64 * 2)
        train, val, meta = build_datasets(config, self.tokenizer)
        self.assertLessEqual(meta.n_train_tokens, 128)
        self.assertGreater(len(train), 0)

    def test_jsonl_text_field_and_caps(self) -> None:
        jsonl = self.root / "corpus.jsonl"
        with jsonl.open("w", encoding="utf-8") as handle:
            for index in range(12):
                handle.write(
                    json.dumps(
                        {"text": f"record {index} " + "hello world. " * 20},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        config = self.config(
            corpus=(str(jsonl),),
            extensions=(".jsonl",),
            block_size=8,
            max_documents=6,
            max_bytes=200_000,
        )
        train, val, meta = build_datasets(config, self.tokenizer)
        self.assertLessEqual(meta.n_documents, 6)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)

    def test_empty_corpus_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "data.corpus is empty"):
            collect_text_files((), (".txt",))

    def test_small_corpus_is_rejected(self) -> None:
        small = self.root / "small"
        small.mkdir()
        (small / "one.txt").write_text("short", encoding="utf-8")
        config = self.config(corpus=(str(small),), block_size=64)
        with self.assertRaisesRegex(ValueError, "too small"):
            build_datasets(config, self.tokenizer)


if __name__ == "__main__":
    unittest.main()
