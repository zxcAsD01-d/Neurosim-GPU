#!/usr/bin/env python3
"""Concatenate the repository's Markdown into a small self-contained corpus.

This exists so the Cortex-LM pipeline can be exercised end to end without
downloading a dataset. It is *not* a training corpus: the resulting file is a
few hundred kilobytes of project documentation, useful for smoke runs and
shape checks only.

Usage:
    python llm/scripts/build_cortex_corpus.py
    python llm/scripts/build_cortex_corpus.py --output data/cortex_corpus/repo_docs.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

DOCUMENTS = [
    "README.md",
    "README.en.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "demo/README.md",
]


def collect(root: Path) -> list[Path]:
    files = [root / name for name in DOCUMENTS]
    files.extend(sorted((root / "docs").glob("*.md")))
    files.extend(sorted((root / "demo" / "docs").glob("*.md")))
    return [path for path in files if path.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument(
        "--output",
        default="data/cortex_corpus/repo_docs.txt",
        help="output text path (gitignored under data/)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = collect(root)
    if not files:
        raise SystemExit(f"no Markdown files found under {root}")

    chunks: list[str] = []
    for path in files:
        relative = path.relative_to(root)
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            chunks.append(f"# source: {relative}\n\n{text}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    total = sum(len(chunk) for chunk in chunks)
    print(
        f"wrote {output} from {len(files)} files, "
        f"{total:,} characters"
    )
    print(
        "note: this is project documentation for pipeline smoke tests, "
        "not a language-model training corpus"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
