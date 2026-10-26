#!/usr/bin/env python3
"""Check that relative Markdown links and image paths resolve.

Usage:
    python scripts/check_links.py [--root .]

Exits non-zero when a local link target is missing, so CI catches documentation
rot. External URLs are not fetched (that would make the check flaky); only the
existence of repository-local targets is verified.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "data", "node_modules", "__pycache__", "build", "dist"}
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCE = re.compile(r"```.*?```", re.DOTALL)


def markdown_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return sorted(files)


def strip_code(text: str) -> str:
    return FENCE.sub("", text)


def check(root: Path) -> list[str]:
    problems: list[str] = []
    for path in markdown_files(root):
        text = strip_code(path.read_text(encoding="utf-8"))
        for match in LINK.finditer(text):
            target = match.group(1).strip()
            if not target:
                continue
            if target.startswith(("http://", "https://", "mailto:", "#", "data:")):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                relative = path.relative_to(root)
                problems.append(f"{relative}: missing '{target}'")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    problems = check(root)
    checked = len(markdown_files(root))
    if problems:
        print(f"{len(problems)} broken link(s) across {checked} Markdown files:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"all local links resolve ({checked} Markdown files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
