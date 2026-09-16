#!/usr/bin/env python3
"""Auto-update the Table of Contents in README.md.

Finds the <!-- TOC --> ... <!-- /TOC --> block and rewrites it with the
current markdown headings. Designed to run as a pre-commit hook.

Usage:
    python scripts/update_readme_toc.py [--path README.md]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TOC_START = "<!-- TOC -->"
TOC_END = "<!-- /TOC -->"


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks so headings inside them are ignored."""
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def parse_headings(text: str) -> list[tuple[int, str, str]]:
    """Return [(level, slug, title), ...] for each markdown heading."""
    headings: list[tuple[int, str, str]] = []
    for line in _strip_code_blocks(text).splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        slug = (
            title.lower()
            .strip()
            .replace(" ", "-")
            .replace("/", "-")
            .replace(":", "")
            .replace("?", "")
            .replace("#", "")
            .replace("`", "")
            .replace("(", "")
            .replace(")", "")
        )
        slug = re.sub(r"[^a-z0-9-]", "", slug)
        slug = re.sub(r"-{2,}", "-", slug).strip("-")
        headings.append((level, slug, title))
    return headings


def render_toc(headings: list[tuple[int, str, str]]) -> str:
    lines = ["## Table of Contents", ""]
    for level, slug, title in headings:
        indent = "  " * (level - 1)
        lines.append(f"{indent}- [{title}](#{slug})")
    return "\n".join(lines)


def update_toc_in_readme(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    headings = parse_headings(text)
    new_toc = render_toc(headings)

    pattern = re.compile(
        re.escape(TOC_START) + r".*?" + re.escape(TOC_END),
        re.DOTALL,
    )
    replacement = f"{TOC_START}\n\n{new_toc}\n\n{TOC_END}"

    if pattern.search(text):
        new_text = pattern.sub(replacement, text)
    else:
        # Insert after the first H1 if no TOC block exists
        new_text = text.replace(
            text.splitlines()[0],
            text.splitlines()[0] + f"\n\n{replacement}\n",
            1,
        )

    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Update README.md Table of Contents")
    parser.add_argument("--path", default="README.md", help="Path to README file")
    args = parser.parse_args()

    readme = Path(args.path)
    if not readme.exists():
        print(f"error: {readme} not found", file=sys.stderr)
        return 1

    changed = update_toc_in_readme(readme)
    if changed:
        print(f"updated: {readme}")
        return 0
    print(f"unchanged: {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
