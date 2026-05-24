#!/usr/bin/env python3
"""Merge corpora from multiple folders into one file in the target folder."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PART_RE = re.compile(
    r"^(?P<prefix>.+)_mode_(?P<mode>finetune|normal_finetune)_(?P<part>\d+)_(?P<tail>finetune_[^/]+?_final\.json)$"
)


def _load_list(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def _pick_latest_per_part(files):
    latest = {}
    for f in files:
        m = PART_RE.match(f.name)
        if not m:
            continue
        part = int(m.group("part"))
        if part not in latest or f.stat().st_mtime > latest[part].stat().st_mtime:
            latest[part] = f
    return latest


def _expand_patterns(pattern: str) -> list[str]:
    pattern = pattern.strip()
    if pattern in {"final", "all"}:
        return ["*_final.json"]
    return [pattern]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge corpora from source dirs.")
    parser.add_argument("--sources", nargs="+", required=True, help="Source directories")
    parser.add_argument("--target", default=None, help="Target directory for merged output")
    parser.add_argument(
        "--pattern",
        default="final",
        help="Pattern: final or a glob (default: final)",
    )
    parser.add_argument(
        "--parts",
        type=int,
        required=True,
        help="Merge by part index (1..parts) for _mode_finetune_*.json files.",
    )
    args = parser.parse_args()

    sources = [Path(p) for p in args.sources]
    target = Path(args.target) if args.target else sources[-1]
    target.mkdir(parents=True, exist_ok=True)

    patterns = _expand_patterns(args.pattern)

    for pat in patterns:
        files = []
        for s in sources:
            files.extend(sorted(s.glob(pat)))

        if not files:
            raise SystemExit(f"No matching files found for pattern {pat}.")

        groups = {}
        for f in files:
            m = PART_RE.match(f.name)
            if not m:
                continue
            key = (m.group("prefix"), m.group("mode"), m.group("tail"))
            groups.setdefault(key, []).append(f)

        if not groups:
            raise SystemExit(f"No _mode_finetune_ part files found for pattern {pat}.")

        for (prefix, mode, tail), group_files in groups.items():
            latest = _pick_latest_per_part(group_files)
            missing = [p for p in range(1, args.parts + 1) if p not in latest]
            if missing:
                raise SystemExit(f"Missing parts {missing} for {prefix}_{tail}.")

            merged = []
            for p in range(1, args.parts + 1):
                merged.extend(_load_list(latest[p]))

            out_name = f"{prefix}_mode_{mode}_merge_{tail}"
            out_path = target / out_name
            out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {out_path} ({len(merged)} records)")


if __name__ == "__main__":
    main()
