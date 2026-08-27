#!/usr/bin/env python
"""
Rebuild RESULTS.md from results/ without re-running the benchmark.

Useful after editing vsbench/findings.py: the numbers are already on disk, only
the prose changes.

  python scripts/make_report.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vsbench import report  # noqa: E402
from vsbench.filters import Filter  # noqa: E402
from vsbench.strategies import elide_vectors  # noqa: E402


def safe_name(label: str) -> str:
    """Must match the sanitisation used when the plans were written."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in label)


class PlansOnDisk(dict):
    """
    Lazily reads results/plans/<sanitised>.txt.

    The sanitisation is lossy, but lookups only ever go forward (key -> file),
    so there is nothing to invert.
    """

    def __init__(self, plans_dir: Path):
        super().__init__()
        self.dir = plans_dir

    def get(self, key, default=None):
        path = self.dir / f"{safe_name(key)}.txt"
        if path.exists():
            # Applied on read as well as on capture, so plans written by an
            # older run are still readable in the report.
            return elide_vectors(path.read_text())
        return default

    def __contains__(self, key) -> bool:  # report.py uses `in`
        return (self.dir / f"{safe_name(key)}.txt").exists()

    def __getitem__(self, key):
        v = self.get(key)
        if v is None:
            raise KeyError(key)
        return v


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key, val in list(r.items()):
            if val in ("", "-"):
                continue
            try:
                r[key] = int(val) if val.lstrip("-").isdigit() else float(val)
            except ValueError:
                pass
    return rows


def main() -> int:
    results = ROOT / "results"
    meta = json.loads((results / "meta.json").read_text())
    ladder = [Filter(label=d["label"], sql=d["sql"], n_rows=d["n_rows"],
                     selectivity=d["selectivity"]) for d in meta["ladder"]]

    report.write(
        ROOT / "RESULTS.md",
        meta=meta,
        ladder=ladder,
        e1=read_csv(results / "e1_baseline.csv"),
        e2=read_csv(results / "e2_selectivity.csv"),
        e3=read_csv(results / "e3_overfetch.csv"),
        e4=read_csv(results / "e4_ef_search.csv"),
        e5=read_csv(results / "e5_iterative_scan.csv"),
        plans=PlansOnDisk(results / "plans"),
    )
    print("wrote RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
