#!/usr/bin/env python
"""
Plain-English search over the corpus. No SQL required.

  python scripts/search.py "oil tanker going dark near Fujairah"
  python scripts/search.py "shell company in a free zone" --jurisdiction AE --since 2025
  python scripts/search.py "drone components" --topic military_end_use --compare

Omit the query for an interactive prompt (model loads once).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vsbench import db  # noqa: E402
from vsbench import strategies as S  # noqa: E402
from vsbench.embedder import MiniLMEmbedder  # noqa: E402

FIELDS = ("doc_id, entity_name, jurisdiction, topic, published, "
          "is_designated, left(text, 150) AS snippet")


def build_filter(args) -> str | None:
    """Turn the plain flags into a readable SQL WHERE fragment."""
    parts = []
    if args.jurisdiction:
        js = ", ".join(f"'{j.upper()}'" for j in args.jurisdiction)
        parts.append(f"jurisdiction IN ({js})")
    if args.topic:
        ts = ", ".join(f"'{t}'" for t in args.topic)
        parts.append(f"topic IN ({ts})")
    if args.since:
        parts.append(f"published >= DATE '{args.since}-01-01'")
    if args.designated:
        parts.append("is_designated")
    return " AND ".join(parts) if parts else None


def fetch(conn, plan, k):
    """Run the plan, then pull readable columns for the doc_ids it returned."""
    res = S.run(conn, plan)
    if not res.doc_ids:
        return [], res
    order = {d: i for i, d in enumerate(res.doc_ids)}
    rows = conn.execute(
        f"SELECT {FIELDS} FROM documents WHERE doc_id = ANY(%s)",
        (res.doc_ids,)).fetchall()
    rows.sort(key=lambda r: order[r[0]])
    return rows, res


def show(rows, res, k, label):
    print(f"\n=== {label} — {len(rows)} of {k} results in {res.elapsed_ms:.1f} ms ===")
    if len(rows) < k:
        print(f"  !! returned fewer than {k}: the filter starved the ANN walk.")
    for i, (did, name, juris, topic, pub, desig, snip) in enumerate(rows, 1):
        flag = " [DESIGNATED]" if desig else ""
        print(f"\n{i:2}. {res.distances[i-1]:.3f}  {juris}  {topic:<18} {pub}  "
              f"doc {did}{flag}\n    {name}\n    {snip.strip()}...")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--jurisdiction", "-j", nargs="+")
    ap.add_argument("--topic", "-t", nargs="+")
    ap.add_argument("--since", type=int, help="year, e.g. 2025")
    ap.add_argument("--designated", action="store_true")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--strategy", default="pre_filter",
                    choices=["pre_filter", "post_filter", "exact"])
    ap.add_argument("--ef-search", type=int, default=40)
    ap.add_argument("--iterative", action="store_true",
                    help="enable hnsw.iterative_scan=relaxed_order")
    ap.add_argument("--compare", action="store_true",
                    help="run all three strategies side by side")
    ap.add_argument("--dsn", default=db.DSN)
    args = ap.parse_args()

    fsql = build_filter(args)
    embedder = MiniLMEmbedder()
    conn = db.connect(args.dsn)

    if fsql:
        n = conn.execute(f"SELECT count(*) FROM documents WHERE {fsql}").fetchone()[0]
        print(f"filter: {fsql}\n        matches {n:,} of 200,000 rows "
              f"({100*n/200000:.3g}%)")
    else:
        print("filter: (none)")

    def run_one(text: str) -> None:
        qvec = embedder.encode([text])[0]
        if args.compare:
            plans = [
                ("pre_filter (naive)", S.pre_filter(qvec, fsql, k=args.k,
                                                    ef_search=args.ef_search)),
                ("pre_filter + iterative", S.pre_filter(qvec, fsql, k=args.k,
                                                        ef_search=args.ef_search,
                                                        iterative_scan="relaxed_order")),
                ("exact (ground truth)", S.exact(qvec, fsql, k=args.k)),
            ]
        else:
            plan = (S.exact(qvec, fsql, k=args.k) if args.strategy == "exact"
                    else S.post_filter(qvec, fsql, k=args.k,
                                       ef_search=args.ef_search)
                    if args.strategy == "post_filter"
                    else S.pre_filter(qvec, fsql, k=args.k,
                                      ef_search=args.ef_search,
                                      iterative_scan=("relaxed_order"
                                                      if args.iterative else "off")))
            plans = [(args.strategy, plan)]

        for label, plan in plans:
            rows, res = fetch(conn, plan, args.k)
            show(rows, res, args.k, label)

    if args.query:
        run_one(args.query)
        return 0

    print("\nType a query, or Ctrl-D to quit.")
    while True:
        try:
            text = input("\nsearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text:
            run_one(text)


if __name__ == "__main__":
    raise SystemExit(main())
