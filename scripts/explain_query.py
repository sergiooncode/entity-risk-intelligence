#!/usr/bin/env python
"""
Interactive poking tool: embed a query string, build one of the three
strategies, print the SQL and the EXPLAIN plan.

  # unfiltered ANN - should show documents_embedding_hnsw
  python scripts/explain_query.py "tanker with an AIS gap off Fujairah"

  # same query, pre-filtered - watch the plan flip as the filter tightens
  python scripts/explain_query.py "..." --filter "jurisdiction = 'AE'"
  python scripts/explain_query.py "..." --filter "jurisdiction = 'AE' AND topic = 'maritime'"

  # the other two strategies
  python scripts/explain_query.py "..." --strategy post_filter --overfetch 50
  python scripts/explain_query.py "..." --strategy exact

  # print SQL with the vector inlined, to paste into psql yourself
  python scripts/explain_query.py "..." --emit-sql
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vsbench import db  # noqa: E402
from vsbench import strategies as S  # noqa: E402
from vsbench.embedder import MiniLMEmbedder, embed_cached  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--filter", default=None, help="SQL WHERE fragment")
    ap.add_argument("--strategy", default="pre_filter",
                    choices=["pre_filter", "post_filter", "exact"])
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--overfetch", type=int, default=10)
    ap.add_argument("--ef-search", type=int, default=40)
    ap.add_argument("--iterative-scan", default="off",
                    choices=["off", "relaxed_order", "strict_order"])
    ap.add_argument("--emit-sql", action="store_true",
                    help="print SQL with the vector literal inlined, then exit")
    ap.add_argument("--rows", action="store_true", help="also show the results")
    ap.add_argument("--dsn", default=db.DSN)
    args = ap.parse_args()

    embedder = MiniLMEmbedder()
    qvec = embed_cached(embedder, [args.query], ROOT / "cache",
                        tag="adhoc")[0]

    if args.strategy == "pre_filter":
        plan = S.pre_filter(qvec, args.filter, k=args.k,
                            ef_search=args.ef_search,
                            iterative_scan=args.iterative_scan)
    elif args.strategy == "post_filter":
        plan = S.post_filter(qvec, args.filter, k=args.k,
                             overfetch=args.overfetch,
                             ef_search=args.ef_search)
    else:
        plan = S.exact(qvec, args.filter, k=args.k)

    if args.emit_sql:
        literal = "'[" + ",".join(f"{x:.6f}" for x in qvec) + "]'::vector"
        print("-- paste into psql:")
        for key, value in plan.gucs.items():
            print(f"SET {key} = '{value}';")
        print(plan.sql.replace("%(q)s", literal) + ";")
        return 0

    conn = db.connect(args.dsn)
    print(f"query    : {args.query!r}")
    print(f"strategy : {plan.label}")
    print(f"filter   : {args.filter or '(none)'}")
    print(f"settings : {plan.gucs}")
    print(f"\n--- SQL ---\n{plan.sql};\n")

    plan_text = S.explain(conn, plan)
    print(f"--- EXPLAIN (ANALYZE, BUFFERS) ---\n{plan_text}\n")
    print(f"scan kind: {S.scan_kind(plan_text)}")

    if args.rows:
        res = S.run(conn, plan)
        print(f"\n--- {len(res.doc_ids)} rows in {res.elapsed_ms:.2f} ms ---")
        for did, dist in zip(res.doc_ids, res.distances):
            print(f"  doc_id={did:<8} distance={dist:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
