#!/usr/bin/env python
"""
End-to-end: generate -> embed -> load -> index -> E1..E5 -> results/ -> RESULTS.md

  python scripts/run_all.py                 # full 200k run
  python scripts/run_all.py --n-docs 20000  # quick smoke run
  python scripts/run_all.py --rebuild       # force regenerate + reload
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vsbench import corpus, db, experiments, filters, queries, report  # noqa: E402
from vsbench import strategies as S  # noqa: E402
from vsbench.embedder import MiniLMEmbedder, embed_cached  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    for r in rows:  # tolerate ragged dicts
        for c in r:
            if c not in cols:
                cols.append(c)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"    wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=corpus.N_DOCS)
    ap.add_argument("--n-entities", type=int, default=corpus.N_ENTITIES)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--reps", type=int, default=experiments.REPS)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--dsn", default=db.DSN)
    args = ap.parse_args()

    results_dir = ROOT / "results"
    cache_dir = ROOT / "cache"
    meta: dict = {"n_docs": args.n_docs, "n_entities": args.n_entities,
                  "k": args.k, "reps": args.reps}

    print("[1/7] corpus")
    t0 = time.perf_counter()
    entities, docs = corpus.build(cache_dir, n_entities=args.n_entities,
                                  n_docs=args.n_docs, rebuild=args.rebuild)
    print(f"  {len(entities):,} entities, {len(docs):,} documents "
          f"({time.perf_counter() - t0:.1f}s)")
    meta["corpus_seconds"] = round(time.perf_counter() - t0, 1)

    print("[2/7] embeddings")
    embedder = MiniLMEmbedder()
    meta["embedder"] = embedder.name
    meta["embed_device"] = embedder.device
    t0 = time.perf_counter()
    doc_vecs = embed_cached(embedder, [d.text for d in docs], cache_dir,
                            tag=f"docs{args.n_docs}")
    meta["embed_seconds"] = round(time.perf_counter() - t0, 1)
    qry_vecs = embed_cached(embedder, queries.query_texts(), cache_dir,
                            tag="queries")

    print("[3/7] database")
    db.wait_for_db(args.dsn)
    conn = db.connect(args.dsn)
    meta["pgvector_version"] = db.pgvector_version(conn)
    meta["server_version"] = db.server_version(conn)
    print(f"  postgres {meta['server_version']}, pgvector {meta['pgvector_version']}")

    if args.rebuild or not db.table_is_loaded(conn, len(docs)):
        print("  creating schema + loading")
        db.create_schema(conn)
        t = db.ingest(conn, docs, doc_vecs)
        meta["ingest_seconds"] = round(t, 1)
        print(f"  copied {len(docs):,} rows in {t:.1f}s")
        print("  building indexes (HNSW m=16 ef_construction=64)...")
        t = db.create_indexes(conn)
        meta["index_build_seconds"] = round(t, 1)
        print(f"  indexes built in {t:.1f}s")
    else:
        print("  table already loaded and indexed, skipping")
    meta["sizes"] = db.index_sizes(conn)
    print(f"  sizes: {meta['sizes']}")

    print("[4/7] sanity check: is HNSW actually used unfiltered?")
    probe = S.pre_filter(qry_vecs[0], None, k=10)
    probe_plan = S.explain(conn, probe)
    kind = S.scan_kind(probe_plan)
    meta["baseline_scan"] = kind
    print(f"  scan kind = {kind}")
    if kind != "hnsw":
        print("\n!! Unfiltered ANN is NOT using the HNSW index. "
              "Everything downstream would be degenerate.\n")
        print(probe_plan)
        return 2

    print("[5/7] selectivity ladder")
    ladder = filters.build_ladder(conn, len(docs))
    for f in ladder:
        print(f"  {f.label:>7} -> {f.n_rows:>7,} rows "
              f"({f.selectivity * 100:.4g}%)  {f.sql}")
    meta["ladder"] = [{"label": f.label, "sql": f.sql, "n_rows": f.n_rows,
                       "selectivity": f.selectivity} for f in ladder]

    bench = experiments.Bench(conn, qry_vecs, k=args.k, reps=args.reps)
    all_plans: dict = {"E0 baseline probe": probe_plan}

    print("[6/7] experiments")
    t0 = time.perf_counter()

    print("  E1 baseline...")
    e1, p = experiments.e1_baseline(bench); all_plans.update(p)
    write_csv(results_dir / "e1_baseline.csv", e1)

    print("  E2 selectivity sweep...")
    e2, p = experiments.e2_selectivity(bench, ladder); all_plans.update(p)
    write_csv(results_dir / "e2_selectivity.csv", e2)

    # E3/E4 anchor on the ~1% and ~10% rungs of the ladder.
    by_label = {f.label: f for f in ladder}
    f1 = by_label.get("1%", ladder[2])
    f10 = by_label.get("10%", ladder[1])

    print("  E3 overfetch sweep...")
    e3, p = experiments.e3_overfetch(bench, f1); all_plans.update(p)
    write_csv(results_dir / "e3_overfetch.csv", e3)

    print("  E4 ef_search sweep...")
    e4, p = experiments.e4_ef_search(bench, [f10, f1]); all_plans.update(p)
    write_csv(results_dir / "e4_ef_search.csv", e4)

    print("  E5 iterative scan...")
    e5, p = experiments.e5_iterative(bench, ladder[2:],
                                     meta["pgvector_version"])
    all_plans.update(p)
    write_csv(results_dir / "e5_iterative_scan.csv", e5)

    meta["experiment_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"  experiments took {meta['experiment_seconds']}s")

    print("[7/7] report")
    plans_dir = results_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    for label, text in all_plans.items():
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        (plans_dir / f"{safe}.txt").write_text(text)
    (results_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    report.write(ROOT / "RESULTS.md", meta=meta, ladder=ladder,
                 e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, plans=all_plans)
    print(f"  wrote RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
