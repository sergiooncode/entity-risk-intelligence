"""
RESULTS.md generation.

Tables, SQL and EXPLAIN output are derived mechanically from the CSVs. The
prose findings live in findings.py and are written by hand after reading the
numbers - a generated paragraph would only restate the table, and the point of
the exercise is the interpretation.
"""
from __future__ import annotations

from pathlib import Path

try:
    from .findings import FINDINGS
except Exception:  # noqa: BLE001 - findings.py is written after the first run
    FINDINGS = {}

PENDING = "_(finding not yet written - run the benchmark, then fill in vsbench/findings.py)_"


def _md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    """cols is [(key, header)]."""
    head = "| " + " | ".join(h for _, h in cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key, "")
            if isinstance(v, float):
                v = f"{v:.4g}"
            cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule] + body)


def _sql_block(plans: dict, key: str) -> str:
    sql = plans.get(key)
    if not sql:
        return ""
    return f"```sql\n{sql.strip()};\n```\n"


def _plan_block(plans: dict, key: str, title: str | None = None) -> str:
    text = plans.get(key)
    if not text:
        return ""
    t = f"**{title or key}**\n\n" if title else ""
    return f"{t}```\n{text.strip()}\n```\n"


def _finding(tag: str) -> str:
    return FINDINGS.get(tag, PENDING).strip() + "\n"


def write(path: Path, *, meta: dict, ladder, e1, e2, e3, e4, e5, plans: dict) -> None:
    out: list[str] = []
    w = out.append

    w("# Metadata filtering and HNSW in pgvector\n")
    w("How pre-filtering, post-filtering and exact search behave as a metadata "
      "filter gets more selective, measured on a 200k-document synthetic "
      "sanctions/export-control corpus.\n")

    # ---------------- setup ----------------
    w("## Setup\n")
    w("| setting | value |\n|---|---|")
    w(f"| Corpus | {meta['n_docs']:,} documents, {meta['n_entities']:,} entities |")
    w(f"| Embedding | {meta['embedder']} (384-dim, cosine), device `{meta['embed_device']}` |")
    w(f"| Postgres | {meta['server_version']} |")
    w(f"| pgvector | {meta['pgvector_version']} |")
    w(f"| ANN index | HNSW, `m=16`, `ef_construction=64`, `vector_cosine_ops` |")
    w(f"| Scalar indexes | btree on `jurisdiction`, `topic`, `published` |")
    w(f"| k | {meta['k']} |")
    w(f"| Latency samples | 48 queries x {meta['reps']} reps, 2 warm-up passes |")
    if "index_build_seconds" in meta:
        w(f"| HNSW build time | {meta['index_build_seconds']:.0f}s |")
    if "embed_seconds" in meta:
        w(f"| Embedding time | {meta['embed_seconds']:.0f}s |")
    w("")
    if meta.get("sizes"):
        w("Object sizes:\n")
        w("| object | size |\n|---|---|")
        for name, size in sorted(meta["sizes"].items()):
            w(f"| `{name}` | {size} |")
        w("")

    w("Column meanings used throughout:\n")
    w("- `recall_mean` / `recall_p10` / `recall_min` - recall@k against exact "
      "search over the *filtered* subset, averaged over the 48 queries, plus "
      "the 10th percentile and worst query. The tail matters more than the mean.")
    w("- `starved_frac` - fraction of queries where the strategy returned fewer "
      "than k rows at all.")
    w("- `scan` - how the planner reached the rows: `hnsw`, `bitmap`, `btree` "
      "or `seq`, read off the EXPLAIN plan.\n")

    # ---------------- filter ladder ----------------
    w("## The selectivity ladder\n")
    w("Predicates were chosen by counting a few hundred readable candidates and "
      "keeping whichever landed nearest each target. Actual selectivity is "
      "reported everywhere, not the nominal target.\n")
    w("| target | actual | rows | predicate |\n|---|---|---|---|")
    for f in ladder:
        w(f"| {f.label} | {f.selectivity * 100:.4g}% | {f.n_rows:,} | `{f.sql}` |")
    w("")

    # ---------------- E1 ----------------
    w("## E1. Baseline: unfiltered ANN vs exact\n")
    w(_md_table(e1, [("strategy", "strategy"), ("ef_search", "ef_search"),
                     ("k", "k"), ("recall_mean", "recall@k"),
                     ("recall_p10", "p10"), ("recall_min", "min"),
                     ("p50_ms", "p50 ms"), ("p95_ms", "p95 ms"),
                     ("scan", "scan")]))
    w("")
    w("Query:\n")
    w(_sql_block(plans, "E1 ann SQL"))
    w(_plan_block(plans, "E1 ann ef_search=40 k=10",
                  "EXPLAIN (ANALYZE, BUFFERS) - unfiltered ANN, ef_search=40"))
    w(_plan_block(plans, "E1 exact k=10",
                  "EXPLAIN (ANALYZE, BUFFERS) - exact, index scans disabled"))
    w("**Finding.** " + _finding("E1"))

    # ---------------- E2 ----------------
    w("## E2. Filter selectivity sweep\n")
    w(_md_table(e2, [("filter_label", "target"), ("selectivity_pct", "actual %"),
                     ("n_rows", "rows"), ("strategy", "strategy"),
                     ("recall_mean", "recall@10"), ("recall_p10", "p10"),
                     ("recall_min", "min"), ("starved_frac", "starved"),
                     ("p50_ms", "p50 ms"), ("p95_ms", "p95 ms"),
                     ("scan", "scan")]))
    w("")
    w("Predicates used:\n")
    seen = set()
    for r in e2:
        if r["filter_label"] not in seen:
            seen.add(r["filter_label"])
            w(f"- **{r['filter_label']}** — `{r['filter_sql']}`")
    w("")
    w("### Queries\n")
    for label in ("post_filter 10x", "pre_filter", "exact"):
        key = f"E2 {e2[0]['filter_label']} {label} SQL"
        if key in plans:
            w(f"`{label}`:\n")
            w(_sql_block(plans, key))
    w("### Plans across the ladder\n")
    for r in e2:
        key = f"E2 {r['filter_label']} {r['strategy']}"
        w(_plan_block(plans, key, f"{r['filter_label']} — {r['strategy']}"))
    w("**Finding.** " + _finding("E2"))

    # ---------------- E3 ----------------
    w("## E3. Post-filter overfetch sweep\n")
    if e3:
        w(f"Filter: `{e3[0]['filter_sql']}` "
          f"({e3[0]['selectivity_pct']}% of the corpus).\n")
        w("Note `ef_search_used`: HNSW returns at most `ef_search` candidates, "
          "so a fetch depth above it is silently truncated. `ef_search` is "
          "raised to the fetch depth, which is part of what the overfetch "
          "actually costs.\n")
        w(_md_table(e3, [("overfetch", "overfetch"), ("fetch_k", "fetch k"),
                         ("ef_search_used", "ef_search"),
                         ("recall_mean", "recall@10"), ("recall_p10", "p10"),
                         ("recall_min", "min"), ("starved_frac", "starved"),
                         ("p50_ms", "p50 ms"), ("p95_ms", "p95 ms")]))
        w("")
        w(_sql_block(plans, "E3 SQL"))
        w(_plan_block(plans, "E3 overfetch=100x",
                      "EXPLAIN (ANALYZE, BUFFERS) - overfetch 100x"))
    w("**Finding.** " + _finding("E3"))

    # ---------------- E4 ----------------
    w("## E4. ef_search sweep\n")
    w(_md_table(e4, [("filter_label", "filter"), ("selectivity_pct", "actual %"),
                     ("strategy", "strategy"), ("ef_search", "ef_search"),
                     ("recall_mean", "recall@10"), ("recall_p10", "p10"),
                     ("starved_frac", "starved"),
                     ("p50_ms", "p50 ms"), ("p95_ms", "p95 ms"),
                     ("scan", "scan")]))
    w("")
    w("**Finding.** " + _finding("E4"))

    # ---------------- E5 ----------------
    w("## E5. pgvector iterative index scan\n")
    if not e5:
        w(plans.get("E5 unsupported", "Not run.") + "\n")
    else:
        w(f"pgvector {meta['pgvector_version']} supports `hnsw.iterative_scan` "
          "(added in 0.8.0). `relaxed_order` allows results to come back "
          "slightly out of distance order; `strict_order` preserves ordering at "
          "higher cost. `hnsw.max_scan_tuples` (default 20,000) caps the work.\n")
        w(_md_table(e5, [("filter_label", "filter"),
                         ("selectivity_pct", "actual %"), ("n_rows", "rows"),
                         ("iterative_scan", "iterative_scan"),
                         ("recall_mean", "recall@10"), ("recall_p10", "p10"),
                         ("recall_min", "min"), ("starved_frac", "starved"),
                         ("p50_ms", "p50 ms"), ("p95_ms", "p95 ms"),
                         ("scan", "scan")]))
        w("")
        for r in e5:
            if r["iterative_scan"] == "relaxed_order":
                w(_plan_block(plans,
                              f"E5 {r['filter_label']} iterative_scan=relaxed_order",
                              f"{r['filter_label']} — relaxed_order"))
    w("**Finding.** " + _finding("E5"))

    # ---------------- decision rule ----------------
    w("## Decision rule\n")
    w(_finding("DECISION"))

    w("## Caveats\n")
    w(_finding("CAVEATS"))

    path.write_text("\n".join(out) + "\n")
