"""
The three query strategies, behind one interface.

Each strategy is a function that *builds* a Plan (SQL text + parameters +
session settings). Nothing executes until you hand the Plan to `run()` or
`explain()`. That split is what lets RESULTS.md print the exact statement that
produced a timing alongside the EXPLAIN output for that same statement.

  post_filter  ANN over the whole table with LIMIT k*overfetch, filtered
               afterwards. The inner LIMIT is an optimisation barrier, so the
               outer WHERE genuinely cannot be pushed down into the index scan.

  pre_filter   WHERE clause and ORDER BY distance in one statement. The planner
               chooses: HNSW index scan with a filter applied on top, or a
               bitmap/seq scan of the matching rows followed by a sort. Which
               one it picks is the subject of E2.

  exact        Brute force over the filtered set. Index scans are disabled for
               the statement, so this is a seq scan plus a top-k sort. This is
               the ground truth every recall number is measured against.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

# A 384-dim vector literal is ~5 KB of digits and appears in the Order By line
# of every plan. Unreadable in a report, and it carries no information.
_VECTOR_LITERAL = re.compile(r"'\[[-0-9eE.,+]{80,}\]'::vector")


def elide_vectors(plan_text: str) -> str:
    return _VECTOR_LITERAL.sub("'[...384 dims...]'::vector", plan_text)


@dataclass
class Plan:
    label: str
    sql: str
    params: dict
    gucs: dict = field(default_factory=dict)


@dataclass
class Result:
    doc_ids: list[int]
    distances: list[float]
    elapsed_ms: float
    plan: Plan


def _where(filter_sql: str | None) -> str:
    return f"WHERE {filter_sql}\n" if filter_sql else ""


# --------------------------------------------------------------------------
# Plan builders
# --------------------------------------------------------------------------

def post_filter(qvec: np.ndarray, filter_sql: str | None, k: int = 10,
                overfetch: int = 10, ef_search: int = 40) -> Plan:
    """
    ANN first, filter second.

    NOTE on ef_search: HNSW returns at most `ef_search` candidates from the
    bottom layer, so LIMIT k*overfetch is silently capped at ef_search. An
    overfetch of 100x with ef_search=40 fetches 40 rows, not 1000. We therefore
    raise ef_search to at least the fetch depth. E3 measures the cost of that.
    """
    fetch_k = k * overfetch
    # The inner query has no WHERE: that is what makes this a post-filter. It
    # must project the columns the outer predicate needs, and its LIMIT is an
    # optimisation barrier, so Postgres cannot push the outer WHERE down into
    # the index scan.
    sql = f"""SELECT doc_id, distance
FROM (
    SELECT doc_id,
           jurisdiction, topic, published, is_designated, hops_to_designated,
           embedding <=> %(q)s AS distance
    FROM documents
    ORDER BY embedding <=> %(q)s
    LIMIT {fetch_k}
) AS ann
{_where(filter_sql)}ORDER BY distance
LIMIT {k}"""
    return Plan(
        label=f"post_filter(k={k}, overfetch={overfetch}x, ef_search={ef_search})",
        sql=sql,
        params={"q": qvec},
        gucs={"hnsw.ef_search": max(ef_search, fetch_k)},
    )


def pre_filter(qvec: np.ndarray, filter_sql: str | None, k: int = 10,
               ef_search: int = 40, iterative_scan: str = "off",
               max_scan_tuples: int | None = None) -> Plan:
    """
    Filter and ANN in one statement. The planner decides how to combine them.

    `iterative_scan` is pgvector 0.8+. When it is not "off", the index scan
    resumes and fetches more candidates if the filter rejected too many, rather
    than stopping at ef_search.
    """
    gucs = {"hnsw.ef_search": ef_search, "hnsw.iterative_scan": iterative_scan}
    if max_scan_tuples is not None:
        gucs["hnsw.max_scan_tuples"] = max_scan_tuples

    sql = f"""SELECT doc_id,
       embedding <=> %(q)s AS distance
FROM documents
{_where(filter_sql)}ORDER BY embedding <=> %(q)s
LIMIT {k}"""
    label = f"pre_filter(k={k}, ef_search={ef_search}"
    if iterative_scan != "off":
        label += f", iterative_scan={iterative_scan}"
    return Plan(label=label + ")", sql=sql, params={"q": qvec}, gucs=gucs)


def exact(qvec: np.ndarray, filter_sql: str | None, k: int = 10) -> Plan:
    """
    Ground truth: brute-force distance over the filtered set.

    enable_indexscan and enable_bitmapscan are switched off so the planner
    cannot reach the HNSW index (or use a bitmap over the btrees, which would
    still be exact but would muddy the "this is the brute-force cost" reading).
    """
    sql = f"""SELECT doc_id,
       embedding <=> %(q)s AS distance
FROM documents
{_where(filter_sql)}ORDER BY embedding <=> %(q)s
LIMIT {k}"""
    return Plan(
        label=f"exact(k={k})",
        sql=sql,
        params={"q": qvec},
        gucs={"enable_indexscan": "off", "enable_bitmapscan": "off"},
    )


STRATEGIES = {"post_filter": post_filter, "pre_filter": pre_filter, "exact": exact}


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def _apply(cur, gucs: dict) -> None:
    """
    Apply session settings for the current transaction only.

    `SET LOCAL` is utility syntax and does not accept bind parameters, so we
    use set_config(name, value, is_local => true), which is the same thing as
    a function call and does.
    """
    for key, value in gucs.items():
        cur.execute("SELECT set_config(%s, %s, true)", (key, str(value)))


def run(conn, plan: Plan) -> Result:
    """Execute a Plan and time it client-side, including row fetch."""
    with conn.transaction():
        with conn.cursor() as cur:
            _apply(cur, plan.gucs)
            t0 = time.perf_counter()
            cur.execute(plan.sql, plan.params)
            rows = cur.fetchall()
            elapsed = (time.perf_counter() - t0) * 1000.0
    return Result(
        doc_ids=[r[0] for r in rows],
        distances=[float(r[1]) for r in rows],
        elapsed_ms=elapsed,
        plan=plan,
    )


def explain(conn, plan: Plan, analyze: bool = True) -> str:
    """Return the EXPLAIN (ANALYZE, BUFFERS) text for exactly this Plan."""
    opts = "ANALYZE, BUFFERS, VERBOSE OFF, COSTS ON" if analyze else "COSTS ON"
    with conn.transaction():
        with conn.cursor() as cur:
            _apply(cur, plan.gucs)
            cur.execute(f"EXPLAIN ({opts}) {plan.sql}", plan.params)
            return elide_vectors("\n".join(r[0] for r in cur.fetchall()))


def scan_kind(plan_text: str) -> str:
    """
    Classify an EXPLAIN plan by how it reached the rows. This is the column
    that answers "did the filter knock us off the HNSW index?".
    """
    t = plan_text.lower()
    if "documents_embedding_hnsw" in t:
        return "hnsw"
    if "bitmap heap scan" in t:
        return "bitmap"
    if "index scan" in t or "index only scan" in t:
        return "btree"
    if "seq scan" in t:
        return "seq"
    return "other"
