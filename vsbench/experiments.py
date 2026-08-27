"""
E1-E5. Each experiment returns (rows, plans): a list of dicts destined for CSV,
and a mapping of label -> EXPLAIN text.

Measurement conventions, applied everywhere:

  * Ground truth is `exact` at GT_K=100 over the *filtered* subset. recall@k is
    |returned_topk & truth_topk| / min(k, |filtered set|). The denominator
    matters at the tight end of the ladder, where a filter passes fewer than k
    rows and perfect recall is still less than k documents.

  * Latency is client-side wall clock around execute+fetch, over
    48 queries x REPS repetitions, after 2 warm-up passes. p50/p95 are taken
    over that whole pool of samples.

  * `starved` is the fraction of queries for which a strategy returned fewer
    than k rows. For post_filter this is the headline failure mode: the ANN
    prefix contained too few surviving rows to fill the result set at all.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from . import strategies as S
from .filters import Filter

GT_K = 100
REPS = 3
WARMUP = 2


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def recall_at(returned: list[int], truth: list[int], k: int) -> float:
    """Overlap of the top-k returned with the true top-k, over min(k, |truth|)."""
    denom = min(k, len(truth))
    if denom == 0:
        return float("nan")
    return len(set(returned[:k]) & set(truth[:k])) / denom


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[idx]


@dataclass
class Measured:
    recalls: list[float]
    latencies: list[float]
    returned_counts: list[int]
    scan: str
    plan_text: str
    plan_sql: str

    def row(self, k: int) -> dict:
        finite = [r for r in self.recalls if r == r]  # drop NaN
        return {
            "recall_mean": round(statistics.fmean(finite), 4) if finite else float("nan"),
            "recall_p10": round(pct(finite, 10), 4) if finite else float("nan"),
            "recall_min": round(min(finite), 4) if finite else float("nan"),
            "starved_frac": round(
                sum(1 for c in self.returned_counts if c < k) / len(self.returned_counts), 4),
            "p50_ms": round(pct(self.latencies, 50), 3),
            "p95_ms": round(pct(self.latencies, 95), 3),
            "scan": self.scan,
        }


class Bench:
    def __init__(self, conn, qvecs: np.ndarray, k: int = 10, reps: int = REPS):
        self.conn = conn
        self.qvecs = qvecs
        self.k = k
        self.reps = reps
        self._truth: dict[str, list[list[int]]] = {}

    # -- ground truth ------------------------------------------------------
    def truth(self, filt: Filter | None) -> list[list[int]]:
        """Exact top-GT_K per query over the filtered subset. Cached."""
        key = filt.sql if filt else "__unfiltered__"
        if key in self._truth:
            return self._truth[key]
        out = []
        for qv in self.qvecs:
            plan = S.exact(qv, filt.sql if filt else None, k=GT_K)
            out.append(S.run(self.conn, plan).doc_ids)
        self._truth[key] = out
        return out

    # -- measurement -------------------------------------------------------
    def measure(self, build, filt: Filter | None, k: int | None = None,
                truth_k: int | None = None) -> Measured:
        """
        `build` is a callable (qvec, filter_sql) -> Plan.
        """
        k = k or self.k
        truth_k = truth_k or k
        truth = self.truth(filt)
        fsql = filt.sql if filt else None

        # Warm up: first touch of an index pulls pages into shared_buffers and
        # would otherwise land entirely in the p95.
        for _ in range(WARMUP):
            S.run(self.conn, build(self.qvecs[0], fsql))

        recalls, lats, counts = [], [], []
        for i, qv in enumerate(self.qvecs):
            plan = build(qv, fsql)
            res = None
            for _ in range(self.reps):
                res = S.run(self.conn, plan)
                lats.append(res.elapsed_ms)
            recalls.append(recall_at(res.doc_ids, truth[i], truth_k))
            counts.append(len(res.doc_ids))

        probe = build(self.qvecs[0], fsql)
        plan_text = S.explain(self.conn, probe)
        return Measured(recalls, lats, counts, S.scan_kind(plan_text),
                        plan_text, probe.sql)


# --------------------------------------------------------------------------
# E1 - baseline, unfiltered
# --------------------------------------------------------------------------

def e1_baseline(b: Bench) -> tuple[list[dict], dict]:
    rows, plans = [], {}

    for ef in (40, 100, 400):
        for k in (10, 100):
            m = b.measure(lambda q, f, ef=ef, k=k: S.pre_filter(q, f, k=k, ef_search=ef),
                          None, k=k, truth_k=k)
            rows.append({
                "strategy": "ann (pre_filter, no WHERE)", "ef_search": ef, "k": k,
                **m.row(k),
            })
            plans[f"E1 ann ef_search={ef} k={k}"] = m.plan_text
            if ef == 40 and k == 10:
                plans["E1 ann SQL"] = m.plan_sql

    for k in (10, 100):
        m = b.measure(lambda q, f, k=k: S.exact(q, f, k=k), None, k=k, truth_k=k)
        rows.append({"strategy": "exact (brute force)", "ef_search": "-", "k": k,
                     **m.row(k)})
        plans[f"E1 exact k={k}"] = m.plan_text
        if k == 10:
            plans["E1 exact SQL"] = m.plan_sql

    return rows, plans


# --------------------------------------------------------------------------
# E2 - filter selectivity sweep
# --------------------------------------------------------------------------

def e2_selectivity(b: Bench, ladder: list[Filter], overfetch: int = 10
                   ) -> tuple[list[dict], dict]:
    rows, plans = [], {}
    k = b.k

    for filt in ladder:
        configs = [
            (f"post_filter {overfetch}x",
             lambda q, f, o=overfetch: S.post_filter(q, f, k=k, overfetch=o)),
            ("pre_filter", lambda q, f: S.pre_filter(q, f, k=k)),
            ("exact", lambda q, f: S.exact(q, f, k=k)),
        ]
        for name, build in configs:
            m = b.measure(build, filt)
            rows.append({
                "filter_label": filt.label, "filter_sql": filt.sql,
                "n_rows": filt.n_rows,
                "selectivity_pct": round(filt.selectivity * 100, 5),
                "strategy": name, **m.row(k),
            })
            plans[f"E2 {filt.label} {name}"] = m.plan_text
            plans[f"E2 {filt.label} {name} SQL"] = m.plan_sql

    return rows, plans


# --------------------------------------------------------------------------
# E3 - post-filter overfetch sweep
# --------------------------------------------------------------------------

def e3_overfetch(b: Bench, filt: Filter,
                 multipliers=(1, 5, 10, 50, 100)) -> tuple[list[dict], dict]:
    rows, plans = [], {}
    k = b.k
    for o in multipliers:
        m = b.measure(lambda q, f, o=o: S.post_filter(q, f, k=k, overfetch=o), filt)
        rows.append({
            "filter_label": filt.label, "filter_sql": filt.sql,
            "selectivity_pct": round(filt.selectivity * 100, 5),
            "overfetch": o, "fetch_k": k * o,
            "ef_search_used": max(40, k * o), **m.row(k),
        })
        plans[f"E3 overfetch={o}x"] = m.plan_text
        if o == 10:
            plans["E3 SQL"] = m.plan_sql
    return rows, plans


# --------------------------------------------------------------------------
# E4 - ef_search sweep
# --------------------------------------------------------------------------

def e4_ef_search(b: Bench, filters: list[Filter],
                 efs=(10, 40, 100, 400)) -> tuple[list[dict], dict]:
    rows, plans = [], {}
    k = b.k
    for filt in filters:
        for ef in efs:
            for name, build in (
                ("pre_filter",
                 lambda q, f, ef=ef: S.pre_filter(q, f, k=k, ef_search=ef)),
                ("post_filter 10x",
                 lambda q, f, ef=ef: S.post_filter(q, f, k=k, overfetch=10,
                                                   ef_search=ef)),
            ):
                m = b.measure(build, filt)
                rows.append({
                    "filter_label": filt.label,
                    "selectivity_pct": round(filt.selectivity * 100, 5),
                    "strategy": name, "ef_search": ef, **m.row(k),
                })
                plans[f"E4 {filt.label} {name} ef={ef}"] = m.plan_text
    return rows, plans


# --------------------------------------------------------------------------
# E5 - pgvector iterative index scan
# --------------------------------------------------------------------------

def e5_iterative(b: Bench, ladder: list[Filter], pgvector_version: str,
                 modes=("off", "relaxed_order", "strict_order")
                 ) -> tuple[list[dict], dict]:
    rows, plans = [], {}
    k = b.k

    major_minor = tuple(int(x) for x in pgvector_version.split(".")[:2])
    if major_minor < (0, 8):
        return [], {"E5 unsupported":
                    f"pgvector {pgvector_version} predates iterative index "
                    f"scan (added in 0.8.0). Experiment not run."}

    for filt in ladder:
        for mode in modes:
            m = b.measure(
                lambda q, f, mode=mode: S.pre_filter(q, f, k=k,
                                                     iterative_scan=mode),
                filt)
            rows.append({
                "filter_label": filt.label, "filter_sql": filt.sql,
                "n_rows": filt.n_rows,
                "selectivity_pct": round(filt.selectivity * 100, 5),
                "iterative_scan": mode, **m.row(k),
            })
            plans[f"E5 {filt.label} iterative_scan={mode}"] = m.plan_text
    return rows, plans
