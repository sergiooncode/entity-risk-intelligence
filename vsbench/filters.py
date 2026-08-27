"""
Building the selectivity ladder for E2.

The corpus has correlated attributes, so you cannot compute a predicate's
selectivity by multiplying marginals - P(topic='maritime' | jurisdiction='AE')
is 0.30, not 0.11. Rather than reason about it, we enumerate a few hundred
readable candidate predicates, count each one in SQL, and keep whichever lands
closest to each target.

Consequence worth stating up front: the filters below are chosen for their row
counts, and the *actual* selectivity is reported everywhere rather than the
nominal target. A filter labelled "1%" is whatever the corpus could supply
nearest to 1%.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import vocab

TARGETS = [0.50, 0.10, 0.01, 0.001, 0.0001]

# A filter needs enough rows for recall@10 to mean anything.
MIN_ROWS = 15


@dataclass
class Filter:
    label: str          # "1%"
    sql: str | None     # WHERE fragment, or None for the unfiltered case
    n_rows: int
    selectivity: float

    @property
    def pretty(self) -> str:
        return f"{self.label} ({self.selectivity * 100:.4g}%, {self.n_rows:,} rows)"


UNFILTERED = None  # sentinel used by the strategies


def _candidates_broad() -> list[str]:
    """Single-attribute and jurisdiction-group predicates."""
    out: list[str] = []

    for j in vocab.JURISDICTIONS:
        out.append(f"jurisdiction = '{j}'")
    for t in vocab.TOPICS:
        out.append(f"topic = '{t}'")

    # Jurisdiction groups, ordered by corpus weight so the prefixes give a
    # smooth ladder from ~22% up to ~90%.
    ordered = sorted(vocab.JURISDICTIONS,
                     key=lambda j: -vocab.JURISDICTION_WEIGHTS[j])
    for n in (2, 3, 4, 5, 6):
        group = ordered[:n]
        lst = ", ".join(f"'{j}'" for j in group)
        out.append(f"jurisdiction IN ({lst})")

    for year in (2020, 2022, 2023, 2024, 2025):
        out.append(f"published >= DATE '{year}-01-01'")

    out.append("is_designated")
    out.append("hops_to_designated <= 2")
    return out


def _candidates_pairs() -> list[str]:
    return [f"jurisdiction = '{j}' AND topic = '{t}'"
            for j in vocab.JURISDICTIONS for t in vocab.TOPICS]


def _candidates_narrow(pairs: list[str]) -> list[str]:
    """Refine promising two-attribute predicates with dates and flags."""
    out: list[str] = []
    for p in pairs:
        for year in (2023, 2024, 2025):
            dated = f"{p} AND published >= DATE '{year}-01-01'"
            out.append(dated)
            out.append(f"{dated} AND is_designated")
        out.append(f"{p} AND is_designated")
        out.append(f"{p} AND hops_to_designated <= 1")
    return out


def _count(conn, predicates: list[str]) -> dict[str, int]:
    counts = {}
    for p in predicates:
        n = conn.execute(f"SELECT count(*) FROM documents WHERE {p}").fetchone()[0]
        counts[p] = n
    return counts


def build_ladder(conn, total_rows: int, targets=TARGETS,
                 verbose: bool = True) -> list[Filter]:
    """
    Count candidate predicates and return one Filter per target selectivity.

    Two passes so we do not run several thousand COUNT(*) queries: the narrow
    candidates are only generated from two-attribute predicates that are
    already within a factor of ~50 of the smallest target.
    """
    if verbose:
        print("  counting broad candidates...")
    counts = _count(conn, _candidates_broad())

    pairs = _candidates_pairs()
    counts.update(_count(conn, pairs))

    smallest = min(targets) * total_rows
    promising = [p for p in pairs
                 if smallest <= counts[p] <= smallest * 400]
    if verbose:
        print(f"  refining {len(promising)} two-attribute predicates...")
    counts.update(_count(conn, _candidates_narrow(promising)))

    if verbose:
        print(f"  counted {len(counts)} candidate predicates")

    chosen: list[Filter] = []
    used: set[str] = set()
    for target in targets:
        want = target * total_rows
        best, best_err = None, None
        for pred, n in counts.items():
            if n < MIN_ROWS or pred in used:
                continue
            # Compare in log space: being 2x off matters equally at every scale.
            err = abs((n / want) if n >= want else (want / n))
            if best_err is None or err < best_err:
                best, best_err = pred, err
        if best is None:
            raise RuntimeError(f"no candidate predicate for target {target}")
        used.add(best)
        n = counts[best]
        label = f"{target * 100:g}%"
        chosen.append(Filter(label=label, sql=best, n_rows=n,
                             selectivity=n / total_rows))

    return chosen
