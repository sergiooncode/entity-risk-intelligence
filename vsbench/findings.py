"""
Hand-written prose for RESULTS.md.

Kept separate from report.py so the report can be regenerated
(`python scripts/make_report.py`) after editing the interpretation, without
re-running the benchmark. Written after reading the numbers.
"""

FINDINGS: dict[str, str] = {

    "E1": """
HNSW is genuinely in use: every unfiltered plan is
`Index Scan using documents_embedding_hnsw`, and the index is doing real work —
1.8 ms p50 against 177 ms for the brute-force seq scan, a ~100x gap on 200k
rows. Recall@10 at the default `ef_search=40` is 0.873, rising to 0.963 at 100
and 0.998 at 400, with latency going 1.8 / 4.0 / 11.0 ms. The recall@100 row at
`ef_search=40` looks alarming at 0.395, but it is not a graph-quality problem:
HNSW returns at most `ef_search` candidates, so a `LIMIT 100` query physically
cannot return more than 40 rows — `starved_frac` is 1.0 for exactly that reason.
The rule this establishes for everything downstream is that `ef_search` is a
hard ceiling on result-set size, not a quality knob you can leave at the default
and ignore.
""",

    "E2": """
There is a clean crossover, and it sits between 1% and 0.1% selectivity. Above
it, the planner stays on the HNSW index and both approximate strategies degrade
badly: at 9.1% selectivity pre-filter and post-filter return 0.123 and 0.125
recall@10, and roughly 7 queries in 8 come back with fewer than 10 rows at all.
At 0.99% it is worse — 0.025 and 0.056, with pre-filter starved on *every*
query. Below the crossover the planner abandons HNSW for `BitmapAnd` over the
three btrees plus a sort, and recall jumps to a perfect 1.0 at 9.2 ms (0.1%) and
4.7 ms (0.01%) — faster *and* exact, with no intervention required. The
important and slightly uncomfortable result is the band between about 1% and
10%: it is where a filtered vector query is most likely to appear in production,
it is where both approximate strategies are near-useless, and nothing in the
query plan flags a problem — you get a fast `Index Scan using
documents_embedding_hnsw` that quietly returns two rows instead of ten. The
mechanism is that a correlated filter does not remove a random 9% of the corpus.
Probing directly: for a maritime seed document, `topic = 'military_end_use'`
matches 9.1% of the corpus but only 6 of the query's global top 1000 — a local
selectivity of 0.6%, 15x worse than the global figure. The filter deletes
precisely the region the graph walk was heading toward.
""",

    "E3": """
Overfetch is a real but weak lever, and it does not rescue the 1% case. Recall@10
climbs 0.002 → 0.031 → 0.056 → 0.194 → 0.235 across 1x / 5x / 10x / 50x / 100x,
while p50 goes 1.9 ms → 22.4 ms. Even at 100x — fetching 1000 candidates with
`ef_search` raised to 1000, which is pgvector's maximum — 79% of queries still
return fewer than 10 rows, and mean recall is 0.235. The comparison that settles
it: exact search over the same filtered set costs 30.7 ms and returns recall
1.0, so the 100x overfetch is both slower (22.4 ms, but with a much worse p95)
and wrong. Note also that overfetch is inert unless `ef_search` is raised to
match the fetch depth; a `LIMIT 1000` with `ef_search=40` fetches 40 rows and
the multiplier does nothing. That interaction is not obvious from the pgvector
documentation and is worth knowing before anyone concludes that overfetch
"didn't help".
""",

    "E4": """
`ef_search` cannot buy your way out of a correlated filter either. At 9.1%
selectivity, sweeping 10 → 400 moves pre-filter recall only from 0.106 to 0.148
while p50 goes from 0.75 ms to 8.6 ms, and the starved fraction barely moves
(0.875 → 0.854). At 0.99% the sweep is slightly more productive in relative
terms — 0.004 → 0.183 — but that is still an 11x latency increase to reach a
recall figure that would be unacceptable in any application. Post-filter at 10x
is flat at 0.125 across the entire sweep at 9.1%, because its bottleneck is the
fetch depth rather than the candidate list. The reading is that `ef_search` and
overfetch are both linear-cost mitigations for a problem that scales with how
strongly the filter anti-correlates with the query, so neither reaches useful
recall before the exact scan becomes cheaper.
""",

    "E5": """
pgvector 0.8.6 is installed, so iterative scan ran. It is the only mitigation
tested that actually fixes starvation: at 0.99% selectivity, `relaxed_order`
takes recall@10 from 0.025 to 0.444 and drops the starved fraction from 1.0 to
0.042, with `strict_order` close behind at 0.398. But the cost is a 34x latency
increase, 1.4 ms → 49.0 ms, and — the part that matters — exact search on the
same filter is 30.7 ms with recall 1.0. So at this selectivity iterative scan is
strictly dominated: slower than brute force and less accurate. At 0.1% and 0.01%
it changes nothing at all, because the planner has already switched to a bitmap
scan and never enters the HNSW code path. The recall ceiling of 0.44 is partly
an artifact of leaving `hnsw.max_scan_tuples` at its 20,000 default; raising it
would trade still more latency for more recall, which does not change the
conclusion. Iterative scan is the right tool when the filtered set is far too
large to scan exactly — much bigger than 200k rows — and a fast approximate
answer beats a slow exact one. It is not the right tool here.
""",

    "DECISION": """
Stated as a rule for this corpus, at 200k rows with `m=16`,
`ef_construction=64` and k=10:

- **Use `exact` (or simply let the planner reach it) when the filter passes
  under ~0.5% of the corpus** — under roughly 1,000 rows here. You do not have
  to do anything: below ~0.1% Postgres already picks `BitmapAnd` + sort on its
  own, and it is both faster (4.7–9.2 ms) and exact. Between ~0.1% and ~1%,
  force it — disable index scans for the statement, or accept the bitmap plan —
  because approximate search in that band returns near-zero recall.

- **Use `post_filter` with 10x overfetch only when the filter passes more than
  ~30% of the corpus** *and* you have checked it is weakly correlated with query
  content. At 47.7% it delivers recall 0.910 at 4.4 ms, which is a good trade.
  At 9.1% it delivers 0.125 and should not be used.

- **Use plain `pre_filter` on HNSW only when the filter is very weak
  (>50%)**, and set `ef_search >= k / local_selectivity`, not
  `k / global_selectivity`. Even at 47.7% it starves on a third of queries at
  the default `ef_search=40`.

- **Reach for `hnsw.iterative_scan` when the filtered set is too large to scan
  exactly.** At 200k rows that condition is never met — exact search costs at
  most 177 ms unfiltered. On a corpus 10–100x larger, iterative scan at ~49 ms
  becomes the sensible answer in the 1–10% band.

- **Never plan capacity from global selectivity.** Measure how many of a
  representative query's top-1000 survive the filter. If that number is below
  ~10x k, approximate search will not give you k good results at any `ef_search`
  you can afford.
""",

    "CAVEATS": """
- `exact` is the ground truth, so its recall column is 1.0 by construction. Only
  its latency column carries information.
- Everything is warm and in memory: the 193 MB table and 391 MB HNSW index both
  fit in `shared_buffers=2GB`. A disk-bound instance would shift the crossover
  toward exact search being relatively more expensive.
- The filters are correlated with document content **on purpose**, because that
  is the realistic case and the one that breaks post-filtering. A benchmark with
  independent filter fields would show post-filter recovering most of its recall
  at 10x overfetch, and would conclude almost the opposite. The specific
  crossover point is therefore a property of this corpus, not a universal
  constant — the method for finding it is the transferable part.
- About 3% of documents are deliberate near-duplicates, so some near-ties exist.
  Where an approximate search returns an equivalent duplicate instead of the
  exact top-k member, recall is understated slightly. This was not corrected
  for.
- `hnsw.max_scan_tuples` was left at its 20,000 default throughout E5.
- One embedding model, one index configuration, one value of k, 48 queries.
  p95 figures come from 144 samples and are noisy at the tail; the p50 and
  recall columns are the reliable ones.
- The 0.01% rung passes only 20 rows, so recall@10 there is measured against a
  set barely twice k. It is included for completeness, not as a strong result.
""",
}
