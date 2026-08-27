# Metadata filtering and HNSW in pgvector

How pre-filtering, post-filtering and exact search behave as a metadata filter gets more selective, measured on a 200k-document synthetic sanctions/export-control corpus.

## Setup

| setting | value |
|---|---|
| Corpus | 200,000 documents, 15,000 entities |
| Embedding | all-MiniLM-L6-v2 (384-dim, cosine), device `mps` |
| Postgres | 16.15 (Debian 16.15-1.pgdg12+2) |
| pgvector | 0.8.6 |
| ANN index | HNSW, `m=16`, `ef_construction=64`, `vector_cosine_ops` |
| Scalar indexes | btree on `jurisdiction`, `topic`, `published` |
| k | 10 |
| Latency samples | 48 queries x 3 reps, 2 warm-up passes |
| HNSW build time | 68s |
| Embedding time | 437s |

Object sizes:

| object | size |
|---|---|
| `documents (table)` | 193 MB |
| `documents_embedding_hnsw` | 391 MB |
| `documents_jurisdiction_idx` | 1368 kB |
| `documents_pkey` | 4408 kB |
| `documents_published_idx` | 1408 kB |
| `documents_topic_idx` | 1408 kB |

Column meanings used throughout:

- `recall_mean` / `recall_p10` / `recall_min` - recall@k against exact search over the *filtered* subset, averaged over the 48 queries, plus the 10th percentile and worst query. The tail matters more than the mean.
- `starved_frac` - fraction of queries where the strategy returned fewer than k rows at all.
- `scan` - how the planner reached the rows: `hnsw`, `bitmap`, `btree` or `seq`, read off the EXPLAIN plan.

## The selectivity ladder

Predicates were chosen by counting a few hundred readable candidates and keeping whichever landed nearest each target. Actual selectivity is reported everywhere, not the nominal target.

| target | actual | rows | predicate |
|---|---|---|---|
| 50% | 47.72% | 95,433 | `jurisdiction IN ('CN', 'RU', 'HK')` |
| 10% | 9.143% | 18,285 | `topic = 'military_end_use'` |
| 1% | 0.9865% | 1,973 | `jurisdiction = 'HK' AND topic = 'sanctions_evasion' AND published >= DATE '2025-01-01'` |
| 0.1% | 0.1% | 200 | `jurisdiction = 'CN' AND topic = 'military_end_use' AND published >= DATE '2024-01-01' AND is_designated` |
| 0.01% | 0.01% | 20 | `jurisdiction = 'TR' AND topic = 'forced_labour' AND published >= DATE '2025-01-01'` |

## E1. Baseline: unfiltered ANN vs exact

| strategy | ef_search | k | recall@k | p10 | min | p50 ms | p95 ms | scan |
|---|---|---|---|---|---|---|---|---|
| ann (pre_filter, no WHERE) | 40 | 10 | 0.8729 | 0.7 | 0 | 1.786 | 15.17 | hnsw |
| ann (pre_filter, no WHERE) | 40 | 100 | 0.3946 | 0.4 | 0.14 | 1.774 | 3.281 | hnsw |
| ann (pre_filter, no WHERE) | 100 | 10 | 0.9625 | 0.9 | 0.7 | 4.022 | 10.55 | hnsw |
| ann (pre_filter, no WHERE) | 100 | 100 | 0.8712 | 0.7 | 0.54 | 3.744 | 6.1 | hnsw |
| ann (pre_filter, no WHERE) | 400 | 10 | 0.9979 | 1 | 0.9 | 11.04 | 23.16 | hnsw |
| ann (pre_filter, no WHERE) | 400 | 100 | 0.9685 | 0.92 | 0.78 | 9.884 | 14.99 | hnsw |
| exact (brute force) | - | 10 | 1 | 1 | 1 | 176.6 | 262.6 | seq |
| exact (brute force) | - | 100 | 1 | 1 | 1 | 171 | 231 | seq |

Query:

```sql
SELECT doc_id,
       embedding <=> %(q)s AS distance
FROM documents
ORDER BY embedding <=> %(q)s
LIMIT 10;
```

**EXPLAIN (ANALYZE, BUFFERS) - unfiltered ANN, ef_search=40**

```
Limit  (cost=204.81..209.11 rows=10 width=12) (actual time=1.437..1.572 rows=10 loops=1)
  Buffers: shared hit=954
  ->  Index Scan using documents_embedding_hnsw on documents  (cost=204.81..86105.10 rows=200000 width=12) (actual time=1.436..1.571 rows=10 loops=1)
        Order By: (embedding <=> '[...384 dims...]'::vector)
        Buffers: shared hit=954
Planning:
  Buffers: shared hit=1
Planning Time: 0.035 ms
Execution Time: 1.582 ms
```

**EXPLAIN (ANALYZE, BUFFERS) - exact, index scans disabled**

```
Limit  (cost=27840.66..27841.84 rows=10 width=12) (actual time=179.412..190.780 rows=10 loops=1)
  Buffers: shared hit=824931
  ->  Gather Merge  (cost=27840.66..50743.84 rows=193548 width=12) (actual time=179.411..190.777 rows=10 loops=1)
        Workers Planned: 3
        Workers Launched: 3
        Buffers: shared hit=824931
        ->  Sort  (cost=26840.62..27001.91 rows=64516 width=12) (actual time=176.915..176.918 rows=8 loops=4)
              Sort Key: ((embedding <=> '[...384 dims...]'::vector))
              Sort Method: top-N heapsort  Memory: 25kB
              Buffers: shared hit=824931
              Worker 0:  Sort Method: top-N heapsort  Memory: 25kB
              Worker 1:  Sort Method: top-N heapsort  Memory: 25kB
              Worker 2:  Sort Method: top-N heapsort  Memory: 25kB
              ->  Parallel Seq Scan on documents  (cost=0.00..25446.45 rows=64516 width=12) (actual time=0.142..172.189 rows=50000 loops=4)
                    Buffers: shared hit=824760
Planning:
  Buffers: shared hit=1
Planning Time: 0.052 ms
Execution Time: 190.801 ms
```

**Finding.** HNSW is genuinely in use: every unfiltered plan is
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

## E2. Filter selectivity sweep

| target | actual % | rows | strategy | recall@10 | p10 | min | starved | p50 ms | p95 ms | scan |
|---|---|---|---|---|---|---|---|---|---|---|
| 50% | 47.72 | 95433 | post_filter 10x | 0.9104 | 0.7 | 0.6 | 0.0208 | 4.365 | 30.8 | hnsw |
| 50% | 47.72 | 95433 | pre_filter | 0.7646 | 0.4 | 0 | 0.3333 | 1.456 | 4.453 | hnsw |
| 50% | 47.72 | 95433 | exact | 1 | 1 | 1 | 0 | 100.5 | 164 | seq |
| 10% | 9.143 | 18285 | post_filter 10x | 0.125 | 0 | 0 | 0.875 | 4.213 | 14.37 | hnsw |
| 10% | 9.143 | 18285 | pre_filter | 0.1229 | 0 | 0 | 0.875 | 1.21 | 2.328 | hnsw |
| 10% | 9.143 | 18285 | exact | 1 | 1 | 1 | 0 | 45.33 | 56.44 | seq |
| 1% | 0.9865 | 1973 | post_filter 10x | 0.0563 | 0 | 0 | 0.9792 | 4.046 | 20.58 | hnsw |
| 1% | 0.9865 | 1973 | pre_filter | 0.025 | 0 | 0 | 1 | 1.329 | 2.552 | hnsw |
| 1% | 0.9865 | 1973 | exact | 1 | 1 | 1 | 0 | 30.71 | 47.05 | seq |
| 0.1% | 0.1 | 200 | post_filter 10x | 0.0042 | 0 | 0 | 1 | 3.425 | 15.86 | hnsw |
| 0.1% | 0.1 | 200 | pre_filter | 1 | 1 | 1 | 0 | 9.157 | 11.94 | bitmap |
| 0.1% | 0.1 | 200 | exact | 1 | 1 | 1 | 0 | 26.1 | 30.94 | seq |
| 0.01% | 0.01 | 20 | post_filter 10x | 0.0021 | 0 | 0 | 1 | 2.691 | 8.971 | hnsw |
| 0.01% | 0.01 | 20 | pre_filter | 1 | 1 | 1 | 0 | 4.741 | 5.874 | bitmap |
| 0.01% | 0.01 | 20 | exact | 1 | 1 | 1 | 0 | 27.09 | 38.58 | seq |

Predicates used:

- **50%** — `jurisdiction IN ('CN', 'RU', 'HK')`
- **10%** — `topic = 'military_end_use'`
- **1%** — `jurisdiction = 'HK' AND topic = 'sanctions_evasion' AND published >= DATE '2025-01-01'`
- **0.1%** — `jurisdiction = 'CN' AND topic = 'military_end_use' AND published >= DATE '2024-01-01' AND is_designated`
- **0.01%** — `jurisdiction = 'TR' AND topic = 'forced_labour' AND published >= DATE '2025-01-01'`

### Queries

`post_filter 10x`:

```sql
SELECT doc_id, distance
FROM (
    SELECT doc_id,
           jurisdiction, topic, published, is_designated, hops_to_designated,
           embedding <=> %(q)s AS distance
    FROM documents
    ORDER BY embedding <=> %(q)s
    LIMIT 100
) AS ann
WHERE jurisdiction IN ('CN', 'RU', 'HK')
ORDER BY distance
LIMIT 10;
```

`pre_filter`:

```sql
SELECT doc_id,
       embedding <=> %(q)s AS distance
FROM documents
WHERE jurisdiction IN ('CN', 'RU', 'HK')
ORDER BY embedding <=> %(q)s
LIMIT 10;
```

`exact`:

```sql
SELECT doc_id,
       embedding <=> %(q)s AS distance
FROM documents
WHERE jurisdiction IN ('CN', 'RU', 'HK')
ORDER BY embedding <=> %(q)s
LIMIT 10;
```

### Plans across the ladder

**50% — post_filter 10x**

```
Limit  (cost=408.60..417.81 rows=10 width=12) (actual time=8.752..9.153 rows=10 loops=1)
  Buffers: shared hit=1775
  ->  Subquery Scan on ann  (cost=408.60..452.82 rows=48 width=12) (actual time=8.751..9.151 rows=10 loops=1)
        Filter: (ann.jurisdiction = ANY ('{CN,RU,HK}'::text[]))
        Buffers: shared hit=1775
        ->  Limit  (cost=408.60..451.45 rows=100 width=54) (actual time=8.747..9.143 rows=10 loops=1)
              Buffers: shared hit=1775
              ->  Index Scan using documents_embedding_hnsw on documents  (cost=408.60..86105.10 rows=200000 width=54) (actual time=8.745..9.140 rows=10 loops=1)
                    Order By: (embedding <=> '[...384 dims...]'::vector)
                    Buffers: shared hit=1775
Planning:
  Buffers: shared hit=1
Planning Time: 0.080 ms
Execution Time: 9.172 ms
```

**50% — pre_filter**

```
Limit  (cost=204.81..213.84 rows=10 width=12) (actual time=0.930..0.982 rows=10 loops=1)
  Buffers: shared hit=954
  ->  Index Scan using documents_embedding_hnsw on documents  (cost=204.81..86594.30 rows=95680 width=12) (actual time=0.930..0.981 rows=10 loops=1)
        Order By: (embedding <=> '[...384 dims...]'::vector)
        Filter: (jurisdiction = ANY ('{CN,RU,HK}'::text[]))
        Buffers: shared hit=954
Planning:
  Buffers: shared hit=1
Planning Time: 0.027 ms
Execution Time: 0.988 ms
```

**50% — exact**

```
Limit  (cost=27271.28..27272.46 rows=10 width=12) (actual time=92.815..102.769 rows=10 loops=1)
  Buffers: shared hit=406672
  ->  Gather Merge  (cost=27271.28..38228.36 rows=92595 width=12) (actual time=92.814..102.767 rows=10 loops=1)
        Workers Planned: 3
        Workers Launched: 3
        Buffers: shared hit=406672
        ->  Sort  (cost=26271.24..26348.40 rows=30865 width=12) (actual time=90.808..90.809 rows=9 loops=4)
              Sort Key: ((embedding <=> '[...384 dims...]'::vector))
              Sort Method: top-N heapsort  Memory: 25kB
              Buffers: shared hit=406672
              Worker 0:  Sort Method: top-N heapsort  Memory: 25kB
              Worker 1:  Sort Method: top-N heapsort  Memory: 25kB
              Worker 2:  Sort Method: top-N heapsort  Memory: 25kB
              ->  Parallel Seq Scan on documents  (cost=0.00..25604.26 rows=30865 width=12) (actual time=0.068..88.901 rows=23858 loops=4)
                    Filter: (jurisdiction = ANY ('{CN,RU,HK}'::text[]))
                    Rows Removed by Filter: 26142
                    Buffers: shared hit=406501
Planning:
  Buffers: shared hit=1
Planning Time: 0.076 ms
Execution Time: 102.786 ms
```

**10% — post_filter 10x**

```
Limit  (cost=408.60..452.70 rows=9 width=12) (actual time=2.513..2.513 rows=0 loops=1)
  Buffers: shared hit=2225
  ->  Subquery Scan on ann  (cost=408.60..452.70 rows=9 width=12) (actual time=2.513..2.513 rows=0 loops=1)
        Filter: (ann.topic = 'military_end_use'::text)
        Rows Removed by Filter: 100
        Buffers: shared hit=2225
        ->  Limit  (cost=408.60..451.45 rows=100 width=65) (actual time=1.967..2.508 rows=100 loops=1)
              Buffers: shared hit=2225
              ->  Index Scan using documents_embedding_hnsw on documents  (cost=408.60..86105.10 rows=200000 width=65) (actual time=1.967..2.500 rows=100 loops=1)
                    Order By: (embedding <=> '[...384 dims...]'::vector)
                    Buffers: shared hit=2225
Planning:
  Buffers: shared hit=1
Planning Time: 0.049 ms
Execution Time: 2.523 ms
```

**10% — pre_filter**

```
Limit  (cost=204.81..253.30 rows=10 width=12) (actual time=1.457..1.457 rows=0 loops=1)
  Buffers: shared hit=944
  ->  Index Scan using documents_embedding_hnsw on documents  (cost=204.81..86149.42 rows=17727 width=12) (actual time=1.456..1.456 rows=0 loops=1)
        Order By: (embedding <=> '[...384 dims...]'::vector)
        Filter: (topic = 'military_end_use'::text)
        Rows Removed by Filter: 40
        Buffers: shared hit=944
Planning:
  Buffers: shared hit=1
Planning Time: 0.046 ms
Execution Time: 1.468 ms
```

**10% — exact**

```
Limit  (cost=26584.35..26585.53 rows=10 width=12) (actual time=41.588..47.649 rows=10 loops=1)
  Buffers: shared hit=98071
  ->  Gather Merge  (cost=26584.35..28614.24 rows=17154 width=12) (actual time=41.587..47.643 rows=10 loops=1)
        Workers Planned: 3
        Workers Launched: 3
        Buffers: shared hit=98071
        ->  Sort  (cost=25584.31..25598.61 rows=5718 width=12) (actual time=38.190..38.191 rows=8 loops=4)
              Sort Key: ((embedding <=> '[...384 dims...]'::vector))
              Sort Method: top-N heapsort  Memory: 25kB
              Buffers: shared hit=98071
              Worker 0:  Sort Method: top-N heapsort  Memory: 25kB
              Worker 1:  Sort Method: top-N heapsort  Memory: 25kB
              Worker 2:  Sort Method: top-N heapsort  Memory: 25kB
              ->  Parallel Seq Scan on documents  (cost=0.00..25460.75 rows=5718 width=12) (actual time=0.077..37.666 rows=4571 loops=4)
                    Filter: (topic = 'military_end_use'::text)
                    Rows Removed by Filter: 45429
                    Buffers: shared hit=97900
Planning:
  Buffers: shared hit=1
Planning Time: 0.090 ms
Execution Time: 47.682 ms
```

**1% — post_filter 10x**

```
Limit  (cost=408.60..453.20 rows=1 width=12) (actual time=3.670..3.671 rows=0 loops=1)
  Buffers: shared hit=2225
  ->  Subquery Scan on ann  (cost=408.60..453.20 rows=1 width=12) (actual time=3.670..3.670 rows=0 loops=1)
        Filter: ((ann.published >= '2025-01-01'::date) AND (ann.jurisdiction = 'HK'::text) AND (ann.topic = 'sanctions_evasion'::text))
        Rows Removed by Filter: 100
        Buffers: shared hit=2225
        ->  Limit  (cost=408.60..451.45 rows=100 width=36) (actual time=2.633..3.662 rows=100 loops=1)
              Buffers: shared hit=2225
              ->  Index Scan using documents_embedding_hnsw on documents  (cost=408.60..86105.10 rows=200000 width=36) (actual time=2.633..3.654 rows=100 loops=1)
                    Order By: (embedding <=> '[...384 dims...]'::vector)
                    Buffers: shared hit=2225
Planning:
  Buffers: shared hit=1
Planning Time: 0.079 ms
Execution Time: 3.687 ms
```

**1% — pre_filter**

```
Limit  (cost=204.81..973.19 rows=10 width=12) (actual time=1.070..1.071 rows=0 loops=1)
  Buffers: shared hit=944
  ->  Index Scan using documents_embedding_hnsw on documents  (cost=204.81..87107.93 rows=1131 width=12) (actual time=1.070..1.070 rows=0 loops=1)
        Order By: (embedding <=> '[...384 dims...]'::vector)
        Filter: ((published >= '2025-01-01'::date) AND (jurisdiction = 'HK'::text) AND (topic = 'sanctions_evasion'::text))
        Rows Removed by Filter: 40
        Buffers: shared hit=944
Planning:
  Buffers: shared hit=1
Planning Time: 0.031 ms
Execution Time: 1.090 ms
```

**1% — exact**

```
Limit  (cost=26777.87..26779.06 rows=10 width=12) (actual time=22.375..28.792 rows=10 loops=1)
  Buffers: shared hit=32823
  ->  Gather Merge  (cost=26777.87..26907.45 rows=1095 width=12) (actual time=22.374..28.790 rows=10 loops=1)
        Workers Planned: 3
        Workers Launched: 3
        Buffers: shared hit=32823
        ->  Sort  (cost=25777.83..25778.74 rows=365 width=12) (actual time=20.318..20.319 rows=9 loops=4)
              Sort Key: ((embedding <=> '[...384 dims...]'::vector))
              Sort Method: top-N heapsort  Memory: 25kB
              Buffers: shared hit=32823
              Worker 0:  Sort Method: top-N heapsort  Memory: 25kB
              Worker 1:  Sort Method: top-N heapsort  Memory: 25kB
              Worker 2:  Sort Method: top-N heapsort  Memory: 25kB
              ->  Parallel Seq Scan on documents  (cost=0.00..25769.94 rows=365 width=12) (actual time=0.177..20.214 rows=493 loops=4)
                    Filter: ((published >= '2025-01-01'::date) AND (jurisdiction = 'HK'::text) AND (topic = 'sanctions_evasion'::text))
                    Rows Removed by Filter: 49507
                    Buffers: shared hit=32652
Planning:
  Buffers: shared hit=1
Planning Time: 0.072 ms
Execution Time: 28.805 ms
```

**0.1% — post_filter 10x**

```
Limit  (cost=408.60..453.20 rows=1 width=12) (actual time=2.824..2.824 rows=0 loops=1)
  Buffers: shared hit=2225
  ->  Subquery Scan on ann  (cost=408.60..453.20 rows=1 width=12) (actual time=2.823..2.824 rows=0 loops=1)
        Filter: (ann.is_designated AND (ann.published >= '2024-01-01'::date) AND (ann.jurisdiction = 'CN'::text) AND (ann.topic = 'military_end_use'::text))
        Rows Removed by Filter: 100
        Buffers: shared hit=2225
        ->  Limit  (cost=408.60..451.45 rows=100 width=36) (actual time=2.213..2.818 rows=100 loops=1)
              Buffers: shared hit=2225
              ->  Index Scan using documents_embedding_hnsw on documents  (cost=408.60..86105.10 rows=200000 width=36) (actual time=2.212..2.810 rows=100 loops=1)
                    Order By: (embedding <=> '[...384 dims...]'::vector)
                    Buffers: shared hit=2225
Planning:
  Buffers: shared hit=1
Planning Time: 0.078 ms
Execution Time: 2.838 ms
```

**0.1% — pre_filter**

```
Limit  (cost=2838.26..2838.29 rows=10 width=12) (actual time=8.834..8.836 rows=10 loops=1)
  Buffers: shared hit=3400
  ->  Sort  (cost=2838.26..2838.52 rows=102 width=12) (actual time=8.833..8.834 rows=10 loops=1)
        Sort Key: ((embedding <=> '[...384 dims...]'::vector))
        Sort Method: top-N heapsort  Memory: 25kB
        Buffers: shared hit=3400
        ->  Bitmap Heap Scan on documents  (cost=1202.24..2836.06 rows=102 width=12) (actual time=6.889..8.812 rows=200 loops=1)
              Recheck Cond: ((topic = 'military_end_use'::text) AND (jurisdiction = 'CN'::text) AND (published >= '2024-01-01'::date))
              Filter: is_designated
              Rows Removed by Filter: 2392
              Heap Blocks: exact=2470
              Buffers: shared hit=3400
              ->  BitmapAnd  (cost=1202.24..1202.24 rows=1540 width=0) (actual time=6.648..6.648 rows=0 loops=1)
                    Buffers: shared hit=130
                    ->  Bitmap Index Scan on documents_topic_idx  (cost=0.00..150.97 rows=17727 width=0) (actual time=1.371..1.371 rows=18285 loops=1)
                          Index Cond: (topic = 'military_end_use'::text)
                          Buffers: shared hit=19
                    ->  Bitmap Index Scan on documents_jurisdiction_idx  (cost=0.00..361.85 rows=42780 width=0) (actual time=1.633..1.633 rows=43196 loops=1)
                          Index Cond: (jurisdiction = 'CN'::text)
                          Buffers: shared hit=39
                    ->  Bitmap Index Scan on documents_published_idx  (cost=0.00..688.85 rows=81247 width=0) (actual time=2.968..2.968 rows=81269 loops=1)
                          Index Cond: (published >= '2024-01-01'::date)
                          Buffers: shared hit=72
Planning:
  Buffers: shared hit=1
Planning Time: 0.045 ms
Execution Time: 8.845 ms
```

**0.1% — exact**

```
Limit  (cost=26769.87..26771.05 rows=10 width=12) (actual time=20.060..25.372 rows=10 loops=1)
  Buffers: shared hit=25731
  ->  Gather Merge  (cost=26769.87..26781.58 rows=99 width=12) (actual time=20.058..25.369 rows=10 loops=1)
        Workers Planned: 3
        Workers Launched: 3
        Buffers: shared hit=25731
        ->  Sort  (cost=25769.83..25769.91 rows=33 width=12) (actual time=17.949..17.950 rows=8 loops=4)
              Sort Key: ((embedding <=> '[...384 dims...]'::vector))
              Sort Method: top-N heapsort  Memory: 25kB
              Buffers: shared hit=25731
              Worker 0:  Sort Method: top-N heapsort  Memory: 25kB
              Worker 1:  Sort Method: top-N heapsort  Memory: 25kB
              Worker 2:  Sort Method: top-N heapsort  Memory: 25kB
              ->  Parallel Seq Scan on documents  (cost=0.00..25769.11 rows=33 width=12) (actual time=0.399..17.887 rows=50 loops=4)
                    Filter: (is_designated AND (published >= '2024-01-01'::date) AND (jurisdiction = 'CN'::text) AND (topic = 'military_end_use'::text))
                    Rows Removed by Filter: 49950
                    Buffers: shared hit=25560
Planning:
  Buffers: shared hit=1
Planning Time: 0.107 ms
Execution Time: 25.389 ms
```

**0.01% — post_filter 10x**

```
Limit  (cost=408.60..453.20 rows=1 width=12) (actual time=2.380..2.380 rows=0 loops=1)
  Buffers: shared hit=2225
  ->  Subquery Scan on ann  (cost=408.60..453.20 rows=1 width=12) (actual time=2.380..2.380 rows=0 loops=1)
        Filter: ((ann.published >= '2025-01-01'::date) AND (ann.jurisdiction = 'TR'::text) AND (ann.topic = 'forced_labour'::text))
        Rows Removed by Filter: 100
        Buffers: shared hit=2225
        ->  Limit  (cost=408.60..451.45 rows=100 width=36) (actual time=1.825..2.374 rows=100 loops=1)
              Buffers: shared hit=2225
              ->  Index Scan using documents_embedding_hnsw on documents  (cost=408.60..86105.10 rows=200000 width=36) (actual time=1.824..2.367 rows=100 loops=1)
                    Order By: (embedding <=> '[...384 dims...]'::vector)
                    Buffers: shared hit=2225
Planning:
  Buffers: shared hit=1
Planning Time: 0.042 ms
Execution Time: 2.387 ms
```

**0.01% — pre_filter**

```
Limit  (cost=1012.21..1012.24 rows=10 width=12) (actual time=4.066..4.068 rows=10 loops=1)
  Buffers: shared hit=174
  ->  Sort  (cost=1012.21..1012.96 rows=298 width=12) (actual time=4.066..4.067 rows=10 loops=1)
        Sort Key: ((embedding <=> '[...384 dims...]'::vector))
        Sort Method: quicksort  Memory: 25kB
        Buffers: shared hit=174
        ->  Bitmap Heap Scan on documents  (cost=676.38..1005.78 rows=298 width=12) (actual time=4.006..4.062 rows=20 loops=1)
              Recheck Cond: ((jurisdiction = 'TR'::text) AND (topic = 'forced_labour'::text) AND (published >= '2025-01-01'::date))
              Heap Blocks: exact=20
              Buffers: shared hit=174
              ->  BitmapAnd  (cost=676.38..676.38 rows=298 width=0) (actual time=3.982..3.982 rows=0 loops=1)
                    Buffers: shared hit=74
                    ->  Bitmap Index Scan on documents_jurisdiction_idx  (cost=0.00..127.94 rows=15113 width=0) (actual time=0.792..0.792 rows=15002 loops=1)
                          Index Cond: (jurisdiction = 'TR'::text)
                          Buffers: shared hit=14
                    ->  Bitmap Index Scan on documents_topic_idx  (cost=0.00..139.87 rows=16393 width=0) (actual time=0.832..0.832 rows=16785 loops=1)
                          Index Cond: (topic = 'forced_labour'::text)
                          Buffers: shared hit=17
                    ->  Bitmap Index Scan on documents_published_idx  (cost=0.00..407.84 rows=48033 width=0) (actual time=1.946..1.946 rows=48273 loops=1)
                          Index Cond: (published >= '2025-01-01'::date)
                          Buffers: shared hit=43
Planning:
  Buffers: shared hit=1
Planning Time: 0.039 ms
Execution Time: 4.078 ms
```

**0.01% — exact**

```
Limit  (cost=26771.39..26772.57 rows=10 width=12) (actual time=19.102..23.665 rows=10 loops=1)
  Buffers: shared hit=25011
  ->  Gather Merge  (cost=26771.39..26805.47 rows=288 width=12) (actual time=19.101..23.660 rows=10 loops=1)
        Workers Planned: 3
        Workers Launched: 3
        Buffers: shared hit=25011
        ->  Sort  (cost=25771.35..25771.59 rows=96 width=12) (actual time=17.107..17.107 rows=4 loops=4)
              Sort Key: ((embedding <=> '[...384 dims...]'::vector))
              Sort Method: quicksort  Memory: 25kB
              Buffers: shared hit=25011
              Worker 0:  Sort Method: quicksort  Memory: 25kB
              Worker 1:  Sort Method: quicksort  Memory: 25kB
              Worker 2:  Sort Method: quicksort  Memory: 25kB
              ->  Parallel Seq Scan on documents  (cost=0.00..25769.27 rows=96 width=12) (actual time=3.508..17.006 rows=5 loops=4)
                    Filter: ((published >= '2025-01-01'::date) AND (jurisdiction = 'TR'::text) AND (topic = 'forced_labour'::text))
                    Rows Removed by Filter: 49995
                    Buffers: shared hit=24840
Planning:
  Buffers: shared hit=1
Planning Time: 0.079 ms
Execution Time: 23.677 ms
```

**Finding.** There is a clean crossover, and it sits between 1% and 0.1% selectivity. Above
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

## E3. Post-filter overfetch sweep

Filter: `jurisdiction = 'HK' AND topic = 'sanctions_evasion' AND published >= DATE '2025-01-01'` (0.9865% of the corpus).

Note `ef_search_used`: HNSW returns at most `ef_search` candidates, so a fetch depth above it is silently truncated. `ef_search` is raised to the fetch depth, which is part of what the overfetch actually costs.

| overfetch | fetch k | ef_search | recall@10 | p10 | min | starved | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 40 | 0.0021 | 0 | 0 | 1 | 1.916 | 9.999 |
| 5 | 50 | 50 | 0.0312 | 0 | 0 | 1 | 2.044 | 4.078 |
| 10 | 100 | 100 | 0.0563 | 0 | 0 | 0.9792 | 3.295 | 9.127 |
| 50 | 500 | 500 | 0.1938 | 0 | 0 | 0.7917 | 13.34 | 28.15 |
| 100 | 1000 | 1000 | 0.2354 | 0 | 0 | 0.7917 | 22.4 | 30.44 |

```sql
SELECT doc_id, distance
FROM (
    SELECT doc_id,
           jurisdiction, topic, published, is_designated, hops_to_designated,
           embedding <=> %(q)s AS distance
    FROM documents
    ORDER BY embedding <=> %(q)s
    LIMIT 100
) AS ann
WHERE jurisdiction = 'HK' AND topic = 'sanctions_evasion' AND published >= DATE '2025-01-01'
ORDER BY distance
LIMIT 10;
```

**EXPLAIN (ANALYZE, BUFFERS) - overfetch 100x**

```
Limit  (cost=2786.15..3220.24 rows=6 width=12) (actual time=14.798..14.801 rows=0 loops=1)
  Buffers: shared hit=12283
  ->  Subquery Scan on ann  (cost=2786.15..3220.24 rows=6 width=12) (actual time=14.797..14.801 rows=0 loops=1)
        Filter: ((ann.published >= '2025-01-01'::date) AND (ann.jurisdiction = 'HK'::text) AND (ann.topic = 'sanctions_evasion'::text))
        Rows Removed by Filter: 1000
        Buffers: shared hit=12283
        ->  Limit  (cost=2786.15..3202.74 rows=1000 width=36) (actual time=9.107..14.748 rows=1000 loops=1)
              Buffers: shared hit=12283
              ->  Index Scan using documents_embedding_hnsw on documents  (cost=2786.15..86105.10 rows=200000 width=36) (actual time=9.106..14.679 rows=1000 loops=1)
                    Order By: (embedding <=> '[...384 dims...]'::vector)
                    Buffers: shared hit=12283
Planning:
  Buffers: shared hit=1
Planning Time: 0.076 ms
Execution Time: 14.816 ms
```

**Finding.** Overfetch is a real but weak lever, and it does not rescue the 1% case. Recall@10
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

## E4. ef_search sweep

| filter | actual % | strategy | ef_search | recall@10 | p10 | starved | p50 ms | p95 ms | scan |
|---|---|---|---|---|---|---|---|---|---|
| 10% | 9.143 | pre_filter | 10 | 0.1063 | 0 | 0.875 | 0.748 | 1.51 | hnsw |
| 10% | 9.143 | post_filter 10x | 10 | 0.125 | 0 | 0.875 | 2.949 | 6.136 | hnsw |
| 10% | 9.143 | pre_filter | 40 | 0.1229 | 0 | 0.875 | 1.17 | 2.348 | hnsw |
| 10% | 9.143 | post_filter 10x | 40 | 0.125 | 0 | 0.875 | 2.927 | 5.512 | hnsw |
| 10% | 9.143 | pre_filter | 100 | 0.125 | 0 | 0.875 | 2.207 | 3.925 | hnsw |
| 10% | 9.143 | post_filter 10x | 100 | 0.125 | 0 | 0.875 | 2.954 | 5.848 | hnsw |
| 10% | 9.143 | pre_filter | 400 | 0.1479 | 0 | 0.8542 | 8.613 | 17.29 | hnsw |
| 10% | 9.143 | post_filter 10x | 400 | 0.125 | 0 | 0.875 | 9.929 | 16.52 | hnsw |
| 1% | 0.9865 | pre_filter | 10 | 0.0042 | 0 | 1 | 0.842 | 1.468 | hnsw |
| 1% | 0.9865 | post_filter 10x | 10 | 0.0563 | 0 | 0.9792 | 3.444 | 6.907 | hnsw |
| 1% | 0.9865 | pre_filter | 40 | 0.025 | 0 | 1 | 1.456 | 3.469 | hnsw |
| 1% | 0.9865 | post_filter 10x | 40 | 0.0563 | 0 | 0.9792 | 2.99 | 6.906 | hnsw |
| 1% | 0.9865 | pre_filter | 100 | 0.0563 | 0 | 0.9792 | 2.909 | 6.705 | hnsw |
| 1% | 0.9865 | post_filter 10x | 100 | 0.0563 | 0 | 0.9792 | 3.718 | 6.864 | hnsw |
| 1% | 0.9865 | pre_filter | 400 | 0.1833 | 0 | 0.875 | 7.606 | 14.52 | hnsw |
| 1% | 0.9865 | post_filter 10x | 400 | 0.0604 | 0 | 1 | 8.293 | 16.48 | hnsw |

**Finding.** `ef_search` cannot buy your way out of a correlated filter either. At 9.1%
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

## E5. pgvector iterative index scan

pgvector 0.8.6 supports `hnsw.iterative_scan` (added in 0.8.0). `relaxed_order` allows results to come back slightly out of distance order; `strict_order` preserves ordering at higher cost. `hnsw.max_scan_tuples` (default 20,000) caps the work.

| filter | actual % | rows | iterative_scan | recall@10 | p10 | min | starved | p50 ms | p95 ms | scan |
|---|---|---|---|---|---|---|---|---|---|---|
| 1% | 0.9865 | 1973 | off | 0.025 | 0 | 0 | 1 | 1.428 | 3.077 | hnsw |
| 1% | 0.9865 | 1973 | relaxed_order | 0.4438 | 0.1 | 0 | 0.0417 | 49.02 | 76.98 | hnsw |
| 1% | 0.9865 | 1973 | strict_order | 0.3979 | 0.1 | 0 | 0.0417 | 46.1 | 68.83 | hnsw |
| 0.1% | 0.1 | 200 | off | 1 | 1 | 1 | 0 | 9.719 | 14.45 | bitmap |
| 0.1% | 0.1 | 200 | relaxed_order | 1 | 1 | 1 | 0 | 9.467 | 14.01 | bitmap |
| 0.1% | 0.1 | 200 | strict_order | 1 | 1 | 1 | 0 | 10.55 | 16.45 | bitmap |
| 0.01% | 0.01 | 20 | off | 1 | 1 | 1 | 0 | 5.292 | 17.81 | bitmap |
| 0.01% | 0.01 | 20 | relaxed_order | 1 | 1 | 1 | 0 | 5.6 | 9.485 | bitmap |
| 0.01% | 0.01 | 20 | strict_order | 1 | 1 | 1 | 0 | 5.528 | 8.819 | bitmap |

**1% — relaxed_order**

```
Limit  (cost=204.81..973.19 rows=10 width=12) (actual time=83.175..98.085 rows=10 loops=1)
  Buffers: shared hit=42798
  ->  Index Scan using documents_embedding_hnsw on documents  (cost=204.81..87107.93 rows=1131 width=12) (actual time=83.174..98.081 rows=10 loops=1)
        Order By: (embedding <=> '[...384 dims...]'::vector)
        Filter: ((published >= '2025-01-01'::date) AND (jurisdiction = 'HK'::text) AND (topic = 'sanctions_evasion'::text))
        Rows Removed by Filter: 16931
        Buffers: shared hit=42798
Planning:
  Buffers: shared hit=1
Planning Time: 0.045 ms
Execution Time: 98.102 ms
```

**0.1% — relaxed_order**

```
Limit  (cost=2838.26..2838.29 rows=10 width=12) (actual time=8.533..8.535 rows=10 loops=1)
  Buffers: shared hit=3400
  ->  Sort  (cost=2838.26..2838.52 rows=102 width=12) (actual time=8.532..8.533 rows=10 loops=1)
        Sort Key: ((embedding <=> '[...384 dims...]'::vector))
        Sort Method: top-N heapsort  Memory: 25kB
        Buffers: shared hit=3400
        ->  Bitmap Heap Scan on documents  (cost=1202.24..2836.06 rows=102 width=12) (actual time=6.444..8.511 rows=200 loops=1)
              Recheck Cond: ((topic = 'military_end_use'::text) AND (jurisdiction = 'CN'::text) AND (published >= '2024-01-01'::date))
              Filter: is_designated
              Rows Removed by Filter: 2392
              Heap Blocks: exact=2470
              Buffers: shared hit=3400
              ->  BitmapAnd  (cost=1202.24..1202.24 rows=1540 width=0) (actual time=6.202..6.203 rows=0 loops=1)
                    Buffers: shared hit=130
                    ->  Bitmap Index Scan on documents_topic_idx  (cost=0.00..150.97 rows=17727 width=0) (actual time=0.894..0.894 rows=18285 loops=1)
                          Index Cond: (topic = 'military_end_use'::text)
                          Buffers: shared hit=19
                    ->  Bitmap Index Scan on documents_jurisdiction_idx  (cost=0.00..361.85 rows=42780 width=0) (actual time=1.654..1.654 rows=43196 loops=1)
                          Index Cond: (jurisdiction = 'CN'::text)
                          Buffers: shared hit=39
                    ->  Bitmap Index Scan on documents_published_idx  (cost=0.00..688.85 rows=81247 width=0) (actual time=2.992..2.992 rows=81269 loops=1)
                          Index Cond: (published >= '2024-01-01'::date)
                          Buffers: shared hit=72
Planning:
  Buffers: shared hit=1
Planning Time: 0.055 ms
Execution Time: 8.545 ms
```

**0.01% — relaxed_order**

```
Limit  (cost=1012.21..1012.24 rows=10 width=12) (actual time=5.741..5.744 rows=10 loops=1)
  Buffers: shared hit=174
  ->  Sort  (cost=1012.21..1012.96 rows=298 width=12) (actual time=5.740..5.741 rows=10 loops=1)
        Sort Key: ((embedding <=> '[...384 dims...]'::vector))
        Sort Method: quicksort  Memory: 25kB
        Buffers: shared hit=174
        ->  Bitmap Heap Scan on documents  (cost=676.38..1005.78 rows=298 width=12) (actual time=5.550..5.724 rows=20 loops=1)
              Recheck Cond: ((jurisdiction = 'TR'::text) AND (topic = 'forced_labour'::text) AND (published >= '2025-01-01'::date))
              Heap Blocks: exact=20
              Buffers: shared hit=174
              ->  BitmapAnd  (cost=676.38..676.38 rows=298 width=0) (actual time=5.466..5.467 rows=0 loops=1)
                    Buffers: shared hit=74
                    ->  Bitmap Index Scan on documents_jurisdiction_idx  (cost=0.00..127.94 rows=15113 width=0) (actual time=0.918..0.918 rows=15002 loops=1)
                          Index Cond: (jurisdiction = 'TR'::text)
                          Buffers: shared hit=14
                    ->  Bitmap Index Scan on documents_topic_idx  (cost=0.00..139.87 rows=16393 width=0) (actual time=0.967..0.967 rows=16785 loops=1)
                          Index Cond: (topic = 'forced_labour'::text)
                          Buffers: shared hit=17
                    ->  Bitmap Index Scan on documents_published_idx  (cost=0.00..407.84 rows=48033 width=0) (actual time=3.145..3.145 rows=48273 loops=1)
                          Index Cond: (published >= '2025-01-01'::date)
                          Buffers: shared hit=43
Planning:
  Buffers: shared hit=1
Planning Time: 0.150 ms
Execution Time: 5.786 ms
```

**Finding.** pgvector 0.8.6 is installed, so iterative scan ran. It is the only mitigation
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

## Decision rule

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

## Caveats

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

