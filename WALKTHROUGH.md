# Hands-on: reproduce the whole result in ten minutes

Open one psql session and stay in it. Every step is one query.

```bash
docker exec -it vsbench-pg psql -U vsbench -d vsbench
```

```sql
\set seed 22   -- an AE / maritime document; we use its embedding as the query
```

Everything below searches for "documents like doc 22". Using an existing row as
the probe vector means no Python and no embedding step.

---

### 1. Unfiltered ANN uses the HNSW index

```sql
SET hnsw.ef_search = 40;

EXPLAIN (ANALYZE)
SELECT doc_id FROM documents
ORDER BY embedding <=> (SELECT embedding FROM documents WHERE doc_id = :seed)
LIMIT 10;
```

Look for `Index Scan using documents_embedding_hnsw`. ~2 ms.
**Say:** with no WHERE clause, pgvector walks the HNSW graph.

---

### 2. `ef_search` caps how many rows can ever come back

```sql
SELECT count(*) FROM (
  SELECT doc_id FROM documents
  ORDER BY embedding <=> (SELECT embedding FROM documents WHERE doc_id = :seed)
  LIMIT 100) t;
```

You asked for 100. You get **40**.
**Say:** HNSW returns at most `ef_search` candidates. `LIMIT` above that is
silently truncated — this is the first thing that surprises people.

---

### 3. A pre-filter returns *nothing*

`topic = 'military_end_use'` matches 9.1% of the corpus — 18,285 documents.

```sql
SELECT count(*) FROM (
  SELECT doc_id FROM documents
  WHERE topic = 'military_end_use'
  ORDER BY embedding <=> (SELECT embedding FROM documents WHERE doc_id = :seed)
  LIMIT 10) t;
```

**0 rows.** Not 10, not 3. Zero.
**Say:** the WHERE is applied *after* the graph walk, as a filter on the 40
candidates HNSW produced. None of them survived.

---

### 4. Why: local selectivity ≠ global selectivity

```sql
SET hnsw.ef_search = 1000;

SELECT count(*) FILTER (WHERE topic = 'military_end_use') AS survivors
FROM (SELECT topic FROM documents
      ORDER BY embedding <=> (SELECT embedding FROM documents WHERE doc_id = :seed)
      LIMIT 1000) t;
```

**6 out of 1000.** The filter matches 9.1% of the corpus but only 0.6% of this
query's nearest neighbours — 15× worse.
**Say:** this is the whole finding. Our seed doc is maritime; the filter asks
for military end-use. Because metadata correlates with content, the filter
deletes exactly the region the ANN walk was heading for. Any capacity planning
based on global selectivity is wrong.

---

### 5. The crossover: tighten the filter and the planner gives up on HNSW

```sql
SET hnsw.ef_search = 40;

EXPLAIN (ANALYZE)
SELECT doc_id FROM documents
WHERE jurisdiction = 'TR' AND topic = 'forced_labour'
  AND published >= DATE '2025-01-01'          -- 20 rows, 0.01%
ORDER BY embedding <=> (SELECT embedding FROM documents WHERE doc_id = :seed)
LIMIT 10;
```

Now the plan is `BitmapAnd` over the three btree indexes → `Bitmap Heap Scan` →
`Sort`. No HNSW. ~17 ms, and the answer is **exact**.
**Say:** below roughly 0.1% selectivity the planner correctly decides that
fetching all matching rows and sorting them beats an approximate graph walk.
Recall goes to 1.0 for free.

---

### 6. Post-filter overfetch helps, but does not rescue you

```sql
-- fetch 10, keep the ones that match
SET hnsw.ef_search = 40;
SELECT count(*) FROM (
  SELECT doc_id FROM (
    SELECT doc_id, topic FROM documents
    ORDER BY embedding <=> (SELECT embedding FROM documents WHERE doc_id = :seed)
    LIMIT 10) ann
  WHERE topic = 'military_end_use' LIMIT 10) t;

-- fetch 1000, keep the ones that match
SET hnsw.ef_search = 1000;
SELECT count(*) FROM (
  SELECT doc_id FROM (
    SELECT doc_id, topic FROM documents
    ORDER BY embedding <=> (SELECT embedding FROM documents WHERE doc_id = :seed)
    LIMIT 1000) ann
  WHERE topic = 'military_end_use' LIMIT 10) t;
```

1× → **0 rows**. 100× → **6 rows**. Still not 10, at 100× the cost.
**Say:** overfetch is a linear fix for an exponential problem. The benchmark
(E3) shows 100× overfetch reaching only 0.24 recall at 1% selectivity.

---

### 7. `hnsw.iterative_scan` is the actual fix

```sql
SET hnsw.ef_search = 40;
SET hnsw.iterative_scan = relaxed_order;   -- pgvector 0.8+

SELECT count(*) FROM (
  SELECT doc_id FROM documents
  WHERE topic = 'military_end_use'
  ORDER BY embedding <=> (SELECT embedding FROM documents WHERE doc_id = :seed)
  LIMIT 10) t;
```

**10 rows.** Same query that returned 0 in step 3.
**Say:** iterative scan resumes the graph walk when the filter rejects too much,
instead of stopping at `ef_search`. It fixes starvation — but E5 shows it costs
~34× latency (1.4 ms → 49 ms) and still only reaches 0.44 recall, which is worse
*and* slower than the 30 ms exact scan. Knowing that is the point.

```sql
RESET hnsw.iterative_scan;
```

---

## The one-sentence version

A metadata filter does not remove a random 9% of your corpus — it removes a
*correlated* 9%, concentrated exactly where the query was looking. So the useful
question is never "how selective is this filter?" but "how selective is it
inside the query's neighbourhood?"
