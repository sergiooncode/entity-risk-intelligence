-- Indexes are created AFTER the bulk load, not with the table.
--
-- Building the HNSW graph incrementally during a 200k-row COPY is roughly an
-- order of magnitude slower than building it once at the end, and it produces a
-- worse graph. Same story for the btrees, to a lesser degree.

-- The ANN index under test. Parameters fixed for the whole benchmark:
--   m = 16               max edges per node per layer
--   ef_construction = 64 candidate list size during build
--   vector_cosine_ops    matches the `<=>` operator used in every query
CREATE INDEX documents_embedding_hnsw
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- The scalar indexes the filter predicates can use. Whether the planner picks
-- these over the HNSW index (or over a seq scan) is the central question of E2.
CREATE INDEX documents_jurisdiction_idx ON documents (jurisdiction);
CREATE INDEX documents_topic_idx        ON documents (topic);
CREATE INDEX documents_published_idx    ON documents (published);

ANALYZE documents;
