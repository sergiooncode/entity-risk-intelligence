-- vsbench schema: one wide document table, one HNSW index, three btree indexes.
--
-- Deliberately minimal. Everything the benchmark measures has to be visible in
-- this file: there is one vector column, one ANN index, and the three scalar
-- indexes the filter predicates can use.

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS documents;

CREATE TABLE documents (
    doc_id              integer     PRIMARY KEY,
    entity_id           integer     NOT NULL,
    entity_name         text        NOT NULL,  -- surface form used in this doc
                                               -- (canonical name or an alias)
    text                text        NOT NULL,
    jurisdiction        text        NOT NULL,  -- CN RU HK AE SG TR KZ DE NL US
    topic               text        NOT NULL,  -- 7 values, see vsbench/corpus.py
    sector              text        NOT NULL,
    published           date        NOT NULL,  -- 2018-01-01 .. 2026-12-31
    is_designated       boolean     NOT NULL,
    hops_to_designated  smallint    NOT NULL,  -- 0-5, 99 = unconnected
    severity            real        NOT NULL,  -- 0..1 latent axis used to pick
                                               -- template + vocabulary register
    embedding           vector(384)
);
