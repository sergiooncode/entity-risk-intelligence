"""
Database access. Raw SQL only - every statement the benchmark issues is
visible in this file or in strategies.py.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DSN = "postgresql://vsbench:vsbench@localhost:5433/vsbench"

ROOT = Path(__file__).resolve().parent.parent


def connect(dsn: str = DSN) -> psycopg.Connection:
    """
    Open a connection with server-side prepared statements disabled.

    psycopg auto-prepares a statement after its fifth execution, at which point
    the planner may switch to a generic plan. The benchmark runs the same query
    text hundreds of times and separately captures EXPLAIN output for it; if
    those two diverge the EXPLAIN plans printed in RESULTS.md would be lies.
    Disabling auto-prepare keeps every execution a fresh custom plan.
    """
    conn = psycopg.connect(dsn, autocommit=True, prepare_threshold=None)
    register_vector(conn)
    return conn


def wait_for_db(dsn: str = DSN, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with psycopg.connect(dsn, connect_timeout=3) as c:
                c.execute("SELECT 1")
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"database not reachable at {dsn}: {last}")


def pgvector_version(conn) -> str:
    row = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    return row[0] if row else "not installed"


def server_version(conn) -> str:
    return conn.execute("SHOW server_version").fetchone()[0]


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

COPY_COLUMNS = [
    "doc_id", "entity_id", "entity_name", "text", "jurisdiction", "topic",
    "sector", "published", "is_designated", "hops_to_designated",
    "severity", "embedding",
]

COPY_TYPES = [
    "int4", "int4", "text", "text", "text", "text",
    "text", "date", "bool", "int2",
    "float4", "vector",
]


def create_schema(conn) -> None:
    sql = (ROOT / "schema.sql").read_text()
    conn.execute(sql)


def create_indexes(conn) -> float:
    """Build the HNSW graph and the btrees. Returns wall-clock seconds."""
    sql = (ROOT / "indexes.sql").read_text()
    t0 = time.perf_counter()
    conn.execute(sql)
    return time.perf_counter() - t0


def ingest(conn, docs, vectors: np.ndarray, batch_log: int = 50_000) -> float:
    """
    Binary COPY of documents + embeddings.

    Binary rather than text format because 200k x 384 float32 rendered as
    decimal text is roughly 2.5 GB pushed through the socket; the binary path
    sends the 307 MB the vectors actually occupy.
    """
    import datetime as dt

    assert len(docs) == len(vectors), "corpus/embedding length mismatch"

    cols = ", ".join(COPY_COLUMNS)
    stmt = f"COPY documents ({cols}) FROM STDIN WITH (FORMAT BINARY)"

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        with cur.copy(stmt) as copy:
            copy.set_types(COPY_TYPES)
            for i, d in enumerate(docs):
                copy.write_row((
                    d.doc_id, d.entity_id, d.entity_name, d.text,
                    d.jurisdiction, d.topic, d.sector,
                    dt.date.fromisoformat(d.published),
                    d.is_designated, d.hops_to_designated,
                    d.severity, vectors[i],
                ))
                if batch_log and (i + 1) % batch_log == 0:
                    print(f"    copied {i + 1:,}/{len(docs):,}")
    return time.perf_counter() - t0


def table_is_loaded(conn, expected_rows: int) -> bool:
    row = conn.execute("""
        SELECT count(*) FROM information_schema.tables
        WHERE table_name = 'documents'
    """).fetchone()
    if not row or row[0] == 0:
        return False
    n = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    if n != expected_rows:
        return False
    idx = conn.execute("""
        SELECT count(*) FROM pg_indexes
        WHERE tablename = 'documents' AND indexname = 'documents_embedding_hnsw'
    """).fetchone()[0]
    return idx == 1


def index_sizes(conn) -> dict:
    rows = conn.execute("""
        SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid))
        FROM pg_stat_user_indexes
        WHERE relname = 'documents'
        ORDER BY indexrelname
    """).fetchall()
    out = {name: size for name, size in rows}
    out["documents (table)"] = conn.execute(
        "SELECT pg_size_pretty(pg_relation_size('documents'))"
    ).fetchone()[0]
    return out
