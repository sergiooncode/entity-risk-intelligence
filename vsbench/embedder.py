"""
Embedding behind a swappable protocol, with a disk cache.

The benchmark never touches sentence-transformers directly: it asks an
`Embedder` for vectors. Swapping in a different model (a bigger local model, an
API model) is a matter of writing another class with the same three members.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    """Anything that turns text into unit-norm float32 vectors."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        """Return an (len(texts), dim) float32 array of unit-norm vectors."""
        ...


class MiniLMEmbedder:
    """
    sentence-transformers/all-MiniLM-L6-v2. 384 dimensions, local, free.

    Vectors are L2-normalised at source. That is not required by
    vector_cosine_ops - pgvector normalises internally for cosine distance -
    but it makes the numpy-side ground-truth checks a plain dot product and
    removes any doubt about whether two distance computations agree.
    """

    name = "all-MiniLM-L6-v2"
    dim = 384

    def __init__(self, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        if device is None:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self._model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )

    def encode(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 5000,
        )
        return np.ascontiguousarray(vecs, dtype=np.float32)


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------

def _fingerprint(embedder_name: str, texts: Sequence[str]) -> str:
    """
    Hash the model name, the corpus size and the full text stream. Any change
    to the generator invalidates the cache, so a stale 200k-row .npy can never
    be silently paired with different documents.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(embedder_name.encode())
    h.update(str(len(texts)).encode())
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()


def embed_cached(embedder: Embedder, texts: Sequence[str], cache_dir: Path,
                 tag: str, batch_size: int = 256) -> np.ndarray:
    """Embed `texts`, reusing a cached .npy when the fingerprint matches."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(embedder.name, texts)
    path = cache_dir / f"emb_{tag}_{fp}.npy"

    if path.exists():
        vecs = np.load(path)
        if vecs.shape == (len(texts), embedder.dim):
            print(f"  embeddings: cache hit  {path.name}")
            return vecs
        print(f"  embeddings: cache shape mismatch, recomputing")

    print(f"  embeddings: computing {len(texts):,} x {embedder.dim} "
          f"on {getattr(embedder, 'device', '?')}")
    vecs = embedder.encode(texts, batch_size=batch_size)
    np.save(path, vecs)
    print(f"  embeddings: wrote {path.name} ({vecs.nbytes / 1e6:.0f} MB)")
    return vecs
