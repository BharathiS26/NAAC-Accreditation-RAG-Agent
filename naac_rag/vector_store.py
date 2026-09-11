"""
Vector store for NAAC metric chunks.
Uses sentence-transformers for embeddings and FAISS for retrieval.
Index is built once and persisted to disk.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Tuple

from sentence_transformers import SentenceTransformer
import faiss

from parser import MetricChunk, parse_manual

_INDEX_PATH = os.path.join(os.path.dirname(__file__), "naac_index.faiss")
_META_PATH = os.path.join(os.path.dirname(__file__), "naac_meta.pkl")
_MODEL_NAME = "all-MiniLM-L6-v2"

_embedder: SentenceTransformer = None
_index: faiss.IndexFlatIP = None
_chunks: List[MetricChunk] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_MODEL_NAME)
    return _embedder


def _embed(texts: List[str]) -> np.ndarray:
    model = _get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype="float32")


def build_index(pdf_path: str, force: bool = False) -> None:
    """Parse the PDF and build the FAISS index. Skips if already built."""
    if not force and os.path.exists(_INDEX_PATH) and os.path.exists(_META_PATH):
        return

    print(f"[VectorStore] Building index from {pdf_path} ...")
    chunks = parse_manual(pdf_path)
    if not chunks:
        raise RuntimeError("No metric chunks extracted from PDF.")

    # Embed enriched text: prepend metric number and KI number as keywords
    # so numeric queries like "1.1.2" and "Key Indicator 1.1" retrieve correctly.
    def _enrich(c):
        return (
            f"Metric {c.metric_num} "
            f"Key Indicator {c.ki_num} {c.ki_name} "
            f"Criterion {c.criterion_num} {c.criterion_name} "
            f"weight {c.weight or 'unspecified'} "
            f"{c.metric_type} metric\n"
            + c.full_text()
        )
    texts = [_enrich(c) for c in chunks]
    vecs = _embed(texts)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, _INDEX_PATH)
    with open(_META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[VectorStore] Indexed {len(chunks)} metric chunks.")


def _load() -> Tuple[faiss.IndexFlatIP, List[MetricChunk]]:
    global _index, _chunks
    if _index is None or _chunks is None:
        if not os.path.exists(_INDEX_PATH) or not os.path.exists(_META_PATH):
            pdf = os.environ.get("PDF_PATH", "")
            if not pdf or not os.path.exists(pdf):
                raise FileNotFoundError(
                    f"FAISS index not found and PDF_PATH not set. "
                    f"Run `python vector_store.py` first or set PDF_PATH."
                )
            build_index(pdf)
        _index = faiss.read_index(_INDEX_PATH)
        with open(_META_PATH, "rb") as f:
            _chunks = pickle.load(f)
    return _index, _chunks


def _keyword_boost(query: str, chunks: List[MetricChunk]) -> List[int]:
    """
    Return indices of chunks that contain explicit numeric references
    from the query (e.g. '1.1', '1.1.2', 'KI 1.1').
    Used to surface exact-match hits that semantic search might miss.
    """
    import re
    # Extract all x.y and x.y.z patterns from the query
    patterns = re.findall(r"\b(\d+\.\d+(?:\.\d+)?)\b", query)
    if not patterns:
        return []
    boosted = []
    for i, c in enumerate(chunks):
        for p in patterns:
            if p in c.metric_num or p in c.ki_num:
                boosted.append(i)
                break
    return boosted


def retrieve(query: str, top_k: int = 3) -> List[MetricChunk]:
    """Return the top_k most relevant MetricChunk objects for a query.
    Combines semantic search with keyword boosting for numeric references.
    """
    index, chunks = _load()
    q_vec = _embed([query])
    # Retrieve more candidates then re-rank with keyword boost at top
    k_large = min(top_k * 6, len(chunks))
    scores, indices = index.search(q_vec, k_large)

    keyword_hits = _keyword_boost(query, chunks)

    # Build ordered result: keyword hits first, then semantic, deduplicated
    seen = set()
    ordered = []
    for idx in keyword_hits:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    for idx in indices[0]:
        if 0 <= idx < len(chunks) and idx not in seen:
            seen.add(idx)
            ordered.append(idx)

    return [chunks[i] for i in ordered[:top_k]]


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    pdf = os.environ.get("PDF_PATH", "")
    if not pdf or not os.path.exists(pdf):
        print(f"PDF not found: {pdf!r}")
        sys.exit(1)
    build_index(pdf, force=True)
    hits = retrieve("What is required under Key Indicator 1.1?")
    for h in hits:
        print("---")
        print(h.metadata_header())
        print(h.text[:300])
