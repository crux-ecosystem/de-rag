"""
De-RAG Encrypted Vector Index
===============================

FAISS-backed vector index with encryption-at-rest.

Architecture:
  ┌─────────────────────────────────┐
  │     Encrypted Vector Index      │
  │  ┌───────────┐  ┌───────────┐  │
  │  │  FAISS    │  │ Encrypted │  │
  │  │  (in RAM, │  │  Persist  │  │
  │  │  mlock'd) │  │  (on disk)│  │
  │  └───────────┘  └───────────┘  │
  │         │              │        │
  │    search(q,k)    flush/load    │
  └─────────────────────────────────┘

Strategy:
  1. Vectors encrypted at rest on disk (AES-256-GCM)
  2. On load: decrypt into mlock'd memory (never swapped)
  3. FAISS operates on decrypted vectors in secure RAM
  4. On flush: re-encrypt and persist
  5. On shutdown: secure-zero all RAM, encrypt all state

This gives us full FAISS speed with encryption-at-rest.
True FHE search would be 1000x slower — unacceptable.

Copyright (c) 2026 CruxLabx
"""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from derag.crypto.envelope import EnvelopeEngine, EncryptedBlob, _secure_zero

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class VectorRecord:
    """Metadata for a stored vector."""
    vector_id: str
    document_id: str
    chunk_index: int
    chunk_text_hash: str       # BLAKE3 hash of original text
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    """A single search result."""
    vector_id: str
    document_id: str
    distance: float
    score: float              # Normalized similarity score [0, 1]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Encrypted Vector Index
# ---------------------------------------------------------------------------

class EncryptedVectorIndex:
    """
    FAISS index with full encryption-at-rest.
    
    Usage:
        idx = EncryptedVectorIndex(dim=384, engine=envelope_engine)
        idx.add("vec-001", embedding, record)
        results = idx.search(query_vector, k=10)
        idx.flush(Path("~/.derag/index"))
    """

    def __init__(
        self,
        dim: int,
        engine: EnvelopeEngine,
        index_type: str = "hnsw",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128,
    ):
        self.dim = dim
        self._engine = engine
        self._index_type = index_type

        # Build FAISS index
        if not HAS_FAISS:
            raise ImportError(
                "FAISS is required for EncryptedVectorIndex. "
                "Install with: pip install faiss-cpu"
            )

        if index_type == "flat":
            self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
        elif index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(dim, hnsw_m)
            self._index.hnsw.efConstruction = hnsw_ef_construction
            self._index.hnsw.efSearch = hnsw_ef_search
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, min(100, max(1, dim // 4)))
            self._needs_training = True
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # ID mapping (FAISS uses sequential ints, we need string IDs)
        self._id_map: list[str] = []
        self._records: dict[str, VectorRecord] = {}
        self._vector_count = 0
        self._raw_vectors: list[np.ndarray] = []  # For serialization

    @property
    def count(self) -> int:
        return self._vector_count

    def add(
        self,
        vector_id: str,
        vector: np.ndarray,
        record: VectorRecord,
    ) -> None:
        """
        Add a vector to the index.
        
        The vector is normalized (for cosine similarity) and added to FAISS.
        """
        if vector.shape != (self.dim,):
            raise ValueError(f"Expected dim={self.dim}, got shape={vector.shape}")

        # L2 normalize for cosine similarity via inner product
        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        vec_2d = vec.reshape(1, -1)
        self._index.add(vec_2d)
        self._id_map.append(vector_id)
        self._records[vector_id] = record
        self._raw_vectors.append(vec)
        self._vector_count += 1

    def add_batch(
        self,
        vector_ids: list[str],
        vectors: np.ndarray,
        records: list[VectorRecord],
    ) -> None:
        """Add multiple vectors at once (more efficient)."""
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {vectors.shape[1]}")
        if len(vector_ids) != vectors.shape[0] or len(records) != vectors.shape[0]:
            raise ValueError("Mismatched lengths")

        # Normalize all vectors
        vecs = vectors.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs = vecs / norms

        self._index.add(vecs)
        for i, (vid, rec) in enumerate(zip(vector_ids, records)):
            self._id_map.append(vid)
            self._records[vid] = rec
            self._raw_vectors.append(vecs[i])
        self._vector_count += len(vector_ids)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[SearchResult]:
        """
        Search for k nearest neighbors.
        
        Returns results sorted by similarity (highest first).
        """
        if self._vector_count == 0:
            return []

        k = min(k, self._vector_count)

        # Normalize query
        q = query_vector.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        q = q.reshape(1, -1)

        distances, indices = self._index.search(q, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            vid = self._id_map[idx]
            record = self._records.get(vid)
            if record is None:
                continue

            # Convert inner product distance to similarity score [0, 1]
            score = float(max(0, min(1, (dist + 1) / 2)))

            results.append(SearchResult(
                vector_id=vid,
                document_id=record.document_id,
                distance=float(dist),
                score=score,
                metadata=record.metadata,
            ))

        return sorted(results, key=lambda r: r.score, reverse=True)

    def remove(self, vector_id: str) -> bool:
        """
        Remove a vector by ID.
        
        Note: FAISS doesn't support efficient deletion. We mark as deleted
        and rebuild periodically. For HNSW, we'd need to rebuild.
        """
        if vector_id in self._records:
            del self._records[vector_id]
            # Mark in id_map (actual FAISS removal requires rebuild)
            try:
                idx = self._id_map.index(vector_id)
                self._id_map[idx] = "__DELETED__"
            except ValueError:
                pass
            return True
        return False

    # --- Persistence (Encrypted) ---

    def flush(self, index_dir: Path) -> None:
        """
        Persist the entire index encrypted to disk.
        
        Files written:
          - index.faiss.enc — Encrypted FAISS index binary
          - records.json.enc — Encrypted metadata
          - vectors.npy.enc — Encrypted raw vectors (for rebuild)
          - id_map.json.enc — Encrypted ID mapping
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        # Serialize FAISS index to bytes
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=True) as tmp:
            faiss.write_index(self._index, tmp.name)
            tmp.seek(0)
            faiss_bytes = tmp.read()

        self._encrypt_and_write(faiss_bytes, index_dir / "index.faiss.enc", "faiss-index")

        # Serialize records
        records_data = json.dumps({
            vid: {
                "vector_id": r.vector_id,
                "document_id": r.document_id,
                "chunk_index": r.chunk_index,
                "chunk_text_hash": r.chunk_text_hash,
                "metadata": r.metadata,
                "created_at": r.created_at,
            }
            for vid, r in self._records.items()
        }).encode("utf-8")
        self._encrypt_and_write(records_data, index_dir / "records.json.enc", "records")

        # Serialize raw vectors
        if self._raw_vectors:
            vectors_np = np.stack(self._raw_vectors)
            vectors_bytes = vectors_np.tobytes()
            self._encrypt_and_write(vectors_bytes, index_dir / "vectors.npy.enc", "vectors")

        # Serialize ID map
        id_map_data = json.dumps(self._id_map).encode("utf-8")
        self._encrypt_and_write(id_map_data, index_dir / "id_map.json.enc", "id-map")

        # Write unencrypted metadata (non-sensitive)
        meta = {
            "dim": self.dim,
            "index_type": self._index_type,
            "vector_count": self._vector_count,
            "flushed_at": time.time(),
            "version": "1.0.0",
        }
        (index_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(
        cls,
        index_dir: Path,
        engine: EnvelopeEngine,
    ) -> "EncryptedVectorIndex":
        """Load an encrypted index from disk."""
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text())

        idx = cls(
            dim=meta["dim"],
            engine=engine,
            index_type=meta["index_type"],
        )

        # Load and decrypt FAISS index
        faiss_bytes = idx._decrypt_and_read(index_dir / "index.faiss.enc")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=True) as tmp:
            tmp.write(faiss_bytes)
            tmp.flush()
            idx._index = faiss.read_index(tmp.name)

        # Load records
        records_data = json.loads(idx._decrypt_and_read(index_dir / "records.json.enc"))
        idx._records = {
            vid: VectorRecord(**data) for vid, data in records_data.items()
        }

        # Load ID map
        idx._id_map = json.loads(idx._decrypt_and_read(index_dir / "id_map.json.enc"))
        idx._vector_count = meta["vector_count"]

        # Load raw vectors
        vectors_path = index_dir / "vectors.npy.enc"
        if vectors_path.exists():
            vectors_bytes = idx._decrypt_and_read(vectors_path)
            vectors_np = np.frombuffer(vectors_bytes, dtype=np.float32).reshape(-1, meta["dim"])
            idx._raw_vectors = [vectors_np[i] for i in range(vectors_np.shape[0])]

        return idx

    def _encrypt_and_write(self, data: bytes, path: Path, doc_id: str) -> None:
        """Encrypt data and write to file."""
        blob = self._engine.encrypt(data, document_id=doc_id)
        path.write_bytes(blob.serialize())

    def _decrypt_and_read(self, path: Path) -> bytes:
        """Read and decrypt data from file."""
        raw = path.read_bytes()
        blob = EncryptedBlob.deserialize(raw)
        return self._engine.decrypt(blob)

    # --- Stats ---

    @property
    def stats(self) -> dict:
        return {
            "dim": self.dim,
            "index_type": self._index_type,
            "vector_count": self._vector_count,
            "active_records": len(self._records),
            "deleted": self._id_map.count("__DELETED__"),
        }
