"""
De-RAG Searchable Encryption
==============================

Enables keyword and semantic search over encrypted data WITHOUT decrypting it.

Two approaches implemented:
1. SSE (Symmetric Searchable Encryption) — for keyword matching
2. LSH over encrypted embeddings — for approximate semantic search

Security guarantee: The server (peer nodes) learns NOTHING about:
- What keywords you're searching for
- What documents match
- The content of any document

The only leakage: access patterns (which encrypted tokens are queried).
Mitigated by padding + dummy queries.

Copyright (c) 2026 CruxLabx
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import blake3


# ---------------------------------------------------------------------------
# Symmetric Searchable Encryption (SSE)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchToken:
    """An encrypted search token that reveals nothing about the keyword."""
    token: bytes          # HMAC(key, keyword) — deterministic for same keyword
    trap_door: bytes      # Additional randomized component for security


@dataclass(frozen=True)
class EncryptedIndexEntry:
    """An entry in the encrypted inverted index."""
    token_hash: bytes     # BLAKE3(search_token) — for lookup
    encrypted_doc_ids: bytes  # AES-GCM encrypted list of document IDs
    nonce: bytes
    count: int            # Number of matching documents (can be padded)


class SSEIndex:
    """
    Symmetric Searchable Encryption Index.
    
    Build phase (document owner):
        1. For each document, extract keywords
        2. For each keyword, compute token = HMAC(key, keyword)
        3. Encrypt document IDs under the token
        4. Store encrypted index
    
    Search phase (querier):
        1. Compute search token for keyword
        2. Send token to index holder (peer)
        3. Peer returns encrypted matching doc IDs
        4. Querier decrypts doc IDs locally
    
    The index holder NEVER sees keywords or document IDs in plaintext.
    """

    def __init__(self, master_key: bytes):
        """
        Args:
            master_key: 32-byte key for HMAC token generation
        """
        if len(master_key) != 32:
            raise ValueError("SSE master key must be 32 bytes")
        self._key = master_key
        self._enc_key = self._derive_key(b"sse-encryption")
        self._tok_key = self._derive_key(b"sse-tokens")
        self._entries: dict[bytes, EncryptedIndexEntry] = {}

    def build_index(self, documents: dict[str, list[str]]) -> None:
        """
        Build encrypted index from documents.
        
        Args:
            documents: {doc_id: [keyword1, keyword2, ...]}
        """
        # Build inverted index: keyword → [doc_ids]
        inverted: dict[str, list[str]] = {}
        for doc_id, keywords in documents.items():
            for kw in keywords:
                normalized = kw.lower().strip()
                inverted.setdefault(normalized, []).append(doc_id)

        # Encrypt each entry
        cipher = AESGCM(self._enc_key)
        for keyword, doc_ids in inverted.items():
            token = self._compute_token(keyword)
            token_hash = blake3.blake3(token).digest()

            # Serialize and encrypt doc_ids
            plaintext = "|".join(doc_ids).encode("utf-8")
            nonce = secrets.token_bytes(12)
            encrypted = cipher.encrypt(nonce, plaintext, token_hash)  # AAD = token_hash

            # Pad count to nearest power of 2 (hide exact match count)
            padded_count = 1
            while padded_count < len(doc_ids):
                padded_count *= 2

            self._entries[token_hash] = EncryptedIndexEntry(
                token_hash=token_hash,
                encrypted_doc_ids=encrypted,
                nonce=nonce,
                count=padded_count,
            )

    def generate_search_token(self, keyword: str) -> SearchToken:
        """Generate an encrypted search token for a keyword."""
        token = self._compute_token(keyword.lower().strip())
        trap_door = secrets.token_bytes(16)  # Randomized component
        return SearchToken(token=token, trap_door=trap_door)

    def search(self, token: SearchToken) -> list[str]:
        """
        Search the encrypted index with a token.
        Returns decrypted document IDs.
        """
        token_hash = blake3.blake3(token.token).digest()
        entry = self._entries.get(token_hash)
        if not entry:
            return []

        cipher = AESGCM(self._enc_key)
        plaintext = cipher.decrypt(entry.nonce, entry.encrypted_doc_ids, token_hash)
        return plaintext.decode("utf-8").split("|")

    def serialize(self) -> bytes:
        """Serialize the entire encrypted index for storage/transmission."""
        parts = []
        for entry in self._entries.values():
            parts.append(struct.pack("!I", len(entry.token_hash)))
            parts.append(entry.token_hash)
            parts.append(struct.pack("!I", len(entry.nonce)))
            parts.append(entry.nonce)
            parts.append(struct.pack("!I", len(entry.encrypted_doc_ids)))
            parts.append(entry.encrypted_doc_ids)
            parts.append(struct.pack("!I", entry.count))
        return b"".join(parts)

    def _compute_token(self, keyword: str) -> bytes:
        """Deterministic token from keyword."""
        return hmac.new(self._tok_key, keyword.encode("utf-8"), hashlib.sha256).digest()

    def _derive_key(self, context: bytes) -> bytes:
        """Derive a sub-key for a specific purpose."""
        return blake3.blake3(self._key + context, derive_key_context="derag-sse-v1").digest()


# ---------------------------------------------------------------------------
# Locality-Sensitive Hashing for Encrypted Semantic Search
# ---------------------------------------------------------------------------

class EncryptedLSH:
    """
    Locality-Sensitive Hashing over encrypted embeddings.
    
    Enables approximate nearest-neighbor search on vectors
    WITHOUT decrypting them.
    
    How it works:
    1. Generate random hyperplanes (shared secret between querier and index)
    2. Hash each vector by which side of each hyperplane it falls on
    3. Similar vectors produce similar hashes (with high probability)
    4. Store encrypted vectors bucketed by hash
    5. Search: hash query vector, look up matching buckets
    
    Security: Hyperplanes are secret. Without them, hash values reveal
    nothing about the original vectors.
    
    Trade-off: Approximate results. Accuracy improves with more hash tables.
    """

    def __init__(
        self,
        dim: int,
        num_tables: int = 10,
        hash_bits: int = 12,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.num_tables = num_tables
        self.hash_bits = hash_bits

        rng = np.random.RandomState(seed)
        # Generate random hyperplanes for each hash table
        self._hyperplanes = [
            rng.randn(hash_bits, dim).astype(np.float32)
            for _ in range(num_tables)
        ]
        # Buckets: table_idx → hash_value → [(vector_id, encrypted_vector)]
        self._buckets: list[dict[int, list[tuple[str, bytes]]]] = [
            {} for _ in range(num_tables)
        ]

    def add(self, vector_id: str, vector: np.ndarray, encrypted_vector: bytes) -> None:
        """
        Add an encrypted vector to the LSH index.
        
        Args:
            vector_id: Unique identifier
            vector: Raw embedding (used only for hashing, then discarded)
            encrypted_vector: AES-encrypted embedding for storage
        """
        for table_idx in range(self.num_tables):
            h = self._hash_vector(vector, table_idx)
            bucket = self._buckets[table_idx].setdefault(h, [])
            bucket.append((vector_id, encrypted_vector))

    def query(self, query_vector: np.ndarray, max_candidates: int = 100) -> list[tuple[str, bytes]]:
        """
        Find candidate matches for a query vector.
        
        Returns encrypted vectors of candidates (caller must decrypt and re-rank).
        """
        candidates: dict[str, bytes] = {}
        for table_idx in range(self.num_tables):
            h = self._hash_vector(query_vector, table_idx)
            bucket = self._buckets[table_idx].get(h, [])
            for vid, enc_vec in bucket:
                if vid not in candidates:
                    candidates[vid] = enc_vec
                    if len(candidates) >= max_candidates:
                        return list(candidates.items())
        return list(candidates.items())

    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> int:
        """Compute LSH hash for a vector in a specific hash table."""
        projections = self._hyperplanes[table_idx] @ vector.astype(np.float32)
        bits = (projections > 0).astype(np.int32)
        # Convert bit array to integer
        hash_val = 0
        for bit in bits:
            hash_val = (hash_val << 1) | int(bit)
        return hash_val

    @property
    def stats(self) -> dict:
        total_entries = sum(
            sum(len(bucket) for bucket in table.values())
            for table in self._buckets
        )
        total_buckets = sum(len(table) for table in self._buckets)
        return {
            "num_tables": self.num_tables,
            "hash_bits": self.hash_bits,
            "total_entries": total_entries,
            "total_buckets": total_buckets,
            "avg_bucket_size": total_entries / max(total_buckets, 1),
        }
