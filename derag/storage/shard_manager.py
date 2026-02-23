"""
De-RAG Shard Manager — Shamir's Secret Sharing
================================================

Splits encrypted data into N shards where any K can reconstruct.
Individual shards are cryptographically meaningless.

This is DIFFERENT from simple replication:
  - Replication: N copies of the same data (compromise 1 = compromise all)
  - Shamir SSS: N shares of a secret (compromise K-1 = learn NOTHING)

Implementation uses GF(2^8) arithmetic for byte-level splitting.
Each byte of the secret is split independently using a random
polynomial of degree K-1 evaluated at N distinct points.

References:
  - Shamir, "How to Share a Secret" (1979)
  - GF(2^8) implementation compatible with SSSS standard

Copyright (c) 2026 CruxLabx
"""

from __future__ import annotations

import os
import secrets
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import blake3


# ---------------------------------------------------------------------------
# GF(2^8) Arithmetic — Galois Field for Shamir's Secret Sharing
# ---------------------------------------------------------------------------

# Irreducible polynomial: x^8 + x^4 + x^3 + x + 1 (0x11b)
_GF_POLY = 0x11b

# Precomputed tables for GF(2^8) multiply and inverse
_EXP_TABLE = [0] * 512
_LOG_TABLE = [0] * 256

def _init_gf_tables() -> None:
    """Initialize GF(2^8) exp and log lookup tables."""
    x = 1
    for i in range(255):
        _EXP_TABLE[i] = x
        _LOG_TABLE[x] = i
        x <<= 1
        if x & 0x100:
            x ^= _GF_POLY
    # Fill upper half of exp table for convenience
    for i in range(255, 512):
        _EXP_TABLE[i] = _EXP_TABLE[i - 255]

_init_gf_tables()


def gf_mul(a: int, b: int) -> int:
    """Multiply two elements in GF(2^8)."""
    if a == 0 or b == 0:
        return 0
    return _EXP_TABLE[_LOG_TABLE[a] + _LOG_TABLE[b]]


def gf_inv(a: int) -> int:
    """Multiplicative inverse in GF(2^8)."""
    if a == 0:
        raise ZeroDivisionError("No inverse for 0 in GF(2^8)")
    return _EXP_TABLE[255 - _LOG_TABLE[a]]


def gf_div(a: int, b: int) -> int:
    """Division in GF(2^8)."""
    if b == 0:
        raise ZeroDivisionError("Division by 0 in GF(2^8)")
    if a == 0:
        return 0
    return _EXP_TABLE[(_LOG_TABLE[a] + 255 - _LOG_TABLE[b]) % 255]


# ---------------------------------------------------------------------------
# Shamir's Secret Sharing Core
# ---------------------------------------------------------------------------

def _eval_polynomial(coeffs: list[int], x: int) -> int:
    """Evaluate polynomial at x in GF(2^8). coeffs[0] is the secret."""
    result = 0
    for coeff in reversed(coeffs):
        result = gf_mul(result, x) ^ coeff
    return result


def _split_byte(secret_byte: int, n: int, k: int) -> list[tuple[int, int]]:
    """
    Split a single byte using Shamir's Secret Sharing.
    
    Args:
        secret_byte: The byte to split (0-255)
        n: Number of shares
        k: Threshold for reconstruction
    
    Returns:
        List of (x, y) pairs where x is the share index (1-based)
    """
    # Random polynomial of degree k-1 with secret as constant term
    coeffs = [secret_byte] + [secrets.randbelow(256) for _ in range(k - 1)]
    
    shares = []
    for i in range(1, n + 1):
        y = _eval_polynomial(coeffs, i)
        shares.append((i, y))
    return shares


def _reconstruct_byte(shares: list[tuple[int, int]]) -> int:
    """
    Reconstruct a byte from k or more shares using Lagrange interpolation.
    """
    k = len(shares)
    secret = 0
    
    for i in range(k):
        xi, yi = shares[i]
        numerator = 1
        denominator = 1
        
        for j in range(k):
            if i == j:
                continue
            xj, _ = shares[j]
            numerator = gf_mul(numerator, xj)       # xj (we evaluate at x=0)
            denominator = gf_mul(denominator, xi ^ xj)  # (xi - xj) = xi ^ xj in GF(2^8)
        
        lagrange = gf_mul(yi, gf_div(numerator, denominator))
        secret ^= lagrange
    
    return secret


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shard:
    """A single shard of a secret-shared document."""
    shard_id: str
    document_id: str
    shard_index: int          # 1-based (the x-coordinate in SSS)
    total_shards: int         # N
    threshold: int            # K
    data: bytes               # The shard bytes (y-coordinates)
    data_hash: str            # BLAKE3 hash for integrity
    original_size: int        # Size of original data
    created_at: float

    def serialize(self) -> bytes:
        """Serialize shard for network transmission."""
        header = struct.pack(
            "!BBBBI",
            self.shard_index,
            self.total_shards,
            self.threshold,
            0,  # reserved
            self.original_size,
        )
        doc_id_bytes = self.document_id.encode("utf-8")
        shard_id_bytes = self.shard_id.encode("utf-8")
        hash_bytes = self.data_hash.encode("utf-8")

        parts = [
            b"SHRD",                                       # Magic
            header,                                         # 8 bytes
            struct.pack("!H", len(shard_id_bytes)),
            shard_id_bytes,
            struct.pack("!H", len(doc_id_bytes)),
            doc_id_bytes,
            struct.pack("!H", len(hash_bytes)),
            hash_bytes,
            struct.pack("!d", self.created_at),            # 8 bytes
            struct.pack("!I", len(self.data)),
            self.data,
        ]
        return b"".join(parts)

    @classmethod
    def deserialize(cls, raw: bytes) -> "Shard":
        """Deserialize shard from network data."""
        if raw[:4] != b"SHRD":
            raise ValueError("Invalid shard magic bytes")

        offset = 4
        shard_index, total_shards, threshold, _, original_size = struct.unpack(
            "!BBBBI", raw[offset:offset + 8]
        )
        offset += 8

        sid_len = struct.unpack("!H", raw[offset:offset + 2])[0]
        offset += 2
        shard_id = raw[offset:offset + sid_len].decode("utf-8")
        offset += sid_len

        did_len = struct.unpack("!H", raw[offset:offset + 2])[0]
        offset += 2
        document_id = raw[offset:offset + did_len].decode("utf-8")
        offset += did_len

        hash_len = struct.unpack("!H", raw[offset:offset + 2])[0]
        offset += 2
        data_hash = raw[offset:offset + hash_len].decode("utf-8")
        offset += hash_len

        created_at = struct.unpack("!d", raw[offset:offset + 8])[0]
        offset += 8

        data_len = struct.unpack("!I", raw[offset:offset + 4])[0]
        offset += 4
        data = raw[offset:offset + data_len]

        return cls(
            shard_id=shard_id,
            document_id=document_id,
            shard_index=shard_index,
            total_shards=total_shards,
            threshold=threshold,
            data=data,
            data_hash=data_hash,
            original_size=original_size,
            created_at=created_at,
        )


# ---------------------------------------------------------------------------
# Shard Manager
# ---------------------------------------------------------------------------

class ShardManager:
    """
    Manages data sharding using Shamir's Secret Sharing.
    
    Usage:
        sm = ShardManager(n=5, k=3)
        shards = sm.split(encrypted_data, doc_id="doc-001")
        # Distribute shards to peers...
        
        # Later, with any 3+ shards:
        recovered = sm.reconstruct(shards[:3])
        assert recovered == encrypted_data
    """

    def __init__(self, n: int = 5, k: int = 3):
        """
        Args:
            n: Total number of shards to create
            k: Minimum shards needed to reconstruct (threshold)
        """
        if k > n:
            raise ValueError(f"Threshold k={k} must be <= total n={n}")
        if n > 255:
            raise ValueError(f"Maximum 255 shards (GF(2^8) limit), got n={n}")
        if k < 2:
            raise ValueError(f"Threshold must be >= 2, got k={k}")
        self.n = n
        self.k = k

    def split(self, data: bytes, document_id: str) -> list[Shard]:
        """
        Split data into n shards using Shamir's Secret Sharing.
        
        Time complexity: O(len(data) * n * k)
        Space complexity: O(len(data) * n)
        """
        original_size = len(data)
        
        # Initialize shard data arrays
        shard_data = [bytearray() for _ in range(self.n)]

        # Split each byte independently
        for byte_val in data:
            shares = _split_byte(byte_val, self.n, self.k)
            for i, (_, y) in enumerate(shares):
                shard_data[i].append(y)

        # Create Shard objects
        now = time.time()
        shards = []
        for i in range(self.n):
            shard_bytes = bytes(shard_data[i])
            shards.append(Shard(
                shard_id=f"shard-{uuid.uuid4().hex[:12]}",
                document_id=document_id,
                shard_index=i + 1,  # 1-based for SSS
                total_shards=self.n,
                threshold=self.k,
                data=shard_bytes,
                data_hash=blake3.blake3(shard_bytes).hexdigest(),
                original_size=original_size,
                created_at=now,
            ))

        return shards

    def reconstruct(self, shards: list[Shard]) -> bytes:
        """
        Reconstruct data from k or more shards.
        
        Raises ValueError if shards are from different documents
        or if insufficient shards are provided.
        """
        if not shards:
            raise ValueError("No shards provided")

        # Validate shards
        doc_ids = set(s.document_id for s in shards)
        if len(doc_ids) > 1:
            raise ValueError(f"Shards from multiple documents: {doc_ids}")

        threshold = shards[0].threshold
        if len(shards) < threshold:
            raise ValueError(
                f"Need at least {threshold} shards for reconstruction, got {len(shards)}"
            )

        original_size = shards[0].original_size

        # Verify shard integrity
        for shard in shards:
            actual_hash = blake3.blake3(shard.data).hexdigest()
            if actual_hash != shard.data_hash:
                raise ValueError(
                    f"Shard {shard.shard_id} integrity check failed: "
                    f"expected {shard.data_hash}, got {actual_hash}"
                )

        # Use only threshold number of shards (more doesn't help)
        active_shards = shards[:threshold]

        # Reconstruct byte by byte
        result = bytearray()
        for byte_idx in range(original_size):
            shares = [
                (s.shard_index, s.data[byte_idx])
                for s in active_shards
            ]
            secret_byte = _reconstruct_byte(shares)
            result.append(secret_byte)

        return bytes(result)

    def verify_shard(self, shard: Shard) -> bool:
        """Verify a shard's integrity."""
        actual_hash = blake3.blake3(shard.data).hexdigest()
        return actual_hash == shard.data_hash

    @property
    def config(self) -> dict:
        return {
            "total_shards": self.n,
            "threshold": self.k,
            "redundancy_factor": self.n / self.k,
        }
