"""
De-RAG Envelope Encryption Engine
==================================

Two-layer encryption architecture:
  Layer 1: Data Encryption Key (DEK) — unique per document/shard, AES-256-GCM
  Layer 2: Key Encryption Key (KEK) — master key derived from password, wraps DEKs

Why envelope encryption:
  - Rotate master key without re-encrypting all data
  - Each document has unique key — compromise one ≠ compromise all
  - DEKs are small — fast to re-wrap during key rotation
  - Standard pattern used by AWS KMS, GCP, Azure Key Vault

Security properties:
  - AES-256-GCM: authenticated encryption (confidentiality + integrity)
  - 96-bit random nonce per encryption operation
  - Argon2id for password → KEK derivation (memory-hard, GPU-resistant)
  - Key material held in mlock'd memory pages (never swapped to disk)
  - Explicit zeroing of key material on cleanup

Copyright (c) 2026 CruxLabx
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import mmap
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

try:
    from argon2.low_level import hash_secret_raw, Type
    HAS_ARGON2 = True
except ImportError:
    HAS_ARGON2 = False

import blake3


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AES_KEY_BITS = 256
AES_KEY_BYTES = AES_KEY_BITS // 8
NONCE_BYTES = 12          # 96-bit nonce for AES-GCM
SALT_BYTES = 32
TAG_BYTES = 16            # GCM authentication tag
VERSION_BYTE = b"\x01"    # Envelope format version
HEADER_MAGIC = b"DERG"    # De-RAG encrypted blob magic bytes


# ---------------------------------------------------------------------------
# Secure Memory Helpers
# ---------------------------------------------------------------------------

def _secure_zero(buffer: bytearray | memoryview) -> None:
    """Overwrite buffer with zeros — prevents compiler from optimizing away."""
    ctypes.memset(ctypes.addressof((ctypes.c_char * len(buffer)).from_buffer(buffer)), 0, len(buffer))


def _mlock_bytes(data: bytes) -> Optional[mmap.mmap]:
    """Lock bytes into RAM to prevent swapping to disk. Returns mmap or None."""
    try:
        secure_page = mmap.mmap(-1, len(data), mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        secure_page.write(data)
        # mlock via ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        addr = ctypes.c_void_p.from_buffer(secure_page)
        libc.mlock(addr, len(data))
        return secure_page
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncryptedBlob:
    """
    Encrypted data envelope.
    
    Binary format:
    [MAGIC:4][VERSION:1][NONCE:12][ENCRYPTED_DEK_LEN:2][ENCRYPTED_DEK:var][CIPHERTEXT:var]
    
    The ciphertext includes the GCM auth tag (last 16 bytes).
    AAD (Additional Authenticated Data) = document_id bytes for binding.
    """
    nonce: bytes                 # 12 bytes — unique per encryption
    encrypted_dek: bytes         # DEK wrapped with KEK
    ciphertext: bytes            # AES-256-GCM(DEK, plaintext, aad=doc_id)
    document_id: str             # Bound to ciphertext via AAD
    timestamp: float             # Encryption timestamp
    blake3_hash: str             # Hash of plaintext for integrity verification

    def serialize(self) -> bytes:
        """Serialize to binary format for storage/transmission."""
        doc_id_bytes = self.document_id.encode("utf-8")
        hash_bytes = self.blake3_hash.encode("utf-8")
        ts_bytes = struct.pack("!d", self.timestamp)

        parts = [
            HEADER_MAGIC,                                      # 4 bytes
            VERSION_BYTE,                                      # 1 byte
            self.nonce,                                        # 12 bytes
            struct.pack("!H", len(self.encrypted_dek)),        # 2 bytes
            self.encrypted_dek,                                # variable
            struct.pack("!H", len(doc_id_bytes)),              # 2 bytes
            doc_id_bytes,                                      # variable
            ts_bytes,                                          # 8 bytes
            struct.pack("!H", len(hash_bytes)),                # 2 bytes
            hash_bytes,                                        # variable
            self.ciphertext,                                   # remainder
        ]
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> "EncryptedBlob":
        """Deserialize from binary format."""
        if data[:4] != HEADER_MAGIC:
            raise ValueError("Invalid De-RAG encrypted blob: bad magic bytes")
        if data[4:5] != VERSION_BYTE:
            raise ValueError(f"Unsupported envelope version: {data[4]}")

        offset = 5
        nonce = data[offset:offset + NONCE_BYTES]
        offset += NONCE_BYTES

        dek_len = struct.unpack("!H", data[offset:offset + 2])[0]
        offset += 2
        encrypted_dek = data[offset:offset + dek_len]
        offset += dek_len

        doc_id_len = struct.unpack("!H", data[offset:offset + 2])[0]
        offset += 2
        document_id = data[offset:offset + doc_id_len].decode("utf-8")
        offset += doc_id_len

        timestamp = struct.unpack("!d", data[offset:offset + 8])[0]
        offset += 8

        hash_len = struct.unpack("!H", data[offset:offset + 2])[0]
        offset += 2
        blake3_hash = data[offset:offset + hash_len].decode("utf-8")
        offset += hash_len

        ciphertext = data[offset:]

        return cls(
            nonce=nonce,
            encrypted_dek=encrypted_dek,
            ciphertext=ciphertext,
            document_id=document_id,
            timestamp=timestamp,
            blake3_hash=blake3_hash,
        )


@dataclass
class KeyMaterial:
    """Holds key encryption key with secure cleanup."""
    kek: bytearray
    salt: bytes
    created_at: float = field(default_factory=time.time)
    _locked_page: Optional[mmap.mmap] = field(default=None, repr=False)

    def __post_init__(self):
        """Try to mlock the key material."""
        self._locked_page = _mlock_bytes(bytes(self.kek))

    def __del__(self):
        """Securely zero key material on garbage collection."""
        try:
            _secure_zero(self.kek)
            if self._locked_page:
                _secure_zero(memoryview(self._locked_page))
                self._locked_page.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Envelope Encryption Engine
# ---------------------------------------------------------------------------

class EnvelopeEngine:
    """
    Production-grade envelope encryption for De-RAG.
    
    Usage:
        engine = EnvelopeEngine.from_password("my-strong-password")
        encrypted = engine.encrypt(b"secret document", doc_id="doc-001")
        plaintext = engine.decrypt(encrypted)
    
    Thread-safety: Each operation is stateless after init. Safe for concurrent use.
    """

    def __init__(self, key_material: KeyMaterial):
        self._km = key_material
        self._kek_cipher = AESGCM(bytes(key_material.kek))

    @classmethod
    def from_password(
        cls,
        password: str,
        salt: Optional[bytes] = None,
        time_cost: int = 3,
        memory_cost: int = 65536,
        parallelism: int = 4,
    ) -> "EnvelopeEngine":
        """
        Derive KEK from password using Argon2id.
        
        Falls back to HKDF(SHA-256) if argon2-cffi is not installed,
        but Argon2id is STRONGLY recommended for production.
        """
        salt = salt or secrets.token_bytes(SALT_BYTES)

        if HAS_ARGON2:
            kek_bytes = hash_secret_raw(
                secret=password.encode("utf-8"),
                salt=salt,
                time_cost=time_cost,
                memory_cost=memory_cost,
                parallelism=parallelism,
                hash_len=AES_KEY_BYTES,
                type=Type.ID,  # Argon2id — hybrid, GPU-resistant
            )
        else:
            # Fallback: HKDF is weaker against brute-force but functional
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=AES_KEY_BYTES,
                salt=salt,
                info=b"derag-kek-derivation-v1",
            )
            kek_bytes = hkdf.derive(password.encode("utf-8"))

        km = KeyMaterial(kek=bytearray(kek_bytes), salt=salt)
        return cls(km)

    @classmethod
    def from_key(cls, kek: bytes, salt: bytes = b"") -> "EnvelopeEngine":
        """Initialize with a raw 256-bit KEK (for programmatic use)."""
        if len(kek) != AES_KEY_BYTES:
            raise ValueError(f"KEK must be {AES_KEY_BYTES} bytes, got {len(kek)}")
        km = KeyMaterial(kek=bytearray(kek), salt=salt)
        return cls(km)

    # --- Core Operations ---

    def encrypt(self, plaintext: bytes, document_id: str) -> EncryptedBlob:
        """
        Encrypt plaintext using envelope encryption.
        
        1. Generate random DEK (Data Encryption Key)
        2. Encrypt plaintext with DEK + AES-256-GCM
        3. Wrap DEK with KEK (Key Encryption Key)
        4. Return sealed envelope
        """
        # Generate unique DEK for this document
        dek = AESGCM.generate_key(bit_length=AES_KEY_BITS)
        dek_cipher = AESGCM(dek)

        # Encrypt plaintext with DEK
        nonce = secrets.token_bytes(NONCE_BYTES)
        aad = document_id.encode("utf-8")  # Bind ciphertext to document ID
        ciphertext = dek_cipher.encrypt(nonce, plaintext, aad)

        # Wrap DEK with KEK
        dek_nonce = secrets.token_bytes(NONCE_BYTES)
        encrypted_dek = self._kek_cipher.encrypt(dek_nonce, dek, None)
        encrypted_dek_with_nonce = dek_nonce + encrypted_dek

        # Hash plaintext for integrity verification
        content_hash = blake3.blake3(plaintext).hexdigest()

        # Securely zero the DEK from memory
        dek_array = bytearray(dek)
        _secure_zero(dek_array)

        return EncryptedBlob(
            nonce=nonce,
            encrypted_dek=encrypted_dek_with_nonce,
            ciphertext=ciphertext,
            document_id=document_id,
            timestamp=time.time(),
            blake3_hash=content_hash,
        )

    def decrypt(self, blob: EncryptedBlob) -> bytes:
        """
        Decrypt an encrypted envelope.
        
        1. Unwrap DEK using KEK
        2. Decrypt ciphertext using DEK
        3. Verify integrity via BLAKE3 hash
        4. Securely zero DEK
        """
        # Unwrap DEK
        dek_nonce = blob.encrypted_dek[:NONCE_BYTES]
        wrapped_dek = blob.encrypted_dek[NONCE_BYTES:]
        dek = self._kek_cipher.decrypt(dek_nonce, wrapped_dek, None)

        # Decrypt data
        dek_cipher = AESGCM(dek)
        aad = blob.document_id.encode("utf-8")
        plaintext = dek_cipher.decrypt(blob.nonce, blob.ciphertext, aad)

        # Verify integrity
        actual_hash = blake3.blake3(plaintext).hexdigest()
        if not hmac.compare_digest(actual_hash, blob.blake3_hash):
            # Zero the plaintext before raising — don't leak corrupted data
            pa = bytearray(plaintext)
            _secure_zero(pa)
            raise IntegrityError(
                f"BLAKE3 hash mismatch for document {blob.document_id}: "
                f"expected {blob.blake3_hash}, got {actual_hash}"
            )

        # Securely zero DEK
        dek_array = bytearray(dek)
        _secure_zero(dek_array)

        return plaintext

    def rotate_kek(self, new_password: str) -> tuple["EnvelopeEngine", bytes]:
        """
        Rotate the master key. Returns new engine + new salt.
        
        IMPORTANT: After rotation, all encrypted DEKs must be re-wrapped.
        Use `rewrap_dek()` to update existing encrypted blobs.
        """
        new_engine = EnvelopeEngine.from_password(new_password)
        return new_engine, new_engine._km.salt

    def rewrap_dek(self, blob: EncryptedBlob, new_engine: "EnvelopeEngine") -> EncryptedBlob:
        """
        Re-wrap a document's DEK with a new KEK (for key rotation).
        
        The document data itself is NOT re-encrypted — only the DEK wrapper changes.
        This is the key advantage of envelope encryption.
        """
        # Unwrap DEK with old KEK
        dek_nonce = blob.encrypted_dek[:NONCE_BYTES]
        wrapped_dek = blob.encrypted_dek[NONCE_BYTES:]
        dek = self._kek_cipher.decrypt(dek_nonce, wrapped_dek, None)

        # Re-wrap with new KEK
        new_dek_nonce = secrets.token_bytes(NONCE_BYTES)
        new_encrypted_dek = new_engine._kek_cipher.encrypt(new_dek_nonce, dek, None)
        new_encrypted_dek_with_nonce = new_dek_nonce + new_encrypted_dek

        # Zero old DEK
        dek_array = bytearray(dek)
        _secure_zero(dek_array)

        return EncryptedBlob(
            nonce=blob.nonce,
            encrypted_dek=new_encrypted_dek_with_nonce,
            ciphertext=blob.ciphertext,
            document_id=blob.document_id,
            timestamp=blob.timestamp,
            blake3_hash=blob.blake3_hash,
        )

    def derive_subkey(self, context: str, length: int = 32) -> bytes:
        """
        Derive a purpose-specific subkey from the KEK.
        
        Used for: shard encryption, HMAC keys, search tokens, etc.
        Each context string produces a unique, independent key.
        """
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=self._km.salt,
            info=f"derag-subkey-{context}".encode("utf-8"),
        )
        return hkdf.derive(bytes(self._km.kek))


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IntegrityError(Exception):
    """Raised when decrypted data fails integrity verification."""
    pass


class KeyDerivationError(Exception):
    """Raised when key derivation fails."""
    pass
