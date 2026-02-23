"""
De-RAG Key Management System
==============================

Handles the full lifecycle of cryptographic keys:
  - Generation and secure storage
  - Key rotation with zero-downtime re-wrapping  
  - Key hierarchy: Master KEK → Document DEKs → Shard Keys → Search Tokens
  - Secure deletion with multi-pass overwrite
  - Key escrow for disaster recovery

Key Hierarchy:
    Master Password
         │
         ▼ (Argon2id)
    KEK (Key Encryption Key)
         │
         ├──▶ DEK per document (AES-256-GCM)
         ├──▶ Shard Key per shard (AES-256-GCM)  
         ├──▶ Search Token Key (HMAC-SHA256)
         ├──▶ Node Identity Key (Ed25519)
         └──▶ Peer Auth Key (X25519)

Copyright (c) 2026 CruxLabx
"""

from __future__ import annotations

import json
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import blake3

from derag.crypto.envelope import (
    EnvelopeEngine,
    KeyMaterial,
    AES_KEY_BYTES,
    NONCE_BYTES,
    SALT_BYTES,
    _secure_zero,
)


@dataclass
class KeyRecord:
    """Metadata about a stored key."""
    key_id: str
    purpose: str           # "dek", "shard", "search", "identity", "peer"
    algorithm: str         # "aes-256-gcm", "ed25519", "x25519"
    created_at: float
    rotated_at: Optional[float] = None
    expires_at: Optional[float] = None
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "key_id": self.key_id,
            "purpose": self.purpose,
            "algorithm": self.algorithm,
            "created_at": self.created_at,
            "rotated_at": self.rotated_at,
            "expires_at": self.expires_at,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeyRecord":
        return cls(**d)


class KeyManager:
    """
    Manages the full cryptographic key hierarchy for a De-RAG node.
    
    Usage:
        km = KeyManager(keys_dir=Path("~/.derag/keys"))
        km.initialize("master-password")
        
        # Get envelope engine for document encryption
        engine = km.get_envelope_engine()
        
        # Get node identity for P2P authentication
        identity = km.get_node_identity()
        
        # Rotate master key
        km.rotate_master("new-password")
    """

    def __init__(self, keys_dir: Path):
        self._dir = Path(keys_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._dir, 0o700)
        
        self._engine: Optional[EnvelopeEngine] = None
        self._records: dict[str, KeyRecord] = {}
        self._identity_key: Optional[Ed25519PrivateKey] = None
        self._peer_key: Optional[X25519PrivateKey] = None

    @property
    def is_initialized(self) -> bool:
        return (self._dir / "master.salt").exists()

    def initialize(self, password: str) -> None:
        """
        Initialize key hierarchy from password.
        Creates master salt, node identity, and peer key if they don't exist.
        """
        salt_path = self._dir / "master.salt"

        if salt_path.exists():
            # Load existing salt
            salt = salt_path.read_bytes()
            self._engine = EnvelopeEngine.from_password(password, salt=salt)
        else:
            # First-time setup
            self._engine = EnvelopeEngine.from_password(password)
            salt_path.write_bytes(self._engine._km.salt)
            os.chmod(salt_path, 0o600)

        # Initialize node identity (Ed25519) for P2P auth
        self._init_identity_key()

        # Initialize peer key (X25519) for key exchange
        self._init_peer_key()

        # Load key records
        self._load_records()

    def get_envelope_engine(self) -> EnvelopeEngine:
        """Get the envelope encryption engine."""
        if not self._engine:
            raise RuntimeError("KeyManager not initialized — call initialize() first")
        return self._engine

    def get_node_identity(self) -> Ed25519PrivateKey:
        """Get node identity key for signing and P2P authentication."""
        if not self._identity_key:
            raise RuntimeError("KeyManager not initialized")
        return self._identity_key

    def get_node_id(self) -> str:
        """Get node ID derived from public key (like IPFS peer IDs)."""
        if not self._identity_key:
            raise RuntimeError("KeyManager not initialized")
        pub_bytes = self._identity_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        return blake3.blake3(pub_bytes).hexdigest()[:32]

    def get_peer_key(self) -> X25519PrivateKey:
        """Get X25519 key for Diffie-Hellman key exchange with peers."""
        if not self._peer_key:
            raise RuntimeError("KeyManager not initialized")
        return self._peer_key

    def derive_shard_key(self, shard_id: str) -> bytes:
        """Derive a unique encryption key for a specific shard."""
        if not self._engine:
            raise RuntimeError("KeyManager not initialized")
        return self._engine.derive_subkey(f"shard:{shard_id}")

    def derive_search_token_key(self) -> bytes:
        """Derive key for searchable encryption tokens."""
        if not self._engine:
            raise RuntimeError("KeyManager not initialized")
        return self._engine.derive_subkey("search-tokens")

    def rotate_master(self, new_password: str) -> None:
        """
        Rotate the master KEK.
        
        This re-wraps all stored DEKs with the new KEK.
        Document data is NOT re-encrypted (envelope encryption advantage).
        """
        if not self._engine:
            raise RuntimeError("KeyManager not initialized")

        new_engine, new_salt = self._engine.rotate_kek(new_password)

        # Update salt file
        salt_path = self._dir / "master.salt"
        # Backup old salt
        backup_path = self._dir / f"master.salt.bak.{int(time.time())}"
        salt_path.rename(backup_path)
        salt_path.write_bytes(new_salt)
        os.chmod(salt_path, 0o600)

        # Re-wrap identity key with new engine
        self._rewrap_identity_key(new_engine)
        self._rewrap_peer_key(new_engine)

        self._engine = new_engine

    def secure_delete_key(self, key_id: str) -> None:
        """Securely delete a key — multi-pass overwrite + unlink."""
        key_path = self._dir / f"{key_id}.key"
        if key_path.exists():
            size = key_path.stat().st_size
            with open(key_path, "r+b") as f:
                # 3-pass overwrite: zeros, ones, random
                for pattern in [b"\x00", b"\xff", None]:
                    f.seek(0)
                    if pattern:
                        f.write(pattern * size)
                    else:
                        f.write(secrets.token_bytes(size))
                    f.flush()
                    os.fsync(f.fileno())
            key_path.unlink()

        # Remove record
        self._records.pop(key_id, None)
        self._save_records()

    # --- Private Methods ---

    def _init_identity_key(self) -> None:
        """Initialize or load Ed25519 identity key."""
        key_path = self._dir / "identity.key"
        if key_path.exists():
            encrypted_key = key_path.read_bytes()
            key_bytes = self._engine.decrypt(
                __import__("derag.crypto.envelope", fromlist=["EncryptedBlob"]).EncryptedBlob.deserialize(encrypted_key)
            )
            self._identity_key = Ed25519PrivateKey.from_private_bytes(key_bytes)
            _secure_zero(bytearray(key_bytes))
        else:
            self._identity_key = Ed25519PrivateKey.generate()
            key_bytes = self._identity_key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
            blob = self._engine.encrypt(key_bytes, document_id="node-identity-key")
            key_path.write_bytes(blob.serialize())
            os.chmod(key_path, 0o600)
            _secure_zero(bytearray(key_bytes))

            self._add_record(KeyRecord(
                key_id="identity",
                purpose="identity",
                algorithm="ed25519",
                created_at=time.time(),
            ))

    def _init_peer_key(self) -> None:
        """Initialize or load X25519 peer key."""
        key_path = self._dir / "peer.key"
        if key_path.exists():
            encrypted_key = key_path.read_bytes()
            from derag.crypto.envelope import EncryptedBlob
            key_bytes = self._engine.decrypt(
                EncryptedBlob.deserialize(encrypted_key)
            )
            self._peer_key = X25519PrivateKey.from_private_bytes(key_bytes)
            _secure_zero(bytearray(key_bytes))
        else:
            self._peer_key = X25519PrivateKey.generate()
            key_bytes = self._peer_key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
            blob = self._engine.encrypt(key_bytes, document_id="node-peer-key")
            key_path.write_bytes(blob.serialize())
            os.chmod(key_path, 0o600)
            _secure_zero(bytearray(key_bytes))

            self._add_record(KeyRecord(
                key_id="peer",
                purpose="peer",
                algorithm="x25519",
                created_at=time.time(),
            ))

    def _rewrap_identity_key(self, new_engine: EnvelopeEngine) -> None:
        """Re-wrap identity key with new KEK."""
        key_path = self._dir / "identity.key"
        if not key_path.exists():
            return
        from derag.crypto.envelope import EncryptedBlob
        old_blob = EncryptedBlob.deserialize(key_path.read_bytes())
        new_blob = self._engine.rewrap_dek(old_blob, new_engine)
        key_path.write_bytes(new_blob.serialize())

    def _rewrap_peer_key(self, new_engine: EnvelopeEngine) -> None:
        """Re-wrap peer key with new KEK."""
        key_path = self._dir / "peer.key"
        if not key_path.exists():
            return
        from derag.crypto.envelope import EncryptedBlob
        old_blob = EncryptedBlob.deserialize(key_path.read_bytes())
        new_blob = self._engine.rewrap_dek(old_blob, new_engine)
        key_path.write_bytes(new_blob.serialize())

    def _add_record(self, record: KeyRecord) -> None:
        self._records[record.key_id] = record
        self._save_records()

    def _load_records(self) -> None:
        records_path = self._dir / "records.json"
        if records_path.exists():
            data = json.loads(records_path.read_text())
            self._records = {r["key_id"]: KeyRecord.from_dict(r) for r in data}

    def _save_records(self) -> None:
        records_path = self._dir / "records.json"
        data = [r.to_dict() for r in self._records.values()]
        records_path.write_text(json.dumps(data, indent=2))
        os.chmod(records_path, 0o600)
