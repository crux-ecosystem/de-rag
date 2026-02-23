"""
De-RAG Test Suite — Crypto Core
=================================

Tests for:
  - Envelope encryption (AES-256-GCM) round-trip
  - Key derivation (Argon2id / HKDF)
  - Key rotation and re-wrapping
  - BLAKE3 integrity hashing
  - Searchable encryption (SSE + LSH)
  - Shamir's Secret Sharing

Run: pytest tests/ -v
"""

import os
import secrets
import tempfile
from pathlib import Path

import numpy as np
import pytest

from derag.crypto.envelope import EnvelopeEngine, EncryptedBlob, KeyMaterial
from derag.crypto.keys import KeyManager, KeyPurpose
from derag.crypto.searchable import SSEIndex, EncryptedLSH
from derag.storage.shard_manager import ShardManager, GF256


# ─── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def password():
    return "test-passphrase-for-derag!"

@pytest.fixture
def engine(password):
    return EnvelopeEngine(password)

@pytest.fixture
def key_manager(tmp_path, password):
    km = KeyManager(tmp_path / "keys")
    km.initialize(password)
    return km

@pytest.fixture
def shard_manager():
    return ShardManager(n=5, k=3)


# ─── Envelope Encryption ─────────────────────────────────────

class TestEnvelopeEncryption:
    """Test the core encryption engine."""

    def test_encrypt_decrypt_roundtrip(self, engine):
        plaintext = b"Hello, De-RAG! This is secret data."
        blob = engine.encrypt(plaintext)
        assert isinstance(blob, EncryptedBlob)
        recovered = engine.decrypt(blob)
        assert recovered == plaintext

    def test_encrypt_decrypt_large_data(self, engine):
        plaintext = os.urandom(1024 * 1024)  # 1 MB
        blob = engine.encrypt(plaintext)
        recovered = engine.decrypt(blob)
        assert recovered == plaintext

    def test_encrypt_decrypt_empty(self, engine):
        blob = engine.encrypt(b"")
        recovered = engine.decrypt(blob)
        assert recovered == b""

    def test_different_encryptions_produce_different_blobs(self, engine):
        data = b"same plaintext"
        blob1 = engine.encrypt(data)
        blob2 = engine.encrypt(data)
        # Different nonces → different ciphertext
        assert blob1.ciphertext != blob2.ciphertext

    def test_blob_serialization_roundtrip(self, engine):
        data = b"serialize me"
        blob = engine.encrypt(data)
        serialized = blob.serialize()
        deserialized = EncryptedBlob.deserialize(serialized)
        recovered = engine.decrypt(deserialized)
        assert recovered == data

    def test_wrong_password_fails(self, password):
        engine_good = EnvelopeEngine(password)
        engine_bad = EnvelopeEngine("wrong-password-oops")

        blob = engine_good.encrypt(b"secret")
        with pytest.raises(Exception):
            engine_bad.decrypt(blob)

    def test_tampered_ciphertext_fails(self, engine):
        blob = engine.encrypt(b"integrity test")
        # Flip a byte
        tampered = bytearray(blob.ciphertext)
        tampered[len(tampered) // 2] ^= 0xFF
        blob.ciphertext = bytes(tampered)
        with pytest.raises(Exception):
            engine.decrypt(blob)

    def test_integrity_hash(self, engine):
        data = b"hash me"
        h = engine.integrity_hash(data)
        assert len(h) == 64  # BLAKE3 hex digest
        # Deterministic
        assert engine.integrity_hash(data) == h

    def test_document_scoped_encryption(self, engine):
        """Encrypt with document_id scoping (AAD)."""
        data = b"scoped data"
        blob = engine.encrypt(data, document_id="doc-123")
        recovered = engine.decrypt(blob, document_id="doc-123")
        assert recovered == data

    def test_kek_rotation(self, engine):
        data = b"will survive rotation"
        blob = engine.encrypt(data)
        
        # Rotate KEK
        new_engine = engine.rotate_kek("new-password-456")
        
        # Old blob should be re-wrappable
        new_blob = new_engine.rewrap_dek(blob)
        recovered = new_engine.decrypt(new_blob)
        assert recovered == data

    def test_subkey_derivation(self, engine):
        key1 = engine.derive_subkey("encryption")
        key2 = engine.derive_subkey("search")
        key3 = engine.derive_subkey("encryption")
        assert key1 != key2
        assert key1 == key3  # Same context → same subkey


# ─── Key Manager ──────────────────────────────────────────────

class TestKeyManager:
    """Test key lifecycle management."""

    def test_initialization(self, key_manager):
        assert key_manager.is_initialized

    def test_node_id_deterministic(self, tmp_path, password):
        km1 = KeyManager(tmp_path / "keys")
        km1.initialize(password)
        node_id = km1.get_node_id()
        
        # Re-open same key store
        km2 = KeyManager(tmp_path / "keys")
        km2.initialize(password)
        assert km2.get_node_id() == node_id

    def test_envelope_engine_creation(self, key_manager):
        engine = key_manager.get_envelope_engine()
        assert isinstance(engine, EnvelopeEngine)

        # Round-trip through manager-created engine
        data = b"via key manager"
        blob = engine.encrypt(data)
        assert engine.decrypt(blob) == data

    def test_key_listing(self, key_manager):
        records = key_manager.list_keys()
        assert len(records) > 0
        # Should have at least KEK and identity
        purposes = {r["purpose"] for r in records}
        assert "kek" in purposes or "identity" in purposes

    def test_key_rotation(self, key_manager):
        engine_before = key_manager.get_envelope_engine()
        data = b"survive rotation"
        blob = engine_before.encrypt(data)

        key_manager.rotate_key("kek")
        engine_after = key_manager.get_envelope_engine()

        # Should still decrypt (re-wrapped)
        new_blob = engine_after.rewrap_dek(blob)
        assert engine_after.decrypt(new_blob) == data


# ─── Searchable Encryption ───────────────────────────────────

class TestSearchableEncryption:
    """Test search over encrypted data."""

    def test_sse_index_add_and_search(self):
        key = secrets.token_bytes(32)
        sse = SSEIndex(key)

        sse.add("doc1", ["hello", "world", "python"])
        sse.add("doc2", ["hello", "rust", "crypto"])
        sse.add("doc3", ["world", "data", "privacy"])

        results = sse.search("hello")
        assert "doc1" in results
        assert "doc2" in results
        assert "doc3" not in results

    def test_sse_remove(self):
        key = secrets.token_bytes(32)
        sse = SSEIndex(key)

        sse.add("doc1", ["test", "data"])
        assert "doc1" in sse.search("test")

        sse.remove("doc1")
        assert "doc1" not in sse.search("test")

    def test_sse_different_keys_no_collision(self):
        key1 = secrets.token_bytes(32)
        key2 = secrets.token_bytes(32)

        sse1 = SSEIndex(key1)
        sse2 = SSEIndex(key2)

        sse1.add("doc1", ["secret"])
        sse2.add("doc2", ["secret"])

        # Different keys → different tokens → independent indexes
        assert "doc1" in sse1.search("secret")
        assert "doc1" not in sse2.search("secret")

    def test_encrypted_lsh_approximate_search(self):
        key = secrets.token_bytes(32)
        lsh = EncryptedLSH(key, dim=128, num_tables=10, num_bits=8)

        # Create clustered vectors
        base_vec = np.random.randn(128).astype(np.float32)
        similar_vec = base_vec + np.random.randn(128).astype(np.float32) * 0.1
        different_vec = np.random.randn(128).astype(np.float32)

        lsh.add("similar", similar_vec)
        lsh.add("different", different_vec)

        results = lsh.query(base_vec, top_k=5)
        # The similar vector should be found
        result_ids = [r[0] for r in results]
        assert "similar" in result_ids


# ─── Shamir's Secret Sharing (GF(256)) ───────────────────────

class TestShamirSecretSharing:
    """Test Shamir's Secret Sharing over GF(2^8)."""

    def test_gf256_multiply(self):
        """Test Galois Field multiplication."""
        assert GF256.multiply(0, 100) == 0
        assert GF256.multiply(1, 100) == 100
        assert GF256.multiply(100, 1) == 100

    def test_gf256_inverse(self):
        """Every non-zero element has a multiplicative inverse."""
        for x in range(1, 256):
            inv = GF256.inverse(x)
            assert GF256.multiply(x, inv) == 1

    def test_split_reconstruct_exact_threshold(self, shard_manager):
        """Reconstruct with exactly k shards."""
        data = b"top secret information"
        shards = shard_manager.split(data, "test-doc-1")
        assert len(shards) == 5

        # Use exactly k=3 shards
        subset = shards[:3]
        recovered = shard_manager.reconstruct(subset)
        assert recovered == data

    def test_split_reconstruct_all_shards(self, shard_manager):
        """Reconstruct with all n shards."""
        data = b"use all shards"
        shards = shard_manager.split(data, "test-doc-2")
        recovered = shard_manager.reconstruct(shards)
        assert recovered == data

    def test_split_reconstruct_different_subsets(self, shard_manager):
        """Any k-of-n subset works."""
        data = b"any subset works"
        shards = shard_manager.split(data, "test-doc-3")

        # Try different 3-shard subsets
        from itertools import combinations
        for subset in combinations(shards, 3):
            recovered = shard_manager.reconstruct(list(subset))
            assert recovered == data

    def test_insufficient_shards_fails(self, shard_manager):
        """k-1 shards should not reconstruct correctly."""
        data = b"need enough shards"
        shards = shard_manager.split(data, "test-doc-4")

        # Only k-1=2 shards (should fail or produce garbage)
        subset = shards[:2]
        try:
            recovered = shard_manager.reconstruct(subset)
            assert recovered != data  # Should not match
        except Exception:
            pass  # Expected

    def test_shard_integrity(self, shard_manager):
        """Each shard should verify its own integrity."""
        data = b"verify integrity"
        shards = shard_manager.split(data, "test-doc-5")

        for shard in shards:
            assert shard_manager.verify(shard)
            # Tamper with shard data
            tampered = bytearray(shard.data)
            tampered[0] ^= 0xFF
            shard.data = bytes(tampered)
            assert not shard_manager.verify(shard)

    def test_large_data_sharding(self, shard_manager):
        """Shard and reconstruct large data."""
        data = os.urandom(100_000)  # 100 KB
        shards = shard_manager.split(data, "large-doc")
        recovered = shard_manager.reconstruct(shards[:3])
        assert recovered == data

    def test_shard_serialization(self, shard_manager):
        """Serialize and deserialize shards."""
        data = b"serialize shards"
        shards = shard_manager.split(data, "serial-doc")

        from derag.storage.shard_manager import Shard
        for shard in shards:
            serialized = shard.serialize()
            deserialized = Shard.deserialize(serialized)
            assert deserialized.shard_id == shard.shard_id
            assert deserialized.shard_index == shard.shard_index
            assert deserialized.data == shard.data


# ─── Audit Log ────────────────────────────────────────────────

class TestAuditLog:
    """Test tamper-proof audit logging."""

    def test_append_and_query(self, tmp_path):
        from derag.lineage.audit import MerkleAuditLog

        log = MerkleAuditLog(tmp_path / "audit", "test-node")
        log.append(action="ingest", target="doc-1")
        log.append(action="query", target="q-1")
        log.append(action="ingest", target="doc-2")

        all_entries = log.query()
        assert len(all_entries) == 3

        ingest_entries = log.query(action="ingest")
        assert len(ingest_entries) == 2

    def test_chain_integrity(self, tmp_path):
        from derag.lineage.audit import MerkleAuditLog

        log = MerkleAuditLog(tmp_path / "audit", "test-node")
        for i in range(10):
            log.append(action="test", target=f"item-{i}")

        assert log.verify_chain()

    def test_chain_tamper_detection(self, tmp_path):
        from derag.lineage.audit import MerkleAuditLog

        log = MerkleAuditLog(tmp_path / "audit", "test-node")
        for i in range(5):
            log.append(action="test", target=f"item-{i}")

        assert log.verify_chain()

        # Tamper with the log file
        log_file = tmp_path / "audit" / "audit.jsonl"
        lines = log_file.read_text().strip().split("\n")
        import json
        entry = json.loads(lines[2])
        entry["action"] = "TAMPERED"
        lines[2] = json.dumps(entry)
        log_file.write_text("\n".join(lines) + "\n")

        # Re-load and verify should fail
        log2 = MerkleAuditLog(tmp_path / "audit", "test-node")
        assert not log2.verify_chain()

    def test_persistence(self, tmp_path):
        from derag.lineage.audit import MerkleAuditLog

        log1 = MerkleAuditLog(tmp_path / "audit", "test-node")
        log1.append(action="persist_test", target="data-1")
        count1 = log1.stats["entry_count"]

        # Re-open
        log2 = MerkleAuditLog(tmp_path / "audit", "test-node")
        assert log2.stats["entry_count"] == count1
        assert log2.verify_chain()


# ─── Protocol Serialization ──────────────────────────────────

class TestProtocol:
    """Test P2P protocol message serialization."""

    def test_message_roundtrip(self):
        from derag.network.protocol import Message, MessageType

        msg = Message.make_hello("node-abc", 9090)
        data = msg.serialize()
        recovered = Message.deserialize(data)

        assert recovered.msg_type == MessageType.HELLO
        assert recovered.sender == "node-abc"
        assert recovered.payload["port"] == 9090

    def test_store_shard_message(self):
        from derag.network.protocol import Message, MessageType

        msg = Message.make_store_shard("node-x", "shard-1", b"encrypted-data", "doc-1")
        data = msg.serialize()
        recovered = Message.deserialize(data)

        assert recovered.msg_type == MessageType.STORE_SHARD
        assert recovered.payload["shard_id"] == "shard-1"

    def test_query_message(self):
        from derag.network.protocol import Message, MessageType

        vector = [0.1, 0.2, 0.3]
        msg = Message.make_query("node-y", "req-1", vector, top_k=10)
        data = msg.serialize()
        recovered = Message.deserialize(data)

        assert recovered.msg_type == MessageType.QUERY
        assert recovered.payload["request_id"] == "req-1"
        assert recovered.payload["top_k"] == 10
