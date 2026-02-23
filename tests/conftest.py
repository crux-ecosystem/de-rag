"""conftest.py — shared fixtures for De-RAG tests."""

import pytest


@pytest.fixture(scope="session")
def test_password():
    return "derag-test-password-2026!"


@pytest.fixture
def sample_text():
    return (
        "De-RAG is a decentralized encrypted retrieval-augmented generation system. "
        "It uses Shamir's Secret Sharing to shard encrypted data across peers. "
        "The envelope encryption scheme uses AES-256-GCM with Argon2id key derivation. "
        "BLAKE3 provides integrity hashing throughout the pipeline. "
        "Searchable encryption enables queries over encrypted document stores."
    )
