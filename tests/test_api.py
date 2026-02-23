"""
De-RAG API Test Suite — v0.2.0
================================

Tests the FastAPI HTTP endpoints including lifecycle, document CRUD,
query, network, audit, and admin dashboard.

Uses FastAPI TestClient (synchronous) for all endpoint tests.

Author: Mounesh Kodi — CruxLabx
Copyright (c) 2026 CruxLabx — AGPL-3.0
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from derag.api.middleware import BearerAuthMiddleware, RateLimitMiddleware
from derag.api.models import (
    AuditListResponse,
    AuditVerifyResponse,
    DashboardStats,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    MessageResponse,
    PeerListResponse,
    QueryResponse,
    SearchResponse,
    ShardDistribution,
    StatusResponse,
)
from derag.api.server import DeRAGAPI, create_app


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_data_dir():
    """Create an isolated temporary data directory for each test."""
    d = tempfile.mkdtemp(prefix="derag_test_api_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def api(tmp_data_dir):
    """DeRAGAPI instance with no auth token."""
    return DeRAGAPI(data_dir=tmp_data_dir, auth_token=None)


@pytest.fixture()
def client(api):
    """FastAPI TestClient for the API."""
    return TestClient(api.app)


@pytest.fixture()
def auth_api(tmp_data_dir):
    """DeRAGAPI instance with auth token."""
    return DeRAGAPI(data_dir=tmp_data_dir, auth_token="test-secret-token-42")


@pytest.fixture()
def auth_client(auth_api):
    """FastAPI TestClient with auth enabled."""
    return TestClient(auth_api.app)


# ────────────────────────────────────────────────────────────
# Health & Lifecycle
# ────────────────────────────────────────────────────────────


class TestLifecycle:
    """Tests for /health, /status, /init, /shutdown."""

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data

    def test_health_model(self, client):
        resp = client.get("/health")
        h = HealthResponse(**resp.json())
        assert h.status == "ok"

    def test_status_requires_init(self, client):
        """Status should return 503 when node is not initialized."""
        resp = client.get("/status")
        assert resp.status_code == 503

    def test_shutdown_without_node(self, client):
        """Shutdown should succeed even if node isn't initialized."""
        resp = client.post("/shutdown")
        assert resp.status_code == 200
        assert "shut down" in resp.json()["message"].lower()

    def test_init_missing_password(self, client):
        """Init without password field should be a validation error."""
        resp = client.post("/init", json={})
        assert resp.status_code == 422


# ────────────────────────────────────────────────────────────
# Authentication middleware
# ────────────────────────────────────────────────────────────


class TestAuth:
    """Tests for Bearer auth middleware."""

    def test_health_bypasses_auth(self, auth_client):
        """Health endpoint should work without a token."""
        resp = auth_client.get("/health")
        assert resp.status_code == 200

    def test_docs_bypasses_auth(self, auth_client):
        """OpenAPI docs endpoint should work without a token."""
        resp = auth_client.get("/docs")
        assert resp.status_code == 200

    def test_protected_endpoint_no_token(self, auth_client):
        """Protected endpoint without token should return 401."""
        resp = auth_client.get("/status")
        assert resp.status_code == 401

    def test_protected_endpoint_wrong_token(self, auth_client):
        """Protected endpoint with wrong token should return 403."""
        resp = auth_client.get(
            "/status", headers={"Authorization": "Bearer wrong-token"}
        )
        assert resp.status_code == 403

    def test_protected_endpoint_correct_token(self, auth_client):
        """Protected endpoint with correct token should pass auth.
        (Still 503 because node not initialized, but NOT 401)."""
        resp = auth_client.get(
            "/status",
            headers={"Authorization": "Bearer test-secret-token-42"},
        )
        # 503 (not initialized) rather than 401 (auth fail)
        assert resp.status_code == 503


# ────────────────────────────────────────────────────────────
# Documents
# ────────────────────────────────────────────────────────────


class TestDocuments:
    """Tests for document endpoints (without a real initialized node)."""

    def test_list_documents_requires_init(self, client):
        resp = client.get("/documents")
        assert resp.status_code == 503

    def test_ingest_requires_init(self, client):
        resp = client.post(
            "/documents",
            json={"content": "hello world", "filename": "test.txt"},
        )
        assert resp.status_code == 503

    def test_upload_requires_init(self, client):
        resp = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 503

    def test_get_document_requires_init(self, client):
        resp = client.get("/documents/abc123")
        assert resp.status_code == 503

    def test_delete_document_requires_init(self, client):
        resp = client.delete("/documents/abc123")
        assert resp.status_code == 503

    def test_get_chunks_requires_init(self, client):
        resp = client.get("/documents/abc123/chunks")
        assert resp.status_code == 503


# ────────────────────────────────────────────────────────────
# Query
# ────────────────────────────────────────────────────────────


class TestQuery:
    """Tests for /query and /search endpoints."""

    def test_query_requires_init(self, client):
        resp = client.post("/query", json={"question": "What is De-RAG?"})
        assert resp.status_code == 503

    def test_search_requires_init(self, client):
        resp = client.post("/search", json={"question": "What is De-RAG?"})
        assert resp.status_code == 503

    def test_query_validation(self, client):
        """Missing question field should be 422."""
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_search_validation(self, client):
        resp = client.post("/search", json={})
        assert resp.status_code == 422


# ────────────────────────────────────────────────────────────
# Network
# ────────────────────────────────────────────────────────────


class TestNetwork:
    """Tests for /network/* endpoints."""

    def test_join_requires_init(self, client):
        resp = client.post("/network/join", json={"port": 9420})
        assert resp.status_code == 503

    def test_peers_requires_init(self, client):
        resp = client.get("/network/peers")
        assert resp.status_code == 503

    def test_disconnect_requires_init(self, client):
        resp = client.post("/network/disconnect/abc123")
        assert resp.status_code == 503


# ────────────────────────────────────────────────────────────
# Shards
# ────────────────────────────────────────────────────────────


class TestShards:
    def test_distribute_requires_init(self, client):
        resp = client.post("/shards/doc123")
        assert resp.status_code == 503


# ────────────────────────────────────────────────────────────
# Keys
# ────────────────────────────────────────────────────────────


class TestKeys:
    def test_list_keys_requires_init(self, client):
        resp = client.get("/keys")
        assert resp.status_code == 503

    def test_rotate_requires_init(self, client):
        resp = client.post("/keys/rotate", json={"new_password": "new-pass"})
        assert resp.status_code == 503


# ────────────────────────────────────────────────────────────
# Audit
# ────────────────────────────────────────────────────────────


class TestAudit:
    def test_get_audit_requires_init(self, client):
        resp = client.get("/audit")
        assert resp.status_code == 503

    def test_post_audit_requires_init(self, client):
        resp = client.post("/audit", json={"limit": 50})
        assert resp.status_code == 503

    def test_verify_audit_requires_init(self, client):
        resp = client.get("/audit/verify")
        assert resp.status_code == 503


# ────────────────────────────────────────────────────────────
# Admin
# ────────────────────────────────────────────────────────────


class TestAdmin:
    def test_dashboard_requires_init(self, client):
        resp = client.get("/admin/dashboard")
        assert resp.status_code == 503


# ────────────────────────────────────────────────────────────
# Pydantic Models Unit Tests
# ────────────────────────────────────────────────────────────


class TestModels:
    """Ensure request/response models validate correctly."""

    def test_health_response(self):
        h = HealthResponse(status="ok", version="0.2.0", timestamp=time.time())
        assert h.status == "ok"

    def test_message_response(self):
        m = MessageResponse(message="done")
        assert m.message == "done"

    def test_status_response_defaults(self):
        s = StatusResponse(
            node_id="abc",
            node_name="test",
            version="0.2.0",
            is_initialized=True,
        )
        assert s.uptime_seconds == 0

    def test_query_response(self):
        q = QueryResponse(answer="hello", question="hi")
        assert q.results == []
        assert q.confidence is None

    def test_search_response(self):
        s = SearchResponse(results=[], total=0, query_time_ms=1.5)
        assert s.total == 0

    def test_shard_distribution(self):
        sd = ShardDistribution(doc_id="d1", shards_distributed=3, distribution=[])
        assert sd.shards_distributed == 3

    def test_audit_verify_response(self):
        av = AuditVerifyResponse(valid=True, total_entries=100)
        assert av.broken_at is None


# ────────────────────────────────────────────────────────────
# Factory function
# ────────────────────────────────────────────────────────────


class TestFactory:
    """Tests for create_app factory."""

    def test_create_app_returns_fastapi(self, tmp_data_dir):
        from fastapi import FastAPI

        app = create_app(data_dir=tmp_data_dir)
        assert isinstance(app, FastAPI)

    def test_create_app_has_routes(self, tmp_data_dir):
        app = create_app(data_dir=tmp_data_dir)
        paths = [r.path for r in app.routes]
        assert "/health" in paths
        assert "/init" in paths
        assert "/documents" in paths
        assert "/query" in paths
        assert "/search" in paths

    def test_create_app_custom_rate_limit(self, tmp_data_dir):
        app = create_app(data_dir=tmp_data_dir, rate_limit=50)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200


# ────────────────────────────────────────────────────────────
# OpenAPI schema
# ────────────────────────────────────────────────────────────


class TestOpenAPI:
    """Ensure the generated OpenAPI schema is correct."""

    def test_openapi_json(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "De-RAG API"
        assert "0.2.0" in schema["info"]["version"]
        assert "/health" in schema["paths"]
        assert "/init" in schema["paths"]
        assert "/query" in schema["paths"]

    def test_openapi_has_tags(self, client):
        resp = client.get("/openapi.json")
        schema = resp.json()
        # Every path should have at least one tag
        for path_name, methods in schema["paths"].items():
            for method, detail in methods.items():
                if method in ("get", "post", "put", "delete", "patch"):
                    assert "tags" in detail, f"{method} {path_name} has no tags"
