"""
De-RAG API — Pydantic request/response models
==============================================

All HTTP request bodies and response shapes for the De-RAG REST API.

Author: Mounesh Kodi — CruxLabx
Copyright (c) 2026 CruxLabx — AGPL-3.0
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Lifecycle ────────────────────────────────────────────────

class InitRequest(BaseModel):
    """Initialize a De-RAG node."""
    password: str = Field(..., min_length=8, description="Master password for key derivation")
    node_name: Optional[str] = Field(None, description="Optional human-readable node name")


class StatusResponse(BaseModel):
    """Node status summary."""
    node_id: str
    node_name: str
    version: str
    is_initialized: bool
    uptime_seconds: float = 0
    documents: int = 0
    chunks: int = 0
    vectors: int = 0
    shards: int = 0
    peers: int = 0
    audit_entries: int = 0
    storage_bytes: int = 0
    index_type: str = "flat"


class HealthResponse(BaseModel):
    """Simple health check."""
    status: str = "ok"
    version: str
    timestamp: float


# ─── Documents ────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Ingest request (for JSON-based ingestion)."""
    content: str = Field(..., description="Raw text content to ingest")
    filename: str = Field("untitled.txt", description="Virtual filename")
    chunk_size: int = Field(512, ge=64, le=8192)
    chunk_overlap: int = Field(50, ge=0, le=512)
    metadata: Optional[dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Single document metadata."""
    doc_id: str
    filename: str
    content_hash: str
    chunk_count: int
    total_size: int
    encrypted_size: int
    mime_type: str
    metadata: dict[str, Any]
    created_at: float
    updated_at: float


class DocumentListResponse(BaseModel):
    """List of documents."""
    documents: list[DocumentResponse]
    total: int


class ChunkResponse(BaseModel):
    """A single chunk."""
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    text_hash: str
    char_count: int
    token_estimate: int
    metadata: dict[str, Any]


class ChunkListResponse(BaseModel):
    """List of chunks for a document."""
    doc_id: str
    chunks: list[ChunkResponse]
    total: int


# ─── Query ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Execute a RAG query."""
    question: str = Field(..., min_length=1, max_length=4096)
    distributed: bool = Field(False, description="Query remote peers too")
    top_k: int = Field(10, ge=1, le=100)


class QueryResultItem(BaseModel):
    """A single search result."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    source_peer: Optional[str] = None


class QueryResponse(BaseModel):
    """Full query response with generated answer."""
    answer: str
    question: str
    results: list[QueryResultItem] = []
    confidence: Optional[float] = None
    generation_model: Optional[str] = None
    total_time_ms: float = 0
    query_time_ms: float = 0
    generation_time_ms: float = 0
    peers_queried: int = 0
    local_results: int = 0
    remote_results: int = 0


class SearchRequest(BaseModel):
    """Similarity search without generation."""
    question: str = Field(..., min_length=1, max_length=4096)
    k: int = Field(10, ge=1, le=100)


class SearchResponse(BaseModel):
    """Search-only results."""
    results: list[QueryResultItem]
    total: int
    query_time_ms: float


# ─── Network ─────────────────────────────────────────────────

class JoinNetworkRequest(BaseModel):
    """Join the P2P network."""
    port: int = Field(9090, ge=1024, le=65535)
    bootstrap_peers: Optional[list[str]] = None


class PeerResponse(BaseModel):
    """A connected peer."""
    peer_id: str
    address: str
    port: int
    shard_count: int
    available_space_mb: float
    connected_at: float
    last_heartbeat: float
    latency_ms: float
    is_alive: bool


class PeerListResponse(BaseModel):
    """List of connected peers."""
    peers: list[PeerResponse]
    total: int


# ─── Shards ───────────────────────────────────────────────────

class DistributeRequest(BaseModel):
    """Optional overrides for shard distribution."""
    target_peers: Optional[list[str]] = None


class ShardDistribution(BaseModel):
    """Result of distributing a document's shards."""
    doc_id: str
    shards_distributed: int
    distribution: list[dict[str, Any]]


# ─── Keys ─────────────────────────────────────────────────────

class KeyRotateRequest(BaseModel):
    """Rotate master key."""
    new_password: str = Field(..., min_length=8)
    current_password: Optional[str] = Field(None, min_length=8)


class KeyInfo(BaseModel):
    """Key metadata (never exposes actual key material)."""
    key_id: str
    purpose: str
    algorithm: str
    created_at: float
    is_active: bool


class KeyListResponse(BaseModel):
    """List of key records."""
    keys: list[KeyInfo]
    total: int


# ─── Audit ────────────────────────────────────────────────────

class AuditEntryResponse(BaseModel):
    """A single audit log entry."""
    entry_id: int
    timestamp: float
    action: str
    actor: str
    target: str
    details: dict[str, Any]
    entry_hash: str


class AuditQueryRequest(BaseModel):
    """Filter audit log."""
    action: Optional[str] = None
    actor: Optional[str] = None
    target: Optional[str] = None
    since: Optional[float] = None
    until: Optional[float] = None
    limit: int = Field(100, ge=1, le=1000)


class AuditListResponse(BaseModel):
    """Audit query results."""
    entries: list[AuditEntryResponse]
    total: int


class AuditVerifyResponse(BaseModel):
    """Merkle chain verification result."""
    valid: bool
    total_entries: int
    broken_at: Optional[int] = None


# ─── Admin Dashboard ─────────────────────────────────────────

class DashboardStats(BaseModel):
    """Aggregated stats for admin dashboard."""
    node: StatusResponse
    crypto: dict[str, Any]
    storage: dict[str, Any]
    network: dict[str, Any]
    query: dict[str, Any]
    audit: dict[str, Any]
    system: dict[str, Any]


class ErrorResponse(BaseModel):
    """Standard error shape."""
    error: str
    detail: Optional[str] = None
    status_code: int


class MessageResponse(BaseModel):
    """Simple success message."""
    message: str
    timestamp: float = Field(default_factory=lambda: __import__("time").time())
