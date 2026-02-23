"""
De-RAG HTTP API Server
======================

FastAPI-based REST API for the De-RAG decentralized encrypted RAG engine.

Endpoints:
    /health              GET   — Health check
    /status              GET   — Node status
    /init                POST  — Initialize node
    /shutdown            POST  — Graceful shutdown

    /documents           GET   — List documents
    /documents           POST  — Ingest text content
    /documents/upload    POST  — Upload file
    /documents/{id}      GET   — Get document
    /documents/{id}      DELETE— Delete document
    /documents/{id}/chunks GET — Get chunks

    /query               POST  — RAG query (search + generate)
    /search              POST  — Search only (no generation)

    /network/join        POST  — Join P2P network
    /network/peers       GET   — List peers
    /network/disconnect  POST  — Disconnect peer

    /shards/{doc_id}     POST  — Distribute shards

    /keys                GET   — List keys
    /keys/rotate         POST  — Rotate master key

    /audit               GET   — Query audit log
    /audit               POST  — Query audit log (with filters)
    /audit/verify        GET   — Verify Merkle chain

    /admin/dashboard     GET   — Full dashboard stats

Author: Mounesh Kodi — CruxLabx
Copyright (c) 2026 CruxLabx — AGPL-3.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from derag import __version__
from derag.api.middleware import (
    BearerAuthMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from derag.api.models import (
    AuditEntryResponse,
    AuditListResponse,
    AuditQueryRequest,
    AuditVerifyResponse,
    ChunkListResponse,
    ChunkResponse,
    DashboardStats,
    DistributeRequest,
    DocumentListResponse,
    DocumentResponse,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    InitRequest,
    JoinNetworkRequest,
    KeyInfo,
    KeyListResponse,
    KeyRotateRequest,
    MessageResponse,
    PeerListResponse,
    PeerResponse,
    QueryRequest,
    QueryResponse,
    QueryResultItem,
    SearchRequest,
    SearchResponse,
    ShardDistribution,
    StatusResponse,
)
from derag.config import DeRAGConfig
from derag.node import DeRAGNode

logger = logging.getLogger("derag.api")


class DeRAGAPI:
    """
    Stateful wrapper around the FastAPI app and the DeRAGNode.

    Usage:
        api = DeRAGAPI(data_dir="/var/derag", auth_token="secret")
        app = api.app
        # Run with: uvicorn derag.api.server:app
    """

    def __init__(
        self,
        data_dir: str = "./derag_data",
        auth_token: Optional[str] = None,
        rate_limit: int = 100,
        cors_origins: Optional[list[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.auth_token = auth_token or os.environ.get("DERAG_API_TOKEN")
        self.rate_limit = rate_limit
        self.cors_origins = cors_origins or ["*"]
        self._start_time = time.time()

        # Node state (initialized lazily via /init endpoint)
        self.node: Optional[DeRAGNode] = None
        self._config: Optional[DeRAGConfig] = None

        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI(
            title="De-RAG API",
            description=(
                "**Decentralized Privacy-First RAG** — encrypted, sharded, "
                "peer-to-peer knowledge retrieval.\n\n"
                "Part of the [Crux Sovereign AI Stack]"
                "(https://github.com/crux-ecosystem)"
            ),
            version=__version__,
            contact={
                "name": "Mounesh Kodi — CruxLabx",
                "url": "https://github.com/crux-ecosystem/de-rag",
            },
            license_info={
                "name": "AGPL-3.0",
                "url": "https://www.gnu.org/licenses/agpl-3.0.html",
            },
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # ── Middleware (order matters: last added = first executed) ──
        app.add_middleware(RequestLoggingMiddleware)
        app.add_middleware(
            RateLimitMiddleware,
            max_requests=self.rate_limit,
            window_seconds=60,
        )
        app.add_middleware(
            BearerAuthMiddleware,
            token=self.auth_token,
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # ── Register routes ──
        self._register_lifecycle(app)
        self._register_documents(app)
        self._register_query(app)
        self._register_network(app)
        self._register_shards(app)
        self._register_keys(app)
        self._register_audit(app)
        self._register_admin(app)

        return app

    # ── Helpers ────────────────────────────────────────────────

    def _require_node(self) -> DeRAGNode:
        """Raise 503 if node is not initialized."""
        if self.node is None or not self.node.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Node not initialized. POST /init first.",
            )
        return self.node

    # ─────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────

    def _register_lifecycle(self, app: FastAPI):

        @app.get("/health", response_model=HealthResponse, tags=["Lifecycle"])
        async def health():
            """Health check — always returns 200."""
            return HealthResponse(
                status="ok",
                version=__version__,
                timestamp=time.time(),
            )

        @app.get(
            "/status",
            response_model=StatusResponse,
            tags=["Lifecycle"],
            responses={503: {"model": ErrorResponse}},
        )
        async def status():
            """Get node status and statistics."""
            node = self._require_node()
            stats = node.stats
            return StatusResponse(
                node_id=node.node_id,
                node_name=self._config.node_name if self._config else "derag-node",
                version=__version__,
                is_initialized=node.is_initialized,
                uptime_seconds=time.time() - self._start_time,
                documents=stats.get("documents", 0),
                chunks=stats.get("chunks", 0),
                vectors=stats.get("vectors", 0),
                shards=stats.get("shards", 0),
                peers=stats.get("peers", 0),
                audit_entries=stats.get("audit_entries", 0),
                storage_bytes=stats.get("storage_bytes", 0),
                index_type=stats.get("index_type", "flat"),
            )

        @app.post(
            "/init",
            response_model=MessageResponse,
            tags=["Lifecycle"],
            responses={400: {"model": ErrorResponse}},
        )
        async def init_node(req: InitRequest):
            """Initialize the De-RAG node with a master password."""
            if self.node is not None and self.node.is_initialized:
                raise HTTPException(400, "Node already initialized")

            try:
                self._config = DeRAGConfig(
                    node_name=req.node_name or "derag-node",
                    data_dir=str(self.data_dir),
                )
                self._config.ensure_dirs()

                self.node = DeRAGNode(config=self._config)
                await self.node.initialize(password=req.password)

                logger.info("Node initialized: %s", self.node.node_id)
                return MessageResponse(
                    message=f"Node initialized successfully. ID: {self.node.node_id}"
                )
            except Exception as e:
                logger.exception("Init failed")
                raise HTTPException(500, f"Initialization failed: {e}") from e

        @app.post("/shutdown", response_model=MessageResponse, tags=["Lifecycle"])
        async def shutdown():
            """Gracefully shut down the node."""
            if self.node is not None:
                await self.node.shutdown()
                logger.info("Node shut down")
            return MessageResponse(message="Node shut down")

    # ─────────────────────────────────────────────────────────
    # DOCUMENTS
    # ─────────────────────────────────────────────────────────

    def _register_documents(self, app: FastAPI):

        def _doc_to_response(doc) -> DocumentResponse:
            return DocumentResponse(
                doc_id=doc.doc_id,
                filename=doc.filename,
                content_hash=doc.content_hash,
                chunk_count=doc.chunk_count,
                total_size=doc.total_size,
                encrypted_size=doc.encrypted_size,
                mime_type=doc.mime_type,
                metadata=doc.metadata or {},
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )

        @app.get(
            "/documents",
            response_model=DocumentListResponse,
            tags=["Documents"],
        )
        async def list_documents():
            """List all ingested documents."""
            node = self._require_node()
            docs = node._local_store.list_documents()
            return DocumentListResponse(
                documents=[_doc_to_response(d) for d in docs],
                total=len(docs),
            )

        @app.post(
            "/documents",
            response_model=DocumentResponse,
            status_code=201,
            tags=["Documents"],
        )
        async def ingest_text(req: IngestRequest):
            """Ingest raw text content."""
            node = self._require_node()
            try:
                # Write temporary file for node.ingest()
                tmp_path = self.data_dir / "tmp" / req.filename
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_text(req.content, encoding="utf-8")

                doc = await node.ingest(
                    file_path=str(tmp_path),
                    chunk_size=req.chunk_size,
                    chunk_overlap=req.chunk_overlap,
                    metadata=req.metadata,
                )

                # Clean up temp
                tmp_path.unlink(missing_ok=True)

                return _doc_to_response(doc)
            except Exception as e:
                logger.exception("Ingest failed")
                raise HTTPException(500, f"Ingestion failed: {e}") from e

        @app.post(
            "/documents/upload",
            response_model=DocumentResponse,
            status_code=201,
            tags=["Documents"],
        )
        async def upload_file(
            file: UploadFile = File(...),
            chunk_size: int = Query(512, ge=64, le=8192),
            chunk_overlap: int = Query(50, ge=0, le=512),
        ):
            """Upload and ingest a file."""
            node = self._require_node()
            try:
                tmp_path = self.data_dir / "tmp" / (file.filename or "upload.bin")
                tmp_path.parent.mkdir(parents=True, exist_ok=True)

                content = await file.read()
                tmp_path.write_bytes(content)

                doc = await node.ingest(
                    file_path=str(tmp_path),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                tmp_path.unlink(missing_ok=True)
                return _doc_to_response(doc)
            except Exception as e:
                logger.exception("Upload ingest failed")
                raise HTTPException(500, f"Upload failed: {e}") from e

        @app.get(
            "/documents/{doc_id}",
            response_model=DocumentResponse,
            tags=["Documents"],
            responses={404: {"model": ErrorResponse}},
        )
        async def get_document(doc_id: str):
            """Get document metadata by ID."""
            node = self._require_node()
            doc = node._local_store.get_document(doc_id)
            if not doc:
                raise HTTPException(404, f"Document {doc_id} not found")
            return _doc_to_response(doc)

        @app.delete(
            "/documents/{doc_id}",
            response_model=MessageResponse,
            tags=["Documents"],
            responses={404: {"model": ErrorResponse}},
        )
        async def delete_document(doc_id: str):
            """Delete a document and its chunks."""
            node = self._require_node()
            success = node._local_store.delete_document(doc_id)
            if not success:
                raise HTTPException(404, f"Document {doc_id} not found")
            return MessageResponse(message=f"Document {doc_id} deleted")

        @app.get(
            "/documents/{doc_id}/chunks",
            response_model=ChunkListResponse,
            tags=["Documents"],
            responses={404: {"model": ErrorResponse}},
        )
        async def get_chunks(doc_id: str):
            """Get all chunks for a document."""
            node = self._require_node()
            doc = node._local_store.get_document(doc_id)
            if not doc:
                raise HTTPException(404, f"Document {doc_id} not found")

            chunks = node._local_store.get_chunks(doc_id)
            return ChunkListResponse(
                doc_id=doc_id,
                chunks=[
                    ChunkResponse(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        chunk_index=c.chunk_index,
                        text=c.text,
                        text_hash=c.text_hash,
                        char_count=c.char_count,
                        token_estimate=c.token_estimate,
                        metadata=c.metadata or {},
                    )
                    for c in chunks
                ],
                total=len(chunks),
            )

    # ─────────────────────────────────────────────────────────
    # QUERY
    # ─────────────────────────────────────────────────────────

    def _register_query(self, app: FastAPI):

        @app.post(
            "/query",
            response_model=QueryResponse,
            tags=["Query"],
            responses={503: {"model": ErrorResponse}},
        )
        async def query(req: QueryRequest):
            """Execute a full RAG query (search + generation)."""
            node = self._require_node()

            try:
                answer = await node.query(
                    question=req.question,
                    distributed=req.distributed,
                )

                return QueryResponse(
                    answer=answer.text,
                    question=answer.question,
                    results=[
                        QueryResultItem(
                            chunk_id=r.chunk_id,
                            document_id=r.document_id,
                            text=r.text,
                            score=r.score,
                            metadata=r.metadata or {},
                            source_peer=r.source_peer,
                        )
                        for r in answer.results
                    ],
                    confidence=answer.confidence,
                    generation_model=answer.generation_model,
                    total_time_ms=answer.total_time_ms,
                    query_time_ms=answer.query_time_ms,
                    generation_time_ms=answer.generation_time_ms,
                    peers_queried=answer.peers_queried,
                    local_results=answer.local_results,
                    remote_results=answer.remote_results,
                )
            except Exception as e:
                logger.exception("Query failed")
                raise HTTPException(500, f"Query failed: {e}") from e

        @app.post(
            "/search",
            response_model=SearchResponse,
            tags=["Query"],
        )
        async def search(req: SearchRequest):
            """Similarity search without generation."""
            node = self._require_node()
            try:
                t0 = time.time()
                results = await node._query_engine.search_only(
                    question=req.question,
                    k=req.k,
                )
                elapsed = (time.time() - t0) * 1000

                return SearchResponse(
                    results=[
                        QueryResultItem(
                            chunk_id=r.chunk_id,
                            document_id=r.document_id,
                            text=r.text,
                            score=r.score,
                            metadata=r.metadata or {},
                            source_peer=getattr(r, "source_peer", None),
                        )
                        for r in results
                    ],
                    total=len(results),
                    query_time_ms=elapsed,
                )
            except Exception as e:
                logger.exception("Search failed")
                raise HTTPException(500, f"Search failed: {e}") from e

    # ─────────────────────────────────────────────────────────
    # NETWORK
    # ─────────────────────────────────────────────────────────

    def _register_network(self, app: FastAPI):

        @app.post(
            "/network/join",
            response_model=MessageResponse,
            tags=["Network"],
        )
        async def join_network(req: JoinNetworkRequest):
            """Join the P2P network."""
            node = self._require_node()
            try:
                await node.join_network(
                    port=req.port,
                    bootstrap_peers=req.bootstrap_peers,
                )
                return MessageResponse(
                    message=f"Joined network on port {req.port}"
                )
            except Exception as e:
                logger.exception("Join network failed")
                raise HTTPException(500, f"Join failed: {e}") from e

        @app.get(
            "/network/peers",
            response_model=PeerListResponse,
            tags=["Network"],
        )
        async def list_peers():
            """List connected peers."""
            node = self._require_node()
            if not hasattr(node, "_peer") or node._peer is None:
                return PeerListResponse(peers=[], total=0)

            peers = node._peer.connected_peers
            return PeerListResponse(
                peers=[
                    PeerResponse(
                        peer_id=p.peer_id,
                        address=p.address,
                        port=p.port,
                        shard_count=p.shard_count,
                        available_space_mb=p.available_space_mb,
                        connected_at=p.connected_at,
                        last_heartbeat=p.last_heartbeat,
                        latency_ms=p.latency_ms,
                        is_alive=p.is_alive,
                    )
                    for p in peers
                ],
                total=len(peers),
            )

        @app.post(
            "/network/disconnect/{peer_id}",
            response_model=MessageResponse,
            tags=["Network"],
        )
        async def disconnect_peer(peer_id: str):
            """Disconnect from a specific peer."""
            node = self._require_node()
            if not hasattr(node, "_peer") or node._peer is None:
                raise HTTPException(400, "Not connected to any network")
            await node._peer.disconnect(peer_id)
            return MessageResponse(message=f"Disconnected from {peer_id}")

    # ─────────────────────────────────────────────────────────
    # SHARDS
    # ─────────────────────────────────────────────────────────

    def _register_shards(self, app: FastAPI):

        @app.post(
            "/shards/{doc_id}",
            response_model=ShardDistribution,
            tags=["Shards"],
        )
        async def distribute_shards(doc_id: str, req: Optional[DistributeRequest] = None):
            """Distribute document shards to the P2P network."""
            node = self._require_node()
            try:
                result = await node.distribute_shards(doc_id)
                return ShardDistribution(
                    doc_id=doc_id,
                    shards_distributed=len(result),
                    distribution=result,
                )
            except Exception as e:
                logger.exception("Shard distribution failed")
                raise HTTPException(500, f"Distribution failed: {e}") from e

    # ─────────────────────────────────────────────────────────
    # KEYS
    # ─────────────────────────────────────────────────────────

    def _register_keys(self, app: FastAPI):

        @app.get(
            "/keys",
            response_model=KeyListResponse,
            tags=["Keys"],
        )
        async def list_keys():
            """List key records (never exposes key material)."""
            node = self._require_node()
            records = []
            keys_dir = self.data_dir / "keys"
            if keys_dir.exists():
                import json
                meta_path = keys_dir / "key_records.json"
                if meta_path.exists():
                    try:
                        data = json.loads(meta_path.read_text())
                        for kr in data:
                            records.append(KeyInfo(
                                key_id=kr.get("key_id", ""),
                                purpose=kr.get("purpose", ""),
                                algorithm=kr.get("algorithm", ""),
                                created_at=kr.get("created_at", 0),
                                is_active=kr.get("is_active", True),
                            ))
                    except Exception:
                        pass

            return KeyListResponse(keys=records, total=len(records))

        @app.post(
            "/keys/rotate",
            response_model=MessageResponse,
            tags=["Keys"],
        )
        async def rotate_keys(req: KeyRotateRequest):
            """Rotate the master encryption key."""
            node = self._require_node()
            try:
                node._key_manager.rotate_master(req.new_password)
                return MessageResponse(message="Master key rotated successfully")
            except Exception as e:
                logger.exception("Key rotation failed")
                raise HTTPException(500, f"Key rotation failed: {e}") from e

    # ─────────────────────────────────────────────────────────
    # AUDIT
    # ─────────────────────────────────────────────────────────

    def _register_audit(self, app: FastAPI):

        def _entry_to_response(e) -> AuditEntryResponse:
            return AuditEntryResponse(
                entry_id=e.entry_id,
                timestamp=e.timestamp,
                action=e.action,
                actor=e.actor,
                target=e.target,
                details=e.details or {},
                entry_hash=e.entry_hash,
            )

        @app.get(
            "/audit",
            response_model=AuditListResponse,
            tags=["Audit"],
        )
        async def get_audit(
            action: Optional[str] = None,
            actor: Optional[str] = None,
            target: Optional[str] = None,
            limit: int = Query(100, ge=1, le=1000),
        ):
            """Query the audit log."""
            node = self._require_node()
            entries = node._audit_log.query(
                action=action,
                actor=actor,
                target=target,
                limit=limit,
            )
            return AuditListResponse(
                entries=[_entry_to_response(e) for e in entries],
                total=len(entries),
            )

        @app.post(
            "/audit",
            response_model=AuditListResponse,
            tags=["Audit"],
        )
        async def query_audit(req: AuditQueryRequest):
            """Query audit log with advanced filters."""
            node = self._require_node()
            entries = node._audit_log.query(
                action=req.action,
                actor=req.actor,
                target=req.target,
                since=req.since,
                until=req.until,
                limit=req.limit,
            )
            return AuditListResponse(
                entries=[_entry_to_response(e) for e in entries],
                total=len(entries),
            )

        @app.get(
            "/audit/verify",
            response_model=AuditVerifyResponse,
            tags=["Audit"],
        )
        async def verify_audit():
            """Verify the Merkle chain integrity of the audit log."""
            node = self._require_node()
            valid, broken_at = node._audit_log.verify_chain()
            return AuditVerifyResponse(
                valid=valid,
                total_entries=node._audit_log.count,
                broken_at=broken_at,
            )

    # ─────────────────────────────────────────────────────────
    # ADMIN DASHBOARD
    # ─────────────────────────────────────────────────────────

    def _register_admin(self, app: FastAPI):

        @app.get(
            "/admin/dashboard",
            response_model=DashboardStats,
            tags=["Admin"],
        )
        async def dashboard():
            """Full aggregated dashboard stats for admin UI."""
            node = self._require_node()
            stats = node.stats

            # System info
            disk = shutil.disk_usage(str(self.data_dir))
            sys_info = {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "disk_total_gb": round(disk.total / (1024 ** 3), 2),
                "disk_free_gb": round(disk.free / (1024 ** 3), 2),
                "data_dir": str(self.data_dir),
            }

            node_status = StatusResponse(
                node_id=node.node_id,
                node_name=self._config.node_name if self._config else "derag-node",
                version=__version__,
                is_initialized=node.is_initialized,
                uptime_seconds=time.time() - self._start_time,
                documents=stats.get("documents", 0),
                chunks=stats.get("chunks", 0),
                vectors=stats.get("vectors", 0),
                shards=stats.get("shards", 0),
                peers=stats.get("peers", 0),
                audit_entries=stats.get("audit_entries", 0),
                storage_bytes=stats.get("storage_bytes", 0),
                index_type=stats.get("index_type", "flat"),
            )

            # Gather per-subsystem stats
            crypto_stats = {}
            storage_stats = stats.get("storage", {})
            network_stats = stats.get("network", {})
            query_stats = {}
            audit_stats = {}

            if hasattr(node, "_local_store"):
                storage_stats = node._local_store.get_stats()
            if hasattr(node, "_query_engine"):
                query_stats = node._query_engine.stats
            if hasattr(node, "_audit_log"):
                audit_stats = node._audit_log.stats
            if hasattr(node, "_vector_index"):
                crypto_stats["vectors"] = node._vector_index.stats

            return DashboardStats(
                node=node_status,
                crypto=crypto_stats,
                storage=storage_stats,
                network=network_stats,
                query=query_stats,
                audit=audit_stats,
                system=sys_info,
            )


# ─── Factory + standalone app ─────────────────────────────────

def create_app(
    data_dir: str = "./derag_data",
    auth_token: Optional[str] = None,
    rate_limit: int = 100,
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """Create a configured FastAPI app for De-RAG."""
    api = DeRAGAPI(
        data_dir=data_dir,
        auth_token=auth_token,
        rate_limit=rate_limit,
        cors_origins=cors_origins,
    )
    return api.app


# Default app instance for `uvicorn derag.api.server:app`
_api_instance = DeRAGAPI(
    data_dir=os.environ.get("DERAG_DATA_DIR", "./derag_data"),
    auth_token=os.environ.get("DERAG_API_TOKEN"),
)
app = _api_instance.app
