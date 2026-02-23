"""
De-RAG Node — The main orchestrator
=====================================

A DeRAGNode is a single instance of the De-RAG system.
It wires together all subsystems:
  - Crypto engine (envelope encryption)
  - Key manager (key hierarchy)
  - Local store (encrypted documents)
  - Vector index (encrypted FAISS)
  - Shard manager (Shamir's Secret Sharing)
  - P2P network (peer communication)
  - Query engine (distributed search)
  - Audit log (tamper-proof lineage)

Lifecycle:
  node = DeRAGNode(config)
  await node.initialize("password")
  doc = await node.ingest("file.pdf")
  answer = await node.query("What is...?")
  await node.shutdown()

Copyright (c) 2026 CruxLabx
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from derag.config import DeRAGConfig
from derag.crypto.envelope import EnvelopeEngine
from derag.crypto.keys import KeyManager
from derag.storage.vector_index import EncryptedVectorIndex, VectorRecord
from derag.storage.local_store import LocalStore, Chunk, Document
from derag.storage.shard_manager import ShardManager
from derag.network.peer import DeRAGPeer
from derag.query.engine import QueryEngine, Answer
from derag.lineage.audit import MerkleAuditLog

import blake3

logger = logging.getLogger("derag.node")


class DeRAGNode:
    """
    A De-RAG node — the primary interface to the entire system.
    
    Usage:
        config = DeRAGConfig(node_name="my-node")
        node = DeRAGNode(config)
        await node.initialize("my-strong-password")
        
        # Ingest documents
        doc = await node.ingest("path/to/document.pdf")
        
        # Query
        answer = await node.query("What is the renewal date?")
        
        # Join network
        await node.join_network(bootstrap_peers=["192.168.1.100:9090"])
        
        # Distribute shards
        await node.distribute_shards(doc.doc_id)
        
        # Shutdown
        await node.shutdown()
    """

    def __init__(self, config: Optional[DeRAGConfig] = None):
        self.config = config or DeRAGConfig()
        self._initialized = False
        self._start_time = 0.0

        # Subsystems (initialized in .initialize())
        self._key_manager: Optional[KeyManager] = None
        self._engine: Optional[EnvelopeEngine] = None
        self._store: Optional[LocalStore] = None
        self._index: Optional[EncryptedVectorIndex] = None
        self._shard_manager: Optional[ShardManager] = None
        self._peer: Optional[DeRAGPeer] = None
        self._query_engine: Optional[QueryEngine] = None
        self._audit: Optional[MerkleAuditLog] = None

        # Embedding function (set by user or auto-loaded)
        self._embed_fn: Optional[callable] = None
        self._generate_fn: Optional[callable] = None

    @property
    def node_id(self) -> str:
        if self._key_manager:
            return self._key_manager.get_node_id()
        return self.config.node_id or "uninitialized"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # --- Lifecycle ---

    async def initialize(self, password: str) -> None:
        """
        Initialize all subsystems.
        
        This is the "boot sequence" for a De-RAG node.
        """
        t_start = time.time()
        self.config.ensure_dirs()

        # 1. Key management
        logger.info("Initializing key management...")
        self._key_manager = KeyManager(self.config.data_dir / "keys")
        self._key_manager.initialize(password)
        self._engine = self._key_manager.get_envelope_engine()

        # 2. Local document store
        logger.info("Initializing local store...")
        self._store = LocalStore(
            db_path=self.config.storage.data_dir / "documents.db",
            engine=self._engine,
        )

        # 3. Vector index
        logger.info("Initializing vector index...")
        index_dir = self.config.data_dir / "index"
        if (index_dir / "meta.json").exists():
            self._index = EncryptedVectorIndex.load(index_dir, self._engine)
            logger.info(f"Loaded existing index with {self._index.count} vectors")
        else:
            self._index = EncryptedVectorIndex(
                dim=self.config.storage.embedding_dim,
                engine=self._engine,
                index_type=self.config.storage.index_type.value,
                hnsw_m=self.config.storage.hnsw_m,
                hnsw_ef_construction=self.config.storage.hnsw_ef_construction,
                hnsw_ef_search=self.config.storage.hnsw_ef_search,
            )

        # 4. Shard manager
        self._shard_manager = ShardManager(
            n=self.config.sharding.total_shards,
            k=self.config.sharding.threshold,
        )

        # 5. Audit log
        logger.info("Initializing audit log...")
        self._audit = MerkleAuditLog(
            log_dir=self.config.data_dir / "lineage",
            node_id=self.node_id,
        )

        # 6. Query engine
        self._query_engine = QueryEngine(
            vector_index=self._index,
            local_store=self._store,
            crypto_engine=self._engine,
            top_k=self.config.query.top_k,
            min_confidence=self.config.query.min_confidence,
            max_context_tokens=self.config.query.max_context_tokens,
        )

        self._initialized = True
        self._start_time = time.time()

        # Log initialization
        self._audit.append(
            action="node_init",
            target=self.node_id,
            details={
                "node_name": self.config.node_name,
                "elapsed_ms": (time.time() - t_start) * 1000,
            },
        )

        logger.info(
            f"De-RAG node '{self.config.node_name}' initialized "
            f"(ID: {self.node_id[:12]}...) in {(time.time() - t_start)*1000:.0f}ms"
        )

    async def shutdown(self) -> None:
        """Gracefully shut down all subsystems."""
        self._assert_initialized()
        logger.info("Shutting down De-RAG node...")

        # Flush index to disk (encrypted)
        if self._index:
            self._index.flush(self.config.data_dir / "index")

        # Stop P2P network
        if self._peer:
            await self._peer.stop()

        # Close store
        if self._store:
            self._store.close()

        # Log shutdown
        if self._audit:
            self._audit.append(
                action="node_shutdown",
                target=self.node_id,
                details={"uptime_sec": time.time() - self._start_time},
            )

        self._initialized = False
        logger.info("De-RAG node shut down cleanly")

    # --- Document Operations ---

    async def ingest(
        self,
        file_path: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        metadata: Optional[dict] = None,
    ) -> Document:
        """
        Ingest a document: parse, chunk, embed, encrypt, store.
        
        Returns the Document record with all metadata.
        """
        self._assert_initialized()
        t_start = time.time()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_bytes()
        text = content.decode("utf-8", errors="replace")

        # Chunk the text
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        logger.info(f"Chunked {path.name} into {len(chunks)} chunks")

        # Embed chunks
        if self._embed_fn:
            embeddings = [self._embed_fn(c.text) for c in chunks]
        else:
            # Fallback: random embeddings for testing (replace with real model)
            logger.warning("No embedding function set — using random vectors for testing")
            embeddings = [
                np.random.randn(self.config.storage.embedding_dim).astype(np.float32)
                for _ in chunks
            ]

        # Store document (encrypted)
        doc = self._store.ingest_document(
            filename=path.name,
            content=content,
            chunks=chunks,
            mime_type=self._guess_mime(path),
            metadata=metadata,
        )

        # Add to vector index
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            record = VectorRecord(
                vector_id=f"{doc.doc_id}:vec-{i}",
                document_id=doc.doc_id,
                chunk_index=i,
                chunk_text_hash=chunk.text_hash,
                metadata={"chunk_id": chunk.chunk_id},
            )
            vec = np.array(embedding, dtype=np.float32) if not isinstance(embedding, np.ndarray) else embedding
            self._index.add(record.vector_id, vec, record)

        # Audit
        self._audit.append(
            action="ingest",
            target=doc.doc_id,
            details={
                "filename": path.name,
                "chunks": len(chunks),
                "size_bytes": len(content),
                "elapsed_ms": (time.time() - t_start) * 1000,
            },
        )

        logger.info(
            f"Ingested {path.name}: {len(chunks)} chunks, "
            f"{len(content)} bytes in {(time.time()-t_start)*1000:.0f}ms"
        )
        return doc

    async def query(self, question: str, distributed: bool = False) -> Answer:
        """Query the knowledge base."""
        self._assert_initialized()

        answer = await self._query_engine.query(question, distributed=distributed)

        self._audit.append(
            action="query",
            target="",
            details={
                "question_hash": blake3.blake3(question.encode()).hexdigest()[:16],
                "results": len(answer.results),
                "confidence": answer.confidence,
                "distributed": distributed,
                "total_time_ms": answer.total_time_ms,
            },
        )

        return answer

    async def distribute_shards(self, doc_id: str) -> list[dict]:
        """
        Shard a document and distribute to peers.
        
        Returns list of shard placement info.
        """
        self._assert_initialized()
        if not self._peer or self._peer.peer_count == 0:
            raise RuntimeError("No peers connected — cannot distribute shards")

        # Get document chunks (encrypted)
        chunks = self._store.get_chunks(doc_id)
        if not chunks:
            raise ValueError(f"Document {doc_id} not found or has no chunks")

        # Serialize all chunk data
        all_data = b""
        for chunk in chunks:
            all_data += chunk.text.encode("utf-8") + b"\x00"

        # Encrypt the combined data
        encrypted = self._engine.encrypt(all_data, document_id=doc_id)
        encrypted_bytes = encrypted.serialize()

        # Split into shards
        shards = self._shard_manager.split(encrypted_bytes, doc_id)

        # Distribute to peers
        peers = self._peer.connected_peers
        placements = []
        for i, shard in enumerate(shards):
            target_peer = peers[i % len(peers)]
            # TODO: Actually send shard to peer via network
            placements.append({
                "shard_id": shard.shard_id,
                "shard_index": shard.shard_index,
                "peer_id": target_peer.peer_id,
                "data_hash": shard.data_hash,
                "size_bytes": len(shard.data),
            })

        self._audit.append(
            action="distribute_shards",
            target=doc_id,
            details={
                "total_shards": len(shards),
                "threshold": self._shard_manager.k,
                "peers_used": len(set(p["peer_id"] for p in placements)),
            },
        )

        return placements

    # --- Network ---

    async def join_network(
        self,
        port: int = 9090,
        bootstrap_peers: Optional[list[str]] = None,
    ) -> None:
        """Start P2P networking and connect to bootstrap peers."""
        self._assert_initialized()

        self._peer = DeRAGPeer(
            node_id=self.node_id,
            port=port,
            identity_key=self._key_manager.get_node_identity(),
            max_peers=self.config.network.max_peers,
            heartbeat_interval=self.config.network.heartbeat_interval_sec,
        )

        await self._peer.start()
        self._query_engine.set_network(self._peer)

        # Connect to bootstrap peers
        if bootstrap_peers:
            for peer_addr in bootstrap_peers:
                host, p = peer_addr.rsplit(":", 1)
                await self._peer.connect_to(host, int(p))

        self._audit.append(
            action="join_network",
            target=self.node_id,
            details={
                "port": port,
                "bootstrap_peers": len(bootstrap_peers or []),
            },
        )

    # --- Configuration ---

    def set_embedding_fn(self, fn: callable) -> None:
        """Set the embedding function for text → vector conversion."""
        self._embed_fn = fn
        if self._query_engine:
            self._query_engine.set_embedding_fn(fn)

    def set_generation_fn(self, fn: callable) -> None:
        """Set the LLM generation function."""
        self._generate_fn = fn
        if self._query_engine:
            self._query_engine.set_generation_fn(fn)

    # --- Stats ---

    @property
    def stats(self) -> dict:
        if not self._initialized:
            return {"status": "not_initialized"}
        return {
            "status": "running",
            "node_id": self.node_id,
            "node_name": self.config.node_name,
            "uptime_sec": time.time() - self._start_time,
            "store": self._store.get_stats() if self._store else {},
            "index": self._index.stats if self._index else {},
            "network": self._peer.stats if self._peer else {"peer_count": 0},
            "audit": self._audit.stats if self._audit else {},
        }

    # --- Helpers ---

    def _assert_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Node not initialized — call await node.initialize(password) first")

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[Chunk]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunk_hash = blake3.blake3(chunk_text.encode("utf-8")).hexdigest()

            chunks.append(Chunk(
                chunk_id="",  # Set by store
                doc_id="",    # Set by store
                chunk_index=len(chunks),
                text=chunk_text,
                text_hash=chunk_hash,
                char_count=len(chunk_text),
                token_estimate=int(len(chunk_text.split()) * 1.3),
            ))

            if end >= len(words):
                break
            start = end - overlap

        return chunks

    @staticmethod
    def _guess_mime(path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".json": "application/json",
            ".csv": "text/csv",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".html": "text/html",
        }.get(suffix, "application/octet-stream")
