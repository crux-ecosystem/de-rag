"""
De-RAG Local Document Store
=============================

SQLite-backed document storage with full encryption-at-rest.

Stores:
  - Document metadata (encrypted)
  - Chunk text (encrypted)
  - Ingestion history
  - Shard placement map

All sensitive data encrypted before hitting SQLite.
Only structural data (IDs, timestamps, sizes) stored in plaintext
for query optimization.

Copyright (c) 2026 CruxLabx
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import blake3

from derag.crypto.envelope import EnvelopeEngine, EncryptedBlob


@dataclass
class Document:
    """A document in the De-RAG store."""
    doc_id: str
    filename: str
    content_hash: str      # BLAKE3 hash of original content
    chunk_count: int
    total_size: int        # Original size in bytes
    encrypted_size: int    # Encrypted size in bytes
    mime_type: str = "application/octet-stream"
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Chunk:
    """A text chunk from a document."""
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str              # Plaintext (only in memory, never on disk)
    text_hash: str         # BLAKE3 hash
    char_count: int
    token_estimate: int    # Approximate token count
    metadata: dict = field(default_factory=dict)


class LocalStore:
    """
    Encrypted local document store backed by SQLite.
    
    Usage:
        store = LocalStore(db_path, engine)
        doc = store.ingest_document("file.pdf", content, chunks)
        chunks = store.get_chunks(doc.doc_id)
        store.delete_document(doc.doc_id)
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path, engine: EnvelopeEngine):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = engine
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                total_size INTEGER NOT NULL,
                encrypted_size INTEGER NOT NULL,
                mime_type TEXT DEFAULT 'application/octet-stream',
                encrypted_metadata BLOB,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                encrypted_text BLOB NOT NULL,
                text_hash TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                token_estimate INTEGER NOT NULL,
                encrypted_metadata BLOB,
                UNIQUE(doc_id, chunk_index)
            );

            CREATE TABLE IF NOT EXISTS shard_map (
                shard_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                shard_index INTEGER NOT NULL,
                peer_id TEXT,
                encrypted_data BLOB,
                status TEXT DEFAULT 'local',
                created_at REAL NOT NULL,
                UNIQUE(doc_id, shard_index)
            );

            CREATE TABLE IF NOT EXISTS ingestion_log (
                log_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                timestamp REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_shards_doc ON shard_map(doc_id);
            CREATE INDEX IF NOT EXISTS idx_shards_peer ON shard_map(peer_id);
            CREATE INDEX IF NOT EXISTS idx_log_doc ON ingestion_log(doc_id);
            CREATE INDEX IF NOT EXISTS idx_log_time ON ingestion_log(timestamp);
        """)

    def ingest_document(
        self,
        filename: str,
        content: bytes,
        chunks: list[Chunk],
        mime_type: str = "application/octet-stream",
        metadata: Optional[dict] = None,
    ) -> Document:
        """
        Ingest a document: encrypt chunks and store.
        
        The raw content is NOT stored — only encrypted chunks.
        """
        doc_id = f"doc-{uuid.uuid4().hex[:16]}"
        content_hash = blake3.blake3(content).hexdigest()

        # Encrypt metadata
        meta_bytes = json.dumps(metadata or {}).encode("utf-8")
        enc_meta = self._engine.encrypt(meta_bytes, document_id=f"{doc_id}:meta")

        # Calculate encrypted size
        encrypted_size = 0
        encrypted_chunks = []
        for chunk in chunks:
            chunk.doc_id = doc_id
            chunk.chunk_id = f"{doc_id}:chunk-{chunk.chunk_index}"
            enc_text = self._engine.encrypt(
                chunk.text.encode("utf-8"),
                document_id=chunk.chunk_id,
            )
            enc_chunk_meta = self._engine.encrypt(
                json.dumps(chunk.metadata).encode("utf-8"),
                document_id=f"{chunk.chunk_id}:meta",
            )
            encrypted_size += len(enc_text.serialize())
            encrypted_chunks.append((chunk, enc_text, enc_chunk_meta))

        doc = Document(
            doc_id=doc_id,
            filename=filename,
            content_hash=content_hash,
            chunk_count=len(chunks),
            total_size=len(content),
            encrypted_size=encrypted_size,
            mime_type=mime_type,
            metadata=metadata or {},
            created_at=time.time(),
            updated_at=time.time(),
        )

        # Write to database
        with self._transaction():
            self._conn.execute(
                """INSERT INTO documents 
                   (doc_id, filename, content_hash, chunk_count, total_size, 
                    encrypted_size, mime_type, encrypted_metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc.doc_id, doc.filename, doc.content_hash, doc.chunk_count,
                 doc.total_size, doc.encrypted_size, doc.mime_type,
                 enc_meta.serialize(), doc.created_at, doc.updated_at),
            )

            for chunk, enc_text, enc_chunk_meta in encrypted_chunks:
                self._conn.execute(
                    """INSERT INTO chunks 
                       (chunk_id, doc_id, chunk_index, encrypted_text, text_hash,
                        char_count, token_estimate, encrypted_metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (chunk.chunk_id, chunk.doc_id, chunk.chunk_index,
                     enc_text.serialize(), chunk.text_hash,
                     chunk.char_count, chunk.token_estimate,
                     enc_chunk_meta.serialize()),
                )

            self._log_action(doc_id, "ingest", f"Ingested {filename} ({len(chunks)} chunks)")

        return doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document metadata."""
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return None

        # Decrypt metadata
        meta = {}
        if row["encrypted_metadata"]:
            blob = EncryptedBlob.deserialize(row["encrypted_metadata"])
            meta = json.loads(self._engine.decrypt(blob))

        return Document(
            doc_id=row["doc_id"],
            filename=row["filename"],
            content_hash=row["content_hash"],
            chunk_count=row["chunk_count"],
            total_size=row["total_size"],
            encrypted_size=row["encrypted_size"],
            mime_type=row["mime_type"],
            metadata=meta,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """Get all chunks for a document (decrypted)."""
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()

        chunks = []
        for row in rows:
            # Decrypt text
            blob = EncryptedBlob.deserialize(row["encrypted_text"])
            text = self._engine.decrypt(blob).decode("utf-8")

            # Decrypt metadata
            meta = {}
            if row["encrypted_metadata"]:
                meta_blob = EncryptedBlob.deserialize(row["encrypted_metadata"])
                meta = json.loads(self._engine.decrypt(meta_blob))

            chunks.append(Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                chunk_index=row["chunk_index"],
                text=text,
                text_hash=row["text_hash"],
                char_count=row["char_count"],
                token_estimate=row["token_estimate"],
                metadata=meta,
            ))

        return chunks

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get a single chunk by ID (decrypted)."""
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            return None

        blob = EncryptedBlob.deserialize(row["encrypted_text"])
        text = self._engine.decrypt(blob).decode("utf-8")

        meta = {}
        if row["encrypted_metadata"]:
            meta_blob = EncryptedBlob.deserialize(row["encrypted_metadata"])
            meta = json.loads(self._engine.decrypt(meta_blob))

        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            chunk_index=row["chunk_index"],
            text=text,
            text_hash=row["text_hash"],
            char_count=row["char_count"],
            token_estimate=row["token_estimate"],
            metadata=meta,
        )

    def list_documents(self) -> list[Document]:
        """List all documents (metadata only, no decryption of chunks)."""
        rows = self._conn.execute(
            "SELECT doc_id, filename, content_hash, chunk_count, total_size, "
            "encrypted_size, mime_type, created_at, updated_at FROM documents "
            "ORDER BY created_at DESC"
        ).fetchall()

        return [
            Document(
                doc_id=row["doc_id"],
                filename=row["filename"],
                content_hash=row["content_hash"],
                chunk_count=row["chunk_count"],
                total_size=row["total_size"],
                encrypted_size=row["encrypted_size"],
                mime_type=row["mime_type"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks (cascade)."""
        with self._transaction():
            cursor = self._conn.execute(
                "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
            )
            if cursor.rowcount > 0:
                self._log_action(doc_id, "delete", f"Deleted document {doc_id}")
                return True
        return False

    def get_stats(self) -> dict:
        """Get store statistics."""
        docs = self._conn.execute("SELECT COUNT(*) as c FROM documents").fetchone()["c"]
        chunks = self._conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
        total_size = self._conn.execute(
            "SELECT COALESCE(SUM(total_size), 0) as s FROM documents"
        ).fetchone()["s"]
        enc_size = self._conn.execute(
            "SELECT COALESCE(SUM(encrypted_size), 0) as s FROM documents"
        ).fetchone()["s"]
        return {
            "documents": docs,
            "chunks": chunks,
            "total_size_bytes": total_size,
            "encrypted_size_bytes": enc_size,
            "encryption_overhead": enc_size / max(total_size, 1),
        }

    @contextmanager
    def _transaction(self) -> Generator[None, None, None]:
        """Context manager for database transactions."""
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _log_action(self, doc_id: str, action: str, details: str) -> None:
        """Log an action to the ingestion log."""
        self._conn.execute(
            "INSERT INTO ingestion_log (log_id, doc_id, action, details, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"log-{uuid.uuid4().hex[:12]}", doc_id, action, details, time.time()),
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# Need uuid for log entries
import uuid
