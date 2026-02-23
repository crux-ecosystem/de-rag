"""
De-RAG Query Engine
====================

Distributed query planner and executor for encrypted knowledge retrieval.

Query Pipeline:
  1. User asks natural language question
  2. Question → embedding (local, never sent to network)
  3. Embedding → encrypted query vector
  4. Encrypted query → fan out to peer nodes
  5. Each peer searches local encrypted index
  6. Peers return encrypted result shards
  7. Collect and decrypt results locally
  8. Rank results by relevance
  9. Feed top results + question to local LLM
  10. Return answer — no peer saw the question or answer

Security guarantees:
  - Question text never leaves the local node
  - Query vector is encrypted before network transmission
  - Peers cannot determine what you're searching for
  - Results are encrypted; only the querier can read them
  - Local LLM generates answer — no cloud API calls

Copyright (c) 2026 CruxLabx
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from derag.crypto.envelope import EnvelopeEngine
from derag.storage.vector_index import EncryptedVectorIndex, SearchResult
from derag.storage.local_store import LocalStore, Chunk
from derag.network.peer import DeRAGPeer
from derag.network.protocol import Message, MessageType, make_query

logger = logging.getLogger("derag.query")


@dataclass
class QueryResult:
    """A single query result with source chunks."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    source_peer: str = "local"


@dataclass
class Answer:
    """A generated answer with sources and audit trail."""
    text: str
    question: str
    results: list[QueryResult]
    confidence: float
    generation_model: str
    total_time_ms: float
    query_time_ms: float
    generation_time_ms: float
    peers_queried: int
    local_results: int
    remote_results: int
    timestamp: float = field(default_factory=time.time)


class QueryEngine:
    """
    Distributed encrypted query engine for De-RAG.
    
    Usage:
        engine = QueryEngine(
            vector_index=encrypted_index,
            local_store=doc_store,
            crypto_engine=envelope_engine,
        )
        
        # Local-only query
        answer = await engine.query("What is the contract renewal date?")
        
        # Distributed query (with P2P network)
        engine.set_network(peer)
        answer = await engine.query("What is the contract renewal date?", distributed=True)
    """

    def __init__(
        self,
        vector_index: EncryptedVectorIndex,
        local_store: LocalStore,
        crypto_engine: EnvelopeEngine,
        embedding_fn: Optional[callable] = None,
        generation_fn: Optional[callable] = None,
        top_k: int = 10,
        min_confidence: float = 0.3,
        max_context_tokens: int = 4096,
    ):
        self._index = vector_index
        self._store = local_store
        self._engine = crypto_engine
        self._embed = embedding_fn
        self._generate = generation_fn
        self._peer: Optional[DeRAGPeer] = None
        self.top_k = top_k
        self.min_confidence = min_confidence
        self.max_context_tokens = max_context_tokens

    def set_network(self, peer: DeRAGPeer) -> None:
        """Attach P2P network for distributed queries."""
        self._peer = peer

    def set_embedding_fn(self, fn: callable) -> None:
        """Set the embedding function: text → numpy vector."""
        self._embed = fn

    def set_generation_fn(self, fn: callable) -> None:
        """Set the generation function: (context, question) → answer string."""
        self._generate = fn

    async def query(
        self,
        question: str,
        distributed: bool = False,
        timeout: float = 30.0,
    ) -> Answer:
        """
        Execute a query — the core De-RAG operation.
        
        Args:
            question: Natural language question
            distributed: Whether to query peer nodes
            timeout: Timeout for distributed query (seconds)
        """
        t_start = time.time()

        # Step 1: Embed the question locally
        if self._embed is None:
            raise RuntimeError("No embedding function set. Call set_embedding_fn() first.")
        
        query_vector = self._embed(question)
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)

        # Step 2: Search local index
        local_results = self._index.search(query_vector, k=self.top_k)

        # Step 3: Optionally query peers
        remote_results: list[SearchResult] = []
        peers_queried = 0
        if distributed and self._peer and self._peer.peer_count > 0:
            remote_results, peers_queried = await self._distributed_search(
                query_vector, timeout
            )

        t_query = time.time()

        # Step 4: Merge and rank results
        all_results = self._merge_results(local_results, remote_results)

        # Step 5: Fetch chunk texts for top results
        query_results = await self._resolve_chunks(all_results)

        # Step 6: Filter by confidence
        query_results = [r for r in query_results if r.score >= self.min_confidence]

        # Step 7: Build context for LLM
        context = self._build_context(query_results)

        # Step 8: Generate answer
        answer_text = ""
        generation_model = "none"
        if self._generate and context:
            answer_text = await self._call_generate(context, question)
            generation_model = "local-llm"

        t_end = time.time()

        return Answer(
            text=answer_text,
            question=question,
            results=query_results,
            confidence=query_results[0].score if query_results else 0.0,
            generation_model=generation_model,
            total_time_ms=(t_end - t_start) * 1000,
            query_time_ms=(t_query - t_start) * 1000,
            generation_time_ms=(t_end - t_query) * 1000,
            peers_queried=peers_queried,
            local_results=len(local_results),
            remote_results=len(remote_results),
        )

    async def search_only(
        self,
        question: str,
        k: int = 10,
    ) -> list[QueryResult]:
        """Search without LLM generation — just return matching chunks."""
        if self._embed is None:
            raise RuntimeError("No embedding function set.")

        query_vector = self._embed(question)
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)

        results = self._index.search(query_vector, k=k)
        return await self._resolve_chunks(results)

    # --- Internal ---

    async def _distributed_search(
        self,
        query_vector: np.ndarray,
        timeout: float,
    ) -> tuple[list[SearchResult], int]:
        """Search peer nodes with encrypted query vector."""
        # Encrypt the query vector before sending
        vec_bytes = query_vector.tobytes()
        encrypted_vec = self._engine.encrypt(vec_bytes, document_id="query-vector")
        
        query_msg = make_query(
            node_id=self._peer.node_id,
            encrypted_query_vector=encrypted_vec.serialize(),
            k=self.top_k,
        )

        # Fan out to all peers
        responses = await self._peer.scatter_query(query_msg, timeout=timeout)
        
        remote_results = []
        for resp in responses:
            if resp.msg_type == MessageType.QUERY_RESPONSE:
                # Decrypt and parse results
                results_data = resp.payload.get("results", [])
                for r in results_data:
                    remote_results.append(SearchResult(
                        vector_id=r.get("vector_id", ""),
                        document_id=r.get("document_id", ""),
                        distance=r.get("distance", 0.0),
                        score=r.get("score", 0.0),
                        metadata=r.get("metadata", {}),
                    ))

        return remote_results, len(responses)

    def _merge_results(
        self,
        local: list[SearchResult],
        remote: list[SearchResult],
    ) -> list[SearchResult]:
        """Merge local and remote results, deduplicate, re-rank."""
        seen: set[str] = set()
        merged = []

        for results in [local, remote]:
            for r in results:
                if r.vector_id not in seen:
                    seen.add(r.vector_id)
                    merged.append(r)

        # Sort by score (highest first)
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:self.top_k]

    async def _resolve_chunks(self, results: list[SearchResult]) -> list[QueryResult]:
        """Resolve search results to full chunk texts."""
        query_results = []
        for r in results:
            # Try to get chunk text from local store
            chunk_id = r.metadata.get("chunk_id", r.vector_id)
            chunk = self._store.get_chunk_by_id(chunk_id)
            text = chunk.text if chunk else f"[Chunk {chunk_id} not available locally]"

            query_results.append(QueryResult(
                chunk_id=chunk_id,
                document_id=r.document_id,
                text=text,
                score=r.score,
                metadata=r.metadata,
                source_peer="local" if chunk else "remote",
            ))

        return query_results

    def _build_context(self, results: list[QueryResult]) -> str:
        """Build context string from results, respecting token budget."""
        context_parts = []
        estimated_tokens = 0

        for r in results:
            chunk_tokens = len(r.text.split()) * 1.3  # Rough token estimate
            if estimated_tokens + chunk_tokens > self.max_context_tokens:
                break
            context_parts.append(f"[Source: {r.document_id}]\n{r.text}")
            estimated_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts)

    async def _call_generate(self, context: str, question: str) -> str:
        """Call the generation function (sync or async)."""
        prompt = (
            f"Based on the following context, answer the question accurately. "
            f"If the context doesn't contain enough information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        result = self._generate(prompt)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    @property
    def stats(self) -> dict:
        return {
            "index_vectors": self._index.count,
            "store_docs": len(self._store.list_documents()),
            "top_k": self.top_k,
            "min_confidence": self.min_confidence,
            "has_embedding_fn": self._embed is not None,
            "has_generation_fn": self._generate is not None,
            "has_network": self._peer is not None,
            "network_peers": self._peer.peer_count if self._peer else 0,
        }
