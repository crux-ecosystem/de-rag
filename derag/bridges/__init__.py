"""
De-RAG ↔ IntraMind Bridge
===========================

Delegates local RAG operations from De-RAG to IntraMind.

When a De-RAG node needs to:
  - Parse a document → delegates to IntraMind's doc_parser
  - Generate embeddings → delegates to IntraMind's embedding_builder
  - Run vector search → delegates to IntraMind's vector_store
  - Generate an answer → delegates to IntraMind's LLM integration
  - Optimize context → delegates to IntraMind's context_optimizer

This creates a clean separation:
  De-RAG handles: encryption, sharding, P2P, lineage, distributed retrieval
  IntraMind handles: parsing, embedding, vector search, LLM generation

Copyright (c) 2026 CruxLabx — AGPL-3.0
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("derag.bridges.intramind")


@dataclass
class DelegationResult:
    """Result from an IntraMind delegation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    delegated_to: str = "unknown"


class IntraMindDelegator:
    """
    Delegates De-RAG operations to IntraMind components.
    
    Acts as a local RAG accelerator for De-RAG nodes.
    Each De-RAG node can have an IntraMind instance running
    alongside it for fast local operations.
    
    Usage:
        delegator = IntraMindDelegator(edition="library")
        delegator.boot()
        
        # Parse a document
        chunks = delegator.parse_and_chunk("paper.pdf")
        
        # Generate embeddings
        embeddings = delegator.embed(chunks)
        
        # Run RAG query locally
        answer = delegator.query("What is quantum computing?")
    """

    def __init__(self, edition: str = "core"):
        self._edition = edition
        self._engine = None
        self._components = {}
        self._booted = False
        self._delegation_count = 0

    def boot(self) -> dict:
        """Initialize IntraMind components for delegation."""
        results = {"status": "booting", "components": {}}

        try:
            # Find IntraMind
            intramind_paths = [
                Path(__file__).parent.parent.parent / "IntraMind",
                Path(__file__).parent.parent.parent.parent / "IntraMind",
                Path.home() / "Documents" / "Cruxlabx" / "IntraMind",
            ]
            intramind_root = None
            for p in intramind_paths:
                if (p / "core" / "engine.py").exists():
                    intramind_root = p
                    if str(p.parent) not in sys.path:
                        sys.path.insert(0, str(p.parent))
                    break

            if not intramind_root:
                results["status"] = "stub"
                results["reason"] = "IntraMind not found"
                self._booted = True
                return results

            # Import IntraMind engine
            from IntraMind.core.engine import IntraMindEngine
            self._engine = IntraMindEngine(edition=self._edition)
            boot_result = self._engine.boot()
            results["engine"] = boot_result

            # Try to get individual components for fine-grained delegation
            self._import_components()
            results["components"] = {k: "loaded" for k in self._components}
            results["status"] = "live"

        except Exception as e:
            logger.warning(f"IntraMind delegation in stub mode: {e}")
            results["status"] = "stub"
            results["reason"] = str(e)

        self._booted = True
        return results

    def _import_components(self):
        """Import individual IntraMind components for targeted delegation."""
        component_map = {
            "doc_parser": "IntraMind.components.doc_parser",
            "embedding_builder": "IntraMind.components.embedding_builder",
            "vector_store": "IntraMind.components.vector_store",
            "llm": "IntraMind.components.llm_integration",
            "context_optimizer": "IntraMind.components.context_optimizer",
            "encryption": "IntraMind.components.encryption",
            "query_cache": "IntraMind.components.query_cache",
        }

        for name, module_path in component_map.items():
            try:
                module = __import__(module_path, fromlist=[name])
                self._components[name] = module
                logger.debug(f"Loaded IntraMind component: {name}")
            except ImportError as e:
                logger.debug(f"Could not load {name}: {e}")

    # ── Document Operations ────────────────────────────────────

    def parse_document(self, file_path: str) -> DelegationResult:
        """Parse a document using IntraMind's doc_parser."""
        start = time.time()
        self._delegation_count += 1

        if "doc_parser" in self._components:
            try:
                parser = self._components["doc_parser"]
                if hasattr(parser, "DocParser"):
                    dp = parser.DocParser()
                    result = dp.parse_document(file_path)
                    return DelegationResult(
                        success=True,
                        data=result,
                        elapsed_ms=(time.time() - start) * 1000,
                        delegated_to="intramind.doc_parser",
                    )
            except Exception as e:
                return DelegationResult(success=False, error=str(e), delegated_to="intramind.doc_parser")

        # Stub: basic file reading
        try:
            text = Path(file_path).read_text(errors="replace")
            return DelegationResult(
                success=True,
                data={"text": text, "source": file_path},
                elapsed_ms=(time.time() - start) * 1000,
                delegated_to="derag.stub",
            )
        except Exception as e:
            return DelegationResult(success=False, error=str(e), delegated_to="derag.stub")

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> DelegationResult:
        """Chunk text using IntraMind's chunking algorithm."""
        start = time.time()
        self._delegation_count += 1

        if "doc_parser" in self._components:
            try:
                parser = self._components["doc_parser"]
                if hasattr(parser, "DocParser"):
                    dp = parser.DocParser()
                    chunks = dp.chunk_text(text, chunk_size=chunk_size, chunk_overlap=overlap)
                    return DelegationResult(
                        success=True,
                        data=chunks,
                        elapsed_ms=(time.time() - start) * 1000,
                        delegated_to="intramind.doc_parser",
                    )
            except Exception as e:
                pass

        # Stub: simple chunking
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append({"text": chunk, "index": len(chunks), "start": i})

        return DelegationResult(
            success=True,
            data=chunks,
            elapsed_ms=(time.time() - start) * 1000,
            delegated_to="derag.stub",
        )

    # ── Embedding Operations ───────────────────────────────────

    def embed(self, texts: list[str]) -> DelegationResult:
        """Generate embeddings using IntraMind's embedding_builder."""
        start = time.time()
        self._delegation_count += 1

        if "embedding_builder" in self._components:
            try:
                eb_module = self._components["embedding_builder"]
                if hasattr(eb_module, "EmbeddingBuilder"):
                    eb = eb_module.EmbeddingBuilder()
                    embeddings = eb.encode_text(texts)
                    return DelegationResult(
                        success=True,
                        data=embeddings,
                        elapsed_ms=(time.time() - start) * 1000,
                        delegated_to="intramind.embedding_builder",
                    )
            except Exception as e:
                logger.debug(f"Embedding delegation failed: {e}")

        # Stub: random vectors (384-dim like MiniLM)
        import hashlib
        embeddings = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            import random
            rng = random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(384)]
            norm = sum(x * x for x in vec) ** 0.5
            embeddings.append([x / norm for x in vec])

        return DelegationResult(
            success=True,
            data=embeddings,
            elapsed_ms=(time.time() - start) * 1000,
            delegated_to="derag.stub",
        )

    # ── Vector Search ──────────────────────────────────────────

    def vector_search(self, query_embedding: list, top_k: int = 5) -> DelegationResult:
        """Search vectors using IntraMind's vector_store."""
        start = time.time()
        self._delegation_count += 1

        if "vector_store" in self._components:
            try:
                vs_module = self._components["vector_store"]
                if hasattr(vs_module, "VectorStore"):
                    vs = vs_module.VectorStore()
                    results = vs.query(query_embedding, top_k=top_k)
                    return DelegationResult(
                        success=True,
                        data=results,
                        elapsed_ms=(time.time() - start) * 1000,
                        delegated_to="intramind.vector_store",
                    )
            except Exception as e:
                logger.debug(f"Vector search delegation failed: {e}")

        return DelegationResult(
            success=True,
            data=[],
            elapsed_ms=(time.time() - start) * 1000,
            delegated_to="derag.stub",
        )

    # ── LLM Generation ─────────────────────────────────────────

    def generate(self, query: str, context: list[str], system_prompt: str = None) -> DelegationResult:
        """Generate an answer using IntraMind's LLM integration."""
        start = time.time()
        self._delegation_count += 1

        if self._engine:
            try:
                result = self._engine.process(query)
                return DelegationResult(
                    success=True,
                    data={
                        "answer": result.data.get("answer", ""),
                        "sources": result.sources,
                    },
                    elapsed_ms=(time.time() - start) * 1000,
                    delegated_to="intramind.engine",
                )
            except Exception as e:
                logger.debug(f"LLM delegation failed: {e}")

        # Stub response
        return DelegationResult(
            success=True,
            data={
                "answer": f"[stub] Response for: {query}",
                "sources": [],
                "context_used": len(context),
            },
            elapsed_ms=(time.time() - start) * 1000,
            delegated_to="derag.stub",
        )

    # ── Full RAG Pipeline ──────────────────────────────────────

    def full_pipeline(self, query: str, top_k: int = 5) -> DelegationResult:
        """Run the complete IntraMind RAG pipeline."""
        start = time.time()
        self._delegation_count += 1

        if self._engine:
            try:
                result = self._engine.process(query, top_k=top_k)
                return DelegationResult(
                    success=result.success,
                    data={
                        "answer": result.data.get("answer", ""),
                        "sources": result.sources,
                        "cached": result.cached,
                        "edition": result.edition,
                    },
                    elapsed_ms=(time.time() - start) * 1000,
                    delegated_to="intramind.engine",
                )
            except Exception as e:
                return DelegationResult(
                    success=False,
                    error=str(e),
                    elapsed_ms=(time.time() - start) * 1000,
                    delegated_to="intramind.engine",
                )

        return DelegationResult(
            success=True,
            data={"answer": f"[stub] {query}", "sources": []},
            elapsed_ms=(time.time() - start) * 1000,
            delegated_to="derag.stub",
        )

    # ── Context Optimization ───────────────────────────────────

    def optimize_context(self, query: str, contexts: list[str], max_tokens: int = 4096) -> DelegationResult:
        """Optimize retrieved context using IntraMind's Neuro-Weaver."""
        start = time.time()

        if "context_optimizer" in self._components:
            try:
                co_module = self._components["context_optimizer"]
                if hasattr(co_module, "ContextOptimizer"):
                    co = co_module.ContextOptimizer()
                    optimized = co.optimize_prompt(query, contexts, max_tokens=max_tokens)
                    return DelegationResult(
                        success=True,
                        data=optimized,
                        elapsed_ms=(time.time() - start) * 1000,
                        delegated_to="intramind.context_optimizer",
                    )
            except Exception as e:
                pass

        # Stub: truncate to fit
        total = ""
        for ctx in contexts:
            if len(total) + len(ctx) < max_tokens * 4:  # rough chars-to-tokens
                total += ctx + "\n\n"
        return DelegationResult(
            success=True,
            data=total.strip(),
            elapsed_ms=(time.time() - start) * 1000,
            delegated_to="derag.stub",
        )

    # ── Status ─────────────────────────────────────────────────

    @property
    def status(self) -> dict:
        return {
            "booted": self._booted,
            "engine": "live" if self._engine else "stub",
            "edition": self._edition,
            "components": list(self._components.keys()),
            "delegations": self._delegation_count,
        }
