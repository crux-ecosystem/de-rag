"""
MOL Bridge — Execute De-RAG operations from MOL language
=========================================================

This module lets MOL programs interact with De-RAG nodes:
  - Define encryption pipelines in MOL
  - Trigger ingestion from MOL scripts
  - Execute queries from MOL
  - Configure shard policies in MOL syntax

Architecture:
  MOL script → mol interpreter → mol_bridge → DeRAGNode → crypto/storage/network

Copyright (c) 2026 CruxLabx — AGPL-3.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("derag.mol_bridge")


@dataclass
class MOLResult:
    """Result from executing a MOL script."""
    success: bool
    output: str = ""
    error: str = ""
    return_value: Any = None
    execution_time_ms: float = 0.0


@dataclass
class MOLPipeline:
    """A De-RAG pipeline defined in MOL language."""
    name: str
    mol_source: str
    compiled: bool = False
    metadata: dict = field(default_factory=dict)


class MOLBridge:
    """
    Bridge between MOL language and De-RAG subsystems.
    
    Registers De-RAG functions as MOL stdlib extensions.
    Provides bidirectional communication:
      - MOL → De-RAG: Call node operations from MOL
      - De-RAG → MOL: Execute MOL pipelines from Python
    """

    # MOL interpreter search paths
    MOL_SEARCH_PATHS = [
        Path(__file__).parent.parent.parent / "MOL",
        Path(__file__).parent.parent.parent / "mol-lang",
        Path.home() / ".mol",
        Path("/usr/local/lib/mol"),
    ]

    def __init__(self, node=None, mol_path: Optional[Path] = None):
        self._node = node
        self._mol_path = mol_path or self._find_mol()
        self._pipelines: dict[str, MOLPipeline] = {}
        self._stdlib_extensions: dict[str, callable] = {}

        # Register De-RAG functions as MOL-callable extensions
        self._register_builtins()

    def _find_mol(self) -> Optional[Path]:
        """Locate the MOL interpreter."""
        # Check if mol is in PATH
        try:
            result = subprocess.run(
                ["which", "mol"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Search known locations
        for search_path in self.MOL_SEARCH_PATHS:
            # Look for the mol interpreter/runner
            for entry_name in ["mol/interpreter.py", "mol/runner.py", "mol"]:
                candidate = search_path / entry_name
                if candidate.exists():
                    logger.info(f"Found MOL at {candidate}")
                    return candidate

        logger.warning("MOL interpreter not found — MOL bridge in limited mode")
        return None

    @property
    def mol_available(self) -> bool:
        return self._mol_path is not None

    def _register_builtins(self):
        """Register De-RAG operations as callable from MOL."""
        self._stdlib_extensions = {
            "derag_encrypt": self._mol_encrypt,
            "derag_decrypt": self._mol_decrypt,
            "derag_hash": self._mol_hash,
            "derag_ingest": self._mol_ingest,
            "derag_query": self._mol_query,
            "derag_shard": self._mol_shard,
            "derag_status": self._mol_status,
        }

    # --- MOL Execution ---

    def execute(self, mol_source: str, context: Optional[dict] = None) -> MOLResult:
        """Execute a MOL script and return the result."""
        import time
        start = time.time()

        if not self.mol_available:
            return MOLResult(
                success=False,
                error="MOL interpreter not found",
            )

        # Write MOL source to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mol", delete=False
        ) as f:
            f.write(mol_source)
            mol_file = f.name

        try:
            # Build command
            if self._mol_path.suffix == ".py":
                cmd = [sys.executable, str(self._mol_path), mol_file]
            else:
                cmd = [str(self._mol_path), mol_file]

            # Pass context as environment variables
            env = {}
            if context:
                for k, v in context.items():
                    env[f"DERAG_{k.upper()}"] = str(v)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env={**dict(__import__("os").environ), **env},
            )

            elapsed = (time.time() - start) * 1000

            if result.returncode == 0:
                # Try to parse JSON output
                output = result.stdout.strip()
                try:
                    return_value = json.loads(output) if output else None
                except json.JSONDecodeError:
                    return_value = output

                return MOLResult(
                    success=True,
                    output=output,
                    return_value=return_value,
                    execution_time_ms=elapsed,
                )
            else:
                return MOLResult(
                    success=False,
                    output=result.stdout.strip(),
                    error=result.stderr.strip(),
                    execution_time_ms=elapsed,
                )

        except subprocess.TimeoutExpired:
            return MOLResult(
                success=False,
                error="MOL script timed out (30s limit)",
                execution_time_ms=30000,
            )
        finally:
            Path(mol_file).unlink(missing_ok=True)

    def execute_file(self, mol_file: str | Path, context: Optional[dict] = None) -> MOLResult:
        """Execute a .mol file."""
        path = Path(mol_file)
        if not path.exists():
            return MOLResult(success=False, error=f"File not found: {mol_file}")
        return self.execute(path.read_text(), context=context)

    # --- Pipeline Management ---

    def register_pipeline(self, name: str, mol_source: str) -> MOLPipeline:
        """Register a named MOL pipeline."""
        pipeline = MOLPipeline(name=name, mol_source=mol_source)
        self._pipelines[name] = pipeline
        return pipeline

    def run_pipeline(self, name: str, **kwargs) -> MOLResult:
        """Execute a registered pipeline."""
        if name not in self._pipelines:
            return MOLResult(success=False, error=f"Pipeline '{name}' not found")
        return self.execute(self._pipelines[name].mol_source, context=kwargs)

    def list_pipelines(self) -> list[str]:
        return list(self._pipelines.keys())

    # --- MOL-callable De-RAG functions ---

    def _mol_encrypt(self, data: str) -> str:
        """Encrypt data via De-RAG envelope encryption."""
        if not self._node or not self._node._engine:
            raise RuntimeError("No De-RAG node available")
        blob = self._node._engine.encrypt(data.encode("utf-8"))
        return blob.serialize().hex()

    def _mol_decrypt(self, hex_data: str) -> str:
        """Decrypt data via De-RAG envelope encryption."""
        if not self._node or not self._node._engine:
            raise RuntimeError("No De-RAG node available")
        from derag.crypto.envelope import EncryptedBlob
        blob = EncryptedBlob.deserialize(bytes.fromhex(hex_data))
        return self._node._engine.decrypt(blob).decode("utf-8")

    def _mol_hash(self, data: str) -> str:
        """Hash data using BLAKE3."""
        import blake3
        return blake3.blake3(data.encode("utf-8")).hexdigest()

    def _mol_ingest(self, filepath: str) -> str:
        """Ingest a document (async wrapper)."""
        if not self._node:
            raise RuntimeError("No De-RAG node available")
        doc = asyncio.get_event_loop().run_until_complete(
            self._node.ingest(filepath)
        )
        return doc.doc_id

    def _mol_query(self, question: str) -> str:
        """Query the knowledge base (async wrapper)."""
        if not self._node:
            raise RuntimeError("No De-RAG node available")
        answer = asyncio.get_event_loop().run_until_complete(
            self._node.query(question)
        )
        return answer.text

    def _mol_shard(self, data: str, n: int = 5, k: int = 3) -> list:
        """Split data into Shamir shards."""
        if not self._node:
            raise RuntimeError("No De-RAG node available")
        shards = self._node._shard_manager.split(data.encode("utf-8"), f"mol-{hash(data)}")
        return [s.shard_id for s in shards]

    def _mol_status(self) -> dict:
        """Get node status."""
        if not self._node:
            return {"status": "no_node"}
        return self._node.stats

    # --- Template pipelines ---

    @staticmethod
    def encryption_pipeline_template() -> str:
        """
        Example MOL pipeline for document encryption.
        
        This is what users write in .mol files to define
        De-RAG operations in a human-readable way.
        """
        return '''
// De-RAG Encryption Pipeline (MOL)
// Encrypts a document, shards it, and distributes to peers

let pipeline = fn(filepath, shard_count, threshold) {
    // Step 1: Read and encrypt
    let content = io::read(filepath)
    let encrypted = derag_encrypt(content)
    
    // Step 2: Shard with Shamir's Secret Sharing
    let shards = derag_shard(encrypted, shard_count, threshold)
    
    // Step 3: Hash for verification
    let content_hash = derag_hash(content)
    
    // Return pipeline result
    {
        "filepath": filepath,
        "content_hash": content_hash,
        "shard_count": len(shards),
        "threshold": threshold,
        "shards": shards
    }
}

// Execute
pipeline(env("DERAG_FILEPATH"), 5, 3)
'''

    @staticmethod
    def query_pipeline_template() -> str:
        """Example MOL pipeline for querying."""
        return '''
// De-RAG Query Pipeline (MOL)
// Queries the knowledge base with privacy guarantees

let query_pipeline = fn(question) {
    let result = derag_query(question)
    let status = derag_status()
    
    {
        "answer": result,
        "node_status": status,
        "question_hash": derag_hash(question)
    }
}

query_pipeline(env("DERAG_QUESTION"))
'''
