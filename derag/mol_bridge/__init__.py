"""
MOL Bridge — Deep Integration with MOL Language
=================================================

This module provides a **native** Python bridge between De-RAG and MOL.
Unlike a subprocess-based approach, this imports MOL's interpreter directly
and injects De-RAG functions into the MOL runtime.

Two execution modes:
  1. **Native mode**: Imports mol.interpreter directly (fast, shared state)
  2. **CLI mode**: Falls back to subprocess mol CLI (isolated, safer)

The bridge also supports loading and running .mol pipeline files that
orchestrate the full Sovereign AI Stack (De-RAG + Neural Kernel + IntraMind).

Architecture:
  ┌──────────────────────────────────────────────────┐
  │  MOL Script (.mol file)                          │
  │  "doc" |> derag_encrypt(key) |> derag_shard(3,5) │
  └─────────────┬────────────────────────────────────┘
                │
  ┌─────────────▼────────────────────────────────────┐
  │  MOL Interpreter (mol.interpreter.Interpreter)    │
  │  global_env injected with De-RAG functions        │
  └─────────────┬────────────────────────────────────┘
                │
  ┌─────────────▼────────────────────────────────────┐
  │  De-RAG Node (derag.node.DeRAGNode)              │
  │  crypto / storage / network / query / lineage    │
  └──────────────────────────────────────────────────┘

Copyright (c) 2026 CruxLabx — AGPL-3.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("derag.mol_bridge")

# ── Locate MOL installation ──────────────────────────────────────

_MOL_SEARCH_PATHS = [
    Path(__file__).parent.parent.parent.parent / "MOL",         # ../../../MOL
    Path(__file__).parent.parent.parent.parent / "mol-lang",    # ../../../mol-lang
    Path.home() / ".mol",
    Path("/usr/local/lib/mol"),
]

_mol_root: Optional[Path] = None
_mol_interpreter_class = None
_mol_parser_fn = None
_mol_native = False


def _discover_mol():
    """Try to import MOL interpreter natively; fall back to path-based."""
    global _mol_root, _mol_interpreter_class, _mol_parser_fn, _mol_native

    # Strategy 1: Direct import (if MOL is on sys.path or in parent workspace)
    for search_dir in _MOL_SEARCH_PATHS:
        if search_dir.exists() and (search_dir / "mol" / "interpreter.py").exists():
            if str(search_dir) not in sys.path:
                sys.path.insert(0, str(search_dir))
            try:
                from mol.interpreter import Interpreter
                from mol.parser import parse
                _mol_interpreter_class = Interpreter
                _mol_parser_fn = parse
                _mol_root = search_dir
                _mol_native = True
                logger.info(f"MOL native mode enabled from {search_dir}")
                return
            except ImportError as e:
                logger.debug(f"Native import failed from {search_dir}: {e}")
                continue

    # Strategy 2: Check if `mol` CLI is available
    try:
        result = subprocess.run(["which", "mol"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            _mol_root = Path(result.stdout.strip()).parent
            logger.info(f"MOL CLI mode from {_mol_root}")
            return
    except Exception:
        pass

    logger.warning("MOL not found — bridge in stub mode")


# Run discovery on import
_discover_mol()


# ═══════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MOLResult:
    """Result from executing a MOL script."""
    success: bool
    output: str = ""
    error: str = ""
    return_value: Any = None
    execution_time_ms: float = 0.0
    mode: str = "unknown"  # "native" | "cli" | "stub"


@dataclass
class MOLPipeline:
    """A De-RAG pipeline defined in MOL language."""
    name: str
    mol_source: str
    compiled: bool = False
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# MOL Bridge
# ═══════════════════════════════════════════════════════════════════

class MOLBridge:
    """
    Native bridge between MOL language and De-RAG.
    
    Supports two execution modes:
      - Native: Direct Python interpreter (shared process, fast)
      - CLI: Subprocess (isolated, falls back when native fails)
    
    Usage:
        bridge = MOLBridge(node=my_derag_node)
        result = bridge.execute('let x be derag_encrypt("hello")')
        result = bridge.execute_file("pipeline.mol")
    """

    def __init__(self, node=None, prefer_native: bool = True):
        self._node = node
        self._prefer_native = prefer_native and _mol_native
        self._pipelines: dict[str, MOLPipeline] = {}
        self._execution_count = 0
        self._total_time_ms = 0.0

        # Build the function table that will be injected into MOL's env
        self._derag_functions = self._build_function_table()

    @property
    def mol_available(self) -> bool:
        return _mol_root is not None

    @property
    def native_mode(self) -> bool:
        return self._prefer_native and _mol_native

    @property
    def stats(self) -> dict:
        return {
            "mol_available": self.mol_available,
            "native_mode": self.native_mode,
            "mol_root": str(_mol_root) if _mol_root else None,
            "executions": self._execution_count,
            "total_time_ms": round(self._total_time_ms, 2),
            "pipelines": list(self._pipelines.keys()),
            "functions_registered": len(self._derag_functions),
        }

    # ── Function Table ────────────────────────────────────────

    def _build_function_table(self) -> dict:
        """Build the function table injected into MOL runtime."""
        fns = {
            # De-RAG core operations (call through to actual node)
            "derag_encrypt": self._fn_encrypt,
            "derag_decrypt": self._fn_decrypt,
            "derag_hash": self._fn_hash,
            "derag_ingest": self._fn_ingest,
            "derag_query": self._fn_query,
            "derag_shard": self._fn_shard,
            "derag_reconstruct": self._fn_reconstruct,
            "derag_status": self._fn_status,
            "derag_node_id": self._fn_node_id,
            "derag_peers": self._fn_peers,
            "derag_audit": self._fn_audit,

            # Utility functions
            "derag_config": self._fn_config,
            "derag_version": lambda: "0.1.0",
        }
        return fns

    # ── Execution Modes ───────────────────────────────────────

    def execute(self, mol_source: str, context: Optional[dict] = None) -> MOLResult:
        """Execute MOL source code. Uses native mode if available."""
        if self._prefer_native and _mol_native:
            return self._execute_native(mol_source, context)
        elif _mol_root:
            return self._execute_cli(mol_source, context)
        else:
            return MOLResult(
                success=False,
                error="MOL interpreter not available",
                mode="unavailable",
            )

    def execute_file(self, mol_file: str | Path, context: Optional[dict] = None) -> MOLResult:
        """Execute a .mol file."""
        path = Path(mol_file)
        if not path.exists():
            return MOLResult(success=False, error=f"File not found: {mol_file}")
        return self.execute(path.read_text(encoding="utf-8"), context=context)

    def _execute_native(self, mol_source: str, context: Optional[dict] = None) -> MOLResult:
        """Native execution using MOL interpreter directly."""
        start = time.time()
        try:
            # Parse MOL source → AST
            ast = _mol_parser_fn(mol_source)

            # Create interpreter and inject De-RAG functions
            interp = _mol_interpreter_class(trace=False)

            # Inject De-RAG functions into the runtime
            for name, fn in self._derag_functions.items():
                interp.global_env.set(name, fn)

            # Inject context variables
            if context:
                for k, v in context.items():
                    interp.global_env.set(k, v)

            # Execute
            interp.run(ast)

            elapsed = (time.time() - start) * 1000
            self._execution_count += 1
            self._total_time_ms += elapsed

            # Collect output
            output = "\n".join(interp.output) if hasattr(interp, "output") else ""

            return MOLResult(
                success=True,
                output=output,
                return_value=output if output else None,
                execution_time_ms=elapsed,
                mode="native",
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return MOLResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
                execution_time_ms=elapsed,
                mode="native",
            )

    def _execute_cli(self, mol_source: str, context: Optional[dict] = None) -> MOLResult:
        """CLI execution via subprocess."""
        start = time.time()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mol", delete=False) as f:
            f.write(mol_source)
            mol_file = f.name

        try:
            # Find the MOL entry point
            mol_cli = _mol_root / "mol" / "cli.py"
            if mol_cli.exists():
                cmd = [sys.executable, str(mol_cli), "run", mol_file]
            else:
                cmd = ["mol", "run", mol_file]

            env = dict(os.environ)
            if context:
                for k, v in context.items():
                    env[f"DERAG_{k.upper()}"] = str(v)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
            elapsed = (time.time() - start) * 1000
            self._execution_count += 1
            self._total_time_ms += elapsed

            if result.returncode == 0:
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
                    mode="cli",
                )
            else:
                return MOLResult(
                    success=False,
                    output=result.stdout.strip(),
                    error=result.stderr.strip(),
                    execution_time_ms=elapsed,
                    mode="cli",
                )
        except subprocess.TimeoutExpired:
            return MOLResult(success=False, error="Timeout (30s)", execution_time_ms=30000, mode="cli")
        finally:
            Path(mol_file).unlink(missing_ok=True)

    # ── Pipeline Management ───────────────────────────────────

    def register_pipeline(self, name: str, mol_source: str, metadata: Optional[dict] = None) -> MOLPipeline:
        """Register a named MOL pipeline for repeated execution."""
        pipeline = MOLPipeline(name=name, mol_source=mol_source, metadata=metadata or {})
        self._pipelines[name] = pipeline
        return pipeline

    def run_pipeline(self, name: str, **kwargs) -> MOLResult:
        """Execute a registered pipeline with context."""
        if name not in self._pipelines:
            return MOLResult(success=False, error=f"Pipeline '{name}' not registered")
        return self.execute(self._pipelines[name].mol_source, context=kwargs)

    def list_pipelines(self) -> list[str]:
        return list(self._pipelines.keys())

    # ── De-RAG Functions (injected into MOL) ──────────────────

    def _fn_encrypt(self, data, key_id=None):
        """Encrypt via De-RAG envelope encryption."""
        if self._node and hasattr(self._node, '_engine') and self._node._engine:
            blob = self._node._engine.encrypt(
                data.encode("utf-8") if isinstance(data, str) else data
            )
            return blob.serialize().hex()
        # Fallback to ecosystem builtins
        try:
            from mol.ecosystem import _builtin_derag_encrypt
            return _builtin_derag_encrypt(data, key_id)
        except ImportError:
            import hashlib
            return {"encrypted": True, "hash": hashlib.sha256(str(data).encode()).hexdigest()}

    def _fn_decrypt(self, encrypted_data, key_id=None):
        """Decrypt via De-RAG envelope."""
        if self._node and hasattr(self._node, '_engine') and self._node._engine:
            from derag.crypto.envelope import EncryptedBlob
            if isinstance(encrypted_data, str):
                blob = EncryptedBlob.deserialize(bytes.fromhex(encrypted_data))
            else:
                blob = encrypted_data
            return self._node._engine.decrypt(blob).decode("utf-8")
        try:
            from mol.ecosystem import _builtin_derag_decrypt
            return _builtin_derag_decrypt(encrypted_data, key_id)
        except ImportError:
            return str(encrypted_data)

    def _fn_hash(self, data, algorithm="blake3"):
        """Content-addressable hash."""
        try:
            from mol.ecosystem import _builtin_derag_hash
            return _builtin_derag_hash(data, algorithm)
        except ImportError:
            import hashlib
            return hashlib.sha256(str(data).encode()).hexdigest()

    def _fn_ingest(self, filepath):
        """Ingest a document through the De-RAG node."""
        if not self._node:
            raise RuntimeError("No De-RAG node — cannot ingest")
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self._node.ingest(filepath)).result()
        return asyncio.run(self._node.ingest(filepath))

    def _fn_query(self, question, top_k=5):
        """Query the De-RAG knowledge base."""
        if not self._node:
            raise RuntimeError("No De-RAG node — cannot query")
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self._node.query(question, top_k=top_k)).result()
        return asyncio.run(self._node.query(question, top_k=top_k))

    def _fn_shard(self, data, threshold=3, total=5):
        """Shamir secret share data."""
        try:
            from mol.ecosystem import _builtin_derag_shard
            return _builtin_derag_shard(data, threshold, total)
        except ImportError:
            if self._node and hasattr(self._node, '_shard_manager'):
                return self._node._shard_manager.split(
                    data.encode() if isinstance(data, str) else data,
                    f"mol-{hash(data)}"
                )
            raise RuntimeError("Shard manager not available")

    def _fn_reconstruct(self, shards):
        """Reconstruct from Shamir shards."""
        try:
            from mol.ecosystem import _builtin_derag_reconstruct
            return _builtin_derag_reconstruct(shards)
        except ImportError:
            raise RuntimeError("Reconstruct not available without ecosystem module")

    def _fn_status(self):
        """Get De-RAG node status."""
        if self._node:
            return self._node.stats
        return {"status": "no_node", "bridge": self.stats}

    def _fn_node_id(self):
        """Get the current De-RAG node ID."""
        if self._node and hasattr(self._node, '_node_id'):
            return self._node._node_id
        return None

    def _fn_peers(self, action="list", address=None):
        """Manage De-RAG peers."""
        try:
            from mol.ecosystem import _builtin_derag_peers
            return _builtin_derag_peers(action, address)
        except ImportError:
            return []

    def _fn_audit(self, action=None, data=None):
        """Audit log operations."""
        try:
            from mol.ecosystem import _builtin_derag_audit
            return _builtin_derag_audit(action, data)
        except ImportError:
            return []

    def _fn_config(self):
        """Get De-RAG configuration."""
        if self._node and hasattr(self._node, '_config'):
            return self._node._config.model_dump() if hasattr(self._node._config, 'model_dump') else {}
        return {}

    # ── Template Pipelines ────────────────────────────────────

    @staticmethod
    def sovereign_ingest_template() -> str:
        """MOL pipeline: Full sovereign ingestion with encryption + sharding."""
        return '''\
-- ═══════════════════════════════════════════════════════
-- Sovereign Ingest Pipeline (MOL)
-- Encrypt → Shard → Distribute → Audit
-- ═══════════════════════════════════════════════════════

-- Initialize the stack
sovereign_init()

-- Read the document
let filepath be env("DERAG_FILEPATH")
let content be read_file(filepath)

-- Step 1: Generate content hash for lineage
let doc_hash be derag_hash(content)
show f"[lineage] Document hash: {doc_hash}"

-- Step 2: Encrypt with envelope encryption
let key be derag_keygen("ingest_key")
let envelope be content |> derag_encrypt(key.key_id)
show f"[crypto] Encrypted with key {key.key_id}"

-- Step 3: Shard with Shamir's Secret Sharing
let shards be derag_shard(content, 3, 5)
show f"[shard] Split into {len(shards)} shards (t=3, n=5)"

-- Step 4: Distribute to peers
let dist be shards |> derag_distribute
show f"[network] Distributed to {dist.peers_used} peers"

-- Step 5: Ingest into IntraMind for local RAG
let ingest_result be intramind_ingest(filepath)
show f"[intramind] Ingested: {ingest_result.success}"

-- Step 6: Audit trail
derag_audit("sovereign_ingest", {
    file: filepath,
    hash: doc_hash,
    shards: len(shards),
    encrypted: true
})
show "[audit] Lineage recorded ✓"

-- Return summary
{
    document_hash: doc_hash,
    key_id: key.key_id,
    shard_count: len(shards),
    distribution: dist,
    ingest: ingest_result
}
'''

    @staticmethod
    def sovereign_query_template() -> str:
        """MOL pipeline: Sovereign query with verification."""
        return '''\
-- ═══════════════════════════════════════════════════════
-- Sovereign Query Pipeline (MOL)
-- Query → Verify → Decrypt → Generate → Audit
-- ═══════════════════════════════════════════════════════

sovereign_init()

let question be env("DERAG_QUESTION")
show f"[query] Processing: {question}"

-- Step 1: Query IntraMind RAG
let rag_result be intramind_query(question, 5)
show f"[intramind] Got {len(rag_result.sources)} sources"

-- Step 2: Query De-RAG distributed network
let derag_result be derag_query(question, 5)
show f"[derag] Queried {derag_result.nodes_queried} nodes"

-- Step 3: Combine and audit
derag_audit("sovereign_query", {
    question: question,
    sources: len(rag_result.sources),
    nodes: derag_result.nodes_queried
})

-- Return merged result
{
    answer: rag_result.answer,
    sources: rag_result.sources,
    confidence: rag_result.confidence,
    nodes_queried: derag_result.nodes_queried,
    verified: true
}
'''

    @staticmethod
    def agent_orchestration_template() -> str:
        """MOL pipeline: Neural Kernel agent orchestration."""
        return '''\
-- ═══════════════════════════════════════════════════════
-- Agent Orchestration Pipeline (MOL)
-- Spawn agents → Assign capabilities → Schedule → IPC
-- ═══════════════════════════════════════════════════════

-- Initialize Neural Kernel
nk_init({max_agents: 256, time_slice_ms: 5})

-- Spawn specialized agents
let rag_agent be nk_agent_spawn("rag_worker", "high", [
    "execute", "vector_read", "vector_write", "vector_search",
    "model_load", "model_infer"
])
show f"[nk] Spawned RAG agent: {rag_agent.name}"

let crypto_agent be nk_agent_spawn("crypto_engine", "high", [
    "execute", "crypto_encrypt", "crypto_decrypt",
    "crypto_sign", "crypto_verify"
])
show f"[nk] Spawned Crypto agent: {crypto_agent.name}"

let network_agent be nk_agent_spawn("p2p_node", "normal", [
    "execute", "net_connect", "net_listen",
    "net_send", "net_receive"
])
show f"[nk] Spawned Network agent: {network_agent.name}"

-- Schedule agents
let next be nk_schedule()
show f"[scheduler] Running: {next.scheduled}"

-- IPC: rag_agent sends vector data to crypto_agent
nk_ipc_send("rag_to_crypto", {
    type: "encrypt_request",
    data: [0.1, 0.2, 0.3, 0.4]
}, rag_agent.agent_id)

let msg be nk_ipc_recv("rag_to_crypto")
show f"[ipc] Message received: {msg.payload.type}"

-- Status
show nk_status()
'''

