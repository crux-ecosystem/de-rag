"""
De-RAG CLI — Command-line interface
=====================================

Commands:
  derag init          Initialize a new De-RAG node
  derag ingest        Ingest documents into the local store
  derag query         Query the knowledge base
  derag network       P2P network management
  derag status        Show node status
  derag audit         View audit log
  derag keys          Key management
  derag shards        Shard operations
  derag serve         Start HTTP API server

Copyright (c) 2026 CruxLabx — AGPL-3.0
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click

from derag import __version__

def _run(coro):
    """Run async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _create_node(data_dir: str):
    from derag.config import DeRAGConfig
    from derag.node import DeRAGNode

    config = DeRAGConfig(data_dir=Path(data_dir))
    return DeRAGNode(config)


# ─── Root Group ───────────────────────────────────────────────

@click.group(
    name="derag",
    help="De-RAG — Decentralized Encrypted Retrieval-Augmented Generation",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--data-dir", "-d",
    default="~/.derag",
    envvar="DERAG_DATA_DIR",
    help="Node data directory",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, data_dir: str, verbose: bool):
    """De-RAG — Privacy-first decentralized RAG"""
    import logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
        stream=sys.stderr,
    )
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = os.path.expanduser(data_dir)
    ctx.obj["verbose"] = verbose


# ─── init ─────────────────────────────────────────────────────

@cli.command()
@click.option("--name", "-n", default=None, help="Node name")
@click.option("--shards", "-s", default=5, help="Total shard count (Shamir n)")
@click.option("--threshold", "-k", default=3, help="Reconstruction threshold (Shamir k)")
@click.option("--dim", default=768, help="Embedding dimension")
@click.pass_context
def init(ctx, name: str, shards: int, threshold: int, dim: int):
    """Initialize a new De-RAG node."""
    from derag.config import DeRAGConfig

    data_dir = Path(ctx.obj["data_dir"])
    if (data_dir / "keys").exists():
        click.echo(f"Node already initialized at {data_dir}", err=True)
        raise SystemExit(1)

    # Ask for password
    password = click.prompt("Enter node password", hide_input=True, confirmation_prompt=True)

    config = DeRAGConfig(
        node_name=name or f"derag-{os.getpid()}",
        data_dir=data_dir,
    )

    # Write config
    config_path = data_dir / "derag.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({
        "node_name": config.node_name,
        "sharding": {"total_shards": shards, "threshold": threshold},
        "storage": {"embedding_dim": dim},
    }, indent=2))

    # Initialize node (creates keys, DB, etc.)
    from derag.node import DeRAGNode
    node = DeRAGNode(config)
    _run(node.initialize(password))
    _run(node.shutdown())

    click.echo(f"\n✓ De-RAG node initialized at {data_dir}")
    click.echo(f"  Node ID: {node.node_id[:16]}...")
    click.echo(f"  Shards: {shards}-of-{threshold}")
    click.echo(f"  Embedding dim: {dim}")


# ─── ingest ───────────────────────────────────────────────────

@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--chunk-size", default=512, help="Words per chunk")
@click.option("--overlap", default=50, help="Word overlap between chunks")
@click.option("--tag", "-t", multiple=True, help="Add metadata tags")
@click.pass_context
def ingest(ctx, files: tuple, chunk_size: int, overlap: int, tag: tuple):
    """Ingest one or more documents."""
    data_dir = ctx.obj["data_dir"]
    password = click.prompt("Node password", hide_input=True)

    node = _create_node(data_dir)

    async def do_ingest():
        await node.initialize(password)
        try:
            for filepath in files:
                meta = {"tags": list(tag)} if tag else None
                doc = await node.ingest(
                    filepath,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    metadata=meta,
                )
                click.echo(f"✓ {filepath} → {doc.doc_id[:12]}... ({doc.chunk_count} chunks)")
        finally:
            await node.shutdown()

    _run(do_ingest())


# ─── query ────────────────────────────────────────────────────

@cli.command()
@click.argument("question")
@click.option("--distributed", "-D", is_flag=True, help="Fan out to peers")
@click.option("--top-k", "-k", default=5, help="Number of results")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.pass_context
def query(ctx, question: str, distributed: bool, top_k: int, json_output: bool):
    """Query the knowledge base."""
    data_dir = ctx.obj["data_dir"]
    password = click.prompt("Node password", hide_input=True)

    node = _create_node(data_dir)

    async def do_query():
        await node.initialize(password)
        try:
            answer = await node.query(question, distributed=distributed)
            if json_output:
                click.echo(json.dumps({
                    "answer": answer.text,
                    "confidence": answer.confidence,
                    "sources": len(answer.results),
                    "time_ms": answer.total_time_ms,
                }, indent=2))
            else:
                click.echo(f"\n{answer.text}")
                click.echo(f"\n─── {len(answer.results)} sources | "
                           f"conf: {answer.confidence:.2f} | "
                           f"{answer.total_time_ms:.0f}ms ───")
                for r in answer.results:
                    click.echo(f"  [{r.score:.3f}] {r.document_id[:12]}... chunk#{r.chunk_index}")
        finally:
            await node.shutdown()

    _run(do_query())


# ─── network ─────────────────────────────────────────────────

@cli.group()
def network():
    """P2P network management."""
    pass


@network.command("start")
@click.option("--port", "-p", default=9090, help="Listen port")
@click.option("--bootstrap", "-b", multiple=True, help="Bootstrap peer (host:port)")
@click.pass_context
def network_start(ctx, port: int, bootstrap: tuple):
    """Start the P2P daemon."""
    data_dir = ctx.obj["data_dir"]
    password = click.prompt("Node password", hide_input=True)

    node = _create_node(data_dir)

    async def run_daemon():
        await node.initialize(password)
        await node.join_network(port=port, bootstrap_peers=list(bootstrap) or None)

        click.echo(f"✓ P2P daemon running on port {port}")
        click.echo(f"  Node ID: {node.node_id[:16]}...")
        click.echo(f"  Peers: {node._peer.peer_count}")
        click.echo("  Press Ctrl+C to stop...")

        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await node.shutdown()

    _run(run_daemon())


@network.command("peers")
@click.pass_context
def network_peers(ctx):
    """List connected peers."""
    data_dir = ctx.obj["data_dir"]
    password = click.prompt("Node password", hide_input=True)

    node = _create_node(data_dir)

    async def show_peers():
        await node.initialize(password)
        try:
            if not node._peer:
                click.echo("Network not started")
                return
            for p in node._peer.connected_peers:
                click.echo(f"  {p.peer_id[:16]}... @ {p.address}:{p.port}")
        finally:
            await node.shutdown()

    _run(show_peers())


# ─── status ───────────────────────────────────────────────────

@cli.command()
@click.option("--json-output", "-j", is_flag=True, help="JSON output")
@click.pass_context
def status(ctx, json_output: bool):
    """Show node status."""
    data_dir = ctx.obj["data_dir"]
    password = click.prompt("Node password", hide_input=True)

    node = _create_node(data_dir)

    async def show_status():
        await node.initialize(password)
        try:
            s = node.stats
            if json_output:
                click.echo(json.dumps(s, indent=2, default=str))
            else:
                click.echo(f"\n De-RAG Node Status")
                click.echo(f"  Name:     {s['node_name']}")
                click.echo(f"  ID:       {s['node_id'][:20]}...")
                click.echo(f"  Status:   {s['status']}")
                click.echo(f"  Uptime:   {s['uptime_sec']:.0f}s")
                if s.get("store"):
                    click.echo(f"  Documents: {s['store'].get('documents', 0)}")
                    click.echo(f"  Chunks:    {s['store'].get('chunks', 0)}")
                if s.get("index"):
                    click.echo(f"  Vectors:   {s['index'].get('count', 0)}")
                if s.get("network"):
                    click.echo(f"  Peers:     {s['network'].get('peer_count', 0)}")
                if s.get("audit"):
                    click.echo(f"  Audit entries: {s['audit'].get('entry_count', 0)}")
        finally:
            await node.shutdown()

    _run(show_status())


# ─── audit ───────────────────────────────────────────────────

@cli.command()
@click.option("--action", "-a", default=None, help="Filter by action")
@click.option("--limit", "-n", default=20, help="Max entries")
@click.option("--verify", is_flag=True, help="Verify chain integrity")
@click.option("--json-output", "-j", is_flag=True, help="JSON output")
@click.pass_context
def audit(ctx, action: Optional[str], limit: int, verify: bool, json_output: bool):
    """View audit log."""
    data_dir = ctx.obj["data_dir"]
    password = click.prompt("Node password", hide_input=True)

    node = _create_node(data_dir)

    async def show_audit():
        await node.initialize(password)
        try:
            if verify:
                ok = node._audit.verify_chain()
                if ok:
                    click.echo("✓ Audit chain integrity verified")
                else:
                    click.echo("✗ AUDIT CHAIN INTEGRITY FAILURE", err=True)
                    raise SystemExit(1)
                return

            entries = node._audit.query(action=action)
            entries = entries[-limit:]

            if json_output:
                click.echo(json.dumps(
                    [{"id": e.entry_id, "ts": e.timestamp, "action": e.action,
                      "actor": e.actor, "target": e.target, "details": e.details}
                     for e in entries],
                    indent=2
                ))
            else:
                for e in entries:
                    click.echo(
                        f"  [{e.entry_id:>5}] {e.timestamp[:19]} "
                        f"{e.action:<20} {e.target[:16] if e.target else ''}"
                    )
        finally:
            await node.shutdown()

    _run(show_audit())


# ─── keys ─────────────────────────────────────────────────────

@cli.group()
def keys():
    """Key management."""
    pass


@keys.command("list")
@click.pass_context
def keys_list(ctx):
    """List all keys."""
    data_dir = ctx.obj["data_dir"]
    password = click.prompt("Node password", hide_input=True)

    node = _create_node(data_dir)

    async def list_keys():
        await node.initialize(password)
        try:
            records = node._key_manager.list_keys()
            for r in records:
                click.echo(
                    f"  {r['purpose']:<12} {r['key_id'][:16]}... "
                    f"created={r['created_at'][:10]} "
                    f"rotations={r.get('rotation_count', 0)}"
                )
        finally:
            await node.shutdown()

    _run(list_keys())


@keys.command("rotate")
@click.option("--purpose", "-p", required=True, help="Key purpose to rotate (kek/dek/shard/search)")
@click.pass_context
def keys_rotate(ctx, purpose: str):
    """Rotate a key."""
    data_dir = ctx.obj["data_dir"]
    password = click.prompt("Node password", hide_input=True)

    node = _create_node(data_dir)

    async def rotate_key():
        await node.initialize(password)
        try:
            node._key_manager.rotate_key(purpose)
            click.echo(f"✓ Rotated {purpose} key")
        finally:
            await node.shutdown()

    _run(rotate_key())


# ─── shards ───────────────────────────────────────────────────

@cli.group()
def shards():
    """Shard operations."""
    pass


@shards.command("distribute")
@click.argument("doc_id")
@click.pass_context
def shards_distribute(ctx, doc_id: str):
    """Distribute shards of a document to peers."""
    data_dir = ctx.obj["data_dir"]
    password = click.prompt("Node password", hide_input=True)

    node = _create_node(data_dir)

    async def distribute():
        await node.initialize(password)
        try:
            placements = await node.distribute_shards(doc_id)
            for p in placements:
                click.echo(
                    f"  shard[{p['shard_index']}] → {p['peer_id'][:12]}... "
                    f"({p['size_bytes']} bytes)"
                )
        finally:
            await node.shutdown()

    _run(distribute())


# ─── serve ────────────────────────────────────────────────────

@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", "-p", default=8420, help="HTTP port")
@click.option("--token", envvar="DERAG_API_TOKEN", default=None,
              help="Bearer auth token (or set DERAG_API_TOKEN)")
@click.option("--rate-limit", default=100, help="Max requests per minute")
@click.option("--workers", default=1, help="Uvicorn workers")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes")
@click.pass_context
def serve(ctx, host: str, port: int, token: str, rate_limit: int,
          workers: int, reload: bool):
    """Start De-RAG HTTP API server."""
    import os
    import uvicorn
    from derag.api.server import DeRAGAPI

    data_dir = ctx.obj["data_dir"]

    # Set env vars so the default app picks them up
    os.environ["DERAG_DATA_DIR"] = data_dir
    if token:
        os.environ["DERAG_API_TOKEN"] = token

    click.echo(f"De-RAG v{__version__} API starting at http://{host}:{port}")
    click.echo(f"  data_dir : {data_dir}")
    click.echo(f"  auth     : {'enabled' if token else 'disabled'}")
    click.echo(f"  rate_limit: {rate_limit} req/min")
    click.echo(f"  docs     : http://{host}:{port}/docs")

    uvicorn.run(
        "derag.api.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


# ─── Entry point ──────────────────────────────────────────────

def main():
    cli()


if __name__ == "__main__":
    main()
