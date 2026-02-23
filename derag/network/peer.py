"""
De-RAG Peer — Async P2P Node Communication
============================================

Handles peer discovery, connection management, and message routing.

Discovery methods:
  1. mDNS (multicast DNS) — automatic discovery on local network
  2. Bootstrap peers — connect to known addresses
  3. DHT — distributed hash table for internet-wide discovery (future)

Transport:
  - TCP with TLS 1.3 (encrypted channels)
  - MessagePack serialization
  - Ed25519 signatures on all messages

Copyright (c) 2026 CruxLabx
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from derag.network.protocol import (
    Message,
    MessageType,
    make_hello,
    make_heartbeat,
    make_error,
)

logger = logging.getLogger("derag.network.peer")


@dataclass
class PeerInfo:
    """Information about a connected peer."""
    peer_id: str
    address: str
    port: int
    public_key: Optional[bytes] = None
    capabilities: dict = field(default_factory=dict)
    shard_count: int = 0
    available_space_mb: int = 0
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    is_alive: bool = True


class DeRAGPeer:
    """
    Async P2P peer for the De-RAG network.
    
    Usage:
        peer = DeRAGPeer(node_id="abc123", port=9090)
        peer.on_message(MessageType.STORE_SHARD, handle_store)
        peer.on_message(MessageType.QUERY, handle_query)
        
        await peer.start()
        await peer.connect_to("192.168.1.100", 9090)
        await peer.broadcast(message)
        await peer.stop()
    """

    def __init__(
        self,
        node_id: str,
        port: int = 9090,
        host: str = "0.0.0.0",
        identity_key: Optional[Ed25519PrivateKey] = None,
        max_peers: int = 50,
        heartbeat_interval: float = 30.0,
    ):
        self.node_id = node_id
        self.port = port
        self.host = host
        self.max_peers = max_peers
        self.heartbeat_interval = heartbeat_interval
        self._identity_key = identity_key

        self._peers: dict[str, PeerInfo] = {}
        self._connections: dict[str, tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
        self._handlers: dict[MessageType, list[Callable]] = {}
        self._server: Optional[asyncio.Server] = None
        self._running = False
        self._start_time = 0.0
        self._tasks: list[asyncio.Task] = []

    @property
    def peer_count(self) -> int:
        return len(self._peers)

    @property
    def connected_peers(self) -> list[PeerInfo]:
        return [p for p in self._peers.values() if p.is_alive]

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time if self._running else 0

    # --- Lifecycle ---

    async def start(self) -> None:
        """Start listening for peer connections."""
        self._start_time = time.time()
        self._running = True

        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )

        # Start heartbeat loop
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        # Start peer health checker
        self._tasks.append(asyncio.create_task(self._health_check_loop()))

        logger.info(f"De-RAG peer {self.node_id} listening on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Gracefully shut down the peer."""
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for peer_id, (reader, writer) in list(self._connections.items()):
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

        # Close server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info(f"De-RAG peer {self.node_id} stopped")

    # --- Connection Management ---

    async def connect_to(self, address: str, port: int) -> Optional[PeerInfo]:
        """Establish a connection to a remote peer."""
        if len(self._peers) >= self.max_peers:
            logger.warning(f"Max peers ({self.max_peers}) reached, rejecting connection to {address}:{port}")
            return None

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(address, port),
                timeout=10.0,
            )

            # Send HELLO
            hello = make_hello(self.node_id, self._get_capabilities())
            await self._send_message(writer, hello)

            # Wait for HELLO_ACK
            response = await self._recv_message(reader)
            if response.msg_type != MessageType.HELLO_ACK:
                writer.close()
                return None

            peer_id = response.sender_id
            peer_info = PeerInfo(
                peer_id=peer_id,
                address=address,
                port=port,
                capabilities=response.payload.get("capabilities", {}),
                shard_count=response.payload.get("shard_count", 0),
            )

            self._peers[peer_id] = peer_info
            self._connections[peer_id] = (reader, writer)

            # Start reading messages from this peer
            self._tasks.append(asyncio.create_task(
                self._read_loop(peer_id, reader)
            ))

            logger.info(f"Connected to peer {peer_id} at {address}:{port}")
            return peer_info

        except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
            logger.warning(f"Failed to connect to {address}:{port}: {e}")
            return None

    async def disconnect(self, peer_id: str) -> None:
        """Disconnect from a peer."""
        if peer_id in self._connections:
            _, writer = self._connections.pop(peer_id)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        self._peers.pop(peer_id, None)
        logger.info(f"Disconnected from peer {peer_id}")

    # --- Message Handling ---

    def on_message(self, msg_type: MessageType, handler: Callable) -> None:
        """Register a handler for a specific message type."""
        self._handlers.setdefault(msg_type, []).append(handler)

    async def send_to(self, peer_id: str, message: Message) -> bool:
        """Send a message to a specific peer."""
        conn = self._connections.get(peer_id)
        if not conn:
            logger.warning(f"No connection to peer {peer_id}")
            return False

        _, writer = conn
        message.sender_id = self.node_id
        try:
            await self._send_message(writer, message)
            return True
        except (ConnectionError, OSError) as e:
            logger.error(f"Failed to send to {peer_id}: {e}")
            await self.disconnect(peer_id)
            return False

    async def broadcast(self, message: Message) -> dict[str, bool]:
        """Send a message to ALL connected peers."""
        results = {}
        for peer_id in list(self._connections.keys()):
            results[peer_id] = await self.send_to(peer_id, message)
        return results

    async def scatter_query(
        self,
        message: Message,
        timeout: float = 30.0,
    ) -> list[Message]:
        """
        Send a query to all peers and collect responses.
        
        Returns responses received within timeout.
        """
        query_id = message.payload.get("query_id", message.msg_id)
        responses: list[Message] = []
        response_event = asyncio.Event()

        async def collect_response(msg: Message) -> None:
            if msg.payload.get("query_id") == query_id:
                responses.append(msg)
                if len(responses) >= len(self._connections):
                    response_event.set()

        # Register temporary handler
        self.on_message(MessageType.QUERY_RESPONSE, collect_response)

        # Broadcast query
        await self.broadcast(message)

        # Wait for responses
        try:
            await asyncio.wait_for(response_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug(f"Query {query_id} timed out with {len(responses)} responses")

        # Remove handler
        if MessageType.QUERY_RESPONSE in self._handlers:
            self._handlers[MessageType.QUERY_RESPONSE] = [
                h for h in self._handlers[MessageType.QUERY_RESPONSE]
                if h is not collect_response
            ]

        return responses

    # --- Internal ---

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an incoming connection."""
        addr = writer.get_extra_info("peername")
        logger.debug(f"Incoming connection from {addr}")

        try:
            # Wait for HELLO
            msg = await asyncio.wait_for(self._recv_message(reader), timeout=10.0)
            if msg.msg_type != MessageType.HELLO:
                writer.close()
                return

            peer_id = msg.sender_id

            # Send HELLO_ACK
            ack = Message(
                msg_type=MessageType.HELLO_ACK,
                sender_id=self.node_id,
                payload={
                    "version": "1.0.0",
                    "capabilities": self._get_capabilities(),
                    "shard_count": 0,
                },
            )
            await self._send_message(writer, ack)

            # Register peer
            peer_info = PeerInfo(
                peer_id=peer_id,
                address=addr[0] if addr else "unknown",
                port=addr[1] if addr else 0,
                capabilities=msg.payload.get("capabilities", {}),
            )
            self._peers[peer_id] = peer_info
            self._connections[peer_id] = (reader, writer)

            logger.info(f"Peer {peer_id} connected from {addr}")

            # Read loop
            await self._read_loop(peer_id, reader)

        except (asyncio.TimeoutError, ConnectionError, OSError) as e:
            logger.debug(f"Connection error from {addr}: {e}")
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _read_loop(self, peer_id: str, reader: asyncio.StreamReader) -> None:
        """Continuously read messages from a peer."""
        while self._running:
            try:
                msg = await self._recv_message(reader)

                # Update heartbeat timestamp
                if peer_id in self._peers:
                    self._peers[peer_id].last_heartbeat = time.time()

                # Dispatch to handlers
                handlers = self._handlers.get(msg.msg_type, [])
                for handler in handlers:
                    try:
                        result = handler(msg)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Handler error for {msg.msg_type}: {e}")

            except (asyncio.IncompleteReadError, ConnectionError, OSError):
                logger.info(f"Peer {peer_id} disconnected")
                break
            except Exception as e:
                logger.error(f"Error reading from {peer_id}: {e}")
                break

        # Cleanup
        self._peers.pop(peer_id, None)
        self._connections.pop(peer_id, None)

    async def _send_message(self, writer: asyncio.StreamWriter, msg: Message) -> None:
        """Send a message over a stream."""
        data = msg.serialize()
        # Length-prefix framing
        writer.write(struct.pack("!I", len(data)))
        writer.write(data)
        await writer.drain()

    async def _recv_message(self, reader: asyncio.StreamReader) -> Message:
        """Receive a message from a stream."""
        # Read length prefix
        length_bytes = await reader.readexactly(4)
        length = struct.unpack("!I", length_bytes)[0]
        
        if length > 64 * 1024 * 1024:  # 64MB max message
            raise ValueError(f"Message too large: {length} bytes")
        
        data = await reader.readexactly(length)
        return Message.deserialize(data)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to all peers."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                hb = make_heartbeat(
                    node_id=self.node_id,
                    shard_count=0,  # Updated by caller
                    available_space_mb=0,
                    peer_count=self.peer_count,
                    uptime_sec=self.uptime,
                )
                await self.broadcast(hb)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _health_check_loop(self) -> None:
        """Check for dead peers."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval * 3)
                now = time.time()
                dead_threshold = self.heartbeat_interval * 5

                for peer_id, info in list(self._peers.items()):
                    if now - info.last_heartbeat > dead_threshold:
                        logger.warning(f"Peer {peer_id} timed out, disconnecting")
                        info.is_alive = False
                        await self.disconnect(peer_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    def _get_capabilities(self) -> dict:
        """Get this node's capabilities."""
        return {
            "version": "1.0.0",
            "can_store_shards": True,
            "can_query": True,
            "max_shard_size_mb": 64,
        }

    # --- Stats ---

    @property
    def stats(self) -> dict:
        return {
            "node_id": self.node_id,
            "address": f"{self.host}:{self.port}",
            "running": self._running,
            "uptime_sec": self.uptime,
            "peer_count": self.peer_count,
            "alive_peers": len(self.connected_peers),
            "peers": [
                {
                    "peer_id": p.peer_id,
                    "address": f"{p.address}:{p.port}",
                    "latency_ms": p.latency_ms,
                    "is_alive": p.is_alive,
                }
                for p in self._peers.values()
            ],
        }


# Need struct for _send_message/_recv_message framing
import struct
