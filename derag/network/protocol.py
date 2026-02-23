"""
De-RAG Wire Protocol
=====================

Defines the message types and serialization for peer communication.

All messages are:
  1. Serialized with MessagePack (compact binary)
  2. Signed with sender's Ed25519 key (authentication)
  3. Encrypted with recipient's X25519 key (confidentiality)

Message flow:
  Sender → serialize(msg) → sign(bytes) → encrypt(signed_bytes) → transmit
  Receiver → decrypt(bytes) → verify(signed_bytes) → deserialize(msg)

Protocol Version: 1.0.0

Copyright (c) 2026 CruxLabx
"""

from __future__ import annotations

import enum
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import msgpack


PROTOCOL_VERSION = "1.0.0"
PROTOCOL_MAGIC = b"DRGP"  # De-RAG Protocol


class MessageType(enum.IntEnum):
    """Wire protocol message types."""
    # Handshake
    HELLO = 0x01
    HELLO_ACK = 0x02
    
    # Shard Operations
    STORE_SHARD = 0x10
    STORE_SHARD_ACK = 0x11
    RETRIEVE_SHARD = 0x12
    RETRIEVE_SHARD_RESPONSE = 0x13
    DELETE_SHARD = 0x14
    DELETE_SHARD_ACK = 0x15
    
    # Query Operations
    QUERY = 0x20
    QUERY_RESPONSE = 0x21
    QUERY_CANCEL = 0x22
    
    # Cluster Management
    HEARTBEAT = 0x30
    PEER_LIST = 0x31
    SHARD_MAP_SYNC = 0x32
    
    # Key Exchange
    KEY_EXCHANGE_INIT = 0x40
    KEY_EXCHANGE_RESPONSE = 0x41
    
    # Error
    ERROR = 0xFF


@dataclass
class Message:
    """Base protocol message."""
    msg_type: MessageType
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    sender_id: str = ""
    timestamp: float = field(default_factory=time.time)
    payload: dict = field(default_factory=dict)
    
    def serialize(self) -> bytes:
        """Serialize to MessagePack binary."""
        data = {
            "t": self.msg_type.value,
            "id": self.msg_id,
            "s": self.sender_id,
            "ts": self.timestamp,
            "p": self.payload,
        }
        body = msgpack.packb(data, use_bin_type=True)
        # Header: magic(4) + version_major(1) + version_minor(1) + body_length(4)
        header = struct.pack("!4sBBI", PROTOCOL_MAGIC, 1, 0, len(body))
        return header + body
    
    @classmethod
    def deserialize(cls, raw: bytes) -> "Message":
        """Deserialize from MessagePack binary."""
        if raw[:4] != PROTOCOL_MAGIC:
            raise ValueError("Invalid protocol magic bytes")
        
        major, minor, body_len = struct.unpack("!BBI", raw[4:10])
        if major != 1:
            raise ValueError(f"Unsupported protocol version: {major}.{minor}")
        
        body = raw[10:10 + body_len]
        data = msgpack.unpackb(body, raw=False)
        
        return cls(
            msg_type=MessageType(data["t"]),
            msg_id=data["id"],
            sender_id=data["s"],
            timestamp=data["ts"],
            payload=data["p"],
        )


# ---------------------------------------------------------------------------
# Typed Message Constructors
# ---------------------------------------------------------------------------

def make_hello(node_id: str, capabilities: dict) -> Message:
    """Create a HELLO handshake message."""
    return Message(
        msg_type=MessageType.HELLO,
        sender_id=node_id,
        payload={
            "version": PROTOCOL_VERSION,
            "capabilities": capabilities,
            "public_key": "",  # Set by caller
        },
    )


def make_store_shard(
    node_id: str,
    shard_id: str,
    document_id: str,
    shard_data: bytes,
    shard_index: int,
    total_shards: int,
    threshold: int,
    data_hash: str,
) -> Message:
    """Create a STORE_SHARD request."""
    return Message(
        msg_type=MessageType.STORE_SHARD,
        sender_id=node_id,
        payload={
            "shard_id": shard_id,
            "document_id": document_id,
            "shard_data": shard_data,
            "shard_index": shard_index,
            "total_shards": total_shards,
            "threshold": threshold,
            "data_hash": data_hash,
        },
    )


def make_query(
    node_id: str,
    encrypted_query_vector: bytes,
    k: int = 10,
    query_id: Optional[str] = None,
) -> Message:
    """Create a QUERY request with encrypted query vector."""
    return Message(
        msg_type=MessageType.QUERY,
        sender_id=node_id,
        payload={
            "query_id": query_id or uuid.uuid4().hex[:16],
            "encrypted_vector": encrypted_query_vector,
            "k": k,
            "timestamp": time.time(),
        },
    )


def make_heartbeat(
    node_id: str,
    shard_count: int,
    available_space_mb: int,
    peer_count: int,
    uptime_sec: float,
) -> Message:
    """Create a HEARTBEAT message."""
    return Message(
        msg_type=MessageType.HEARTBEAT,
        sender_id=node_id,
        payload={
            "shard_count": shard_count,
            "available_space_mb": available_space_mb,
            "peer_count": peer_count,
            "uptime_sec": uptime_sec,
        },
    )


def make_error(node_id: str, error_code: int, error_msg: str, ref_msg_id: str = "") -> Message:
    """Create an ERROR message."""
    return Message(
        msg_type=MessageType.ERROR,
        sender_id=node_id,
        payload={
            "error_code": error_code,
            "error_msg": error_msg,
            "ref_msg_id": ref_msg_id,
        },
    )
