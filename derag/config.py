"""
De-RAG Configuration — Pydantic-validated settings for every subsystem.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class EncryptionAlgorithm(str, Enum):
    AES_256_GCM = "aes-256-gcm"
    CHACHA20_POLY1305 = "chacha20-poly1305"


class ShardingStrategy(str, Enum):
    SHAMIR = "shamir"           # Shamir's Secret Sharing
    REED_SOLOMON = "reed-solomon"  # Erasure coding
    REPLICATE = "replicate"     # Simple replication


class IndexType(str, Enum):
    FLAT = "flat"               # Exact search (small datasets)
    HNSW = "hnsw"               # Approximate (large datasets)
    IVF = "ivf"                 # Inverted file index


class CryptoConfig(BaseModel):
    """Encryption subsystem configuration."""
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    key_derivation: str = "argon2id"
    argon2_time_cost: int = Field(default=3, ge=1, le=32)
    argon2_memory_cost: int = Field(default=65536, ge=8192)  # KB
    argon2_parallelism: int = Field(default=4, ge=1, le=16)
    key_rotation_interval_hours: int = Field(default=720, ge=24)  # 30 days
    envelope_encryption: bool = True
    secure_memory: bool = True  # mlock key material


class StorageConfig(BaseModel):
    """Storage engine configuration."""
    data_dir: Path = Path("~/.derag/data").expanduser()
    index_type: IndexType = IndexType.HNSW
    embedding_dim: int = Field(default=384, ge=32, le=4096)
    hnsw_m: int = Field(default=32, ge=8, le=128)
    hnsw_ef_construction: int = Field(default=200, ge=64, le=800)
    hnsw_ef_search: int = Field(default=128, ge=16, le=512)
    max_vectors: int = Field(default=1_000_000, ge=1000)
    encrypt_at_rest: bool = True
    compress_shards: bool = True


class ShardConfig(BaseModel):
    """Shard manager configuration."""
    strategy: ShardingStrategy = ShardingStrategy.SHAMIR
    total_shards: int = Field(default=5, ge=2, le=255)
    threshold: int = Field(default=3, ge=2, le=255)
    shard_size_limit_mb: int = Field(default=64, ge=1, le=1024)
    redundancy_factor: float = Field(default=1.5, ge=1.0, le=5.0)

    @field_validator("threshold")
    @classmethod
    def threshold_lte_total(cls, v: int, info) -> int:
        total = info.data.get("total_shards", 5)
        if v > total:
            raise ValueError(f"threshold ({v}) must be <= total_shards ({total})")
        return v


class NetworkConfig(BaseModel):
    """P2P network configuration."""
    listen_addr: str = "/ip4/0.0.0.0/tcp/9090"
    bootstrap_peers: list[str] = Field(default_factory=list)
    enable_mdns: bool = True          # Local network discovery
    enable_dht: bool = True           # Internet-wide discovery
    max_peers: int = Field(default=50, ge=1, le=500)
    heartbeat_interval_sec: int = Field(default=30, ge=5, le=300)
    connection_timeout_sec: int = Field(default=10, ge=1, le=60)
    protocol_id: str = "/derag/1.0.0"
    node_key_path: Path = Path("~/.derag/node_key").expanduser()


class QueryConfig(BaseModel):
    """Query engine configuration."""
    top_k: int = Field(default=10, ge=1, le=100)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    timeout_sec: int = Field(default=30, ge=1, le=300)
    max_context_tokens: int = Field(default=4096, ge=256, le=128000)
    rerank: bool = True
    differential_privacy: bool = False
    dp_epsilon: float = Field(default=1.0, ge=0.01, le=10.0)


class LLMConfig(BaseModel):
    """Local LLM configuration (Ollama)."""
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=64, le=32000)
    embedding_model: str = "all-MiniLM-L6-v2"


class LineageConfig(BaseModel):
    """Data lineage and audit configuration."""
    enable_merkle: bool = True
    enable_audit_log: bool = True
    audit_log_path: Path = Path("~/.derag/audit.log").expanduser()
    log_retention_days: int = Field(default=365, ge=30)
    sign_audit_entries: bool = True


class DeRAGConfig(BaseSettings):
    """
    Root configuration for a De-RAG node.
    
    Loads from environment variables prefixed with DERAG_,
    e.g. DERAG_NODE_NAME=mynode, DERAG_CRYPTO__ALGORITHM=chacha20-poly1305
    """
    model_config = {"env_prefix": "DERAG_", "env_nested_delimiter": "__"}

    node_name: str = Field(default="derag-node-01")
    node_id: Optional[str] = None  # Auto-generated if not set
    data_dir: Path = Path("~/.derag").expanduser()
    log_level: str = "INFO"

    crypto: CryptoConfig = Field(default_factory=CryptoConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    sharding: ShardConfig = Field(default_factory=ShardConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    lineage: LineageConfig = Field(default_factory=LineageConfig)

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        dirs = [
            self.data_dir,
            self.storage.data_dir,
            self.data_dir / "shards",
            self.data_dir / "keys",
            self.data_dir / "index",
            self.data_dir / "lineage",
            self.data_dir / "logs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions on key directory
            if "keys" in str(d):
                os.chmod(d, 0o700)
