<div align="center">

# De-RAG

### Decentralized Encrypted Retrieval-Augmented Generation

*Privacy-first knowledge infrastructure for the sovereign AI stack*

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Part of Crux Ecosystem](https://img.shields.io/badge/crux-ecosystem-orange)](https://github.com/crux-ecosystem)

</div>

---

## What is De-RAG?

De-RAG is a **fully encrypted, decentralized RAG system** where *no single node ever holds complete data*. It combines:

- **Envelope Encryption** — AES-256-GCM with Argon2id key derivation (two-layer DEK/KEK)
- **Shamir's Secret Sharing** — Documents sharded across peers; reconstruction requires k-of-n threshold
- **Searchable Encryption** — Query encrypted indexes without decryption (SSE + encrypted LSH)
- **P2P Network** — Zero-trust peer communication with Ed25519 identity and length-prefix framing
- **BLAKE3 Audit Chain** — Tamper-proof Merkle-chained lineage log

This is **Phase 2** of the [Crux Sovereign AI Stack](https://github.com/crux-ecosystem):

```
MOL Language → De-RAG → Neural Kernel
(compute)      (data)    (orchestration)
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        De-RAG Node                           │
├──────────────────────────────────────────────────────────────┤
│  CLI / HTTP API                                              │
├──────────┬──────────┬───────────┬──────────┬────────────────┤
│  Crypto  │ Storage  │  Network  │  Query   │   Lineage      │
│  ────    │ ────     │  ────     │  ────    │   ────         │
│ Envelope │ SQLite   │ P2P TCP   │ Embed    │ BLAKE3 chain   │
│ KeyMgr   │ FAISS    │ Protocol  │ Search   │ Merkle audit   │
│ SSE/LSH  │ Shards   │ Peers     │ Dist.    │ Tamper-proof   │
├──────────┴──────────┴───────────┴──────────┴────────────────┤
│                      MOL Bridge                              │
│            (execute pipelines in MOL language)                │
└──────────────────────────────────────────────────────────────┘
                 ↕ P2P (DRGP protocol)
┌──────────────────────────────────────────────────────────────┐
│                     Other De-RAG Nodes                       │
│         ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│         │ Shard 1 │  │ Shard 3 │  │ Shard 5 │              │
│         └─────────┘  └─────────┘  └─────────┘              │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install de-rag

# Initialize a node
derag init --name my-node

# Ingest documents (encrypted locally)
derag ingest documents/*.pdf

# Query
derag query "What is the contract renewal date?"

# Start P2P daemon  
derag network start --port 9090

# Join peers
derag network start --port 9091 --bootstrap 192.168.1.100:9090

# View audit log
derag audit --verify
```

## Installation

### From Source (development)

```bash
git clone https://github.com/crux-ecosystem/de-rag.git
cd de-rag
pip install -e ".[dev]"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `cryptography` | AES-256-GCM, Ed25519, X25519, HKDF |
| `blake3` | BLAKE3 hashing (integrity + audit) |
| `numpy` | Vector operations |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Text → embeddings |
| `pydantic` | Configuration validation |
| `msgpack` | Wire protocol serialization |
| `click` | CLI framework |

## Security Model

### Encryption

```
Password → Argon2id → KEK (Key Encryption Key)
                        │
                        ▼
                   DEK (Data Encryption Key)
                        │
                        ▼
              AES-256-GCM(plaintext, DEK, nonce)
                        │
                        ▼
                  EncryptedBlob {
                    magic: "DERG"
                    version: 1
                    nonce: 12 bytes
                    encrypted_dek: 72 bytes
                    ciphertext: variable
                  }
```

### Shamir's Secret Sharing

Data is split across peers using polynomial interpolation over GF(2^8). With n=5, k=3:

- 5 shards distributed to 5 peers
- Any 3 shards reconstruct the original
- 2 shards reveal *zero information*
- Each shard is independently encrypted

### Searchable Encryption

Two complementary approaches:

1. **SSE (Symmetric Searchable Encryption)** — HMAC-based token matching for exact keyword search
2. **Encrypted LSH** — Locality-Sensitive Hashing over encrypted embeddings for approximate semantic search

### Audit Trail

Every operation is logged in a BLAKE3 hash-chained append-only log:

```
entry[0].hash = BLAKE3(entry[0].data)
entry[n].hash = BLAKE3(entry[n].data || entry[n-1].hash)
```

Tampering with any entry breaks the chain. Verification: `derag audit --verify`

## Programmatic Usage

```python
import asyncio
from derag import DeRAGNode, DeRAGConfig

async def main():
    config = DeRAGConfig(node_name="research-node")
    node = DeRAGNode(config)
    
    await node.initialize("strong-password-here")
    
    # Ingest
    doc = await node.ingest("paper.pdf")
    print(f"Ingested: {doc.doc_id}")
    
    # Query
    answer = await node.query("What are the main findings?")
    print(answer.text)
    
    # Distribute shards to peers
    await node.join_network(port=9090)
    placements = await node.distribute_shards(doc.doc_id)
    
    await node.shutdown()

asyncio.run(main())
```

## MOL Integration

Define encryption pipelines in the [MOL language](https://github.com/crux-ecosystem/mol-lang):

```mol
// encrypt_pipeline.mol
let pipeline = fn(filepath) {
    let content = io::read(filepath)
    let encrypted = derag_encrypt(content)
    let shards = derag_shard(encrypted, 5, 3)
    let hash = derag_hash(content)
    
    { "hash": hash, "shards": shards }
}

pipeline("secret_doc.pdf")
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run crypto tests only
pytest tests/test_crypto.py -v

# With coverage
pytest tests/ --cov=derag --cov-report=term-missing
```

## Project Structure

```
De-RAG/
├── derag/
│   ├── __init__.py          # Package root
│   ├── config.py            # Pydantic configuration
│   ├── node.py              # Main orchestrator
│   ├── cli.py               # Click CLI
│   ├── crypto/
│   │   ├── envelope.py      # AES-256-GCM envelope encryption
│   │   ├── keys.py          # Key lifecycle management
│   │   └── searchable.py    # SSE + Encrypted LSH
│   ├── storage/
│   │   ├── vector_index.py  # Encrypted FAISS index
│   │   ├── local_store.py   # SQLite encrypted document store
│   │   └── shard_manager.py # Shamir's Secret Sharing (GF(2^8))
│   ├── network/
│   │   ├── protocol.py      # DRGP wire protocol
│   │   └── peer.py          # Async P2P communication
│   ├── query/
│   │   └── engine.py        # Distributed query processing
│   ├── lineage/
│   │   └── audit.py         # BLAKE3 Merkle audit chain
│   └── mol_bridge/
│       └── __init__.py      # MOL language integration
├── tests/
│   ├── conftest.py
│   └── test_crypto.py
├── pyproject.toml
└── README.md
```

## Roadmap

- [ ] **v0.1.0** — Core engine (encryption, storage, sharding, P2P) ← *current*
- [ ] **v0.2.0** — HTTP API server (FastAPI), web dashboard
- [ ] **v0.3.0** — Neural Kernel integration (agent-controlled RAG)
- [ ] **v0.4.0** — Multi-modal support (images, audio, video)
- [ ] **v1.0.0** — Production hardening, formal security audit

## Part of the Crux Ecosystem

| Project | Role | Status |
|---------|------|--------|
| [MOL](https://github.com/crux-ecosystem/mol-lang) | Sovereign compute language | v1.1.0 |
| **De-RAG** | Decentralized encrypted data layer | v0.1.0 |
| [Neural Kernel](https://github.com/crux-ecosystem/neural-kernel) | AI microkernel & orchestration | In development |
| [IntraMind](https://github.com/crux-ecosystem/IntraMind-Showcase) | Campus RAG showcase | Pre-production |

## License

AGPL-3.0 — See [LICENSE](LICENSE) for details.

**Copyright (c) 2026 CruxLabx**
