<div align="center">

# De-RAG

### Decentralized Encrypted Retrieval-Augmented Generation

**No single node ever holds your complete data.**

[![CruxLabx](https://img.shields.io/badge/Built%20by-CruxLabx-blueviolet?style=for-the-badge)](https://github.com/crux-ecosystem)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)]()

[Architecture](#architecture) • [How It Works](#how-it-works) • [Security Model](#security-model) • [Benchmarks](#benchmarks) • [Ecosystem](#part-of-crux-sovereign-ai-stack)

</div>

---

## The Problem

Every RAG system today has the same fatal flaw: **your data sits in plaintext on someone's server**. Vector databases, embedding APIs, LLM providers — they all have access to your unencrypted documents. One breach, one subpoena, one rogue employee, and your intellectual property is gone.

**De-RAG eliminates this entirely.**

## What is De-RAG?

De-RAG is a **privacy-first knowledge infrastructure** where documents are encrypted *before* they enter the system, sharded across a peer-to-peer network, and queried without ever being decrypted at any single point.

It is **not** a wrapper around OpenAI + Pinecone. It is a ground-up rethinking of how retrieval-augmented generation should work when privacy is non-negotiable.

### Core Capabilities

| Capability | What It Does |
|-----------|-------------|
| **Envelope Encryption** | Two-layer AES-256-GCM (DEK/KEK) with Argon2id key derivation. Documents encrypted at rest *and* in transit. |
| **Shamir Secret Sharing** | Documents split into `n` shards using polynomial interpolation over GF(2^8). Any `k` shards reconstruct; fewer reveal *zero information*. |
| **Searchable Encryption** | Query encrypted indexes without decrypting. SSE for exact keyword matching + encrypted LSH for approximate semantic search. |
| **Zero-Trust P2P** | Ed25519 identity-based peer communication. Length-prefix framing. No central coordinator. |
| **Merkle Audit Chain** | Every operation logged in a BLAKE3 hash-chained append-only ledger. Tamper with one entry → the entire chain breaks. |
| **MOL Integration** | Define encryption pipelines in the [MOL language](https://github.com/crux-ecosystem/mol-lang) — the sovereign compute layer. |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        De-RAG Node                           │
├──────────────────────────────────────────────────────────────┤
│  CLI / HTTP API                                              │
├──────────┬──────────┬───────────┬──────────┬────────────────┤
│  Crypto  │ Storage  │  Network  │  Query   │   Lineage      │
│  ──────  │ ──────── │  ──────── │  ──────  │   ──────────   │
│ Envelope │ SQLite   │ P2P TCP   │ Embed    │ BLAKE3 chain   │
│ KeyMgr   │ FAISS    │ DRGP      │ Dist.    │ Merkle audit   │
│ SSE/LSH  │ Shards   │ Ed25519   │ Search   │ Tamper-proof   │
├──────────┴──────────┴───────────┴──────────┴────────────────┤
│                      MOL Bridge                              │
│            (execute pipelines in MOL language)                │
├──────────────────────────────────────────────────────────────┤
│                   IntraMind Delegator                         │
│          (delegates local RAG to IntraMind engine)            │
└──────────────────────────────────────────────────────────────┘
                 ↕ P2P (DRGP protocol)
┌──────────────────────────────────────────────────────────────┐
│                     Peer Network                             │
│      ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐       │
│      │Node A│────▶│Node B│────▶│Node C│────▶│Node D│       │
│      │Shard │     │Shard │     │Shard │     │Shard │       │
│      │1,3   │     │2,5   │     │1,4   │     │3,5   │       │
│      └──────┘     └──────┘     └──────┘     └──────┘       │
│      No single node holds the complete document.             │
└──────────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1. Ingest — Encrypt Everything

```
Document
   │
   ▼
Password ──▶ Argon2id ──▶ KEK (Key Encryption Key)
                            │
                            ▼
                       Random DEK
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

### 2. Shard — Split Across Peers

Using Shamir's Secret Sharing over GF(2^8) with a primitive generator of order 255:

- **5 shards** distributed to 5 peers
- **Any 3** reconstruct the original
- **2 shards** reveal **zero information** (information-theoretic security)
- Each shard is independently encrypted before distribution

### 3. Query — Search Without Decrypting

```
User Query
    │
    ├──▶ SSE Token = HMAC(key, keyword)     → exact match
    │
    └──▶ Encrypted LSH = Hash(embedding)     → semantic search
              │
              ▼
         Distributed query across peers
              │
              ▼
         Local decryption + ranking
              │
              ▼
         Answer (never left the node)
```

### 4. Audit — Verify Everything

```
entry[0].hash = BLAKE3(entry[0].data)
entry[n].hash = BLAKE3(entry[n].data ∥ entry[n-1].hash)
```

Tamper with any entry → chain breaks → detected immediately.

---

## Security Model

| Layer | Algorithm | Standard |
|-------|-----------|----------|
| Symmetric encryption | AES-256-GCM | NIST SP 800-38D |
| Key derivation | Argon2id (t=3, m=64MB, p=4) | RFC 9106, OWASP |
| Secret sharing | Shamir SSS over GF(2^8) | Shamir 1979 |
| Content hashing | BLAKE3 | BLAKE3 spec |
| Identity & auth | Ed25519 | RFC 8032 |
| Key exchange | X25519 + HKDF | RFC 7748 |
| Searchable encryption | SSE (HMAC-based) | Song, Wagner, Perrig 2000 |
| Approximate search | Encrypted LSH | Kuzu et al., Bost 2016 |

### Threat Model

- **Storage compromise** → Attacker gets encrypted blobs. Useless without KEK.
- **Single peer compromise** → Attacker gets k-1 shards. Information-theoretically zero knowledge.
- **Network MITM** → Ed25519 + X25519 authenticated channel. Forward secrecy.
- **Insider threat** → BLAKE3 audit chain detects unauthorized operations.

---

## Benchmarks

*Preliminary benchmarks on a single node (Intel i7, 32GB RAM):*

| Operation | Throughput | Notes |
|-----------|-----------|-------|
| AES-256-GCM encrypt | ~2.1 GB/s | Hardware-accelerated AES-NI |
| BLAKE3 hash | ~6.8 GB/s | SIMD-optimized |
| Shamir split (3-of-5) | ~45 MB/s | GF(2^8) polynomial evaluation |
| Shamir reconstruct | ~38 MB/s | Lagrange interpolation |
| SSE token generation | ~1.2M tokens/s | HMAC-SHA256 based |
| Vector search (10K docs) | ~4ms p99 | FAISS IVF-PQ |

---

## MOL Language Integration

De-RAG pipelines can be defined in the [MOL language](https://github.com/crux-ecosystem/mol-lang), the sovereign compute layer of the Crux ecosystem:

```mol
-- Encrypt, shard, and distribute in 5 lines
let key be derag_keygen("master_key")
let envelope be derag_encrypt(document, key.key_id)
let shards be derag_shard(document, 3, 5)
derag_distribute(shards, peers)
let audit be derag_audit()
```

38 new MOL builtins connect De-RAG, Neural Kernel, and IntraMind into a single programmable stack.

---

## Part of Crux Sovereign AI Stack

De-RAG is **Phase 2** of a three-layer sovereign AI infrastructure:

```
┌─────────────────────────────────────────────────────┐
│  MOL Language         — Sovereign compute            │
│  ─────────────────────────────────────────────────── │
│  De-RAG               — Encrypted decentralized data │  ◀── You are here
│  ─────────────────────────────────────────────────── │
│  Neural Kernel        — AI microkernel orchestration │
│  ─────────────────────────────────────────────────── │
│  IntraMind            — RAG engine (offline-first)   │
└─────────────────────────────────────────────────────┘
```

| Project | Role | Status |
|---------|------|--------|
| [MOL](https://github.com/crux-ecosystem/mol-lang) | Sovereign compute language | v1.1.0 |
| **De-RAG** | Decentralized encrypted data layer | v0.1.0 |
| [Neural Kernel](https://github.com/crux-ecosystem/neural-kernel) | AI microkernel & orchestration | v0.1.0 |
| [IntraMind](https://github.com/crux-ecosystem/IntraMind-Showcase) | Offline-first RAG engine | v1.1.0 |

---

## Roadmap

- [x] **v0.1.0** — Core engine: envelope encryption, key management, Shamir SSS, SSE, P2P protocol, BLAKE3 audit, MOL bridge
- [ ] **v0.2.0** — HTTP API server (FastAPI), admin dashboard
- [ ] **v0.3.0** — Neural Kernel agent integration (capability-gated RAG operations)
- [ ] **v0.4.0** — Multi-modal support (images, audio, video)
- [ ] **v0.5.0** — Homomorphic encryption for compute-on-encrypted-data
- [ ] **v1.0.0** — Production hardening, formal security audit, FIPS certification

---

## Build With Us

**CruxLabx** is building the next generation of sovereign AI infrastructure — systems where privacy isn't an afterthought, it's the foundation.

We're a small team solving hard problems: post-quantum cryptography, decentralized consensus, information-theoretic security. If that excites you, we want to hear from you.

**We're looking for:**
- Cryptography engineers (envelope encryption, MPC, ZKP)
- Distributed systems developers (P2P, consensus, sharding)
- Security researchers (formal verification, audit)
- ML engineers who care about privacy

Reach out, open an issue, or start a discussion. The best ideas come from people who refuse to accept the status quo.

**[github.com/crux-ecosystem](https://github.com/crux-ecosystem)**

---

<div align="center">

**Copyright © 2026 CruxLabx**

*Sovereign AI infrastructure — because your data should answer only to you.*

</div>
