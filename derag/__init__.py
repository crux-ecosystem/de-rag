"""
De-RAG — Decentralized Privacy-First Retrieval-Augmented Generation
===================================================================

A peer-to-peer encrypted knowledge network where no single node holds
complete data, queries are encrypted end-to-end, and AI answers personal
questions without any central authority seeing your information.

Architecture:
    ┌─────────────────────────────────┐
    │          De-RAG Node            │
    │  ┌───────┐ ┌────────┐ ┌─────┐  │
    │  │Crypto │ │Storage │ │ P2P │  │
    │  │Engine │→│Engine  │→│Net  │  │
    │  └───────┘ └────────┘ └─────┘  │
    │  ┌───────┐ ┌────────┐ ┌─────┐  │
    │  │Query  │ │Lineage │ │ MOL │  │
    │  │Planner│→│Tracker │→│Brdg │  │
    │  └───────┘ └────────┘ └─────┘  │
    └─────────────────────────────────┘

Copyright (c) 2026 CruxLabx — Mounesh Kodi
License: AGPL-3.0
"""

__version__ = "0.1.0"
__author__ = "Mounesh Kodi"
__org__ = "CruxLabx"

from derag.node import DeRAGNode
from derag.config import DeRAGConfig

__all__ = ["DeRAGNode", "DeRAGConfig", "__version__"]
