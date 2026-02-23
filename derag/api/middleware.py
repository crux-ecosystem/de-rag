"""
De-RAG API — Middleware
=======================

Authentication, rate limiting, CORS, and request logging.

Author: Mounesh Kodi — CruxLabx
Copyright (c) 2026 CruxLabx — AGPL-3.0
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from collections import defaultdict
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("derag.api")


# ─── Bearer Token Auth ───────────────────────────────────────

class BearerAuthMiddleware(BaseHTTPMiddleware):
    """
    Simple bearer-token authentication.

    Public endpoints (health, docs) are exempted.
    Token is set at server start and compared via constant-time HMAC.
    """

    PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, token: Optional[str] = None):
        super().__init__(app)
        self.token = token  # None = auth disabled

    async def dispatch(self, request: Request, call_next: Callable):
        if self.token is None:
            return await call_next(request)

        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401,
            )

        provided = auth_header[7:]
        if not hmac.compare_digest(provided, self.token):
            return JSONResponse(
                {"error": "Invalid bearer token"},
                status_code=403,
            )

        return await call_next(request)


# ─── Rate Limiting ───────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory sliding window rate limiter.

    Limits per client IP. Default: 100 requests / 60 seconds.
    """

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - self.window

        # Prune old entries
        self._hits[client_ip] = [t for t in self._hits[client_ip] if t > cutoff]

        if len(self._hits[client_ip]) >= self.max_requests:
            return JSONResponse(
                {
                    "error": "Rate limit exceeded",
                    "detail": f"Max {self.max_requests} requests per {self.window}s",
                },
                status_code=429,
                headers={"Retry-After": str(self.window)},
            )

        self._hits[client_ip].append(now)
        return await call_next(request)


# ─── Request Logging ─────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"
        return response
