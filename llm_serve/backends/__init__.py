"""Serving backends behind a single protocol.

Importing this package is CPU-safe: concrete engines are resolved lazily through
the registry in :mod:`llm_serve.backends.base`.
"""

from .base import (
    Backend,
    BackendStats,
    BackendUnavailable,
    HealthStatus,
    available_backends,
    create_backend,
    get_backend_class,
    register_backend,
)

__all__ = [
    "Backend",
    "BackendStats",
    "BackendUnavailable",
    "HealthStatus",
    "available_backends",
    "create_backend",
    "get_backend_class",
    "register_backend",
]
