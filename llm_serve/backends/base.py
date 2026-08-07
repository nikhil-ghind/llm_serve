"""Backend protocol and registry.

Every engine — vLLM, Triton, Ray Serve, TensorRT-LLM and the CPU mock — implements
the same small async surface. Nothing above this layer knows which engine is live,
which is what makes the cross-backend benchmark an apples-to-apples comparison.

The registry maps a name to a *lazy* factory so selecting one backend never imports
the others (and never imports CUDA-dependent packages on a CPU-only host).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Callable

from ..types import GenerationRequest, TokenChunk

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config


class BackendUnavailable(RuntimeError):
    """Raised when a backend's runtime dependencies or hardware are missing.

    Carries the install hint so the operator sees the fix, not just the ImportError.
    """

    def __init__(self, backend: str, reason: str, hint: str | None = None) -> None:
        message = f"backend {backend!r} is unavailable: {reason}"
        if hint:
            message += f"\n  hint: {hint}"
        super().__init__(message)
        self.backend = backend
        self.reason = reason
        self.hint = hint


@dataclass
class BackendStats:
    """Point-in-time engine state, surfaced on /metrics and in benchmark runs."""

    backend: str
    running: int = 0
    waiting: int = 0
    swapped: int = 0
    kv_cache_usage: float = 0.0
    prefix_cache_hit_rate: float = 0.0
    total_prompt_tokens: int = 0
    total_generation_tokens: int = 0
    preemptions: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int | str]:
        base: dict[str, float | int | str] = {
            "backend": self.backend,
            "running": self.running,
            "waiting": self.waiting,
            "swapped": self.swapped,
            "kv_cache_usage": self.kv_cache_usage,
            "prefix_cache_hit_rate": self.prefix_cache_hit_rate,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generation_tokens": self.total_generation_tokens,
            "preemptions": self.preemptions,
        }
        base.update(self.extra)
        return base


@dataclass
class HealthStatus:
    ok: bool
    backend: str
    detail: str = ""
    model: str | None = None


class Backend(abc.ABC):
    """Common interface implemented by every serving backend."""

    #: Registry name, e.g. ``"vllm"``.
    name: str = "base"

    def __init__(self, config: "Config") -> None:
        self.config = config
        self._started = False

    async def start(self) -> None:
        """Load the engine. Heavy imports belong here, never at module scope."""
        self._started = True

    async def stop(self) -> None:
        """Release engine resources. Safe to call when never started."""
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @abc.abstractmethod
    def generate_stream(self, request: GenerationRequest) -> AsyncIterator[TokenChunk]:
        """Yield :class:`TokenChunk` objects; the last one carries a finish reason."""
        raise NotImplementedError

    @abc.abstractmethod
    async def abort(self, request_id: str) -> None:
        """Cancel an in-flight request (client disconnect)."""
        raise NotImplementedError

    async def health(self) -> HealthStatus:
        return HealthStatus(
            ok=self._started,
            backend=self.name,
            detail="started" if self._started else "not started",
            model=self.config.model.name,
        )

    async def stats(self) -> BackendStats:
        return BackendStats(backend=self.name)

    async def __aenter__(self) -> "Backend":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()


BackendFactory = Callable[["Config"], Backend]

_REGISTRY: dict[str, Callable[[], type[Backend]]] = {}


def register_backend(name: str, loader: Callable[[], type[Backend]]) -> None:
    """Register a lazy loader returning the backend class on first use."""
    if name in _REGISTRY:
        raise ValueError(f"backend {name!r} is already registered")
    _REGISTRY[name] = loader


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_backend_class(name: str) -> type[Backend]:
    """Resolve a backend name to its class, importing the module on demand."""
    try:
        loader = _REGISTRY[name]
    except KeyError:
        raise BackendUnavailable(
            name,
            "no such backend",
            f"available backends: {', '.join(available_backends())}",
        ) from None
    return loader()


def create_backend(config: "Config", name: str | None = None) -> Backend:
    """Instantiate the configured backend without starting it."""
    return get_backend_class(name or config.backend.kind)(config)


def _load_mock() -> type[Backend]:
    from .mock import MockBackend

    return MockBackend


def _load_vllm() -> type[Backend]:
    from .vllm_backend import VLLMBackend

    return VLLMBackend


def _load_triton() -> type[Backend]:
    from .triton_backend import TritonBackend

    return TritonBackend


def _load_ray() -> type[Backend]:
    from .ray_backend import RayBackend

    return RayBackend


def _load_trtllm() -> type[Backend]:
    from .trtllm_backend import TensorRTLLMBackend

    return TensorRTLLMBackend


register_backend("mock", _load_mock)
register_backend("vllm", _load_vllm)
register_backend("triton", _load_triton)
register_backend("ray", _load_ray)
register_backend("trtllm", _load_trtllm)
