"""Core request/response types shared by every backend.

Deliberately dependency-free (stdlib only) so the whole type layer imports and
runs on a CPU-only machine with no CUDA, no torch and no model weights.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class ValidationError(ValueError):
    """Raised when a client-supplied request is malformed."""


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"
    ERROR = "error"


def _require_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def _check_range(name: str, value: float, low: float, high: float) -> float:
    if not low <= value <= high:
        raise ValidationError(f"{name} must be within [{low}, {high}], got {value}")
    return value


@dataclass(frozen=True)
class SamplingParams:
    """Decoding parameters, mirroring the OpenAI completions surface."""

    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: tuple[str, ...] = ()
    seed: int | None = None
    n: int = 1
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise ValidationError("max_tokens must be an integer")
        if self.max_tokens < 1:
            raise ValidationError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.max_tokens > 32768:
            raise ValidationError("max_tokens exceeds the 32768 hard ceiling")
        _check_range("temperature", _require_number("temperature", self.temperature), 0.0, 2.0)
        _check_range("top_p", _require_number("top_p", self.top_p), 0.0, 1.0)
        if self.top_p == 0.0:
            raise ValidationError("top_p must be > 0")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ValidationError("top_k must be an integer")
        if self.top_k == 0 or self.top_k < -1:
            raise ValidationError("top_k must be -1 (disabled) or >= 1")
        _check_range(
            "presence_penalty", _require_number("presence_penalty", self.presence_penalty), -2.0, 2.0
        )
        _check_range(
            "frequency_penalty",
            _require_number("frequency_penalty", self.frequency_penalty),
            -2.0,
            2.0,
        )
        if _require_number("repetition_penalty", self.repetition_penalty) <= 0:
            raise ValidationError("repetition_penalty must be > 0")
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise ValidationError("n must be an integer >= 1")
        if len(self.stop) > 8:
            raise ValidationError("at most 8 stop sequences are supported")
        for s in self.stop:
            if not isinstance(s, str):
                raise ValidationError("stop sequences must be strings")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SamplingParams":
        """Build from a loosely-typed client payload, ignoring unknown keys."""
        stop = payload.get("stop", ())
        if stop is None:
            stop = ()
        elif isinstance(stop, str):
            stop = (stop,)
        else:
            stop = tuple(stop)
        known = {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "seed",
            "n",
            "ignore_eos",
        }
        kwargs = {k: v for k, v in payload.items() if k in known and v is not None}
        return cls(stop=stop, **kwargs)


@dataclass
class GenerationRequest:
    """A single unit of work handed to a backend."""

    prompt: str
    sampling: SamplingParams = field(default_factory=SamplingParams)
    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex[:16]}")
    model: str = "mistral-7b-qlora"
    prompt_token_ids: Sequence[int] | None = None
    arrival_time: float = field(default_factory=time.monotonic)
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise ValidationError("prompt must be a string")
        if not self.prompt and not self.prompt_token_ids:
            raise ValidationError("prompt must not be empty")

    @property
    def prompt_len(self) -> int:
        """Prompt length in tokens (falls back to a ~4 chars/token estimate)."""
        if self.prompt_token_ids is not None:
            return len(self.prompt_token_ids)
        return max(1, len(self.prompt) // 4)


@dataclass
class TokenChunk:
    """One streamed decode step."""

    request_id: str
    index: int
    text: str
    token_id: int | None = None
    finish_reason: FinishReason | None = None
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None


@dataclass
class GenerationResult:
    """Terminal result for a request, assembled from its chunks."""

    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason = FinishReason.STOP
    ttft_s: float | None = None
    e2e_latency_s: float | None = None
    inter_token_latencies_s: list[float] = field(default_factory=list)
    cached_prompt_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def mean_itl_s(self) -> float | None:
        if not self.inter_token_latencies_s:
            return None
        return sum(self.inter_token_latencies_s) / len(self.inter_token_latencies_s)

    @property
    def output_tokens_per_s(self) -> float | None:
        if not self.e2e_latency_s or self.e2e_latency_s <= 0:
            return None
        return self.completion_tokens / self.e2e_latency_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "finish_reason": self.finish_reason.value,
            "ttft_s": self.ttft_s,
            "e2e_latency_s": self.e2e_latency_s,
            "mean_itl_s": self.mean_itl_s,
        }


class ResultAccumulator:
    """Folds a chunk stream into a :class:`GenerationResult` with timings."""

    def __init__(self, request: GenerationRequest, start_time: float | None = None) -> None:
        self.request = request
        self.start = start_time if start_time is not None else time.monotonic()
        self._pieces: list[str] = []
        self._first_token_time: float | None = None
        self._last_token_time: float | None = None
        self._itls: list[float] = []
        self._finish = FinishReason.STOP
        self._n_tokens = 0
        self.cached_prompt_tokens = 0

    def add(self, chunk: TokenChunk) -> None:
        if chunk.text:
            self._pieces.append(chunk.text)
            self._n_tokens += 1
            if self._first_token_time is None:
                self._first_token_time = chunk.timestamp
            elif self._last_token_time is not None:
                self._itls.append(chunk.timestamp - self._last_token_time)
            self._last_token_time = chunk.timestamp
        if chunk.finish_reason is not None:
            self._finish = chunk.finish_reason

    def finish(self, end_time: float | None = None) -> GenerationResult:
        end = end_time if end_time is not None else time.monotonic()
        ttft = None if self._first_token_time is None else self._first_token_time - self.start
        return GenerationResult(
            request_id=self.request.request_id,
            text="".join(self._pieces),
            prompt_tokens=self.request.prompt_len,
            completion_tokens=self._n_tokens,
            finish_reason=self._finish,
            ttft_s=ttft,
            e2e_latency_s=end - self.start,
            inter_token_latencies_s=list(self._itls),
            cached_prompt_tokens=self.cached_prompt_tokens,
        )
