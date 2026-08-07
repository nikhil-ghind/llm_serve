"""Synthetic workload generation for the load generator.

Two things decide what a serving benchmark actually measures, and both live here:

* **Length distribution.** A fixed 512-in/128-out workload makes every engine
  look good; real traffic has a spread, and the spread is what exercises
  continuous batching (short requests must not queue behind long ones).
* **Shared prefix fraction.** Chat and RAG traffic repeats a long system prompt.
  Generating a controllable shared prefix is what makes the prefix-cache win
  visible instead of accidental.

Everything is seeded, so two runs against different backends see byte-identical
prompts — otherwise the comparison is noise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator, Sequence

# A small deterministic vocabulary; prompts must be reproducible, not readable.
_WORDS = (
    "system context document passage retrieve summarize analyze token latency "
    "throughput inference kernel attention batch sequence prompt completion cache "
    "memory tensor engine schedule replica request response stream decode prefill"
).split()

CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class WorkloadRequest:
    """One request in a generated workload."""

    index: int
    prompt: str
    max_tokens: int
    arrival_offset_s: float = 0.0
    shared_prefix_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return max(1, len(self.prompt) // CHARS_PER_TOKEN)


def _words_for_tokens(rng: random.Random, num_tokens: int) -> str:
    """Build text of roughly ``num_tokens`` tokens (~4 characters each)."""
    target_chars = max(CHARS_PER_TOKEN, num_tokens * CHARS_PER_TOKEN)
    parts: list[str] = []
    length = 0
    # One extra character of slack: " ".join drops the final separator that the
    # running length counts, which would otherwise leave the text a char short.
    while length < target_chars + 1:
        word = _WORDS[rng.randrange(len(_WORDS))]
        parts.append(word)
        length += len(word) + 1
    return " ".join(parts)[:target_chars]


def _sample_length(rng: random.Random, mean: int, std: int, minimum: int = 1) -> int:
    if std <= 0:
        return max(minimum, mean)
    return max(minimum, int(round(rng.gauss(mean, std))))


def poisson_arrivals(rng: random.Random, rate: float, count: int) -> list[float]:
    """Cumulative arrival offsets for a Poisson process of ``rate`` req/s.

    Inter-arrival times are exponential, which produces the bursts that a fixed
    interval schedule hides — and bursts are where queueing delay shows up.
    """
    if rate <= 0:
        raise ValueError("rate must be > 0")
    offsets: list[float] = []
    t = 0.0
    for _ in range(count):
        t += -math.log(1.0 - rng.random()) / rate
        offsets.append(t)
    return offsets


def generate_workload(
    num_requests: int,
    input_len: int = 512,
    input_len_std: int = 64,
    output_len: int = 128,
    output_len_std: int = 16,
    shared_prefix_len: int = 0,
    seed: int = 42,
    request_rate: float | None = None,
) -> list[WorkloadRequest]:
    """Generate a deterministic list of requests.

    ``request_rate`` set means open-loop: arrivals follow a Poisson process and
    the client does not wait for one response before sending the next. Left
    unset, the workload is closed-loop and arrival offsets are all zero.
    """
    if num_requests < 1:
        raise ValueError("num_requests must be >= 1")
    if shared_prefix_len >= input_len:
        raise ValueError("shared_prefix_len must be shorter than input_len")

    rng = random.Random(seed)
    prefix = _words_for_tokens(random.Random(seed ^ 0x5EED), shared_prefix_len) if shared_prefix_len else ""

    arrivals = (
        poisson_arrivals(random.Random(seed + 1), request_rate, num_requests)
        if request_rate
        else [0.0] * num_requests
    )

    requests: list[WorkloadRequest] = []
    for i in range(num_requests):
        total_in = _sample_length(rng, input_len, input_len_std, minimum=shared_prefix_len + 1)
        body = _words_for_tokens(rng, total_in - shared_prefix_len)
        prompt = f"{prefix} {body}".strip() if prefix else body
        requests.append(
            WorkloadRequest(
                index=i,
                prompt=prompt,
                max_tokens=_sample_length(rng, output_len, output_len_std),
                arrival_offset_s=arrivals[i],
                shared_prefix_tokens=shared_prefix_len,
            )
        )
    return requests


def workload_summary(requests: Sequence[WorkloadRequest]) -> dict[str, float]:
    """Descriptive stats for the generated workload, recorded alongside results."""
    if not requests:
        return {}
    prompt_tokens = [r.prompt_tokens for r in requests]
    output_tokens = [r.max_tokens for r in requests]
    return {
        "num_requests": float(len(requests)),
        "mean_input_tokens": sum(prompt_tokens) / len(prompt_tokens),
        "min_input_tokens": float(min(prompt_tokens)),
        "max_input_tokens": float(max(prompt_tokens)),
        "mean_output_tokens": sum(output_tokens) / len(output_tokens),
        "total_input_tokens": float(sum(prompt_tokens)),
        "total_output_tokens": float(sum(output_tokens)),
        "shared_prefix_tokens": float(requests[0].shared_prefix_tokens),
    }


def cycle(requests: Sequence[WorkloadRequest]) -> Iterator[WorkloadRequest]:
    """Endless repetition of a workload, for duration-bounded runs."""
    while True:
        for request in requests:
            yield request
