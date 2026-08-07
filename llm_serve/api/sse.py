"""Server-sent-event framing for token streaming.

The OpenAI streaming protocol is SSE with a JSON object per event and a literal
``data: [DONE]`` sentinel at the end. Framing is pure string work, so it lives
here and is unit-tested directly rather than through a live HTTP server.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterable

DONE = "[DONE]"
SSE_MEDIA_TYPE = "text/event-stream"
#: Headers that keep proxies from buffering a token stream into one blob.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def format_sse(data: str, event: str | None = None, retry_ms: int | None = None) -> str:
    """Frame a raw payload as one SSE event.

    Multi-line payloads become several ``data:`` lines, per the SSE spec.
    """
    lines: list[str] = []
    if event is not None:
        lines.append(f"event: {event}")
    if retry_ms is not None:
        lines.append(f"retry: {retry_ms}")
    for line in data.split("\n"):
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def format_json_sse(payload: dict[str, Any], event: str | None = None) -> str:
    """Frame a JSON object as one SSE event (compact, no spurious whitespace)."""
    return format_sse(json.dumps(payload, separators=(",", ":")), event=event)


def done_event() -> str:
    """The terminating ``data: [DONE]`` event."""
    return format_sse(DONE)


def comment(text: str = "") -> str:
    """An SSE comment line — used as a keep-alive ping through idle proxies."""
    return f": {text}\n\n"


def parse_sse_stream(raw: str) -> list[str]:
    """Extract the ``data`` payloads from an SSE byte stream. Test helper."""
    payloads: list[str] = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        data_lines = [
            line[len("data:") :].lstrip(" ")
            for line in block.split("\n")
            if line.startswith("data:")
        ]
        if data_lines:
            payloads.append("\n".join(data_lines))
    return payloads


def parse_sse_json(raw: str) -> list[dict[str, Any]]:
    """Parsed JSON payloads from a stream, excluding the ``[DONE]`` sentinel."""
    return [json.loads(p) for p in parse_sse_stream(raw) if p != DONE]


def iter_sse(payloads: Iterable[dict[str, Any]], send_done: bool = True) -> Iterable[str]:
    """Synchronous framing of a payload sequence."""
    for payload in payloads:
        yield format_json_sse(payload)
    if send_done:
        yield done_event()


async def aiter_sse(
    payloads: AsyncIterator[dict[str, Any]], send_done: bool = True
) -> AsyncIterator[str]:
    """Async framing of a payload stream, for use as an ASGI response body."""
    async for payload in payloads:
        yield format_json_sse(payload)
    if send_done:
        yield done_event()
