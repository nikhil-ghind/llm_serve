"""Mistral chat-template rendering.

Mistral Instruct models are trained on

    <s>[INST] {system + user} [/INST] {assistant}</s>[INST] {user} [/INST]

with no dedicated system role: the system prompt is folded into the first user
turn. Getting this wrong is a silent quality regression rather than an error,
which is why it is a small pure function with its own tests instead of an
inline f-string in the request handler.
"""

from __future__ import annotations

from typing import Sequence

BOS = "<s>"
EOS = "</s>"
INST_OPEN = "[INST]"
INST_CLOSE = "[/INST]"


class TemplateError(ValueError):
    """Raised when a message list cannot be rendered into a prompt."""


def render_mistral_prompt(
    messages: Sequence[dict[str, str]],
    add_bos: bool = True,
    add_generation_prompt: bool = True,
) -> str:
    """Render OpenAI-style messages into a Mistral Instruct prompt."""
    if not messages:
        raise TemplateError("messages must not be empty")

    turns = [dict(m) for m in messages]
    system_parts = [m.get("content", "") for m in turns if m.get("role") == "system"]
    turns = [m for m in turns if m.get("role") != "system"]
    if not turns:
        raise TemplateError("at least one user message is required")
    if any(m.get("role") == "tool" for m in turns):
        raise TemplateError("tool messages are not supported by the Mistral template")

    # Fold the system prompt into the first user turn.
    if system_parts:
        if turns[0].get("role") != "user":
            raise TemplateError("a system prompt must be followed by a user message")
        system = "\n".join(p for p in system_parts if p)
        turns[0]["content"] = f"{system}\n\n{turns[0].get('content', '')}".strip()

    for i, message in enumerate(turns):
        expected = "user" if i % 2 == 0 else "assistant"
        if message.get("role") != expected:
            raise TemplateError(
                f"messages must alternate user/assistant; position {i} is "
                f"{message.get('role')!r}, expected {expected!r}"
            )

    out = BOS if add_bos else ""
    for i, message in enumerate(turns):
        content = (message.get("content") or "").strip()
        if i % 2 == 0:
            out += f"{INST_OPEN} {content} {INST_CLOSE}"
        else:
            out += f" {content}{EOS}"

    if turns[-1]["role"] == "assistant" and add_generation_prompt:
        raise TemplateError("the last message must be from the user to generate a reply")
    return out


def render_completion_prompt(prompt: str, lora_adapter: str | None = None) -> str:
    """Raw-completion passthrough.

    Text completions are sent verbatim: applying a chat template here would
    corrupt few-shot prompts that already carry their own formatting. The
    adapter name is accepted for symmetry with the chat path and for logging.
    """
    del lora_adapter
    return prompt


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token) for accounting.

    Used only where a real tokenizer is unavailable — the GPU backends report
    exact counts from the engine.
    """
    return max(1, len(text) // 4)
