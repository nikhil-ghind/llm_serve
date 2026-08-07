"""OpenAI-compatible HTTP surface.

Only the pure pieces (schemas, SSE framing, chat template) are re-exported here;
``create_app`` is imported from :mod:`llm_serve.api.app` on demand so that this
package stays importable without FastAPI installed.
"""

from . import openai_schemas, sse
from .chat_template import TemplateError, render_completion_prompt, render_mistral_prompt

__all__ = [
    "TemplateError",
    "openai_schemas",
    "render_completion_prompt",
    "render_mistral_prompt",
    "sse",
]
