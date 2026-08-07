"""llm_serve — multi-backend LLM inference serving for a fine-tuned Mistral 7B.

Importing this package is always safe on a CPU-only machine: nothing here pulls in
vLLM, TensorRT-LLM, Ray or torch. Heavy engine imports happen lazily inside the
individual backend implementations when they are actually started.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
