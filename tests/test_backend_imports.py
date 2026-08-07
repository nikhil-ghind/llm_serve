"""Import safety: nothing in this repo may pull in a GPU stack at import time.

If any of these fail, the package has stopped being usable (or testable) on a
machine without CUDA — which is most developer laptops and all of CI.
"""

import importlib
import subprocess
import sys
import unittest

HEAVY = ("vllm", "tensorrt_llm", "ray", "torch", "tritonclient", "pynvml")

MODULES = [
    "llm_serve",
    "llm_serve.types",
    "llm_serve.config",
    "llm_serve.server",
    "llm_serve.backends",
    "llm_serve.backends.base",
    "llm_serve.backends.mock",
    "llm_serve.backends.vllm_backend",
    "llm_serve.backends.triton_backend",
    "llm_serve.backends.ray_backend",
    "llm_serve.backends.trtllm_backend",
    "llm_serve.engine",
    "llm_serve.engine.block_manager",
    "llm_serve.engine.prefix_cache",
    "llm_serve.engine.scheduler",
    "llm_serve.engine.stats",
    "llm_serve.api",
    "llm_serve.api.app",
    "llm_serve.api.sse",
    "llm_serve.api.openai_schemas",
    "llm_serve.api.chat_template",
    "llm_serve.metrics",
    "llm_serve.metrics.math",
    "llm_serve.metrics.registry",
]


class TestImportSafety(unittest.TestCase):
    def test_every_module_imports_on_cpu(self):
        for name in MODULES:
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))

    def test_no_gpu_package_is_imported(self):
        """Import the whole package in a clean interpreter and check sys.modules."""
        code = (
            "import sys;"
            f"mods={list(MODULES)!r};"
            "[__import__(m) for m in mods];"
            f"heavy=[h for h in {list(HEAVY)!r} if h in sys.modules];"
            "print(','.join(heavy))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(
            result.stdout.strip(),
            "",
            f"GPU packages imported at module scope: {result.stdout.strip()}",
        )

    def test_api_module_does_not_require_fastapi(self):
        code = (
            "import sys, llm_serve.api.app as app;"
            "print('fastapi' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
