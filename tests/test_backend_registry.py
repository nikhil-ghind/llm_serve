import asyncio
import json
import os
import tempfile
import unittest

from llm_serve.backends.base import BackendUnavailable, create_backend, get_backend_class
from llm_serve.backends.ray_backend import RayBackend
from llm_serve.backends.triton_backend import TritonBackend
from llm_serve.backends.trtllm_backend import TensorRTLLMBackend, inspect_engine_dir
from llm_serve.backends.vllm_backend import VLLMBackend
from llm_serve.config import build_config
from llm_serve.types import GenerationRequest, SamplingParams


def _config(*overrides):
    return build_config(None, environ={}, cli_overrides=list(overrides))


class TestResolution(unittest.TestCase):
    def test_names_map_to_classes(self):
        self.assertIs(get_backend_class("vllm"), VLLMBackend)
        self.assertIs(get_backend_class("triton"), TritonBackend)
        self.assertIs(get_backend_class("ray"), RayBackend)
        self.assertIs(get_backend_class("trtllm"), TensorRTLLMBackend)

    def test_create_backend_does_not_start_engine(self):
        backend = create_backend(_config("backend.kind=vllm"))
        self.assertIsInstance(backend, VLLMBackend)
        self.assertFalse(backend.started)

    def test_explicit_name_overrides_config(self):
        backend = create_backend(_config(), name="trtllm")
        self.assertIsInstance(backend, TensorRTLLMBackend)


class TestUnavailableEngines(unittest.TestCase):
    """Starting a backend without its runtime must fail with a usable message."""

    def test_vllm_reports_install_hint(self):
        backend = VLLMBackend(_config("backend.kind=vllm"))
        with self.assertRaises(BackendUnavailable) as ctx:
            asyncio.run(backend.start())
        self.assertIn("vllm", str(ctx.exception).lower())
        self.assertIn("hint", str(ctx.exception))

    def test_trtllm_requires_an_engine_directory(self):
        cfg = _config("backend.kind=trtllm", "backend.engine_dir=/nonexistent/engine")
        with self.assertRaises(BackendUnavailable) as ctx:
            asyncio.run(TensorRTLLMBackend(cfg).start())
        self.assertIn("does not exist", str(ctx.exception))

    def test_backends_are_stoppable_before_start(self):
        for cls, overrides in (
            (VLLMBackend, ("backend.kind=vllm",)),
            (TritonBackend, ("backend.kind=triton", "backend.endpoint=localhost:8001")),
            (RayBackend, ("backend.kind=ray", "backend.endpoint=http://localhost:8000")),
            (TensorRTLLMBackend, ("backend.kind=trtllm", "backend.engine_dir=/tmp/x")),
        ):
            with self.subTest(backend=cls.__name__):
                backend = cls(_config(*overrides))
                asyncio.run(backend.stop())
                self.assertFalse(backend.started)

    def test_health_before_start_is_not_ok(self):
        backend = RayBackend(_config("backend.kind=ray", "backend.endpoint=http://x:8000"))
        status = asyncio.run(backend.health())
        self.assertFalse(status.ok)
        self.assertEqual(status.backend, "ray")


class TestEngineArgTranslation(unittest.TestCase):
    def test_vllm_engine_args_carry_config(self):
        cfg = _config(
            "backend.kind=vllm",
            "backend.tensor_parallel_size=2",
            "backend.gpu_memory_utilization=0.85",
            "scheduler.max_num_seqs=64",
            "model.max_model_len=4096",
        )
        args = VLLMBackend(cfg)._engine_args()
        self.assertEqual(args["tensor_parallel_size"], 2)
        self.assertAlmostEqual(args["gpu_memory_utilization"], 0.85)
        self.assertEqual(args["max_num_seqs"], 64)
        self.assertEqual(args["max_model_len"], 4096)
        self.assertTrue(args["enable_prefix_caching"])
        self.assertNotIn("enable_lora", args)

    def test_lora_adapter_enables_lora(self):
        cfg = _config("backend.kind=vllm", "model.lora_adapter=/models/qlora")
        args = VLLMBackend(cfg)._engine_args()
        self.assertTrue(args["enable_lora"])
        self.assertGreaterEqual(args["max_lora_rank"], 8)

    def test_quantization_is_forwarded(self):
        cfg = _config("backend.kind=vllm", "model.quantization=awq")
        self.assertEqual(VLLMBackend(cfg)._engine_args()["quantization"], "awq")

    def test_ray_payload_shape(self):
        cfg = _config("backend.kind=ray", "backend.endpoint=http://localhost:8000/")
        backend = RayBackend(cfg)
        self.assertEqual(backend.base_url, "http://localhost:8000")
        request = GenerationRequest(
            prompt="hello", sampling=SamplingParams(max_tokens=16, stop=("END",))
        )
        payload = backend._payload(request)
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["max_tokens"], 16)
        self.assertEqual(payload["stop"], ["END"])
        self.assertEqual(payload["request_id"], request.request_id)


class TestEngineInspection(unittest.TestCase):
    def _engine_dir(self, config):
        path = tempfile.mkdtemp()
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh)
        open(os.path.join(path, "rank0.engine"), "wb").close()
        return path

    def test_reads_build_shape(self):
        path = self._engine_dir(
            {
                "build_config": {
                    "max_batch_size": 64,
                    "max_input_len": 4096,
                    "max_seq_len": 8192,
                    "plugin_config": {"paged_kv_cache": True},
                },
                "pretrained_config": {
                    "dtype": "bfloat16",
                    "quantization": {"quant_algo": "FP8"},
                    "mapping": {"tp_size": 2},
                },
            }
        )
        info = inspect_engine_dir(path)
        self.assertEqual(info["max_batch_size"], 64)
        self.assertEqual(info["max_seq_len"], 8192)
        self.assertEqual(info["quantization"], "FP8")
        self.assertEqual(info["tensor_parallel_size"], 2)
        self.assertEqual(info["engine_files"], ["rank0.engine"])

    def test_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            inspect_engine_dir("/definitely/not/here")

    def test_missing_config_mentions_the_build_script(self):
        path = tempfile.mkdtemp()
        with self.assertRaises(FileNotFoundError) as ctx:
            inspect_engine_dir(path)
        self.assertIn("export_trtllm.py", str(ctx.exception))


class TestExportPlan(unittest.TestCase):
    """The TensorRT-LLM build pipeline, verified without running any of it."""

    def setUp(self):
        import importlib.util

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "export_trtllm", os.path.join(here, "scripts", "export_trtllm.py")
        )
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def _args(self, *argv):
        return self.mod.build_parser().parse_args(list(argv))

    def test_plan_without_adapter_skips_merge(self):
        steps = self.mod.plan(self._args())
        self.assertEqual([label for label, _ in steps], ["convert checkpoint", "build engine"])

    def test_plan_with_adapter_merges_first(self):
        steps = self.mod.plan(self._args("--lora-adapter=/models/qlora"))
        self.assertEqual(steps[0][0], "merge QLoRA adapter")
        self.assertIn("/models/qlora", " ".join(steps[0][1]))

    def test_fp8_uses_the_modelopt_quantizer(self):
        command = " ".join(self.mod.convert_command(self._args("--quantization=fp8")))
        self.assertIn("quantize_by_modelopt", command)
        self.assertIn("--qformat=fp8", command)

    def test_int8_weight_only_uses_convert_checkpoint(self):
        command = " ".join(self.mod.convert_command(self._args("--quantization=int8_wo")))
        self.assertIn("convert_checkpoint.py", command)
        self.assertIn("--use_weight_only", command)

    def test_build_command_enables_paged_kv_and_batching(self):
        command = " ".join(self.mod.build_command(self._args("--max-batch-size=32")))
        self.assertIn("trtllm-build", command)
        self.assertIn("--max_batch_size=32", command)
        self.assertIn("--kv_cache_type=paged", command)
        self.assertIn("--remove_input_padding=enable", command)

    def test_single_stage_selection(self):
        steps = self.mod.plan(self._args("--stage=build"))
        self.assertEqual([label for label, _ in steps], ["build engine"])

    def test_dry_run_executes_nothing(self):
        self.assertEqual(self.mod.main(["--dry-run", "--engine-dir=/should/not/be/created"]), 0)
        self.assertFalse(os.path.exists("/should/not/be/created"))


if __name__ == "__main__":
    unittest.main()
