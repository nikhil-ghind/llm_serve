import os
import tempfile
import unittest

from llm_serve.config import (
    Config,
    ConfigError,
    build_config,
    env_overrides,
    parse_cli_overrides,
)

YAML = """
model:
  name: mistral-7b-qlora
  max_model_len: 4096
backend:
  kind: mock
  gpu_memory_utilization: 0.8
scheduler:
  block_size: 32
  max_num_seqs: 64
"""


class TestConfigLayers(unittest.TestCase):
    def _write(self, text):
        fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        fh.write(text)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_defaults_are_valid(self):
        cfg = Config().validate()
        self.assertEqual(cfg.backend.kind, "mock")
        self.assertEqual(cfg.scheduler.block_size, 16)

    def test_yaml_overrides_defaults(self):
        cfg = build_config(self._write(YAML), environ={})
        self.assertEqual(cfg.model.max_model_len, 4096)
        self.assertEqual(cfg.scheduler.block_size, 32)
        self.assertAlmostEqual(cfg.backend.gpu_memory_utilization, 0.8)
        # untouched keys keep their defaults
        self.assertTrue(cfg.metrics.enabled)

    def test_env_beats_yaml_and_cli_beats_env(self):
        path = self._write(YAML)
        env = {"LLM_SERVE_SCHEDULER__MAX_NUM_SEQS": "128", "PATH": "/ignored"}
        cfg = build_config(path, environ=env)
        self.assertEqual(cfg.scheduler.max_num_seqs, 128)
        cfg = build_config(path, environ=env, cli_overrides=["scheduler.max_num_seqs=256"])
        self.assertEqual(cfg.scheduler.max_num_seqs, 256)

    def test_bool_and_list_coercion(self):
        cfg = build_config(
            None,
            environ={},
            cli_overrides=[
                "backend.enable_prefix_caching=false",
                "metrics.ttft_buckets_s=0.1,0.5,1.0",
            ],
        )
        self.assertFalse(cfg.backend.enable_prefix_caching)
        self.assertEqual(cfg.metrics.ttft_buckets_s, [0.1, 0.5, 1.0])

    def test_env_parsing_ignores_unprefixed(self):
        out = env_overrides({"HOME": "/root", "LLM_SERVE_SERVER__PORT": "9000"})
        self.assertEqual(out, {"server": {"port": "9000"}})

    def test_cli_override_form_is_checked(self):
        with self.assertRaises(ConfigError):
            parse_cli_overrides(["nonsense"])
        with self.assertRaises(ConfigError):
            parse_cli_overrides(["nosection=1"])

    def test_unknown_section_and_option_rejected(self):
        with self.assertRaises(ConfigError):
            build_config(None, environ={}, cli_overrides=["nope.x=1"])
        with self.assertRaises(ConfigError):
            build_config(None, environ={}, cli_overrides=["server.nope=1"])


class TestConfigValidation(unittest.TestCase):
    def test_unknown_backend_rejected(self):
        with self.assertRaises(ConfigError):
            build_config(None, environ={}, cli_overrides=["backend.kind=tensorflow"])

    def test_remote_backends_require_endpoint(self):
        with self.assertRaises(ConfigError) as ctx:
            build_config(None, environ={}, cli_overrides=["backend.kind=triton"])
        self.assertIn("endpoint", str(ctx.exception))
        cfg = build_config(
            None,
            environ={},
            cli_overrides=["backend.kind=triton", "backend.endpoint=localhost:8001"],
        )
        self.assertEqual(cfg.backend.endpoint, "localhost:8001")

    def test_trtllm_requires_engine_dir(self):
        with self.assertRaises(ConfigError):
            build_config(None, environ={}, cli_overrides=["backend.kind=trtllm"])

    def test_block_size_must_be_power_of_two_choice(self):
        with self.assertRaises(ConfigError):
            build_config(None, environ={}, cli_overrides=["scheduler.block_size=17"])

    def test_gpu_memory_utilization_bounds(self):
        with self.assertRaises(ConfigError):
            build_config(None, environ={}, cli_overrides=["backend.gpu_memory_utilization=1.5"])

    def test_metric_buckets_must_be_sorted(self):
        with self.assertRaises(ConfigError):
            build_config(None, environ={}, cli_overrides=["metrics.itl_buckets_s=1.0,0.5"])

    def test_bench_shared_prefix_shorter_than_input(self):
        with self.assertRaises(ConfigError):
            build_config(
                None,
                environ={},
                cli_overrides=["bench.input_len=128", "bench.shared_prefix_len=256"],
            )

    def test_to_dict_roundtrips_sections(self):
        data = Config().to_dict()
        self.assertEqual(set(data), {"model", "backend", "scheduler", "server", "metrics", "bench"})
        self.assertEqual(data["backend"]["kind"], "mock")


if __name__ == "__main__":
    unittest.main()
