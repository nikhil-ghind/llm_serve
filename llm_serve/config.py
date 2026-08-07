"""Layered configuration: YAML file -> environment overrides -> CLI overrides.

Kept to the stdlib plus PyYAML. If PyYAML is unavailable a small fallback parser
handles the flat/nested scalar YAML this project ships, so config loading never
becomes a hard dependency for tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

ENV_PREFIX = "LLM_SERVE_"
KNOWN_BACKENDS = ("mock", "vllm", "triton", "ray", "trtllm")


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed or contradictory."""


@dataclass
class ModelConfig:
    name: str = "mistral-7b-qlora"
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    lora_adapter: str | None = None
    tokenizer: str | None = None
    max_model_len: int = 8192
    dtype: str = "bfloat16"
    quantization: str | None = None
    trust_remote_code: bool = False

    def validate(self) -> None:
        if self.max_model_len < 128:
            raise ConfigError("model.max_model_len must be >= 128")
        if self.dtype not in ("auto", "float16", "bfloat16", "float32"):
            raise ConfigError(f"model.dtype {self.dtype!r} is not supported")
        if self.quantization not in (None, "awq", "gptq", "fp8", "bitsandbytes"):
            raise ConfigError(f"model.quantization {self.quantization!r} is not supported")

    @property
    def tokenizer_id(self) -> str:
        return self.tokenizer or self.base_model


@dataclass
class BackendConfig:
    kind: str = "mock"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    swap_space_gb: int = 4
    endpoint: str | None = None
    engine_dir: str | None = None
    mock_prefill_s: float = 0.05
    mock_decode_s: float = 0.01

    def validate(self) -> None:
        if self.kind not in KNOWN_BACKENDS:
            raise ConfigError(
                f"backend.kind {self.kind!r} unknown; expected one of {', '.join(KNOWN_BACKENDS)}"
            )
        if self.tensor_parallel_size < 1:
            raise ConfigError("backend.tensor_parallel_size must be >= 1")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ConfigError("backend.gpu_memory_utilization must be in (0, 1]")
        if self.kind in ("triton", "ray") and not self.endpoint:
            raise ConfigError(f"backend.endpoint is required for the {self.kind} backend")
        if self.kind == "trtllm" and not self.engine_dir:
            raise ConfigError("backend.engine_dir is required for the trtllm backend")


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    block_size: int = 16
    num_gpu_blocks: int = 4096
    watermark: float = 0.01
    enable_chunked_prefill: bool = True
    max_waiting: int = 2048
    preemption_mode: str = "recompute"

    def validate(self) -> None:
        if self.block_size not in (8, 16, 32, 64, 128):
            raise ConfigError("scheduler.block_size must be one of 8, 16, 32, 64, 128")
        if self.max_num_seqs < 1:
            raise ConfigError("scheduler.max_num_seqs must be >= 1")
        if self.max_num_batched_tokens < self.block_size:
            raise ConfigError(
                "scheduler.max_num_batched_tokens must be at least one block worth of tokens"
            )
        if not 0.0 <= self.watermark < 1.0:
            raise ConfigError("scheduler.watermark must be in [0, 1)")
        if self.preemption_mode not in ("recompute", "swap"):
            raise ConfigError("scheduler.preemption_mode must be 'recompute' or 'swap'")


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    max_concurrent_requests: int = 512
    request_timeout_s: float = 300.0
    served_model_name: str = "mistral-7b-qlora"

    def validate(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ConfigError(f"server.port {self.port} is out of range")
        if self.request_timeout_s <= 0:
            raise ConfigError("server.request_timeout_s must be > 0")


@dataclass
class MetricsConfig:
    enabled: bool = True
    namespace: str = "llm_serve"
    ttft_buckets_s: list[float] = field(
        default_factory=lambda: [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    itl_buckets_s: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    )
    e2e_buckets_s: list[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    )

    def validate(self) -> None:
        for name in ("ttft_buckets_s", "itl_buckets_s", "e2e_buckets_s"):
            buckets = getattr(self, name)
            if not buckets:
                raise ConfigError(f"metrics.{name} must not be empty")
            if list(buckets) != sorted(buckets):
                raise ConfigError(f"metrics.{name} must be sorted ascending")


@dataclass
class BenchConfig:
    concurrency: int = 16
    num_requests: int = 200
    duration_s: float | None = None
    request_rate: float | None = None
    warmup_requests: int = 8
    input_len: int = 512
    input_len_std: int = 64
    output_len: int = 128
    output_len_std: int = 16
    shared_prefix_len: int = 0
    seed: int = 42
    slo_ttft_s: float = 1.0
    slo_itl_s: float = 0.05
    gpu_sample_interval_s: float = 0.5

    def validate(self) -> None:
        if self.concurrency < 1:
            raise ConfigError("bench.concurrency must be >= 1")
        if self.duration_s is None and self.num_requests < 1:
            raise ConfigError("bench.num_requests must be >= 1 when no duration is set")
        if self.request_rate is not None and self.request_rate <= 0:
            raise ConfigError("bench.request_rate must be > 0 when set")
        if self.input_len < 1 or self.output_len < 1:
            raise ConfigError("bench.input_len and bench.output_len must be >= 1")
        if self.shared_prefix_len >= self.input_len:
            raise ConfigError("bench.shared_prefix_len must be shorter than bench.input_len")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)

    def validate(self) -> "Config":
        for f in fields(self):
            getattr(self, f.name).validate()
        if self.scheduler.max_num_batched_tokens > self.model.max_model_len * self.scheduler.max_num_seqs:
            raise ConfigError(
                "scheduler.max_num_batched_tokens exceeds what max_num_seqs sequences can hold"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _dataclass_to_dict(getattr(self, f.name)) for f in fields(self)}


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def _coerce(value: Any, target: Any) -> Any:
    """Coerce a scalar from YAML/env into the type of the existing default."""
    if value is None or target is None:
        return value
    if isinstance(target, bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(target, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(target, float):
        return float(value)
    if isinstance(target, list):
        if isinstance(value, str):
            return [float(v) for v in value.split(",") if v.strip()]
        return list(value)
    if isinstance(target, str):
        return str(value)
    return value


def _apply_section(section: Any, overrides: dict[str, Any]) -> None:
    valid = {f.name for f in fields(section)}
    for key, value in overrides.items():
        if key not in valid:
            raise ConfigError(f"unknown option {key!r} in section {type(section).__name__}")
        setattr(section, key, _coerce(value, getattr(section, key)))


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal two-level YAML reader used when PyYAML is not installed."""
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        val = val.strip()
        if indent == 0:
            if val:
                root[key] = _scalar(val)
                current = None
            else:
                current = {}
                root[key] = current
        else:
            if current is None:
                raise ConfigError(f"unexpected indented line: {raw!r}")
            current[key] = _scalar(val)
    return root


def _scalar(text: str) -> Any:
    if text in ("null", "~", ""):
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_scalar(p.strip()) for p in inner.split(",")] if inner else []
    if text.startswith(("'", '"')) and text[-1] == text[0]:
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # noqa: PLC0415 - optional dependency
    except ImportError:
        return _parse_simple_yaml(text)
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return data


def env_overrides(environ: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Read ``LLM_SERVE_<SECTION>__<OPTION>`` variables into nested overrides."""
    environ = os.environ if environ is None else environ
    out: dict[str, dict[str, Any]] = {}
    for key, value in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        remainder = key[len(ENV_PREFIX) :].lower()
        if "__" not in remainder:
            continue
        section, option = remainder.split("__", 1)
        out.setdefault(section, {})[option] = value
    return out


def parse_cli_overrides(pairs: list[str]) -> dict[str, dict[str, Any]]:
    """Parse ``--set section.option=value`` style strings."""
    out: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"override {pair!r} must be of the form section.option=value")
        dotted, _, value = pair.partition("=")
        if "." not in dotted:
            raise ConfigError(f"override key {dotted!r} must be of the form section.option")
        section, option = dotted.split(".", 1)
        out.setdefault(section.strip(), {})[option.strip()] = value.strip()
    return out


def build_config(
    path: str | None = None,
    *,
    cli_overrides: list[str] | None = None,
    environ: dict[str, str] | None = None,
) -> Config:
    """Compose the effective config: defaults < YAML < environment < CLI."""
    cfg = Config()
    layers: list[dict[str, dict[str, Any]]] = []
    if path:
        raw = load_yaml(path)
        layers.append({k: v for k, v in raw.items() if isinstance(v, dict)})
        stray = [k for k, v in raw.items() if not isinstance(v, dict)]
        if stray:
            raise ConfigError(f"top-level scalar keys are not allowed: {', '.join(sorted(stray))}")
    layers.append(env_overrides(environ))
    if cli_overrides:
        layers.append(parse_cli_overrides(cli_overrides))

    valid_sections = {f.name for f in fields(cfg)}
    for layer in layers:
        for section, values in layer.items():
            if section not in valid_sections:
                raise ConfigError(
                    f"unknown config section {section!r}; expected one of "
                    f"{', '.join(sorted(valid_sections))}"
                )
            _apply_section(getattr(cfg, section), values)
    return cfg.validate()
