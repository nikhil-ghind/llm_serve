# LLM Inference Serving — Implementation Plan

Serve a fine-tuned **Mistral 7B (QLoRA)** model behind **vLLM**, **NVIDIA Triton**,
**Ray Serve**, and **TensorRT-LLM**, exposing a single OpenAI-compatible completions
API with SSE token streaming. Benchmark throughput (tokens/s, req/s), time-to-first-token
(TTFT), inter-token latency (ITL), and GPU utilization across every backend, then publish
a cross-backend comparison report.

## Design constraints

The engine-facing code targets an NVIDIA GPU host (A100/L4/4090-class, CUDA 12.x), but the
repository must remain **import-safe and unit-testable on a CPU-only machine with no model
weights present**. That drives three rules applied everywhere:

1. **Backend abstraction.** Every engine sits behind one `Backend` protocol
   (`generate_stream` / `health` / `stats`). Nothing above the backend layer knows what
   engine is running.
2. **Lazy heavy imports.** `vllm`, `tensorrt_llm`, `ray`, `torch`, `triton_python_backend_utils`
   and `pynvml` are imported *inside* methods, never at module import time. Importing any
   module in this repo on a laptop must succeed.
3. **Pure-Python core.** The scheduler, KV block manager, metrics math, benchmark
   aggregation, SSE framing and config parsing carry no GPU dependency and are covered by
   stdlib `unittest` tests that run in milliseconds on CPU. A `mock` backend implements the
   same protocol with a deterministic token generator so the whole server, load generator
   and report pipeline can be exercised end to end without a GPU.

---

## Phase 1 — Core abstractions, config, and the mock backend

Foundation everything else is built on: typed request/response objects, YAML-driven config,
the backend protocol plus registry, and a CPU-only mock engine.

**Deliverables**
- Sampling/request/response dataclasses (`CompletionRequest`, `TokenChunk`, `GenerationResult`)
  with validation of `temperature`, `top_p`, `max_tokens`, `stop`, `n`.
- Layered config loader: YAML file → environment overrides → CLI overrides, with typed
  sections (`model`, `backend`, `scheduler`, `server`, `metrics`, `bench`) and clear errors.
- `Backend` protocol + a name→factory registry so `--backend vllm|triton|ray|trtllm|mock`
  selects an implementation without importing the others.
- `MockBackend`: deterministic pseudo-token stream with configurable prefill/decode delays,
  used by every test and by the smoke path of the load generator.

**Files**
`llm_serve/__init__.py`, `llm_serve/types.py`, `llm_serve/config.py`,
`llm_serve/backends/__init__.py`, `llm_serve/backends/base.py`,
`llm_serve/backends/mock.py`, `configs/serving.yaml`, `requirements.txt`,
`tests/test_config.py`, `tests/test_types.py`, `tests/test_mock_backend.py`

---

## Phase 2 — Continuous batching scheduler and KV cache reuse

The heart of the serving story, written as a pure-Python simulation of what vLLM does
internally so the policy is inspectable and testable. Used directly by the mock backend and
by the Triton/Ray fronting layers where the engine does not do its own admission control.

**Deliverables**
- Paged KV block manager: fixed-size blocks, allocation/free, watermark-based admission,
  reference counting for shared blocks.
- **Prefix-based KV cache reuse**: rolling hash over block-aligned prompt token spans,
  hash→block table, `cache_hit_blocks` accounting and hit-rate reporting.
- Continuous-batching scheduler: `waiting`/`running`/`swapped` queues, per-step token budget,
  `max_num_seqs` / `max_num_batched_tokens` caps, chunked prefill, FCFS with preemption
  (recompute policy) when blocks run out; each `step()` returns the admitted batch.
- Scheduler stats: running/waiting depth, KV utilization, preemption count, batch-size
  histogram.

**Files**
`llm_serve/engine/__init__.py`, `llm_serve/engine/block_manager.py`,
`llm_serve/engine/prefix_cache.py`, `llm_serve/engine/scheduler.py`,
`llm_serve/engine/stats.py`, `tests/test_block_manager.py`,
`tests/test_prefix_cache.py`, `tests/test_scheduler.py`

---

## Phase 3 — OpenAI-compatible API server, SSE streaming, Prometheus metrics

One FastAPI app that fronts whichever backend is configured, speaking the OpenAI wire
protocol so existing clients work unchanged.

**Deliverables**
- `POST /v1/completions`, `POST /v1/chat/completions` (streaming and non-streaming),
  `GET /v1/models`, `GET /health`, `GET /metrics`.
- SSE framing helpers producing exact OpenAI chunk objects and the terminating `data: [DONE]`,
  including `finish_reason` on the last chunk and usage accounting — pure functions, fully
  unit-tested without a running server.
- Prometheus metrics: TTFT and ITL histograms, end-to-end latency, prompt/generation token
  counters, running/waiting gauges, KV-cache utilization and prefix-hit-rate gauges. Metric
  math (bucketing, quantile estimation, rate) lives in a dependency-free module.
- Mistral chat-template rendering (`[INST] ... [/INST]`) for the chat endpoint, plus the
  QLoRA adapter path wired through config.

**Files**
`llm_serve/api/__init__.py`, `llm_serve/api/app.py`, `llm_serve/api/openai_schemas.py`,
`llm_serve/api/sse.py`, `llm_serve/api/chat_template.py`, `llm_serve/metrics/__init__.py`,
`llm_serve/metrics/registry.py`, `llm_serve/metrics/math.py`, `llm_serve/server.py`,
`tests/test_sse.py`, `tests/test_openai_schemas.py`, `tests/test_metrics.py`,
`tests/test_chat_template.py`

---

## Phase 4 — Real backends: vLLM, Triton, Ray Serve, TensorRT-LLM

Four adapters against the Phase 1 protocol. All GPU imports are lazy; each module imports
cleanly on CPU and raises a descriptive `BackendUnavailable` only when actually started.

**Deliverables**
- `VLLMBackend`: `AsyncLLMEngine` with continuous batching, `enable_prefix_caching`,
  `gpu_memory_utilization`, tensor parallelism, LoRA adapter loading for the QLoRA weights,
  streaming deltas mapped onto `TokenChunk`.
- Triton deployment: Python-backend `model.py` wrapping the same engine with decoupled
  (streaming) responses, `config.pbtxt` with dynamic batching, and a `TritonBackend` client
  over gRPC/HTTP streaming.
- Ray Serve deployment: autoscaling `@serve.deployment` fronting the engine, plus a
  `RayBackend` client; replica/ongoing-request config surfaced through YAML.
- TensorRT-LLM export path: quantization + `trtllm-build` invocation script, engine layout
  documentation, and a `TensorRTLLMBackend` reading the built engine.
- Dockerfiles / compose for the Triton and Ray images.

**Files**
`llm_serve/backends/vllm_backend.py`, `llm_serve/backends/triton_backend.py`,
`llm_serve/backends/ray_backend.py`, `llm_serve/backends/trtllm_backend.py`,
`deploy/triton/model_repository/mistral7b/1/model.py`,
`deploy/triton/model_repository/mistral7b/config.pbtxt`,
`deploy/ray/serve_app.py`, `scripts/export_trtllm.py`, `deploy/Dockerfile.triton`,
`deploy/Dockerfile.ray`, `deploy/docker-compose.yml`,
`tests/test_backend_registry.py`, `tests/test_backend_imports.py`

---

## Phase 5 — Load generator, benchmark aggregation, comparison report

Turn the serving stack into measured numbers and a report that ranks the backends.

**Deliverables**
- Async load generator: configurable concurrency, request rate (Poisson or closed-loop),
  prompt-length and output-length distributions, warmup, fixed duration or fixed request
  count; records per-request TTFT, per-token ITL, e2e latency, token counts.
- Aggregation module: p50/p90/p95/p99 percentiles, output/total tokens per second,
  requests per second, goodput under an SLO, all pure functions over recorded samples.
- GPU utilization sampler: background NVML poller (lazy `pynvml`, no-op when absent)
  producing mean/max SM utilization and memory-used timelines aligned to the run window.
- Result store (JSON per run) and a Markdown/CSV **cross-backend comparison report**
  ranking vLLM / Triton / Ray / TensorRT-LLM on throughput, TTFT, ITL and $/1M tokens,
  with a sweep driver that runs a concurrency ladder per backend.
- Prometheus/Grafana scrape config for live dashboards.

**Files**
`llm_serve/bench/__init__.py`, `llm_serve/bench/loadgen.py`, `llm_serve/bench/workload.py`,
`llm_serve/bench/aggregate.py`, `llm_serve/bench/gpu_monitor.py`, `llm_serve/bench/report.py`,
`scripts/run_benchmark.py`, `scripts/compare_backends.py`, `configs/bench.yaml`,
`deploy/prometheus.yml`, `tests/test_aggregate.py`, `tests/test_workload.py`,
`tests/test_report.py`

---

## Verification strategy

No GPU is available in the development environment, so verification is:

- `python -m compileall llm_serve scripts deploy` — every module parses.
- `python -m unittest discover -s tests -v` — CPU-only unit tests over the scheduler, block
  manager, prefix cache, metrics math, SSE framing, config parsing, workload generation and
  benchmark aggregation.
- An import-safety test that imports every backend module and asserts none of them pull in
  `vllm` / `tensorrt_llm` / `ray` / `torch` at import time.
- End-to-end smoke through the `mock` backend: load generator → aggregation → report,
  producing a real comparison report without touching a GPU.

Model downloads, engine builds and live server runs are documented in the README as GPU-host
steps and are deliberately not executed here.
