"""Triton Python-backend model wrapping the vLLM engine.

Triton loads this file inside its own Python stub process, where
``triton_python_backend_utils`` exists as a builtin module. The model is declared
*decoupled* in ``config.pbtxt``, so ``execute`` returns ``None`` and instead
pushes many responses per request through the response sender — that is how
Triton expresses token-by-token streaming.

The engine runs on a background asyncio loop because Triton calls ``execute``
from its own threads.
"""

import asyncio
import json
import os
import threading

import numpy as np
import triton_python_backend_utils as pb_utils

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


class TritonPythonModel:
    def initialize(self, args):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self.model_config = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in self.model_config.get("parameters", {}).items()}

        base_model = params.get("model", os.environ.get("LLM_SERVE_MODEL", DEFAULT_MODEL))
        lora_adapter = params.get("lora_adapter") or os.environ.get("LLM_SERVE_LORA_ADAPTER")

        engine_args = AsyncEngineArgs(
            model=base_model,
            dtype=params.get("dtype", "bfloat16"),
            max_model_len=int(params.get("max_model_len", 8192)),
            tensor_parallel_size=int(params.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(params.get("gpu_memory_utilization", 0.9)),
            enable_prefix_caching=params.get("enable_prefix_caching", "true").lower() == "true",
            max_num_seqs=int(params.get("max_num_seqs", 256)),
            max_num_batched_tokens=int(params.get("max_num_batched_tokens", 8192)),
            enable_chunked_prefill=True,
            enable_lora=bool(lora_adapter),
            disable_log_requests=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.lora_request = None
        if lora_adapter:
            from vllm.lora.request import LoRARequest

            self.lora_request = LoRARequest("qlora-adapter", 1, lora_adapter)

        # Triton drives execute() from its own threads; run the engine on a
        # dedicated event loop so async generation is not tied to those threads.
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()
        self.logger = pb_utils.Logger
        self.logger.log_info(f"[llm_serve] vLLM engine ready: {base_model}")

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def execute(self, requests):
        for request in requests:
            asyncio.run_coroutine_threadsafe(self._generate(request), self.loop)
        return None  # decoupled mode: responses are sent asynchronously

    async def _generate(self, request):
        from vllm import SamplingParams

        sender = request.get_response_sender()
        try:
            prompt = _scalar_str(request, "text_input")
            raw_params = _scalar_str(request, "sampling_parameters", default="{}")
            options = json.loads(raw_params or "{}")
            stop = options.get("stop") or None
            sampling = SamplingParams(
                max_tokens=int(options.get("max_tokens", 128)),
                temperature=float(options.get("temperature", 0.7)),
                top_p=float(options.get("top_p", 1.0)),
                top_k=int(options.get("top_k", -1)),
                presence_penalty=float(options.get("presence_penalty", 0.0)),
                frequency_penalty=float(options.get("frequency_penalty", 0.0)),
                repetition_penalty=float(options.get("repetition_penalty", 1.0)),
                stop=stop,
                seed=options.get("seed"),
                ignore_eos=bool(options.get("ignore_eos", False)),
            )

            kwargs = {}
            if self.lora_request is not None:
                kwargs["lora_request"] = self.lora_request

            previous = ""
            request_id = request.request_id() or f"triton-{id(request)}"
            async for output in self.engine.generate(prompt, sampling, request_id, **kwargs):
                completion = output.outputs[0]
                delta = completion.text[len(previous):]
                previous = completion.text
                if delta:
                    sender.send(_text_response(delta))
        except Exception as exc:  # surface engine errors to the client
            sender.send(
                pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"generation failed: {exc}")
                )
            )
        finally:
            sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def finalize(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=10)


def _scalar_str(request, name, default=""):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return default
    value = tensor.as_numpy()[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _text_response(text):
    tensor = pb_utils.Tensor("text_output", np.array([text.encode("utf-8")], dtype=object))
    return pb_utils.InferenceResponse(output_tensors=[tensor])
