import asyncio
import unittest

from llm_serve.backends import available_backends, create_backend
from llm_serve.backends.base import BackendUnavailable, get_backend_class
from llm_serve.backends.mock import MockBackend
from llm_serve.config import build_config
from llm_serve.types import FinishReason, GenerationRequest, ResultAccumulator, SamplingParams


def _config(**overrides):
    cli = [f"{k.replace('_', '.', 1)}={v}" for k, v in overrides.items()]
    return build_config(None, environ={}, cli_overrides=cli)


async def _collect(backend, request):
    chunks = []
    async for chunk in backend.generate_stream(request):
        chunks.append(chunk)
    return chunks


class TestRegistry(unittest.TestCase):
    def test_all_backends_registered(self):
        self.assertEqual(
            available_backends(), ["mock", "ray", "triton", "trtllm", "vllm"]
        )

    def test_unknown_backend_raises_with_hint(self):
        with self.assertRaises(BackendUnavailable) as ctx:
            get_backend_class("keras")
        self.assertIn("available backends", str(ctx.exception))

    def test_create_backend_uses_config_kind(self):
        backend = create_backend(_config())
        self.assertIsInstance(backend, MockBackend)
        self.assertFalse(backend.started)


class TestMockBackend(unittest.TestCase):
    def setUp(self):
        cfg = _config(backend_mock_prefill_s=0.0, backend_mock_decode_s=0.0)
        self.backend = MockBackend(cfg)

    def test_streams_requested_number_of_tokens(self):
        req = GenerationRequest(prompt="hello world", sampling=SamplingParams(max_tokens=5))
        chunks = asyncio.run(_collect(self.backend, req))
        text_chunks = [c for c in chunks if c.text]
        self.assertEqual(len(text_chunks), 5)
        self.assertTrue(chunks[-1].is_final)
        self.assertIs(chunks[-1].finish_reason, FinishReason.LENGTH)

    def test_output_is_deterministic_for_same_prompt(self):
        req_a = GenerationRequest(prompt="same prompt", sampling=SamplingParams(max_tokens=8))
        req_b = GenerationRequest(prompt="same prompt", sampling=SamplingParams(max_tokens=8))
        a = "".join(c.text for c in asyncio.run(_collect(self.backend, req_a)))
        b = "".join(c.text for c in asyncio.run(_collect(self.backend, req_b)))
        self.assertEqual(a, b)
        self.assertTrue(a)

    def test_different_prompts_differ(self):
        a = "".join(
            c.text
            for c in asyncio.run(
                _collect(self.backend, GenerationRequest(prompt="alpha", sampling=SamplingParams(max_tokens=20)))
            )
        )
        b = "".join(
            c.text
            for c in asyncio.run(
                _collect(self.backend, GenerationRequest(prompt="beta", sampling=SamplingParams(max_tokens=20)))
            )
        )
        self.assertNotEqual(a, b)

    def test_seed_pins_output(self):
        reqs = [
            GenerationRequest(prompt=p, sampling=SamplingParams(max_tokens=6, seed=7))
            for p in ("one", "two")
        ]
        outs = ["".join(c.text for c in asyncio.run(_collect(self.backend, r))) for r in reqs]
        self.assertEqual(outs[0], outs[1])

    def test_abort_terminates_stream(self):
        async def run():
            req = GenerationRequest(prompt="abort me", sampling=SamplingParams(max_tokens=50))
            got = []
            async for chunk in self.backend.generate_stream(req):
                got.append(chunk)
                if len(got) == 3:
                    await self.backend.abort(req.request_id)
            return got

        chunks = asyncio.run(run())
        self.assertLess(len(chunks), 50)
        self.assertIs(chunks[-1].finish_reason, FinishReason.ABORT)

    def test_stop_sequence_ends_generation_early(self):
        async def run():
            req = GenerationRequest(prompt="stopper", sampling=SamplingParams(max_tokens=40))
            first = []
            async for chunk in self.backend.generate_stream(req):
                if chunk.text:
                    first.append(chunk.text.strip())
            stop_word = first[3]
            req2 = GenerationRequest(
                prompt="stopper", sampling=SamplingParams(max_tokens=40, stop=(stop_word,))
            )
            return await _collect(self.backend, req2)

        chunks = asyncio.run(run())
        self.assertIs(chunks[-1].finish_reason, FinishReason.STOP)
        self.assertLess(len([c for c in chunks if c.text]), 40)

    def test_stats_and_health(self):
        async def run():
            async with MockBackend(self.backend.config) as backend:
                health = await backend.health()
                await _collect(
                    backend,
                    GenerationRequest(prompt="a" * 40, sampling=SamplingParams(max_tokens=4)),
                )
                return health, await backend.stats()

        health, stats = asyncio.run(run())
        self.assertTrue(health.ok)
        self.assertEqual(health.backend, "mock")
        self.assertEqual(stats.total_generation_tokens, 4)
        self.assertEqual(stats.total_prompt_tokens, 10)
        self.assertEqual(stats.extra["completed_requests"], 1.0)
        self.assertEqual(stats.running, 0)

    def test_accumulator_over_mock_stream(self):
        async def run():
            req = GenerationRequest(prompt="accumulate", sampling=SamplingParams(max_tokens=6))
            acc = ResultAccumulator(req)
            async for chunk in self.backend.generate_stream(req):
                acc.add(chunk)
            return acc.finish()

        result = asyncio.run(run())
        self.assertEqual(result.completion_tokens, 6)
        self.assertIsNotNone(result.ttft_s)
        self.assertEqual(len(result.inter_token_latencies_s), 5)

    def test_concurrent_requests_are_isolated(self):
        async def run():
            reqs = [
                GenerationRequest(prompt=f"p{i}", sampling=SamplingParams(max_tokens=5))
                for i in range(8)
            ]
            results = await asyncio.gather(*[_collect(self.backend, r) for r in reqs])
            return reqs, results

        reqs, results = asyncio.run(run())
        for req, chunks in zip(reqs, results):
            self.assertTrue(all(c.request_id == req.request_id for c in chunks))
            self.assertEqual(len([c for c in chunks if c.text]), 5)


if __name__ == "__main__":
    unittest.main()
