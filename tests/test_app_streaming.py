"""End-to-end streaming through the handler layer, without an HTTP server.

`llm_serve.api.app` keeps FastAPI inside `create_app`, so the request handlers
themselves can be driven directly on a CPU-only machine against the mock backend.
"""

import asyncio
import unittest

from llm_serve.api import openai_schemas as schemas
from llm_serve.api.app import ServingContext, run_completion, stream_completion
from llm_serve.api.chat_template import render_mistral_prompt
from llm_serve.api.sse import DONE, parse_sse_json, parse_sse_stream
from llm_serve.config import build_config


def _ctx():
    cfg = build_config(
        None,
        environ={},
        cli_overrides=["backend.mock_prefill_s=0.0", "backend.mock_decode_s=0.0"],
    )
    return ServingContext(cfg)


class TestStreamingHandler(unittest.TestCase):
    def _stream(self, payload, chat):
        ctx = _ctx()

        async def run():
            await ctx.startup()
            if chat:
                req = schemas.parse_chat_request(payload, "mistral-7b-qlora", render_mistral_prompt)
            else:
                req = schemas.parse_completion_request(payload, "mistral-7b-qlora")
            frames = [frame async for frame in stream_completion(ctx, req, chat)]
            await ctx.shutdown()
            return ctx, "".join(frames)

        return asyncio.run(run())

    def test_completion_stream_ends_with_done(self):
        _, raw = self._stream({"prompt": "hello", "max_tokens": 5, "stream": True}, chat=False)
        payloads = parse_sse_stream(raw)
        self.assertEqual(payloads[-1], DONE)
        objects = parse_sse_json(raw)
        self.assertEqual(len([o for o in objects if o["choices"][0]["text"]]), 5)
        self.assertEqual(objects[-1]["choices"][0]["finish_reason"], "length")

    def test_chat_stream_starts_with_role_delta(self):
        _, raw = self._stream(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "stream": True},
            chat=True,
        )
        objects = parse_sse_json(raw)
        self.assertEqual(objects[0]["choices"][0]["delta"], {"role": "assistant"})
        text = "".join(o["choices"][0]["delta"].get("content", "") for o in objects)
        self.assertTrue(text.strip())
        self.assertEqual(objects[-1]["choices"][0]["finish_reason"], "length")

    def test_streaming_records_metrics(self):
        ctx, _ = self._stream({"prompt": "hello", "max_tokens": 4, "stream": True}, chat=False)
        labels = {"backend": "mock", "model": "mistral-7b-qlora"}
        self.assertEqual(ctx.metrics.generation_tokens_total.value(labels), 4)
        self.assertEqual(ctx.metrics.requests_total.value({**labels, "status": "ok"}), 1)
        self.assertEqual(ctx.metrics.ttft_seconds.count(labels), 1)
        self.assertIn("llm_serve_time_to_first_token_seconds_bucket", ctx.metrics.render())

    def test_all_chunks_share_one_id_and_created(self):
        _, raw = self._stream({"prompt": "hello", "max_tokens": 4, "stream": True}, chat=False)
        objects = parse_sse_json(raw)
        self.assertEqual(len({o["id"] for o in objects}), 1)
        self.assertEqual(len({o["created"] for o in objects}), 1)


class TestNonStreamingHandler(unittest.TestCase):
    def test_full_response_body(self):
        ctx = _ctx()

        async def run():
            await ctx.startup()
            req = schemas.parse_completion_request({"prompt": "hello", "max_tokens": 6}, "m")
            result = await run_completion(ctx, req)
            await ctx.shutdown()
            return req, result

        req, result = asyncio.run(run())
        body = schemas.completion_response(req, result)
        self.assertEqual(body["usage"]["completion_tokens"], 6)
        self.assertTrue(body["choices"][0]["text"])
        self.assertEqual(body["choices"][0]["finish_reason"], "length")

    def test_concurrency_limit_is_respected(self):
        cfg = build_config(
            None,
            environ={},
            cli_overrides=[
                "backend.mock_prefill_s=0.0",
                "backend.mock_decode_s=0.0",
                "server.max_concurrent_requests=2",
            ],
        )
        ctx = ServingContext(cfg)
        peak = 0
        current = 0

        async def one():
            nonlocal peak, current
            async with ctx.semaphore:
                current += 1
                peak = max(peak, current)
                await asyncio.sleep(0)
                req = schemas.parse_completion_request({"prompt": "x", "max_tokens": 2}, "m")
                await run_completion(ctx, req)
                current -= 1

        async def run():
            await ctx.startup()
            await asyncio.gather(*[one() for _ in range(8)])
            await ctx.shutdown()

        asyncio.run(run())
        self.assertLessEqual(peak, 2)

    def test_refresh_gauges_reads_backend_stats(self):
        ctx = _ctx()

        async def run():
            await ctx.startup()
            req = schemas.parse_completion_request({"prompt": "x", "max_tokens": 2}, "m")
            await run_completion(ctx, req)
            await ctx.refresh_gauges()
            await ctx.shutdown()

        asyncio.run(run())
        labels = {"backend": "mock", "model": "mistral-7b-qlora"}
        self.assertEqual(ctx.metrics.running.value(labels), 0)


if __name__ == "__main__":
    unittest.main()
