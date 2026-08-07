import asyncio
import json
import unittest

from llm_serve.api import openai_schemas as schemas
from llm_serve.api.sse import (
    DONE,
    aiter_sse,
    comment,
    done_event,
    format_json_sse,
    format_sse,
    iter_sse,
    parse_sse_json,
    parse_sse_stream,
)
from llm_serve.types import FinishReason, GenerationRequest


class TestFraming(unittest.TestCase):
    def test_single_line_event(self):
        self.assertEqual(format_sse("hello"), "data: hello\n\n")

    def test_multiline_payload_gets_one_data_line_each(self):
        self.assertEqual(format_sse("a\nb"), "data: a\ndata: b\n\n")

    def test_event_and_retry_fields(self):
        frame = format_sse("x", event="ping", retry_ms=3000)
        self.assertTrue(frame.startswith("event: ping\nretry: 3000\ndata: x"))

    def test_json_frame_is_compact(self):
        frame = format_json_sse({"a": 1, "b": "two"})
        self.assertEqual(frame, 'data: {"a":1,"b":"two"}\n\n')
        self.assertNotIn(", ", frame)

    def test_done_sentinel(self):
        self.assertEqual(done_event(), "data: [DONE]\n\n")

    def test_comment_keepalive(self):
        self.assertEqual(comment("ping"), ": ping\n\n")

    def test_roundtrip_parsing(self):
        raw = format_json_sse({"n": 1}) + format_json_sse({"n": 2}) + done_event()
        self.assertEqual(parse_sse_stream(raw), ['{"n":1}', '{"n":2}', DONE])
        self.assertEqual(parse_sse_json(raw), [{"n": 1}, {"n": 2}])

    def test_iter_sse_appends_done(self):
        frames = list(iter_sse([{"a": 1}]))
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[-1], done_event())
        self.assertEqual(list(iter_sse([{"a": 1}], send_done=False)), frames[:1])

    def test_aiter_sse(self):
        async def gen():
            for i in range(3):
                yield {"i": i}

        async def run():
            return [frame async for frame in aiter_sse(gen())]

        frames = asyncio.run(run())
        self.assertEqual(len(frames), 4)
        self.assertEqual(parse_sse_json("".join(frames)), [{"i": 0}, {"i": 1}, {"i": 2}])


class TestStreamShape(unittest.TestCase):
    """The exact chunk objects an OpenAI client expects."""

    def setUp(self):
        self.req = GenerationRequest(prompt="hello", model="mistral-7b-qlora")
        self.req.request_id = "cmpl-abc"

    def test_completion_chunk_shape(self):
        payload = schemas.completion_chunk(self.req, "Hi", created=1700000000)
        self.assertEqual(payload["object"], "text_completion")
        self.assertEqual(payload["id"], "cmpl-abc")
        self.assertEqual(payload["created"], 1700000000)
        choice = payload["choices"][0]
        self.assertEqual(choice["text"], "Hi")
        self.assertIsNone(choice["finish_reason"])

    def test_final_chunk_carries_finish_reason(self):
        payload = schemas.completion_chunk(self.req, "", FinishReason.LENGTH)
        self.assertEqual(payload["choices"][0]["finish_reason"], "length")

    def test_chat_first_chunk_has_role_only(self):
        payload = schemas.chat_chunk(self.req, "", role="assistant")
        delta = payload["choices"][0]["delta"]
        self.assertEqual(delta, {"role": "assistant"})
        self.assertEqual(payload["object"], "chat.completion.chunk")

    def test_chat_content_chunk(self):
        delta = schemas.chat_chunk(self.req, "tok")["choices"][0]["delta"]
        self.assertEqual(delta, {"content": "tok"})

    def test_abort_maps_to_length(self):
        payload = schemas.chat_chunk(self.req, "", FinishReason.ABORT)
        self.assertEqual(payload["choices"][0]["finish_reason"], "length")

    def test_full_stream_is_valid_json_lines_then_done(self):
        frames = [
            format_json_sse(schemas.chat_chunk(self.req, "", role="assistant")),
            format_json_sse(schemas.chat_chunk(self.req, "Hel")),
            format_json_sse(schemas.chat_chunk(self.req, "lo")),
            format_json_sse(schemas.chat_chunk(self.req, "", FinishReason.STOP)),
            done_event(),
        ]
        raw = "".join(frames)
        payloads = parse_sse_stream(raw)
        self.assertEqual(payloads[-1], DONE)
        objects = [json.loads(p) for p in payloads[:-1]]
        text = "".join(o["choices"][0]["delta"].get("content", "") for o in objects)
        self.assertEqual(text, "Hello")
        self.assertEqual(objects[-1]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(all(o["id"] == "cmpl-abc" for o in objects))


if __name__ == "__main__":
    unittest.main()
