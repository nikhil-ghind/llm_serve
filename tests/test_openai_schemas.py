import unittest

from llm_serve.api import openai_schemas as schemas
from llm_serve.api.chat_template import render_mistral_prompt
from llm_serve.types import FinishReason, GenerationRequest, GenerationResult, ValidationError


class TestParseCompletion(unittest.TestCase):
    def test_minimal_request(self):
        req = schemas.parse_completion_request({"prompt": "hello"}, "mistral-7b-qlora")
        self.assertEqual(req.prompt, "hello")
        self.assertEqual(req.model, "mistral-7b-qlora")
        self.assertFalse(req.stream)
        self.assertTrue(req.request_id.startswith("cmpl-"))

    def test_sampling_params_are_carried(self):
        req = schemas.parse_completion_request(
            {"prompt": "x", "max_tokens": 64, "temperature": 0.2, "stop": ["\n\n"], "stream": True},
            "m",
        )
        self.assertEqual(req.sampling.max_tokens, 64)
        self.assertAlmostEqual(req.sampling.temperature, 0.2)
        self.assertEqual(req.sampling.stop, ("\n\n",))
        self.assertTrue(req.stream)

    def test_model_override(self):
        req = schemas.parse_completion_request({"prompt": "x", "model": "other"}, "default")
        self.assertEqual(req.model, "other")

    def test_single_element_prompt_list_allowed(self):
        req = schemas.parse_completion_request({"prompt": ["only"]}, "m")
        self.assertEqual(req.prompt, "only")

    def test_rejections(self):
        for payload in (
            {},
            {"prompt": None},
            {"prompt": 5},
            {"prompt": []},
            {"prompt": ["a", "b"]},
            {"prompt": [[1, 2, 3]]},
            {"prompt": "x", "n": 2},
            {"prompt": "x", "temperature": 9},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    schemas.parse_completion_request(payload, "m")

    def test_non_object_body(self):
        with self.assertRaises(ValidationError):
            schemas.parse_completion_request(["not", "an", "object"], "m")


class TestParseChat(unittest.TestCase):
    def _parse(self, payload):
        return schemas.parse_chat_request(payload, "m", render_mistral_prompt)

    def test_renders_prompt_through_template(self):
        req = self._parse({"messages": [{"role": "user", "content": "Hi"}]})
        self.assertIn("[INST] Hi [/INST]", req.prompt)
        self.assertTrue(req.request_id.startswith("chatcmpl-"))

    def test_system_prompt_is_folded_in(self):
        req = self._parse(
            {
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "Hi"},
                ]
            }
        )
        self.assertIn("Be terse.", req.prompt)

    def test_rejections(self):
        for payload in (
            {},
            {"messages": []},
            {"messages": "hello"},
            {"messages": ["not an object"]},
            {"messages": [{"role": "wizard", "content": "x"}]},
            {"messages": [{"role": "user", "content": 42}]},
            {"messages": [{"role": "user", "content": "x"}], "n": 3},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    self._parse(payload)


class TestResponses(unittest.TestCase):
    def setUp(self):
        self.req = GenerationRequest(prompt="hello", model="mistral-7b-qlora")
        self.req.request_id = "cmpl-1"
        self.result = GenerationResult(
            request_id="cmpl-1",
            text="world",
            prompt_tokens=5,
            completion_tokens=2,
            finish_reason=FinishReason.STOP,
        )

    def test_completion_response(self):
        body = schemas.completion_response(self.req, self.result, created=1)
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["choices"][0]["text"], "world")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"], {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})

    def test_chat_completion_response(self):
        body = schemas.chat_completion_response(self.req, self.result, created=1)
        self.assertEqual(body["object"], "chat.completion")
        message = body["choices"][0]["message"]
        self.assertEqual(message, {"role": "assistant", "content": "world"})

    def test_models_response(self):
        body = schemas.models_response(["mistral-7b-qlora"])
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["data"][0]["id"], "mistral-7b-qlora")
        self.assertEqual(body["data"][0]["object"], "model")

    def test_error_response(self):
        body = schemas.error_response("bad prompt", code="prompt_invalid")
        self.assertEqual(body["error"]["message"], "bad prompt")
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["error"]["code"], "prompt_invalid")

    def test_request_ids_are_unique(self):
        ids = {schemas.new_request_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)


if __name__ == "__main__":
    unittest.main()
