import unittest

from llm_serve.types import (
    FinishReason,
    GenerationRequest,
    ResultAccumulator,
    SamplingParams,
    TokenChunk,
    ValidationError,
)


class TestSamplingParams(unittest.TestCase):
    def test_defaults(self):
        p = SamplingParams()
        self.assertEqual(p.max_tokens, 128)
        self.assertEqual(p.stop, ())

    def test_rejects_bad_values(self):
        for kwargs in (
            {"max_tokens": 0},
            {"max_tokens": 1.5},
            {"temperature": -0.1},
            {"temperature": 2.5},
            {"top_p": 0.0},
            {"top_p": 1.5},
            {"top_k": 0},
            {"top_k": -3},
            {"presence_penalty": 3.0},
            {"frequency_penalty": -3.0},
            {"repetition_penalty": 0.0},
            {"n": 0},
            {"stop": tuple(f"s{i}" for i in range(9))},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValidationError):
                    SamplingParams(**kwargs)

    def test_from_dict_normalizes_stop_and_drops_unknown(self):
        p = SamplingParams.from_dict(
            {"max_tokens": 32, "stop": "END", "logit_bias": {"1": 2}, "temperature": None}
        )
        self.assertEqual(p.stop, ("END",))
        self.assertEqual(p.max_tokens, 32)
        self.assertEqual(p.temperature, 0.7)

    def test_from_dict_accepts_stop_list_and_none(self):
        self.assertEqual(SamplingParams.from_dict({"stop": ["a", "b"]}).stop, ("a", "b"))
        self.assertEqual(SamplingParams.from_dict({"stop": None}).stop, ())


class TestGenerationRequest(unittest.TestCase):
    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValidationError):
            GenerationRequest(prompt="")

    def test_prompt_len_prefers_token_ids(self):
        r = GenerationRequest(prompt="x" * 100, prompt_token_ids=[1, 2, 3])
        self.assertEqual(r.prompt_len, 3)

    def test_prompt_len_estimates_from_chars(self):
        self.assertEqual(GenerationRequest(prompt="a" * 40).prompt_len, 10)
        self.assertEqual(GenerationRequest(prompt="hi").prompt_len, 1)

    def test_request_ids_are_unique(self):
        ids = {GenerationRequest(prompt="hello").request_id for _ in range(50)}
        self.assertEqual(len(ids), 50)


class TestResultAccumulator(unittest.TestCase):
    def _run(self):
        req = GenerationRequest(prompt="a" * 40, sampling=SamplingParams(max_tokens=4))
        acc = ResultAccumulator(req, start_time=100.0)
        for i, ts in enumerate([100.5, 100.6, 100.75]):
            acc.add(TokenChunk(req.request_id, i, f"t{i}", timestamp=ts))
        acc.add(TokenChunk(req.request_id, 3, "", finish_reason=FinishReason.STOP))
        return acc.finish(end_time=101.0)

    def test_timings(self):
        res = self._run()
        self.assertEqual(res.text, "t0t1t2")
        self.assertEqual(res.completion_tokens, 3)
        self.assertEqual(res.prompt_tokens, 10)
        self.assertEqual(res.total_tokens, 13)
        self.assertAlmostEqual(res.ttft_s, 0.5)
        self.assertAlmostEqual(res.e2e_latency_s, 1.0)
        self.assertEqual(len(res.inter_token_latencies_s), 2)
        self.assertAlmostEqual(res.mean_itl_s, (0.1 + 0.15) / 2)
        self.assertAlmostEqual(res.output_tokens_per_s, 3.0)
        self.assertIs(res.finish_reason, FinishReason.STOP)

    def test_no_tokens_gives_no_ttft(self):
        req = GenerationRequest(prompt="hello")
        acc = ResultAccumulator(req, start_time=0.0)
        acc.add(TokenChunk(req.request_id, 0, "", finish_reason=FinishReason.ABORT))
        res = acc.finish(end_time=1.0)
        self.assertIsNone(res.ttft_s)
        self.assertIsNone(res.mean_itl_s)
        self.assertEqual(res.completion_tokens, 0)
        self.assertIs(res.finish_reason, FinishReason.ABORT)

    def test_to_dict_is_json_friendly(self):
        d = self._run().to_dict()
        self.assertEqual(d["finish_reason"], "stop")
        self.assertEqual(d["completion_tokens"], 3)

    def test_chunk_is_final(self):
        self.assertFalse(TokenChunk("r", 0, "hi").is_final)
        self.assertTrue(TokenChunk("r", 0, "", finish_reason=FinishReason.LENGTH).is_final)


if __name__ == "__main__":
    unittest.main()
