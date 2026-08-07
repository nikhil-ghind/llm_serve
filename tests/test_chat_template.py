import unittest

from llm_serve.api.chat_template import (
    TemplateError,
    estimate_tokens,
    render_completion_prompt,
    render_mistral_prompt,
)


class TestMistralTemplate(unittest.TestCase):
    def test_single_user_turn(self):
        out = render_mistral_prompt([{"role": "user", "content": "Hello"}])
        self.assertEqual(out, "<s>[INST] Hello [/INST]")

    def test_multi_turn_conversation(self):
        out = render_mistral_prompt(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ]
        )
        self.assertEqual(out, "<s>[INST] Hi [/INST] Hello!</s>[INST] How are you? [/INST]")

    def test_system_prompt_folds_into_first_user_turn(self):
        out = render_mistral_prompt(
            [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Hi"},
            ]
        )
        self.assertEqual(out, "<s>[INST] You are terse.\n\nHi [/INST]")
        self.assertNotIn("system", out)

    def test_multiple_system_messages_are_joined(self):
        out = render_mistral_prompt(
            [
                {"role": "system", "content": "A."},
                {"role": "system", "content": "B."},
                {"role": "user", "content": "Hi"},
            ]
        )
        self.assertIn("A.\nB.\n\nHi", out)

    def test_bos_can_be_omitted(self):
        out = render_mistral_prompt([{"role": "user", "content": "Hi"}], add_bos=False)
        self.assertTrue(out.startswith("[INST]"))

    def test_content_is_stripped(self):
        out = render_mistral_prompt([{"role": "user", "content": "  Hi  "}])
        self.assertEqual(out, "<s>[INST] Hi [/INST]")

    def test_empty_messages_rejected(self):
        with self.assertRaises(TemplateError):
            render_mistral_prompt([])

    def test_system_only_rejected(self):
        with self.assertRaises(TemplateError):
            render_mistral_prompt([{"role": "system", "content": "x"}])

    def test_non_alternating_roles_rejected(self):
        with self.assertRaises(TemplateError):
            render_mistral_prompt(
                [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
            )

    def test_trailing_assistant_message_rejected(self):
        with self.assertRaises(TemplateError):
            render_mistral_prompt(
                [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
            )

    def test_assistant_first_rejected(self):
        with self.assertRaises(TemplateError):
            render_mistral_prompt([{"role": "assistant", "content": "a"}])

    def test_tool_role_rejected(self):
        with self.assertRaises(TemplateError):
            render_mistral_prompt(
                [{"role": "user", "content": "a"}, {"role": "tool", "content": "b"}]
            )

    def test_system_must_precede_a_user_message(self):
        with self.assertRaises(TemplateError):
            render_mistral_prompt(
                [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]
            )


class TestCompletionPassthrough(unittest.TestCase):
    def test_prompt_is_untouched(self):
        raw = "Q: 2+2?\nA:"
        self.assertEqual(render_completion_prompt(raw), raw)
        self.assertEqual(render_completion_prompt(raw, lora_adapter="qlora"), raw)

    def test_token_estimate(self):
        self.assertEqual(estimate_tokens(""), 1)
        self.assertEqual(estimate_tokens("a" * 40), 10)


if __name__ == "__main__":
    unittest.main()
