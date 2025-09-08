import json
import unittest

from modules.prompt_utils import render_prompt_with_schema


class TestRenderPromptWithSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.schema_obj = {"type": "object", "properties": {"a": {"type": "string"}}}
        self.schema_str = json.dumps(self.schema_obj, indent=2, ensure_ascii=False)

    def test_replace_token(self):
        prompt = """
You are a model.
Schema below:
{{TRANSCRIPTION_SCHEMA}}
"""
        out = render_prompt_with_schema(prompt, self.schema_obj)
        self.assertIn(self.schema_str, out)
        self.assertNotIn("{{TRANSCRIPTION_SCHEMA}}", out)

    def test_replace_after_marker_existing_block(self):
        prompt = """
System: do X.
The JSON schema:
{
  "type": "object",
  "properties": {
    "old": {"type": "string"}
  }
}
Some trailing text.
"""
        out = render_prompt_with_schema(prompt, self.schema_obj)
        # Should still include the marker
        self.assertIn("The JSON schema:", out)
        # Should include our new schema
        self.assertIn(self.schema_str, out)
        # Old key should be removed
        self.assertNotIn("\"old\"", out)
        # Trailing text should remain
        self.assertIn("Some trailing text.", out)

    def test_append_when_no_marker(self):
        prompt = "No schema marker here."
        out = render_prompt_with_schema(prompt, self.schema_obj)
        self.assertTrue(out.startswith(prompt))
        self.assertIn("The JSON schema:", out)
        self.assertIn(self.schema_str, out)


if __name__ == "__main__":
    unittest.main()
