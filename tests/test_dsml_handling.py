import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llm_service import parse_dsml_tool_calls, strip_dsml_tags, resolve_tool_name


class TestDSMLHandling(unittest.TestCase):
    def test_parses_dsml_tool_calls(self):
        text = """I have the article.
< | DSML | tool_calls>
< | DSML | invoke name="fetch_page">
< | DSML | parameter name="url" string="true">https://en.wikipedia.org/wiki/T-54/T-55</ | DSML | parameter>
</ | DSML | invoke>
< | DSML | invoke name="fetch_page">
< | DSML | parameter name="url" string="true">https://en.wikipedia.org/wiki/T-54/T-55_operators_and_variants</ | DSML | parameter>
</ | DSML | invoke>
</ | DSML | tool_calls>"""
        calls = parse_dsml_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["function"]["name"], "fetch_page")
        self.assertEqual(calls[0]["function"]["arguments"], '{"url": "https://en.wikipedia.org/wiki/T-54/T-55"}')
        self.assertEqual(calls[1]["function"]["name"], "fetch_page")
        self.assertEqual(calls[1]["function"]["arguments"], '{"url": "https://en.wikipedia.org/wiki/T-54/T-55_operators_and_variants"}')

    def test_strips_dsml_tags_completely(self):
        text = """I have the operators/variants article but it got truncated mid-content.

< | DSML | tool_calls>
< | DSML | invoke name="fetch_page">
< | DSML | parameter name="url" string="true">https://en.wikipedia.org/wiki/T-54/T-55</ | DSML | parameter>
</ | DSML | invoke>
</ | DSML | tool_calls>"""
        stripped = strip_dsml_tags(text)
        self.assertNotIn("DSML", stripped)
        self.assertNotIn("tool_calls", stripped)
        self.assertNotIn("fetch_page", stripped)
        self.assertEqual(stripped, "I have the operators/variants article but it got truncated mid-content.")

    def test_resolve_tool_name_aliases(self):
        available = {"wiki__search_wikipedia", "wiki__get_article", "fetch__fetch", "parallel__web_fetch"}
        
        # Exact match
        self.assertEqual(resolve_tool_name("wiki__get_article", available), "wiki__get_article")
        
        # Unprefixed match
        self.assertEqual(resolve_tool_name("get_article", available), "wiki__get_article")
        self.assertEqual(resolve_tool_name("search_wikipedia", available), "wiki__search_wikipedia")
        
        # Well-known alias match
        self.assertIn(resolve_tool_name("fetch_page", available), {"fetch__fetch", "parallel__web_fetch"})
        self.assertIn(resolve_tool_name("fetch", available), {"fetch__fetch", "parallel__web_fetch"})
        self.assertEqual(resolve_tool_name("search", available), "wiki__search_wikipedia")
        
        # Unmatched returns raw name
        self.assertEqual(resolve_tool_name("unregistered_tool", available), "unregistered_tool")


if __name__ == "__main__":
    unittest.main()
