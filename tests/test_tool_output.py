import unittest


class TestToolOutput(unittest.TestCase):
    """Regression tests for Task 11: Faithful bounded tool outcomes and format-aware presentation (F11)."""

    def test_preserves_short_source_tags_and_indentation(self) -> None:
        """Source code containing <script> tags, HTML, or indentation must be preserved faithfully."""
        from mcp_manager import format_tool_result

        code_snippet = (
            "def render_html():\n"
            "    # Code with script tag\n"
            "    return '<script>alert(1)</script>'\n"
        )

        formatted = format_tool_result("read_file", code_snippet)
        # Must retain <script>alert(1)</script> exactly without stripping
        self.assertIn("<script>alert(1)</script>", formatted)
        self.assertIn("    return", formatted)

    def test_preserves_json_structures(self) -> None:
        """JSON output from query or API tools must preserve quotes and structure."""
        from mcp_manager import format_tool_result

        raw_json = '{\n  "status": "ok",\n  "users": [1, 2, 3]\n}'
        formatted = format_tool_result("query_db", raw_json)
        self.assertEqual(formatted, raw_json)

    def test_strips_huge_base64_payloads(self) -> None:
        """Huge inline base64 data payloads must be replaced with a concise placeholder."""
        from mcp_manager import format_tool_result

        giant_base64 = "data:image/png;base64," + ("A" * 500)
        input_text = f"Image loaded: {giant_base64}"
        formatted = format_tool_result("fetch", input_text)
        self.assertNotIn("A" * 500, formatted)
        self.assertIn("[base64_data_omitted]", formatted)

    def test_strict_cap_at_boundary_lengths(self) -> None:
        """Result must never exceed MAX_TOOL_RESULT_CHARS including all markers."""
        from mcp_manager import format_tool_result, MAX_TOOL_RESULT_CHARS

        # Test at boundaries: MAX-1, MAX, MAX+1, and MAX+20000 characters
        for length in (MAX_TOOL_RESULT_CHARS - 1, MAX_TOOL_RESULT_CHARS, MAX_TOOL_RESULT_CHARS + 1, MAX_TOOL_RESULT_CHARS + 20000):
            raw = "x" * length
            formatted = format_tool_result("fetch_page", raw)
            self.assertLessEqual(
                len(formatted),
                MAX_TOOL_RESULT_CHARS,
                f"Formatted output exceeded limit ({len(formatted)} > {MAX_TOOL_RESULT_CHARS}) for input len {length}",
            )
            if length <= MAX_TOOL_RESULT_CHARS:
                self.assertEqual(len(formatted), length)
            else:
                self.assertIn("sampled", formatted.lower())

    def test_error_outcomes_preserve_error_status(self) -> None:
        """Tool outcomes marked as error must clearly indicate the error state."""
        from mcp_manager import ToolOutcome, format_tool_outcome

        outcome = ToolOutcome(text="Database constraint violation", is_error=True)
        rendered = format_tool_outcome("query_db", outcome)
        self.assertIn("error", rendered.lower())
        self.assertIn("Database constraint violation", rendered)


if __name__ == "__main__":
    unittest.main()
