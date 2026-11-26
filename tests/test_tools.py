import unittest
from unittest.mock import patch

from finance_agent.key_rotator import KeyRotator
from finance_agent.prompt import INSTRUCTIONS_PROMPT
from finance_agent.tools import EDGARSearch, TavilyWebSearch, _html_bytes_to_text


_CUTOFF = "2025-04-07"
_VALID_RANGES = (
    ({}, None, _CUTOFF),
    ({"end_date": None}, None, _CUTOFF),
    ({"end_date": ""}, None, _CUTOFF),
    ({"end_date": "2026-10-12"}, None, _CUTOFF),
    ({"start_date": "2026-10-12"}, _CUTOFF, _CUTOFF),
    (
        {"start_date": "2026-10-12", "end_date": "2025-12-31"},
        _CUTOFF,
        _CUTOFF,
    ),
    (
        {"start_date": "2024-01-01", "end_date": "2026-10-12"},
        "2024-01-01",
        _CUTOFF,
    ),
    (
        {"start_date": "2024-01-01", "end_date": "2024-12-31"},
        "2024-01-01",
        "2024-12-31",
    ),
)
_INVALID_RANGES = (
    {"start_date": "2025/04/07"},
    {"end_date": "2025/04/07"},
    {"start_date": "2025-04-01", "end_date": "2025-03-31"},
    {"start_date": "2026-10-12", "end_date": "2025-03-31"},
)


class _FakeTavilyClient:
    def __init__(self, calls):
        self.calls = calls

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": [{"title": "test result"}]}


class _FakeEDGARResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return {"filings": [{"id": "test filing"}]}


class _FakeEDGARSession:
    def __init__(self, calls):
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeEDGARResponse()


class PromptDateCutoffTests(unittest.TestCase):
    def test_formatted_prompt_sets_the_as_of_date(self):
        prompt = INSTRUCTIONS_PROMPT.format(question="test question")

        self.assertIn(
            "You should answer all questions as if the current date is April 07, 2025.",
            prompt,
        )
        self.assertIn("test question", prompt)


class TavilyDateCutoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_applies_cutoff_to_actual_search_request(self):
        for date_args, expected_start, expected_end in _VALID_RANGES:
            with self.subTest(date_args=date_args):
                calls = []
                tool = TavilyWebSearch(
                    key_rotator=KeyRotator(["fake-tavily-key"]),
                    client_factory=lambda key: _FakeTavilyClient(calls),
                )

                results = await tool._execute_search("test query", **date_args)

                self.assertEqual(results, [{"title": "test result"}])
                self.assertEqual(len(calls), 1)
                request = calls[0]
                self.assertEqual(request["query"], "test query")
                self.assertEqual(request["end_date"], expected_end)
                if expected_start is None:
                    self.assertNotIn("start_date", request)
                else:
                    self.assertEqual(request["start_date"], expected_start)

    async def test_rejects_invalid_ranges_before_request(self):
        for date_args in _INVALID_RANGES:
            with self.subTest(date_args=date_args):
                calls = []
                tool = TavilyWebSearch(
                    key_rotator=KeyRotator(["fake-tavily-key"]),
                    client_factory=lambda key: _FakeTavilyClient(calls),
                )

                with self.assertRaises(ValueError):
                    await tool._execute_search("test query", **date_args)

                self.assertEqual(calls, [])


class EDGARDateCutoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_applies_cutoff_to_actual_search_request(self):
        for date_args, expected_start, expected_end in _VALID_RANGES:
            with self.subTest(date_args=date_args):
                calls = []
                tool = EDGARSearch(key_rotator=KeyRotator(["fake-sec-key"]))
                with patch(
                    "finance_agent.tools.aiohttp.ClientSession",
                    return_value=_FakeEDGARSession(calls),
                ):
                    results = await tool._execute_search("test query", **date_args)

                self.assertEqual(results, [{"id": "test filing"}])
                self.assertEqual(len(calls), 1)
                url, request = calls[0]
                self.assertEqual(url, tool.sec_api_url)
                self.assertEqual(request["json"]["query"], "test query")
                self.assertEqual(
                    request["json"]["startDate"], expected_start or "1900-01-01"
                )
                self.assertEqual(request["json"]["endDate"], expected_end)

    async def test_rejects_invalid_ranges_before_request(self):
        for date_args in _INVALID_RANGES:
            with self.subTest(date_args=date_args):
                calls = []
                tool = EDGARSearch(key_rotator=KeyRotator(["fake-sec-key"]))
                with patch(
                    "finance_agent.tools.aiohttp.ClientSession",
                    return_value=_FakeEDGARSession(calls),
                ) as session_factory:
                    with self.assertRaises(ValueError):
                        await tool._execute_search("test query", **date_args)

                session_factory.assert_not_called()
                self.assertEqual(calls, [])


class ParseHtmlPageDecodingTests(unittest.TestCase):
    def test_decodes_utf16_bom(self) -> None:
        content = "<html><body>Revenue £123</body></html>".encode("utf-16")

        text = _html_bytes_to_text(
            content,
            declared_encoding=None,
            content_type="text/html",
        )

        self.assertEqual(text, "Revenue £123")

    def test_falls_back_when_server_incorrectly_declares_utf8(self) -> None:
        content = b"<html><body>Price \xfe 123</body></html>"

        text = _html_bytes_to_text(
            content,
            declared_encoding="utf-8",
            content_type="text/html; charset=utf-8",
        )

        self.assertEqual(text, "Price þ 123")

    def test_uses_encoding_declared_inside_html(self) -> None:
        content = (
            '<html><head><meta charset="gb18030"></head>'
            "<body>营业收入</body></html>"
        ).encode("gb18030")

        text = _html_bytes_to_text(
            content,
            declared_encoding=None,
            content_type="text/html",
        )

        self.assertEqual(text, "营业收入")

    def test_removes_scripts_and_styles_after_decoding(self) -> None:
        content = (
            b"<html><style>hidden</style><body>Visible"
            b"<script>also hidden</script></body></html>"
        )

        text = _html_bytes_to_text(
            content,
            declared_encoding="utf-8",
            content_type="text/html",
        )

        self.assertEqual(text, "Visible")

    def test_rejects_pdf_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary content"):
            _html_bytes_to_text(
                b"%PDF-1.7 binary",
                declared_encoding=None,
                content_type="application/pdf",
            )


if __name__ == "__main__":
    unittest.main()
