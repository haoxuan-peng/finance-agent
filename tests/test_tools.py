import unittest

from finance_agent.tools import _html_bytes_to_text


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
