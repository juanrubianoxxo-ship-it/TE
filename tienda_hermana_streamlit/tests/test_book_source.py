from urllib.parse import parse_qs, urlsplit

import pytest

from book_source import BookSourceError, build_onedrive_download_url, download_onedrive_book


def test_onedrive_shared_link_gets_download_parameter():
    url = build_onedrive_download_url("https://1drv.ms/x/s!abc123?e=12345")
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    assert parts.scheme == "https"
    assert parts.netloc == "1drv.ms"
    assert query["e"] == ["12345"]
    assert query["download"] == ["1"]


def test_onedrive_existing_download_parameter_is_replaced():
    url = build_onedrive_download_url("https://contoso.sharepoint.com/:x:/g/personal/book?download=0")
    query = parse_qs(urlsplit(url).query)
    assert query["download"] == ["1"]


def test_rejects_non_microsoft_host():
    with pytest.raises(BookSourceError, match="BOOK_ONEDRIVE_URL"):
        build_onedrive_download_url("https://example.com/private-book.xlsx")


def test_download_returns_xlsx_bytes_from_shared_link(monkeypatch):
    class Response:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def iter_content(self, chunk_size):
            assert chunk_size == 1024 * 1024
            return iter([b"PK\x03\x04test-xlsx"])

    def fake_get(url, **kwargs):
        assert "download=1" in url
        assert kwargs["allow_redirects"] is True
        return Response()

    monkeypatch.setattr("book_source.requests.get", fake_get)
    assert download_onedrive_book("https://1drv.ms/x/s!secret") == b"PK\x03\x04test-xlsx"


def test_download_rejects_login_html_without_echoing_secret(monkeypatch):
    shared = "https://1drv.ms/x/s!do-not-print-this"

    class Response:
        status_code = 200
        headers = {"Content-Type": "text/html"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def iter_content(self, chunk_size):
            return iter([b"<html>sign in</html>"])

    monkeypatch.setattr("book_source.requests.get", lambda *args, **kwargs: Response())
    with pytest.raises(BookSourceError) as error:
        download_onedrive_book(shared)
    assert shared not in str(error.value)
    assert "inicio de sesión" in str(error.value)
