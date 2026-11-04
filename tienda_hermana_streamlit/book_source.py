from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests


class BookSourceError(ValueError):
    """Error legible al obtener el Book desde OneDrive."""


_ALLOWED_HOSTS = (
    "1drv.ms",
    "1drv.com",
    "onedrive.live.com",
    "onedrive.com",
    "sharepoint.com",
    "office.com",
)
_MAX_BYTES = 100 * 1024 * 1024


def _allowed_host(host: str) -> bool:
    normalized = host.lower().rstrip(".")
    return any(normalized == domain or normalized.endswith("." + domain) for domain in _ALLOWED_HOSTS)


def build_onedrive_download_url(shared_url: str) -> str:
    """Add download=1 to a OneDrive/SharePoint shared URL without exposing it."""
    value = str(shared_url or "").strip()
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or not _allowed_host(parts.hostname):
        raise BookSourceError(
            "El valor de BOOK_ONEDRIVE_URL debe ser un enlace compartido de OneDrive o SharePoint (https://...)."
        )
    query = [(key, item) for key, item in parse_qsl(parts.query, keep_blank_values=True) if key.lower() != "download"]
    query.append(("download", "1"))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def download_onedrive_book(shared_url: str, timeout_seconds: int = 45) -> bytes:
    """Download an anonymously accessible .xlsx shared from OneDrive/SharePoint.

    The link is treated as a secret. It is never included in exceptions or logs.
    The owner must enable link access according to their organization's policy.
    """
    download_url = build_onedrive_download_url(shared_url)
    try:
        with requests.get(
            download_url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; TiendaHermana/1.0)",
                "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/octet-stream,*/*",
            },
            allow_redirects=True,
            stream=True,
            timeout=(10, timeout_seconds),
        ) as response:
            if response.status_code != 200:
                raise BookSourceError(
                    f"OneDrive respondió HTTP {response.status_code}. Revisa permisos del enlace y vuelve a intentar."
                )
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit() and int(content_length) > _MAX_BYTES:
                raise BookSourceError("El Book supera el límite de descarga de 100 MB.")
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > _MAX_BYTES:
                    raise BookSourceError("El Book supera el límite de descarga de 100 MB.")
                chunks.append(chunk)
            content = b"".join(chunks)
    except BookSourceError:
        raise
    except requests.Timeout as exc:
        raise BookSourceError("La descarga de OneDrive tardó demasiado. Intenta actualizar de nuevo.") from exc
    except requests.RequestException as exc:
        raise BookSourceError(
            "No se pudo conectar con OneDrive. Verifica el enlace y la disponibilidad del servicio."
        ) from exc

    if not content.startswith(b"PK\x03\x04"):
        raise BookSourceError(
            "OneDrive no devolvió un archivo .xlsx. Configura un enlace de descarga compartido con acceso permitido; "
            "un enlace que exige inicio de sesión no puede leerse solo con la URL."
        )
    return content
