"""Credential-free video results through a trusted proxy or pinned public DNS."""

from __future__ import annotations

import asyncio
import math
from urllib.parse import urlsplit

import httpx

from .errors import ImageTooLarge, MediaDownloadError, UnsafeMediaSource
from .media_download import PublicMediaDownloader

VIDEO_RESULT_HOST = "vidgen.x.ai"
_VIDEO_TYPES = ("video/mp4", "application/octet-stream")


class CredentialFreeVideoDownloader:
    """Proxy only the fixed xAI video origin; other hosts retain DNS pinning.

    The proxy is an administrator-configured transport, independent of the
    OAuth inference client. Exact-origin validation and ordinary TLS verification
    prevent using this compatibility path to reach arbitrary media destinations.
    """

    PROXY_HOST = VIDEO_RESULT_HOST

    def __init__(
        self,
        allowed_hosts,
        *,
        max_bytes=20 * 1024 * 1024,
        result_proxy: str | None = None,
        proxy_client=None,
        public_downloader=None,
    ):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if result_proxy is not None and not isinstance(result_proxy, str):
            raise ValueError("result_proxy must be a string")
        self.max_bytes = max_bytes
        self._public = public_downloader or PublicMediaDownloader(
            allowed_hosts,
            max_bytes=max_bytes,
            allowed_content_types=_VIDEO_TYPES,
            accept=",".join(_VIDEO_TYPES),
        )
        self._proxy = (
            proxy_client
            or httpx.AsyncClient(
                proxy=result_proxy,
                trust_env=False,
                follow_redirects=False,
                timeout=None,
            )
            if result_proxy
            else None
        )
        self._closed = False

    @classmethod
    def _fixed_origin(cls, url):
        if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) == 127 for c in url):
            raise UnsafeMediaSource("Video result URL is invalid")
        try:
            parsed = urlsplit(url)
            host, port = parsed.hostname, parsed.port
        except ValueError:
            raise UnsafeMediaSource("Video result URL is invalid") from None
        if host != cls.PROXY_HOST:
            return False
        if (
            parsed.scheme != "https"
            or port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise UnsafeMediaSource("Video result URL must use the fixed HTTPS origin")
        return True

    async def fetch(self, url, *, deadline):
        if self._closed:
            raise RuntimeError("Video downloader is closed")
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
        ):
            raise TypeError("deadline must be a finite monotonic timestamp")
        fixed = self._fixed_origin(url)
        if self._proxy is None or not fixed:
            return await self._public.fetch(url, deadline=deadline)
        try:
            async with asyncio.timeout_at(deadline):
                # Construct an independent request; no API bearer, default client
                # headers, cookie jar, or auth flow is inherited by this send.
                request = httpx.Request("GET", url, headers={"Accept": ",".join(_VIDEO_TYPES)})
                response = await self._proxy.send(
                    request, stream=True, auth=None, follow_redirects=False
                )
                try:
                    if response.is_redirect:
                        raise UnsafeMediaSource("Video result redirects are forbidden")
                    if response.status_code != 200:
                        raise MediaDownloadError("Video download returned an unsuccessful status")
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if content_type and content_type not in _VIDEO_TYPES:
                        raise UnsafeMediaSource("Video response Content-Type is not allowed")
                    length = response.headers.get("content-length")
                    if length:
                        try:
                            length = int(length)
                        except ValueError:
                            raise MediaDownloadError("Video Content-Length is invalid") from None
                        if length < 0:
                            raise MediaDownloadError("Video Content-Length is invalid")
                        if length > self.max_bytes:
                            raise UnsafeMediaSource("Video exceeds the download byte limit")
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(content) + len(chunk) > self.max_bytes:
                            raise UnsafeMediaSource("Video exceeds the download byte limit")
                        content.extend(chunk)
                    return bytes(content)
                finally:
                    await response.aclose()
        except TimeoutError:
            raise MediaDownloadError("Video download deadline expired") from None
        except httpx.HTTPError:
            raise MediaDownloadError("Video download failed") from None

    async def close(self):
        if not self._closed:
            self._closed = True
            try:
                await self._public.close()
            finally:
                if self._proxy is not None:
                    await self._proxy.aclose()


class _CredentialFreeSourceClient:
    """Keep injected clients from adding auth or following unchecked redirects."""

    def __init__(self, client):
        self._client = client

    async def send(self, request, *, stream):
        return await self._client.send(request, stream=stream, auth=None, follow_redirects=False)

    async def aclose(self):
        await self._client.aclose()


class ExternalVideoDownloader(CredentialFreeVideoDownloader):
    """QQ source proxy access is separate from xAI result proxy access."""

    PROXY_HOST = "multimedia.nt.qq.com.cn"

    def __init__(
        self,
        allowed_hosts=(),
        *,
        max_bytes=20 * 1024 * 1024,
        source_proxy=None,
        proxy_client=None,
        client=None,
        resolver=None,
    ):
        public = PublicMediaDownloader(
            allowed_hosts,
            max_bytes=max_bytes,
            client=_CredentialFreeSourceClient(
                client
                or httpx.AsyncClient(
                    trust_env=False,
                    follow_redirects=False,
                    timeout=None,
                    limits=httpx.Limits(max_keepalive_connections=0),
                    http2=False,
                )
            ),
            resolver=resolver,
            allowed_content_types=_VIDEO_TYPES,
            accept=",".join(_VIDEO_TYPES),
        )
        super().__init__(
            allowed_hosts,
            max_bytes=max_bytes,
            result_proxy=source_proxy,
            proxy_client=proxy_client,
            public_downloader=public,
        )

    async def fetch(self, url, *, deadline):
        if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) == 127 for c in url):
            raise UnsafeMediaSource("Video source URL is invalid")
        try:
            parsed = urlsplit(url)
            if parsed.fragment:
                raise UnsafeMediaSource("Video source URL fragments are forbidden")
        except ValueError:
            raise UnsafeMediaSource("Video source URL is invalid") from None
        try:
            return await super().fetch(url, deadline=deadline)
        except ImageTooLarge:
            raise UnsafeMediaSource("Video exceeds the import byte limit") from None
