"""Credential-free, DNS-pinned downloads from administrator allowlisted hosts."""

from __future__ import annotations

import asyncio
import ipaddress
import math
import socket
from collections.abc import Awaitable, Callable, Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from .errors import ImageTooLarge, MediaDownloadError, UnsafeMediaSource

Resolver = Callable[[str, int], Awaitable[list[ipaddress.IPv4Address | ipaddress.IPv6Address]]]


class PublicMediaDownloader:
    """Download HTTPS media while pinning each request to a validated public IP."""

    def __init__(
        self,
        allowed_hosts: Iterable[str],
        *,
        max_bytes: int = 20 * 1024 * 1024,
        max_redirects: int = 3,
        client: httpx.AsyncClient | None = None,
        resolver: Resolver | None = None,
        allowed_content_types: Iterable[str] = (
            "image/png",
            "image/jpeg",
            "image/webp",
            "application/octet-stream",
        ),
        accept: str = "image/png,image/jpeg,image/webp",
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(max_redirects, bool)
            or not isinstance(max_redirects, int)
            or max_redirects < 0
        ):
            raise ValueError("max_redirects must be a non-negative integer")
        self.allowed_hosts = frozenset(self._normalize_host(host) for host in allowed_hosts)
        self.allowed_content_types = frozenset(allowed_content_types)
        if not self.allowed_content_types or not all(
            isinstance(value, str) and value for value in self.allowed_content_types
        ):
            raise ValueError("allowed_content_types must contain media types")
        if (
            not isinstance(accept, str)
            or not accept
            or any(ord(c) < 32 or ord(c) == 127 for c in accept)
        ):
            raise ValueError("accept header is invalid")
        self.accept = accept
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self._client = client or httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            timeout=None,
            limits=httpx.Limits(max_keepalive_connections=0),
            http2=False,
        )
        self._resolver = resolver or self._resolve
        self._closed = False

    @staticmethod
    def _normalize_host(host: str) -> str:
        if not isinstance(host, str) or not host or "://" in host or "/" in host:
            raise ValueError("allowed media hosts must be bare hostnames")
        try:
            normalized = host.rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise ValueError("allowed media hostname is invalid") from exc
        if not normalized:
            raise ValueError("allowed media hostname is invalid")
        return normalized

    @staticmethod
    async def _resolve(host: str, port: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        loop = asyncio.get_running_loop()
        try:
            records = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise MediaDownloadError("Media hostname resolution failed") from exc
        addresses = []
        for record in records:
            address = ipaddress.ip_address(record[4][0])
            if address not in addresses:
                addresses.append(address)
        return addresses

    async def _validated_target(self, url: str) -> tuple[str, str, str]:
        try:
            parsed = urlsplit(url)
            host_value = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise UnsafeMediaSource("Media URL is invalid") from exc
        if parsed.scheme.lower() != "https" or not host_value:
            raise UnsafeMediaSource("Only HTTPS media URLs are accepted")
        if parsed.username is not None or parsed.password is not None:
            raise UnsafeMediaSource("Media URLs cannot contain credentials")
        if port not in {None, 443}:
            raise UnsafeMediaSource("Media URL must use HTTPS port 443")
        host = self._normalize_host(host_value)
        if host not in self.allowed_hosts:
            raise UnsafeMediaSource("Media host is not allowlisted")
        addresses = await self._resolver(host, 443)
        if not addresses or any(not address.is_global for address in addresses):
            raise UnsafeMediaSource(
                "Media hostname did not resolve exclusively to public addresses"
            )
        address = addresses[0]
        pinned_host = f"[{address}]" if address.version == 6 else str(address)
        path = parsed.path or "/"
        pinned_url = urlunsplit(("https", pinned_host, path, parsed.query, ""))
        logical_url = urlunsplit(("https", host, path, parsed.query, ""))
        return logical_url, pinned_url, host

    async def fetch(self, url: str, *, deadline: float) -> bytes:
        if self._closed:
            raise RuntimeError("PublicMediaDownloader is closed")
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
        ):
            raise TypeError("deadline must be a finite monotonic timestamp")
        logical_url = url
        try:
            async with asyncio.timeout_at(float(deadline)):
                for redirect_count in range(self.max_redirects + 1):
                    logical_url, pinned_url, host = await self._validated_target(logical_url)
                    request = httpx.Request(
                        "GET",
                        pinned_url,
                        headers={"Host": host, "Accept": self.accept},
                        extensions={"sni_hostname": host},
                    )
                    for attempt in range(2):
                        try:
                            response = await self._client.send(request, stream=True)
                            break
                        except httpx.TransportError as exc:
                            if attempt:
                                raise MediaDownloadError("Media download failed") from exc
                    try:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location or redirect_count >= self.max_redirects:
                                raise UnsafeMediaSource("Media redirect limit exceeded")
                            logical_url = urljoin(logical_url, location)
                            continue
                        if response.status_code != 200:
                            raise MediaDownloadError(
                                "Media download returned an unsuccessful status"
                            )
                        content_type = (
                            response.headers.get("content-type", "").split(";", 1)[0].lower()
                        )
                        if content_type and content_type not in self.allowed_content_types:
                            raise UnsafeMediaSource("Media response Content-Type is not allowed")
                        content_length = response.headers.get("content-length")
                        if content_length:
                            try:
                                declared_length = int(content_length)
                            except ValueError as exc:
                                raise MediaDownloadError("Media Content-Length is invalid") from exc
                            if declared_length < 0:
                                raise MediaDownloadError("Media Content-Length is invalid")
                            if declared_length > self.max_bytes:
                                raise ImageTooLarge("Downloaded image exceeds the byte limit")
                        chunks = bytearray()
                        async for chunk in response.aiter_bytes():
                            chunks.extend(chunk)
                            if len(chunks) > self.max_bytes:
                                raise ImageTooLarge("Downloaded image exceeds the byte limit")
                        return bytes(chunks)
                    finally:
                        await response.aclose()
        except TimeoutError as exc:
            raise MediaDownloadError("Media download deadline expired") from exc
        raise MediaDownloadError("Media download did not produce a response")

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._client.aclose()
