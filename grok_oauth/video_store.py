"""Bounded, scoped MP4 assets and durable video operation records."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import os
import re
import stat
import tempfile
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlsplit

from filelock import FileLock, Timeout

from .errors import (
    AssetExpired,
    AssetNotFound,
    Busy,
    CredentialPersistenceError,
    MediaDownloadError,
    ProtocolError,
    UnsafeMediaSource,
)
from .media import _scope_dict
from .models import AssetScope
from .video_download import CredentialFreeVideoDownloader, ExternalVideoDownloader

MAX_VIDEO_BYTES = 20 * 1024 * 1024
DEFAULT_VIDEO_HOSTS = ("vidgen.x.ai",)
DEFAULT_VIDEO_SOURCE_HOSTS = ("multimedia.nt.qq.com.cn",)
MAX_EDIT_DURATION = 8.7

_ID = re.compile(r"[a-f0-9]{32}")


@dataclass(frozen=True)
class VideoAsset:
    asset_id: str
    path: str = field(repr=False)
    mime_type: str = "video/mp4"
    size_bytes: int = 0
    duration: float = 0
    sha256: str = ""


@dataclass(frozen=True)
class VideoJob:
    job_id: str
    status: str
    asset_id: str = ""
    error: str = ""
    request_id: str = field(default="", repr=False)
    model: str = ""
    duration: float = 0
    submission_state: str = "unknown"

    def public(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "submission_state": self.submission_state,
            "asset_id": self.asset_id,
            "error": self.error,
            "model": self.model,
            "duration": self.duration,
        }


def _boxes(data: bytes, start=0, end=None):
    end = len(data) if end is None else end
    position = start
    while position < end:
        if position + 8 > end:
            raise UnsafeMediaSource("MP4 box is truncated")
        size = int.from_bytes(data[position : position + 4], "big")
        kind = data[position + 4 : position + 8]
        header = 8
        if size == 1:
            if position + 16 > end:
                raise UnsafeMediaSource("MP4 box is truncated")
            size = int.from_bytes(data[position + 8 : position + 16], "big")
            header = 16
        elif size == 0:
            size = end - position
        if size < header or position + size > end:
            raise UnsafeMediaSource("MP4 box size is invalid")
        yield kind, position + header, position + size
        position += size


def mp4_duration(data: bytes) -> float:
    """Validate MP4 framing and obtain the movie duration without external processes."""
    movie_duration = None
    brands = False
    media = False
    for kind, start, end in _boxes(data):
        if kind == b"ftyp":
            payload = data[start:end]
            if len(payload) < 8 or (len(payload) - 8) % 4:
                raise UnsafeMediaSource("MP4 file type is invalid")
            supported = {b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1"}
            brands = payload[:4] in supported or any(
                payload[i : i + 4] in supported for i in range(8, len(payload), 4)
            )
        elif kind == b"mdat":
            media = end > start
        elif kind == b"moov":
            for child, left, right in _boxes(data, start, end):
                if child != b"mvhd":
                    continue
                payload = data[left:right]
                if len(payload) < 20 or payload[0] not in {0, 1}:
                    raise UnsafeMediaSource("MP4 duration is invalid")
                offset = 20 if payload[0] == 1 else 12
                width = 8 if payload[0] == 1 else 4
                if len(payload) < offset + 4 + width:
                    raise UnsafeMediaSource("MP4 duration is truncated")
                timescale = int.from_bytes(payload[offset : offset + 4], "big")
                duration = int.from_bytes(payload[offset + 4 : offset + 4 + width], "big")
                if not timescale or duration == (1 << (width * 8)) - 1:
                    raise UnsafeMediaSource("MP4 duration is unavailable")
                movie_duration = duration / timescale
    if not brands or not media or not movie_duration or not math.isfinite(movie_duration):
        raise UnsafeMediaSource("A complete MP4 movie is required")
    if movie_duration > 60:
        raise UnsafeMediaSource("Video duration exceeds the supported limit")
    return movie_duration


def _header_duration(payload: bytes, *, track=False) -> float:
    if not payload or payload[0] not in {0, 1}:
        raise UnsafeMediaSource("MP4 track duration is invalid")
    version = payload[0]
    offset = (28 if version else 20) if track else (20 if version else 12)
    width = 8 if version else 4
    if len(payload) < offset + (0 if track else 4) + width:
        raise UnsafeMediaSource("MP4 track duration is truncated")
    timescale = 1 if track else int.from_bytes(payload[offset : offset + 4], "big")
    left = offset if track else offset + 4
    duration = int.from_bytes(payload[left : left + width], "big")
    if not timescale or not duration or duration == (1 << (width * 8)) - 1:
        raise UnsafeMediaSource("MP4 track duration is unavailable")
    return duration / timescale


def _import_duration(content: bytes) -> float:
    """Check movie and all track timelines before admitting an edit source."""
    duration = mp4_duration(content)
    movie_count = 0
    brand_count = 0
    for kind, _, _ in _boxes(content):
        movie_count += kind == b"moov"
        brand_count += kind == b"ftyp"
    if movie_count != 1 or brand_count != 1:
        raise UnsafeMediaSource("MP4 movie structure is invalid")
    movie_scale = None
    video_track = False
    track_count = 0
    for kind, start, end in _boxes(content):
        if kind != b"moov":
            continue
        header = None
        for child, left, right in _boxes(content, start, end):
            if child == b"mvhd":
                if header is not None:
                    raise UnsafeMediaSource("MP4 movie header is invalid")
                header = (left, right)
        if header is None:
            raise UnsafeMediaSource("MP4 movie header is invalid")
        left, right = header
        payload = content[left:right]
        _header_duration(payload)
        offset = 20 if payload[0] else 12
        movie_scale = int.from_bytes(payload[offset : offset + 4], "big")
        for child, left, right in _boxes(content, start, end):
            if child != b"trak":
                continue
            track_count += 1
            track_headers = 0
            media_headers = 0
            handlers = 0
            for entry, a, b in _boxes(content, left, right):
                if entry == b"tkhd":
                    track_headers += 1
                    timeline = _header_duration(content[a:b], track=True) / movie_scale
                    if timeline > MAX_EDIT_DURATION:
                        raise UnsafeMediaSource("Video exceeds the edit duration limit")
                elif entry == b"mdia":
                    for item, x, y in _boxes(content, a, b):
                        if item == b"mdhd":
                            media_headers += 1
                            if _header_duration(content[x:y]) > MAX_EDIT_DURATION:
                                raise UnsafeMediaSource("Video exceeds the edit duration limit")
                        elif item == b"hdlr":
                            handlers += 1
                            if y - x < 12:
                                raise UnsafeMediaSource("MP4 track handler is truncated")
                            video_track |= content[x + 8 : x + 12] == b"vide"
            if track_headers != 1 or media_headers != 1 or handlers != 1:
                raise UnsafeMediaSource("A complete MP4 track is required")
    if not movie_scale or not track_count or not video_track:
        raise UnsafeMediaSource("A complete MP4 video track is required")
    if duration > MAX_EDIT_DURATION:
        raise UnsafeMediaSource("Video exceeds the edit duration limit")
    return duration


def _read_import_file(reference: str, allowed_roots, max_bytes: int) -> bytes:
    """Open beneath an explicitly trusted root without following any child symlink."""
    try:
        parsed = urlsplit(reference)
    except ValueError:
        raise UnsafeMediaSource("Local video URI is invalid") from None
    if "\x00" in reference:
        raise UnsafeMediaSource("Local video source is invalid")
    if parsed.scheme == "file":
        if parsed.netloc or parsed.query or parsed.fragment:
            raise UnsafeMediaSource("Local video URI is invalid")
        decoded_path = unquote(parsed.path)
        if "\x00" in decoded_path:
            raise UnsafeMediaSource("Local video source is invalid")
        path = Path(decoded_path)
    elif parsed.scheme:
        raise UnsafeMediaSource("Video source scheme is unsupported")
    else:
        path = Path(reference)
    if not path.is_absolute() or ".." in path.parts or not allowed_roots:
        raise UnsafeMediaSource("Local video source is not allowed")
    for value in allowed_roots:
        try:
            root = Path(value).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise UnsafeMediaSource("Local video root is unavailable") from None
        if root == Path(root.anchor):
            raise UnsafeMediaSource("A filesystem root cannot allow local video imports")
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if not relative.parts:
            raise UnsafeMediaSource("Local video source must be a regular file")
        descriptor = None
        try:
            descriptor = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            for part in (*root.parts[1:], *relative.parts[:-1]):
                following = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
                os.close(descriptor)
                descriptor = following
            following = os.open(
                relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = following
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
                raise UnsafeMediaSource("Local video must be a bounded regular file")
            with os.fdopen(descriptor, "rb") as source:
                descriptor = None
                content = source.read(max_bytes + 1)
            if len(content) > max_bytes:
                raise UnsafeMediaSource("Video exceeds the import byte limit")
            return content
        except OSError:
            raise UnsafeMediaSource("Local video source is unavailable") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
    raise UnsafeMediaSource("Local video source is not allowed")


class VideoStore:
    def __init__(
        self,
        root: Path,
        *,
        media_hosts=(),
        media_downloader=None,
        source_media_hosts=(),
        source_downloader=None,
        source_proxy: str | None = None,
        result_proxy: str | None = None,
        max_video_bytes: int = MAX_VIDEO_BYTES,
        max_total_bytes: int = 256 * 1024 * 1024,
        max_jobs: int = 256,
        ttl_seconds: float = 604800,
    ):
        for name, value in (
            ("max_video_bytes", max_video_bytes),
            ("max_total_bytes", max_total_bytes),
            ("max_jobs", max_jobs),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise ValueError("ttl_seconds must be positive")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._files = self.root / "files"
        self._files.mkdir(exist_ok=True, mode=0o700)
        self._metadata = self.root / "metadata.json"
        self._file_lock = FileLock(str(self.root / ".metadata.lock"), timeout=0)
        self._lock = asyncio.Lock()
        self._leases = Counter()
        self._volatile_jobs = {}
        self._closed = False
        self.max_video_bytes = max_video_bytes
        self.max_total_bytes = max_total_bytes
        self.max_jobs = max_jobs
        self.ttl_seconds = float(ttl_seconds)
        media_hosts = tuple(dict.fromkeys((*DEFAULT_VIDEO_HOSTS, *media_hosts)))
        self._sources = source_downloader or ExternalVideoDownloader(
            tuple(dict.fromkeys((*DEFAULT_VIDEO_SOURCE_HOSTS, *source_media_hosts))),
            max_bytes=min(max_video_bytes, MAX_VIDEO_BYTES),
            source_proxy=source_proxy,
        )
        self._media = media_downloader or (
            CredentialFreeVideoDownloader(
                media_hosts, max_bytes=max_video_bytes, result_proxy=result_proxy
            )
            if media_hosts
            else None
        )

    def _load(self):
        if not self._metadata.exists():
            return {"assets": {}, "jobs": {}}
        try:
            data = json.loads(self._metadata.read_text())
            if not isinstance(data, dict) or not all(
                isinstance(data.get(k), dict) for k in ("assets", "jobs")
            ):
                raise ValueError
            if len(data["jobs"]) > self.max_jobs or len(data["assets"]) > self.max_jobs:
                raise ValueError
            for table in ("assets", "jobs"):
                for key, value in data[table].items():
                    if not _ID.fullmatch(key) or not isinstance(value, dict):
                        raise ValueError
                    if not isinstance(value.get("scope"), dict) or not isinstance(
                        value.get("expires"), (float, int)
                    ):
                        raise ValueError
            return data
        except (ValueError, OSError, TypeError) as exc:
            raise ProtocolError("Video metadata is unavailable") from exc

    def _write(self, data):
        descriptor, name = tempfile.mkstemp(dir=self.root, prefix=".metadata-")
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(data, output, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, self._metadata)
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _persist(self, data):
        try:
            self._write(data)
        except OSError as exc:
            raise CredentialPersistenceError("Video metadata could not be persisted") from exc
        self._volatile_jobs.clear()

    @asynccontextmanager
    async def _transaction(self):
        if self._closed:
            raise RuntimeError("VideoStore is closed")
        async with self._lock:
            if self._closed:
                raise RuntimeError("VideoStore is closed")
            try:
                with self._file_lock:
                    data = self._load()
                    data["jobs"].update(self._volatile_jobs)
                    yield data
            except Timeout as exc:
                raise Busy("Video storage is busy") from exc

    def _lookup(self, data, table, item_id, scope):
        if not isinstance(item_id, str) or not _ID.fullmatch(item_id):
            raise AssetNotFound()
        value = data[table].get(item_id)
        if value is None or value["scope"] != _scope_dict(scope):
            raise AssetNotFound()
        if value["expires"] <= time.time():
            raise AssetExpired()
        return value

    @staticmethod
    def _job(job_id, value):
        status = value["status"]
        submission_state = value.get("submission_state")
        if status == "submitting":
            submission_state = "unknown"
        elif value.get("request_id"):
            submission_state = "submitted"
        elif submission_state not in {"submitted", "not_submitted", "unknown"}:
            submission_state = "unknown"
        # An interrupted submit never becomes an implicit second POST after restart.
        return VideoJob(
            job_id=job_id,
            status="unknown" if status == "submitting" else status,
            asset_id=value.get("asset_id", ""),
            error=value.get("error", ""),
            request_id=value.get("request_id", ""),
            model=value.get("model", ""),
            duration=value.get("duration", 0),
            submission_state=submission_state,
        )

    async def create_job(self, *, scope, operation_key, fingerprint, binding, model, duration):
        scope_data = _scope_dict(scope)
        async with self._transaction() as data:
            for job_id, value in data["jobs"].items():
                if value["scope"] == scope_data and value.get("operation_key") == operation_key:
                    if value["expires"] <= time.time():
                        raise AssetExpired("Video operation expired")
                    if value.get("fingerprint") != fingerprint:
                        raise ProtocolError("Video operation key has different arguments")
                    if value.get("binding") != binding:
                        raise AssetNotFound()
                    return self._job(job_id, value), False
            self._cleanup(data)
            if len(data["jobs"]) >= self.max_jobs:
                raise Busy("Video operation storage is full")
            job_id = uuid.uuid4().hex
            value = {
                "scope": scope_data,
                "expires": time.time() + self.ttl_seconds,
                "operation_key": operation_key,
                "fingerprint": fingerprint,
                "binding": binding,
                "status": "submitting",
                "submission_state": "not_submitted",
                "model": model,
                "duration": duration,
            }
            data["jobs"][job_id] = value
            self._persist(data)
            return self._job(job_id, value), True

    async def get_job(self, job_id, *, scope):
        async with self._transaction() as data:
            return self._job(job_id, self._lookup(data, "jobs", job_id, scope))

    async def job_binding(self, job_id, *, scope):
        async with self._transaction() as data:
            return self._lookup(data, "jobs", job_id, scope).get("binding", "")

    async def update_job(self, job_id, *, scope, **changes):
        if set(changes) - {
            "status",
            "asset_id",
            "error",
            "request_id",
            "duration",
            "submission_state",
        }:
            raise TypeError("Unsupported video job field")
        if "submission_state" in changes and changes["submission_state"] not in {
            "submitted",
            "not_submitted",
            "unknown",
        }:
            raise ValueError("Unsupported video submission state")
        async with self._transaction() as data:
            value = self._lookup(data, "jobs", job_id, scope)
            value.update(changes)
            try:
                self._persist(data)
            except CredentialPersistenceError:
                # Acknowledged remote handles must survive a transient disk error.
                # The persisted pre-submit tombstone still prevents a repeated POST
                # if the process stops before storage can recover.
                value["error"] = "CredentialPersistenceError"
                self._volatile_jobs[job_id] = dict(value)
            return self._job(job_id, value)

    async def list_jobs(self, *, scope):
        async with self._transaction() as data:
            return [
                self._job(key, value)
                for key, value in data["jobs"].items()
                if value["scope"] == _scope_dict(scope) and value["expires"] > time.time()
            ]

    def _cleanup(self, data):
        now = time.time()
        for asset_id, value in tuple(data["assets"].items()):
            if value["expires"] <= now and not self._leases[asset_id]:
                (self._files / f"{asset_id}.mp4").unlink(missing_ok=True)
                del data["assets"][asset_id]
        for job_id, value in tuple(data["jobs"].items()):
            if value["expires"] <= now:
                del data["jobs"][job_id]

    async def cleanup(self):
        async with self._transaction() as data:
            self._cleanup(data)
            self._persist(data)

    async def store_bytes(
        self, content: bytes, *, scope: AssetScope, binding: str = "", provenance: str = "generated"
    ) -> VideoAsset:
        if not isinstance(content, bytes) or not content:
            raise UnsafeMediaSource("Video must contain MP4 bytes")
        if len(content) > self.max_video_bytes:
            raise UnsafeMediaSource("Video exceeds the byte limit")
        duration = mp4_duration(content)
        digest = sha256(content).hexdigest()
        async with self._transaction() as data:
            self._cleanup(data)
            if (
                len(data["assets"]) >= self.max_jobs
                or sum(v["size_bytes"] for v in data["assets"].values()) + len(content)
                > self.max_total_bytes
            ):
                raise Busy("Video asset storage is full")
            asset_id = uuid.uuid4().hex
            path = self._files / f"{asset_id}.mp4"
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                data["assets"][asset_id] = {
                    "scope": _scope_dict(scope),
                    "expires": time.time() + self.ttl_seconds,
                    "size_bytes": len(content),
                    "duration": duration,
                    "sha256": digest,
                    "binding": binding,
                    "provenance": provenance,
                }
                self._persist(data)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            return VideoAsset(
                asset_id, str(path), size_bytes=len(content), duration=duration, sha256=digest
            )

    async def store_url(self, url, *, scope, deadline, binding=""):
        if self._media is None:
            raise UnsafeMediaSource("Video download hosts are not configured")
        content = await self._media.fetch(url, deadline=deadline)
        return await self.store_bytes(content, scope=scope, binding=binding)

    async def import_reference(
        self, reference: str, *, scope, binding, allowed_roots=(), deadline=None
    ) -> str:
        if self._closed:
            raise RuntimeError("VideoStore is closed")
        if (
            not isinstance(reference, str)
            or not reference
            or not isinstance(binding, str)
            or not binding
        ):
            raise UnsafeMediaSource("A video reference and account binding are required")
        if deadline is None:
            deadline = time.monotonic() + 120
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
        ):
            raise TypeError("deadline must be a finite monotonic timestamp")
        if deadline <= time.monotonic():
            raise MediaDownloadError("Video import deadline expired")
        limit = min(self.max_video_bytes, MAX_VIDEO_BYTES)
        prefix = "data:video/mp4;base64,"
        try:
            async with asyncio.timeout_at(deadline):
                if reference.startswith(prefix):
                    encoded_length = len(reference) - len(prefix)
                    if encoded_length > 4 * ((limit + 2) // 3):
                        raise UnsafeMediaSource("Video exceeds the import byte limit")
                    encoded = reference[len(prefix) :]
                    padding = len(encoded) - len(encoded.rstrip("="))
                    if encoded_length % 4 or padding > 2:
                        raise UnsafeMediaSource("Video data URI is invalid")
                    if encoded_length // 4 * 3 - padding > limit:
                        raise UnsafeMediaSource("Video exceeds the import byte limit")
                    try:
                        content = base64.b64decode(encoded, validate=True)
                    except (ValueError, binascii.Error):
                        raise UnsafeMediaSource("Video data URI is invalid") from None
                elif reference.startswith("https:"):
                    content = await self._sources.fetch(reference, deadline=deadline)
                else:
                    content = await asyncio.to_thread(
                        _read_import_file, reference, allowed_roots, limit
                    )
                if len(content) > limit:
                    raise UnsafeMediaSource("Video exceeds the import byte limit")
                _import_duration(content)
                asset = await self.store_bytes(
                    content, scope=scope, binding=binding, provenance="imported"
                )
                return asset.asset_id
        except TimeoutError:
            raise MediaDownloadError("Video import deadline expired") from None

    async def asset_binding(self, asset_id, *, scope):
        async with self._transaction() as data:
            return self._lookup(data, "assets", asset_id, scope).get("binding", "")

    async def get_asset(self, asset_id, *, scope):
        async with self._transaction() as data:
            value = self._lookup(data, "assets", asset_id, scope)
            return VideoAsset(
                asset_id,
                str(self._files / f"{asset_id}.mp4"),
                size_bytes=value["size_bytes"],
                duration=value["duration"],
                sha256=value["sha256"],
            )

    @asynccontextmanager
    async def lease(self, asset_id, *, scope):
        asset = await self.get_asset(asset_id, scope=scope)
        self._leases[asset_id] += 1
        try:
            yield Path(asset.path)
        finally:
            self._leases[asset_id] -= 1

    async def read_bytes(self, asset_id, *, scope):
        async with self.lease(asset_id, scope=scope):
            asset = await self.get_asset(asset_id, scope=scope)
            try:
                descriptor = os.open(asset.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as source:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        raise UnsafeMediaSource("Video asset is not a regular file")
                    content = source.read(self.max_video_bytes + 1)
            except OSError as exc:
                raise AssetNotFound() from exc
            if len(content) != asset.size_bytes or sha256(content).hexdigest() != asset.sha256:
                raise UnsafeMediaSource("Video asset integrity check failed")
            return content

    async def to_base64(self, asset_id, *, scope):
        return base64.b64encode(await self.read_bytes(asset_id, scope=scope)).decode("ascii")

    async def to_data_uri(self, asset_id, *, scope):
        return "data:video/mp4;base64," + await self.to_base64(asset_id, scope=scope)

    async def close(self):
        async with self._lock:
            self._closed = True
        try:
            if self._media is not None:
                await self._media.close()
        finally:
            await self._sources.close()
