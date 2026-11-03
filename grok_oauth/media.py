"""Scoped, persistent storage for generated and imported image assets."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import os
import stat
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlsplit

from filelock import FileLock
from PIL import Image, UnidentifiedImageError

from .errors import AssetExpired, AssetNotFound, ImageTooLarge, UnsafeMediaSource
from .media_download import PublicMediaDownloader
from .models import AssetScope, GeneratedImage

_MIB = 1024 * 1024
_FORMATS = {
    "PNG": ("image/png", ".png"),
    "JPEG": ("image/jpeg", ".jpg"),
    "WEBP": ("image/webp", ".webp"),
}
_DATA_MIMES = frozenset(mime for mime, _ in _FORMATS.values())


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _scope_dict(scope: AssetScope) -> dict[str, str]:
    if not isinstance(scope, AssetScope):
        raise TypeError("scope must be AssetScope")
    if not all(
        isinstance(value, str) and value
        for value in (scope.platform_id, scope.umo, scope.conversation_id)
    ):
        raise TypeError("scope fields must be non-empty strings")
    return {
        "platform_id": scope.platform_id,
        "umo": scope.umo,
        "conversation_id": scope.conversation_id,
    }


def _decode_data_uri(source: str, max_image_bytes: int) -> tuple[str, bytes]:
    header, separator, encoded = source.partition(",")
    if not separator or not header.endswith(";base64"):
        raise UnsafeMediaSource("Only base64 image data URIs are accepted")
    declared_mime = header[5:-7].lower()
    if declared_mime not in _DATA_MIMES:
        raise UnsafeMediaSource("Unsupported data URI media type")
    if len(encoded) > ((max_image_bytes + 2) // 3) * 4 + 4:
        raise ImageTooLarge("Image exceeds the encoded byte limit")
    try:
        return declared_mime, base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UnsafeMediaSource("Image data URI is invalid") from exc


def _is_data_uri(source: str | Path) -> bool:
    return isinstance(source, str) and source.startswith("data:")


def _read_local_reference(
    candidate: Path, allowed_roots: Iterable[Path], max_image_bytes: int
) -> bytes:
    roots = tuple(Path(root).resolve(strict=True) for root in allowed_roots)
    if not roots:
        raise UnsafeMediaSource("Local image roots are not configured")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UnsafeMediaSource("Local image path is unavailable") from exc
    containing_roots = [root for root in roots if resolved.is_relative_to(root)]
    if not resolved.is_file() or not containing_roots:
        raise UnsafeMediaSource("Local image path is outside allowed roots")
    root = max(containing_roots, key=lambda value: len(value.parts))
    relative = resolved.relative_to(root)
    if not relative.parts:
        raise UnsafeMediaSource("Local image path is not a regular file")
    descriptors: list[int] = []
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    try:
        current = os.open(root, os.O_RDONLY | os.O_DIRECTORY | nofollow | cloexec)
        descriptors.append(current)
        for part in relative.parts[:-1]:
            current = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | nofollow | cloexec,
                dir_fd=current,
            )
            descriptors.append(current)
        file_descriptor = os.open(
            relative.parts[-1], os.O_RDONLY | nofollow | cloexec, dir_fd=current
        )
        descriptors.append(file_descriptor)
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            raise UnsafeMediaSource("Local image path is not a regular file")
        output = bytearray()
        while len(output) <= max_image_bytes:
            chunk = os.read(file_descriptor, min(1024 * 1024, max_image_bytes + 1 - len(output)))
            if not chunk:
                break
            output.extend(chunk)
        if len(output) > max_image_bytes:
            raise ImageTooLarge("Image exceeds the encoded byte limit")
        return bytes(output)
    except UnsafeMediaSource:
        raise
    except OSError as exc:
        raise UnsafeMediaSource("Local image changed during secure open") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


class _ThreadBarrier:
    """Keep worker capacity until a thread ends and defer cancellation to consistency points."""

    def __init__(self, store: AssetStore) -> None:
        self.store = store
        self.cancelled: asyncio.CancelledError | None = None

    async def run(self, function, /, *args, protect_acquire: bool = False):
        if protect_acquire:
            acquire = asyncio.create_task(self.store._workers.acquire())
            while not acquire.done():
                try:
                    await asyncio.shield(acquire)
                except asyncio.CancelledError as exc:
                    self.cancelled = self.cancelled or exc
            acquire.result()
        else:
            # Ordinary queued work remains cancellable. asyncio.Semaphore
            # compensates its counter if cancellation races with wakeup.
            await self.store._workers.acquire()
        try:
            task = asyncio.create_task(asyncio.to_thread(function, *args))
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as exc:
                    self.cancelled = self.cancelled or exc
            return task.result()
        finally:
            self.store._workers.release()

    def raise_if_cancelled(self) -> None:
        if self.cancelled is not None:
            raise self.cancelled


class AssetStore:
    """Store validated images behind unguessable, same-scope handles."""

    def __init__(
        self,
        root: Path,
        *,
        ttl_seconds: float = 604800,
        max_total_bytes: int = 2 * 1024 * 1024 * 1024,
        max_image_bytes: int = 20 * _MIB,
        max_pixels: int = 40_000_000,
        max_decoded_bytes: int = 80 * _MIB,
        max_workers: int = 2,
        media_hosts: Iterable[str] = (),
        media_downloader: PublicMediaDownloader | None = None,
    ) -> None:
        self.root = Path(root)
        self.ttl_seconds = _positive_number(ttl_seconds, "ttl_seconds")
        self.max_total_bytes = _positive_integer(max_total_bytes, "max_total_bytes")
        self.max_image_bytes = _positive_integer(max_image_bytes, "max_image_bytes")
        self.max_pixels = _positive_integer(max_pixels, "max_pixels")
        self.max_decoded_bytes = _positive_integer(max_decoded_bytes, "max_decoded_bytes")
        self.max_workers = _positive_integer(max_workers, "max_workers")
        self._files = self.root / "files"
        self._metadata_path = self.root / "metadata.json"
        self._file_lock = FileLock(str(self.root / ".metadata.lock"))
        self._lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._workers = asyncio.Semaphore(self.max_workers)
        self._idle = asyncio.Event()
        self._idle.set()
        self._operations: dict[asyncio.Task, int] = {}
        self._leases: Counter[str] = Counter()
        self._closing = False
        self._closed = False
        configured_media_hosts = tuple(media_hosts)
        self._media = media_downloader or (
            PublicMediaDownloader(configured_media_hosts, max_bytes=self.max_image_bytes)
            if configured_media_hosts
            else None
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self._files.mkdir(parents=True, exist_ok=True)
        self._assets = self._load_metadata()
        if not self._metadata_path.exists():
            self._persist_metadata()

    def _load_metadata(self) -> dict[str, dict]:
        if not self._metadata_path.exists():
            return {}
        try:
            payload = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UnsafeMediaSource("Asset metadata is unreadable") from exc
        assets = payload.get("assets") if isinstance(payload, dict) else None
        if not isinstance(assets, dict):
            raise UnsafeMediaSource("Asset metadata is invalid")
        return {
            key: value
            for key, value in assets.items()
            if isinstance(key, str) and isinstance(value, dict)
        }

    def _persist_metadata(self) -> None:
        payload = json.dumps(
            {"version": 1, "assets": self._assets}, sort_keys=True, separators=(",", ":")
        )
        with self._file_lock:
            fd, name = tempfile.mkstemp(prefix=".metadata-", suffix=".tmp", dir=self.root)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(name, self._metadata_path)
            finally:
                try:
                    os.unlink(name)
                except FileNotFoundError:
                    pass

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AssetStore is closed")

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[None]:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("AssetStore operation requires an asyncio task")
        # These updates contain no await, so they are atomic on the event loop
        # and cannot be interrupted by repeated cancellation.
        depth = self._operations.get(task, 0)
        if depth == 0:
            if self._closing or self._closed:
                raise RuntimeError("AssetStore is closing")
            self._idle.clear()
        self._operations[task] = depth + 1
        try:
            yield
        finally:
            depth = self._operations[task] - 1
            if depth:
                self._operations[task] = depth
            else:
                del self._operations[task]
                if not self._operations:
                    self._idle.set()

    def _thread_barrier(self) -> _ThreadBarrier:
        return _ThreadBarrier(self)

    async def _thread(self, function, /, *args):
        barrier = self._thread_barrier()
        result = await barrier.run(function, *args)
        barrier.raise_if_cancelled()
        return result

    async def _remove_asset(self, asset_id: str, metadata: dict, barrier: _ThreadBarrier) -> None:
        async with self._lock:
            self._assets.pop(asset_id, None)
            await barrier.run(self._persist_metadata, protect_acquire=True)
        path = self._asset_path(metadata)
        if path is not None:
            await barrier.run(path.unlink, True, protect_acquire=True)

    async def _rollback_store(
        self,
        *,
        asset_id: str,
        metadata: dict,
        target: Path,
        metadata_added: bool,
        file_written: bool,
        barrier: _ThreadBarrier,
    ) -> None:
        if metadata_added:
            await self._remove_asset(asset_id, metadata, barrier)
        elif file_written:
            await barrier.run(target.unlink, True, protect_acquire=True)

    @staticmethod
    async def _finish_protected(coroutine, barrier: _ThreadBarrier) -> None:
        task = asyncio.create_task(coroutine)
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                barrier.cancelled = barrier.cancelled or exc
        task.result()

    def _inspect_image(self, data: bytes) -> tuple[str, str, int, int]:
        if len(data) > self.max_image_bytes:
            raise ImageTooLarge("Image exceeds the encoded byte limit")
        try:
            with Image.open(BytesIO(data)) as image:
                image_format = image.format
                width, height = image.size
                if image_format not in _FORMATS:
                    raise UnsafeMediaSource("Unsupported image format")
                if width <= 0 or height <= 0:
                    raise UnsafeMediaSource("Invalid image dimensions")
                pixels = width * height
                if pixels > self.max_pixels:
                    raise ImageTooLarge("Image exceeds the pixel limit")
                bands = max(1, len(image.getbands()))
                if pixels * bands > self.max_decoded_bytes:
                    raise ImageTooLarge("Image exceeds the decoded byte limit")
                image.load()
        except ImageTooLarge:
            raise
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise UnsafeMediaSource("Image bytes are invalid") from exc
        mime, extension = _FORMATS[image_format]
        return mime, extension, width, height

    async def store_bytes(
        self,
        data: bytes,
        *,
        scope: AssetScope,
        request_id: str,
        item_id: str | None = None,
    ) -> GeneratedImage:
        async with self._operation():
            return await self._store_bytes(
                data, scope=scope, request_id=request_id, item_id=item_id
            )

    async def _store_bytes(
        self,
        data: bytes,
        *,
        scope: AssetScope,
        request_id: str,
        item_id: str | None = None,
    ) -> GeneratedImage:
        self._ensure_open()
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if len(data) > self.max_total_bytes:
            raise ImageTooLarge("Image exceeds the total asset capacity")
        scope_data = _scope_dict(scope)
        barrier = self._thread_barrier()
        mime, extension, width, height = await barrier.run(self._inspect_image, data)
        barrier.raise_if_cancelled()
        asset_id = uuid.uuid4().hex
        relative_path = f"files/{asset_id}{extension}"
        target = self.root / relative_path
        created_at = time.time()
        metadata = {
            "scope": scope_data,
            "sha256": (await barrier.run(sha256, data)).hexdigest(),
            "request_id": request_id if isinstance(request_id, str) else "",
            "item_id": item_id if isinstance(item_id, str) else "",
            "created_at": created_at,
            "expires_at": created_at + self.ttl_seconds,
            "relative_path": relative_path,
            "mime_type": mime,
            "width": width,
            "height": height,
            "size": len(data),
        }

        def write_file() -> None:
            fd, name = tempfile.mkstemp(prefix=f".{asset_id}-", suffix=".tmp", dir=self._files)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(name, target)
            finally:
                try:
                    os.unlink(name)
                except FileNotFoundError:
                    pass

        file_written = False
        metadata_added = False
        try:
            await barrier.run(write_file)
            file_written = True
            barrier.raise_if_cancelled()
            async with self._lock:
                self._assets[asset_id] = metadata
                metadata_added = True
                await barrier.run(self._persist_metadata)
            barrier.raise_if_cancelled()
            await self._cleanup()
        except BaseException:
            await self._finish_protected(
                self._rollback_store(
                    asset_id=asset_id,
                    metadata=metadata,
                    target=target,
                    metadata_added=metadata_added,
                    file_written=file_written,
                    barrier=barrier,
                ),
                barrier,
            )
            raise
        return self._generated(asset_id, metadata)

    def _asset_path(self, metadata: dict) -> Path | None:
        relative = metadata.get("relative_path")
        if not isinstance(relative, str):
            return None
        candidate = self.root / relative
        try:
            resolved = candidate.resolve(strict=True)
            root = self.root.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        return resolved if resolved.is_relative_to(root) and resolved.is_file() else None

    def _generated(self, asset_id: str, metadata: dict) -> GeneratedImage:
        path = self._asset_path(metadata)
        if path is None:
            raise AssetNotFound()
        return GeneratedImage(
            path=str(path),
            mime_type=metadata["mime_type"],
            asset_id=asset_id,
            request_id=metadata.get("request_id", ""),
            width=metadata.get("width"),
            height=metadata.get("height"),
            raw={"sha256": metadata.get("sha256", ""), "item_id": metadata.get("item_id", "")},
        )

    async def resolve(self, asset_id: str, *, scope: AssetScope) -> Path:
        async with self._operation():
            return await self._resolve(asset_id, scope=scope)

    async def _resolve(self, asset_id: str, *, scope: AssetScope) -> Path:
        self._ensure_open()
        scope_data = _scope_dict(scope)
        if not isinstance(asset_id, str):
            raise AssetNotFound()
        async with self._lock:
            metadata = self._assets.get(asset_id)
            if metadata is None or metadata.get("scope") != scope_data:
                raise AssetNotFound()
            if metadata.get("expires_at", 0) <= time.time():
                raise AssetExpired()
            path = self._asset_path(metadata)
            if path is None:
                raise AssetNotFound()
            return path

    @asynccontextmanager
    async def lease(self, asset_id: str, *, scope: AssetScope) -> AsyncIterator[Path]:
        async with self._operation():
            async with self._lease(asset_id, scope=scope) as path:
                yield path

    @asynccontextmanager
    async def _lease(self, asset_id: str, *, scope: AssetScope) -> AsyncIterator[Path]:
        self._ensure_open()
        scope_data = _scope_dict(scope)
        async with self._lock:
            metadata = self._assets.get(asset_id)
            if metadata is None or metadata.get("scope") != scope_data:
                raise AssetNotFound()
            if metadata.get("expires_at", 0) <= time.time():
                raise AssetExpired()
            path = self._asset_path(metadata)
            if path is None:
                raise AssetNotFound()
            self._leases[asset_id] += 1
        try:
            yield path
        finally:
            async with self._lock:
                self._leases[asset_id] -= 1
                if self._leases[asset_id] <= 0:
                    del self._leases[asset_id]

    async def to_data_uri(self, asset_id: str, *, scope: AssetScope) -> str:
        """Materialize a same-scope asset without exposing unrestricted file reads."""
        async with self.lease(asset_id, scope=scope) as path:
            suffix = path.suffix.lower()
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }.get(suffix)
            if mime is None:
                raise UnsafeMediaSource("Asset image format is unsupported")
            data = await self._thread(path.read_bytes)
            encoded = await self._thread(base64.b64encode, data)
        return f"data:{mime};base64,{encoded.decode('ascii')}"

    async def import_reference(
        self,
        source: str | Path,
        *,
        scope: AssetScope,
        allowed_roots: Iterable[Path] = (),
    ) -> str:
        async with self._operation():
            return await self._import_reference(source, scope=scope, allowed_roots=allowed_roots)

    async def _import_reference(
        self,
        source: str | Path,
        *,
        scope: AssetScope,
        allowed_roots: Iterable[Path] = (),
    ) -> str:
        self._ensure_open()
        if isinstance(source, Path):
            candidate = source
        elif _is_data_uri(source):
            declared_mime, data = await self._thread(_decode_data_uri, source, self.max_image_bytes)
            image = await self._store_bytes(data, scope=scope, request_id="import")
            if image.mime_type != declared_mime:
                async with self._lock:
                    metadata = self._assets.pop(image.asset_id, None)
                    if metadata:
                        barrier = self._thread_barrier()
                        path = self._asset_path(metadata)
                        if path:
                            await barrier.run(path.unlink, True, protect_acquire=True)
                        await barrier.run(self._persist_metadata, protect_acquire=True)
                        barrier.raise_if_cancelled()
                raise UnsafeMediaSource("Declared media type does not match image bytes")
            return image.asset_id
        elif isinstance(source, str):
            parsed = urlsplit(source)
            if parsed.scheme in {"http", "https"}:
                image = await self._store_url(
                    source,
                    scope=scope,
                    request_id="import",
                    deadline=time.monotonic() + 30,
                )
                return image.asset_id
            if parsed.scheme == "file":
                if parsed.netloc not in {"", "localhost"}:
                    raise UnsafeMediaSource("Remote file URLs are not accepted")
                candidate = Path(unquote(parsed.path))
            elif parsed.scheme:
                raise UnsafeMediaSource("Unsupported media source")
            else:
                candidate = Path(source)
        else:
            raise TypeError("source must be a path or string")

        data = await self._thread(
            _read_local_reference, candidate, allowed_roots, self.max_image_bytes
        )
        return (await self._store_bytes(data, scope=scope, request_id="import")).asset_id

    async def store_url(
        self,
        url: str,
        *,
        scope: AssetScope,
        request_id: str,
        item_id: str | None = None,
        deadline: float,
    ) -> GeneratedImage:
        async with self._operation():
            return await self._store_url(
                url,
                scope=scope,
                request_id=request_id,
                item_id=item_id,
                deadline=deadline,
            )

    async def _store_url(
        self,
        url: str,
        *,
        scope: AssetScope,
        request_id: str,
        item_id: str | None = None,
        deadline: float,
    ) -> GeneratedImage:
        self._ensure_open()
        if self._media is None:
            raise UnsafeMediaSource("Public URL image inputs are disabled")
        data = await self._media.fetch(url, deadline=deadline)
        return await self._store_bytes(data, scope=scope, request_id=request_id, item_id=item_id)

    async def cleanup(self) -> None:
        async with self._operation():
            await self._cleanup()

    async def _cleanup(self) -> None:
        self._ensure_open()
        now = time.time()
        removed: list[Path] = []
        barrier = self._thread_barrier()
        try:
            async with self._lock:
                candidates = sorted(
                    self._assets.items(), key=lambda item: item[1].get("created_at", 0)
                )
                total = sum(
                    metadata.get("size", 0)
                    for _, metadata in candidates
                    if isinstance(metadata.get("size"), int)
                )
                changed = False
                for asset_id, metadata in candidates:
                    expired = metadata.get("expires_at", 0) <= now
                    oversized = total > self.max_total_bytes
                    if self._leases.get(asset_id, 0) or not (expired or oversized):
                        continue
                    path = self._asset_path(metadata)
                    if path:
                        removed.append(path)
                    size = metadata.get("size", 0)
                    total -= size if isinstance(size, int) else 0
                    self._assets.pop(asset_id, None)
                    changed = True
                if changed:
                    await barrier.run(self._persist_metadata, protect_acquire=True)
            for path in removed:
                await barrier.run(path.unlink, True, protect_acquire=True)
        finally:
            barrier.raise_if_cancelled()

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            current = asyncio.current_task()
            if current in self._operations:
                raise RuntimeError("AssetStore cannot close from an active operation")
            self._closing = True
            # Cancellation while operations are still active leaves the store in
            # closing state. A later close call resumes from the same safe point.
            await self._idle.wait()
            barrier = self._thread_barrier()
            async with self._lock:
                await barrier.run(self._persist_metadata, protect_acquire=True)
            if self._media is not None:
                close_task = asyncio.create_task(self._media.close())
                while not close_task.done():
                    try:
                        await asyncio.shield(close_task)
                    except asyncio.CancelledError as exc:
                        barrier.cancelled = barrier.cancelled or exc
                close_task.result()
            self._closed = True
            barrier.raise_if_cancelled()
