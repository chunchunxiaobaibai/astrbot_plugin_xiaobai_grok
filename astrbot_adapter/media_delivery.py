"""Durable bounded outbox: retry delivery without regenerating images."""

import asyncio
import copy
import json
import os
import tempfile
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from ..grok_oauth.errors import (
    AssetNotFound,
    Busy,
    GrokOAuthError,
    OutcomeUnknown,
    ProtocolError,
    ServiceClosed,
)
from .compat import Image, MessageChain
from .runtime import _finish_owned_operation


class MediaDelivery:
    def __init__(self, path, assets, *, max_records=256):
        self.path = Path(path)
        self.assets = assets
        self.max_records = max_records
        self._lock = asyncio.Lock()
        self._operation_locks = {}
        self._tasks = set()
        self.closed = False
        self._close_task = None
        if self.path.exists():
            try:
                self.records = json.loads(self.path.read_text())
                if not isinstance(self.records, dict) or len(self.records) > max_records:
                    raise ValueError()
                for record in self.records.values():
                    if not isinstance(record, dict) or record.get("status") not in {
                        "pending",
                        "generated",
                        "sending",
                        "sent",
                        "failed",
                        "unknown",
                        "error",
                    }:
                        raise ValueError()
                    if record["status"] in {"pending", "sending"}:
                        record["status"] = "unknown"
            except (ValueError, OSError):
                raise ProtocolError("Outbox state cannot be loaded") from None
        else:
            self.records = {}

    async def _commit(self, records):
        def write():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".outbox-", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(records, stream, ensure_ascii=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, self.path)
                dirfd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dirfd)
                finally:
                    os.close(dirfd)
                self.records = records
            finally:
                if os.path.exists(name):
                    os.unlink(name)

        try:
            await _finish_owned_operation(asyncio.to_thread(write))
        except OSError:
            raise ProtocolError("Outbox could not be persisted") from None

    async def _update(self, key, **values):
        async with self._lock:
            records = copy.deepcopy(self.records)
            records[key].update(values)
            await self._commit(records)

    @staticmethod
    def _public(record):
        result = {
            "status": record["status"],
            "operation_id": record["operation_id"],
            "asset_ids": list(record.get("asset_ids", [])),
            "count": len(record.get("asset_ids", [])),
        }
        if record.get("error"):
            result["error"] = record["error"]
        return result

    @asynccontextmanager
    async def _active_operation(self):
        if self.closed:
            raise ServiceClosed()
        task = asyncio.current_task()
        self._tasks.add(task)
        try:
            yield
        finally:
            self._tasks.discard(task)

    async def run(self, key, *, scope, event, generate):
        async with self._active_operation():
            return await self._run(key, scope=scope, event=event, generate=generate)

    async def _run(self, key, *, scope, event, generate):
        if self.closed:
            raise ServiceClosed()
        async with self._lock:
            created = key not in self.records
            if created:
                records = copy.deepcopy(self.records)
                if len(records) >= self.max_records:
                    candidates = [
                        k
                        for k, r in records.items()
                        if r["status"] in {"sent", "error"}
                        and not (k in self._operation_locks and self._operation_locks[k].locked())
                    ]
                    if not candidates:
                        raise Busy("Image outbox is full")
                    victim = min(candidates, key=lambda k: records[k]["created_at"])
                    records.pop(victim)
                    self._operation_locks.pop(victim, None)
                records[key] = {
                    "operation_id": uuid.uuid4().hex,
                    "scope": asdict(scope),
                    "status": "pending",
                    "asset_ids": [],
                    "created_at": time.time(),
                }
                await self._commit(records)
            lock = self._operation_locks.setdefault(key, asyncio.Lock())
        current = asyncio.current_task()
        self._tasks.add(current)
        try:
            async with lock:
                if self.closed:
                    raise ServiceClosed()
                record = self.records[key]
                if record["scope"] != asdict(scope):
                    raise AssetNotFound()
                if created:
                    try:
                        images = await generate()
                    except asyncio.CancelledError:
                        await self._update(key, status="unknown", error="OutcomeUnknown")
                        raise
                    except GrokOAuthError as error:
                        await self._update(
                            key,
                            status="unknown" if isinstance(error, OutcomeUnknown) else "error",
                            error=error.code,
                            asset_ids=[image.asset_id for image in error.assets],
                        )
                        return self._public(self.records[key])
                    try:
                        await self._update(
                            key, status="generated", asset_ids=[image.asset_id for image in images]
                        )
                    except asyncio.CancelledError:
                        await _finish_owned_operation(self._retain_generated(key, images))
                        raise
                    except GrokOAuthError:
                        return await self._retain_generated(key, images)
                if self.records[key]["status"] in {"generated", "failed"}:
                    await self._send(key, scope=scope, event=event)
                return self._public(self.records[key])
        finally:
            self._tasks.discard(current)

    async def _retain_generated(self, key, images):
        # The upstream operation already succeeded. Even if disk remains broken,
        # retain the asset handles in memory and never return to generation.
        async with self._lock:
            records = copy.deepcopy(self.records)
            records[key].update(
                status="unknown",
                asset_ids=[image.asset_id for image in images],
                error="OutboxPersistenceError",
            )
            self.records = records
            try:
                await self._commit(records)
            except GrokOAuthError:
                pass
            return self._public(self.records[key])

    async def _send(self, key, *, scope, event):
        async with AsyncExitStack() as stack:
            chain = []
            for asset_id in self.records[key]["asset_ids"]:
                path = await stack.enter_async_context(self.assets.lease(asset_id, scope=scope))
                chain.append(Image.fromFileSystem(str(path)))
            if not chain:
                raise AssetNotFound("No stored images for this operation")
            await self._update(key, status="sending")
            try:
                accepted = await event.send(MessageChain(chain=chain, type="tool_direct_result"))
            except asyncio.CancelledError:
                await self._update(key, status="unknown", error="DeliveryOutcomeUnknown")
                raise
            except Exception:
                await self._update(key, status="unknown", error="DeliveryOutcomeUnknown")
            else:
                await self._update(
                    key,
                    status="failed" if accepted is False else "sent",
                    error="DeliveryRejected" if accepted is False else None,
                )

    async def resend(self, operation_id, *, scope, event):
        async with self._active_operation():
            return await self._resend(operation_id, scope=scope, event=event)

    async def _resend(self, operation_id, *, scope, event):
        if self.closed:
            raise ServiceClosed()
        async with self._lock:
            found = [
                (key, record)
                for key, record in self.records.items()
                if record["operation_id"] == operation_id and record["scope"] == asdict(scope)
            ]
            if len(found) != 1:
                raise AssetNotFound()
            key, _ = found[0]
            lock = self._operation_locks.setdefault(key, asyncio.Lock())
        current = asyncio.current_task()
        self._tasks.add(current)
        try:
            async with lock:
                if self.closed:
                    raise ServiceClosed()
                await self._send(key, scope=scope, event=event)
                return self._public(self.records[key])
        finally:
            self._tasks.discard(current)

    async def close(self):
        self.closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="grok-outbox-close")
        await _finish_owned_operation(self._close_task)

    async def _close(self):
        tasks = [task for task in self._tasks if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
