"""In-memory credential lifecycle with durable single-flight refresh."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace

from .errors import (
    AuthorizationChanged,
    CredentialPersistenceError,
    ReauthorizationRequired,
    ServiceClosed,
)
from .models import TokenSnapshot
from .token_store import TokenStore

RefreshFunction = Callable[[TokenSnapshot], Awaitable[TokenSnapshot]]


class OAuthService:
    def __init__(self, store: TokenStore, refresh_fn: RefreshFunction | None, *, clock=time.time):
        self._store = store
        self._refresh_fn = refresh_fn
        self._clock = clock
        self._snapshot: TokenSnapshot | None = None
        self._epoch = 0
        self._binding_generation = 0
        self._opened = False
        self._closed = False
        self._blocked = False
        self._reauth_required = False
        self._state_lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[TokenSnapshot] | None = None
        self._retired_tasks: set[asyncio.Task[TokenSnapshot]] = set()
        self._close_task: asyncio.Task[None] | None = None

    @property
    def binding_generation(self) -> int:
        return self._binding_generation

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def status(self) -> str:
        if self._closed:
            return "closed"
        if self._blocked:
            return "persistence_error"
        if self._reauth_required:
            return "reauth_required"
        return "authorized" if self._snapshot is not None else "unbound"

    async def open(self) -> None:
        async with self._state_lock:
            if self._opened:
                return
            if self._closed:
                raise ServiceClosed()
            await self._store.open()
            try:
                loaded = await self._store.load()
            except BaseException:
                await self._store.close()
                raise
            if loaded is not None:
                self._snapshot = loaded
                self._epoch = loaded.epoch
            self._opened = True

    def snapshot(self) -> TokenSnapshot | None:
        return self._snapshot

    async def get_token(
        self, *, force_refresh: bool = False, rejected_version: int | None = None
    ) -> TokenSnapshot:
        async with self._state_lock:
            self._require_active()
            if self._blocked:
                raise CredentialPersistenceError()
            if self._reauth_required:
                raise ReauthorizationRequired()
            current = self._snapshot
            if current is None:
                raise ReauthorizationRequired()
            if rejected_version is not None and current.version != rejected_version:
                return current
            expired = current.expires_at is not None and current.expires_at <= self._clock() + 120.0
            if not force_refresh and not expired:
                return current
            if self._refresh_task is None:
                self._refresh_task = asyncio.create_task(
                    self._refresh_once(current, self._binding_generation)
                )
            task = self._refresh_task
        try:
            return await asyncio.shield(task)
        finally:
            async with self._state_lock:
                if self._refresh_task is task and task.done():
                    self._refresh_task = None

    async def bind(self, token: TokenSnapshot, *, expected_epoch: int) -> None:
        await _shield_operation(self._bind(token, expected_epoch))

    async def disconnect(self) -> None:
        await _shield_operation(self._disconnect())

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await _shield_task(self._close_task)

    async def _refresh_once(self, previous: TokenSnapshot, generation: int) -> TokenSnapshot:
        if self._refresh_fn is None:
            raise ReauthorizationRequired()
        try:
            candidate = await self._refresh_fn(previous)
        except ReauthorizationRequired:
            async with self._state_lock:
                if (
                    self._binding_generation == generation
                    and self._epoch == previous.epoch
                    and self._snapshot is not None
                    and self._snapshot.version == previous.version
                ):
                    try:
                        await self._store.clear()
                    except CredentialPersistenceError:
                        self._blocked = True
                    self._reauth_required = True
                    self._snapshot = None
            raise
        if not isinstance(candidate, TokenSnapshot):
            raise ReauthorizationRequired()
        refreshed = replace(
            candidate,
            slot=previous.slot,
            refresh_token=candidate.refresh_token or previous.refresh_token,
            client_id=previous.client_id,
            user_id=previous.user_id or candidate.user_id,
            version=previous.version + 1,
            epoch=previous.epoch,
        )
        async with self._state_lock:
            self._require_active()
            if (
                self._binding_generation != generation
                or self._epoch != previous.epoch
                or self._snapshot is None
                or self._snapshot.version != previous.version
            ):
                raise AuthorizationChanged()
            try:
                await self._store.commit(refreshed)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                    raise
                self._blocked = True
                self._snapshot = None
                try:
                    await self._store.clear()
                except CredentialPersistenceError:
                    pass
                raise CredentialPersistenceError() from None
            self._snapshot = refreshed
            return refreshed

    async def _bind(self, token: TokenSnapshot, expected_epoch: int) -> None:
        async with self._state_lock:
            self._require_active()
            if expected_epoch != self._epoch:
                raise AuthorizationChanged()
            next_version = max(token.version, (self._snapshot.version + 1) if self._snapshot else 1)
            candidate = replace(token, version=next_version, epoch=self._epoch)
            try:
                await self._store.commit(candidate)
            except CredentialPersistenceError:
                self._blocked = True
                self._snapshot = None
                try:
                    await self._store.clear()
                except CredentialPersistenceError:
                    pass
                raise
            self._retire_refresh_locked()
            self._snapshot = candidate
            self._binding_generation += 1
            self._blocked = False
            self._reauth_required = False

    async def _disconnect(self) -> None:
        async with self._state_lock:
            self._require_active()
            self._retire_refresh_locked()
            self._epoch += 1
            self._binding_generation += 1
            self._snapshot = None
            self._blocked = False
            self._reauth_required = False
            await self._store.clear()

    async def _close(self) -> None:
        async with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._opened = False
            self._retire_refresh_locked()
            self._epoch += 1
            self._binding_generation += 1
            self._snapshot = None
            pending = tuple(self._retired_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        async with self._state_lock:
            await self._store.close()

    def _retire_refresh_locked(self) -> None:
        task = self._refresh_task
        self._refresh_task = None
        if task is None:
            return
        if task.done():
            _consume_task(task)
            return
        self._retired_tasks.add(task)
        task.add_done_callback(self._retired_done)

    def _retired_done(self, task: asyncio.Task[TokenSnapshot]) -> None:
        self._retired_tasks.discard(task)
        _consume_task(task)

    def _require_active(self) -> None:
        if self._closed or not self._opened:
            raise ServiceClosed()


async def _shield_operation(operation) -> None:
    task = asyncio.create_task(operation)
    await _shield_task(task)


async def _shield_task(task: asyncio.Task) -> None:
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(task)
        except Exception:
            pass
        raise


def _consume_task(task: asyncio.Task) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
