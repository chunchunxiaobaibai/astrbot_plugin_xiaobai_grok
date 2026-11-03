"""Process-owned, crash-safe storage for OAuth credential snapshots."""

from __future__ import annotations

import errno
import json
import math
import os
import tempfile
from pathlib import Path

from filelock import FileLock, Timeout

from .errors import CredentialPersistenceError, CredentialStoreInUse, ServiceClosed
from .models import TokenSnapshot


class TokenStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = FileLock(str(self.path.with_name(f".{self.path.name}.lock")))
        self._opened = False

    async def open(self) -> None:
        if self._opened:
            return
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
            self._lock.acquire(timeout=0)
        except Timeout:
            raise CredentialStoreInUse() from None
        except OSError:
            raise CredentialPersistenceError() from None
        self._opened = True

    async def load(self) -> TokenSnapshot | None:
        self._require_open()
        if not self.path.exists():
            return None
        try:
            raw = self.path.read_bytes()
            if len(raw) > 64 * 1024:
                raise ValueError("oversize")
            data = json.loads(
                raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("constant"))
            )
            return _decode_snapshot(data)
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            raise CredentialPersistenceError() from None

    async def commit(self, snapshot: TokenSnapshot) -> None:
        self._require_open()
        data = _encode_snapshot(snapshot)
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            os.chmod(self.path, 0o600)
            _sync_directory(self.path.parent)
        except (OSError, TypeError, ValueError):
            raise CredentialPersistenceError() from None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    async def clear(self) -> None:
        self._require_open()
        try:
            self.path.unlink(missing_ok=True)
            _sync_directory(self.path.parent)
        except OSError:
            raise CredentialPersistenceError() from None

    async def close(self) -> None:
        if not self._opened:
            return
        self._opened = False
        self._lock.release()

    def _require_open(self) -> None:
        if not self._opened:
            raise ServiceClosed()


def _valid_text(value: object, *, allow_empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or (not allow_empty and not value)
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ValueError("invalid text")
    return value


def _valid_counter(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid counter")
    return value


def _valid_expiry(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid expiry")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("invalid expiry")
    return result


def _encode_snapshot(snapshot: TokenSnapshot) -> dict[str, object]:
    if not isinstance(snapshot, TokenSnapshot):
        raise CredentialPersistenceError()
    try:
        return {
            "slot": _valid_text(snapshot.slot),
            "access_token": _valid_text(snapshot.access_token),
            "refresh_token": _valid_text(snapshot.refresh_token),
            "expires_at": _valid_expiry(snapshot.expires_at),
            "scope": _valid_text(snapshot.scope, allow_empty=True),
            "client_id": _valid_text(snapshot.client_id),
            "version": _valid_counter(snapshot.version),
            "epoch": _valid_counter(snapshot.epoch),
            "user_id": _valid_text(snapshot.user_id) if snapshot.user_id is not None else None,
        }
    except ValueError:
        raise CredentialPersistenceError() from None


def _decode_snapshot(data: object) -> TokenSnapshot:
    if not isinstance(data, dict) or set(data) - {"user_id"} != {
        "slot",
        "access_token",
        "refresh_token",
        "expires_at",
        "scope",
        "client_id",
        "version",
        "epoch",
    }:
        raise ValueError("invalid schema")
    return TokenSnapshot(
        slot=_valid_text(data["slot"]),
        access_token=_valid_text(data["access_token"]),
        refresh_token=_valid_text(data["refresh_token"]),
        expires_at=_valid_expiry(data["expires_at"]),
        scope=_valid_text(data["scope"], allow_empty=True),
        client_id=_valid_text(data["client_id"]),
        version=_valid_counter(data["version"]),
        epoch=_valid_counter(data["epoch"]),
        user_id=_valid_text(data["user_id"]) if data.get("user_id") is not None else None,
    )


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
            raise
    finally:
        os.close(descriptor)
