"""Bounded, content-free transport diagnostics through AstrBot and private JSONL."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import stat
from datetime import UTC, datetime
from pathlib import Path

from astrbot.api import logger

from ..grok_oauth.diagnostic_fields import (
    DIAGNOSTIC_ERROR_CODES,
    DIAGNOSTIC_ERROR_PARAMS,
    DIAGNOSTIC_REQUEST_KINDS,
)
from ..grok_oauth.oauth import OAUTH_OUTCOMES, OAUTH_REASONS, OAUTH_STAGES, safe_oauth_diagnostic

_TEMPLATE = "grok_oauth_transport_diag %s"
_MAX_BYTES = 1024 * 1024
_QUEUE_SIZE = 128
_STOP = object()
_ENUMS = {
    "event": {"provider_response", "oauth_request"},
    "operation": {"json", "stream"},
    "mode": {"chat", "stream", "unknown"},
    "outcome": {"success", "failure", "cancelled", "timeout", "scope_returned", "stream_exhausted"}
    | OAUTH_OUTCOMES,
    "oauth_stage": OAUTH_STAGES,
    "oauth_reason": OAUTH_REASONS,
    "phase": {
        "validation",
        "authentication",
        "authentication_refresh",
        "send_or_headers",
        "response",
        "read_body",
        "stream_body",
        "connect",
        "decode",
        "retry_backoff",
        "error_body",
    },
    "termination_reason": {"", "request_deadline", "transport_timeout", "external_cancel"},
    "request_kind": DIAGNOSTIC_REQUEST_KINDS,
    "upstream_error_code": DIAGNOSTIC_ERROR_CODES,
    "upstream_error_param": DIAGNOSTIC_ERROR_PARAMS,
    "connection_error_kind": {"dns", "tls", "proxy", "connect_timeout", "connect", "other"},
    "error_body_state": {
        "empty",
        "json",
        "unparseable",
        "too_large",
        "read_failed",
        "deadline_skipped",
    },
}
_CAUSES = {
    "",
    "CancelledError",
    "TimeoutError",
    "ReadTimeout",
    "WriteTimeout",
    "ConnectTimeout",
    "PoolTimeout",
    "ConnectError",
    "ProxyError",
    "ReadError",
    "WriteError",
    "RemoteProtocolError",
    "LocalProtocolError",
    "ProtocolError",
    "DeviceCodeExpired",
    "AuthorizationDenied",
    "AuthorizationChanged",
    "ClientNotEligible",
    "OutcomeUnknown",
    "PaymentRequired",
    "PermissionDenied",
    "RateLimited",
    "ReauthorizationRequired",
    "UnsafeTarget",
    "InvalidRequest",
    "ServiceClosed",
    "HTTPStatusError",
    "JSONDecodeError",
    "ValueError",
    "RuntimeError",
    "OSError",
    "DecodingError",
}
_IDENTIFIERS = {"request_id", "provider_id", "configured_model", "actual_model", "response_id"}
_NUMBERS = {
    "elapsed_ms",
    "remaining_ms",
    "body_bytes",
    "connect_retries",
    "auth_retries",
    "retry_backoff_ms",
    "poll_retries",
    "retry_in_ms",
}
_BOOLS = {"headers_received", "body_exhausted"}


def _safe_record(value: str) -> dict | None:
    if not isinstance(value, str) or len(value) > 8192:
        return None
    raw = json.loads(value)
    if not isinstance(raw, dict):
        return None
    result = {}
    for key, item in raw.items():
        if key in _ENUMS and isinstance(item, str) and item in _ENUMS[key]:
            result[key] = item
        elif key == "trace_id" and isinstance(item, str) and re.fullmatch(r"[0-9a-fA-F]{32}", item):
            result[key] = item
        elif key == "cause" and isinstance(item, str):
            result[key] = item if item in _CAUSES else "OtherError"
        elif key in _NUMBERS:
            if item is None:
                result[key] = None
            elif type(item) in {int, float} and math.isfinite(item) and 0 <= item <= 10**15:
                result[key] = item
        elif key == "status" and (item is None or type(item) is int and 100 <= item <= 599):
            result[key] = item
        elif key in _BOOLS and type(item) is bool:
            result[key] = item
        elif key in _IDENTIFIERS and isinstance(item, str):
            if (
                re.fullmatch(r"[A-Za-z0-9_./:-]{1,128}", item)
                and not item.startswith(("/", "eyJ", "xai-", "sk-", "ghp_", "github_pat_"))
                and "://" not in item
                and not re.match(r"^[A-Za-z]:/", item)
            ):
                result[key] = item
    return result or None


class TransportDiagnostics:
    """Synchronous logger-shaped input; asynchronous bounded private-file output."""

    def __init__(self, root: Path, *, native_enabled: bool, file_enabled: bool, native_logger=None):
        self.root = Path(root).absolute()
        self.native_enabled = native_enabled is True
        self.file_enabled = file_enabled is True
        self._logger = logger if native_logger is None else native_logger
        self._queue = asyncio.Queue(maxsize=_QUEUE_SIZE)
        self._task = None
        self._closed = False
        self._warned = False
        self._file_failed = False
        self._close_task = None

    def _warn(self):
        if self._warned:
            return
        self._warned = True
        try:
            self._logger.warning("Grok OAuth transport diagnostic file unavailable or queue full.")
        except Exception:
            pass

    def info(self, template, value):
        if self._closed or not (self.native_enabled or self.file_enabled) or template != _TEMPLATE:
            return
        try:
            record = _safe_record(value)
            if record is None:
                return
            serialized = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
        except Exception:
            return
        if self.native_enabled:
            try:
                self._logger.info(_TEMPLATE, serialized)
            except Exception:
                pass
        if self.file_enabled and not self._file_failed:
            try:
                loop = asyncio.get_running_loop()
                if self._task is None:
                    self._task = loop.create_task(self._write_loop(), name="grok-oauth-diagnostics")
                record["timestamp_utc"] = datetime.now(UTC).isoformat(timespec="milliseconds")
                file_record = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
                self._queue.put_nowait((file_record + "\n").encode("utf-8"))
            except (RuntimeError, asyncio.QueueFull):
                self._warn()

    async def _write_loop(self):
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    return
                if not self._file_failed:
                    try:
                        await asyncio.to_thread(self._append, item)
                    except Exception:
                        self._file_failed = True
                        self._warn()
            finally:
                self._queue.task_done()

    def _append(self, data: bytes):
        # Check existing ancestors too: resolving them would hide a symlink.
        for part in (self.root, *self.root.parents):
            if part.is_symlink():
                raise OSError("unsafe diagnostic directory")
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        debug_fd = None
        try:
            try:
                os.mkdir("debug", mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            debug_fd = os.open(
                "debug", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
            )
            os.fchmod(debug_fd, 0o700)
            for name in ("transport.jsonl", "transport.jsonl.1"):
                try:
                    existing = os.stat(name, dir_fd=debug_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1:
                    raise OSError("unsafe diagnostic file")
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
            file_fd = os.open("transport.jsonl", flags, 0o600, dir_fd=debug_fd)
            try:
                opened = os.fstat(file_fd)
                if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                    raise OSError("unsafe diagnostic file")
                os.fchmod(file_fd, 0o600)
                if opened.st_size + len(data) > _MAX_BYTES:
                    os.close(file_fd)
                    file_fd = None
                    os.replace(
                        "transport.jsonl",
                        "transport.jsonl.1",
                        src_dir_fd=debug_fd,
                        dst_dir_fd=debug_fd,
                    )
                    file_fd = os.open("transport.jsonl", flags | os.O_EXCL, 0o600, dir_fd=debug_fd)
                offset = 0
                while offset < len(data):
                    written = os.write(file_fd, data[offset:])
                    if written <= 0:
                        raise OSError("diagnostic write failed")
                    offset += written
            finally:
                if file_fd is not None:
                    os.close(file_fd)
        finally:
            if debug_fd is not None:
                os.close(debug_fd)
            os.close(root_fd)

    async def _finish_close(self):
        if not self._task.done():
            await self._queue.put(_STOP)
        await self._task

    async def close(self):
        self._closed = True
        if self._task is not None:
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._finish_close())
            await asyncio.shield(self._close_task)


_NETWORK_PROXY_HINT = (
    "无法连接 xAI 授权服务（auth.x.ai）。国内网络请在插件配置填写 proxy，"
    "例如 http://127.0.0.1:7890，保存并重载后再试。"
)
_OAUTH_REASON_MESSAGES = {
    "dns": "无法解析 xAI 授权域名，请检查 DNS 或代理。",
    "tls": "与 xAI 授权服务的 TLS 连接失败，请检查代理或系统证书。",
    "proxy": "出站代理不可用，请检查插件配置中的 proxy 地址是否可访问。",
    "connect_timeout": _NETWORK_PROXY_HINT,
    "connect": _NETWORK_PROXY_HINT,
    "network": _NETWORK_PROXY_HINT,
    "read_timeout": "读取 xAI 授权响应超时，请稍后重试或检查代理。",
    "write_timeout": "向 xAI 授权服务发送请求超时，请稍后重试或检查代理。",
    "pool_timeout": "代理连接池繁忙，请稍后重试。",
}


def user_facing_oauth_error(error):
    """Stable, credential-free text for plugin page and command replies."""
    from ..grok_oauth.errors import (
        AuthorizationDenied,
        ClientNotEligible,
        DeviceCodeExpired,
        InvalidRequest,
        PermissionDenied,
        ProtocolError,
        RateLimited,
        ReauthorizationRequired,
        ServiceClosed,
    )

    if isinstance(error, ProtocolError):
        reason = (getattr(error, "oauth_diagnostic", None) or {}).get("oauth_reason")
        return _OAUTH_REASON_MESSAGES.get(reason, _NETWORK_PROXY_HINT)
    if isinstance(error, DeviceCodeExpired):
        return "认证码已过期，请重新开始授权。"
    if isinstance(error, AuthorizationDenied):
        return "官方服务未批准本次授权，可以重新尝试。"
    if isinstance(error, ReauthorizationRequired):
        return "账号授权已失效，请重新连接。"
    if isinstance(error, ClientNotEligible):
        return "当前 OAuth 客户端未被批准，请核对客户端 ID。"
    if isinstance(error, RateLimited):
        return "请求过于频繁，请稍后再试。"
    if isinstance(error, PermissionDenied):
        return "当前账号无权完成此操作。"
    if isinstance(error, ServiceClosed):
        return "插件正在关闭，请稍后重试。"
    if isinstance(error, InvalidRequest):
        return str(error) if str(error) and str(error) != "InvalidRequest" else "请求参数无效。"
    return _NETWORK_PROXY_HINT


def format_oauth_diagnostic(value):
    """Short safe details for private administrator command replies."""
    detail = safe_oauth_diagnostic(value)
    if not detail.get("trace_id"):
        return ""
    parts = ["阶段=" + detail.get("oauth_stage", "unknown")]
    if detail.get("oauth_reason"):
        parts.append("原因=" + detail["oauth_reason"])
    if detail.get("cause"):
        parts.append("类型=" + detail["cause"])
    if detail.get("status"):
        parts.append("HTTP=" + str(detail["status"]))
    if detail.get("outcome") == "retrying":
        parts.append("正在重试")
    elif detail.get("outcome") == "pending":
        parts.append("等待授权")
    parts.append("关联编号=" + detail["trace_id"])
    return "；" + "；".join(parts)
