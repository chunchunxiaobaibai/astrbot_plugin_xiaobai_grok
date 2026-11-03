"""OAuth-only HTTP transport pinned to the xAI inference origin."""

from __future__ import annotations

import asyncio
import json as json_module
import math
import random
import re
import socket
import ssl
import time
import uuid
from contextlib import asynccontextmanager
from hashlib import sha256
from urllib.parse import unquote, urlsplit

import httpx

from .diagnostic_fields import request_kind, safe_error_metadata
from .errors import (
    AuthorizationChanged,
    ClientNotEligible,
    OutcomeUnknown,
    PaymentRequired,
    PermissionDenied,
    ProtocolError,
    RateLimited,
    ReauthorizationRequired,
    UnsafeTarget,
)
from .models import RequestPolicy
from .oauth import _access_identity_subject
from .version import USER_AGENT

API_ORIGIN = "https://api.x.ai"
API_BASE = f"{API_ORIGIN}/v1"


def validate_connect_timeout(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 60:
        raise ValueError("connect_timeout must be between 0 and 60 seconds")
    return float(value)


class _DiagnosticTrace:
    """A fixed, content-free record for one inference request."""

    def __init__(
        self, enabled: bool, logger, operation: str, policy: RequestPolicy, *, kind="other"
    ):
        self.enabled = enabled
        self.logger = logger
        self.operation = operation
        self.trace_id = uuid.uuid4().hex
        self.request_kind = kind
        self.connection_error_kind = ""
        self.retry_backoff_ms = 0
        self.error_body_state = ""
        self.error_metadata = safe_error_metadata(b"")
        try:
            self.started = time.monotonic()
            self.deadline = getattr(policy, "deadline", None)
        except Exception:
            self.started = None
            self.deadline = None
        self.phase = "validation"
        self.cause = ""
        self.cancelled_at = None
        self.status = None
        self.request_id = ""
        self.headers_received = False
        self.body_bytes = 0
        self.body_exhausted = False
        self.connect_retries = 0
        self.auth_retries = 0

    def begin_attempt(self, phase: str):
        self.phase = phase
        self.cause = ""
        self.cancelled_at = None
        self.status = None
        self.request_id = ""
        self.headers_received = False
        self.body_bytes = 0
        self.body_exhausted = False
        self.error_body_state = ""
        self.error_metadata = safe_error_metadata(b"")

    def response(self, response: httpx.Response):
        self.phase = "response"
        self.cause = ""
        self.status = response.status_code
        self.headers_received = True
        request_id = _request_id(response)
        if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id):
            self.request_id = request_id

    def failure(self, phase: str, exc: BaseException):
        self.phase = phase
        self.cause = type(exc).__name__
        if isinstance(exc, httpx.RequestError):
            self.connection_error_kind = _connection_error_kind(exc)
        if isinstance(exc, asyncio.CancelledError) and self.cancelled_at is None:
            self.cancelled_at = time.monotonic()

    def emit(self, outcome: str, exc: BaseException | None = None):
        if not self.enabled:
            return
        try:
            now = time.monotonic()
            remaining = self.deadline - now if isinstance(self.deadline, (int, float)) else None
            cause = self.cause or (type(exc).__name__ if exc else "")
            termination_reason = ""
            cancellation_remaining = (
                self.deadline - self.cancelled_at
                if self.cancelled_at is not None and isinstance(self.deadline, (int, float))
                else remaining
            )
            if outcome in {"cancelled", "failure"}:
                if cause == "TimeoutError" or (
                    cause == "CancelledError"
                    and cancellation_remaining is not None
                    and math.isfinite(cancellation_remaining)
                    and cancellation_remaining <= 0
                ):
                    outcome = "timeout"
                    termination_reason = "request_deadline"
                elif cause in {"ReadTimeout", "WriteTimeout", "ConnectTimeout", "PoolTimeout"}:
                    outcome = "timeout"
                    termination_reason = "transport_timeout"
                elif cause == "CancelledError":
                    termination_reason = "external_cancel"
            record = {
                "operation": self.operation,
                "trace_id": self.trace_id,
                "request_kind": self.request_kind,
                "connection_error_kind": self.connection_error_kind,
                "retry_backoff_ms": self.retry_backoff_ms,
                "error_body_state": self.error_body_state,
                **self.error_metadata,
                "outcome": outcome,
                "phase": self.phase,
                "cause": cause,
                "termination_reason": termination_reason,
                "elapsed_ms": round((now - self.started) * 1000)
                if self.started is not None
                else None,
                "remaining_ms": round(max(0, remaining) * 1000)
                if remaining is not None and math.isfinite(remaining)
                else None,
                "status": self.status,
                "request_id": self.request_id,
                "headers_received": self.headers_received,
                "body_bytes": self.body_bytes,
                "body_exhausted": self.body_exhausted,
                "connect_retries": self.connect_retries,
                "auth_retries": self.auth_retries,
            }
            self.logger.info("grok_oauth_transport_diag %s", json_module.dumps(record))
        except Exception:
            pass


class AuthorizedHttp:
    def __init__(
        self,
        client: httpx.AsyncClient,
        oauth,
        *,
        max_json_bytes: int = 140 * 1024 * 1024,
        diagnostic_logging: bool = False,
        diagnostic_logger=None,
        connect_timeout: float = 10.0,
    ):
        self._connect_timeout = validate_connect_timeout(connect_timeout)
        if not isinstance(max_json_bytes, int) or max_json_bytes < 1:
            raise ValueError("max_json_bytes must be positive")
        self._client = client
        self._oauth = oauth
        self._max_json_bytes = max_json_bytes
        self._diagnostic_logging = diagnostic_logging is True and diagnostic_logger is not None
        self._diagnostic_logger = diagnostic_logger

    def _transport_timeout(self, policy):
        remaining = _remaining(policy)
        return httpx.Timeout(remaining, connect=min(self._connect_timeout, remaining))

    def log_completion(self, *, mode: str, provider_id, configured_model, result):
        """Record only validated identifiers from a completed provider response."""
        if not self._diagnostic_logging:
            return
        try:

            def safe(value):
                return (
                    value
                    if isinstance(value, str)
                    and not value.startswith("/")
                    and "://" not in value
                    and not re.match(r"^[A-Za-z]:/", value)
                    and re.fullmatch(r"[A-Za-z0-9_./:-]{1,128}", value)
                    else ""
                )

            record = {
                "event": "provider_response",
                "mode": mode if mode in {"chat", "stream"} else "unknown",
                "provider_id": safe(provider_id),
                "configured_model": safe(configured_model),
                "actual_model": safe(result.model),
                "response_id": safe(result.id),
            }
            self._diagnostic_logger.info("grok_oauth_transport_diag %s", json_module.dumps(record))
        except Exception:
            pass

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        json=None,
        policy: RequestPolicy,
        account_binding: str | None = None,
        binding_generation: int | None = None,
    ) -> dict:
        trace = _DiagnosticTrace(
            self._diagnostic_logging,
            self._diagnostic_logger,
            "json",
            policy,
            kind=request_kind(method, path),
        )
        try:
            result = await self._request_json(
                method,
                path,
                json=json,
                policy=policy,
                trace=trace,
                account_binding=account_binding,
                binding_generation=binding_generation,
            )
        except asyncio.CancelledError as exc:
            trace.emit("cancelled", exc)
            raise
        except Exception as exc:
            trace.emit("cancelled" if trace.cause == "CancelledError" else "failure", exc)
            raise
        trace.emit("success")
        return result

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json=None,
        policy: RequestPolicy,
        trace: _DiagnosticTrace,
        account_binding: str | None = None,
        binding_generation: int | None = None,
    ) -> dict:
        _validate_policy(policy)
        url = _api_url(path)
        trace.phase = "authentication"
        snapshot = await _token_before_deadline(self._oauth, policy)
        _check_video_authorization(self._oauth, snapshot, account_binding, binding_generation)
        auth_retries = 0
        connect_retries = 0
        authorization_generation = getattr(self._oauth, "binding_generation", None)
        while True:
            if getattr(self._oauth, "binding_generation", None) != authorization_generation:
                error = AuthorizationChanged()
                trace.failure("authentication", error)
                raise error
            response = None
            body = b""
            trace.begin_attempt("send_or_headers")
            try:
                async with asyncio.timeout_at(policy.deadline):
                    async with self._client.stream(
                        method,
                        url,
                        json=json,
                        headers=_headers(snapshot.access_token),
                        timeout=self._transport_timeout(policy),
                        follow_redirects=False,
                    ) as response:
                        trace.response(response)
                        if response.status_code < 400 and not response.is_redirect:
                            trace.phase = "read_body"
                            body = await _read_bounded(response, self._max_json_bytes, trace)
                        elif response.status_code == 403:
                            try:
                                trace.phase = "read_body"
                                body = await _read_bounded(response, 64 * 1024, trace)
                            except _BodyTooLarge:
                                body = b""
                                trace.error_body_state = "too_large"
                            if trace.enabled:
                                _set_error_metadata(trace, body)
                        elif response.status_code >= 400:
                            await _capture_error_response(response, policy, trace)
            except TimeoutError:
                trace.cause = "TimeoutError"
                if policy.side_effecting:
                    raise OutcomeUnknown() from None
                raise ProtocolError() from None
            except _BodyTooLarge:
                trace.cause = "BodyTooLarge"
                request_id = _request_id(response)
                if policy.side_effecting:
                    raise OutcomeUnknown(
                        request_id=request_id, operation_id=_operation_id(response)
                    ) from None
                raise ProtocolError(request_id=request_id) from None
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                trace.failure("connect", exc)
                if connect_retries < policy.safe_pre_send_retries and await _retry_backoff(
                    policy, trace, connect_retries
                ):
                    connect_retries += 1
                    trace.connect_retries = connect_retries
                    continue
                raise ProtocolError() from None
            except asyncio.CancelledError as exc:
                trace.failure(trace.phase, exc)
                if policy.side_effecting:
                    raise OutcomeUnknown() from None
                raise
            except (
                httpx.ReadTimeout,
                httpx.ReadError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
            ) as exc:
                trace.failure(trace.phase, exc)
                if policy.side_effecting:
                    raise OutcomeUnknown() from None
                raise ProtocolError() from None
            except httpx.RequestError as exc:
                trace.failure(trace.phase, exc)
                if policy.side_effecting:
                    raise OutcomeUnknown() from None
                raise ProtocolError() from None

            request_id = _request_id(response)
            if response.is_redirect:
                raise UnsafeTarget(request_id=request_id)
            if response.status_code == 401:
                if policy.allow_one_auth_retry and auth_retries == 0:
                    auth_retries += 1
                    trace.auth_retries = auth_retries
                    trace.phase = "authentication_refresh"
                    refreshed = await _token_before_deadline(
                        self._oauth,
                        policy,
                        force_refresh=True,
                        rejected_version=snapshot.version,
                    )
                    if refreshed.epoch != snapshot.epoch:
                        raise AuthorizationChanged()
                    _check_video_authorization(
                        self._oauth, refreshed, account_binding, binding_generation
                    )
                    snapshot = refreshed
                    continue
                raise ReauthorizationRequired(request_id=request_id)
            if response.status_code == 402:
                raise PaymentRequired(request_id=request_id)
            if response.status_code == 403:
                if _safe_error_code(body) in {
                    "unauthorized_client",
                    "invalid_client",
                    "client_not_eligible",
                }:
                    raise ClientNotEligible(request_id=request_id)
                raise PermissionDenied(request_id=request_id)
            if response.status_code == 429:
                error = RateLimited(request_id=request_id)
                error.retry_after_seconds = _retry_after(response, policy)
                raise error
            if 500 <= response.status_code:
                if policy.side_effecting:
                    raise OutcomeUnknown(
                        request_id=request_id, operation_id=_operation_id(response)
                    )
                raise ProtocolError(request_id=request_id)
            if response.status_code >= 400:
                raise ProtocolError(request_id=request_id)
            try:
                trace.phase = "decode"
                if (
                    response.status_code == 202
                    and method == "GET"
                    and re.fullmatch(r"/videos/[A-Za-z0-9_-]{1,128}", path)
                    and not body.strip()
                ):
                    result = {"status": "pending"}
                else:
                    result = json_module.loads(body)
            except (ValueError, UnicodeError, json_module.JSONDecodeError):
                if policy.side_effecting:
                    raise OutcomeUnknown(
                        request_id=request_id, operation_id=_operation_id(response)
                    ) from None
                raise ProtocolError(request_id=request_id) from None
            if not _has_time(policy):
                trace.cause = "TimeoutError"
                if policy.side_effecting:
                    raise OutcomeUnknown(
                        request_id=request_id, operation_id=_operation_id(response)
                    )
                raise ProtocolError(request_id=request_id)
            if not isinstance(result, dict):
                if policy.side_effecting:
                    raise OutcomeUnknown(
                        request_id=request_id, operation_id=_operation_id(response)
                    )
                raise ProtocolError(request_id=request_id)
            return result

    @asynccontextmanager
    async def stream_sse(self, path: str, *, json, policy: RequestPolicy):
        trace = _DiagnosticTrace(
            self._diagnostic_logging,
            self._diagnostic_logger,
            "stream",
            policy,
            kind=request_kind("POST", path),
        )
        try:
            async with self._stream_sse(path, json=json, policy=policy, trace=trace) as chunks:
                yield chunks
        except asyncio.CancelledError as exc:
            trace.emit("cancelled", exc)
            raise
        except Exception as exc:
            trace.emit("cancelled" if trace.cause == "CancelledError" else "failure", exc)
            raise
        else:
            trace.emit(
                "failure"
                if trace.cause
                else "stream_exhausted"
                if trace.body_exhausted
                else "scope_returned"
            )

    @asynccontextmanager
    async def _stream_sse(self, path: str, *, json, policy: RequestPolicy, trace: _DiagnosticTrace):
        _validate_policy(policy)
        url = _api_url(path)
        trace.phase = "authentication"
        snapshot = await _token_before_deadline(self._oauth, policy)
        auth_retries = 0
        connect_retries = 0
        authorization_generation = getattr(self._oauth, "binding_generation", None)
        while True:
            if getattr(self._oauth, "binding_generation", None) != authorization_generation:
                error = AuthorizationChanged()
                trace.failure("authentication", error)
                raise error
            trace.begin_attempt("send_or_headers")
            manager = self._client.stream(
                "POST",
                url,
                json=json,
                headers=_headers(snapshot.access_token),
                timeout=self._transport_timeout(policy),
                follow_redirects=False,
            )
            try:
                async with asyncio.timeout_at(policy.deadline):
                    response = await manager.__aenter__()
            except TimeoutError:
                trace.cause = "TimeoutError"
                if policy.side_effecting:
                    raise OutcomeUnknown() from None
                raise ProtocolError() from None
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                trace.failure("connect", exc)
                if connect_retries < policy.safe_pre_send_retries and await _retry_backoff(
                    policy, trace, connect_retries
                ):
                    connect_retries += 1
                    trace.connect_retries = connect_retries
                    continue
                raise ProtocolError() from None
            except asyncio.CancelledError as exc:
                trace.failure("send_or_headers", exc)
                if policy.side_effecting:
                    raise OutcomeUnknown() from None
                raise
            except httpx.RequestError as exc:
                trace.failure("send_or_headers", exc)
                if policy.side_effecting:
                    raise OutcomeUnknown() from None
                raise ProtocolError() from None
            request_id = _request_id(response)
            trace.response(response)
            if response.status_code == 401 and policy.allow_one_auth_retry and auth_retries == 0:
                await manager.__aexit__(None, None, None)
                auth_retries += 1
                trace.auth_retries = auth_retries
                trace.phase = "authentication_refresh"
                refreshed = await _token_before_deadline(
                    self._oauth,
                    policy,
                    force_refresh=True,
                    rejected_version=snapshot.version,
                )
                if refreshed.epoch != snapshot.epoch:
                    raise AuthorizationChanged()
                snapshot = refreshed
                continue
            try:
                if response.status_code >= 400:
                    await _capture_error_response(response, policy, trace)
                _raise_stream_status(response, request_id, policy)
                trace.phase = "stream_body"
                yield _protected_chunks(response, policy, request_id, trace)
            finally:
                await manager.__aexit__(None, None, None)
            return


async def _retry_backoff(policy, trace, retries):
    delay = random.uniform(0.1, 0.25) * min(2 ** min(retries, 2), 4)
    remaining = policy.deadline - time.monotonic()
    if remaining <= delay + 0.05:
        return False
    trace.phase = "retry_backoff"
    started = time.monotonic()
    try:
        async with asyncio.timeout_at(policy.deadline):
            await asyncio.sleep(delay)
    except asyncio.CancelledError as exc:
        trace.failure("retry_backoff", exc)
        if policy.side_effecting:
            raise OutcomeUnknown() from None
        raise
    except TimeoutError as exc:
        trace.failure("retry_backoff", exc)
        if policy.side_effecting:
            raise OutcomeUnknown() from None
        raise ProtocolError() from None
    finally:
        trace.retry_backoff_ms += round((time.monotonic() - started) * 1000)
    return _has_time(policy)


def _connection_error_kind(exc):
    current = exc
    seen = set()
    for _ in range(5):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if isinstance(current, socket.gaierror):
            return "dns"
        if isinstance(current, ssl.SSLError):
            return "tls"
        if isinstance(current, httpx.ProxyError):
            return "proxy"
        current = current.__cause__ or current.__context__
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout"
    return "connect" if isinstance(exc, httpx.ConnectError) else "other"


def _set_error_metadata(trace, body):
    if not body:
        if not trace.error_body_state:
            trace.error_body_state = "empty"
        return
    try:
        json_module.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        trace.error_body_state = "unparseable"
    else:
        trace.error_body_state = "json"
        trace.error_metadata = safe_error_metadata(body)


async def _capture_error_response(response, policy, trace):
    # Diagnostics must not turn a known HTTP rejection into a body-read failure.
    if not trace.enabled:
        return
    remaining = policy.deadline - time.monotonic()
    if remaining <= 0.05:
        trace.error_body_state = "deadline_skipped"
        return
    trace.phase = "error_body"
    try:
        async with asyncio.timeout(min(0.25, remaining / 2)):
            body = await _read_bounded(response, 8192)
        trace.body_bytes = len(body)
        _set_error_metadata(trace, body)
    except _BodyTooLarge:
        trace.error_body_state = "too_large"
    except Exception:
        trace.error_body_state = "read_failed"
    except asyncio.CancelledError as exc:
        trace.failure("error_body", exc)
        if policy.side_effecting:
            raise OutcomeUnknown(
                request_id=_request_id(response), operation_id=_operation_id(response)
            ) from None
        raise
    trace.phase = "response"


async def _protected_chunks(
    response: httpx.Response, policy: RequestPolicy, request_id: str, trace=None
):
    iterator = response.aiter_bytes().__aiter__()
    try:
        while True:
            if not _has_time(policy):
                _raise_chunk_deadline(policy, response, request_id, trace)
            try:
                async with asyncio.timeout_at(policy.deadline):
                    chunk = await anext(iterator)
            except StopAsyncIteration:
                if trace is not None:
                    trace.body_exhausted = True
                return
            if not _has_time(policy):
                _raise_chunk_deadline(policy, response, request_id, trace)
            if trace is not None:
                trace.body_bytes += len(chunk)
            yield chunk
    except TimeoutError:
        if trace is not None:
            trace.cause = "TimeoutError"
        if policy.side_effecting:
            raise OutcomeUnknown(
                request_id=request_id, operation_id=_operation_id(response)
            ) from None
        raise ProtocolError(request_id=request_id) from None
    except asyncio.CancelledError as exc:
        if trace is not None:
            trace.failure("stream_body", exc)
        if policy.side_effecting:
            raise OutcomeUnknown(
                request_id=request_id, operation_id=_operation_id(response)
            ) from None
        raise
    except httpx.RequestError as exc:
        if trace is not None:
            trace.failure("stream_body", exc)
        if policy.side_effecting:
            raise OutcomeUnknown(
                request_id=request_id, operation_id=_operation_id(response)
            ) from None
        raise ProtocolError(request_id=request_id) from None


def _raise_chunk_deadline(
    policy: RequestPolicy, response: httpx.Response, request_id: str, trace=None
) -> None:
    if trace is not None:
        trace.cause = "TimeoutError"
    if policy.side_effecting:
        raise OutcomeUnknown(request_id=request_id, operation_id=_operation_id(response))
    raise ProtocolError(request_id=request_id)


def _raise_stream_status(response: httpx.Response, request_id: str, policy: RequestPolicy) -> None:
    if response.is_redirect:
        raise UnsafeTarget(request_id=request_id)
    if response.status_code == 401:
        raise ReauthorizationRequired(request_id=request_id)
    if response.status_code == 402:
        raise PaymentRequired(request_id=request_id)
    if response.status_code == 403:
        raise PermissionDenied(request_id=request_id)
    if response.status_code == 429:
        raise RateLimited(request_id=request_id)
    if response.status_code >= 500 and policy.side_effecting:
        raise OutcomeUnknown(request_id=request_id, operation_id=_operation_id(response))
    if response.status_code >= 400:
        raise ProtocolError(request_id=request_id)


def _api_url(path: str) -> str:
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or path.startswith("//")
        or "\\" in path
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in path)
    ):
        raise UnsafeTarget()
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise UnsafeTarget()
    decoded = path
    for _ in range(3):
        decoded = unquote(decoded)
    if (
        decoded.startswith("//")
        or "\\" in decoded
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in decoded)
        or any(part == ".." for part in decoded.split("/"))
    ):
        raise UnsafeTarget()
    return API_BASE + path


def _video_account_binding(snapshot) -> str:
    identity = snapshot.user_id or _access_identity_subject(
        snapshot.access_token, snapshot.client_id
    )
    if (
        not isinstance(identity, str)
        or not 0 < len(identity) <= 256
        or any(not 0x21 <= ord(c) <= 0x7E for c in identity)
    ):
        raise ReauthorizationRequired("Video account identity is unavailable")
    return sha256(
        json_module.dumps(
            [snapshot.epoch, snapshot.client_id, identity], separators=(",", ":")
        ).encode()
    ).hexdigest()


def _check_video_authorization(oauth, snapshot, account_binding, binding_generation):
    if account_binding is None:
        return
    if (
        not isinstance(account_binding, str)
        or not re.fullmatch(r"[a-f0-9]{64}", account_binding)
        or _video_account_binding(snapshot) != account_binding
        or (
            binding_generation is not None
            and getattr(oauth, "binding_generation", 0) != binding_generation
        )
    ):
        raise AuthorizationChanged()


def _headers(access_token: str) -> dict[str, str]:
    if not isinstance(access_token, str) or not re.fullmatch(r"[A-Za-z0-9._~+/=-]+", access_token):
        raise ReauthorizationRequired()
    return {
        "accept": "application/json",
        "authorization": f"Bearer {access_token}",
        "user-agent": USER_AGENT,
    }


def _remaining(policy: RequestPolicy) -> float:
    value = policy.deadline - time.monotonic()
    if not math.isfinite(value) or value <= 0:
        raise ProtocolError()
    return value


def _has_time(policy: RequestPolicy) -> bool:
    return math.isfinite(policy.deadline) and policy.deadline > time.monotonic()


def _request_id(response: httpx.Response) -> str:
    return response.headers.get("x-request-id") or response.headers.get("request-id") or ""


def _operation_id(response: httpx.Response) -> str:
    value = response.headers.get("x-operation-id", "")
    return value if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) else ""


def _safe_error_code(body: bytes) -> str:
    try:
        data = json_module.loads(body)
    except (ValueError, UnicodeError):
        return ""
    if not isinstance(data, dict):
        return ""
    code = data.get("code") or data.get("error")
    return code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", code) else ""


class _BodyTooLarge(Exception):
    pass


async def _read_bounded(response: httpx.Response, limit: int, trace=None) -> bytes:
    chunks = bytearray()
    try:
        async for chunk in response.aiter_bytes():
            if len(chunks) + len(chunk) > limit:
                raise _BodyTooLarge()
            chunks.extend(chunk)
            if trace is not None:
                trace.body_bytes += len(chunk)
    except asyncio.CancelledError as exc:
        if trace is not None:
            trace.failure("read_body", exc)
        raise
    if trace is not None:
        trace.body_exhausted = True
    return bytes(chunks)


async def _token_before_deadline(oauth, policy: RequestPolicy, **kwargs):
    _remaining(policy)
    try:
        async with asyncio.timeout_at(policy.deadline):
            token = await oauth.get_token(**kwargs)
    except TimeoutError:
        raise ProtocolError() from None
    if not _has_time(policy):
        raise ProtocolError()
    return token


def _validate_policy(policy: RequestPolicy) -> None:
    if (
        not isinstance(policy, RequestPolicy)
        or not isinstance(policy.capability, str)
        or not policy.capability
        or isinstance(policy.deadline, bool)
        or not isinstance(policy.deadline, (int, float))
        or not math.isfinite(policy.deadline)
        or isinstance(policy.safe_pre_send_retries, bool)
        or not isinstance(policy.safe_pre_send_retries, int)
        or policy.safe_pre_send_retries < 0
        or not isinstance(policy.allow_one_auth_retry, bool)
        or not isinstance(policy.side_effecting, bool)
    ):
        raise ProtocolError()


def _retry_after(response: httpx.Response, policy: RequestPolicy) -> float | None:
    try:
        delay = float(response.headers.get("retry-after", ""))
    except ValueError:
        return None
    remaining = max(0.0, policy.deadline - time.monotonic())
    return min(delay, remaining) if math.isfinite(delay) and delay >= 0 else None
