"""Pinned xAI OIDC discovery and RFC 8628 device authorization wire protocol."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import socket
import ssl
import time
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import httpx

from .errors import (
    AuthorizationDenied,
    ClientNotEligible,
    DeviceCodeExpired,
    GrokOAuthError,
    ProtocolError,
    ReauthorizationRequired,
)
from .models import DeviceFlow, TokenSnapshot
from .version import USER_AGENT

DISCOVERY_URL = "https://auth.x.ai/.well-known/openid-configuration"
DEVICE_ENDPOINT = "https://auth.x.ai/oauth2/device/code"
TOKEN_ENDPOINT = "https://auth.x.ai/oauth2/token"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
MAX_OPERATION_TIMEOUT = 30.0


OAUTH_STAGES = {"discovery", "device_code", "token_poll", "token_refresh"}
OAUTH_REASONS = {
    "",
    "dns",
    "tls",
    "proxy",
    "connect_timeout",
    "connect",
    "read_timeout",
    "write_timeout",
    "pool_timeout",
    "read",
    "write",
    "remote_protocol",
    "network",
    "invalid_json",
    "invalid_response",
    "response_too_large",
    "http_status",
    "authorization_pending",
    "slow_down",
    "deadline",
    "cancelled",
    "other",
}
OAUTH_OUTCOMES = {"success", "failure", "cancelled", "timeout", "pending", "retrying"}
_OAUTH_CAUSES = {
    "",
    "ConnectError",
    "ConnectTimeout",
    "ProxyError",
    "ReadError",
    "WriteError",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "RemoteProtocolError",
    "LocalProtocolError",
    "ProtocolError",
    "DeviceCodeExpired",
    "AuthorizationDenied",
    "ClientNotEligible",
    "ReauthorizationRequired",
    "CancelledError",
    "TimeoutError",
    "DecodingError",
    "OtherError",
}


def safe_oauth_diagnostic(value):
    """Copy only fixed, content-free OAuth metadata for status and error responses."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, allowed in (
        ("oauth_stage", OAUTH_STAGES),
        ("oauth_reason", OAUTH_REASONS),
        ("outcome", OAUTH_OUTCOMES),
        ("cause", _OAUTH_CAUSES),
    ):
        if isinstance(value.get(key), str) and value[key] in allowed:
            result[key] = value[key]
    if value.get("termination_reason") in {"request_deadline", "external_cancel"}:
        result["termination_reason"] = value["termination_reason"]
    trace = value.get("trace_id")
    if isinstance(trace, str) and re.fullmatch(r"[0-9a-f]{32}", trace):
        result["trace_id"] = trace
    status = value.get("status")
    if status is None or type(status) is int and 100 <= status <= 599:
        result["status"] = status
    for key in ("poll_retries", "retry_in_ms", "elapsed_ms"):
        number = value.get(key)
        if type(number) in (int, float) and math.isfinite(number) and 0 <= number <= 10**15:
            result[key] = number
    return result


def _oauth_network_reason(error):
    current, seen = error, set()
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
    for cls, reason in (
        (httpx.ConnectTimeout, "connect_timeout"),
        (httpx.ConnectError, "connect"),
        (httpx.ReadTimeout, "read_timeout"),
        (httpx.WriteTimeout, "write_timeout"),
        (httpx.PoolTimeout, "pool_timeout"),
        (httpx.ReadError, "read"),
        (httpx.WriteError, "write"),
        (httpx.RemoteProtocolError, "remote_protocol"),
    ):
        if isinstance(error, cls):
            return reason
    return "network"


class _OAuthTrace:
    def __init__(self, wire, stage, flow):
        self.wire, self.stage, self.flow = wire, stage, flow
        self.trace_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.reason, self.cause, self.status = "", "", None
        self.retries, self.retry_in_ms = 0, 0
        self.termination_reason = ""

    def begin(self):
        self.reason, self.cause, self.status = "", "", None
        self.retry_in_ms = 0

    def emit(self, outcome):
        record = safe_oauth_diagnostic(
            {
                "oauth_stage": self.stage,
                "oauth_reason": self.reason,
                "cause": self.cause if self.cause in _OAUTH_CAUSES else "OtherError",
                "status": self.status,
                "trace_id": self.trace_id,
                "outcome": outcome,
                "poll_retries": self.retries,
                "retry_in_ms": self.retry_in_ms,
                "elapsed_ms": round((time.monotonic() - self.started) * 1000),
                "termination_reason": self.termination_reason,
            }
        )
        if self.flow is not None and self.wire._flow_diagnostic is not None:
            try:
                self.wire._flow_diagnostic(self.flow, dict(record))
            except Exception:
                pass
        if self.wire._diagnostic_logging and self.wire._diagnostic_logger is not None:
            try:
                self.wire._diagnostic_logger.info(
                    "grok_oauth_transport_diag %s",
                    json.dumps({"event": "oauth_request", **record}, separators=(",", ":")),
                )
            except Exception:
                pass
        return record


class _PollingConnectFailure(Exception):
    """A device poll failed before its HTTP request could be sent."""


class OAuthWireClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        client_id: str,
        scope: str,
        operation_timeout: float = MAX_OPERATION_TIMEOUT,
        diagnostic_logging: bool = False,
        diagnostic_logger=None,
        flow_diagnostic=None,
    ):
        if (
            not isinstance(client_id, str)
            or not client_id
            or not isinstance(scope, str)
            or not scope
        ):
            raise ValueError("client_id and scope are required")
        if (
            isinstance(operation_timeout, bool)
            or not isinstance(operation_timeout, (int, float))
            or not math.isfinite(operation_timeout)
            or operation_timeout <= 0
            or operation_timeout > MAX_OPERATION_TIMEOUT
        ):
            raise ValueError("operation_timeout must be within 30 seconds")
        self._http = http
        self.client_id = client_id
        self.scope = scope
        self._operation_timeout = float(operation_timeout)
        self._device_endpoint: str | None = None
        self._token_endpoint: str | None = None
        self._discovery_lock = asyncio.Lock()
        self._diagnostic_logging = diagnostic_logging is True
        self._diagnostic_logger = diagnostic_logger
        self._flow_diagnostic = flow_diagnostic

    @asynccontextmanager
    async def _diagnostic_scope(self, stage, flow=None):
        trace = _OAuthTrace(self, stage, flow)
        try:
            yield trace
        except BaseException as error:
            trace.retry_in_ms = 0
            if isinstance(error, asyncio.CancelledError):
                trace.reason, trace.cause = "cancelled", "CancelledError"
                trace.termination_reason = "external_cancel"
            elif isinstance(error, DeviceCodeExpired):
                trace.termination_reason = "request_deadline"
                if trace.reason in {"", "authorization_pending", "slow_down"}:
                    trace.reason = "deadline"
            if not trace.reason:
                trace.reason = (
                    "cancelled"
                    if isinstance(error, asyncio.CancelledError)
                    else "deadline"
                    if isinstance(error, DeviceCodeExpired)
                    else "invalid_response"
                    if isinstance(error, GrokOAuthError)
                    else "other"
                )
            if not trace.cause:
                trace.cause = type(error).__name__
            outcome = (
                "cancelled"
                if isinstance(error, asyncio.CancelledError)
                else "timeout"
                if trace.termination_reason == "request_deadline"
                else "failure"
            )
            record = trace.emit(outcome)
            if isinstance(error, GrokOAuthError):
                error.oauth_diagnostic = record
            raise
        else:
            trace.reason, trace.cause, trace.retry_in_ms = "", "", 0
            trace.emit("success")

    async def start_device_flow(self, *, owner_id: str, epoch: int) -> DeviceFlow:
        async with self._diagnostic_scope("discovery") as trace:
            return await self._start_device_flow(owner_id=owner_id, epoch=epoch, trace=trace)

    async def poll_device_flow(
        self, flow: DeviceFlow, *, clock=time.monotonic, sleep=asyncio.sleep
    ) -> TokenSnapshot:
        async with self._diagnostic_scope("discovery", flow) as trace:
            return await self._poll_device_flow(flow, clock=clock, sleep=sleep, trace=trace)

    async def refresh(self, snapshot: TokenSnapshot) -> TokenSnapshot:
        async with self._diagnostic_scope("discovery") as trace:
            return await self._refresh(snapshot, trace=trace)

    async def _start_device_flow(self, *, owner_id: str, epoch: int, trace) -> DeviceFlow:
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("owner_id is required")
        deadline = time.monotonic() + self._operation_timeout
        await self._discover(deadline=deadline, timeout_error=ProtocolError, trace=trace)
        trace.stage = "device_code"
        response, data = await _bounded_request_json(
            self._http,
            "POST",
            self._device_endpoint,
            data={
                "client_id": self.client_id,
                "scope": self.scope,
                "referrer": "astrbot_plugin_xiaobai_grok",
            },
            deadline=deadline,
            timeout_error=ProtocolError,
            trace=trace,
        )
        if response.is_redirect:
            raise ProtocolError()
        if not response.is_success:
            _raise_oauth_failure(data)
        device_code = _required_text(data, "device_code")
        user_code = _required_text(data, "user_code")
        verification_uri = _safe_auth_url(_required_text(data, "verification_uri"))
        complete = data.get("verification_uri_complete", "")
        if complete:
            complete = _safe_auth_url(complete)
        expires = _positive_number(data.get("expires_in"), 300.0)
        interval = max(1.0, _positive_number(data.get("interval"), 5.0))
        now_mono = time.monotonic()
        return DeviceFlow(
            flow_id=uuid.uuid4().hex,
            owner_id=owner_id,
            user_code=user_code,
            verification_uri=verification_uri,
            verification_uri_complete=complete,
            device_code=device_code,
            expires_in=expires,
            interval=interval,
            epoch=epoch,
            deadline=now_mono + expires,
            expires_at=time.time() + expires,
        )

    async def _poll_device_flow(
        self, flow: DeviceFlow, *, clock=time.monotonic, sleep=asyncio.sleep, trace
    ) -> TokenSnapshot:
        deadline = flow.deadline
        if not math.isfinite(deadline) or deadline <= clock():
            raise DeviceCodeExpired()
        await self._discover(
            deadline=deadline, clock=clock, timeout_error=DeviceCodeExpired, trace=trace
        )
        interval = flow.interval
        while clock() < deadline:
            trace.stage = "token_poll"
            try:
                response, data = await _bounded_request_json(
                    self._http,
                    "POST",
                    self._token_endpoint,
                    data={
                        "grant_type": DEVICE_GRANT,
                        "client_id": self.client_id,
                        "device_code": flow.device_code,
                    },
                    deadline=deadline,
                    clock=clock,
                    timeout_error=DeviceCodeExpired,
                    report_connect_failure=True,
                    request_timeout=self._operation_timeout,
                    trace=trace,
                )
            except _PollingConnectFailure:
                # RFC 8628 §3.5: reduce polling frequency after connection timeouts.
                # This branch only covers failures before sending, not lost token responses.
                interval = max(interval, min(60.0, interval * 2))
                trace.retries += 1
                trace.retry_in_ms = round(min(interval, max(0, deadline - clock())) * 1000)
                trace.emit("retrying")
                remaining = deadline - clock()
                if remaining <= 0:
                    break
                await _within_deadline(
                    sleep(min(interval, remaining)), deadline, clock, DeviceCodeExpired, trace=trace
                )
                continue
            if clock() >= deadline:
                raise DeviceCodeExpired()
            if response.is_redirect:
                raise ProtocolError()
            if response.is_success:
                return _token_snapshot(data, client_id=self.client_id, epoch=flow.epoch)
            code = data.get("error") if isinstance(data, dict) else None
            if code == "authorization_pending":
                trace.reason = "authorization_pending"
            elif code == "slow_down":
                interval += 5.0
                trace.reason = "slow_down"
            elif isinstance(code, str) and code in {
                "access_denied",
                "authorization_denied",
            }:
                raise AuthorizationDenied()
            elif code == "expired_token":
                raise DeviceCodeExpired()
            elif response.status_code == 429 or 500 <= response.status_code < 600:
                interval = max(interval, _retry_after(response) or interval)
            else:
                _raise_oauth_failure(data)
            remaining = deadline - clock()
            if remaining <= 0:
                break
            trace.retry_in_ms = round(min(interval, remaining) * 1000)
            trace.emit("pending" if code in ("authorization_pending", "slow_down") else "retrying")
            await _within_deadline(
                sleep(min(interval, remaining)), deadline, clock, DeviceCodeExpired, trace=trace
            )
        raise DeviceCodeExpired()

    async def _refresh(self, snapshot: TokenSnapshot, *, trace) -> TokenSnapshot:
        deadline = time.monotonic() + self._operation_timeout
        await self._discover(deadline=deadline, timeout_error=ProtocolError, trace=trace)
        trace.stage = "token_refresh"
        response, data = await _bounded_request_json(
            self._http,
            "POST",
            self._token_endpoint,
            data={
                "grant_type": "refresh_token",
                "refresh_token": snapshot.refresh_token,
                "client_id": self.client_id,
            },
            deadline=deadline,
            timeout_error=ProtocolError,
            trace=trace,
        )
        if response.is_redirect:
            raise ProtocolError()
        if not response.is_success:
            if isinstance(data, dict) and data.get("error") == "invalid_grant":
                raise ReauthorizationRequired() from None
            _raise_oauth_failure(data)
        result = _token_snapshot(data, client_id=self.client_id, epoch=snapshot.epoch)
        return TokenSnapshot(
            snapshot.slot,
            result.access_token,
            result.refresh_token,
            result.expires_at,
            result.scope or snapshot.scope,
            snapshot.client_id,
            snapshot.version,
            snapshot.epoch,
            snapshot.user_id or result.user_id,
        )

    async def _discover(
        self,
        *,
        deadline: float | None = None,
        clock=time.monotonic,
        timeout_error=DeviceCodeExpired,
        trace=None,
    ) -> None:
        if self._token_endpoint is not None:
            return

        async def discover_locked() -> None:
            async with self._discovery_lock:
                if self._token_endpoint is not None:
                    return
                response, data = await _bounded_request_json(
                    self._http,
                    "GET",
                    DISCOVERY_URL,
                    deadline=deadline,
                    clock=clock,
                    timeout_error=timeout_error,
                    trace=trace,
                )
                if response.is_redirect or not response.is_success:
                    raise ProtocolError()
                if data.get("issuer") != "https://auth.x.ai":
                    raise ProtocolError()
                device = data.get("device_authorization_endpoint")
                token = data.get("token_endpoint")
                if device != DEVICE_ENDPOINT or token != TOKEN_ENDPOINT:
                    raise ProtocolError()
                _pinned_auth_endpoint(device, "/oauth2/device/code")
                _pinned_auth_endpoint(token, "/oauth2/token")
                self._device_endpoint, self._token_endpoint = device, token

        if deadline is None:
            await discover_locked()
        else:
            await _within_deadline(discover_locked(), deadline, clock, timeout_error, trace=trace)


def _headers() -> dict[str, str]:
    return {
        "accept": "application/json",
        "user-agent": USER_AGENT,
        "content-type": "application/x-www-form-urlencoded",
    }


async def _bounded_request_json(
    http: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    data: dict | None = None,
    deadline: float | None = None,
    clock=time.monotonic,
    timeout_error=DeviceCodeExpired,
    report_connect_failure: bool = False,
    request_timeout: float | None = None,
    trace=None,
) -> tuple[httpx.Response, dict]:
    async def send() -> tuple[httpx.Response, dict]:
        if trace is not None:
            trace.begin()
        options = {"data": data, "headers": _headers(), "follow_redirects": False}
        if deadline is not None:
            remaining = deadline - clock()
            if not math.isfinite(remaining) or remaining <= 0:
                raise timeout_error()
            options["timeout"] = min(remaining, request_timeout or remaining)
        try:
            async with http.stream(method, url, **options) as response:
                if trace is not None:
                    trace.status = response.status_code
                    if not response.is_success:
                        trace.reason = "http_status"
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > 1024 * 1024:
                        if trace is not None:
                            trace.reason = "response_too_large"
                        raise ProtocolError()
                    body.extend(chunk)
        except (httpx.ConnectTimeout, httpx.ConnectError) as error:
            if trace is not None:
                trace.reason, trace.cause = _oauth_network_reason(error), type(error).__name__
            if report_connect_failure:
                raise _PollingConnectFailure() from None
            raise ProtocolError() from None
        except httpx.RequestError as error:
            if trace is not None:
                trace.reason, trace.cause = _oauth_network_reason(error), type(error).__name__
            raise ProtocolError() from None
        try:
            parsed = json.loads(body)
        except (ValueError, UnicodeError):
            if trace is not None:
                trace.reason = "invalid_json"
            raise ProtocolError() from None
        if not isinstance(parsed, dict):
            if trace is not None:
                trace.reason = "invalid_response"
            raise ProtocolError()
        return response, parsed

    if deadline is None:
        return await send()
    return await _within_deadline(send(), deadline, clock, timeout_error, trace=trace)


async def _within_deadline(operation, deadline: float, clock, timeout_error, *, trace=None):
    def mark_deadline():
        if trace is not None:
            trace.termination_reason = "request_deadline"
            if trace.reason in {"", "authorization_pending", "slow_down"}:
                trace.reason = "deadline"
            trace.cause = "TimeoutError"

    remaining = deadline - clock()
    if not math.isfinite(remaining) or remaining <= 0:
        operation.close()
        mark_deadline()
        raise timeout_error()
    try:
        async with asyncio.timeout(remaining):
            result = await operation
    except TimeoutError:
        mark_deadline()
        raise timeout_error() from None
    if clock() >= deadline:
        mark_deadline()
        raise timeout_error()
    return result


def _required_text(data: dict, key: str) -> str:
    value = data.get(key)
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ProtocolError()
    return value


def _positive_number(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return default
    return result if math.isfinite(result) and result > 0 else default


def _pinned_auth_endpoint(value: str, path: str) -> None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ProtocolError() from None
    if (
        parsed.scheme != "https"
        or parsed.hostname != "auth.x.ai"
        or port not in {None, 443}
        or parsed.path != path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ProtocolError()


def _safe_auth_url(value: object) -> str:
    if not isinstance(value, str):
        raise ProtocolError()
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ProtocolError() from None
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"auth.x.ai", "accounts.x.ai"}
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ProtocolError()
    return value


def _token_snapshot(data: dict, *, client_id: str, epoch: int) -> TokenSnapshot:
    token_type = data.get("token_type", "Bearer")
    if not isinstance(token_type, str) or token_type.casefold() != "bearer":
        raise ProtocolError()
    access = _required_text(data, "access_token")
    refresh = data.get("refresh_token", "")
    if not isinstance(refresh, str) or any(
        ord(character) < 0x20 or ord(character) == 0x7F for character in refresh
    ):
        raise ProtocolError()
    scope = data.get("scope", "")
    if not isinstance(scope, str):
        raise ProtocolError()
    seconds = _positive_number(data.get("expires_in"), 0.0)
    expires_at = time.time() + seconds if seconds else _jwt_exp(access)
    return TokenSnapshot(
        "default",
        access,
        refresh,
        expires_at,
        scope,
        client_id,
        0,
        epoch,
        _identity_subject(data.get("id_token")) or _access_identity_subject(access, client_id),
    )


def _jwt_exp(token: str) -> float | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        value = json.loads(base64.urlsafe_b64decode(payload))
        if not isinstance(value, dict):
            return None
        exp = value.get("exp")
        if (
            isinstance(exp, bool)
            or not isinstance(exp, (int, float))
            or not math.isfinite(exp)
            or exp < 0
        ):
            return None
        return float(exp)
    except (OverflowError, ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        return None


def _raise_oauth_failure(data: object) -> None:
    code = data.get("error") if isinstance(data, dict) else None
    if isinstance(code, str) and code in {
        "unauthorized_client",
        "invalid_client",
        "client_not_eligible",
    }:
        raise ClientNotEligible()
    raise ProtocolError()


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        delay = float(value) if value is not None else None
    except ValueError:
        return None
    return delay if delay is not None and math.isfinite(delay) and delay >= 0 else None


def _identity_subject(token: object) -> str | None:
    """Identity metadata from our fixed HTTPS token endpoint, never model input.

    The bearer credential still authenticates billing; this claim is only its
    routing identity. Missing/invalid ID tokens never fall back to guessed IDs.
    """
    if not isinstance(token, str) or len(token) > 32768:
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        value = data.get("sub") if isinstance(data, dict) else None
        if (
            isinstance(value, str)
            and 0 < len(value) <= 256
            and all(0x21 <= ord(c) <= 0x7E for c in value)
        ):
            return value
    except (ValueError, TypeError, UnicodeError):
        pass
    return None


def _access_identity_subject(token: object, client_id: str) -> str | None:
    """Read routing metadata from this plugin's existing xAI user credential.

    Like the official CLI principal reader, this does not establish local
    authorization. The fixed upstream validates the bearer signature. Require
    matching issuer, client and User principal; a bare arbitrary sub is not used.
    """
    if not isinstance(client_id, str) or not client_id:
        return None
    subject = _identity_subject(token)
    if subject is None:
        return None
    payload = token.split(".")[1]
    data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))

    def claim(snake, camel):
        values = [data[key] for key in (snake, camel) if key in data]
        return values[0] if values and all(value == values[0] for value in values) else None

    if (
        data.get("iss") == "https://auth.x.ai"
        and data.get("client_id") == client_id
        and claim("principal_type", "principalType") == "User"
        and claim("principal_id", "principalId") == subject
    ):
        return subject
    return None
