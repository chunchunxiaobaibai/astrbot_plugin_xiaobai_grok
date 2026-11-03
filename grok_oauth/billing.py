"""Fixed OAuth billing destination and account-isolated, bounded shared cache."""

import asyncio
import json
import math
import time
from dataclasses import replace
from datetime import datetime
from email.utils import parsedate_to_datetime

import httpx

from .billing_parser import parse_billing
from .errors import AuthorizationChanged, GrokOAuthError, ReauthorizationRequired
from .oauth import _access_identity_subject
from .version import USER_AGENT

BILLING_URL = "https://cli-chat-proxy.grok.com/v1/billing?format=credits"
CACHE_SECONDS = 120
STALE_SECONDS = 600


class BillingClient:
    def __init__(self, http, oauth, *, clock=time.time):
        self._http = http
        self._oauth = oauth
        self._clock = clock
        self._closed = False
        self._key = None
        self._cache = None
        self._cache_time = 0.0
        self._cooldown = 0.0
        self._task = None
        self._tasks = set()

    def _identity(self):
        return self._oauth.epoch, self._oauth.binding_generation

    def _empty(self, status):
        return replace(parse_billing({}, observed_at=self._clock()), status=status)

    def invalidate(self):
        self._key = None
        self._cache = None
        self._cache_time = 0.0
        self._cooldown = 0.0
        self._task = None

    def _same_account(self, key):
        return not self._closed and self._identity() == key and self._oauth.status == "authorized"

    def _before_period_end(self):
        if self._cache is None or not self._cache.reset_at:
            return True
        return (
            self._clock()
            < datetime.fromisoformat(self._cache.reset_at.replace("Z", "+00:00")).timestamp()
        )

    async def get_usage(self, *, force_refresh=False):
        if self._closed:
            return self._empty("closed")
        if self._oauth.status != "authorized":
            self.invalidate()
            return self._empty(self._oauth.status)
        key = self._identity()
        if self._key != key:
            self.invalidate()
            self._key = key
        if self._clock() < self._cooldown:
            return self._failure("rate_limited")
        if (
            not force_refresh
            and self._cache is not None
            and 0 <= self._clock() - self._cache_time < CACHE_SECONDS
            and self._before_period_end()
        ):
            return replace(self._cache, cached=True)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._fetch(key), name="grok-usage")
            self._tasks.add(self._task)
            self._task.add_done_callback(self._done)
        return await asyncio.shield(self._task)

    def _done(self, task):
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def _failure(self, status):
        if (
            status in {"unavailable", "rate_limited"}
            and self._cache is not None
            and self._cache.status == "success"
            and 0 <= self._clock() - self._cache_time <= STALE_SECONDS
            and self._before_period_end()
        ):
            return replace(self._cache, status=status, cached=True, stale=True)
        return self._empty(status)

    async def _fetch(self, key):
        try:
            async with asyncio.timeout(30):
                result = await self._request(key)
        except (TimeoutError, httpx.HTTPError):
            result = self._empty("unavailable")
        except AuthorizationChanged:
            result = self._empty("authorization_changed")
        except ReauthorizationRequired:
            result = self._empty("reauth_required")
        except GrokOAuthError:
            result = self._empty("unavailable")
        if not self._same_account(key):
            return self._empty("authorization_changed")
        if result.status in {"success", "unknown", "unparseable", "identity_unavailable"}:
            self._cache = result
            self._cache_time = self._clock()
        elif result.status in {"forbidden", "reauth_required"}:
            self._cache = None
        return (
            self._failure(result.status)
            if result.status in {"unavailable", "rate_limited"}
            else result
        )

    async def _request(self, key):
        token = await self._oauth.get_token()
        refreshed = False
        user_id = token.user_id or _access_identity_subject(token.access_token, token.client_id)
        if not user_id:
            token = await self._oauth.get_token(force_refresh=True, rejected_version=token.version)
            refreshed = True
            user_id = token.user_id or _access_identity_subject(token.access_token, token.client_id)
        if not self._same_account(key):
            raise AuthorizationChanged()
        if not user_id or not _safe_identity(user_id):
            return self._empty("identity_unavailable")
        while True:
            async with asyncio.timeout(15):
                async with self._http.stream(
                    "GET",
                    BILLING_URL,
                    headers={
                        "Authorization": f"Bearer {token.access_token}",
                        "X-XAI-Token-Auth": "xai-grok-cli",
                        "x-userid": user_id,
                        "x-grok-client-version": USER_AGENT,
                        "x-grok-client-mode": "headless",
                        "Accept": "application/json",
                    },
                    follow_redirects=False,
                    timeout=15,
                ) as response:
                    if not self._same_account(key):
                        raise AuthorizationChanged()
                    if response.status_code == 401:
                        status = 401
                    elif response.status_code == 403:
                        return self._empty("forbidden")
                    elif response.status_code == 429:
                        self._cooldown = self._clock() + _retry_after(
                            response.headers.get("Retry-After"), self._clock()
                        )
                        return self._empty("rate_limited")
                    elif not response.is_success:
                        return self._empty("unavailable")
                    else:
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > 65536:
                                return self._empty("unparseable")
                        try:
                            data = json.loads(body)
                        except (ValueError, UnicodeError):
                            return self._empty("unparseable")
                        return parse_billing(data, observed_at=self._clock())
            if status == 401 and not refreshed:
                token = await self._oauth.get_token(
                    force_refresh=True, rejected_version=token.version
                )
                refreshed = True
                user_id = token.user_id or _access_identity_subject(
                    token.access_token, token.client_id
                )
                if not user_id or not _safe_identity(user_id):
                    return self._empty("identity_unavailable")
                if not self._same_account(key):
                    raise AuthorizationChanged()
            else:
                return self._empty("reauth_required")

    async def close(self):
        self._closed = True
        self.invalidate()
        pending = tuple(self._tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def _safe_identity(value):
    return (
        isinstance(value, str)
        and 0 < len(value) <= 256
        and all(0x21 <= ord(c) <= 0x7E for c in value)
    )


def _retry_after(value, now):
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - now
        except (ValueError, TypeError, OverflowError):
            return 60.0
    return min(86400.0, max(1.0, seconds)) if math.isfinite(seconds) else 60.0
