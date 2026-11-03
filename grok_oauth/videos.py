"""OAuth video submit, resumable polling, and scoped result materialization."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
import uuid
import weakref
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256

from .errors import (
    AuthorizationChanged,
    Busy,
    GrokOAuthError,
    InvalidRequest,
    OutcomeUnknown,
    ProtocolError,
    ServiceClosed,
)
from .http import _video_account_binding
from .models import AssetScope, RequestPolicy
from .video_store import VideoJob, VideoStore

GENERATION_MODEL = "grok-imagine-video-1.5"
EDIT_MODEL = "grok-imagine-video"
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


@dataclass(frozen=True)
class VideoRequest:
    prompt: str
    reference_asset_id: str = ""
    action: str = "generate"
    duration: int = 6
    resolution: str = "480p"


def _timeout(value, *, maximum=600):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidRequest("Video timeout must be numeric")
    if not math.isfinite(value) or value <= 0 or value > maximum:
        raise InvalidRequest("Video timeout is outside the supported range")
    return float(value)


async def _durable(awaitable):
    """Complete a consistency write even when a caller is being cancelled."""
    task = asyncio.create_task(awaitable)
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
    result = task.result()
    if cancelled is not None:
        raise cancelled
    return result


class VideoService:
    def __init__(
        self,
        http,
        image_assets,
        video_store: VideoStore,
        *,
        oauth=None,
        max_running=2,
        max_pending=8,
    ):
        for value in (max_running, max_pending):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("Video queue limits must be integers")
        if max_running <= 0 or max_pending < 0:
            raise ValueError("Video queue limits are invalid")
        self.http = http
        self.image_assets = image_assets
        self.video_store = video_store
        self.oauth = oauth or getattr(http, "_oauth", None)
        self._running = asyncio.Semaphore(max_running)
        self._capacity = max_running + max_pending
        self._admitted = 0
        self._active = set()
        self._job_locks = weakref.WeakValueDictionary()
        self._closed = False

    @asynccontextmanager
    async def _slot(self, deadline):
        if self._closed:
            raise ServiceClosed()
        if self._admitted >= self._capacity:
            raise Busy("Video queue is full")
        self._admitted += 1
        task = asyncio.current_task()
        self._active.add(task)
        acquired = False
        try:
            async with asyncio.timeout_at(deadline):
                await self._running.acquire()
                acquired = True
                yield
        finally:
            if acquired:
                self._running.release()
            self._admitted -= 1
            self._active.discard(task)

    async def _binding(self):
        if self.oauth is None:
            raise ServiceClosed("Video OAuth service is unavailable")
        token = await self.oauth.get_token()
        return _video_account_binding(token)

    async def check_job(self, job_id, *, scope: AssetScope) -> VideoJob:
        if self._closed:
            raise ServiceClosed()
        job = await self.video_store.get_job(job_id, scope=scope)
        if await self.video_store.job_binding(job_id, scope=scope) != await self._binding():
            raise AuthorizationChanged()
        return job

    @staticmethod
    def _validate(request):
        if not isinstance(request, VideoRequest):
            raise InvalidRequest("request must be VideoRequest")
        if (
            not isinstance(request.prompt, str)
            or not request.prompt.strip()
            or len(request.prompt) > 16000
        ):
            raise InvalidRequest("Video prompt must contain at most 16000 characters")
        if not isinstance(request.action, str) or request.action not in {
            "generate",
            "edit",
            "text",
        }:
            raise InvalidRequest("Video action must be generate, edit or text")
        if not isinstance(request.reference_asset_id, str):
            raise InvalidRequest("Video reference must be a scoped asset ID")
        if request.action == "text":
            if request.reference_asset_id:
                raise InvalidRequest("Text generation does not accept a reference asset")
        elif not request.reference_asset_id:
            raise InvalidRequest("A scoped reference asset is required")
        if (
            isinstance(request.duration, bool)
            or not isinstance(request.duration, int)
            or request.duration not in {6, 10}
        ):
            raise InvalidRequest("Video duration must be 6 or 10 seconds")
        if request.resolution != "480p":
            raise InvalidRequest("Video resolution must be 480p")

    async def submit(
        self,
        request: VideoRequest,
        *,
        scope: AssetScope,
        operation_key="",
        timeout=60,  # noqa: ASYNC109 - converted into an absolute deadline
    ) -> VideoJob:  # noqa: ASYNC109 - public API converts timeout into an absolute deadline
        try:
            self._validate(request)
            deadline = time.monotonic() + _timeout(timeout, maximum=120)
            if not isinstance(operation_key, str) or len(operation_key) > 256:
                raise InvalidRequest("Video operation key is invalid")
        except GrokOAuthError as exc:
            exc.submission_state = "not_submitted"
            raise
        # Hash all caller keys; neither message contents nor channel identifiers are persisted as keys.
        operation_key = sha256((operation_key or uuid.uuid4().hex).encode()).hexdigest()
        fingerprint = sha256(json.dumps(asdict(request), sort_keys=True).encode()).hexdigest()
        model = EDIT_MODEL if request.action == "edit" else GENERATION_MODEL
        job = None
        started = False
        remote_id = ""
        try:
            async with self._slot(deadline):
                binding = await self._binding()
                generation = getattr(self.oauth, "binding_generation", 0)
                job, created = await self.video_store.create_job(
                    scope=scope,
                    operation_key=operation_key,
                    fingerprint=fingerprint,
                    binding=binding,
                    model=model,
                    duration=request.duration,
                )
                if not created:
                    return job
                async with AsyncExitStack() as inputs:
                    uri = ""
                    if request.action != "text":
                        store = (
                            self.image_assets if request.action == "generate" else self.video_store
                        )
                        await inputs.enter_async_context(
                            store.lease(request.reference_asset_id, scope=scope)
                        )
                        if request.action == "edit":
                            if (
                                await store.asset_binding(request.reference_asset_id, scope=scope)
                                != binding
                            ):
                                raise AuthorizationChanged()
                            asset = await store.get_asset(request.reference_asset_id, scope=scope)
                            if asset.duration > 8.7:
                                raise InvalidRequest("Video editing input exceeds 8.7 seconds")
                        uri = await store.to_data_uri(request.reference_asset_id, scope=scope)
                    if (
                        binding != await self._binding()
                        or getattr(self.oauth, "binding_generation", 0) != generation
                    ):
                        raise AuthorizationChanged()
                    body = {"model": model, "prompt": request.prompt}
                    path = "/videos/generations"
                    if request.action != "edit":
                        body.update(duration=request.duration, resolution=request.resolution)
                        if request.action == "generate":
                            body["image"] = {"url": uri}
                    else:
                        path = "/videos/edits"
                        body["video"] = {"url": uri}
                    # A crash after this write is conservatively unknown, never an implicit retry.
                    await _durable(
                        self.video_store.update_job(
                            job.job_id, scope=scope, submission_state="unknown"
                        )
                    )
                    started = True
                    response = await self.http.request_json(
                        "POST",
                        path,
                        json=body,
                        account_binding=binding,
                        binding_generation=generation,
                        policy=RequestPolicy(
                            capability="videos",
                            deadline=deadline,
                            safe_pre_send_retries=0,
                            allow_one_auth_retry=True,
                            side_effecting=True,
                        ),
                    )
                    request_id = response.get("request_id") if isinstance(response, dict) else None
                    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
                        raise OutcomeUnknown("Video submission has no recoverable task ID")
                    remote_id = request_id
                    # Persist the returned remote handle before any cancellable follow-up.
                    return await _durable(
                        self.video_store.update_job(
                            job.job_id,
                            scope=scope,
                            status="pending",
                            request_id=request_id,
                            submission_state="submitted",
                        )
                    )
        except BaseException as exc:
            unknown = started and isinstance(
                exc, (OutcomeUnknown, TimeoutError, asyncio.CancelledError)
            )
            # Capture local submission knowledge before any fallible storage cleanup.
            # A returned handle stays submitted even when persisting or reading it fails.
            state = "submitted" if remote_id else "unknown" if unknown else "not_submitted"
            if isinstance(exc, GrokOAuthError):
                exc.operation_id = job.job_id if job is not None else ""
                exc.submission_state = state
            if job is not None:
                try:
                    current = await _durable(self.video_store.get_job(job.job_id, scope=scope))
                    if current.request_id:
                        state = "submitted"
                    elif remote_id:
                        await _durable(
                            self.video_store.update_job(
                                job.job_id,
                                scope=scope,
                                status="pending",
                                request_id=remote_id,
                                submission_state="submitted",
                            )
                        )
                    else:
                        error = (
                            "OutcomeUnknown"
                            if unknown
                            else type(exc).__name__
                            if isinstance(exc, GrokOAuthError)
                            else "ProtocolError"
                        )
                        await _durable(
                            self.video_store.update_job(
                                job.job_id,
                                scope=scope,
                                status="unknown" if unknown else "failed",
                                error=error,
                                submission_state=state,
                            )
                        )
                except BaseException:
                    # The pre-send tombstone still forbids a repeated POST. Do not let
                    # cleanup failures replace an acknowledged or ambiguous result.
                    pass
                if isinstance(exc, GrokOAuthError):
                    exc.submission_state = state
                if isinstance(exc, OutcomeUnknown) or (started and isinstance(exc, TimeoutError)):
                    raise OutcomeUnknown(operation_id=job.job_id, submission_state=state) from None
            if isinstance(exc, TimeoutError):
                raise Busy(
                    "Video submission deadline expired before sending",
                    operation_id=job.job_id if job is not None else "",
                    submission_state=state,
                ) from None
            raise

    async def poll(self, job_id, *, scope: AssetScope, timeout=30) -> VideoJob:  # noqa: ASYNC109 - absolute deadline

        deadline = time.monotonic() + _timeout(timeout, maximum=120)
        async with self._slot(deadline):
            job = await self.check_job(job_id, scope=scope)
            if job.status in {"done", "failed", "unknown"}:
                return job
            if not _REQUEST_ID.fullmatch(job.request_id):
                raise ProtocolError("Video task ID is invalid")
            lock = self._job_locks.setdefault(job_id, asyncio.Lock())
            async with lock:
                job = await self.check_job(job_id, scope=scope)
                if job.status == "done":
                    return job
                binding = await self.video_store.job_binding(job_id, scope=scope)
                generation = getattr(self.oauth, "binding_generation", 0)
                response = await self.http.request_json(
                    "GET",
                    f"/videos/{job.request_id}",
                    account_binding=binding,
                    binding_generation=generation,
                    policy=RequestPolicy(
                        capability="videos",
                        deadline=deadline,
                        safe_pre_send_retries=1,
                        side_effecting=False,
                    ),
                )
                if not isinstance(response, dict):
                    raise ProtocolError("Video polling response is invalid")
                status = response.get("status")
                if status in {"pending", "processing", "queued", "running"}:
                    return job
                if status in {"failed", "error", "expired"}:
                    return await _durable(
                        self.video_store.update_job(
                            job_id, scope=scope, status="failed", error="VideoGenerationFailed"
                        )
                    )
                if status != "done":
                    raise ProtocolError("Video polling status is invalid")
                video = response.get("video")
                if not isinstance(video, dict) or not isinstance(video.get("url"), str):
                    raise ProtocolError("Video completion has no download URL")
                # URLs are transient: failed downloads can re-poll the same task without a new POST.
                await self.check_job(job_id, scope=scope)
                asset = await self.video_store.store_url(
                    video["url"],
                    scope=scope,
                    deadline=deadline,
                    binding=await self.video_store.job_binding(job_id, scope=scope),
                )
                await self.check_job(job_id, scope=scope)
                return await _durable(
                    self.video_store.update_job(
                        job_id,
                        scope=scope,
                        status="done",
                        asset_id=asset.asset_id,
                        duration=asset.duration,
                        error="",
                    )
                )

    async def wait(self, job_id, *, scope: AssetScope, timeout=300, poll_interval=5) -> VideoJob:  # noqa: ASYNC109 - absolute deadline

        deadline = time.monotonic() + _timeout(timeout)
        interval = _timeout(poll_interval, maximum=30)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return await self.check_job(job_id, scope=scope)
            try:
                job = await self.poll(job_id, scope=scope, timeout=min(remaining, 120))
            except TimeoutError:
                return await self.check_job(job_id, scope=scope)
            if job.status in {"done", "failed", "unknown"}:
                return job
            await asyncio.sleep(min(interval, max(0, deadline - time.monotonic())))

    async def close(self):
        self._closed = True
        active = [task for task in self._active if task is not asyncio.current_task()]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
