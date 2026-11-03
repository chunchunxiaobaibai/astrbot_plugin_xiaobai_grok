"""xAI Images request encoding and bounded result materialization."""

from __future__ import annotations

import asyncio
import base64
import binascii
import math
import time
from contextlib import AsyncExitStack
from typing import Any, Protocol

from .errors import Busy, ImageTooLarge, InvalidImageRequest, OutcomeUnknown, ProtocolError
from .media import AssetStore
from .models import AssetScope, GeneratedImage, ImageRequest, RequestPolicy

DEFAULT_IMAGE_MODEL = "grok-imagine-image-2.0"
_CURRENT_MODEL = "grok-imagine-image-2.0"
_SIZE_MAP = {
    "1024x1024": ("1:1", "1k"),
    "2048x2048": ("1:1", "2k"),
    "auto": ("auto", None),
}
_ASPECT_RATIOS = {"auto", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "5:2"}
_RESOLUTIONS = {"1k", "2k"}


class AuthorizedHttp(Protocol):
    async def request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        policy: RequestPolicy,
    ) -> dict: ...


class ImageService:
    def __init__(
        self,
        http: AuthorizedHttp,
        assets: AssetStore,
        *,
        max_running: int = 2,
        max_pending: int = 8,
    ) -> None:
        if isinstance(max_running, bool) or not isinstance(max_running, int) or max_running <= 0:
            raise ValueError("max_running must be a positive integer")
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 0:
            raise ValueError("max_pending must be a non-negative integer")
        self.http = http
        self.assets = assets
        self._running = asyncio.Semaphore(max_running)
        self._capacity = max_running + max_pending
        self._admission_lock = asyncio.Lock()
        self._admitted = 0

    async def _admit(self) -> None:
        async with self._admission_lock:
            if self._admitted >= self._capacity:
                raise Busy("Image queue is full")
            self._admitted += 1

    def _validate(self, request: ImageRequest) -> tuple[str, str, dict[str, Any], float]:
        if not isinstance(request, ImageRequest):
            raise TypeError("request must be ImageRequest")
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise InvalidImageRequest("prompt must be a non-empty string")
        if isinstance(request.n, bool) or not isinstance(request.n, int) or not 1 <= request.n <= 4:
            raise InvalidImageRequest("n must be an integer from 1 through 4")
        if isinstance(request.timeout, bool) or not isinstance(request.timeout, (int, float)):
            raise InvalidImageRequest("timeout must be a number")
        timeout = float(request.timeout)
        if not math.isfinite(timeout) or timeout <= 0 or timeout > 600:
            raise InvalidImageRequest("timeout must be finite and between 0 and 600 seconds")
        if isinstance(request.reference_images, (str, bytes)) or not isinstance(
            request.reference_images, tuple
        ):
            raise InvalidImageRequest("reference_images must be a tuple")
        if not all(
            isinstance(reference, str) and reference for reference in request.reference_images
        ):
            raise InvalidImageRequest("reference_images must contain non-empty strings")

        if request.action is not None and not isinstance(request.action, str):
            raise InvalidImageRequest("action must be generate or edit")
        action = request.action or ("edit" if request.reference_images else "generate")
        if action not in {"generate", "edit"}:
            raise InvalidImageRequest("action must be generate or edit")
        if action == "edit" and not request.reference_images:
            raise InvalidImageRequest("editing requires at least one reference image")
        if action == "generate" and request.reference_images:
            raise InvalidImageRequest("generation cannot include reference images")

        model = request.model or DEFAULT_IMAGE_MODEL
        if not isinstance(model, str) or not model.startswith("grok-imagine-image"):
            raise InvalidImageRequest("model is not an Imagine image model")
        reference_limit = 5 if model == _CURRENT_MODEL else 3
        if len(request.reference_images) > reference_limit:
            raise InvalidImageRequest(f"model supports at most {reference_limit} reference images")

        geometry: dict[str, Any] = {}
        if request.size is not None:
            if not isinstance(request.size, str):
                raise InvalidImageRequest("size must be a string")
            if request.size not in _SIZE_MAP:
                raise InvalidImageRequest("unsupported size compatibility value")
            if request.aspect_ratio is not None or request.resolution is not None:
                raise InvalidImageRequest("size conflicts with aspect_ratio or resolution")
            aspect_ratio, resolution = _SIZE_MAP[request.size]
            geometry["aspect_ratio"] = aspect_ratio
            if resolution:
                geometry["resolution"] = resolution
        else:
            if request.aspect_ratio is not None:
                if not isinstance(request.aspect_ratio, str):
                    raise InvalidImageRequest("aspect_ratio must be a string")
                if request.aspect_ratio not in _ASPECT_RATIOS:
                    raise InvalidImageRequest("unsupported aspect_ratio")
                geometry["aspect_ratio"] = request.aspect_ratio
            if request.resolution is not None:
                if not isinstance(request.resolution, str):
                    raise InvalidImageRequest("resolution must be a string")
                if request.resolution not in _RESOLUTIONS:
                    raise InvalidImageRequest("unsupported resolution")
                geometry["resolution"] = request.resolution
        return action, model, geometry, timeout

    async def generate(
        self,
        request: ImageRequest,
        *,
        scope: AssetScope,
    ) -> list[GeneratedImage]:
        action, model, geometry, timeout = self._validate(request)
        deadline = time.monotonic() + timeout
        await self._admit()
        acquired = False
        request_started = False
        try:
            try:
                async with asyncio.timeout_at(deadline):
                    await self._running.acquire()
                    acquired = True
                    body: dict[str, Any] = {
                        "prompt": request.prompt,
                        "model": model,
                        "n": request.n,
                        "response_format": "b64_json",
                        **geometry,
                    }
                    path = "/images/generations"
                    async with AsyncExitStack() as stack:
                        if action == "edit":
                            encoded: list[dict[str, str]] = []
                            input_bytes = 0
                            for asset_id in request.reference_images:
                                # Keep the lease through the upstream request so cleanup
                                # cannot remove an input after it is materialized.
                                image_path = await stack.enter_async_context(
                                    self.assets.lease(asset_id, scope=scope)
                                )
                                input_bytes += (await asyncio.to_thread(image_path.stat)).st_size
                                if input_bytes > self.assets.max_decoded_bytes:
                                    raise ImageTooLarge("Reference batch exceeds the byte limit")
                                encoded.append(
                                    {
                                        "type": "image_url",
                                        "url": await self.assets.to_data_uri(asset_id, scope=scope),
                                    }
                                )
                            body["image" if len(encoded) == 1 else "images"] = (
                                encoded[0] if len(encoded) == 1 else encoded
                            )
                            path = "/images/edits"
                        policy = RequestPolicy(
                            capability="images",
                            deadline=deadline,
                            safe_pre_send_retries=0,
                            allow_one_auth_retry=True,
                            side_effecting=True,
                        )
                        request_started = True
                        response = await self.http.request_json(
                            "POST", path, json=body, policy=policy
                        )
                    return await self._materialize(
                        response, request=request, model=model, scope=scope, deadline=deadline
                    )
            except TimeoutError as exc:
                if request_started:
                    raise OutcomeUnknown("Image request outcome is unknown") from exc
                raise Busy("Image request deadline expired in queue") from exc
        finally:
            if acquired:
                self._running.release()
            # No await is needed: event-loop tasks cannot interleave this update.
            self._admitted -= 1

    async def _materialize(
        self,
        response: dict,
        *,
        request: ImageRequest,
        model: str,
        scope: AssetScope,
        deadline: float,
    ) -> list[GeneratedImage]:
        if not isinstance(response, dict) or not isinstance(response.get("data"), list):
            raise ProtocolError("Images response data is invalid")
        items = response["data"]
        if not items:
            raise ProtocolError("Images response is empty")
        results: list[GeneratedImage] = []
        decoded_total = 0
        request_id = response.get("id") if isinstance(response.get("id"), str) else ""
        operation_id = (
            response.get("operation_id") if isinstance(response.get("operation_id"), str) else ""
        )
        response_model = response.get("model") if isinstance(response.get("model"), str) else model
        try:
            for index, item in enumerate(items):
                if time.monotonic() >= deadline:
                    raise TimeoutError
                if not isinstance(item, dict):
                    raise ProtocolError("Image result item is invalid")
                item_id = item.get("id") if isinstance(item.get("id"), str) else str(index)
                encoded = item.get("b64_json")
                if isinstance(encoded, str):
                    try:
                        data = base64.b64decode(encoded, validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise ProtocolError("Image result base64 is invalid") from exc
                    decoded_total += len(data)
                    if decoded_total > self.assets.max_decoded_bytes:
                        raise ProtocolError("Image batch exceeds the decoded byte limit")
                    stored = await self.assets.store_bytes(
                        data, scope=scope, request_id=request_id, item_id=item_id
                    )
                elif isinstance(item.get("url"), str):
                    stored = await self.assets.store_url(
                        item["url"],
                        scope=scope,
                        request_id=request_id,
                        item_id=item_id,
                        deadline=deadline,
                    )
                else:
                    raise ProtocolError("Image result has no image payload")
                results.append(
                    GeneratedImage(
                        path=stored.path,
                        mime_type=stored.mime_type,
                        revised_prompt=item.get("revised_prompt", "")
                        if isinstance(item.get("revised_prompt", ""), str)
                        else "",
                        raw={"index": index},
                        asset_id=stored.asset_id,
                        request_id=request_id,
                        model=response_model,
                        width=stored.width,
                        height=stored.height,
                    )
                )
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                if results:
                    raise OutcomeUnknown(
                        "Image result processing was cancelled",
                        request_id=request_id,
                        operation_id=operation_id,
                        partial=True,
                        assets=results,
                    ) from exc
                raise
            if results:
                if isinstance(exc, TimeoutError):
                    raise OutcomeUnknown(
                        "Image result processing exceeded its deadline",
                        request_id=request_id,
                        operation_id=operation_id,
                        partial=True,
                        assets=results,
                    ) from exc
                raise ProtocolError(
                    "Image batch was only partially stored",
                    request_id=request_id,
                    operation_id=operation_id,
                    partial=True,
                    assets=results,
                ) from exc
            if isinstance(exc, TimeoutError):
                raise OutcomeUnknown(
                    "Image result processing exceeded its deadline",
                    request_id=request_id,
                    operation_id=operation_id,
                ) from exc
            raise
        if len(results) != request.n:
            raise ProtocolError(
                "Images response count does not match n",
                request_id=request_id,
                operation_id=operation_id,
                partial=True,
                assets=results,
            )
        return results
