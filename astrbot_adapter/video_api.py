"""Public caller-owned video operations for explicitly selected Grok providers."""

from contextlib import asynccontextmanager
from hashlib import sha256

from ..grok_oauth.errors import (
    AuthorizationChanged,
    GrokOAuthError,
    InvalidRequest,
    PermissionDenied,
    ServiceClosed,
    UnsafeMediaSource,
    VideoUnsupported,
)
from ..grok_oauth.models import AssetScope
from ..grok_oauth.video_store import DEFAULT_VIDEO_SOURCE_HOSTS, MAX_EDIT_DURATION, MAX_VIDEO_BYTES
from ..grok_oauth.videos import VideoRequest


class VideoProviderAPI:
    """Generate and read scoped video assets; callers own scheduling and delivery."""

    async def _video_scope(self, event, *, require_available=True):
        if require_available and not self._runtime.config.get("videos_enabled", True):
            raise PermissionDenied("Grok videos are disabled")
        if require_available and (
            self._runtime.videos is None or self._runtime.video_assets is None
        ):
            raise ServiceClosed("Grok video service is not ready")
        context = getattr(self._runtime, "host_context", None)
        if context is None:
            raise InvalidRequest("A host context is required for video operations")
        try:
            platform = event.get_platform_id()
            umo = event.unified_msg_origin
        except (AttributeError, TypeError) as exc:
            raise InvalidRequest("A message event is required for video operations") from exc
        if not all(isinstance(value, str) and value.strip() for value in (platform, umo)):
            raise InvalidRequest("A current session is required for video operations")
        parts = umo.split(":", 2)
        if len(parts) != 3 or parts[0] != platform or not all(parts):
            raise InvalidRequest("The message origin must match its platform")
        conversation = await context.conversation_manager.get_curr_conversation_id(umo)
        if not isinstance(conversation, str) or not conversation.strip():
            raise InvalidRequest("A current conversation is required for video operations")
        return AssetScope(platform, umo, conversation)

    def _video_operation_key(self, operation_key):
        if (
            not isinstance(operation_key, str)
            or not operation_key.strip()
            or len(operation_key) > 256
        ):
            raise InvalidRequest(
                "A stable video operation key of at most 256 characters is required"
            )
        provider_id = self.provider_config.get("id")
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise InvalidRequest("An explicit Grok provider ID is required")
        # Keep the service key bounded without persisting caller IDs or message text.
        return "sdk:" + sha256((provider_id + "\0" + operation_key).encode()).hexdigest()

    @staticmethod
    def _public_video_job(job):
        return {**job.public(), "delivery_owner": "caller"}

    async def get_video_capabilities(self, *, event) -> dict:
        """Describe implemented modes without refreshing credentials or querying the network."""
        async with self._request():
            await self._video_scope(event, require_available=False)
            self._video_operation_key("capabilities")
            enabled = bool(self._runtime.config.get("videos_enabled", True))
            ready = self._runtime.videos is not None and self._runtime.video_assets is not None
            authorization = getattr(self._runtime.oauth, "status", "unbound")
            available = enabled and ready and authorization == "authorized"
            implemented = ["image_to_video", "text_to_video", "video_edit", "external_video_import"]
            configured_hosts = self._runtime.config.get("video_source_hosts", ())
            source_hosts = sorted(set(DEFAULT_VIDEO_SOURCE_HOSTS) | set(configured_hosts))
            return {
                "schema_version": 1,
                "available": available,
                "enabled": enabled,
                "authorization": authorization,
                "account_entitlement": "unknown",
                "implemented_modes": implemented,
                "modes": implemented.copy() if available else [],
                "unsupported_modes": {
                    "video_edit_with_images": "upstream_reference_edit_combination_unsupported"
                },
                "durations": [6, 10],
                "resolution": "480p",
                "max_reference_images": 1,
                "max_video_bytes": MAX_VIDEO_BYTES,
                "max_edit_duration": MAX_EDIT_DURATION,
                "mode_constraints": {
                    "image_to_video": {
                        "reference_images": 1,
                        "durations": [6, 10],
                        "resolution": "480p",
                    },
                    "text_to_video": {
                        "reference_images": 0,
                        "durations": [6, 10],
                        "resolution": "480p",
                    },
                    "video_edit": {
                        "max_output_resolution": "720p",
                        "preserves_source_duration": True,
                        "preserves_source_aspect_ratio": True,
                        "reference_images": 0,
                        "source_videos": 1,
                        "max_duration": MAX_EDIT_DURATION,
                        "max_video_bytes": MAX_VIDEO_BYTES,
                    },
                    "external_video_import": {
                        "mime_type": "video/mp4",
                        "max_duration": MAX_EDIT_DURATION,
                        "max_video_bytes": MAX_VIDEO_BYTES,
                        "source_types": ["data_uri", "https"]
                        + (
                            ["local_file"]
                            if getattr(self._runtime, "allowed_video_roots", ())
                            else []
                        ),
                        "https_hosts": source_hosts,
                    },
                },
                "delivery_owner": "caller",
            }

    @asynccontextmanager
    async def _video_submission(self):
        """Keep all preflight failures explicit, including a closed Provider lifecycle."""
        try:
            async with self._request():
                yield
        except GrokOAuthError as exc:
            if not hasattr(exc, "submission_state"):
                exc.submission_state = "not_submitted"
            raise

    async def _submit_video_request(self, request, *, event, operation_key):
        backend_entered = False
        try:
            scope = await self._video_scope(event)
            key = self._video_operation_key(operation_key)
            backend_entered = True
            job = await self._runtime.videos.submit(request, scope=scope, operation_key=key)
            return self._public_video_job(job)
        except GrokOAuthError as exc:
            if not hasattr(exc, "submission_state"):
                exc.submission_state = "unknown" if backend_entered else "not_submitted"
            raise

    async def submit_text_video(
        self, prompt: str, *, event, operation_key: str, duration: int = 6
    ) -> dict:
        """Generate directly from unchanged text; no image call or synthesized reference."""
        async with self._video_submission():
            return await self._submit_video_request(
                VideoRequest(prompt, action="text", duration=duration),
                event=event,
                operation_key=operation_key,
            )

    async def import_video_source(self, reference: str, *, event) -> str:
        """Import a bounded external MP4 into this account and current conversation."""
        async with self._request():
            scope = await self._video_scope(event)
            binding = await self._runtime.videos._binding()
            generation = getattr(self._runtime.oauth, "binding_generation", 0)
            asset_id = await self._runtime.video_assets.import_reference(
                reference,
                scope=scope,
                binding=binding,
                allowed_roots=getattr(self._runtime, "allowed_video_roots", ()),
            )
            if (
                binding != await self._runtime.videos._binding()
                or getattr(self._runtime.oauth, "binding_generation", 0) != generation
            ):
                raise AuthorizationChanged()
            return asset_id

    async def edit_video_with_images(
        self,
        prompt: str,
        reference_asset_id: str,
        reference_image_asset_ids: list[str],
        *,
        event,
        operation_key: str,
    ) -> dict:
        """Reject unsupported combined reference/edit input before any upstream request."""
        async with self._video_submission():
            await self._video_scope(event)
            self._video_operation_key(operation_key)
            raise VideoUnsupported(
                "upstream_reference_edit_combination_unsupported", submission_state="not_submitted"
            )

    async def import_video_image(self, reference: str, *, event) -> str:
        """Import an image into the event's current conversation and return its asset ID."""
        async with self._request():
            scope = await self._video_scope(event)
            return await self._runtime.assets.import_reference(
                reference, scope=scope, allowed_roots=self._runtime.allowed_roots
            )

    async def submit_video(
        self, prompt: str, reference_asset_id: str, *, event, operation_key: str, duration: int = 6
    ) -> dict:
        """Submit once per stable caller key; a retry returns the same durable job."""
        async with self._video_submission():
            return await self._submit_video_request(
                VideoRequest(prompt, reference_asset_id, duration=duration),
                event=event,
                operation_key=operation_key,
            )

    async def edit_video(
        self, prompt: str, reference_asset_id: str, *, event, operation_key: str
    ) -> dict:
        """Edit a scoped MP4 owned by the current account and event conversation."""
        async with self._video_submission():
            return await self._submit_video_request(
                VideoRequest(prompt, reference_asset_id, action="edit"),
                event=event,
                operation_key=operation_key,
            )

    async def get_video_job(self, job_id: str, *, event, timeout=30) -> dict:  # noqa: ASYNC109 - backend bounds the timeout
        """Poll once and materialize a completed result without sending it."""
        async with self._request():
            scope = await self._video_scope(event)
            job = await self._runtime.videos.poll(job_id, scope=scope, timeout=timeout)
            return self._public_video_job(job)

    async def wait_video_job(self, job_id: str, *, event, timeout=300) -> dict:  # noqa: ASYNC109 - backend bounds the timeout
        """Wait within a bounded budget; a pending result can be queried again later."""
        async with self._request():
            scope = await self._video_scope(event)
            job = await self._runtime.videos.wait(job_id, scope=scope, timeout=timeout)
            return self._public_video_job(job)

    async def read_video_bytes(self, job_id: str, *, event) -> bytes:
        """Read a completed MP4 of at most 20 MiB, without exposing paths or signed URLs."""
        async with self._request():
            scope = await self._video_scope(event)
            generation = getattr(self._runtime.oauth, "binding_generation", 0)
            job = await self._runtime.videos.check_job(job_id, scope=scope)
            if job.status != "done" or not job.asset_id:
                raise InvalidRequest("Only a completed video job can be read")
            store = self._runtime.video_assets
            async with store.lease(job.asset_id, scope=scope):
                asset = await store.get_asset(job.asset_id, scope=scope)
                if asset.size_bytes > MAX_VIDEO_BYTES:
                    raise UnsafeMediaSource("Video exceeds the 20 MiB delivery limit")
                content = await store.read_bytes(job.asset_id, scope=scope)
                if not isinstance(content, bytes) or not 0 < len(content) <= MAX_VIDEO_BYTES:
                    raise UnsafeMediaSource("Video exceeds the supported byte range")
                await self._runtime.videos.check_job(job_id, scope=scope)
                if getattr(self._runtime.oauth, "binding_generation", 0) != generation:
                    raise AuthorizationChanged()
                if self._closed or self._runtime.closed:
                    raise ServiceClosed("Grok provider is closed")
                return content
