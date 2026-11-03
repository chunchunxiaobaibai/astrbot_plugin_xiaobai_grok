"""Plugin-owned services, device-flow ownership and bounded lifecycle."""

import asyncio
import copy
import uuid
import weakref
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..grok_oauth.billing import BillingClient
from ..grok_oauth.catalog import ModelCatalog
from ..grok_oauth.credentials import OAuthService
from ..grok_oauth.errors import (
    AuthorizationDenied,
    DeviceCodeExpired,
    GrokOAuthError,
    InvalidRequest,
    PermissionDenied,
    ServiceClosed,
)
from ..grok_oauth.http import AuthorizedHttp, validate_connect_timeout
from ..grok_oauth.images import ImageService
from ..grok_oauth.media import AssetStore
from ..grok_oauth.oauth import OAuthWireClient, safe_oauth_diagnostic
from ..grok_oauth.responses import ResponsesClient
from ..grok_oauth.search import SearchClient
from ..grok_oauth.token_store import TokenStore
from ..grok_oauth.version import USER_AGENT
from ..grok_oauth.video_store import VideoStore
from ..grok_oauth.videos import VideoService
from .diagnostics import TransportDiagnostics

DEFAULT_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
DEFAULT_SCOPE = "openid profile email offline_access grok-cli:access api:access"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def normalize_outbound_proxy(value):
    """Accept host:port and rewrite loopback https:// to http:// for Clash mixed-port."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("proxy must be a string")
    text = value.strip()
    if not text:
        return None
    if "://" not in text:
        text = "http://" + text
    parsed = urlsplit(text)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https", "socks5", "socks5h"}:
        raise ValueError("proxy must be an http, https, or socks5 URL")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("proxy host is required")
    if scheme == "https" and host in _LOOPBACK_HOSTS:
        scheme = "http"
    return urlunsplit(parsed._replace(scheme=scheme))


async def _finish_owned_operation(operation):
    task = asyncio.ensure_future(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError()
    return result


class GrokRuntime:
    def __init__(
        self,
        config,
        data_root,
        *,
        transport=None,
        allowed_roots=(),
        diagnostic_logger=None,
        host_context=None,
    ):
        self.config = copy.deepcopy(dict(config))
        self.host_context = host_context
        self.data_root = Path(data_root)
        self.owner_id = uuid.uuid4().hex
        self.client_id = self.config.get("oauth_client_id") or DEFAULT_CLIENT_ID
        self.client_profile = self.config.get("oauth_client_profile") or "public-grok-cli-reference"
        self.allowed_roots = tuple(Path(root) for root in allowed_roots)
        self.allowed_video_roots = tuple(
            Path(root) for root in self.config.get("allowed_video_roots", [])
        )
        self.closed = False
        self._opened = False
        self._control = asyncio.Lock()
        self._lifecycle = asyncio.Lock()
        self._close_task = None
        self._poll_task = None
        self._active_flow = None
        self._flow_info = None
        self._flow_state = "unbound"
        self._flow_error = None
        self._flow_diagnostic = None
        self._flow_diagnostic_owner = None
        self.tasks = set()
        self.providers = weakref.WeakSet()
        self.catalog = ModelCatalog()
        self.assets = None
        self.images = None
        self.video_assets = None
        self.videos = None
        proxy = normalize_outbound_proxy(self.config.get("proxy"))
        connect_timeout = validate_connect_timeout(self.config.get("transport_connect_timeout", 10))
        self.outbound_proxy = proxy
        self.client = httpx.AsyncClient(
            transport=transport,
            proxy=proxy,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(30),
            headers={"User-Agent": USER_AGENT},
        )
        native_diagnostics = self.config.get("transport_diagnostics", False) is True
        file_diagnostics = self.config.get("transport_debug_file", False) is True
        self.diagnostics = TransportDiagnostics(
            self.data_root,
            native_enabled=native_diagnostics,
            file_enabled=file_diagnostics,
            native_logger=diagnostic_logger,
        )
        self.wire = OAuthWireClient(
            self.client,
            client_id=self.client_id,
            scope=DEFAULT_SCOPE,
            diagnostic_logging=native_diagnostics or file_diagnostics,
            diagnostic_logger=self.diagnostics,
            flow_diagnostic=self._update_flow_diagnostic,
        )
        self.store = TokenStore(self.data_root / "credentials.json")
        self.oauth = OAuthService(self.store, self.wire.refresh)
        self.http = AuthorizedHttp(
            self.client,
            self.oauth,
            diagnostic_logging=native_diagnostics or file_diagnostics,
            diagnostic_logger=self.diagnostics,
            connect_timeout=connect_timeout,
        )
        self.responses = ResponsesClient(self.http)
        self.search = SearchClient(self.responses)
        self.billing = BillingClient(self.client, self.oauth)

    async def get_usage(self, *, force_refresh=False):
        self._require_open()
        return (await self.billing.get_usage(force_refresh=force_refresh)).to_dict()

    async def get_usage_breakdown(self, *, force_refresh=False):
        self._require_open()
        return (await self.billing.get_usage(force_refresh=force_refresh)).to_breakdown_dict()

    async def open(self):
        async with self._lifecycle:
            await self._open()

    async def _construct_assets(self):
        self.assets = await asyncio.to_thread(
            AssetStore,
            self.data_root / "assets",
            media_hosts=self.config.get("media_hosts", []),
        )

    async def _construct_video_assets(self):
        self.video_assets = await asyncio.to_thread(
            VideoStore,
            self.data_root / "videos",
            media_hosts=self.config.get("media_hosts", []),
            result_proxy=self.outbound_proxy,
            source_proxy=self.outbound_proxy,
            source_media_hosts=self.config.get("video_source_hosts", []),
        )

    async def _open(self):
        if self.closed:
            raise ServiceClosed()
        if self._opened:
            return
        try:
            await self.oauth.open()
            await _finish_owned_operation(self._construct_assets())
            self.allowed_roots += (self.assets.root,)
            self.images = ImageService(self.http, self.assets)
            await _finish_owned_operation(self._construct_video_assets())
            self.videos = VideoService(self.http, self.assets, self.video_assets, oauth=self.oauth)
            self._opened = True
        except BaseException:
            self.closed = True
            try:
                await _finish_owned_operation(self.oauth.close())
            finally:
                try:
                    if self.video_assets is not None:
                        await _finish_owned_operation(self.video_assets.close())
                    if self.assets is not None:
                        await _finish_owned_operation(self.assets.close())
                finally:
                    try:
                        await _finish_owned_operation(self.client.aclose())
                    finally:
                        await _finish_owned_operation(self.diagnostics.close())
            raise

    def _require_open(self):
        if self.closed or not self._opened:
            raise ServiceClosed("Grok plugin is not ready")

    def client_description(self):
        return {
            "client_id": self.client_id,
            "client_profile": self.client_profile,
            "scope": DEFAULT_SCOPE,
            "notice": "Reference public Grok CLI client; eligibility for this plugin and account capabilities require verification.",
        }

    def status(self, owner, flow_id=None):
        self._require_open()
        info = self._flow_info
        if flow_id is not None:
            if not info or flow_id != info["flow_id"] or owner != info["owner"]:
                raise PermissionDenied("Authorization flow is not owned by this administrator")
        elif self._flow_state == "pending" and info and info["owner"] != owner:
            raise PermissionDenied("Authorization flow is owned by another administrator")
        state = self.oauth.status
        if self._flow_state in {"pending", "cancelled", "denied", "expired", "error"}:
            state = self._flow_state
        result = {"status": state, "account_slot": "default"}
        if info and info["owner"] == owner:
            result.update({key: info[key] for key in ("flow_id", "expires_at")})
            if state == "pending":
                result.update({key: info[key] for key in ("user_code", "verification_uri")})
        if self._flow_error:
            result["error"] = self._flow_error
        if self._flow_diagnostic_owner == owner and self._flow_diagnostic:
            result["diagnostic"] = dict(self._flow_diagnostic)
        return result

    def _update_flow_diagnostic(self, flow, detail):
        # Refresh requests never enter this callback; identity also rejects an old flow.
        if self._active_flow is flow:
            self._flow_diagnostic = safe_oauth_diagnostic(detail)
            self._flow_diagnostic_owner = flow.owner_id

    async def start_flow(self, owner, *, confirmed_client_id=None, client_profile=None):
        self._require_open()
        if not isinstance(owner, str) or not owner:
            raise PermissionDenied()
        if confirmed_client_id != self.client_id or client_profile != self.client_profile:
            raise InvalidRequest(
                "Confirm the current OAuth client ID and profile before authorization"
            )
        async with self._control:
            self._require_open()
            if self._flow_state == "pending":
                return self.status(owner)
            if self.oauth.status == "authorized":
                return self.status(owner)
            self._flow_diagnostic = None
            self._flow_diagnostic_owner = None
            self._flow_info = None
            self._flow_error = None
            self._flow_state = "unbound"
            try:
                flow = await self.wire.start_device_flow(owner_id=owner, epoch=self.oauth.epoch)
            except GrokOAuthError as error:
                self._flow_state, self._flow_error = "error", error.code
                self._flow_diagnostic = safe_oauth_diagnostic(
                    getattr(error, "oauth_diagnostic", None)
                )
                self._flow_diagnostic_owner = owner
                raise
            self._active_flow = flow
            self._flow_info = {
                "flow_id": flow.flow_id,
                "owner": owner,
                "expires_at": flow.expires_at,
                "user_code": flow.user_code,
                "verification_uri": flow.verification_uri,
            }
            self._flow_state = "pending"
            self._flow_error = None
            self._poll_task = asyncio.create_task(self._run_flow(flow), name="grok-device-flow")
            self.tasks.add(self._poll_task)
            self._poll_task.add_done_callback(self.tasks.discard)
            return self.status(owner)

    async def _run_flow(self, flow):
        try:
            token = await self.wire.poll_device_flow(flow)
            await self.oauth.bind(token, expected_epoch=flow.epoch)
            if self._active_flow is flow:
                self._flow_state = "authorized"
                self._flow_diagnostic = None
            # Model discovery is explicitly separated from any media/paid test.
            try:
                await self.catalog.refresh(self.http)
            except GrokOAuthError:
                pass  # Authorization remains valid if catalog discovery is denied.
        except asyncio.CancelledError:
            if self._active_flow is flow and self._flow_state == "pending":
                self._flow_state = "cancelled"
            raise
        except GrokOAuthError as error:
            if self._active_flow is flow:
                # A later bind/store error must not reuse a successful token diagnostic.
                if self._flow_diagnostic and self._flow_diagnostic.get("outcome") == "success":
                    self._flow_diagnostic = None
                self._flow_state = (
                    "denied"
                    if isinstance(error, AuthorizationDenied)
                    else "expired"
                    if isinstance(error, DeviceCodeExpired)
                    else "error"
                )
                self._flow_error = error.code
        except Exception:
            if self._active_flow is flow:
                self._flow_diagnostic = None
                self._flow_state = "error"
                self._flow_error = "ProtocolError"
        finally:
            if self._active_flow is flow:
                self._active_flow = None
            if self._flow_info and self._flow_info["flow_id"] == flow.flow_id:
                self._flow_info.pop("user_code", None)
                self._flow_info.pop("verification_uri", None)

    async def wait_for_flow(self):
        if self._poll_task:
            await asyncio.shield(self._poll_task)

    async def _cancel_pending(self):
        task = self._poll_task
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._active_flow = None

    async def cancel_flow(self, owner, flow_id):
        self._require_open()
        async with self._control:
            self.status(owner, flow_id)
            if self._flow_state == "pending":
                # Invalidate the epoch before waiting for the poller to stop.
                await self.oauth.disconnect()
                self._flow_state = "cancelled"
                await self._cancel_pending()
            return self.status(owner, flow_id)

    async def disconnect(self, owner):
        self._require_open()
        if not owner:
            raise PermissionDenied()
        async with self._control:
            await self.oauth.disconnect()
            self.billing.invalidate()
            self._flow_state = "unbound"
            await self._cancel_pending()
            self._flow_info = None
            self._flow_error = None
            self._flow_diagnostic = None
            self._flow_diagnostic_owner = None
            return {"status": "unbound", "account_slot": "default"}

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="grok-runtime-close")
        await _finish_owned_operation(self._close_task)

    async def _close(self):
        async with self._lifecycle:
            self.closed = True
            errors = []
            async with self._control:
                operations = [provider.terminate for provider in list(self.providers)]
                if self.videos is not None:
                    operations.append(self.videos.close)
                if self.video_assets is not None:
                    operations.append(self.video_assets.close)
                operations.extend((self.billing.close, self.oauth.close, self._cancel_pending))
                if self.assets is not None:
                    operations.append(self.assets.close)
                operations.extend((self.client.aclose, self.diagnostics.close))
                for operation in operations:
                    try:
                        await operation()
                    except Exception as error:
                        errors.append(type(error).__name__)
                self.tasks.clear()
                self._opened = False
            if errors:
                raise ServiceClosed("Runtime cleanup failed; see component status") from None
