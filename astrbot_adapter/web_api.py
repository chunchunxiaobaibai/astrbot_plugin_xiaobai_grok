"""Dashboard-authenticated control endpoints with no credential serialization."""

from ..grok_oauth.errors import (
    AuthenticationRequired,
    GrokOAuthError,
    InvalidRequest,
    PermissionDenied,
    RateLimited,
    ServiceClosed,
)
from ..grok_oauth.oauth import safe_oauth_diagnostic
from .compat import json_response, request
from .provider import GrokOAuthProvider


class GrokWebAPI:
    def __init__(self, context, runtime, *, logger=None):
        self.context = context
        self.runtime = runtime
        self.logger = logger
        self.handlers = []
        for action, method in [
            ("auth/client", "GET"),
            ("auth/start", "POST"),
            ("auth/status", "GET"),
            ("auth/cancel", "POST"),
            ("auth/disconnect", "POST"),
            ("test/chat", "POST"),
            ("capabilities", "GET"),
            ("usage", "GET"),
        ]:

            async def handler(action=action):
                return await self.handle(action)

            for prefix in ("/xiaobai-grok/", "/astrbot_plugin_xiaobai_grok/"):
                self.handlers.append((prefix + action, handler, [method], "Grok OAuth control"))

    def register(self):
        existing = {item[0] for item in self.context.registered_web_apis}
        if any(item[0] in existing for item in self.handlers):
            raise InvalidRequest("Grok management route already registered")
        for path, handler, methods, description in self.handlers:
            self.context.register_web_api(path, handler, methods, description)

    def unregister(self):
        handlers = {id(item[1]) for item in self.handlers}
        self.context.registered_web_apis[:] = [
            item for item in self.context.registered_web_apis if id(item[1]) not in handlers
        ]

    async def handle(self, action):
        try:
            owner = request.username
            if not isinstance(owner, str) or not owner:
                raise AuthenticationRequired("Dashboard administrator authentication is required")
            if request.method == "POST":
                if (request.content_type or "").split(";", 1)[0] != "application/json":
                    raise InvalidRequest("A JSON request body is required")
                raw = await request.body()
                if len(raw) > 16 * 1024:
                    raise InvalidRequest("Management request exceeds size limit")
                payload = await request.json()
                if not isinstance(payload, dict):
                    raise InvalidRequest("A JSON object is required")
            else:
                payload = dict(request.query.items())
            allowed = {
                "auth/client": set(),
                "auth/start": {"account_slot", "client_profile", "confirmed_client_id"},
                "auth/status": {"flow_id"},
                "auth/cancel": {"flow_id"},
                "auth/disconnect": {"account_slot"},
                "test/chat": {"provider_id", "run"},
                "capabilities": {"provider_id"},
                "usage": {"refresh"},
            }[action]
            if set(payload) - allowed or payload.get("account_slot", "default") != "default":
                raise InvalidRequest("Unsupported management parameter")
            if action == "usage":
                if payload.get("refresh", "false") not in {"true", "false"}:
                    raise InvalidRequest("refresh must be true or false")
                result = await self.runtime.get_usage(
                    force_refresh=payload.get("refresh") == "true"
                )
            elif action == "auth/client":
                result = self.runtime.client_description()
            elif action == "auth/start":
                result = await self.runtime.start_flow(
                    owner,
                    confirmed_client_id=payload.get("confirmed_client_id"),
                    client_profile=payload.get("client_profile"),
                )
            elif action == "auth/status":
                result = self.runtime.status(owner, payload.get("flow_id"))
            elif action == "auth/cancel":
                if not isinstance(payload.get("flow_id"), str):
                    raise InvalidRequest("flow_id is required")
                result = await self.runtime.cancel_flow(owner, payload["flow_id"])
            elif action == "auth/disconnect":
                result = await self.runtime.disconnect(owner)
            else:
                provider = self.context.get_provider_by_id(payload.get("provider_id"))
                if (
                    not isinstance(provider, GrokOAuthProvider)
                    or provider._runtime is not self.runtime
                ):
                    raise InvalidRequest("Select a configured Grok OAuth provider")
                if action == "capabilities":
                    result = provider.capabilities
                elif payload.get("run") is not True:
                    raise InvalidRequest("Explicit run=true is required for a live test")
                elif action == "test/chat":
                    response = await provider.text_chat(prompt="Reply PONG only.")
                    result = {
                        "status": "success",
                        "model": provider.get_model(),
                        "request_id": response.id,
                    }
                else:
                    raise InvalidRequest("Unknown management action")
            return json_response({"status": "ok", "data": result})
        except GrokOAuthError as error:
            status = (
                401
                if isinstance(error, AuthenticationRequired)
                else 403
                if isinstance(error, PermissionDenied)
                else 429
                if isinstance(error, RateLimited)
                else 503
                if isinstance(error, ServiceClosed)
                else 400
                if isinstance(error, InvalidRequest)
                else 502
            )
            payload = {"status": "error", **error.summary()}
            detail = safe_oauth_diagnostic(getattr(error, "oauth_diagnostic", None))
            if detail:
                payload["diagnostic"] = detail
            return json_response(payload, status_code=status)
        except Exception as error:
            if self.logger:
                self.logger.error("Grok management failure: %s", type(error).__name__)
            return json_response({"status": "error", "error": "InternalError"}, status_code=500)
