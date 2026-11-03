"""AstrBot plugin entry: Grok OAuth, usage, and chat provider."""

import asyncio
from pathlib import Path

from .astrbot_adapter.compat import (
    AstrBotConfig,
    AstrMessageEvent,
    Context,
    Star,
    detach_owned_providers,
    filter,
    get_astrbot_plugin_data_path,
    get_astrbot_temp_path,
    install_owned_tools,
    migrate_legacy_provider_sources,
    remove_owned_tools,
    restore_owned_providers,
)
from .astrbot_adapter.diagnostics import format_oauth_diagnostic
from .astrbot_adapter.provider import GrokOAuthProvider
from .astrbot_adapter.registration import register_provider, unregister_provider
from .astrbot_adapter.registry_state import bind_runtime, clear_runtime
from .astrbot_adapter.runtime import GrokRuntime, _finish_owned_operation
from .astrbot_adapter.usage_tools import (
    UsageToolService,
    format_usage,
    format_usage_breakdown,
    require_usage_access,
)
from .astrbot_adapter.web_api import GrokWebAPI
from .grok_oauth.errors import GrokOAuthError, PermissionDenied


class GrokOAuthPlugin(Star):
    name = "astrbot_plugin_xiaobai_grok"

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)
        self.config = config
        self.runtime = None
        self.usage_tools = None
        self.web_api = None
        self.registration = None
        self._tools = []
        self._lifecycle = asyncio.Lock()
        self._close_task = None

    async def initialize(self):
        async with self._lifecycle:
            if self.runtime and not self.runtime.closed:
                return
            root = Path(get_astrbot_plugin_data_path()) / self.name
            roots = [Path(get_astrbot_temp_path())]
            roots[0].mkdir(parents=True, exist_ok=True)
            for key in (
                "images_enabled",
                "videos_enabled",
                "tools_enabled",
                "search_enabled",
                "search_tools_enabled",
            ):
                self.config[key] = False
            self.runtime = GrokRuntime(
                self.config,
                root,
                allowed_roots=roots,
                host_context=self.context,
            )
            try:
                await self.runtime.open()
                bind_runtime(self.runtime)
                self.registration = register_provider(self.runtime.owner_id, GrokOAuthProvider)
                self.usage_tools = UsageToolService(self.context, self.runtime)
                self._tools = self.usage_tools.build_tools()
                install_owned_tools(self.context, self._tools)
                self.web_api = GrokWebAPI(self.context, self.runtime, logger=self.logger)
                self.web_api.register()
                migrated = await migrate_legacy_provider_sources(self.context)
                if migrated:
                    self.logger.info(
                        "Normalized %d Grok models for native provider-source configuration",
                        migrated,
                    )
                await restore_owned_providers(self.context, self.runtime)
                self.logger.info("Grok OAuth provider and control endpoints initialized")
            except BaseException:
                await _finish_owned_operation(self._cleanup())
                raise

    async def terminate(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._terminate(), name="grok-plugin-close")
        await _finish_owned_operation(self._close_task)

    async def _terminate(self):
        async with self._lifecycle:
            await self._cleanup()

    async def _cleanup(self):
        errors = []
        if self.runtime:
            try:
                await detach_owned_providers(self.context, self.runtime)
            except Exception as error:
                errors.append(type(error).__name__)
        if self.usage_tools:
            await self.usage_tools.close()
        if self.web_api:
            self.web_api.unregister()
        remove_owned_tools(self.context, self._tools)
        if self.runtime:
            try:
                await self.runtime.close()
            except Exception as error:
                errors.append(type(error).__name__)
            clear_runtime(self.runtime)
        if self.registration:
            unregister_provider(self.registration)
        if errors:
            raise RuntimeError("Grok plugin resource cleanup failed")

    @staticmethod
    def _admin(event):
        if not event.is_admin() or not event.is_private_chat():
            raise PermissionDenied("此命令仅允许管理员在私聊使用。")
        return f"{event.get_platform_id()}:{event.get_sender_id()}"

    @filter.command("grok_oauth_login")
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def login(self, event: AstrMessageEvent, confirmation: str = ""):
        try:
            owner = self._admin(event)
            profile = self.runtime.client_description()
            if confirmation != "confirm":
                yield event.plain_result(
                    "客户端："
                    + profile["client_profile"]
                    + "\nID："
                    + profile["client_id"]
                    + "\n默认客户端来自公开 Grok CLI 参考实现，是否适用于本插件及当前账号需要实际确认。"
                    + "\n确认使用此客户端后，发送 /grok_oauth_login confirm。"
                )
                return
            result = await self.runtime.start_flow(
                owner,
                confirmed_client_id=profile["client_id"],
                client_profile=profile["client_profile"],
            )
            if result["status"] == "pending":
                yield event.plain_result(
                    "打开 "
                    + result["verification_uri"]
                    + "\n输入代码："
                    + result["user_code"]
                    + "\n完成后发送 /grok_oauth_status 查看结果。"
                )
            else:
                yield event.plain_result("授权状态：" + result["status"])
        except GrokOAuthError as error:
            yield event.plain_result(
                "Grok OAuth："
                + error.code
                + format_oauth_diagnostic(getattr(error, "oauth_diagnostic", None))
            )

    @filter.command("grok_oauth_usage")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def usage(self, event: AstrMessageEvent):
        try:
            require_usage_access(self.runtime, event)
            yield event.plain_result(format_usage(await self.runtime.get_usage()))
        except GrokOAuthError as error:
            yield event.plain_result("Grok 额度：" + error.code)

    @filter.command("grok_oauth_usage_breakdown")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def usage_breakdown(self, event: AstrMessageEvent):
        try:
            require_usage_access(self.runtime, event)
            yield event.plain_result(
                format_usage_breakdown(await self.runtime.get_usage_breakdown())
            )
        except GrokOAuthError as error:
            yield event.plain_result("Grok 额度消耗来源：" + error.code)

    @filter.command("grok_oauth_status")
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def status(self, event: AstrMessageEvent):
        try:
            result = self.runtime.status(self._admin(event))
            yield event.plain_result(
                "授权状态："
                + result["status"]
                + ("；" + result["error"] if result.get("error") else "")
                + format_oauth_diagnostic(result.get("diagnostic"))
            )
        except GrokOAuthError as error:
            yield event.plain_result(
                "Grok OAuth："
                + error.code
                + format_oauth_diagnostic(getattr(error, "oauth_diagnostic", None))
            )

    @filter.command("grok_oauth_cancel")
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def cancel(self, event: AstrMessageEvent):
        try:
            owner = self._admin(event)
            state = self.runtime.status(owner)
            if state.get("flow_id"):
                result = await self.runtime.cancel_flow(owner, state["flow_id"])
                yield event.plain_result("授权状态：" + result["status"])
            else:
                yield event.plain_result("当前没有设备码授权流程。")
        except GrokOAuthError as error:
            yield event.plain_result(
                "Grok OAuth："
                + error.code
                + format_oauth_diagnostic(getattr(error, "oauth_diagnostic", None))
            )

    @filter.command("grok_oauth_disconnect")
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def disconnect(self, event: AstrMessageEvent):
        try:
            await self.runtime.disconnect(self._admin(event))
            yield event.plain_result("Grok OAuth 绑定已解除，图片缓存保留。")
        except GrokOAuthError as error:
            yield event.plain_result(
                "Grok OAuth："
                + error.code
                + format_oauth_diagnostic(getattr(error, "oauth_diagnostic", None))
            )

    @filter.command("grok_oauth_test")
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def test_chat(self, event: AstrMessageEvent, provider_id: str):
        try:
            self._admin(event)
            provider = self.context.get_provider_by_id(provider_id)
            if not isinstance(provider, GrokOAuthProvider) or provider._runtime is not self.runtime:
                raise PermissionDenied()
            response = await provider.text_chat(prompt="Reply PONG only.")
            yield event.plain_result("Grok 测试成功，响应 ID：" + str(response.id))
        except GrokOAuthError as error:
            yield event.plain_result(
                "Grok OAuth："
                + error.code
                + format_oauth_diagnostic(getattr(error, "oauth_diagnostic", None))
            )

    @filter.on_llm_request()
    async def prepare_tools(self, event: AstrMessageEvent, req):
        if self.usage_tools:
            await self.usage_tools.prepare(event, req)
