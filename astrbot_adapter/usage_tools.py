"""Account-information permissions, independent of inference capability access."""

import copy
import json
from datetime import datetime

from ..grok_oauth.errors import GrokOAuthError, PermissionDenied, ServiceClosed
from .compat import FunctionTool, ToolSet, selected_chat_provider
from .provider import is_runtime_provider
from .tool_scope import (
    USAGE_BREAKDOWN_TOOL_NAME,
    USAGE_TOOL_NAME,
    USAGE_TOOL_NAMES,
    consume_usage_call,
)


def require_usage_access(runtime, event, *, require_admin=True):
    if require_admin and not event.is_admin():
        raise PermissionDenied("额度仅允许管理员查看。")
    if not event.is_private_chat():
        allowed = runtime.config.get("usage_group_allowlist", [])
        if not isinstance(allowed, list) or event.unified_msg_origin not in allowed:
            raise PermissionDenied("当前群会话未开放额度查询。")


def _usage_heading(data, title):
    labels = {
        "success": "可用",
        "unknown": "额度信息不完整",
        "partial": "部分来源信息不完整",
        "unparseable": "额度信息无法解析",
        "unbound": "未绑定",
        "reauth_required": "授权需要更新",
        "forbidden": "账户无额度查询权限",
        "identity_unavailable": "缺少账户身份，请重新进行设备授权",
        "rate_limited": "查询限流，请稍后重试",
        "unavailable": "额度服务暂不可用",
        "authorization_changed": "授权已变更，请重新查询",
        "closed": "服务已关闭",
        "persistence_error": "授权存储异常",
    }
    lines = [title + labels.get(data["status"], "暂不可用")]
    if data.get("stale"):
        lines.append("以下为历史快照，不代表当前额度。")
    return lines


def _usage_footer(data):
    lines = []
    lines.append(
        "周期：" + {"weekly": "每周", "monthly": "每月"}.get(data.get("period_type"), "未知")
    )
    lines.append("共享范围：" + ("账户共享" if data.get("scope") == "account_shared" else "未确认"))
    lines.append("重置时间：" + _local_time(data.get("reset_at")))
    lines.append("采集时间：" + _local_time(data.get("observed_at")))
    if data.get("cached"):
        lines.append("数据来自缓存。")
    return lines


def _local_time(value):
    """Format an absolute timestamp in the process's local zone at that instant."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return "未知"
        local = parsed.astimezone()
        offset = local.strftime("%z")
        return local.strftime("%Y-%m-%d %H:%M:%S") + " UTC" + offset[:3] + ":" + offset[3:]
    except (AttributeError, TypeError, ValueError, OverflowError, OSError):
        return "未知"


def format_usage(data):
    lines = _usage_heading(data, "Grok 额度：")
    for key, title in (("used_percent", "已用"), ("remaining_percent", "剩余")):
        value = data.get(key)
        lines.append(f"{title}：{value:g}%" if value is not None else f"{title}：未知")
    return "\n".join(lines + _usage_footer(data))


def format_usage_breakdown(data):
    lines = _usage_heading(data, "Grok 额度消耗来源：")
    products = data.get("products", [])
    for product in products:
        value = product["usage_percent"]
        text = f"{value:g}%" if value is not None else "未知"
        lines.append(product["display_name"] + "：" + text)
    if not products:
        lines.append("暂无可用的产品来源数据。")
    unknown = data.get("unrecognized_product_count", 0)
    if unknown:
        lines.append(f"另有 {unknown} 项未识别来源。")
    lines.append("百分比为上游原值，不代表各产品独立额度，也不保证合计为 100%。")
    return "\n".join(lines + _usage_footer(data))


class GrokUsageFunctionTool(FunctionTool):
    """Plugin-owned registration type."""


class UsageToolSet(ToolSet):
    """Resolve each issued name to its own handler; shared tools never change."""

    def __init__(self, tools, service, event, authorization=None):
        super().__init__(tools)
        self._usage_service = service
        self._usage_event = event
        self._usage_authorization = authorization or (
            service.runtime.oauth.epoch,
            service.runtime.oauth.binding_generation,
        )

    def get_tool(self, name):
        tool = super().get_tool(name)
        if tool is None or name not in USAGE_TOOL_NAMES:
            return tool
        bound = copy.copy(tool)
        service, expected_event, authorization = (
            self._usage_service,
            self._usage_event,
            self._usage_authorization,
        )

        async def execute(event, **arguments):
            return await service.run(
                event,
                arguments,
                call_name=name,
                expected_event=expected_event,
                expected_authorization=authorization,
            )

        bound.handler = execute
        return bound

    def get_light_tool_set(self):
        return UsageToolSet(
            super().get_light_tool_set().tools,
            self._usage_service,
            self._usage_event,
            self._usage_authorization,
        )

    def get_param_only_tool_set(self):
        return UsageToolSet(
            super().get_param_only_tool_set().tools,
            self._usage_service,
            self._usage_event,
            self._usage_authorization,
        )


class UsageToolService:
    def __init__(self, context, runtime):
        self.context = context
        self.runtime = runtime
        self.closed = False

    def build_tools(self):
        descriptions = {
            USAGE_TOOL_NAME: "查询 Grok 额度总量：已用、剩余、周期及重置时间。用户询问‘Grok 额度’时使用此工具。仅在明确询问时调用，不包含产品消耗来源。",
            USAGE_BREAKDOWN_TOOL_NAME: "查询 Grok 额度消耗来源：Grok、Grok Build、Imagine 各产品的用量百分比。用户询问‘Grok 额度消耗来源’或各产品用了多少时使用此工具。按上游 usagePercent 原值回复，不归一化为总和 100%，不推断各产品独立额度；缺失值不是零。",
        }
        return [
            GrokUsageFunctionTool(
                name=name,
                description=description
                + " Exclusively through Grok OAuth models in private or configured group sessions. Empty arguments. Unknown values are not zero; stale data is historical.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                handler=self._global_handler,
            )
            for name, description in descriptions.items()
        ]

    def _check(self, event):
        if self.closed or self.runtime.closed:
            raise ServiceClosed()
        require_usage_access(self.runtime, event, require_admin=False)
        if not self.runtime.config.get("usage_tools_enabled", True):
            raise PermissionDenied()

    async def prepare(self, event, req):
        if not req.func_tool:
            return
        tools = list(req.func_tool.tools)
        if not any(tool.name in USAGE_TOOL_NAMES for tool in tools):
            return
        allowed = False
        try:
            self._check(event)
            current = await selected_chat_provider(self.context, event)
            allowed = is_runtime_provider(current, self.runtime)
        except GrokOAuthError:
            pass
        selected = [
            tool for tool in tools if tool.name not in USAGE_TOOL_NAMES or (allowed and tool.active)
        ]
        req.func_tool = UsageToolSet(selected, self, event)

    async def _global_handler(self, event, **arguments):
        # A bare executor has no request/name proof and cannot query the account.
        return await self.run(event, arguments)

    async def run(
        self, event, arguments, *, call_name=None, expected_event=None, expected_authorization=None
    ):
        try:
            self._check(event)
            if arguments or expected_event is None or event is not expected_event:
                raise PermissionDenied()
            if expected_authorization != (
                self.runtime.oauth.epoch,
                self.runtime.oauth.binding_generation,
            ):
                raise PermissionDenied()
            provider = consume_usage_call(self.runtime, call_name)
            if not is_runtime_provider(provider, self.runtime):
                raise PermissionDenied()
            data = (
                await provider.get_usage_breakdown()
                if call_name == USAGE_BREAKDOWN_TOOL_NAME
                else await provider.get_usage()
            )
            return json.dumps(data, ensure_ascii=False)
        except GrokOAuthError:
            return json.dumps({"status": "denied"})

    async def close(self):
        self.closed = True
