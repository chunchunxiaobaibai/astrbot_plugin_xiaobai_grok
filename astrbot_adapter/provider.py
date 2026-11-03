"""Grok OAuth provider implementing AstrBot's Responses provider contract."""

import asyncio
import copy
import math
import time
import uuid
from contextlib import aclosing, asynccontextmanager

from ..grok_oauth.errors import (
    GrokOAuthError,
    InvalidRequest,
    OutcomeUnknown,
    PermissionDenied,
    ProtocolError,
    ServiceClosed,
    StreamIncomplete,
    UnsupportedModelParameter,
)
from ..grok_oauth.models import AssetScope, ImageRequest, RequestPolicy
from ..grok_oauth.responses import encode_messages
from ..grok_oauth.search import SEARCH_TOOL_NAME, web_search_tool
from .compat import (
    LLMResponse,
    ProviderBase,
    ProviderOpenAIResponses,
    inherited_oauth_web_search,
    to_llm_response,
)
from .registry_state import get_runtime
from .tool_scope import (
    IMAGE_TOOL_NAMES,
    OWNED_TOOL_NAMES,
    USAGE_TOOL_NAMES,
    VIDEO_TOOL_NAMES,
    issue_tool_call,
    issue_usage_call,
)


class GrokOAuthProvider(ProviderOpenAIResponses):
    """Use the host's provider interface with an independent OAuth transport."""

    def __init__(self, provider_config, provider_settings):
        # Do not construct an SDK client or inherit key rotation/retry behavior.
        ProviderBase.__init__(self, copy.deepcopy(provider_config), provider_settings)
        if provider_config.get("grok_account_slot", "default") != "default":
            raise InvalidRequest("Only the default account slot is supported")
        if (
            provider_config.get("api_base", "https://api.x.ai/v1").rstrip("/")
            != "https://api.x.ai/v1"
        ):
            raise InvalidRequest("Grok API destination is fixed")
        self.set_model(provider_config.get("model") or "grok-4.6")
        self._runtime = get_runtime()
        # Core routes image requests using provider_config, not capabilities.
        # Keep explicit administrator choices, including the legacy empty list.
        if self.provider_config.get("modalities") is None:
            capabilities = self.capabilities
            if capabilities["chat"]["model_support"] == "available":
                modalities = ["text"]
                for capability, modality in (("vision", "image"), ("function_tools", "tool_use")):
                    declared = capabilities[capability]
                    if (
                        declared["implementation"]
                        and declared["enabled"]
                        and declared["model_support"] == "available"
                    ):
                        modalities.append(modality)
                self.provider_config["modalities"] = modalities
        self._closed = False
        self._tasks = set()
        self._runtime.providers.add(self)

    @property
    def capabilities(self):
        return self._runtime.catalog.capabilities(
            self.get_model(),
            images_enabled=False,
            search_enabled=False,
            videos_enabled=False,
        )

    def get_current_key(self):
        return "oauth-managed"

    def get_keys(self):
        return ["oauth-managed"]

    def _log_completion(self, *, mode, payload, result):
        http = getattr(self._runtime, "http", None)
        if http is not None:
            http.log_completion(
                mode=mode,
                provider_id=self.provider_config.get("id"),
                configured_model=payload["model"],
                result=result,
            )

    def set_key(self, key):
        if key != "oauth-managed":
            raise InvalidRequest("Credentials are managed by the OAuth service")

    async def get_usage(self, *, force_refresh=False):
        """Backend API; callers enforce the account-information access policy."""
        async with self._request():
            return await self._runtime.get_usage(force_refresh=force_refresh)

    async def get_usage_breakdown(self, *, force_refresh=False):
        """Product percentages from the same account snapshot and access policy."""
        async with self._request():
            return await self._runtime.get_usage_breakdown(force_refresh=force_refresh)

    async def get_models(self):
        return self._runtime.catalog.chat_models()

    @asynccontextmanager
    async def _request(self):
        if self._closed or self._runtime.closed:
            raise ServiceClosed("Grok provider is closed")
        task = asyncio.current_task()
        self._tasks.add(task)
        try:
            yield
        finally:
            self._tasks.discard(task)

    async def _payload(
        self,
        *,
        prompt=None,
        image_urls=None,
        audio_urls=None,
        func_tool=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        extra_user_content_parts=None,
        tool_choice="auto",
        oauth_web_search=None,
        search_allowed_domains=None,
        retry_rate_limits=None,
        **kwargs,
    ):
        # Native Dashboard stores per-model inference settings in this field.
        configured = self.provider_config.get("custom_extra_body")
        configured = {} if configured is None else copy.deepcopy(configured)
        if not isinstance(configured, dict) or set(configured) - {
            "reasoning",
            "reasoning_effort",
            "temperature",
            "top_p",
            "max_tokens",
            "max_output_tokens",
        }:
            raise UnsupportedModelParameter("Unsupported custom model parameter")
        if "max_tokens" in configured:
            if "max_output_tokens" in configured:
                raise UnsupportedModelParameter("Conflicting output token limits")
            configured["max_output_tokens"] = configured.pop("max_tokens")
        if "reasoning" in kwargs or "reasoning_effort" in kwargs:
            configured.pop("reasoning", None)
            configured.pop("reasoning_effort", None)
        kwargs = {**configured, **kwargs}
        # Host request controls are consumed here, never sent as model parameters.
        # The transport never retries 429, even when the host permits it.
        if retry_rate_limits is not None and type(retry_rate_limits) is not bool:
            raise InvalidRequest("retry_rate_limits must be a boolean")
        search = inherited_oauth_web_search() if oauth_web_search is None else oauth_web_search
        if not isinstance(search, str) or search not in {"inherit", "disabled", "cached", "live"}:
            raise InvalidRequest("Invalid OAuth web search policy")
        if search == "cached":
            raise UnsupportedModelParameter("Cached OAuth web search is not supported")
        native_search = search == "live"
        if native_search and not self._runtime.config.get("search_enabled", True):
            raise PermissionDenied("Grok web search is disabled")
        if search_allowed_domains is not None and not native_search:
            raise InvalidRequest("Search domain filters require live search")
        native_tool = web_search_tool(search_allowed_domains) if native_search else None
        unsupported = set(kwargs) - {
            "reasoning",
            "reasoning_effort",
            "temperature",
            "top_p",
            "max_output_tokens",
            "native_tools",
            "abort_signal",
        }
        if unsupported:
            raise UnsupportedModelParameter("Unsupported request parameter")
        if audio_urls or kwargs.get("native_tools"):
            raise UnsupportedModelParameter("Requested media or native tool capability is disabled")
        history = copy.deepcopy(self._ensure_message_to_dicts(contexts))
        if system_prompt:
            history.insert(0, {"role": "system", "content": system_prompt})
        if prompt is not None or image_urls or extra_user_content_parts:
            parts = []
            if prompt:
                parts.append({"type": "text", "text": prompt})
            parts.extend(
                part.model_dump() if hasattr(part, "model_dump") else copy.deepcopy(part)
                for part in (extra_user_content_parts or [])
            )
            parts.extend(
                {"type": "image_url", "image_url": {"url": image}} for image in (image_urls or [])
            )
            history.append({"role": "user", "content": parts})
        if tool_calls_result:
            values = (
                tool_calls_result if isinstance(tool_calls_result, list) else [tool_calls_result]
            )
            for value in values:
                history.extend(value.to_openai_messages())
        scope = AssetScope("provider", str(self.provider_config.get("id")), uuid.uuid4().hex)
        for message in history:
            if not isinstance(message.get("content"), list):
                continue
            for part in message["content"]:
                if part.get("type") in {"image_url", "input_image"}:
                    image = part.get("image_url")
                    ref = image.get("url") if isinstance(image, dict) else image
                    asset = await self._runtime.assets.import_reference(
                        ref, scope=scope, allowed_roots=self._runtime.allowed_roots
                    )
                    encoded = await self._runtime.assets.to_data_uri(asset, scope=scope)
                    part["image_url"] = {"url": encoded} if part["type"] == "image_url" else encoded
        selected_model = model or self.get_model()
        payload = {
            "model": selected_model,
            "input": encode_messages(history),
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        reasoning = kwargs.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict) or set(reasoning) != {"effort"}:
                raise UnsupportedModelParameter("Only reasoning.effort is supported")
            effort = reasoning["effort"]
        else:
            effort = kwargs.get(
                "reasoning_effort",
                self.provider_config.get(
                    "reasoning_effort", self._runtime.config.get("reasoning_effort")
                ),
            )
        validated = self._runtime.catalog.validate_reasoning(selected_model, effort)
        if validated:
            payload["reasoning"] = validated
        for name in ("temperature", "top_p", "max_output_tokens"):
            value = kwargs.get(name, self.provider_config.get(name))
            if value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise UnsupportedModelParameter("Invalid numeric request parameter")
            if name == "max_output_tokens" and (type(value) is not int or value <= 0):
                raise UnsupportedModelParameter("Invalid output token limit")
            if name == "top_p" and not 0 <= value <= 1:
                raise UnsupportedModelParameter("Invalid top_p")
            if name == "temperature" and not 0 <= value <= 2:
                raise UnsupportedModelParameter("Invalid temperature")
            payload[name] = value
        tools = []
        if func_tool:
            tools = [
                {"type": "function", **copy.deepcopy(item["function"])}
                for item in func_tool.openai_schema()
                if not (
                    item["function"]["name"] == SEARCH_TOOL_NAME
                    and (
                        search == "disabled"
                        or native_search
                        or not self._runtime.config.get("search_enabled", True)
                    )
                )
            ]
        if not self._runtime.config.get("images_enabled", True) or not self._runtime.config.get(
            "tools_enabled", True
        ):
            tools = [tool for tool in tools if tool.get("name") not in IMAGE_TOOL_NAMES]
        if not self._runtime.config.get("videos_enabled", True) or not self._runtime.config.get(
            "tools_enabled", True
        ):
            tools = [tool for tool in tools if tool.get("name") not in VIDEO_TOOL_NAMES]
        if not self._runtime.config.get("search_tools_enabled", True):
            tools = [tool for tool in tools if tool.get("name") != SEARCH_TOOL_NAME]
        if not self._runtime.config.get("usage_tools_enabled", True):
            tools = [tool for tool in tools if tool.get("name") not in USAGE_TOOL_NAMES]
        if native_tool:
            tools.append(native_tool)
            payload["include"].append("web_search_call.action.sources")
        if tool_choice not in {"auto", "required", "none"}:
            raise UnsupportedModelParameter("Unsupported tool choice")
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        elif tool_choice == "required":
            raise InvalidRequest("Tool choice requires available tools")
        return payload

    def _response(self, payload, result):
        allow_search = payload.get("tool_choice") != "none" and any(
            tool.get("type") == "web_search" for tool in payload.get("tools", [])
        )
        exposed_functions = {
            tool.get("name") for tool in payload.get("tools", []) if tool.get("type") == "function"
        }
        if any(
            call["name"] in OWNED_TOOL_NAMES
            and (call["name"] not in exposed_functions or payload.get("tool_choice") == "none")
            for call in result.function_calls
        ):
            raise PermissionDenied("Grok tool was not enabled for this request")
        response = to_llm_response(result, allow_web_search=allow_search)
        for index, (name, arguments) in enumerate(
            zip(response.tools_call_name, response.tools_call_args)
        ):
            if name in USAGE_TOOL_NAMES:
                if arguments:
                    raise InvalidRequest("Usage tool accepts no arguments")
                response.tools_call_name[index] = issue_usage_call(self, name)
            if name in OWNED_TOOL_NAMES:
                issue_tool_call(self, name, arguments)
        if allow_search and result.native_items:
            self._runtime.catalog.observe(payload["model"], "web_search", "available")
        return response

    async def search_web(
        self,
        query,
        *,
        allowed_domains=None,
        timeout=None,  # noqa: ASYNC109 - public SDK request deadline
        oauth_web_search="live",
    ):
        if oauth_web_search == "cached":
            raise UnsupportedModelParameter("Cached OAuth web search is not supported")
        if oauth_web_search == "disabled" or not self._runtime.config.get("search_enabled", True):
            raise PermissionDenied("Grok web search is disabled")
        if oauth_web_search != "live":
            raise InvalidRequest("Explicit search requires the live policy")
        policy = self._policy(
            self._runtime.config.get("search_timeout", 60) if timeout is None else timeout
        )
        try:
            async with self._request(), asyncio.timeout_at(policy.deadline):
                result = await self._runtime.search.search(
                    query, model=self.get_model(), policy=policy, allowed_domains=allowed_domains
                )
                self._runtime.catalog.observe(self.get_model(), "web_search", "available")
                return result
        except TimeoutError:
            raise ProtocolError("Web search timed out") from None

    def _policy(self, timeout):
        value = self.provider_config.get("timeout", 180) if timeout is None else timeout
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 600:
            raise InvalidRequest("Timeout must be between 0 and 600 seconds")
        return RequestPolicy(deadline=time.monotonic() + value)

    async def text_chat(
        self,
        prompt=None,
        session_id=None,
        image_urls=None,
        audio_urls=None,
        func_tool=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        extra_user_content_parts=None,
        tool_choice="auto",
        request_max_retries=None,
        *,
        conversation_id: str | None = None,
        **kwargs,
    ):
        # Host-local correlation only. Explicit contexts carry conversation history;
        # never forward this identifier or retain it on a shared provider instance.
        policy = self._policy(kwargs.pop("timeout", None))
        async with self._request(), asyncio.timeout_at(policy.deadline):
            payload = await self._payload(
                prompt=prompt,
                image_urls=image_urls,
                audio_urls=audio_urls,
                func_tool=func_tool,
                contexts=contexts,
                system_prompt=system_prompt,
                tool_calls_result=tool_calls_result,
                model=model,
                extra_user_content_parts=extra_user_content_parts,
                tool_choice=tool_choice,
                **kwargs,
            )
            result = await self._runtime.responses.create(payload, policy=policy)
            response = self._response(payload, result)
            self._runtime.catalog.observe(payload["model"], "chat", "available")
            self._log_completion(mode="chat", payload=payload, result=result)
            return response

    async def text_chat_stream(
        self,
        prompt=None,
        session_id=None,
        image_urls=None,
        audio_urls=None,
        func_tool=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        extra_user_content_parts=None,
        tool_choice="auto",
        request_max_retries=None,
        *,
        conversation_id: str | None = None,
        **kwargs,
    ):
        # Host-local correlation only. Explicit contexts carry conversation history;
        # never forward this identifier or retain it on a shared provider instance.
        policy = self._policy(kwargs.pop("timeout", None))
        queue = asyncio.Queue(maxsize=8)
        terminal = {"error": None}

        async def produce():
            partial = False
            text_streamed = False
            search_tail = ""
            final = None
            completed_result = None
            try:
                async with self._request():
                    async with asyncio.timeout_at(policy.deadline):
                        payload = await self._payload(
                            prompt=prompt,
                            image_urls=image_urls,
                            audio_urls=audio_urls,
                            func_tool=func_tool,
                            contexts=contexts,
                            system_prompt=system_prompt,
                            tool_calls_result=tool_calls_result,
                            model=model,
                            extra_user_content_parts=extra_user_content_parts,
                            tool_choice=tool_choice,
                            **kwargs,
                        )
                        async with aclosing(
                            self._runtime.responses.stream(payload, policy=policy)
                        ) as stream:
                            async for event in stream:
                                if event.kind == "text_delta":
                                    text_streamed = text_streamed or bool(event.delta)
                                    partial = partial or bool(event.delta)
                                    await queue.put(
                                        LLMResponse(
                                            "assistant", completion_text=event.delta, is_chunk=True
                                        )
                                    )
                                elif event.kind == "reasoning_delta":
                                    partial = partial or bool(event.delta)
                                    await queue.put(
                                        LLMResponse(
                                            "assistant",
                                            reasoning_content=event.delta,
                                            is_chunk=True,
                                        )
                                    )
                                elif event.kind == "completed":
                                    completed_result = event.result
                                    final = self._response(payload, event.result)
                                    if event.result.native_items:
                                        # Sources can arrive only with response.completed.
                                        # Emit them for hosts which discard nonchunk aggregates.
                                        search_tail = (
                                            (final.completion_text or "")[len(event.result.text) :]
                                            if text_streamed
                                            else (final.completion_text or "")
                                        )
                        if final is None:
                            raise StreamIncomplete("Missing complete response", partial=partial)
                    # The HTTP response has completed and closed. Slow downstream
                    # delivery must not turn this successful result into a timeout.
                    self._runtime.catalog.observe(payload["model"], "chat", "available")
                    self._log_completion(mode="stream", payload=payload, result=completed_result)
                    if search_tail:
                        await queue.put(
                            LLMResponse("assistant", completion_text=search_tail, is_chunk=True)
                        )
                    await queue.put(final)
            except BaseException as exc:
                if isinstance(exc, TimeoutError):
                    exc = StreamIncomplete("Response stream timed out", partial=partial)
                if isinstance(exc, GrokOAuthError):
                    exc.partial = exc.partial or partial
                terminal["error"] = exc

        # Each host anext() may run in a distinct task; one producer owns the
        # complete request. Terminal state never occupies a data queue slot.
        producer = asyncio.create_task(produce(), name="grok-response-stream")
        try:
            while True:
                if terminal["error"] is not None:
                    raise terminal["error"]
                if not queue.empty():
                    yield queue.get_nowait()
                    continue
                if producer.done():
                    break
                take = asyncio.create_task(queue.get())
                try:
                    await asyncio.wait((producer, take), return_when=asyncio.FIRST_COMPLETED)
                    if terminal["error"] is not None:
                        raise terminal["error"]
                    if take.done():
                        yield take.result()
                finally:
                    if not take.done():
                        take.cancel()
                    await asyncio.gather(take, return_exceptions=True)
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def generate_image(
        self,
        prompt,
        *,
        model=None,
        size=None,
        aspect_ratio=None,
        resolution=None,
        n=1,
        timeout=None,
        reference_images=None,
        action=None,
    ):
        raise InvalidRequest("Image generation is disabled in this plugin")

    async def generate_for_scope(self, request, *, scope):
        raise InvalidRequest("Image generation is disabled in this plugin")

    async def terminate(self):
        if self._closed:
            return
        self._closed = True
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def is_runtime_provider(provider, runtime):
    """Shared identity check; capability-specific access policies stay with callers."""
    return (
        isinstance(provider, GrokOAuthProvider)
        and provider._runtime is runtime
        and not provider._closed
    )
