"""Scoped Agent image tools and controlled current-message asset discovery."""

import copy
import hashlib
import json
import uuid
from dataclasses import asdict

from ..grok_oauth.errors import GrokOAuthError, InvalidRequest, PermissionDenied, ServiceClosed
from ..grok_oauth.models import AssetScope, ImageRequest
from .compat import FunctionTool, Image, Reply, selected_chat_provider
from .provider import is_runtime_provider
from .tool_scope import IMAGE_TOOL_NAMES, consume_tool_call

TOOL_NAMES = IMAGE_TOOL_NAMES
_BASE_PROPERTIES = {
    "prompt": {
        "type": "string",
        "minLength": 1,
        "description": "Describe the desired image or edit.",
    },
    "n": {"type": "integer", "minimum": 1, "maximum": 4},
    "aspect_ratio": {
        "type": "string",
        "enum": ["auto", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "5:2"],
    },
    "resolution": {"type": "string", "enum": ["1k", "2k"]},
}


class GrokImageFunctionTool(FunctionTool):
    """Keep host tool ownership attached to this plugin module."""


class ImageToolService:
    def __init__(self, context, runtime, delivery):
        self.context = context
        self.runtime = runtime
        self.delivery = delivery
        self.closed = False

    async def resolve_provider(self, event, *, from_tool=False, calling_provider=None):
        if self.closed or self.runtime.closed:
            raise ServiceClosed()
        if not self.runtime.config.get("images_enabled", True) or not self.runtime.config.get(
            "tools_enabled", True
        ):
            raise PermissionDenied("Image tools are disabled")
        selected = self.runtime.config.get("tools_provider_id")
        if selected:
            provider = self.context.get_provider_by_id(selected)
        elif from_tool:
            provider = calling_provider
        else:
            provider = await selected_chat_provider(self.context, event)
        if not is_runtime_provider(provider, self.runtime):
            raise PermissionDenied("Select a Grok image provider in plugin settings")
        return provider

    async def scope(self, event):
        platform = event.get_platform_id()
        umo = event.unified_msg_origin
        conversation = await self.context.conversation_manager.get_curr_conversation_id(umo)
        if not all(isinstance(value, str) and value for value in (platform, umo, conversation)):
            raise InvalidRequest("A current conversation is required for image tools")
        return AssetScope(platform, umo, conversation)

    def build_tools(self):
        generate = GrokImageFunctionTool(
            name=TOOL_NAMES[0],
            description="Generate an image with Grok and deliver it as a real attachment. Repeating the same request in this message reuses its stored result. Available to Grok OAuth chat models by default; other chat models require explicit administrator configuration.",
            parameters={
                "type": "object",
                "properties": copy.deepcopy(_BASE_PROPERTIES),
                "required": ["prompt"],
                "additionalProperties": False,
            },
            handler=self._tool_generate,
        )
        edit_properties = copy.deepcopy(_BASE_PROPERTIES)
        edit_properties["reference_asset_ids"] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
            "description": "Current-conversation asset IDs from earlier image results or attached-image discovery, in input order. Never pass paths or URLs.",
        }
        edit = GrokImageFunctionTool(
            name=TOOL_NAMES[1],
            description="Edit current-conversation images with Grok and deliver the result as an attachment. Available to Grok OAuth chat models by default; other chat models require explicit administrator configuration.",
            parameters={
                "type": "object",
                "properties": edit_properties,
                "required": ["prompt", "reference_asset_ids"],
                "additionalProperties": False,
            },
            handler=self._tool_edit,
        )
        return [generate, edit]

    async def _tool_generate(self, event, **arguments):
        return await self._run("generate", event, arguments, from_tool=True)

    async def _tool_edit(self, event, **arguments):
        return await self._run("edit", event, arguments, from_tool=True)

    async def generate(self, event, **arguments):
        return await self._run("generate", event, arguments)

    async def edit(self, event, **arguments):
        return await self._run("edit", event, arguments)

    async def _run(self, action, event, arguments, *, from_tool=False):
        try:
            allowed = set(_BASE_PROPERTIES) | (
                {"reference_asset_ids"} if action == "edit" else set()
            )
            if set(arguments) - allowed:
                raise InvalidRequest("Unknown image tool argument")
            calling_provider = (
                consume_tool_call(self.runtime, "grok_image_" + action, arguments)
                if from_tool
                else None
            )
            provider = await self.resolve_provider(
                event, from_tool=from_tool, calling_provider=calling_provider
            )
            scope = await self.scope(event)
            refs = arguments.get("reference_asset_ids", [])
            if not isinstance(refs, list) or not all(isinstance(ref, str) and ref for ref in refs):
                raise InvalidRequest("Reference asset IDs must be a list of strings")
            prompt = arguments.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise InvalidRequest("An image prompt is required")
            message_id = str(getattr(event.message_obj, "message_id", "") or "")
            if not message_id:
                message_id = event.get_extra("grok_operation_nonce")
                if not message_id:
                    message_id = uuid.uuid4().hex
                    event.set_extra("grok_operation_nonce", message_id)
            request = ImageRequest(
                prompt=prompt,
                model=self.runtime.config.get("image_model") or None,
                n=arguments.get("n", 1),
                reference_images=tuple(refs),
                action=action,
                timeout=self.runtime.config.get("image_timeout", 180),
                aspect_ratio=arguments.get("aspect_ratio"),
                resolution=arguments.get("resolution"),
            )

            identity = [asdict(scope), message_id, action, asdict(request)]
            key = hashlib.sha256(
                json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()

            async def generate():
                return await provider.generate_for_scope(request, scope=scope)

            result = await self.delivery.run(key, scope=scope, event=event, generate=generate)
            return json.dumps(result, ensure_ascii=False)
        except GrokOAuthError as error:
            return json.dumps({"status": "error", **error.summary()}, ensure_ascii=False)

    async def import_message_images(self, event):
        await self.resolve_provider(event)
        scope = await self.scope(event)
        components = list(event.message_obj.message or [])
        images = []
        for component in components:
            if isinstance(component, Image):
                images.append(component)
            elif isinstance(component, Reply):
                images.extend(part for part in (component.chain or []) if isinstance(part, Image))
        if len(images) > 5:
            raise InvalidRequest("At most five message images can be imported")
        result = []
        for image in images:
            ref = image.url or image.file or image.path
            asset_id = await self.runtime.assets.import_reference(
                ref, scope=scope, allowed_roots=self.runtime.allowed_roots
            )
            result.append(asset_id)
        return result

    async def close(self):
        self.closed = True
        await self.delivery.close()
