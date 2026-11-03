"""Pure capability snapshots; only explicit refresh performs network I/O."""

import copy
import time

from .errors import ProtocolError, UnsupportedModelParameter
from .models import RequestPolicy

# Primary-source snapshot: docs.x.ai/developers/model-capabilities/text/reasoning,
# checked 2026-09-21. Fast shares the 4.7 model; OAuth probe evidence is recorded
# in docs/model-47-evidence.json. Unknown models keep no forced effort setting.
REASONING_EFFORTS = {
    "grok-4.7": {"low", "medium", "high", "xhigh"},
    "grok-4.7-build-fast": {"low", "medium", "high", "xhigh"},
    "grok-4.6": {"low", "medium", "high", "xhigh"},
    "grok-4.5": {"low", "medium", "high"},
    "grok-4.3": {"none", "low", "medium", "high"},
    "grok-4.20-multi-agent": {"low", "medium", "high", "xhigh"},
}


class ModelCatalog:
    def __init__(self):
        self._models = ["grok-4.7", "grok-4.7-build-fast", "grok-4.6", "grok-4.5", "grok-4.3"]
        self._observed = {}
        self.refreshed_at = None

    def chat_models(self):
        return list(self._models)

    def snapshot(self):
        return {
            "models": self.chat_models(),
            "refreshed_at": self.refreshed_at,
            "observed": copy.deepcopy(self._observed),
        }

    def capabilities(self, model, *, images_enabled=True, search_enabled=True, videos_enabled=True):
        result = {}
        for capability in (
            "chat",
            "streaming",
            "vision",
            "function_tools",
            "image_generation",
            "image_editing",
        ):
            result[capability] = {
                "implementation": True,
                "enabled": images_enabled if capability.startswith("image_") else True,
                "model_support": "available"
                if model in REASONING_EFFORTS and not capability.startswith("image_")
                else "unknown",
                "observed": self._observed.get(model, {}).get(capability, "unknown"),
            }
        for capability in (
            "native_image_generation",
            "web_search",
            "x_search",
            "code_execution",
            "audio",
            "video",
        ):
            result[capability] = {
                "implementation": False,
                "enabled": False,
                "model_support": "unknown",
                "observed": "unknown",
            }
        result["web_search"] = {
            "implementation": True,
            "enabled": search_enabled,
            "model_support": "unknown",
            "observed": self._observed.get(model, {}).get("web_search", "unknown"),
        }
        for capability in ("video_generation", "video_editing"):
            result[capability] = {
                "implementation": True,
                "enabled": videos_enabled,
                "model_support": "unknown",
                "observed": "unknown",
            }
        return result

    def observe(self, model, capability, state):
        if state not in {"unknown", "available", "denied", "temporary_error"}:
            raise ValueError("Invalid capability state")
        self._observed.setdefault(model, {})[capability] = state

    @staticmethod
    def validate_reasoning(model, effort):
        if effort is None or effort == "":
            return None
        if not isinstance(effort, str) or effort not in REASONING_EFFORTS.get(model, set()):
            raise UnsupportedModelParameter("Reasoning effort is not supported for this model")
        return {"effort": effort}

    async def refresh(self, http):
        raw = await http.request_json(
            "GET", "/models", policy=RequestPolicy(deadline=time.monotonic() + 30)
        )
        entries = raw.get("data")
        if not isinstance(entries, list):
            raise ProtocolError("Invalid model catalog")
        models = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                raise ProtocolError("Invalid model catalog entry")
            model = entry["id"]
            if model.startswith("grok-") and not any(
                word in model for word in ("imagine", "video", "audio", "voice")
            ):
                models.append(model)
        # OAuth accepts this Build-only variant even when /models omits it.
        # Listing a known variant does not claim account entitlement.
        if "grok-4.7" in models:
            models.append("grok-4.7-build-fast")
        self._models = sorted(set(models))
        self.refreshed_at = time.time()
        return self.snapshot()
