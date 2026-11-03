"""Non-serializable, one-use proof that a local tool call came from Grok.

The host passes argument values unchanged to function handlers but serializes
history as JSON. A string subtype preserves the public argument value while its
in-memory proof cannot be recreated by another model's JSON response.
"""

import hashlib
import json
import time
import weakref
from dataclasses import dataclass

IMAGE_TOOL_NAMES = ("grok_image_generate", "grok_image_edit")
VIDEO_TOOL_NAMES = ("grok_video_generate", "grok_video_edit", "grok_video_status")
TOOL_TEXT_FIELDS = {
    "grok_video_generate": "prompt",
    "grok_video_edit": "prompt",
    "grok_video_status": "job_id",
    "grok_web_search": "query",
    "grok_image_generate": "prompt",
    "grok_image_edit": "prompt",
}
USAGE_TOOL_NAME = "grok_usage_status"
USAGE_BREAKDOWN_TOOL_NAME = "grok_usage_breakdown"
USAGE_TOOL_NAMES = frozenset((USAGE_TOOL_NAME, USAGE_BREAKDOWN_TOOL_NAME))
OWNED_TOOL_NAMES = frozenset((*TOOL_TEXT_FIELDS, *USAGE_TOOL_NAMES))


def _digest(arguments):
    encoded = json.dumps(
        arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).digest()


@dataclass
class _CallProof:
    provider: weakref.ReferenceType
    owner_id: str
    tool_name: str
    digest: bytes
    expires: float
    consumed: bool = False
    authorization: tuple[int, int] | None = None

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class _IssuedToolText(str):
    def __new__(cls, text, proof):
        instance = super().__new__(cls, text)
        instance._proof = proof
        return instance

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # A copied response must not duplicate authority to execute the call.
        return self


def issue_tool_call(provider, name, arguments):
    field = TOOL_TEXT_FIELDS.get(name)
    if not field or not isinstance(arguments.get(field), str):
        return
    try:
        digest = _digest(arguments)
    except (TypeError, ValueError, UnicodeError):
        return
    proof = _CallProof(
        weakref.ref(provider), provider._runtime.owner_id, name, digest, time.monotonic() + 600
    )
    arguments[field] = _IssuedToolText(arguments[field], proof)


def consume_tool_call(runtime, name, arguments):
    """Validate and consume before any await or string normalization occurs."""
    field = TOOL_TEXT_FIELDS.get(name)
    value = arguments.get(field) if field else None
    if not isinstance(value, _IssuedToolText):
        return None
    proof = value._proof
    provider = proof.provider()
    if (
        proof.consumed
        or proof.tool_name != name
        or proof.owner_id != runtime.owner_id
        or proof.expires <= time.monotonic()
        or runtime.closed
        or provider is None
        or provider._closed
        or provider._runtime is not runtime
    ):
        return None
    try:
        if proof.digest != _digest(arguments):
            return None
    except (TypeError, ValueError, UnicodeError):
        return None
    proof.consumed = True
    return provider


class _IssuedToolName(_IssuedToolText):
    """Tool-name proof survives the host's empty-argument reconstruction."""


def issue_usage_call(provider, name=USAGE_TOOL_NAME):
    if name not in USAGE_TOOL_NAMES:
        raise ValueError("Unknown usage tool")
    proof = _CallProof(
        weakref.ref(provider),
        provider._runtime.owner_id,
        name,
        _digest({}),
        time.monotonic() + 600,
        authorization=(provider._runtime.oauth.epoch, provider._runtime.oauth.binding_generation),
    )
    return _IssuedToolName(name, proof)


def consume_usage_call(runtime, name):
    if not isinstance(name, _IssuedToolName):
        return None
    proof = name._proof
    provider = proof.provider()
    if (
        proof.authorization != (runtime.oauth.epoch, runtime.oauth.binding_generation)
        or proof.consumed
        or name not in USAGE_TOOL_NAMES
        or proof.tool_name != name
        or proof.owner_id != runtime.owner_id
        or proof.expires <= time.monotonic()
        or runtime.closed
        or provider is None
        or provider._closed
        or provider._runtime is not runtime
    ):
        return None
    proof.consumed = True
    return provider
