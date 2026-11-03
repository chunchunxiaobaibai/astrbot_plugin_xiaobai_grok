"""Versioned, validated continuation state carried by the host's opaque ThinkPart."""

import copy
import hashlib
import json
import re

from .errors import ProtocolError

STATE_TYPE = "grok_oauth_reasoning"


def decode_envelope(value):
    """Ignore foreign state, but fail closed for damaged plugin-owned JSON."""
    try:
        state = json.loads(value)
    except (TypeError, ValueError):
        if isinstance(value, str) and re.search(r'"type"\s*:\s*"grok_oauth_reasoning"', value):
            raise ProtocolError("Invalid saved reasoning state") from None
        return None
    return state if isinstance(state, dict) and state.get("type") == STATE_TYPE else None


def _arguments(value):
    try:
        parsed = json.loads(value) if isinstance(value, str) else None
    except (TypeError, ValueError):
        raise ProtocolError("Invalid continuation arguments") from None
    if not isinstance(parsed, dict):
        raise ProtocolError("Invalid continuation arguments")
    return parsed


def _digest(text, calls):
    try:
        data = json.dumps(
            {"text": text, "calls": calls},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        raise ProtocolError("Invalid continuation display") from None
    return hashlib.sha256(data.encode()).hexdigest()


def _calls(items):
    return [
        {"id": item["call_id"], "name": item["name"], "arguments": _arguments(item["arguments"])}
        for item in items
        if item.get("type") == "function_call"
    ]


def _validate_output(items):
    if not isinstance(items, list) or not items:
        raise ProtocolError("Invalid continuation output")
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ProtocolError("Invalid continuation item")
        kind = item.get("type")
        if kind == "reasoning":
            for key in ("summary", "content"):
                value = item.get(key)
                if value is not None and (
                    not isinstance(value, list) or any(not isinstance(x, dict) for x in value)
                ):
                    raise ProtocolError("Invalid continuation reasoning")
        elif kind == "message":
            if item.get("role", "assistant") != "assistant" or not isinstance(
                item.get("content"), list
            ):
                raise ProtocolError("Invalid continuation message")
            for part in item["content"]:
                if not isinstance(part, dict) or part.get("type") not in {"output_text", "refusal"}:
                    raise ProtocolError("Unsupported continuation message part")
                key = "text" if part["type"] == "output_text" else "refusal"
                if not isinstance(part.get(key), str):
                    raise ProtocolError("Invalid continuation text")
        elif kind == "function_call":
            call_id = item.get("call_id")
            if (
                not isinstance(call_id, str)
                or not call_id
                or call_id in seen
                or not isinstance(item.get("name"), str)
                or not item["name"]
            ):
                raise ProtocolError("Invalid continuation function")
            seen.add(call_id)
            _arguments(item.get("arguments"))
        elif kind == "web_search_call":
            if item.get("status") != "completed":
                raise ProtocolError("Incomplete continuation search")
        else:
            raise ProtocolError("Unsupported continuation item")
        if item.get("status") not in (None, "completed"):
            raise ProtocolError("Incomplete continuation item")


def make_continuation(result, display_text):
    if not result.reasoning_items and not result.native_items:
        return None
    state = {"type": STATE_TYPE, "items": result.reasoning_items}
    if result.output_items:
        _validate_output(result.output_items)
        state.update(
            version=2,
            output=result.output_items,
            display_digest=_digest(display_text, _calls(result.output_items)),
        )
    return json.dumps(state, ensure_ascii=False)


def restore_continuation(message):
    """Return original ordered output only while host-visible history is unchanged.

    Older reasoning-only envelopes are handled by the legacy encoder. Edited
    histories use the host's text/calls and discard their stale opaque state.
    """
    content = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, list):
        return None
    states = []
    own_states = 0
    text = []
    for part in content:
        if not isinstance(part, dict):
            raise ProtocolError("Invalid history content")
        if part.get("type") in {"text", "input_text", "output_text"}:
            if not isinstance(part.get("text"), str):
                raise ProtocolError("Invalid history text")
            text.append(part["text"])
        elif part.get("type") == "think" and part.get("encrypted"):
            state = decode_envelope(part["encrypted"])
            if isinstance(state, dict) and state.get("type") == STATE_TYPE:
                own_states += 1
            if isinstance(state, dict) and state.get("type") == STATE_TYPE and "version" in state:
                if type(state["version"]) is not int or state["version"] != 2:
                    raise ProtocolError("Unsupported continuation version")
                states.append(state)
        elif part.get("type") != "think":
            raise ProtocolError("Unsupported history content")
    if not states:
        return None
    if len(states) != 1 or own_states != 1:
        raise ProtocolError("Ambiguous continuation state")
    state = states[0]
    output = state.get("output")
    _validate_output(output)
    if state.get("items") != [item for item in output if item["type"] == "reasoning"]:
        raise ProtocolError("Inconsistent continuation reasoning")
    digest = state.get("display_digest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ProtocolError("Invalid continuation display digest")
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ProtocolError("Invalid history tool calls")
    logical_calls = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ProtocolError("Invalid history function")
        function = call["function"]
        logical_calls.append(
            {
                "id": call.get("id"),
                "name": function.get("name"),
                "arguments": _arguments(function.get("arguments")),
            }
        )
    if _digest("".join(text), logical_calls) != digest:
        return None
    return copy.deepcopy(output)
