"""Finite, content-free fields shared by transport diagnostics and host sinks."""

from __future__ import annotations

import json
import re

DIAGNOSTIC_ERROR_CODES = frozenset(
    {
        "",
        "unknown",
        "invalid_request_error",
        "invalid_argument",
        "context_length_exceeded",
        "unsupported_parameter",
        "invalid_parameter",
        "invalid_image",
        "content_policy_violation",
        "rate_limit_exceeded",
        "insufficient_quota",
        "unauthorized_client",
        "invalid_client",
        "client_not_eligible",
        "model_not_found",
        "server_error",
    }
)
DIAGNOSTIC_ERROR_PARAMS = frozenset(
    {
        "",
        "unknown",
        "model",
        "input",
        "tools",
        "tool_choice",
        "reasoning",
        "reasoning_effort",
        "max_output_tokens",
        "temperature",
        "top_p",
        "image",
        "images",
        "prompt",
        "duration",
        "aspect_ratio",
        "resolution",
    }
)
DIAGNOSTIC_REQUEST_KINDS = frozenset(
    {
        "chat",
        "image_generate",
        "image_edit",
        "video_generate",
        "video_edit",
        "video_status",
        "models",
        "other",
    }
)
_MAX_ERROR_BYTES = 64 * 1024
_POST_KINDS = {
    "/responses": "chat",
    "/images/generations": "image_generate",
    "/images/edits": "image_edit",
    "/videos/generations": "video_generate",
    "/videos/edits": "video_edit",
}


def _first_present(*values):
    return next((value for value in values if value is not None and value != ""), None)


def _known_value(value, allowed: frozenset[str]) -> str:
    if value is None or value == "":
        return ""
    return value if isinstance(value, str) and value in allowed else "unknown"


def safe_error_metadata(body: bytes) -> dict[str, str]:
    """Extract only finite code/parameter categories, never upstream free text."""
    empty = {"upstream_error_code": "", "upstream_error_param": ""}
    if not isinstance(body, bytes) or not body or len(body) > _MAX_ERROR_BYTES:
        return empty
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        return empty
    if not isinstance(payload, dict):
        return empty
    error = payload.get("error")
    nested = error if isinstance(error, dict) else {}
    code = _first_present(
        nested.get("code"),
        nested.get("type"),
        payload.get("code"),
        payload.get("type"),
        error if isinstance(error, str) else None,
    )
    parameter = _first_present(nested.get("param"), payload.get("param"))
    return {
        "upstream_error_code": _known_value(code, DIAGNOSTIC_ERROR_CODES),
        "upstream_error_param": _known_value(parameter, DIAGNOSTIC_ERROR_PARAMS),
    }


def request_kind(method: str, path: str) -> str:
    """Return a route category without recording URLs, query strings or job IDs."""
    if not isinstance(method, str) or not isinstance(path, str):
        return "other"
    if method.upper() == "POST":
        return _POST_KINDS.get(path, "other")
    if method.upper() == "GET":
        if path == "/models":
            return "models"
        if re.fullmatch(r"/videos/[A-Za-z0-9_-]{1,128}", path):
            return "video_status"
    return "other"
