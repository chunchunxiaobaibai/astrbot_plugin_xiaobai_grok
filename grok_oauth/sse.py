"""Bounded SSE framing with byte-safe UTF-8 and explicit incomplete input."""

import json
import re
from collections.abc import AsyncIterator

from .errors import ProtocolError, StreamIncomplete


async def iter_json_sse_events(
    chunks: AsyncIterator[bytes], *, max_event_bytes: int = 2 * 1024 * 1024
) -> AsyncIterator[dict]:
    """Decode complete JSON events while limiting buffered event bytes.

    Args:
        chunks: Raw HTTP body chunks.
        max_event_bytes: Maximum encoded bytes per event, including comments.

    Yields:
        JSON object events. The legacy DONE marker is not a Responses terminal.

    Raises:
        ProtocolError: Invalid UTF-8, JSON, shape or size.
        StreamIncomplete: The connection ends inside an event.
    """
    if type(max_event_bytes) is not int or max_event_bytes < 1:
        raise ValueError("max_event_bytes must be positive")
    pending = bytearray()
    data: list[bytes] = []
    event_size = 0

    def consume(line: bytes):
        nonlocal event_size
        event_size += len(line) + 1
        if event_size > max_event_bytes:
            raise ProtocolError("SSE event exceeds byte limit")
        if not line:
            event_size = 0
            if not data:
                return None
            payload = b"\n".join(data)
            data.clear()
            if payload == b"[DONE]":
                return None
            try:
                obj = json.loads(payload.decode("utf-8"))
            except (ValueError, UnicodeError):
                raise ProtocolError("Invalid SSE JSON event") from None
            if not isinstance(obj, dict):
                raise ProtocolError("SSE event must be an object")
            return obj
        if line.startswith(b"data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(b" ") else value)
        elif line == b"data":
            data.append(b"")
        return None

    async for chunk in chunks:
        pending.extend(chunk)
        while match := re.search(rb"\r\n|\r|\n", pending):
            if pending[match.start() : match.end()] == b"\r" and match.end() == len(pending):
                break  # A CRLF may be split over two network chunks.
            line = bytes(pending[: match.start()])
            del pending[: match.end()]
            obj = consume(line)
            if obj is not None:
                yield obj
        if event_size + len(pending) > max_event_bytes:
            raise ProtocolError("SSE event exceeds byte limit")
    if pending.endswith(b"\r"):
        obj = consume(bytes(pending[:-1]))
        pending.clear()
        if obj is not None:
            yield obj
    if pending or data:
        raise StreamIncomplete("Connection ended within an SSE event")
