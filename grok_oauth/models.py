"""Immutable values shared by the protocol and host adapters."""

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TokenSnapshot:
    slot: str
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: float | None
    scope: str
    client_id: str
    version: int = 0
    epoch: int = 0
    user_id: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class DeviceFlow:
    flow_id: str
    owner_id: str
    user_code: str = field(repr=False)
    verification_uri: str
    device_code: str = field(repr=False)
    expires_in: float
    interval: float
    epoch: int
    deadline: float
    expires_at: float
    verification_uri_complete: str = field(default="", repr=False)


@dataclass(frozen=True)
class RequestPolicy:
    capability: str = "chat"
    deadline: float = field(default_factory=lambda: time.monotonic() + 60)
    safe_pre_send_retries: int = 1
    allow_one_auth_retry: bool = True
    side_effecting: bool = False


@dataclass(frozen=True)
class AssetScope:
    platform_id: str
    umo: str
    conversation_id: str


@dataclass(frozen=True)
class GeneratedImage:
    path: str = field(repr=False)
    mime_type: str
    revised_prompt: str = ""
    raw: dict | None = field(default=None, repr=False)
    asset_id: str = ""
    request_id: str = ""
    model: str = ""
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    model: str | None = None
    size: str | None = None
    n: int = 1
    reference_images: tuple[str, ...] = ()
    action: str | None = None
    timeout: float = 180.0
    aspect_ratio: str | None = None
    resolution: str | None = None
