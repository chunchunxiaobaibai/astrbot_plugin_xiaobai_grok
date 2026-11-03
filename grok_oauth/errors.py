"""Typed public failures; upstream bodies and credentials never form messages."""

import re


class GrokOAuthError(Exception):
    def __init__(
        self,
        message=None,
        *,
        request_id="",
        operation_id="",
        partial=False,
        assets=(),
        submission_state=None,
    ):
        self.code = type(self).__name__
        self.request_id = request_id if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id) else ""
        self.operation_id = operation_id
        self.partial = partial
        self.assets = tuple(assets)
        if submission_state in {"not_submitted", "submitted", "unknown"}:
            self.submission_state = submission_state
        super().__init__(message or self.code)

    def summary(self):
        result = {
            "error": self.code,
            "request_id": self.request_id,
            "operation_id": self.operation_id,
            "partial": self.partial,
        }
        if hasattr(self, "submission_state"):
            result["submission_state"] = self.submission_state
        return result


class ReauthorizationRequired(GrokOAuthError):
    pass


class CredentialPersistenceError(GrokOAuthError):
    pass


class CredentialStoreInUse(GrokOAuthError):
    pass


class ServiceClosed(GrokOAuthError):
    pass


class AuthorizationChanged(GrokOAuthError):
    pass


class AuthorizationDenied(GrokOAuthError):
    pass


class DeviceCodeExpired(GrokOAuthError):
    pass


class ClientNotEligible(GrokOAuthError):
    pass


class PermissionDenied(GrokOAuthError):
    pass


class PaymentRequired(PermissionDenied):
    """The service requires available credits or an eligible subscription."""


class RateLimited(GrokOAuthError):
    status_code = 429


class OutcomeUnknown(GrokOAuthError):
    pass


class UnsafeTarget(GrokOAuthError):
    pass


class ProtocolError(GrokOAuthError):
    pass


class StreamIncomplete(GrokOAuthError):
    pass


class EmptyOutput(GrokOAuthError):
    pass


class UnsupportedModelParameter(GrokOAuthError):
    pass


class InvalidImageRequest(GrokOAuthError):
    pass


class ImageTooLarge(GrokOAuthError):
    pass


class UnsafeMediaSource(GrokOAuthError):
    pass


class AssetExpired(GrokOAuthError):
    pass


class AssetNotFound(GrokOAuthError):
    pass


class Busy(GrokOAuthError):
    pass


class UnsupportedAstrBotVersion(GrokOAuthError):
    pass


class RegistrationConflict(GrokOAuthError):
    pass


class AuthenticationRequired(GrokOAuthError):
    pass


class InvalidRequest(GrokOAuthError):
    pass


class MediaDownloadError(GrokOAuthError):
    pass


class VideoUnsupported(GrokOAuthError):
    """The requested video mode has no supported upstream contract."""
