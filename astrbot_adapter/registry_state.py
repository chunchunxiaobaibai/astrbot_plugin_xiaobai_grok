"""One plugin-owned service shared by all configured provider instances."""

from ..grok_oauth.errors import RegistrationConflict, ServiceClosed

_active = None


def bind_runtime(runtime):
    global _active
    if _active is not None and _active is not runtime and not _active.closed:
        raise RegistrationConflict("Another Grok runtime is already active")
    _active = runtime


def get_runtime():
    if _active is None or _active.closed:
        raise ServiceClosed("Grok OAuth plugin is not ready")
    return _active


def clear_runtime(runtime):
    global _active
    if _active is runtime:
        _active = None
