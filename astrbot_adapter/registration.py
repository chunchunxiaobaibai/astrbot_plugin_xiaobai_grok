"""Owned provider and named Dashboard source-template registration."""

from dataclasses import dataclass

from ..grok_oauth.errors import RegistrationConflict
from .compat import (
    ProviderType,
    provider_cls_map,
    provider_registry,
    provider_source_templates,
    register_provider_adapter,
)
from .source_compat import DISPLAY_NAME, PROVIDER_TYPE, is_legacy_source_template, source_template


@dataclass(frozen=True)
class RegistrationHandle:
    owner_id: str
    metadata: object
    template: dict


def register_provider(owner_id, provider_class):
    templates = provider_source_templates()
    existing = provider_cls_map.get(PROVIDER_TYPE)
    if existing is not None:
        if (
            getattr(existing, "_grok_oauth_owner", None) != owner_id
            or existing.cls_type is not provider_class
            or templates.get(DISPLAY_NAME) is not getattr(existing, "_grok_source_template", None)
        ):
            raise RegistrationConflict("Grok provider type belongs to another registration")
        return RegistrationHandle(owner_id, existing, existing._grok_source_template)
    if DISPLAY_NAME in templates:
        raise RegistrationConflict("Grok source template belongs to another registration")
    missing = object()
    legacy = templates.get(PROVIDER_TYPE, missing)
    if legacy is not missing and not is_legacy_source_template(legacy):
        raise RegistrationConflict(
            "Unrecognized Grok legacy template belongs to another registration"
        )
    template = source_template()
    # AstrBot's source picker displays template dictionary keys. A registry
    # default_config_tmpl would add a second entry named after the internal type.
    register_provider_adapter(
        PROVIDER_TYPE,
        "Grok subscription OAuth",
        ProviderType.CHAT_COMPLETION,
        provider_display_name=DISPLAY_NAME,
    )(provider_class)
    metadata = provider_cls_map[PROVIDER_TYPE]
    metadata._grok_oauth_owner = owner_id
    metadata._grok_source_template = template
    templates[DISPLAY_NAME] = template
    if legacy is not missing and templates.get(PROVIDER_TYPE) is legacy:
        del templates[PROVIDER_TYPE]
    return RegistrationHandle(owner_id, metadata, template)


def unregister_provider(handle):
    existing = provider_cls_map.get(PROVIDER_TYPE)
    if (
        existing is handle.metadata
        and getattr(existing, "_grok_oauth_owner", None) == handle.owner_id
    ):
        del provider_cls_map[PROVIDER_TYPE]
        provider_registry[:] = [item for item in provider_registry if item is not existing]
    templates = provider_source_templates()
    if templates.get(DISPLAY_NAME) is handle.template:
        del templates[DISPLAY_NAME]
