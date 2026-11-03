"""Plan source links for legacy models without depending on model IDs."""

import copy
import json

PROVIDER_TYPE = "grok_oauth_chat_completion"
DISPLAY_NAME = "Xiaobai Grok"


def source_template():
    return {
        "id": "grok_oauth",
        "provider": "xai",
        "type": PROVIDER_TYPE,
        "provider_type": "chat_completion",
        "enable": True,
        "api_base": "https://api.x.ai/v1",
        "timeout": 180,
    }


def plan_legacy_source_links(models, sources):
    """Return changed lists only when this adapter has source-less models.

    An unambiguous existing source is reused. With zero or multiple matching
    sources, create a dedicated source instead of choosing an arbitrary account.
    Existing model IDs, valid links and other provider types remain untouched.
    """
    source_ids = {
        source["id"] for source in sources if isinstance(source.get("id"), str) and source["id"]
    }
    orphan_indexes = [
        i
        for i, model in enumerate(models)
        if model.get("type") == PROVIDER_TYPE
        and model.get("grok_account_slot", "default") == "default"
        and model.get("provider_source_id") not in source_ids
    ]
    if not orphan_indexes:
        return None
    compatible = [
        source
        for source in sources
        if source.get("type") == PROVIDER_TYPE
        and source.get("enable", True) is not False
        and source.get("grok_account_slot", "default") == "default"
        and isinstance(source.get("id"), str)
        and source["id"]
    ]
    next_sources = copy.deepcopy(sources)
    if len(compatible) == 1:
        source_id = compatible[0]["id"]
    else:
        base = "grok_oauth" if not compatible else "grok_oauth_legacy"
        source_id = base
        suffix = 2
        while source_id in source_ids:
            source_id = f"{base}_{suffix}"
            suffix += 1
        next_sources.append({**source_template(), "id": source_id})
    next_models = copy.deepcopy(models)
    for i in orphan_indexes:
        next_models[i]["provider_source_id"] = source_id
    return next_models, next_sources, len(orphan_indexes)


def is_legacy_source_template(value):
    """Recognize the v0.1.0 template cached by the host's schema endpoint."""
    expected = {
        "id": PROVIDER_TYPE,
        "type": PROVIDER_TYPE,
        "provider_type": "chat_completion",
        "enable": False,
        "key": ["oauth-managed"],
        "api_base": "https://api.x.ai/v1",
        "model": "grok-4.6",
        "grok_account_slot": "default",
        "timeout": 180,
    }
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True
        )
    except (TypeError, ValueError):
        return False


def plan_native_model_configs(models, sources, *, model_defaults=None):
    """Normalize this provider's model rows without changing effective overrides.

    Native Dashboard model editors render the keys present in the model row.
    Connection settings inherited unchanged from its source are not model fields.
    """
    model_defaults = model_defaults or {}
    next_models = copy.deepcopy(models)
    for model in next_models:
        matching = [
            source
            for source in sources
            if source.get("id") == model.get("provider_source_id") and source.get("id")
        ]
        if len(matching) != 1 or matching[0].get("type") != PROVIDER_TYPE:
            continue
        source = matching[0]
        if model.get("grok_account_slot", source.get("grok_account_slot", "default")) != "default":
            continue
        for key in (
            "type",
            "provider",
            "provider_type",
            "api_base",
            "timeout",
            "grok_account_slot",
            "key",
        ):
            if key in model and key in source and model[key] == source[key]:
                del model[key]
        if model.get("key") == ["oauth-managed"] and "key" not in source:
            del model["key"]
        if model.get("grok_account_slot") == "default" and "grok_account_slot" not in source:
            del model["grok_account_slot"]
        defaults = {"modalities": ["text"], "max_context_tokens": 0, "custom_extra_body": {}}
        defaults.update(model_defaults.get(model.get("model") or "grok-4.6", {}))
        for key, fallback in defaults.items():
            if key not in model:
                model[key] = copy.deepcopy(source[key] if key in source else fallback)
    return next_models if next_models != models else None
