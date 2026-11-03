"""Strict, side-effect-free normalization for OAuth billing responses."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from numbers import Real
from typing import Any
from zoneinfo import ZoneInfo

from .product_usage import ProductUsage, parse_product_usage

_SOURCE = "grok_oauth_billing"
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
_PERIOD_TYPES = {
    "USAGE_PERIOD_TYPE_WEEKLY": "weekly",
    "USAGE_PERIOD_TYPE_MONTHLY": "monthly",
}
_QUOTA_STATUSES = {"success", "unknown", "unparseable"}


@dataclass(frozen=True, slots=True)
class QuotaSnapshot:
    """Public quota view; fields intentionally match the output allowlist."""

    status: str
    observed_at: str
    scope: str = "unknown"
    period_type: str = "unknown"
    used_percent: float | None = None
    remaining_percent: float | None = None
    period_start: str | None = None
    reset_at: str | None = None
    reset_at_local: str | None = None
    cached: bool = False
    stale: bool = False
    source: str = _SOURCE
    products: tuple[ProductUsage, ...] = ()
    unrecognized_product_count: int = 0
    product_usage_status: str = "unknown"

    def to_dict(self) -> dict[str, str | float | bool | None]:
        """Return only the stable public fields."""

        return {
            "status": self.status,
            "scope": self.scope,
            "period_type": self.period_type,
            "used_percent": self.used_percent,
            "remaining_percent": self.remaining_percent,
            "period_start": self.period_start,
            "reset_at": self.reset_at,
            "reset_at_local": self.reset_at_local,
            "observed_at": self.observed_at,
            "cached": self.cached,
            "stale": self.stale,
            "source": self.source,
        }

    def to_breakdown_dict(self) -> dict[str, object]:
        """Return the independent, allowlisted product-usage view."""

        status = self.product_usage_status if self.status in _QUOTA_STATUSES else self.status
        return {
            "status": status,
            "scope": self.scope,
            "period_type": self.period_type,
            "period_start": self.period_start,
            "reset_at": self.reset_at,
            "reset_at_local": self.reset_at_local,
            "observed_at": self.observed_at,
            "cached": self.cached,
            "stale": self.stale,
            "source": self.source,
            "products": [product.to_dict() for product in self.products],
            "unrecognized_product_count": self.unrecognized_product_count,
            "percent_basis": "upstream_product_usage",
        }


@dataclass(frozen=True, slots=True)
class _Period:
    period_type: str
    start: datetime
    end: datetime


def _utc_text(value: datetime) -> str:
    text = value.astimezone(UTC).isoformat()
    return text.removesuffix("+00:00") + "Z"


def _observed_text(value: datetime | int | float) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at datetime must include a timezone")
        return _utc_text(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("observed_at must be a datetime or Unix timestamp")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError("observed_at timestamp is out of range") from error
    if not math.isfinite(numeric):
        raise ValueError("observed_at timestamp must be finite")
    try:
        return _utc_text(datetime.fromtimestamp(numeric, tz=UTC))
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError("observed_at timestamp is out of range") from error


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or _RFC3339.fullmatch(value) is None:
        return None
    try:
        parsed = datetime.fromisoformat(
            value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else "")
        )
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    try:
        normalized = parsed.astimezone(UTC)
        normalized.astimezone(_SHANGHAI)
        return normalized
    except (OverflowError, ValueError):
        return None


def _parse_period(config: Mapping[str, Any]) -> tuple[_Period | None, bool]:
    if "currentPeriod" in config:
        raw = config["currentPeriod"]
        if not isinstance(raw, Mapping):
            return None, False
        raw_type = raw.get("type")
        if raw_type is not None and not isinstance(raw_type, str):
            return None, False
        period_type = _PERIOD_TYPES.get(raw_type, "unknown")
        start = _parse_time(raw.get("start"))
        end = _parse_time(raw.get("end"))
        if start is None or end is None or end <= start:
            return None, False
        return _Period(period_type, start, end), True

    has_start = "billingPeriodStart" in config
    has_end = "billingPeriodEnd" in config
    if not has_start and not has_end:
        return None, True
    start = _parse_time(config.get("billingPeriodStart"))
    end = _parse_time(config.get("billingPeriodEnd"))
    if start is None or end is None or end <= start:
        return None, False
    return _Period("monthly", start, end), True


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        parsed = float(value)
    except OverflowError:
        return None
    return parsed if math.isfinite(parsed) else None


def _new_percentage(config: Mapping[str, Any]) -> tuple[float | None, bool]:
    value = _number(config.get("creditUsagePercent"))
    if value is None or not 0 <= value <= 100:
        return None, False
    return value, True


def _legacy_percentage(
    config: Mapping[str, Any], period: _Period | None
) -> tuple[float | None, bool]:
    limit = config.get("monthlyLimit")
    used = config.get("used")
    if not isinstance(limit, Mapping) or not isinstance(used, Mapping):
        return None, False
    if "val" not in limit or "val" not in used:
        return None, False
    limit_value = _number(limit["val"])
    used_value = _number(used["val"])
    if (
        period is None
        or limit_value is None
        or limit_value <= 0
        or used_value is None
        or used_value < 0
        or used_value > limit_value
    ):
        return None, False
    percent = (used_value / limit_value) * 100
    return (percent, True) if math.isfinite(percent) and 0 <= percent <= 100 else (None, False)


def _parse_quota(
    data: object,
    observed_at: datetime | int | float,
) -> QuotaSnapshot:
    observed = _observed_text(observed_at)
    empty = QuotaSnapshot(status="unknown", observed_at=observed)
    if not isinstance(data, Mapping):
        return QuotaSnapshot(status="unparseable", observed_at=observed)

    config = data.get("config")
    if config is None:
        return empty
    if not isinstance(config, Mapping):
        return QuotaSnapshot(status="unparseable", observed_at=observed)
    if not config:
        return empty

    scope = "account_shared" if config.get("isUnifiedBillingUser") is True else "unknown"
    period, period_valid = _parse_period(config)
    if not period_valid:
        return QuotaSnapshot(status="unparseable", observed_at=observed, scope=scope)

    has_new_percentage = "creditUsagePercent" in config
    has_legacy_percentage = "monthlyLimit" in config or "used" in config
    if has_new_percentage:
        used_percent, percentage_valid = _new_percentage(config)
    elif has_legacy_percentage:
        used_percent, percentage_valid = _legacy_percentage(config, period)
    elif (
        scope == "account_shared"
        and "currentPeriod" in config
        and period is not None
        and period.period_type == "weekly"
        and period.start <= datetime.fromisoformat(observed) < period.end
    ):
        # The app confirms 0% when this current weekly response omits the percentage.
        used_percent, percentage_valid = 0.0, True
    else:
        used_percent, percentage_valid = None, True

    period_values = {}
    if period is not None:
        period_values = {
            "period_type": period.period_type,
            "period_start": _utc_text(period.start),
            "reset_at": _utc_text(period.end),
            "reset_at_local": period.end.astimezone(_SHANGHAI).isoformat(),
        }
    if not percentage_valid:
        return QuotaSnapshot(
            status="unparseable",
            observed_at=observed,
            scope=scope,
            **period_values,
        )
    if used_percent is None:
        return QuotaSnapshot(
            status="unknown",
            observed_at=observed,
            scope=scope,
            **period_values,
        )
    if period is None:
        return QuotaSnapshot(
            status="unknown",
            observed_at=observed,
            scope=scope,
            used_percent=used_percent,
            remaining_percent=100 - used_percent,
        )
    return QuotaSnapshot(
        status="success",
        observed_at=observed,
        scope=scope,
        used_percent=used_percent,
        remaining_percent=100 - used_percent,
        **period_values,
    )


def parse_billing(
    data: object,
    observed_at: datetime | int | float,
) -> QuotaSnapshot:
    """Normalize quota and product usage independently without I/O."""

    quota = _parse_quota(data, observed_at)
    breakdown = parse_product_usage(data)
    return replace(
        quota,
        products=breakdown.products,
        unrecognized_product_count=breakdown.unrecognized_product_count,
        product_usage_status=breakdown.status,
    )
