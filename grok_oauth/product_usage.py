"""Strict normalization for the allowlisted upstream product-usage breakdown."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

_MAX_PRODUCT_ROWS = 32
_PRODUCTS = {
    "GrokBuild": ("GrokBuild", "Grok Build"),
    "PRODUCT_GROK_BUILD": ("GrokBuild", "Grok Build"),
    "GrokChat": ("GrokChat", "Grok"),
    "GrokImagine": ("GrokImagine", "Imagine"),
}


@dataclass(frozen=True, slots=True)
class ProductUsage:
    """One recognized upstream product percentage."""

    display_name: str
    product: str
    usage_percent: float | None

    def to_dict(self) -> dict[str, str | float | None]:
        return {
            "display_name": self.display_name,
            "product": self.product,
            "usage_percent": self.usage_percent,
        }


@dataclass(frozen=True, slots=True)
class ProductUsageResult:
    status: str = "unknown"
    products: tuple[ProductUsage, ...] = ()
    unrecognized_product_count: int = 0


def _percentage(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        parsed = float(value)
    except OverflowError:
        return None
    if not math.isfinite(parsed) or not 0 <= parsed <= 100:
        return None
    return parsed


def parse_product_usage(data: object) -> ProductUsageResult:
    """Parse product percentages without retaining unrecognized names or other fields."""

    if not isinstance(data, Mapping):
        return ProductUsageResult(status="unparseable")
    config = data.get("config")
    if config is None:
        return ProductUsageResult()
    if not isinstance(config, Mapping):
        return ProductUsageResult(status="unparseable")
    if "productUsage" not in config:
        return ProductUsageResult()

    rows = config["productUsage"]
    if not isinstance(rows, list) or len(rows) > _MAX_PRODUCT_ROWS:
        return ProductUsageResult(status="unparseable")
    if not rows:
        return ProductUsageResult()

    products: list[ProductUsage] = []
    seen: set[str] = set()
    unrecognized = 0
    has_percentage = False
    incomplete = False
    for row in rows:
        if not isinstance(row, Mapping):
            return ProductUsageResult(status="unparseable")
        upstream_product = row.get("product")
        if not isinstance(upstream_product, str) or not upstream_product:
            return ProductUsageResult(status="unparseable")
        has_usage_percent = "usagePercent" in row
        usage_percent = _percentage(row.get("usagePercent")) if has_usage_percent else None
        if has_usage_percent and usage_percent is None:
            return ProductUsageResult(status="unparseable")
        mapped = _PRODUCTS.get(upstream_product)
        if mapped is None:
            unrecognized += 1
            continue

        product, display_name = mapped
        if product in seen:
            return ProductUsageResult(status="unparseable")
        seen.add(product)
        if not has_usage_percent:
            incomplete = True
        else:
            has_percentage = True
        products.append(
            ProductUsage(
                display_name=display_name,
                product=product,
                usage_percent=usage_percent,
            )
        )

    if not has_percentage:
        status = "unknown"
    elif incomplete or unrecognized:
        status = "partial"
    else:
        status = "success"
    return ProductUsageResult(
        status=status,
        products=tuple(products),
        unrecognized_product_count=unrecognized,
    )
