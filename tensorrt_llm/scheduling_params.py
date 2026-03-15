from dataclasses import dataclass
from typing import Optional

DEFAULT_ORGANIZATION_ID = "default"
MAX_PRIORITY_TIER = 15


def normalize_priority_tier(priority_tier: object) -> int:
    try:
        # Keep the scheduler tier space bounded so request.priority() stays in a
        # small, predictable range for the downstream C++ scheduler.
        return min(MAX_PRIORITY_TIER, max(0, int(priority_tier)))
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True, kw_only=True)
class SchedulingParams:
    """Schedule parameters.

    Args:
        attention_dp_rank (int): The rank of target attention dp
        attention_dp_relax (bool): Whether to allow the request to be scheduled to other attention dp for better throughput
        priority_tier (int): Relative request priority for the PyTorch worker scheduler. Higher tiers dominate lower
            tiers; fairness and aging are applied within a tier-aware policy rather than allowing lower tiers to
            overtake higher tiers.
        organization_id (str): Organization identifier used by the PyTorch worker scheduler for spike protection.
    """

    attention_dp_rank: Optional[int] = None
    attention_dp_relax: Optional[bool] = None
    priority_tier: int = 0
    organization_id: Optional[str] = None


def get_py_scheduling_params(request: object) -> Optional[SchedulingParams]:
    """Return Python-only scheduling params attached to a request-like object.

    Missing `py_scheduling_params` is treated as absent configuration. Only
    `AttributeError` is swallowed so unexpected failures still surface.
    """

    try:
        return request.py_scheduling_params
    except AttributeError:
        return None


def normalize_organization_id(organization_id: Optional[str]) -> str:
    if organization_id is None:
        return DEFAULT_ORGANIZATION_ID

    normalized = str(organization_id).strip()
    return normalized or DEFAULT_ORGANIZATION_ID


def hash_organization_id(organization_id: Optional[str]) -> int:
    """Return a stable 64-bit organization hash.

    We intentionally avoid Python's built-in hash because it is salted per
    process. The scheduler state needs a stable key across worker restarts.
    """

    normalized = normalize_organization_id(organization_id)
    value = 0xCBF29CE484222325
    for byte in normalized.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value
