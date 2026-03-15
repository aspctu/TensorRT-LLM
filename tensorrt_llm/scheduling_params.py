from dataclasses import dataclass
from typing import Optional, Protocol, cast

DEFAULT_ORGANIZATION_ID = "default"


class SchedulingParamsLike(Protocol):
    attention_dp_rank: Optional[int]
    attention_dp_relax: Optional[bool]
    priority_tier: int
    organization_id: Optional[str]


class HasSchedulingParams(Protocol):
    py_scheduling_params: Optional[SchedulingParamsLike]


def normalize_priority_tier(priority_tier: object) -> int:
    try:
        return max(0, int(priority_tier))
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True, kw_only=True)
class SchedulingParams:
    """Schedule parameters.

    Args:
        attention_dp_rank (int): The rank of target attention dp
        attention_dp_relax (bool): Whether to allow the request to be scheduled to other attention dp for better throughput
        priority_tier (int): Relative request priority for the PyTorch worker scheduler. Higher tiers are favored,
            but lower tiers age into service over time.
        organization_id (str): Organization identifier used by the PyTorch worker scheduler for spike protection.
    """

    attention_dp_rank: Optional[int] = None
    attention_dp_relax: Optional[bool] = None
    priority_tier: int = 0
    organization_id: Optional[str] = None


def get_py_scheduling_params(request: object) -> Optional[SchedulingParamsLike]:
    """Return Python-only scheduling params attached to a request-like object.

    Missing `py_scheduling_params` is treated as absent configuration. Only
    `AttributeError` is swallowed so unexpected failures still surface.
    """

    try:
        return cast(HasSchedulingParams, request).py_scheduling_params
    except AttributeError:
        return None


def get_priority_tier(scheduling_params: Optional[SchedulingParamsLike]) -> int:
    if scheduling_params is None:
        return 0
    return scheduling_params.priority_tier


def get_organization_id(scheduling_params: Optional[SchedulingParamsLike]) -> Optional[str]:
    if scheduling_params is None:
        return None
    return scheduling_params.organization_id


def normalize_organization_id(organization_id: Optional[str]) -> str:
    if organization_id is None:
        return DEFAULT_ORGANIZATION_ID

    normalized = str(organization_id).strip()
    return normalized or DEFAULT_ORGANIZATION_ID


def hash_organization_id(organization_id: Optional[str]) -> int:
    normalized = normalize_organization_id(organization_id)
    value = 0xCBF29CE484222325
    for byte in normalized.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value
