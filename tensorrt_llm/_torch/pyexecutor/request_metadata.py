from dataclasses import dataclass
from typing import Optional

from tensorrt_llm.scheduling_params import (
    DEFAULT_ORGANIZATION_ID,
    get_organization_id,
    get_priority_tier,
    get_py_scheduling_params,
    normalize_organization_id,
    normalize_priority_tier,
)

_MISSING = object()


@dataclass(slots=True)
class RequestStatsExtra:
    priorityTier: int
    priorityCredit: float
    priorityPauseCount: int
    organizationId: str
    schedulerScore: Optional[float] = None
    schedulerAgeCredit: Optional[float] = None
    schedulerOrgTokenBalance: Optional[float] = None

    def to_dict(self) -> dict[str, int | float | str]:
        result: dict[str, int | float | str] = {
            "priorityTier": self.priorityTier,
            "priorityCredit": self.priorityCredit,
            "priorityPauseCount": self.priorityPauseCount,
            "organizationId": self.organizationId,
        }
        if self.schedulerScore is not None:
            result["schedulerScore"] = self.schedulerScore
        if self.schedulerAgeCredit is not None:
            result["schedulerAgeCredit"] = self.schedulerAgeCredit
        if self.schedulerOrgTokenBalance is not None:
            result["schedulerOrgTokenBalance"] = self.schedulerOrgTokenBalance
        return result


def _unwrap_request(request_or_item: object) -> Optional[object]:
    if request_or_item is None:
        return None

    request = getattr(request_or_item, "request", _MISSING)
    if request is _MISSING:
        return request_or_item
    return request


def request_priority_tier(request_or_item: object) -> int:
    request = _unwrap_request(request_or_item)
    if request is None:
        return 0

    priority_tier = getattr(request, "py_priority_tier", _MISSING)
    if priority_tier is not _MISSING:
        return normalize_priority_tier(priority_tier)

    scheduling_params = get_py_scheduling_params(request)
    return normalize_priority_tier(get_priority_tier(scheduling_params))


def request_organization_id(request_or_item: object) -> str:
    request = _unwrap_request(request_or_item)
    if request is None:
        return DEFAULT_ORGANIZATION_ID

    organization_id = getattr(request, "py_organization_id", _MISSING)
    if organization_id is not _MISSING:
        return normalize_organization_id(organization_id)

    scheduling_params = get_py_scheduling_params(request)
    return normalize_organization_id(get_organization_id(scheduling_params))


def request_organization_key(request_or_item: object) -> str:
    return f"{request_priority_tier(request_or_item)}:{request_organization_id(request_or_item)}"


def scheduled_token_cost(request: object) -> int:
    if getattr(request, "is_encoder_init_state", False):
        return max(1, int(getattr(request, "encoder_output_len", 1)))

    if getattr(request, "is_context_init_state", False) or getattr(
        request, "is_disagg_generation_init_state", False
    ):
        chunk_size = int(
            getattr(
                request,
                "context_chunk_size",
                getattr(request, "context_remaining_length", 0),
            )
            or 0
        )
        if chunk_size <= 0 and hasattr(request, "get_num_tokens"):
            chunk_size = int(request.get_num_tokens(0))

        draft_tokens = 0
        if getattr(request, "has_draft_tokens", False) and getattr(
            request, "is_last_context_chunk", False
        ):
            draft_tokens = int(getattr(request, "num_draft_tokens", 0))
        return max(1, chunk_size + draft_tokens)

    if hasattr(request, "get_beam_width_by_iter"):
        beam_width = int(request.get_beam_width_by_iter(for_next_iteration=False))
    else:
        sampling_config = getattr(request, "sampling_config", None)
        beam_width = int(getattr(sampling_config, "beam_width", 1))
    return max(1, beam_width + int(getattr(request, "num_draft_tokens", 0)))
