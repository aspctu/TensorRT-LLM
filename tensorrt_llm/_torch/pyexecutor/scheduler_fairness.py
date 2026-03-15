import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol

from tensorrt_llm.scheduling_params import (
    DEFAULT_ORGANIZATION_ID,
    get_organization_id,
    get_priority_tier,
    get_py_scheduling_params,
    normalize_organization_id,
    normalize_priority_tier,
)

# Tiering is intentionally dominant. Org balancing and age only arbitrate
# within a tier; they should not let lower-tier traffic age its way above a
# higher-tier request.
PRIORITY_TIER_BIAS = 1024.0

WAITING_AGE_CREDIT_PER_SEC = 2.0
WAITING_MAX_AGE_CREDIT = 16.0

ORG_TOKEN_BALANCE_DECAY = 0.85
ORG_RECENT_SERVICE_DECAY = 0.85
ORG_TOKEN_BALANCE_MIN = -256.0
ORG_TOKEN_SCORE_SCALE = 16.0
ADMISSION_PROMPT_TOKEN_COST_DIVISOR = 128.0
ADMISSION_MAX_NEW_TOKEN_COST_WEIGHT = 0.25
ADMISSION_MAX_CHARGE = 96.0
ORG_ACTIVE_PRESSURE_SCALE = 48.0

_MISSING = object()


class RequestQueueItemLike(Protocol):
    """Minimal queue-item surface the fairness controller consumes."""

    request: Optional[object]
    is_normal_request: bool


@dataclass(slots=True)
class SchedulerScore:
    score: float
    age_credit: float
    org_token_balance: float


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


@dataclass(slots=True)
class SchedulerRequestView:
    """Typed view over scheduler-owned request metadata.

    The underlying request objects still carry dynamic `py_*` state, but this
    view centralizes all reads and writes so the scheduler logic has one
    discoverable interface boundary.
    """

    raw_request: object

    @classmethod
    def from_request_or_item(cls, request_or_item: object) -> Optional["SchedulerRequestView"]:
        if request_or_item is None:
            return None

        request = getattr(request_or_item, "request", _MISSING)
        if request is _MISSING:
            return cls(request_or_item)
        if request is None:
            return None
        return cls(request)

    @property
    def request_id(self) -> int:
        return int(getattr(self.raw_request, "request_id", 0))

    @property
    def priority_tier(self) -> int:
        current = getattr(self.raw_request, "py_priority_tier", _MISSING)
        if current is not _MISSING:
            return normalize_priority_tier(current)

        scheduling_params = get_py_scheduling_params(self.raw_request)
        return normalize_priority_tier(get_priority_tier(scheduling_params))

    @priority_tier.setter
    def priority_tier(self, value: object) -> None:
        self.raw_request.py_priority_tier = normalize_priority_tier(value)

    @property
    def organization_id(self) -> str:
        current = getattr(self.raw_request, "py_organization_id", _MISSING)
        if current is not _MISSING:
            return normalize_organization_id(current)

        scheduling_params = get_py_scheduling_params(self.raw_request)
        return normalize_organization_id(get_organization_id(scheduling_params))

    @organization_id.setter
    def organization_id(self, value: object) -> None:
        self.raw_request.py_organization_id = normalize_organization_id(value)

    @property
    def priority_credit(self) -> float:
        return float(getattr(self.raw_request, "py_priority_credit", 0.0))

    @priority_credit.setter
    def priority_credit(self, value: object) -> None:
        self.raw_request.py_priority_credit = float(value)

    @property
    def priority_pause_count(self) -> int:
        return int(getattr(self.raw_request, "py_priority_pause_count", 0))

    @priority_pause_count.setter
    def priority_pause_count(self, value: object) -> None:
        self.raw_request.py_priority_pause_count = int(value)

    @property
    def enqueue_time(self) -> Optional[float]:
        enqueue_time = getattr(self.raw_request, "py_scheduler_enqueue_time", _MISSING)
        if enqueue_time not in (_MISSING, None):
            return float(enqueue_time)

        arrival_time = getattr(self.raw_request, "arrival_time", None)
        if arrival_time is None:
            return None
        return float(arrival_time)

    @enqueue_time.setter
    def enqueue_time(self, value: float) -> None:
        self.raw_request.py_scheduler_enqueue_time = float(value)

    @property
    def scheduler_score(self) -> float:
        return float(getattr(self.raw_request, "py_scheduler_score", 0.0))

    @scheduler_score.setter
    def scheduler_score(self, value: object) -> None:
        self.raw_request.py_scheduler_score = float(value)

    @property
    def scheduler_age_credit(self) -> float:
        return float(getattr(self.raw_request, "py_scheduler_age_credit", 0.0))

    @scheduler_age_credit.setter
    def scheduler_age_credit(self, value: object) -> None:
        self.raw_request.py_scheduler_age_credit = float(value)

    @property
    def scheduler_org_token_balance(self) -> float:
        return float(getattr(self.raw_request, "py_scheduler_org_token_balance", 0.0))

    @scheduler_org_token_balance.setter
    def scheduler_org_token_balance(self, value: object) -> None:
        self.raw_request.py_scheduler_org_token_balance = float(value)

    @property
    def is_waiting(self) -> bool:
        # Activated LlmRequest objects expose `state_value`; queued executor-side
        # requests do not. Waiting-age credit is only for not-yet-activated work.
        return not hasattr(self.raw_request, "state_value")

    def ensure_defaults(self, now: float) -> None:
        self.priority_tier = self.priority_tier
        self.organization_id = self.organization_id

        if getattr(self.raw_request, "py_priority_credit", _MISSING) is _MISSING:
            self.priority_credit = 0.0
        if getattr(self.raw_request, "py_priority_pause_count", _MISSING) is _MISSING:
            self.priority_pause_count = 0
        if getattr(self.raw_request, "py_scheduler_enqueue_time", _MISSING) in (_MISSING, None):
            self.enqueue_time = now
        if getattr(self.raw_request, "py_scheduler_score", _MISSING) is _MISSING:
            self.scheduler_score = 0.0
        if getattr(self.raw_request, "py_scheduler_age_credit", _MISSING) is _MISSING:
            self.scheduler_age_credit = 0.0
        if getattr(self.raw_request, "py_scheduler_org_token_balance", _MISSING) is _MISSING:
            self.scheduler_org_token_balance = 0.0

    def apply_score(self, score: SchedulerScore) -> None:
        self.scheduler_score = score.score
        self.scheduler_age_credit = score.age_credit
        self.scheduler_org_token_balance = score.org_token_balance


def _request_view(request_or_item: object) -> Optional[SchedulerRequestView]:
    return SchedulerRequestView.from_request_or_item(request_or_item)


def _queued_request_view(request_item: RequestQueueItemLike) -> Optional[SchedulerRequestView]:
    if not getattr(request_item, "is_normal_request", True):
        return None

    request = getattr(request_item, "request", None)
    if request is None:
        raise ValueError("Normal request item is missing request payload")

    return SchedulerRequestView(request)


def request_priority_tier(request_or_item: object) -> int:
    view = _request_view(request_or_item)
    if view is None:
        return 0
    return view.priority_tier


def request_organization_id(request_or_item: object) -> str:
    view = _request_view(request_or_item)
    if view is None:
        return DEFAULT_ORGANIZATION_ID
    return view.organization_id


def _organization_key(priority_tier: int, organization_id: str) -> str:
    return f"{priority_tier}:{organization_id}"


def _request_org_key(view: SchedulerRequestView) -> str:
    return _organization_key(view.priority_tier, view.organization_id)


def _count_by_organization(requests: Iterable[object]) -> dict[str, int]:
    counts = Counter(
        _request_org_key(view)
        for request in requests
        if (view := _request_view(request)) is not None
    )
    return {organization_key: count for organization_key, count in sorted(counts.items())}


def _prompt_token_count(request: object) -> int:
    for attr in (
        "py_orig_prompt_len",
        "orig_prompt_len",
        "py_prompt_len",
        "prompt_len",
    ):
        value = getattr(request, attr, None)
        if value is not None:
            return max(0, int(value))

    for attr in ("input_token_ids", "prompt_token_ids"):
        value = getattr(request, attr, None)
        if value is not None:
            return len(value)

    return 0


def _max_new_token_count(request: object) -> int:
    for attr in ("py_max_new_tokens", "max_new_tokens", "max_tokens"):
        value = getattr(request, attr, None)
        if value is not None:
            return max(0, int(value))

    sampling_config = getattr(request, "sampling_config", None)
    if sampling_config is None:
        return 0
    return max(0, int(getattr(sampling_config, "max_tokens", 0) or 0))


def _estimate_activation_charge(request: object) -> float:
    prompt_cost = _prompt_token_count(request) / ADMISSION_PROMPT_TOKEN_COST_DIVISOR
    decode_cost = _max_new_token_count(request) * ADMISSION_MAX_NEW_TOKEN_COST_WEIGHT
    return min(ADMISSION_MAX_CHARGE, prompt_cost + decode_cost)


def _request_org_pressure(request: object) -> float:
    return 1.0 + (_estimate_activation_charge(request) / ORG_ACTIVE_PRESSURE_SCALE)


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


def _allocate_reserved_slots(ordered_tiers: list[int], total_slots: int) -> dict[int, int]:
    if total_slots <= 0 or not ordered_tiers:
        return {}

    reserved_slots: dict[int, int] = {}
    remaining_slots = total_slots
    remaining_tiers = len(ordered_tiers)
    for tier in ordered_tiers:
        if remaining_slots <= 0:
            reserved_slots[tier] = 0
        elif remaining_tiers == 1:
            reserved_slots[tier] = remaining_slots
        else:
            reserved_slots[tier] = max(1, remaining_slots - (remaining_tiers - 1))

        remaining_slots -= reserved_slots[tier]
        remaining_tiers -= 1

    return reserved_slots


def _fair_pressure_share(
    organization_keys: Iterable[str],
    pressure_by_org: Counter[str],
) -> float:
    keys = tuple(dict.fromkeys(organization_keys))
    if not keys:
        return 0.0

    total_pressure = sum(float(pressure_by_org.get(key, 0.0)) for key in keys)
    return total_pressure / len(keys)


@dataclass
class OrganizationState:
    priority_tier: int
    organization_id: str
    token_balance: float = 0.0
    recent_service_tokens: float = 0.0
    total_enqueued: int = 0
    total_scheduled_tokens: float = 0.0


class SchedulerFairnessController:
    def __init__(self, time_fn: Optional[Callable[[], float]] = None):
        self._time_fn = time_fn or time.monotonic
        self._org_states: dict[str, OrganizationState] = {}

    def now(self) -> float:
        return float(self._time_fn())

    def _score_request(self, view: SchedulerRequestView, now: float) -> SchedulerScore:
        view.ensure_defaults(now)
        age_credit = self._waiting_age_credit(view, now)
        org_state = self._get_org_state(view.priority_tier, view.organization_id)
        org_token_balance = org_state.token_balance if org_state is not None else 0.0
        score = (
            PRIORITY_TIER_BIAS * view.priority_tier
            + view.priority_credit
            + age_credit
            + (org_token_balance / ORG_TOKEN_SCORE_SCALE)
            - (_estimate_activation_charge(view.raw_request) / ORG_TOKEN_SCORE_SCALE if view.is_waiting else 0.0)
        )
        score_state = SchedulerScore(
            score=score,
            age_credit=age_credit,
            org_token_balance=org_token_balance,
        )
        view.apply_score(score_state)
        return score_state

    def ensure_request_state(self, request: object, now: Optional[float] = None) -> None:
        view = _request_view(request)
        if view is None:
            return

        if now is None:
            now = self.now()
        view.ensure_defaults(now)

    def _get_org_state(
        self, priority_tier: int, organization_id: str, create: bool = True
    ) -> Optional[OrganizationState]:
        org_key = _organization_key(priority_tier, organization_id)
        state = self._org_states.get(org_key)
        if state is None and create:
            state = OrganizationState(
                priority_tier=priority_tier,
                organization_id=organization_id,
            )
            self._org_states[org_key] = state
        return state

    def _tier_slot_targets(
        self,
        request_items: Iterable[object],
        *,
        max_active_requests: int,
    ) -> dict[int, int]:
        waiting_tiers = sorted(
            {
                view.priority_tier
                for request_item in request_items
                if (view := _queued_request_view(request_item)) is not None
            },
            reverse=True,
        )
        return _allocate_reserved_slots(waiting_tiers, max_active_requests)

    def _per_tier_org_active_cap(
        self,
        priority_tier: int,
        *,
        request_items: Iterable[object],
        tier_slot_targets: dict[int, int],
        max_service_requests: int,
        single_tier_mode: bool,
    ) -> float:
        waiting_orgs = {
            _request_org_key(view)
            for request_item in request_items
            if (view := _queued_request_view(request_item)) is not None
            and view.priority_tier == priority_tier
        }
        tier_target = tier_slot_targets.get(priority_tier, 1)
        capacity_basis = max(
            1.0,
            float(tier_target if single_tier_mode else min(tier_target, max_service_requests)),
        )
        if not waiting_orgs:
            return capacity_basis

        return max(1.0, capacity_basis / len(waiting_orgs))

    def _advance_balancing_round(self) -> None:
        for state in self._org_states.values():
            state.token_balance = max(
                ORG_TOKEN_BALANCE_MIN,
                state.token_balance * ORG_TOKEN_BALANCE_DECAY,
            )
            state.recent_service_tokens *= ORG_RECENT_SERVICE_DECAY

    def on_requests_enqueued(
        self, request_items: Iterable[object], now: Optional[float] = None
    ) -> None:
        if now is None:
            now = self.now()

        for request_item in request_items:
            view = _queued_request_view(request_item)
            if view is None:
                continue

            view.ensure_defaults(now)
            state = self._get_org_state(view.priority_tier, view.organization_id)
            if state is None:
                continue
            state.total_enqueued += 1

    def on_requests_activated(
        self, requests: Iterable[object], now: Optional[float] = None
    ) -> None:
        if now is None:
            now = self.now()

        for request in requests:
            view = _request_view(request)
            if view is None:
                continue
            view.ensure_defaults(now)
            state = self._get_org_state(view.priority_tier, view.organization_id)
            if state is None:
                continue
            state.token_balance = max(
                ORG_TOKEN_BALANCE_MIN,
                state.token_balance - _estimate_activation_charge(view.raw_request),
            )
            view.scheduler_org_token_balance = state.token_balance

    def _waiting_age_credit(self, view: SchedulerRequestView, now: float) -> float:
        if not view.is_waiting:
            return 0.0

        enqueue_time = view.enqueue_time
        if enqueue_time is None:
            return 0.0

        waited_seconds = max(0.0, now - enqueue_time)
        return min(WAITING_MAX_AGE_CREDIT, waited_seconds * WAITING_AGE_CREDIT_PER_SEC)

    def on_requests_scheduled(
        self,
        context_requests: Iterable[object],
        generation_requests: Iterable[object],
    ) -> None:
        self._advance_balancing_round()

        for request in list(context_requests) + list(generation_requests):
            view = _request_view(request)
            if view is None:
                continue

            state = self._get_org_state(view.priority_tier, view.organization_id)
            if state is None:
                continue

            scheduled_tokens = float(scheduled_token_cost(view.raw_request))
            state.token_balance = max(
                ORG_TOKEN_BALANCE_MIN,
                state.token_balance - scheduled_tokens,
            )
            state.recent_service_tokens += scheduled_tokens
            state.total_scheduled_tokens += scheduled_tokens
            view.scheduler_org_token_balance = state.token_balance

    def score_waiting_request(
        self, request_item: object, now: Optional[float] = None
    ) -> float:
        view = _queued_request_view(request_item)
        if view is None:
            return 0.0

        if now is None:
            now = self.now()
        return self._score_request(view, now).score

    def order_waiting_requests(
        self,
        request_items: Iterable[object],
        active_requests: Iterable[object],
        now: Optional[float] = None,
    ) -> list[object]:
        """Order queued requests for admission.

        Tier remains dominant. Within a tier, this treats organization as the
        scheduling unit and prefers orgs with less already-active pressure so a
        single busy org cannot fill every newly available active slot.
        """

        if now is None:
            now = self.now()

        active_pressure: Counter[str] = Counter()
        for request in active_requests:
            view = _request_view(request)
            if view is None:
                continue
            active_pressure[_request_org_key(view)] += _request_org_pressure(view.raw_request)

        @dataclass(slots=True)
        class OrderedRequest:
            request_item: object
            order_index: int
            view: SchedulerRequestView
            score: SchedulerScore

        ordered_passthrough: list[tuple[int, object]] = []
        requests_by_tier: dict[int, dict[str, list[OrderedRequest]]] = {}
        for order_index, request_item in enumerate(request_items):
            view = _queued_request_view(request_item)
            if view is None:
                ordered_passthrough.append((order_index, request_item))
                continue

            score = self._score_request(view, now)
            organization_key = _request_org_key(view)
            tier_requests = requests_by_tier.setdefault(view.priority_tier, {})
            tier_requests.setdefault(organization_key, []).append(
                OrderedRequest(
                    request_item=request_item,
                    order_index=order_index,
                    view=view,
                    score=score,
                )
            )

        ordered_requests: list[tuple[int, object]] = []
        for priority_tier in sorted(requests_by_tier, reverse=True):
            org_requests = requests_by_tier[priority_tier]
            for requests in org_requests.values():
                requests.sort(
                    key=lambda request: (-request.score.score, request.order_index)
                )

            selected_pressure: Counter[str] = Counter()
            selected_token_costs: Counter[str] = Counter()
            while True:
                candidate_orgs = [
                    organization_key
                    for organization_key, requests in org_requests.items()
                    if requests
                ]
                if not candidate_orgs:
                    break

                current_pressure_by_org: Counter[str] = Counter(
                    {
                        key: active_pressure.get(key, 0.0)
                        + selected_pressure.get(key, 0.0)
                        for key in candidate_orgs
                    }
                )
                fair_pressure_share = _fair_pressure_share(
                    candidate_orgs, current_pressure_by_org
                )

                organization_key = min(
                    candidate_orgs,
                    key=lambda key: (
                        max(
                            0.0,
                            current_pressure_by_org.get(key, 0.0) - fair_pressure_share,
                        ),
                        current_pressure_by_org.get(key, 0.0),
                        -(
                            org_requests[key][0].score.score
                            - (selected_token_costs.get(key, 0.0) / ORG_TOKEN_SCORE_SCALE)
                        ),
                        org_requests[key][0].order_index,
                    ),
                )
                next_request = org_requests[organization_key].pop(0)
                selected_pressure[organization_key] += _request_org_pressure(
                    next_request.view.raw_request
                )
                selected_token_costs[organization_key] += _estimate_activation_charge(
                    next_request.view.raw_request
                )
                ordered_requests.append((next_request.order_index, next_request.request_item))

        passthrough_requests = [
            request_item for _, request_item in sorted(ordered_passthrough, key=lambda item: item[0])
        ]
        return [request_item for _, request_item in ordered_requests] + passthrough_requests

    def select_waiting_requests(
        self,
        request_items: Iterable[object],
        active_requests: Iterable[object],
        max_new_requests: int,
        max_active_requests: int,
        max_service_requests: Optional[int] = None,
        now: Optional[float] = None,
    ) -> list[object]:
        """Select queued requests for activation.

        Tiering remains dominant. Within a tier, orgs are limited by a fair
        share of the active set so one org cannot immediately repopulate every
        newly free slot under GUARANTEED_NO_EVICT.
        """

        if max_new_requests <= 0:
            return []

        if now is None:
            now = self.now()
        if max_service_requests is None:
            max_service_requests = max_active_requests

        ordered_requests = self.order_waiting_requests(
            request_items,
            active_requests,
            now=now,
        )

        tier_slot_targets = self._tier_slot_targets(
            request_items,
            max_active_requests=max_active_requests,
        )
        if not tier_slot_targets:
            return ordered_requests[:max_new_requests]
        single_tier_mode = len(tier_slot_targets) == 1

        active_count_by_tier = Counter(
            view.priority_tier
            for request in active_requests
            if (view := _request_view(request)) is not None
        )
        active_pressure_by_org: Counter[str] = Counter()
        for request in active_requests:
            view = _request_view(request)
            if view is None:
                continue
            active_pressure_by_org[_request_org_key(view)] += _request_org_pressure(
                view.raw_request
            )
        selected_count_by_tier: Counter[int] = Counter()
        selected_pressure_by_org: Counter[str] = Counter()
        active_count = sum(active_count_by_tier.values())
        selected_requests: list[object] = []
        waiting_orgs_by_tier = {
            tier: {
                _request_org_key(view)
                for request_item in request_items
                if (view := _queued_request_view(request_item)) is not None
                and view.priority_tier == tier
            }
            for tier in tier_slot_targets
        }

        for request_item in ordered_requests:
            if len(selected_requests) >= max_new_requests:
                break

            view = _queued_request_view(request_item)
            if view is None:
                selected_requests.append(request_item)
                continue

            higher_tiers_needing_reserve = [
                tier
                for tier in tier_slot_targets
                if tier > view.priority_tier
                and active_count_by_tier.get(tier, 0) + selected_count_by_tier.get(tier, 0)
                < tier_slot_targets[tier]
            ]
            if higher_tiers_needing_reserve:
                remaining_reserved_slots = sum(
                    max(
                        0,
                        tier_slot_targets[tier]
                        - active_count_by_tier.get(tier, 0)
                        - selected_count_by_tier.get(tier, 0),
                    )
                    for tier in higher_tiers_needing_reserve
                )
                future_active_count = active_count + len(selected_requests) + 1
                if future_active_count > (max_active_requests - remaining_reserved_slots):
                    continue

            org_cap = self._per_tier_org_active_cap(
                view.priority_tier,
                request_items=request_items,
                tier_slot_targets=tier_slot_targets,
                max_service_requests=max_service_requests,
                single_tier_mode=single_tier_mode,
            )
            org_key = _request_org_key(view)
            current_org_pressure = (
                active_pressure_by_org.get(org_key, 0.0)
                + selected_pressure_by_org.get(org_key, 0.0)
            )
            peer_can_still_use_capacity = any(
                (
                    active_pressure_by_org.get(peer_org_key, 0.0)
                    + selected_pressure_by_org.get(peer_org_key, 0.0)
                )
                < org_cap
                for peer_org_key in waiting_orgs_by_tier.get(view.priority_tier, set())
                if peer_org_key != org_key
            )
            if current_org_pressure >= org_cap and peer_can_still_use_capacity:
                continue

            selected_requests.append(request_item)
            selected_count_by_tier[view.priority_tier] += 1
            selected_pressure_by_org[org_key] += _request_org_pressure(view.raw_request)

        return selected_requests

    def build_organization_stats(
        self,
        active_requests: Iterable[object],
        queued_requests: Iterable[object],
        waiting_requests: Iterable[object],
        scheduled_context_requests: Iterable[object],
        scheduled_generation_requests: Iterable[object],
        paused_requests: Iterable[object],
        now: Optional[float] = None,
    ) -> dict[str, dict[str, float | int]]:
        active_counts = _count_by_organization(active_requests)
        queued_counts = _count_by_organization(queued_requests)
        waiting_counts = _count_by_organization(waiting_requests)
        scheduled_context_counts = _count_by_organization(scheduled_context_requests)
        scheduled_generation_counts = _count_by_organization(scheduled_generation_requests)
        paused_counts = _count_by_organization(paused_requests)

        organization_ids = set(self._org_states)
        organization_ids.update(active_counts)
        organization_ids.update(queued_counts)
        organization_ids.update(waiting_counts)
        organization_ids.update(scheduled_context_counts)
        organization_ids.update(scheduled_generation_counts)
        organization_ids.update(paused_counts)

        stats: dict[str, dict[str, float | int]] = {}
        for organization_key in sorted(organization_ids):
            state = self._org_states.get(organization_key)
            total_enqueued = state.total_enqueued if state is not None else 0
            total_scheduled_tokens = (
                float(state.total_scheduled_tokens) if state is not None else 0.0
            )
            token_balance = float(state.token_balance) if state is not None else 0.0
            recent_service_tokens = (
                float(state.recent_service_tokens) if state is not None else 0.0
            )

            if state is not None:
                priority_tier = state.priority_tier
                organization_id = state.organization_id
            else:
                priority_tier_str, organization_id = organization_key.split(":", 1)
                priority_tier = int(priority_tier_str)

            stats[organization_key] = {
                "priorityTier": int(priority_tier),
                "organizationId": organization_id,
                "active": active_counts.get(organization_key, 0),
                "queued": queued_counts.get(organization_key, 0),
                "waiting": waiting_counts.get(organization_key, 0),
                "scheduled": scheduled_context_counts.get(organization_key, 0)
                + scheduled_generation_counts.get(organization_key, 0),
                "scheduledContext": scheduled_context_counts.get(organization_key, 0),
                "scheduledGeneration": scheduled_generation_counts.get(organization_key, 0),
                "paused": paused_counts.get(organization_key, 0),
                "orgTokenBalance": token_balance,
                "recentServiceTokens": recent_service_tokens,
                "totalEnqueued": int(total_enqueued),
                "totalScheduledTokens": total_scheduled_tokens,
            }

        return stats
