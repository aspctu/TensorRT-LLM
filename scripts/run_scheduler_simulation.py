#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, TypeVar

TIER_PRIORITY_BIAS = 1024.0
WAITING_AGE_CREDIT_PER_SEC = 2.0
WAITING_MAX_AGE_CREDIT = 16.0
REQUEST_WAIT_CREDIT = 0.75
REQUEST_PAUSE_BONUS = 1.5
REQUEST_PAUSE_PROTECTION = 0.25
REQUEST_SERVICE_COST = 0.75
REQUEST_MAX_CREDIT = 8.0
ORG_TOKEN_BALANCE_DECAY = 0.85
ORG_TOKEN_BALANCE_MIN = -256.0
ORG_TOKEN_SCORE_SCALE = 16.0
ORG_DRR_QUANTUM = 2.0
ADMISSION_PROMPT_TOKEN_COST_DIVISOR = 128.0
ADMISSION_MAX_NEW_TOKEN_COST_WEIGHT = 0.25
ADMISSION_MAX_CHARGE = 96.0
ORG_ACTIVE_PRESSURE_SCALE = 48.0
ORG_ACTIVE_PRESSURE_PENALTY_SCALE = 8.0
SAME_TIER_HARM_WAIT_S = 60.0
STARVATION_GAP_S = 120.0
TTFT_SLA_S = 10.0
LATENCY_SLA_S = 60.0


@dataclass(frozen=True)
class RequestShape:
    name: str
    prompt_tokens: int
    max_tokens: int
    weight: float


@dataclass(frozen=True)
class OrgProfile:
    org_id: str
    priority_tier: int
    weight: float
    shape_weight_scale: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrafficPhase:
    name: str
    fraction: float
    qps_multiplier: float = 1.0
    org_weight_scale: dict[str, float] = field(default_factory=dict)
    shape_weight_scale: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrafficProfile:
    name: str
    description: str
    org_profiles: tuple[OrgProfile, ...]
    request_shapes: tuple[RequestShape, ...]
    phases: tuple[TrafficPhase, ...]
    default_total_requests: int
    default_qps: float


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    arrival_time_s: float
    organization_id: str
    priority_tier: int
    shape_name: str
    phase_name: str
    prompt_tokens: int
    max_tokens: int


@dataclass
class RequestState:
    spec: RequestSpec
    remaining_prompt_tokens: int
    remaining_decode_tokens: int
    scheduler_credit: float = 0.0
    admitted_time_s: Optional[float] = None
    first_token_time_s: Optional[float] = None
    completion_time_s: Optional[float] = None
    last_progress_time_s: Optional[float] = None
    max_no_progress_gap_s: float = 0.0
    total_service_tokens: int = 0
    total_generated_tokens: int = 0
    pause_count: int = 0

    @property
    def request_id(self) -> int:
        return self.spec.request_id

    @property
    def priority_tier(self) -> int:
        return self.spec.priority_tier

    @property
    def organization_id(self) -> str:
        return self.spec.organization_id

    @property
    def organization_key(self) -> str:
        return f"{self.priority_tier}:{self.organization_id}"

    @property
    def is_complete(self) -> bool:
        return self.completion_time_s is not None

    @property
    def is_prompt_phase(self) -> bool:
        return self.remaining_prompt_tokens > 0


@dataclass
class OrganizationState:
    token_balance: float = 0.0
    recent_service_tokens: float = 0.0
    total_service_tokens: float = 0.0


@dataclass
class TierCrowdoutStats:
    backlog_ticks: int = 0
    service_ticks: int = 0
    blocked_by_higher_tier_ticks: int = 0
    service_tokens: int = 0
    leaked_service_tokens: int = 0


DEFAULT_SHAPES = (
    RequestShape(name="short-chat", prompt_tokens=384, max_tokens=48, weight=4),
    RequestShape(name="medium-chat", prompt_tokens=1280, max_tokens=128, weight=3),
    RequestShape(name="long-prompt", prompt_tokens=4608, max_tokens=96, weight=2),
    RequestShape(name="long-decode", prompt_tokens=1536, max_tokens=320, weight=3),
)


def make_profiles() -> dict[str, TrafficProfile]:
    return {
        "same_tier_burst": TrafficProfile(
            name="same_tier_burst",
            description="All orgs share one tier, but one bursty org sends decode-heavy traffic mid-run.",
            org_profiles=(
                OrgProfile("tier1-steady-a", 1, 4, {"short-chat": 1.5, "medium-chat": 1.2}),
                OrgProfile("tier1-steady-b", 1, 4, {"short-chat": 1.3, "medium-chat": 1.3}),
                OrgProfile("tier1-burst", 1, 1, {"long-decode": 3.2, "long-prompt": 1.5}),
                OrgProfile("tier1-drain", 1, 2, {"long-decode": 2.5}),
            ),
            request_shapes=DEFAULT_SHAPES,
            phases=(
                TrafficPhase("steady", 0.25, 1.0),
                TrafficPhase(
                    "burst",
                    0.50,
                    1.8,
                    org_weight_scale={"tier1-burst": 8.0, "tier1-drain": 1.8},
                    shape_weight_scale={"long-decode": 1.8},
                ),
                TrafficPhase("recover", 0.25, 1.2),
            ),
            default_total_requests=360,
            default_qps=4.2,
        ),
        "tier_mix_pressure": TrafficProfile(
            name="tier_mix_pressure",
            description="Premium tiers stay interactive while tier-0 traffic floods the worker with decode-heavy requests.",
            org_profiles=(
                OrgProfile("tier3-gold-a", 3, 2, {"short-chat": 2.0, "medium-chat": 1.4}),
                OrgProfile("tier3-gold-b", 3, 2, {"short-chat": 1.8, "medium-chat": 1.5}),
                OrgProfile("tier2-silver", 2, 4, {"medium-chat": 1.5, "long-prompt": 1.2}),
                OrgProfile("tier1-bronze-chat", 1, 6, {"medium-chat": 1.3}),
                OrgProfile("tier0-bronze-flood", 0, 10, {"long-decode": 3.5, "long-prompt": 2.2}),
            ),
            request_shapes=DEFAULT_SHAPES,
            phases=(
                TrafficPhase("steady", 0.40, 1.0),
                TrafficPhase(
                    "overload",
                    0.40,
                    2.1,
                    org_weight_scale={"tier0-bronze-flood": 1.6},
                    shape_weight_scale={"long-decode": 1.5},
                ),
                TrafficPhase("tail", 0.20, 1.2),
            ),
            default_total_requests=384,
            default_qps=4.5,
        ),
        "premium_protection": TrafficProfile(
            name="premium_protection",
            description="Tier-4 premium traffic competes with a very large tier-0 background flood.",
            org_profiles=(
                OrgProfile("tier4-premium-a", 4, 1, {"short-chat": 2.2, "medium-chat": 1.4}),
                OrgProfile("tier4-premium-b", 4, 1, {"short-chat": 2.2, "medium-chat": 1.4}),
                OrgProfile("tier2-standard-a", 2, 4, {"medium-chat": 1.5}),
                OrgProfile("tier2-standard-b", 2, 4, {"medium-chat": 1.5}),
                OrgProfile("tier0-background", 0, 12, {"long-decode": 4.0, "long-prompt": 2.0}),
            ),
            request_shapes=DEFAULT_SHAPES,
            phases=(
                TrafficPhase("steady", 0.20, 1.0),
                TrafficPhase(
                    "flood",
                    0.60,
                    2.6,
                    org_weight_scale={"tier0-background": 2.5},
                    shape_weight_scale={"long-decode": 2.0},
                ),
                TrafficPhase("recovery", 0.20, 1.1),
            ),
            default_total_requests=320,
            default_qps=4.0,
        ),
        "premium_noisy_neighbor": TrafficProfile(
            name="premium_noisy_neighbor",
            description="One premium org is chat-heavy while another premium org turns into a decode-heavy noisy neighbor.",
            org_profiles=(
                OrgProfile("tier4-chat", 4, 2, {"short-chat": 2.4, "medium-chat": 1.6}),
                OrgProfile("tier4-noisy", 4, 2, {"long-decode": 4.2, "long-prompt": 1.8}),
                OrgProfile("tier2-standard", 2, 5, {"medium-chat": 1.4}),
                OrgProfile("tier0-flood", 0, 10, {"long-decode": 3.8, "long-prompt": 2.0}),
            ),
            request_shapes=DEFAULT_SHAPES,
            phases=(
                TrafficPhase("steady", 0.20, 1.0),
                TrafficPhase(
                    "premium_burst",
                    0.50,
                    2.0,
                    org_weight_scale={"tier4-noisy": 2.8, "tier0-flood": 1.6},
                    shape_weight_scale={"long-decode": 1.8},
                ),
                TrafficPhase("tail", 0.30, 1.1),
            ),
            default_total_requests=320,
            default_qps=4.0,
        ),
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, quantile)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def allocate_phase_counts(total_requests: int, phases: tuple[TrafficPhase, ...]) -> list[int]:
    counts: list[int] = []
    assigned = 0
    cumulative_fraction = 0.0
    for index, phase in enumerate(phases):
        cumulative_fraction += phase.fraction
        if index == len(phases) - 1:
            target_assigned = total_requests
        else:
            target_assigned = int(round(total_requests * cumulative_fraction))
        counts.append(max(0, target_assigned - assigned))
        assigned = target_assigned
    return counts


def weighted_choice(rng: random.Random, items: tuple[T, ...], weights: tuple[float, ...]) -> T:
    total = sum(weights)
    target = rng.uniform(0.0, total)
    running = 0.0
    for item, weight in zip(items, weights):
        running += weight
        if target <= running:
            return item
    return items[-1]


def generate_requests(
    *,
    profile: TrafficProfile,
    seed: int,
    total_requests: int,
    qps: float,
) -> list[RequestSpec]:
    rng = random.Random(seed)
    phase_counts = allocate_phase_counts(total_requests, profile.phases)
    requests: list[RequestSpec] = []
    next_request_id = 1
    arrival_time_s = 0.0
    org_profiles = tuple(profile.org_profiles)
    request_shapes = tuple(profile.request_shapes)

    for phase, phase_count in zip(profile.phases, phase_counts):
        phase_qps = max(0.05, qps * phase.qps_multiplier)
        for _ in range(phase_count):
            arrival_time_s += rng.expovariate(phase_qps)
            org_weights = tuple(
                org.weight * phase.org_weight_scale.get(org.org_id, 1.0)
                for org in org_profiles
            )
            org = weighted_choice(rng, org_profiles, org_weights)
            shape_weights = tuple(
                shape.weight
                * org.shape_weight_scale.get(shape.name, 1.0)
                * phase.shape_weight_scale.get(shape.name, 1.0)
                for shape in request_shapes
            )
            shape = weighted_choice(rng, request_shapes, shape_weights)
            requests.append(
                RequestSpec(
                    request_id=next_request_id,
                    arrival_time_s=arrival_time_s,
                    organization_id=org.org_id,
                    priority_tier=org.priority_tier,
                    shape_name=shape.name,
                    phase_name=phase.name,
                    prompt_tokens=shape.prompt_tokens,
                    max_tokens=shape.max_tokens,
                )
            )
            next_request_id += 1
    return requests


def _score_waiting_request(
    request: RequestState,
    *,
    now_s: float,
    org_states: dict[str, OrganizationState],
) -> float:
    waited_s = max(0.0, now_s - request.spec.arrival_time_s)
    age_credit = min(WAITING_MAX_AGE_CREDIT, waited_s * WAITING_AGE_CREDIT_PER_SEC)
    org_state = org_states[request.organization_key]
    return (
        request.priority_tier * TIER_PRIORITY_BIAS
        + request.scheduler_credit
        + (request.pause_count * REQUEST_PAUSE_PROTECTION)
        + age_credit
        + (org_state.token_balance / ORG_TOKEN_SCORE_SCALE)
        - (_estimate_activation_charge(request) / ORG_TOKEN_SCORE_SCALE)
    )


def _score_active_request(
    request: RequestState,
    *,
    org_states: dict[str, OrganizationState],
) -> float:
    org_state = org_states[request.organization_key]
    return (
        request.priority_tier * TIER_PRIORITY_BIAS
        + request.scheduler_credit
        + (request.pause_count * REQUEST_PAUSE_PROTECTION)
        + (org_state.token_balance / ORG_TOKEN_SCORE_SCALE)
    )


def _candidate_waiting_order(
    waiting: list[RequestState],
    active: list[RequestState],
    *,
    now_s: float,
    org_states: dict[str, OrganizationState],
    same_tier_harm_by_tier: Optional[dict[int, bool]] = None,
) -> list[RequestState]:
    if same_tier_harm_by_tier is None:
        same_tier_harm_by_tier = _same_tier_harm_by_tier(
            waiting=waiting,
            active=active,
            now_s=now_s,
        )

    active_pressure = defaultdict(float)
    for request in active:
        active_pressure[request.organization_key] += _request_org_pressure(request)
    requests_by_tier: dict[int, dict[str, list[RequestState]]] = defaultdict(lambda: defaultdict(list))
    for request in waiting:
        requests_by_tier[request.priority_tier][request.organization_key].append(request)

    ordered: list[RequestState] = []
    for priority_tier in sorted(requests_by_tier, reverse=True):
        requests_by_org = requests_by_tier[priority_tier]
        for org_requests in requests_by_org.values():
            org_requests.sort(
                key=lambda request: (
                    -_score_waiting_request(request, now_s=now_s, org_states=org_states),
                    request.spec.arrival_time_s,
                    request.request_id,
                )
            )

        if not same_tier_harm_by_tier.get(priority_tier, False):
            ordered.extend(
                sorted(
                    (request for org_requests in requests_by_org.values() for request in org_requests),
                    key=lambda request: (
                        -_score_waiting_request(request, now_s=now_s, org_states=org_states),
                        request.spec.arrival_time_s,
                        request.request_id,
                    ),
                )
            )
            continue

        selected_pressure: Counter[str] = Counter()
        selected_token_costs: Counter[str] = Counter()
        while True:
            candidate_orgs = [org_key for org_key, org_requests in requests_by_org.items() if org_requests]
            if not candidate_orgs:
                break
            current_pressure_by_org = {
                org_key: active_pressure.get(org_key, 0.0) + selected_pressure.get(org_key, 0.0)
                for org_key in candidate_orgs
            }
            fair_pressure_share = sum(current_pressure_by_org.values()) / len(current_pressure_by_org)
            organization_key = min(
                candidate_orgs,
                key=lambda org_key: (
                    max(0.0, current_pressure_by_org.get(org_key, 0.0) - fair_pressure_share),
                    current_pressure_by_org.get(org_key, 0.0),
                    -(
                        _score_waiting_request(
                            requests_by_org[org_key][0], now_s=now_s, org_states=org_states
                        )
                        - (selected_token_costs.get(org_key, 0.0) / ORG_TOKEN_SCORE_SCALE)
                    ),
                    requests_by_org[org_key][0].spec.arrival_time_s,
                    requests_by_org[org_key][0].request_id,
                ),
            )
            next_request = requests_by_org[organization_key].pop(0)
            selected_pressure[organization_key] += _request_org_pressure(next_request)
            selected_token_costs[organization_key] += _estimate_activation_charge(next_request)
            ordered.append(next_request)
    return ordered


def _order_requests_within_tier(
    requests: list[RequestState],
    *,
    org_states: dict[str, OrganizationState],
    service_quota: int,
    same_tier_harm: bool,
) -> list[RequestState]:
    if not same_tier_harm:
        return sorted(
            requests,
            key=lambda request: (
                -_score_active_request(request, org_states=org_states),
                request.spec.arrival_time_s,
                request.request_id,
            ),
        )

    requests_by_org: dict[str, list[RequestState]] = defaultdict(list)
    for request in requests:
        requests_by_org[request.organization_key].append(request)

    for org_key, org_requests in requests_by_org.items():
        org_requests.sort(
            key=lambda request: (
                -_score_active_request(request, org_states=org_states),
                request.spec.arrival_time_s,
                request.request_id,
            )
        )

    deficits = {
        org_key: (org_states[org_key].token_balance / ORG_TOKEN_SCORE_SCALE)
        for org_key in requests_by_org
    }
    ordered: list[RequestState] = []
    while True:
        active_orgs = [
            org_key
            for org_key, org_requests in requests_by_org.items()
            if org_requests
        ]
        if not active_orgs:
            break

        emitted_in_round = False
        round_orgs = sorted(
            active_orgs,
            key=lambda org_key: (
                -deficits[org_key],
                -_score_active_request(requests_by_org[org_key][0], org_states=org_states),
                requests_by_org[org_key][0].spec.arrival_time_s,
                requests_by_org[org_key][0].request_id,
            ),
        )
        for org_key in round_orgs:
            if not requests_by_org[org_key]:
                continue
            deficits[org_key] += ORG_DRR_QUANTUM
            request = requests_by_org[org_key][0]
            request_cost = _request_org_pressure(request)
            if deficits[org_key] < request_cost:
                continue

            ordered.append(requests_by_org[org_key].pop(0))
            deficits[org_key] -= request_cost
            emitted_in_round = True

        if emitted_in_round:
            continue

        fallback_org_key = max(
            active_orgs,
            key=lambda org_key: (
                deficits[org_key],
                _score_active_request(requests_by_org[org_key][0], org_states=org_states),
                -requests_by_org[org_key][0].spec.arrival_time_s,
                -requests_by_org[org_key][0].request_id,
            ),
        )
        fallback_request = requests_by_org[fallback_org_key].pop(0)
        ordered.append(fallback_request)
        deficits[fallback_org_key] -= _request_org_pressure(fallback_request)

    return ordered


def _stock_waiting_order(waiting: list[RequestState]) -> list[RequestState]:
    return sorted(
        waiting,
        key=lambda request: (
            request.spec.arrival_time_s,
            request.request_id,
        ),
    )


def order_waiting_requests(
    *,
    policy: str,
    waiting: list[RequestState],
    active: list[RequestState],
    now_s: float,
    org_states: dict[str, OrganizationState],
    same_tier_harm_by_tier: Optional[dict[int, bool]] = None,
) -> list[RequestState]:
    if policy == "stock_default":
        return _stock_waiting_order(waiting)
    if policy == "tier_aware_max_utilization":
        return _candidate_waiting_order(
            waiting,
            active,
            now_s=now_s,
            org_states=org_states,
            same_tier_harm_by_tier=same_tier_harm_by_tier,
        )
    raise ValueError(f"unsupported policy: {policy}")


def _allocate_reserved_slots(
    ordered_tiers: list[int],
    *,
    total_slots: int,
) -> dict[int, int]:
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


def _estimate_activation_charge(request: RequestState) -> float:
    prompt_cost = request.spec.prompt_tokens / ADMISSION_PROMPT_TOKEN_COST_DIVISOR
    decode_cost = request.spec.max_tokens * ADMISSION_MAX_NEW_TOKEN_COST_WEIGHT
    return min(ADMISSION_MAX_CHARGE, prompt_cost + decode_cost)


def _request_org_pressure(request: RequestState) -> float:
    return 1.0 + (_estimate_activation_charge(request) / ORG_ACTIVE_PRESSURE_SCALE)


def _same_tier_harm_by_tier(
    *,
    waiting: list[RequestState],
    active: list[RequestState],
    now_s: float,
) -> dict[int, bool]:
    all_orgs_by_tier: dict[int, set[str]] = defaultdict(set)
    active_pressure_by_tier_org: dict[int, Counter[str]] = defaultdict(Counter)
    old_waiting_orgs_by_tier: dict[int, set[str]] = defaultdict(set)

    for request in active:
        all_orgs_by_tier[request.priority_tier].add(request.organization_key)
        active_pressure_by_tier_org[request.priority_tier][request.organization_key] += _request_org_pressure(request)

    for request in waiting:
        all_orgs_by_tier[request.priority_tier].add(request.organization_key)
        if (now_s - request.spec.arrival_time_s) >= SAME_TIER_HARM_WAIT_S:
            old_waiting_orgs_by_tier[request.priority_tier].add(request.organization_key)

    harmed: dict[int, bool] = {}
    for priority_tier, org_keys in all_orgs_by_tier.items():
        if len(org_keys) < 2:
            harmed[priority_tier] = False
            continue

        old_waiting_orgs = old_waiting_orgs_by_tier.get(priority_tier, set())
        if len(old_waiting_orgs) >= 2:
            harmed[priority_tier] = True
            continue

        if not old_waiting_orgs:
            harmed[priority_tier] = False
            continue

        active_pressures = active_pressure_by_tier_org.get(priority_tier, Counter())
        if not active_pressures:
            harmed[priority_tier] = False
            continue

        fair_pressure_share = sum(active_pressures.values()) / len(org_keys)
        harmed[priority_tier] = any(
            org_key not in old_waiting_orgs and pressure > fair_pressure_share
            for org_key, pressure in active_pressures.items()
        )

    return harmed


def _tier_slot_targets(
    requests: list[RequestState],
    *,
    total_slots: int,
) -> dict[int, int]:
    backlogged_tiers = sorted({request.priority_tier for request in requests}, reverse=True)
    return _allocate_reserved_slots(backlogged_tiers, total_slots=total_slots)


def _per_tier_org_active_cap(
    priority_tier: int,
    *,
    waiting: list[RequestState],
    tier_slot_targets: dict[int, int],
    max_service_requests: int,
    single_tier_mode: bool,
) -> float:
    waiting_orgs = {
        request.organization_key
        for request in waiting
        if request.priority_tier == priority_tier
    }
    tier_target = tier_slot_targets.get(priority_tier, 1)
    capacity_basis = max(
        1.0,
        float(tier_target if single_tier_mode else min(tier_target, max_service_requests)),
    )
    if not waiting_orgs:
        return capacity_basis

    return max(1.0, capacity_basis / len(waiting_orgs))



def select_waiting_requests(
    *,
    policy: str,
    waiting: list[RequestState],
    active: list[RequestState],
    now_s: float,
    org_states: dict[str, OrganizationState],
    max_new_requests: int,
    max_active_requests: int,
    max_service_requests: int,
    same_tier_harm_by_tier: Optional[dict[int, bool]] = None,
) -> list[RequestState]:
    ordered_waiting = order_waiting_requests(
        policy=policy,
        waiting=waiting,
        active=active,
        now_s=now_s,
        org_states=org_states,
        same_tier_harm_by_tier=same_tier_harm_by_tier,
    )
    if policy != "tier_aware_max_utilization":
        return ordered_waiting[:max_new_requests]

    tier_slot_targets = _tier_slot_targets(
        waiting,
        total_slots=max_active_requests,
    )
    single_tier_mode = len(tier_slot_targets) == 1
    active_count_by_tier = Counter(request.priority_tier for request in active)
    active_pressure_by_org = defaultdict(float)
    for request in active:
        active_pressure_by_org[request.organization_key] += _request_org_pressure(request)
    selected_count_by_tier: Counter[int] = Counter()
    selected_pressure_by_org: Counter[str] = Counter()
    selected: list[RequestState] = []
    total_active = len(active)
    waiting_by_tier_org = {
        tier: {
            request.organization_key
            for request in waiting
            if request.priority_tier == tier
        }
        for tier in tier_slot_targets
    }

    for request in ordered_waiting:
        if len(selected) >= max_new_requests:
            break

        higher_tiers_needing_reserve = [
            tier
            for tier in tier_slot_targets
            if tier > request.priority_tier
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
            future_occupancy = total_active + len(selected) + 1
            if future_occupancy > (max_active_requests - remaining_reserved_slots):
                continue

        if same_tier_harm_by_tier and same_tier_harm_by_tier.get(request.priority_tier, False):
            org_cap = _per_tier_org_active_cap(
                request.priority_tier,
                waiting=waiting,
                tier_slot_targets=tier_slot_targets,
                max_service_requests=max_service_requests,
                single_tier_mode=single_tier_mode,
            )
            org_key = request.organization_key
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
                for peer_org_key in waiting_by_tier_org.get(request.priority_tier, set())
                if peer_org_key != org_key
            )
            if current_org_pressure >= org_cap and peer_can_still_use_capacity:
                continue

        selected.append(request)
        selected_count_by_tier[request.priority_tier] += 1
        selected_pressure_by_org[request.organization_key] += _request_org_pressure(request)

    return selected


def _hierarchical_active_order(
    active: list[RequestState],
    *,
    org_states: dict[str, OrganizationState],
    max_scheduled_per_tick: int,
    same_tier_harm_by_tier: Optional[dict[int, bool]] = None,
) -> list[RequestState]:
    if not active:
        return []

    requests_by_tier: dict[int, list[RequestState]] = defaultdict(list)
    for request in active:
        requests_by_tier[request.priority_tier].append(request)

    ordered_tiers = sorted(requests_by_tier, reverse=True)
    quotas = _allocate_reserved_slots(
        ordered_tiers,
        total_slots=max_scheduled_per_tick,
    )
    tier_queues = {
        tier: _order_requests_within_tier(
            requests_by_tier[tier],
            org_states=org_states,
            service_quota=quotas.get(tier, 0),
            same_tier_harm=(same_tier_harm_by_tier or {}).get(tier, False),
        )
        for tier in ordered_tiers
    }

    scheduled: list[RequestState] = []
    for tier in ordered_tiers:
        quota = quotas.get(tier, 0)
        for _ in range(min(quota, len(tier_queues[tier]))):
            scheduled.append(tier_queues[tier].pop(0))
            if len(scheduled) >= max_scheduled_per_tick:
                return scheduled

    while len(scheduled) < max_scheduled_per_tick:
        added = False
        for tier in ordered_tiers:
            if not tier_queues[tier]:
                continue
            scheduled.append(tier_queues[tier].pop(0))
            added = True
            if len(scheduled) >= max_scheduled_per_tick:
                break
        if not added:
            break
    return scheduled


def _stock_active_order(active: list[RequestState]) -> list[RequestState]:
    return sorted(
        active,
        key=lambda request: (
            -request.priority_tier,
            request.spec.arrival_time_s,
            request.request_id,
        ),
    )


def order_active_requests(
    *,
    policy: str,
    active: list[RequestState],
    org_states: dict[str, OrganizationState],
    max_active_requests: int,
    max_scheduled_per_tick: int,
    same_tier_harm_by_tier: Optional[dict[int, bool]] = None,
) -> list[RequestState]:
    if policy == "stock_default":
        return _stock_active_order(active)
    if policy == "tier_aware_max_utilization":
        return _hierarchical_active_order(
            active,
            org_states=org_states,
            max_scheduled_per_tick=max_scheduled_per_tick,
            same_tier_harm_by_tier=same_tier_harm_by_tier,
        )
    raise ValueError(f"unsupported policy: {policy}")


def decay_org_states(org_states: dict[str, OrganizationState]) -> None:
    for state in org_states.values():
        state.token_balance = max(ORG_TOKEN_BALANCE_MIN, state.token_balance * ORG_TOKEN_BALANCE_DECAY)
        state.recent_service_tokens *= ORG_TOKEN_BALANCE_DECAY


def service_request(
    request: RequestState,
    *,
    now_s: float,
    tick_s: float,
    ctx_chunk_tokens: int,
) -> int:
    last_progress_time_s = request.spec.arrival_time_s if request.last_progress_time_s is None else request.last_progress_time_s
    request.max_no_progress_gap_s = max(request.max_no_progress_gap_s, now_s - last_progress_time_s)
    request.last_progress_time_s = now_s

    if request.remaining_prompt_tokens > 0:
        served = min(ctx_chunk_tokens, request.remaining_prompt_tokens)
        request.remaining_prompt_tokens -= served
        request.total_service_tokens += served
        return served

    request.remaining_decode_tokens -= 1
    request.total_service_tokens += 1
    request.total_generated_tokens += 1
    if request.first_token_time_s is None:
        request.first_token_time_s = now_s + tick_s
    if request.remaining_decode_tokens == 0:
        request.completion_time_s = now_s + tick_s
    return 1


def finalize_request_gaps(requests: Iterable[RequestState], *, end_time_s: float) -> None:
    for request in requests:
        last_progress_time_s = request.spec.arrival_time_s if request.last_progress_time_s is None else request.last_progress_time_s
        request.max_no_progress_gap_s = max(request.max_no_progress_gap_s, end_time_s - last_progress_time_s)


def summarize_group(
    requests: list[RequestState],
    *,
    wall_time_s: float,
) -> dict[str, float | int | None]:
    submitted = len(requests)
    completed = sum(1 for request in requests if request.is_complete)
    admission_delays = [
        request.admitted_time_s - request.spec.arrival_time_s
        for request in requests
        if request.admitted_time_s is not None
    ]
    ttfts = [
        request.first_token_time_s - request.spec.arrival_time_s
        for request in requests
        if request.first_token_time_s is not None
    ]
    latencies = [
        request.completion_time_s - request.spec.arrival_time_s
        for request in requests
        if request.completion_time_s is not None
    ]
    max_gaps = [request.max_no_progress_gap_s for request in requests]
    generated_tokens = sum(request.total_generated_tokens for request in requests)
    service_tokens = sum(request.total_service_tokens for request in requests)
    starved = sum(
        1
        for request in requests
        if (request.max_no_progress_gap_s >= STARVATION_GAP_S) or not request.is_complete
    )
    ttft_sla_misses = sum(
        1
        for request in requests
        if request.first_token_time_s is None
        or (request.first_token_time_s - request.spec.arrival_time_s) > TTFT_SLA_S
    )
    latency_sla_misses = sum(
        1
        for request in requests
        if request.completion_time_s is None
        or (request.completion_time_s - request.spec.arrival_time_s) > LATENCY_SLA_S
    )
    return {
        "submitted": submitted,
        "completed": completed,
        "completion_rate": round_or_none(completed / submitted if submitted else None, 4),
        "admission_p50_s": round_or_none(percentile(admission_delays, 0.50)),
        "admission_p95_s": round_or_none(percentile(admission_delays, 0.95)),
        "ttft_p50_s": round_or_none(percentile(ttfts, 0.50)),
        "ttft_p95_s": round_or_none(percentile(ttfts, 0.95)),
        "latency_p50_s": round_or_none(percentile(latencies, 0.50)),
        "latency_p95_s": round_or_none(percentile(latencies, 0.95)),
        "max_no_progress_gap_p95_s": round_or_none(percentile(max_gaps, 0.95)),
        "starved_share": round_or_none(starved / submitted if submitted else None, 4),
        "ttft_sla_miss_share": round_or_none(ttft_sla_misses / submitted if submitted else None, 4),
        "latency_sla_miss_share": round_or_none(
            latency_sla_misses / submitted if submitted else None, 4
        ),
        "generated_tokens": generated_tokens,
        "service_tokens": service_tokens,
        "generated_tps": round_or_none(generated_tokens / wall_time_s if wall_time_s > 0 else None, 4),
        "service_tps": round_or_none(service_tokens / wall_time_s if wall_time_s > 0 else None, 4),
    }


def summarize_requests(
    *,
    requests: list[RequestState],
    wall_time_s: float,
    crowdout_by_tier: dict[int, TierCrowdoutStats],
) -> dict[str, Any]:
    by_tier: dict[str, list[RequestState]] = defaultdict(list)
    by_org: dict[str, list[RequestState]] = defaultdict(list)
    by_shape: dict[str, list[RequestState]] = defaultdict(list)
    by_phase: dict[str, list[RequestState]] = defaultdict(list)
    by_tier_org: dict[str, dict[str, list[RequestState]]] = defaultdict(lambda: defaultdict(list))
    for request in requests:
        tier_key = str(request.priority_tier)
        by_tier[tier_key].append(request)
        by_org[request.organization_id].append(request)
        by_shape[request.spec.shape_name].append(request)
        by_phase[request.spec.phase_name].append(request)
        by_tier_org[tier_key][request.organization_id].append(request)

    per_tier = {tier: summarize_group(group, wall_time_s=wall_time_s) for tier, group in sorted(by_tier.items(), key=lambda item: int(item[0]), reverse=True)}
    per_org = {org_id: summarize_group(group, wall_time_s=wall_time_s) for org_id, group in sorted(by_org.items())}
    for org_id, metrics in per_org.items():
        if by_org[org_id]:
            metrics["priority_tier"] = by_org[org_id][0].priority_tier

    per_shape = {shape_name: summarize_group(group, wall_time_s=wall_time_s) for shape_name, group in sorted(by_shape.items())}
    per_phase = {phase_name: summarize_group(group, wall_time_s=wall_time_s) for phase_name, group in sorted(by_phase.items())}

    p95_ttfts = [metrics["ttft_p95_s"] for metrics in per_org.values() if metrics["ttft_p95_s"] is not None]
    p95_latencies = [metrics["latency_p95_s"] for metrics in per_org.values() if metrics["latency_p95_s"] is not None]

    per_tier_fairness: dict[str, dict[str, float | None]] = {}
    for tier_key, tier_orgs in sorted(by_tier_org.items(), key=lambda item: int(item[0]), reverse=True):
        tier_ttfts = []
        tier_latencies = []
        for group_requests in tier_orgs.values():
            metrics = summarize_group(group_requests, wall_time_s=wall_time_s)
            if metrics["ttft_p95_s"] is not None:
                tier_ttfts.append(metrics["ttft_p95_s"])
            if metrics["latency_p95_s"] is not None:
                tier_latencies.append(metrics["latency_p95_s"])
        per_tier_fairness[tier_key] = {
            "ttft_p95_spread_s": round_or_none(max(tier_ttfts) - min(tier_ttfts) if len(tier_ttfts) >= 2 else None),
            "latency_p95_spread_s": round_or_none(max(tier_latencies) - min(tier_latencies) if len(tier_latencies) >= 2 else None),
        }

    total_service_tokens = sum(tier_stats.service_tokens for tier_stats in crowdout_by_tier.values())
    total_completed = sum(1 for request in requests if request.is_complete)
    crowdout_summary: dict[str, dict[str, float | int | None]] = {}
    for tier_key, group_requests in sorted(by_tier.items(), key=lambda item: int(item[0]), reverse=True):
        tier = int(tier_key)
        crowdout = crowdout_by_tier[tier]
        completed = sum(1 for request in group_requests if request.is_complete)
        starved = sum(
            1
            for request in group_requests
            if (request.max_no_progress_gap_s >= STARVATION_GAP_S) or not request.is_complete
        )
        crowdout_summary[tier_key] = {
            "backlog_ticks": crowdout.backlog_ticks,
            "service_ticks": crowdout.service_ticks,
            "blocked_by_higher_tier_ticks": crowdout.blocked_by_higher_tier_ticks,
            "blocked_ratio": round_or_none(
                crowdout.blocked_by_higher_tier_ticks / crowdout.backlog_ticks
                if crowdout.backlog_ticks > 0 else None,
                4,
            ),
            "service_tokens": crowdout.service_tokens,
            "service_share": round_or_none(
                crowdout.service_tokens / total_service_tokens if total_service_tokens > 0 else None, 4
            ),
            "completed_share": round_or_none(
                completed / total_completed if total_completed > 0 else None, 4
            ),
            "starved_share": round_or_none(starved / len(group_requests) if group_requests else None, 4),
            "leaked_service_tokens": crowdout.leaked_service_tokens,
        }

    return {
        "overall": summarize_group(requests, wall_time_s=wall_time_s),
        "per_tier": per_tier,
        "per_org": per_org,
        "per_shape": per_shape,
        "per_phase": per_phase,
        "fairness": {
            "ttft_p95_spread_s": round_or_none(max(p95_ttfts) - min(p95_ttfts) if len(p95_ttfts) >= 2 else None),
            "latency_p95_spread_s": round_or_none(max(p95_latencies) - min(p95_latencies) if len(p95_latencies) >= 2 else None),
            "per_tier": per_tier_fairness,
        },
        "crowdout": {
            "per_tier": crowdout_summary,
        },
    }


def policy_display_name(policy: str) -> str:
    return {
        "stock_default": "stock_default",
        "tier_aware_max_utilization": "tier_aware_max_utilization",
    }[policy]


def run_simulation(
    *,
    profile: TrafficProfile,
    policy: str,
    seed: int,
    total_requests: int,
    qps: float,
    tick_s: float,
    max_active_requests: int,
    max_scheduled_per_tick: int,
    ctx_chunk_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    specs = generate_requests(profile=profile, seed=seed, total_requests=total_requests, qps=qps)
    waiting: list[RequestState] = []
    active: list[RequestState] = []
    completed: list[RequestState] = []
    org_states: dict[str, OrganizationState] = defaultdict(OrganizationState)
    crowdout_by_tier: dict[int, TierCrowdoutStats] = defaultdict(TierCrowdoutStats)

    now_s = 0.0
    next_arrival_index = 0
    progress_step = max(1, total_requests // 4)
    next_progress_mark = progress_step
    preemption_count = 0
    preempted_victims_by_tier: Counter[int] = Counter()
    replay_tokens_due_to_preemption = 0

    while len(completed) < total_requests:
        while next_arrival_index < len(specs) and specs[next_arrival_index].arrival_time_s <= now_s:
            spec = specs[next_arrival_index]
            waiting.append(
                RequestState(
                    spec=spec,
                    remaining_prompt_tokens=spec.prompt_tokens,
                    remaining_decode_tokens=spec.max_tokens,
                    last_progress_time_s=spec.arrival_time_s,
                )
            )
            next_arrival_index += 1

        if policy == "tier_aware_max_utilization" and waiting and len(active) >= max_active_requests:
            while waiting and len(active) >= max_active_requests:
                same_tier_harm_by_tier = _same_tier_harm_by_tier(
                    waiting=waiting,
                    active=active,
                    now_s=now_s,
                )
                ordered_waiting = order_waiting_requests(
                    policy=policy,
                    waiting=waiting,
                    active=active,
                    now_s=now_s,
                    org_states=org_states,
                    same_tier_harm_by_tier=same_tier_harm_by_tier,
                )
                if not ordered_waiting:
                    break

                incoming = ordered_waiting[0]
                eligible_victims = []
                for request in active:
                    if request.priority_tier >= incoming.priority_tier:
                        continue
                    eligible_victims.append(request)
                if not eligible_victims:
                    break

                victim = min(
                    eligible_victims,
                    key=lambda request: (
                        request.priority_tier,
                        _score_active_request(request, org_states=org_states),
                        request.spec.arrival_time_s,
                        request.request_id,
                    ),
                )
                replay_prompt_tokens = victim.spec.prompt_tokens + victim.total_generated_tokens
                replay_tokens_due_to_preemption += replay_prompt_tokens
                victim.remaining_prompt_tokens = replay_prompt_tokens
                victim.pause_count += 1
                victim.scheduler_credit = min(
                    REQUEST_MAX_CREDIT,
                    victim.scheduler_credit + REQUEST_WAIT_CREDIT + REQUEST_PAUSE_BONUS,
                )
                active = [request for request in active if request.request_id != victim.request_id]
                waiting = [request for request in waiting if request.request_id != incoming.request_id]
                waiting.append(victim)
                if incoming.admitted_time_s is None:
                    incoming.admitted_time_s = now_s
                    org_state = org_states[incoming.organization_key]
                    org_state.token_balance = max(
                        ORG_TOKEN_BALANCE_MIN,
                        org_state.token_balance - _estimate_activation_charge(incoming),
                    )
                active.append(incoming)
                preemption_count += 1
                preempted_victims_by_tier[victim.priority_tier] += 1

        open_slots = max(0, max_active_requests - len(active))
        if open_slots > 0 and waiting:
            same_tier_harm_by_tier = _same_tier_harm_by_tier(
                waiting=waiting,
                active=active,
                now_s=now_s,
            )
            admitted = select_waiting_requests(
                policy=policy,
                waiting=waiting,
                active=active,
                now_s=now_s,
                org_states=org_states,
                max_new_requests=open_slots,
                max_active_requests=max_active_requests,
                max_service_requests=max_scheduled_per_tick,
                same_tier_harm_by_tier=same_tier_harm_by_tier,
            )
            admitted_ids = {request.request_id for request in admitted}
            waiting = [request for request in waiting if request.request_id not in admitted_ids]
            for request in admitted:
                if request.admitted_time_s is None:
                    request.admitted_time_s = now_s
                    org_state = org_states[request.organization_key]
                    org_state.token_balance = max(
                        ORG_TOKEN_BALANCE_MIN,
                        org_state.token_balance - _estimate_activation_charge(request),
                    )
                active.append(request)

        decay_org_states(org_states)

        same_tier_harm_by_tier = _same_tier_harm_by_tier(
            waiting=waiting,
            active=active,
            now_s=now_s,
        )
        active_order = order_active_requests(
            policy=policy,
            active=active,
            org_states=org_states,
            max_active_requests=max_active_requests,
            max_scheduled_per_tick=max_scheduled_per_tick,
            same_tier_harm_by_tier=same_tier_harm_by_tier,
        )
        scheduled = active_order[:max_scheduled_per_tick]
        scheduled_ids = {request.request_id for request in scheduled}

        tier_backlog: dict[int, bool] = {}
        for request in waiting + active:
            tier_backlog[request.priority_tier] = True
        scheduled_tiers = {request.priority_tier for request in scheduled}
        highest_scheduled_tier = max(scheduled_tiers) if scheduled_tiers else None
        highest_backlogged_tier = max(tier_backlog) if tier_backlog else None

        for tier in sorted(tier_backlog, reverse=True):
            tier_stats = crowdout_by_tier[tier]
            tier_stats.backlog_ticks += 1
            if tier in scheduled_tiers:
                tier_stats.service_ticks += 1
            elif highest_scheduled_tier is not None and highest_scheduled_tier > tier:
                tier_stats.blocked_by_higher_tier_ticks += 1

        if highest_backlogged_tier is not None:
            for request in scheduled:
                if request.priority_tier < highest_backlogged_tier:
                    crowdout_by_tier[request.priority_tier].leaked_service_tokens += 1

        active_orgs_by_tier: dict[int, set[str]] = defaultdict(set)
        for request in active:
            active_orgs_by_tier[request.priority_tier].add(request.organization_key)

        served_tokens_by_org: Counter[str] = Counter()
        served_tokens_by_tier: Counter[int] = Counter()
        for request in active:
            if request.request_id in scheduled_ids:
                served_tokens = service_request(
                    request,
                    now_s=now_s,
                    tick_s=tick_s,
                    ctx_chunk_tokens=ctx_chunk_tokens,
                )
                request.pause_count = max(0, request.pause_count - 1)
                request.scheduler_credit = max(0.0, request.scheduler_credit - REQUEST_SERVICE_COST)
                org_state = org_states[request.organization_key]
                org_state.token_balance = max(ORG_TOKEN_BALANCE_MIN, org_state.token_balance - served_tokens)
                org_state.recent_service_tokens += served_tokens
                org_state.total_service_tokens += served_tokens
                served_tokens_by_org[request.organization_key] += served_tokens
                served_tokens_by_tier[request.priority_tier] += served_tokens
                crowdout_by_tier[request.priority_tier].service_tokens += served_tokens
            else:
                request.scheduler_credit = min(REQUEST_MAX_CREDIT, request.scheduler_credit + REQUEST_WAIT_CREDIT)

        for tier, org_keys in active_orgs_by_tier.items():
            if not org_keys:
                continue
            tier_served_tokens = served_tokens_by_tier[tier]
            if tier_served_tokens <= 0:
                continue
            fair_service_grant = tier_served_tokens / len(org_keys)
            for org_key in org_keys:
                org_states[org_key].token_balance += fair_service_grant

        still_active: list[RequestState] = []
        for request in active:
            if request.is_complete:
                completed.append(request)
            else:
                still_active.append(request)
        active = still_active

        if stream and len(completed) >= next_progress_mark:
            summary = summarize_requests(
                requests=completed,
                wall_time_s=max(now_s, tick_s),
                crowdout_by_tier=crowdout_by_tier,
            )
            overall = summary["overall"]
            print(
                f"progress profile={profile.name} policy={policy_display_name(policy)} "
                f"completed={len(completed)}/{total_requests} elapsed_s={round(now_s, 2)} "
                f"ttft_p95_s={overall['ttft_p95_s']} latency_p95_s={overall['latency_p95_s']}"
            )
            next_progress_mark += progress_step

        now_s += tick_s

    finalize_request_gaps(completed, end_time_s=now_s)
    summary = summarize_requests(requests=completed, wall_time_s=now_s, crowdout_by_tier=crowdout_by_tier)
    return {
        "run": {
            "profile": profile.name,
            "policy": policy_display_name(policy),
            "seed": seed,
            "total_requests": total_requests,
            "qps": qps,
            "tick_s": tick_s,
            "max_active_requests": max_active_requests,
            "max_scheduled_per_tick": max_scheduled_per_tick,
            "ctx_chunk_tokens": ctx_chunk_tokens,
            "higher_numeric_tier_is_higher_priority": True,
            "preemption_count": preemption_count,
            "replay_tokens_due_to_preemption": replay_tokens_due_to_preemption,
            "preempted_victims_by_tier": {
                str(tier): count for tier, count in sorted(preempted_victims_by_tier.items(), reverse=True)
            },
        },
        "summary": summary,
    }


def compare_metric(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, 2)


def build_comparison(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "overall": {},
        "per_tier": {},
        "per_org": {},
    }
    overall_keys = (
        "completion_rate",
        "ttft_p95_s",
        "latency_p95_s",
        "admission_p95_s",
        "starved_share",
        "ttft_sla_miss_share",
        "latency_sla_miss_share",
        "generated_tps",
        "service_tps",
    )
    for key in overall_keys:
        comparison["overall"][key] = compare_metric(
            candidate["summary"]["overall"].get(key),
            baseline["summary"]["overall"].get(key),
        )

    tiers = sorted(
        set(baseline["summary"]["per_tier"]) | set(candidate["summary"]["per_tier"]),
        key=int,
        reverse=True,
    )
    for tier_key in tiers:
        baseline_tier = baseline["summary"]["per_tier"].get(tier_key, {})
        candidate_tier = candidate["summary"]["per_tier"].get(tier_key, {})
        baseline_crowdout = baseline["summary"]["crowdout"]["per_tier"].get(tier_key, {})
        candidate_crowdout = candidate["summary"]["crowdout"]["per_tier"].get(tier_key, {})
        comparison["per_tier"][tier_key] = {
            "completion_rate": compare_metric(
                candidate_tier.get("completion_rate"),
                baseline_tier.get("completion_rate"),
            ),
            "ttft_p95_s": compare_metric(candidate_tier.get("ttft_p95_s"), baseline_tier.get("ttft_p95_s")),
            "latency_p95_s": compare_metric(candidate_tier.get("latency_p95_s"), baseline_tier.get("latency_p95_s")),
            "ttft_sla_miss_share": compare_metric(
                candidate_tier.get("ttft_sla_miss_share"),
                baseline_tier.get("ttft_sla_miss_share"),
            ),
            "latency_sla_miss_share": compare_metric(
                candidate_tier.get("latency_sla_miss_share"),
                baseline_tier.get("latency_sla_miss_share"),
            ),
            "generated_tps": compare_metric(
                candidate_tier.get("generated_tps"),
                baseline_tier.get("generated_tps"),
            ),
            "service_tps": compare_metric(
                candidate_tier.get("service_tps"),
                baseline_tier.get("service_tps"),
            ),
            "starved_share": compare_metric(candidate_tier.get("starved_share"), baseline_tier.get("starved_share")),
            "blocked_ratio": compare_metric(
                candidate_crowdout.get("blocked_ratio"),
                baseline_crowdout.get("blocked_ratio"),
            ),
            "service_share": compare_metric(
                candidate_crowdout.get("service_share"),
                baseline_crowdout.get("service_share"),
            ),
        }

    organizations = sorted(
        set(baseline["summary"]["per_org"]) | set(candidate["summary"]["per_org"])
    )
    for organization_id in organizations:
        baseline_org = baseline["summary"]["per_org"].get(organization_id, {})
        candidate_org = candidate["summary"]["per_org"].get(organization_id, {})
        comparison["per_org"][organization_id] = {
            "priority_tier": candidate_org.get(
                "priority_tier",
                baseline_org.get("priority_tier"),
            ),
            "completion_rate": compare_metric(
                candidate_org.get("completion_rate"),
                baseline_org.get("completion_rate"),
            ),
            "ttft_p95_s": compare_metric(
                candidate_org.get("ttft_p95_s"),
                baseline_org.get("ttft_p95_s"),
            ),
            "latency_p95_s": compare_metric(
                candidate_org.get("latency_p95_s"),
                baseline_org.get("latency_p95_s"),
            ),
            "ttft_sla_miss_share": compare_metric(
                candidate_org.get("ttft_sla_miss_share"),
                baseline_org.get("ttft_sla_miss_share"),
            ),
            "latency_sla_miss_share": compare_metric(
                candidate_org.get("latency_sla_miss_share"),
                baseline_org.get("latency_sla_miss_share"),
            ),
            "generated_tps": compare_metric(
                candidate_org.get("generated_tps"),
                baseline_org.get("generated_tps"),
            ),
            "service_tps": compare_metric(
                candidate_org.get("service_tps"),
                baseline_org.get("service_tps"),
            ),
        }
    return comparison


def print_profile_summary(label: str, result: dict[str, Any]) -> None:
    overall = result["summary"]["overall"]
    run = result["run"]
    print(
        f"{label} overall "
        f"ttft_p95_s={overall['ttft_p95_s']} "
        f"latency_p95_s={overall['latency_p95_s']} "
        f"admission_p95_s={overall['admission_p95_s']} "
        f"ttft_sla_miss_share={overall['ttft_sla_miss_share']} "
        f"latency_sla_miss_share={overall['latency_sla_miss_share']} "
        f"generated_tps={overall['generated_tps']} "
        f"service_tps={overall['service_tps']} "
        f"completion_rate={overall['completion_rate']} "
        f"preemptions={run['preemption_count']} "
        f"replay_tokens={run['replay_tokens_due_to_preemption']}"
    )
    for tier_key, metrics in sorted(result["summary"]["per_tier"].items(), key=lambda item: int(item[0]), reverse=True):
        crowdout = result["summary"]["crowdout"]["per_tier"].get(tier_key, {})
        print(
            f"{label} tier={tier_key} "
            f"ttft_p95_s={metrics['ttft_p95_s']} "
            f"latency_p95_s={metrics['latency_p95_s']} "
            f"generated_tps={metrics['generated_tps']} "
            f"service_tps={metrics['service_tps']} "
            f"ttft_sla_miss_share={metrics['ttft_sla_miss_share']} "
            f"latency_sla_miss_share={metrics['latency_sla_miss_share']} "
            f"blocked_ratio={crowdout.get('blocked_ratio')} "
            f"service_share={crowdout.get('service_share')}"
        )


def render_markdown_report(matrix_results: dict[str, Any]) -> str:
    run = matrix_results["run"]
    lines = [
        "# Scheduler Simulation",
        "",
        "Higher numeric `priority_tier` means higher priority, matching the current TensorRT-LLM scheduler semantics.",
        "",
        f"Baseline policy: `{run['baseline_policy']}`",
        f"Candidate policy: `{run['candidate_policy']}`",
        "",
    ]
    for profile_name, profile_result in matrix_results["profiles"].items():
        baseline = profile_result["baseline"]
        candidate = profile_result["candidate"]
        comparison = profile_result["comparison"]
        lines.extend(
            [
                f"## {profile_name}",
                "",
                profile_result["description"],
                "",
                "| scope | baseline preemptions | candidate preemptions | baseline replay tokens | candidate replay tokens |",
                "| --- | ---: | ---: | ---: | ---: |",
                (
                    f"| run | {baseline['run']['preemption_count']} | {candidate['run']['preemption_count']} | "
                    f"{baseline['run']['replay_tokens_due_to_preemption']} | "
                    f"{candidate['run']['replay_tokens_due_to_preemption']} |"
                ),
                "",
                "| scope | baseline completion rate | candidate completion rate | delta | baseline ttft p95 s | candidate ttft p95 s | delta | baseline latency p95 s | candidate latency p95 s | delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                (
                    f"| overall | {baseline['summary']['overall']['completion_rate']} | "
                    f"{candidate['summary']['overall']['completion_rate']} | {comparison['overall']['completion_rate']} | "
                    f"{baseline['summary']['overall']['ttft_p95_s']} | "
                    f"{candidate['summary']['overall']['ttft_p95_s']} | {comparison['overall']['ttft_p95_s']} | "
                    f"{baseline['summary']['overall']['latency_p95_s']} | "
                    f"{candidate['summary']['overall']['latency_p95_s']} | {comparison['overall']['latency_p95_s']} |"
                ),
                "",
                "| scope | baseline generated tps | candidate generated tps | delta | baseline service tps | candidate service tps | delta | baseline ttft SLA miss | candidate ttft SLA miss | delta | baseline latency SLA miss | candidate latency SLA miss | delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                (
                    f"| overall | {baseline['summary']['overall']['generated_tps']} | "
                    f"{candidate['summary']['overall']['generated_tps']} | "
                    f"{comparison['overall']['generated_tps']} | "
                    f"{baseline['summary']['overall']['service_tps']} | "
                    f"{candidate['summary']['overall']['service_tps']} | "
                    f"{comparison['overall']['service_tps']} | "
                    f"{baseline['summary']['overall']['ttft_sla_miss_share']} | "
                    f"{candidate['summary']['overall']['ttft_sla_miss_share']} | "
                    f"{comparison['overall']['ttft_sla_miss_share']} | "
                    f"{baseline['summary']['overall']['latency_sla_miss_share']} | "
                    f"{candidate['summary']['overall']['latency_sla_miss_share']} | "
                    f"{comparison['overall']['latency_sla_miss_share']} |"
                ),
                "",
                "| tier | baseline ttft p95 s | candidate ttft p95 s | delta | baseline latency p95 s | candidate latency p95 s | delta | baseline generated tps | candidate generated tps | delta | baseline service tps | candidate service tps | delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for tier_key in sorted(
            set(baseline["summary"]["per_tier"]) | set(candidate["summary"]["per_tier"]),
            key=int,
            reverse=True,
        ):
            baseline_tier = baseline["summary"]["per_tier"].get(tier_key, {})
            candidate_tier = candidate["summary"]["per_tier"].get(tier_key, {})
            baseline_crowdout = baseline["summary"]["crowdout"]["per_tier"].get(tier_key, {})
            candidate_crowdout = candidate["summary"]["crowdout"]["per_tier"].get(tier_key, {})
            delta = comparison["per_tier"].get(tier_key, {})
            lines.append(
                f"| {tier_key} | {baseline_tier.get('ttft_p95_s')} | {candidate_tier.get('ttft_p95_s')} | "
                f"{delta.get('ttft_p95_s')} | {baseline_tier.get('latency_p95_s')} | "
                f"{candidate_tier.get('latency_p95_s')} | {delta.get('latency_p95_s')} | "
                f"{baseline_tier.get('generated_tps')} | {candidate_tier.get('generated_tps')} | "
                f"{delta.get('generated_tps')} | {baseline_tier.get('service_tps')} | "
                f"{candidate_tier.get('service_tps')} | {delta.get('service_tps')} |"
            )
        lines.extend(
            [
                "",
                "| tier | baseline ttft SLA miss | candidate ttft SLA miss | delta | baseline latency SLA miss | candidate latency SLA miss | delta | baseline blocked ratio | candidate blocked ratio | delta | baseline service share | candidate service share | delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for tier_key in sorted(
            set(baseline["summary"]["per_tier"]) | set(candidate["summary"]["per_tier"]),
            key=int,
            reverse=True,
        ):
            baseline_tier = baseline["summary"]["per_tier"].get(tier_key, {})
            candidate_tier = candidate["summary"]["per_tier"].get(tier_key, {})
            baseline_crowdout = baseline["summary"]["crowdout"]["per_tier"].get(tier_key, {})
            candidate_crowdout = candidate["summary"]["crowdout"]["per_tier"].get(tier_key, {})
            delta = comparison["per_tier"].get(tier_key, {})
            lines.append(
                f"| {tier_key} | {baseline_tier.get('ttft_sla_miss_share')} | "
                f"{candidate_tier.get('ttft_sla_miss_share')} | {delta.get('ttft_sla_miss_share')} | "
                f"{baseline_tier.get('latency_sla_miss_share')} | "
                f"{candidate_tier.get('latency_sla_miss_share')} | {delta.get('latency_sla_miss_share')} | "
                f"{baseline_crowdout.get('blocked_ratio')} | {candidate_crowdout.get('blocked_ratio')} | "
                f"{delta.get('blocked_ratio')} | {baseline_crowdout.get('service_share')} | "
                f"{candidate_crowdout.get('service_share')} | {delta.get('service_share')} |"
            )
        lines.extend(
            [
                "",
                "| org | tier | baseline ttft p95 s | candidate ttft p95 s | delta | baseline latency p95 s | candidate latency p95 s | delta | baseline generated tps | candidate generated tps | delta | baseline service tps | candidate service tps | delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for organization_id in sorted(
            set(baseline["summary"]["per_org"]) | set(candidate["summary"]["per_org"])
        ):
            baseline_org = baseline["summary"]["per_org"].get(organization_id, {})
            candidate_org = candidate["summary"]["per_org"].get(organization_id, {})
            delta = comparison["per_org"].get(organization_id, {})
            lines.append(
                f"| {organization_id} | {candidate_org.get('priority_tier', baseline_org.get('priority_tier'))} | "
                f"{baseline_org.get('ttft_p95_s')} | {candidate_org.get('ttft_p95_s')} | {delta.get('ttft_p95_s')} | "
                f"{baseline_org.get('latency_p95_s')} | {candidate_org.get('latency_p95_s')} | {delta.get('latency_p95_s')} | "
                f"{baseline_org.get('generated_tps')} | {candidate_org.get('generated_tps')} | {delta.get('generated_tps')} | "
                f"{baseline_org.get('service_tps')} | {candidate_org.get('service_tps')} | {delta.get('service_tps')} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a synthetic scheduler simulation matrix for noisy-neighbor and tier crowd-out analysis."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["premium_protection", "tier_mix_pressure", "same_tier_burst", "premium_noisy_neighbor"],
        help="Profiles to run. Use 'all' to run every built-in profile.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tick-s", type=float, default=0.05)
    parser.add_argument("--max-active-requests", type=int, default=16)
    parser.add_argument("--max-scheduled-per-tick", type=int, default=8)
    parser.add_argument("--ctx-chunk-tokens", type=int, default=128)
    parser.add_argument(
        "--request-scale",
        type=float,
        default=1.0,
        help="Scale each profile's default request count for heavier or lighter runs.",
    )
    parser.add_argument(
        "--qps-scale",
        type=float,
        default=1.0,
        help="Scale each profile's default arrival rate for heavier or lighter runs.",
    )
    parser.add_argument("--baseline-policy", default="stock_default", choices=["stock_default"])
    parser.add_argument(
        "--candidate-policy",
        default="tier_aware_max_utilization",
        choices=["tier_aware_max_utilization"],
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    profiles = make_profiles()
    selected_profiles = list(profiles) if args.profiles == ["all"] else args.profiles

    matrix_results: dict[str, Any] = {
        "run": {
            "seed": args.seed,
            "tick_s": args.tick_s,
            "max_active_requests": args.max_active_requests,
            "max_scheduled_per_tick": args.max_scheduled_per_tick,
            "ctx_chunk_tokens": args.ctx_chunk_tokens,
            "request_scale": args.request_scale,
            "qps_scale": args.qps_scale,
            "baseline_policy": policy_display_name(args.baseline_policy),
            "candidate_policy": policy_display_name(args.candidate_policy),
            "higher_numeric_tier_is_higher_priority": True,
        },
        "profiles": {},
    }

    print("simulation note: higher numeric priority_tier means higher priority.")
    print(
        f"simulation policies: baseline={policy_display_name(args.baseline_policy)} "
        f"candidate={policy_display_name(args.candidate_policy)}"
    )
    for profile_name in selected_profiles:
        profile = profiles[profile_name]
        total_requests = max(1, int(round(profile.default_total_requests * args.request_scale)))
        qps = max(0.05, profile.default_qps * args.qps_scale)
        print(
            f"profile name={profile.name} description={profile.description} "
            f"total_requests={total_requests} qps={qps}"
        )
        baseline = run_simulation(
            profile=profile,
            policy=args.baseline_policy,
            seed=args.seed,
            total_requests=total_requests,
            qps=qps,
            tick_s=args.tick_s,
            max_active_requests=args.max_active_requests,
            max_scheduled_per_tick=args.max_scheduled_per_tick,
            ctx_chunk_tokens=args.ctx_chunk_tokens,
            stream=not args.no_stream,
        )
        candidate = run_simulation(
            profile=profile,
            policy=args.candidate_policy,
            seed=args.seed,
            total_requests=total_requests,
            qps=qps,
            tick_s=args.tick_s,
            max_active_requests=args.max_active_requests,
            max_scheduled_per_tick=args.max_scheduled_per_tick,
            ctx_chunk_tokens=args.ctx_chunk_tokens,
            stream=not args.no_stream,
        )
        comparison = build_comparison(baseline=baseline, candidate=candidate)
        matrix_results["profiles"][profile_name] = {
            "description": profile.description,
            "total_requests": total_requests,
            "qps": qps,
            "baseline": baseline,
            "candidate": candidate,
            "comparison": comparison,
        }

        print_profile_summary("baseline", baseline)
        print_profile_summary("candidate", candidate)
        print(
            f"delta profile={profile_name} "
            f"overall_ttft_p95_s={comparison['overall']['ttft_p95_s']} "
            f"overall_latency_p95_s={comparison['overall']['latency_p95_s']} "
            f"overall_ttft_sla_miss_share={comparison['overall']['ttft_sla_miss_share']} "
            f"overall_latency_sla_miss_share={comparison['overall']['latency_sla_miss_share']}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(matrix_results, indent=2), encoding="utf-8")

    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown_report(matrix_results), encoding="utf-8")


if __name__ == "__main__":
    main()
T = TypeVar("T")
