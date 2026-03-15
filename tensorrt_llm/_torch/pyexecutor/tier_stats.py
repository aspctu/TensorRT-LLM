from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Iterable, Optional

from .request_metadata import request_priority_tier, scheduled_token_cost
from .scheduler import ScheduledRequests

LATENCY_SAMPLE_LIMIT = 256
THROUGHPUT_WINDOW_SECONDS = 30.0


@dataclass(slots=True)
class _LatencySample:
    ttft_ms: float
    latency_ms: float


@dataclass(slots=True)
class _ThroughputSample:
    timestamp: float
    duration_s: float
    generated_tokens: int
    service_tokens: int


def _request_id(request: object) -> Optional[int]:
    value = getattr(request, "request_id", getattr(request, "py_request_id", None))
    if value is None:
        return None
    return int(value)


def _request_arrival_time(request: object) -> Optional[float]:
    for attr in ("arrival_time", "py_scheduler_enqueue_time", "py_arrival_time"):
        value = getattr(request, attr, None)
        if value is not None:
            return float(value)
    return None


def _request_first_scheduled_time(request: object) -> Optional[float]:
    value = getattr(request, "py_first_scheduled_time", None)
    if value is None:
        return None
    return float(value)


def _request_first_token_time(request: object) -> Optional[float]:
    value = getattr(request, "py_first_token_time", None)
    if value is None:
        return None
    return float(value)


def _request_last_token_time(request: object) -> Optional[float]:
    value = getattr(request, "py_last_token_time", None)
    if value is None:
        return None
    return float(value)


def _request_num_generated_tokens(request: object) -> int:
    for attr in ("num_generated_tokens", "max_beam_num_tokens"):
        value = getattr(request, attr, None)
        if value is not None:
            generated_tokens = int(value)
            if attr == "max_beam_num_tokens":
                generated_tokens -= int(getattr(request, "orig_prompt_len", 0))
            return max(0, generated_tokens)
    return 0


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    if not values:
        return None

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * percentile
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = rank - lower_index
    return lower_value + (upper_value - lower_value) * weight


class TierRuntimeStatsCollector:
    """Rolling per-tier latency and throughput stats for iteration JSON."""

    def __init__(
        self,
        *,
        time_fn: Callable[[], float],
        latency_sample_limit: int = LATENCY_SAMPLE_LIMIT,
        throughput_window_seconds: float = THROUGHPUT_WINDOW_SECONDS,
    ) -> None:
        self._time_fn = time_fn
        self._latency_sample_limit = latency_sample_limit
        self._throughput_window_seconds = throughput_window_seconds
        self._latency_samples_by_tier: dict[int, Deque[_LatencySample]] = defaultdict(deque)
        self._throughput_samples_by_tier: dict[int, Deque[_ThroughputSample]] = defaultdict(deque)
        self._last_generated_tokens_by_request: dict[int, int] = {}

    def now(self) -> float:
        return float(self._time_fn())

    def mark_requests_scheduled(
        self, requests: Iterable[object], *, now: Optional[float] = None
    ) -> None:
        timestamp = self.now() if now is None else float(now)
        for request in requests:
            if getattr(request, "py_first_scheduled_time", None) is None:
                request.py_first_scheduled_time = timestamp

    def mark_first_token(self, request: object, *, now: Optional[float] = None) -> None:
        if getattr(request, "py_first_token_time", None) is not None:
            return
        request.py_first_token_time = self.now() if now is None else float(now)

    def mark_request_completed(
        self, request: object, *, now: Optional[float] = None
    ) -> None:
        if getattr(request, "py_last_token_time", None) is None:
            request.py_last_token_time = self.now() if now is None else float(now)

    def observe_iteration(
        self,
        *,
        active_requests: Iterable[object],
        finished_requests: Iterable[object],
        scheduled_batch: ScheduledRequests,
        iter_latency_ms: float,
        now: Optional[float] = None,
    ) -> None:
        timestamp = self.now() if now is None else float(now)
        duration_s = max(float(iter_latency_ms) / 1000.0, 1e-6)

        service_tokens_by_tier: Counter[int] = Counter()
        for request in list(scheduled_batch.context_requests) + list(
            scheduled_batch.generation_requests
        ):
            service_tokens_by_tier[request_priority_tier(request)] += scheduled_token_cost(
                request
            )

        generated_tokens_by_tier: Counter[int] = Counter()
        seen_request_ids: set[int] = set()
        for request in list(active_requests) + list(finished_requests):
            request_id = _request_id(request)
            if request_id is None or request_id in seen_request_ids:
                continue
            seen_request_ids.add(request_id)
            generated_tokens = _request_num_generated_tokens(request)
            previous_generated_tokens = self._last_generated_tokens_by_request.get(
                request_id, 0
            )
            generated_delta = max(0, generated_tokens - previous_generated_tokens)
            if generated_delta > 0:
                generated_tokens_by_tier[request_priority_tier(request)] += generated_delta
            self._last_generated_tokens_by_request[request_id] = generated_tokens

        touched_tiers = set(service_tokens_by_tier) | set(generated_tokens_by_tier)
        for priority_tier in touched_tiers:
            self._throughput_samples_by_tier[priority_tier].append(
                _ThroughputSample(
                    timestamp=timestamp,
                    duration_s=duration_s,
                    generated_tokens=generated_tokens_by_tier.get(priority_tier, 0),
                    service_tokens=service_tokens_by_tier.get(priority_tier, 0),
                )
            )
            self._prune_throughput_samples(priority_tier, now=timestamp)

        for request in finished_requests:
            self._record_completion_latency(request)
            request_id = _request_id(request)
            if request_id is not None:
                self._last_generated_tokens_by_request.pop(request_id, None)

    def _prune_throughput_samples(self, priority_tier: int, *, now: float) -> None:
        cutoff = now - self._throughput_window_seconds
        samples = self._throughput_samples_by_tier.get(priority_tier)
        if samples is None:
            return
        while samples and samples[0].timestamp < cutoff:
            samples.popleft()

    def _record_completion_latency(self, request: object) -> None:
        arrival_time = _request_arrival_time(request)
        first_token_time = _request_first_token_time(request)
        last_token_time = _request_last_token_time(request)
        if arrival_time is None or first_token_time is None or last_token_time is None:
            return

        priority_tier = request_priority_tier(request)
        samples = self._latency_samples_by_tier[priority_tier]
        samples.append(
            _LatencySample(
                ttft_ms=max(0.0, (first_token_time - arrival_time) * 1000.0),
                latency_ms=max(0.0, (last_token_time - arrival_time) * 1000.0),
            )
        )
        while len(samples) > self._latency_sample_limit:
            samples.popleft()

    def build_tier_stats(
        self,
        *,
        active_requests: Iterable[object],
        queued_requests: Iterable[object],
        waiting_requests: Iterable[object],
        scheduled_context_requests: Iterable[object],
        scheduled_generation_requests: Iterable[object],
        paused_requests: Iterable[object],
        now: Optional[float] = None,
    ) -> dict[str, dict[str, int | float | None]]:
        timestamp = self.now() if now is None else float(now)

        active_counts = Counter(request_priority_tier(request) for request in active_requests)
        queued_counts = Counter(request_priority_tier(request) for request in queued_requests)
        waiting_counts = Counter(request_priority_tier(request) for request in waiting_requests)
        scheduled_context_counts = Counter(
            request_priority_tier(request) for request in scheduled_context_requests
        )
        scheduled_generation_counts = Counter(
            request_priority_tier(request) for request in scheduled_generation_requests
        )
        paused_counts = Counter(request_priority_tier(request) for request in paused_requests)

        all_tiers = set(active_counts)
        all_tiers.update(queued_counts)
        all_tiers.update(waiting_counts)
        all_tiers.update(scheduled_context_counts)
        all_tiers.update(scheduled_generation_counts)
        all_tiers.update(paused_counts)
        all_tiers.update(self._latency_samples_by_tier)
        all_tiers.update(self._throughput_samples_by_tier)

        stats: dict[str, dict[str, int | float | None]] = {}
        for priority_tier in sorted(all_tiers, reverse=True):
            self._prune_throughput_samples(priority_tier, now=timestamp)
            throughput_samples = self._throughput_samples_by_tier.get(priority_tier, ())
            total_duration_s = sum(sample.duration_s for sample in throughput_samples)
            generated_tps = (
                sum(sample.generated_tokens for sample in throughput_samples) / total_duration_s
                if total_duration_s > 0.0
                else 0.0
            )
            service_tps = (
                sum(sample.service_tokens for sample in throughput_samples) / total_duration_s
                if total_duration_s > 0.0
                else 0.0
            )

            latency_samples = self._latency_samples_by_tier.get(priority_tier, ())
            ttft_samples = [sample.ttft_ms for sample in latency_samples]
            latency_values = [sample.latency_ms for sample in latency_samples]

            stats[str(priority_tier)] = {
                "active": active_counts.get(priority_tier, 0),
                "queued": queued_counts.get(priority_tier, 0),
                "waiting": waiting_counts.get(priority_tier, 0),
                "scheduled": scheduled_context_counts.get(priority_tier, 0)
                + scheduled_generation_counts.get(priority_tier, 0),
                "scheduledContext": scheduled_context_counts.get(priority_tier, 0),
                "scheduledGeneration": scheduled_generation_counts.get(priority_tier, 0),
                "paused": paused_counts.get(priority_tier, 0),
                "completedSamples": len(latency_values),
                "ttftMsP50": _percentile(ttft_samples, 0.50),
                "ttftMsP95": _percentile(ttft_samples, 0.95),
                "latencyMsP50": _percentile(latency_values, 0.50),
                "latencyMsP95": _percentile(latency_values, 0.95),
                "generatedTokensPerSecond": generated_tps,
                "serviceTokensPerSecond": service_tps,
            }

        return stats
