from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import (
    CapacityScheduler,
    MicroBatchScheduler,
    PrioritySimpleScheduler,
)


class _StubCapacityScheduler(CapacityScheduler):

    def __init__(self):
        self.last_active_requests = None

    def schedule_request(self, active_requests):
        self.last_active_requests = list(active_requests)
        return active_requests, [], []


class _StubMicroBatchScheduler(MicroBatchScheduler):

    def schedule(self, active_requests, inflight_request_ids):
        return active_requests, []


def _make_request(request_id, priority=None):
    req = SimpleNamespace()
    if priority is not None:
        req.priority = lambda p=priority: p
    req.state = LlmRequestState.CONTEXT_INIT
    req.mRequestId = request_id
    return req


def test_priority_simple_scheduler_orders_active_requests():
    high = _make_request(1, 0.9)
    medium = _make_request(2, 0.6)
    low = _make_request(3, 0.2)

    capacity = _StubCapacityScheduler()
    micro = _StubMicroBatchScheduler()
    scheduler = PrioritySimpleScheduler(capacity, micro)

    active_requests = [low, medium, high]
    output = scheduler.schedule_request(active_requests, set())

    # Ensure capacity scheduler saw the requests reordered by priority
    ordered_ids = [req.mRequestId for req in capacity.last_active_requests]
    assert ordered_ids == [high.mRequestId, medium.mRequestId, low.mRequestId]

    # Ensure outputs preserve the priority ordering
    context_ids = [req.mRequestId for req in output.context_requests]
    assert context_ids == ordered_ids


def test_priority_simple_scheduler_defaults_priority_when_missing():
    high = _make_request(1, 0.9)
    neutral = _make_request(2)  # No explicit priority set.
    low = _make_request(3, 0.1)

    scheduler = PrioritySimpleScheduler(_StubCapacityScheduler(),
                                        _StubMicroBatchScheduler())

    ordered = scheduler.schedule_request([neutral, high, low],
                                         set()).context_requests
    ordered_ids = [req.mRequestId for req in ordered]

    # Expect high priority first, default (0.5) next, and low priority last.
    assert ordered_ids == [high.mRequestId, neutral.mRequestId, low.mRequestId]
