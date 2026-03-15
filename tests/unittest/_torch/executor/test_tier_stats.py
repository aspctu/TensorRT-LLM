from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.tier_stats import TierRuntimeStatsCollector


def _make_generation_request(
    request_id: int,
    *,
    priority_tier: int,
    arrival_time: float,
    generated_tokens: int = 0,
    beam_width: int = 1,
):
    request = SimpleNamespace(
        request_id=request_id,
        py_priority_tier=priority_tier,
        arrival_time=arrival_time,
        orig_prompt_len=32,
        max_beam_num_tokens=32 + generated_tokens,
        num_draft_tokens=0,
        is_encoder_init_state=False,
        is_context_init_state=False,
        is_disagg_generation_init_state=False,
    )
    request.get_beam_width_by_iter = lambda for_next_iteration=False: beam_width
    return request


def _make_context_request(
    request_id: int,
    *,
    priority_tier: int,
    arrival_time: float,
    context_chunk_size: int,
):
    return SimpleNamespace(
        request_id=request_id,
        py_priority_tier=priority_tier,
        arrival_time=arrival_time,
        orig_prompt_len=context_chunk_size,
        max_beam_num_tokens=context_chunk_size,
        num_draft_tokens=0,
        is_encoder_init_state=False,
        is_context_init_state=True,
        is_disagg_generation_init_state=False,
        is_last_context_chunk=True,
        has_draft_tokens=False,
        context_chunk_size=context_chunk_size,
        context_remaining_length=context_chunk_size,
    )


def test_tier_runtime_stats_collects_latency_and_throughput():
    clock = [100.0]
    collector = TierRuntimeStatsCollector(
        time_fn=lambda: clock[0],
        throughput_window_seconds=60.0,
    )

    generation_request = _make_generation_request(
        7, priority_tier=2, arrival_time=clock[0]
    )
    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [generation_request]

    collector.mark_requests_scheduled([generation_request], now=clock[0] + 0.25)
    collector.mark_first_token(generation_request, now=clock[0] + 1.0)
    generation_request.max_beam_num_tokens = generation_request.orig_prompt_len + 16
    collector.mark_request_completed(generation_request, now=clock[0] + 4.0)

    clock[0] += 4.0
    collector.observe_iteration(
        active_requests=[],
        finished_requests=[generation_request],
        scheduled_batch=scheduled_batch,
        iter_latency_ms=500.0,
        now=clock[0],
    )

    stats = collector.build_tier_stats(
        active_requests=[],
        queued_requests=[],
        waiting_requests=[],
        scheduled_context_requests=[],
        scheduled_generation_requests=[],
        paused_requests=[],
        now=clock[0],
    )

    tier_stats = stats["2"]
    assert tier_stats["completedSamples"] == 1
    assert tier_stats["ttftMsP50"] == pytest.approx(1000.0)
    assert tier_stats["ttftMsP95"] == pytest.approx(1000.0)
    assert tier_stats["latencyMsP95"] == pytest.approx(4000.0)
    assert tier_stats["generatedTokensPerSecond"] == pytest.approx(32.0)
    assert tier_stats["serviceTokensPerSecond"] == pytest.approx(2.0)


def test_tier_runtime_stats_include_live_counts():
    clock = [50.0]
    collector = TierRuntimeStatsCollector(time_fn=lambda: clock[0])

    active = _make_generation_request(1, priority_tier=3, arrival_time=clock[0])
    queued = _make_generation_request(2, priority_tier=1, arrival_time=clock[0])
    waiting = _make_generation_request(3, priority_tier=1, arrival_time=clock[0])
    scheduled_context = _make_context_request(
        4, priority_tier=3, arrival_time=clock[0], context_chunk_size=64
    )
    paused = _make_generation_request(5, priority_tier=1, arrival_time=clock[0])

    stats = collector.build_tier_stats(
        active_requests=[active],
        queued_requests=[queued],
        waiting_requests=[waiting],
        scheduled_context_requests=[scheduled_context],
        scheduled_generation_requests=[],
        paused_requests=[paused],
        now=clock[0],
    )

    assert stats["3"]["active"] == 1
    assert stats["3"]["scheduledContext"] == 1
    assert stats["1"]["queued"] == 1
    assert stats["1"]["waiting"] == 1
    assert stats["1"]["paused"] == 1
