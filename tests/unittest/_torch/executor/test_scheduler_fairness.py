from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.scheduler_fairness import SchedulerFairnessController
from tensorrt_llm.scheduling_params import SchedulingParams


def _make_waiting_item(
    request_id: int,
    *,
    priority_tier: int = 0,
    organization_id: str = "default",
    prompt_tokens: int = 128,
    max_tokens: int = 32,
) -> RequestQueueItem:
    request = SimpleNamespace(
        py_scheduling_params=SchedulingParams(
            priority_tier=priority_tier,
            organization_id=organization_id,
        ),
        input_token_ids=[0] * prompt_tokens,
        max_tokens=max_tokens,
    )
    return RequestQueueItem(request_id, request)


def _make_scheduled_context_request(
    request_id: int,
    *,
    priority_tier: int = 0,
    organization_id: str = "default",
    context_chunk_size: int = 32,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        py_priority_tier=priority_tier,
        py_organization_id=organization_id,
        is_encoder_init_state=False,
        is_context_init_state=True,
        is_disagg_generation_init_state=False,
        is_last_context_chunk=True,
        has_draft_tokens=False,
        num_draft_tokens=0,
        context_chunk_size=context_chunk_size,
        context_remaining_length=context_chunk_size,
    )


class TestSchedulerFairnessController:

    def test_higher_tier_wins_waiting_queue(self):
        clock = [100.0]
        fairness = SchedulerFairnessController(time_fn=lambda: clock[0])

        low = _make_waiting_item(1, priority_tier=0, organization_id="org-a")
        fairness.on_requests_enqueued([low])

        clock[0] += 4.0
        high = _make_waiting_item(2, priority_tier=1, organization_id="org-a")
        fairness.on_requests_enqueued([high])

        assert fairness.score_waiting_request(high) > fairness.score_waiting_request(low)

    def test_higher_tier_still_wins_after_low_tier_ages_and_other_org_is_served(self):
        clock = [0.0]
        fairness = SchedulerFairnessController(time_fn=lambda: clock[0])

        served = _make_scheduled_context_request(
            100, priority_tier=1, organization_id="org-high", context_chunk_size=128
        )
        fairness.on_requests_scheduled([served], [])

        low = _make_waiting_item(1, priority_tier=0, organization_id="org-low")
        fairness.on_requests_enqueued([low])
        clock[0] += 20.0

        high = _make_waiting_item(2, priority_tier=1, organization_id="org-high")
        fairness.on_requests_enqueued([high])

        assert fairness.score_waiting_request(high) > fairness.score_waiting_request(low)

    def test_waiting_age_breaks_ties_within_org_and_tier(self):
        clock = [10.0]
        fairness = SchedulerFairnessController(time_fn=lambda: clock[0])

        first = _make_waiting_item(1, priority_tier=1, organization_id="org-a")
        fairness.on_requests_enqueued([first])

        clock[0] += 2.0
        second = _make_waiting_item(2, priority_tier=1, organization_id="org-a")
        fairness.on_requests_enqueued([second])

        assert fairness.score_waiting_request(first) > fairness.score_waiting_request(second)

    def test_org_token_balance_prefers_less_served_org_within_tier(self):
        clock = [0.0]
        fairness = SchedulerFairnessController(time_fn=lambda: clock[0])

        served = _make_scheduled_context_request(
            100, priority_tier=1, organization_id="org-a", context_chunk_size=64
        )
        fairness.on_requests_scheduled([served], [])

        org_a = _make_waiting_item(1, priority_tier=1, organization_id="org-a")
        org_b = _make_waiting_item(2, priority_tier=1, organization_id="org-b")
        fairness.on_requests_enqueued([org_a, org_b])

        assert fairness.score_waiting_request(org_b) > fairness.score_waiting_request(org_a)

    def test_waiting_score_penalizes_large_requests_within_tier(self):
        fairness = SchedulerFairnessController(time_fn=lambda: 0.0)

        short = _make_waiting_item(
            1,
            priority_tier=1,
            organization_id="org-a",
            prompt_tokens=256,
            max_tokens=32,
        )
        long = _make_waiting_item(
            2,
            priority_tier=1,
            organization_id="org-b",
            prompt_tokens=1536,
            max_tokens=320,
        )
        fairness.on_requests_enqueued([short, long])

        assert fairness.score_waiting_request(short) > fairness.score_waiting_request(long)

    def test_waiting_order_prefers_org_with_fewer_active_requests_within_tier(self):
        fairness = SchedulerFairnessController(time_fn=lambda: 0.0)

        active_requests = [
            _make_scheduled_context_request(101, priority_tier=2, organization_id="org-hot"),
            _make_scheduled_context_request(102, priority_tier=2, organization_id="org-hot"),
        ]
        hot_waiting = _make_waiting_item(1, priority_tier=2, organization_id="org-hot")
        cool_waiting = _make_waiting_item(2, priority_tier=2, organization_id="org-cool")
        fairness.on_requests_enqueued([hot_waiting, cool_waiting])

        ordered = fairness.order_waiting_requests(
            [hot_waiting, cool_waiting],
            active_requests=active_requests,
        )

        assert [item.id for item in ordered] == [2, 1]

    def test_waiting_order_keeps_higher_tier_ahead_of_less_loaded_lower_tier(self):
        fairness = SchedulerFairnessController(time_fn=lambda: 0.0)

        active_requests = [
            _make_scheduled_context_request(101, priority_tier=3, organization_id="org-hot"),
            _make_scheduled_context_request(102, priority_tier=3, organization_id="org-hot"),
        ]
        high = _make_waiting_item(1, priority_tier=3, organization_id="org-hot")
        low = _make_waiting_item(2, priority_tier=2, organization_id="org-cool")
        fairness.on_requests_enqueued([high, low])

        ordered = fairness.order_waiting_requests(
            [high, low],
            active_requests=active_requests,
        )

        assert [item.id for item in ordered] == [1, 2]

    def test_select_waiting_requests_reserves_capacity_for_waiting_top_tier(self):
        fairness = SchedulerFairnessController(time_fn=lambda: 0.0)

        top = _make_waiting_item(100, priority_tier=3, organization_id="top")
        low = _make_waiting_item(1, priority_tier=1, organization_id="org-a")
        fairness.on_requests_enqueued([top, low])

        active_requests = [
            _make_scheduled_context_request(200 + idx, priority_tier=1, organization_id=f"low-{idx}")
            for idx in range(7)
        ]

        selected = fairness.select_waiting_requests(
            [low, top],
            active_requests=active_requests,
            max_new_requests=1,
            max_active_requests=8,
        )

        assert [item.id for item in selected] == [100]

    def test_select_waiting_requests_allows_lower_tier_to_borrow_without_higher_tier_backlog(self):
        fairness = SchedulerFairnessController(time_fn=lambda: 0.0)

        low = _make_waiting_item(2, priority_tier=1, organization_id="low")
        fairness.on_requests_enqueued([low])

        active_requests = [
            _make_scheduled_context_request(200 + idx, priority_tier=1, organization_id=f"low-{idx}")
            for idx in range(7)
        ]

        selected = fairness.select_waiting_requests(
            [low],
            active_requests=active_requests,
            max_new_requests=1,
            max_active_requests=8,
        )

        assert [item.id for item in selected] == [2]

    def test_select_waiting_requests_caps_hot_org_when_peer_waits_in_same_tier(self):
        fairness = SchedulerFairnessController(time_fn=lambda: 0.0)

        hot = _make_waiting_item(1, priority_tier=2, organization_id="org-hot")
        cool = _make_waiting_item(2, priority_tier=2, organization_id="org-cool")
        fairness.on_requests_enqueued([hot, cool])

        active_requests = [
            _make_scheduled_context_request(100 + idx, priority_tier=2, organization_id="org-hot")
            for idx in range(4)
        ]

        selected = fairness.select_waiting_requests(
            [hot, cool],
            active_requests=active_requests,
            max_new_requests=1,
            max_active_requests=8,
        )

        assert [item.id for item in selected] == [2]

    def test_select_waiting_requests_requires_peer_orgs_before_expanding_hot_org(self):
        fairness = SchedulerFairnessController(time_fn=lambda: 0.0)

        hot_a = _make_waiting_item(1, priority_tier=2, organization_id="org-hot")
        hot_b = _make_waiting_item(2, priority_tier=2, organization_id="org-hot")
        cool_a = _make_waiting_item(3, priority_tier=2, organization_id="org-cool-a")
        cool_b = _make_waiting_item(4, priority_tier=2, organization_id="org-cool-b")
        fairness.on_requests_enqueued([hot_a, hot_b, cool_a, cool_b])

        active_requests = [
            _make_scheduled_context_request(100 + idx, priority_tier=2, organization_id="org-hot")
            for idx in range(2)
        ]

        selected = fairness.select_waiting_requests(
            [hot_a, hot_b, cool_a, cool_b],
            active_requests=active_requests,
            max_new_requests=3,
            max_active_requests=8,
            max_service_requests=8,
        )

        assert [item.id for item in selected] == [3, 4, 1]

    def test_select_waiting_requests_single_tier_uses_active_capacity_for_org_cap(self):
        fairness = SchedulerFairnessController(time_fn=lambda: 0.0)

        hot = _make_waiting_item(1, priority_tier=2, organization_id="org-hot")
        cool = _make_waiting_item(2, priority_tier=2, organization_id="org-cool")
        fairness.on_requests_enqueued([hot, cool])

        active_requests = [
            _make_scheduled_context_request(100 + idx, priority_tier=2, organization_id="org-hot")
            for idx in range(3)
        ]

        selected = fairness.select_waiting_requests(
            [hot, cool],
            active_requests=active_requests,
            max_new_requests=2,
            max_active_requests=8,
            max_service_requests=4,
        )

        assert [item.id for item in selected] == [2, 1]

    def test_on_requests_activated_charges_large_request_org_balance(self):
        fairness = SchedulerFairnessController(time_fn=lambda: 0.0)

        small = SimpleNamespace(
            request_id=1,
            py_priority_tier=2,
            py_organization_id="org-small",
            py_orig_prompt_len=256,
            py_max_new_tokens=32,
            state_value=1,
        )
        large = SimpleNamespace(
            request_id=2,
            py_priority_tier=2,
            py_organization_id="org-large",
            py_orig_prompt_len=1536,
            py_max_new_tokens=320,
            state_value=1,
        )

        fairness.on_requests_activated([small, large])

        stats = fairness.build_organization_stats(
            active_requests=[small, large],
            queued_requests=[],
            waiting_requests=[],
            scheduled_context_requests=[],
            scheduled_generation_requests=[],
            paused_requests=[],
        )
        assert stats["2:org-large"]["orgTokenBalance"] < stats["2:org-small"]["orgTokenBalance"]

    def test_build_organization_stats_reports_token_metrics(self):
        fairness = SchedulerFairnessController(time_fn=lambda: 0.0)

        queued = _make_waiting_item(1, priority_tier=2, organization_id="acme")
        fairness.on_requests_enqueued([queued])
        fairness.on_requests_scheduled(
            [_make_scheduled_context_request(
                2, priority_tier=2, organization_id="acme", context_chunk_size=48
            )],
            [],
        )

        stats = fairness.build_organization_stats(
            active_requests=[],
            queued_requests=[queued.request],
            waiting_requests=[],
            scheduled_context_requests=[],
            scheduled_generation_requests=[],
            paused_requests=[],
        )

        assert stats["2:acme"]["queued"] == 1
        assert stats["2:acme"]["priorityTier"] == 2
        assert stats["2:acme"]["organizationId"] == "acme"
        assert stats["2:acme"]["totalEnqueued"] == 1
        assert stats["2:acme"]["totalScheduledTokens"] == 48.0
        assert stats["2:acme"]["orgTokenBalance"] < 0.0

    def test_on_requests_enqueued_rejects_missing_normal_request(self):
        fairness = SchedulerFairnessController(time_fn=lambda: 0.0)
        broken_item = SimpleNamespace(is_normal_request=True, request=None)

        try:
            fairness.on_requests_enqueued([broken_item])
        except ValueError as exc:
            assert "missing request payload" in str(exc)
        else:
            raise AssertionError("expected ValueError for missing request payload")
