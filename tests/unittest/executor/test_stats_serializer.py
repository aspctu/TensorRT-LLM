import json

from tensorrt_llm._torch.pyexecutor.request_metadata import RequestStatsExtra
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.base_worker import BaseWorker


def test_stats_serializer_includes_priority_metrics():
    iteration_stats = tllm.IterationStats()
    iteration_stats.iter = 7

    request_stats = tllm.RequestStats()
    request_stats.id = 42

    serialized = BaseWorker._stats_serializer(
        (
            iteration_stats,
            [request_stats],
            {
                "priorityStats": {
                    "scheduled": {"2": 1},
                    "queued": {"0": 1},
                },
                "tierStats": {
                    "2": {
                        "active": 1,
                        "completedSamples": 4,
                        "ttftMsP95": 1250.0,
                        "generatedTokensPerSecond": 32.0,
                        "serviceTokensPerSecond": 64.0,
                    }
                },
                "organizationStats": {
                    "active": {"2:acme": 1},
                    "queued": {"0:default": 1},
                },
                "requestStatsExtra": {
                    42: RequestStatsExtra(
                        priorityTier=2,
                        priorityCredit=3.0,
                        priorityPauseCount=1,
                        organizationId="acme",
                        schedulerScore=12.0,
                        schedulerAgeCredit=0.0,
                        schedulerOrgTokenBalance=-48.0,
                    )
                },
            },
        )
    )

    stats = json.loads(serialized)
    assert stats["priorityStats"]["scheduled"] == {"2": 1}
    assert stats["priorityStats"]["queued"] == {"0": 1}
    assert stats["tierStats"]["2"]["ttftMsP95"] == 1250.0
    assert stats["tierStats"]["2"]["generatedTokensPerSecond"] == 32.0
    assert stats["organizationStats"]["active"] == {"2:acme": 1}
    assert stats["requestStats"][0]["id"] == 42
    assert stats["requestStats"][0]["priorityTier"] == 2
    assert stats["requestStats"][0]["priorityCredit"] == 3.0
    assert stats["requestStats"][0]["priorityPauseCount"] == 1
    assert stats["requestStats"][0]["organizationId"] == "acme"
    assert stats["requestStats"][0]["schedulerScore"] == 12.0
    assert stats["requestStats"][0]["schedulerOrgTokenBalance"] == -48.0
