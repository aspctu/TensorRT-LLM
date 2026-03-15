## PyTorch Scheduler Notes

This branch adds a request-side `priority_tier` knob and a C++-side
`TIER_AWARE_MAX_UTILIZATION` scheduler policy for the default PyTorch backend.

### Runtime Path

- Request metadata enters through [scheduling_params.py](/workspace/work/TensorRT-LLM/tensorrt_llm/scheduling_params.py).
- PyTorch request conversion copies `priority_tier` and `organization_id` onto
  the bound request in [llm_request.py](/workspace/work/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/llm_request.py).
- The real active-request scheduler is the C++ capacity scheduler in
  [capacityScheduler.cpp](/workspace/work/TensorRT-LLM/cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp).

Use it with:

```python
from tensorrt_llm.llmapi import CapacitySchedulerPolicy, SchedulerConfig
from tensorrt_llm.scheduling_params import SchedulingParams

scheduler_config = SchedulerConfig(
    capacity_scheduler_policy=CapacitySchedulerPolicy.TIER_AWARE_MAX_UTILIZATION
)

scheduling_params = SchedulingParams(priority_tier=4, organization_id="org-a")
```

Higher numeric `priority_tier` means higher priority.

### Per-Tier Metrics

Per-tier iteration metrics are emitted in the normal worker stats payload under
`tierStats` when `enable_iter_perf_stats=True`.

Each tier entry currently includes:

- `active`
- `queued`
- `waiting`
- `scheduled`
- `scheduledContext`
- `scheduledGeneration`
- `paused`
- `completedSamples`
- `ttftMsP50`
- `ttftMsP95`
- `latencyMsP50`
- `latencyMsP95`
- `generatedTokensPerSecond`
- `serviceTokensPerSecond`

The collector lives in [tier_stats.py](/workspace/work/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/tier_stats.py)
and is wired from [py_executor.py](/workspace/work/TensorRT-LLM/tensorrt_llm/_torch/pyexecutor/py_executor.py).

### Running The Local Simulator

The synthetic scheduler simulator is intentionally kept out of the tracked repo
at `.local/run_scheduler_simulation.py` so it does not bloat the runtime diff.
`.local/` is already gitignored in this checkout.

From the repo root:

```bash
python3 .local/run_scheduler_simulation.py \
  --no-stream \
  --output-json /tmp/trtllm-scheduler-sim.json \
  --output-markdown /tmp/trtllm-scheduler-sim.md
```

For a heavier run:

```bash
python3 .local/run_scheduler_simulation.py \
  --request-scale 1.5 \
  --qps-scale 1.5 \
  --no-stream \
  --output-json /tmp/trtllm-scheduler-sim-heavy.json \
  --output-markdown /tmp/trtllm-scheduler-sim-heavy.md
```

The simulator compares:

- baseline: `stock_default`
- candidate: `vtc_tier_shares`

The built-in profiles are:

- `premium_protection`
- `tier_mix_pressure`
- `same_tier_burst`
- `premium_noisy_neighbor`
