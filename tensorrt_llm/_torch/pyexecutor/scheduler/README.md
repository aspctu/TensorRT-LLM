# PyExecutor Scheduler Notes

This directory is the Python-facing scheduler surface for the PyTorch executor.
The real scheduling logic is split across Python and C++.

The short version is:

- queued requests are ordered in Python
- active requests are scheduled in C++ by default
- the final scheduler output is `ScheduledRequests`
- `model_engine.forward(scheduled_requests, ...)` is the execution boundary

This file is for maintainers. The higher-level feature doc is
`docs/source/torch/scheduler.md`.

## What runs by default

The default PyTorch path is not the pure-Python scheduler.

Scheduler selection happens in `../_util.py`:

- `use_python_scheduler=False` by default
- `SchedulerConfig.capacity_scheduler_policy=GUARANTEED_NO_EVICT` by default
- the executor builds:
  - `SimpleScheduler`
  - `BindCapacityScheduler`
  - `BindMicroBatchScheduler`

So the default execution path is:

1. Python waiting queue
2. C++ capacity scheduler
3. C++ microbatch scheduler
4. Python `ScheduledRequests`
5. `model_engine.forward(...)`

The optional pure-Python path is `SimpleUnifiedScheduler`, but it only runs if
the caller explicitly enables `use_python_scheduler=True`.

## End-to-end flow

The request path in the PyTorch backend is:

1. Requests arrive as executor-side request objects and are wrapped in
   `RequestQueueItem`.
2. `PyExecutor` inserts them into a waiting queue.
3. The waiting queue decides which queued request should be activated next.
4. Activation converts the executor request into an `LlmRequest`.
5. The scheduler runs on the active `LlmRequest` set.
6. Capacity scheduling produces a fitting set and possibly a paused set.
7. Microbatch scheduling partitions the fitting set into context and generation
   work for this iteration.
8. `PyExecutor._schedule()` turns the result into `ScheduledRequests`.
9. `PyExecutor._forward_step()` passes `ScheduledRequests` into
   `model_engine.forward(...)`.

Once the ordered `ScheduledRequests` object is built, the order is real. It
drives downstream batching, slot assignment, and model input construction.

## Component map

### Python queue-side scheduling

Files:

- `../py_executor.py`
- `waiting_queue.py`
- `../scheduler_fairness.py`
- `../llm_request.py`
- `../../scheduling_params.py`

Responsibilities:

- track queued requests before they become active
- score queued work
- activate requests in score order
- copy Python scheduling params onto `LlmRequest`
- expose fairness metrics back through Python stats

### C++ active scheduling

Files:

- `cpp/include/tensorrt_llm/batch_manager/capacityScheduler.h`
- `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp`
- `cpp/include/tensorrt_llm/batch_manager/microBatchScheduler.h`
- `cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp`
- `cpp/include/tensorrt_llm/batch_manager/llmRequest.h`
- `cpp/tensorrt_llm/nanobind/batch_manager/bindings.cpp`

Responsibilities:

- choose which active requests fit this iteration
- decide whether started requests are paused
- compute the final per-iteration context/generation split
- enforce token-count and batch-size limits
- carry default-path fairness state on `LlmRequest`

## Python waiting-queue algorithm

The waiting queue currently uses `PriorityWaitingQueue`, not plain FIFO, when a
priority function is installed.

That priority function comes from `SchedulerFairnessController` in
`../scheduler_fairness.py`.

The queue-side score is:

- `priority tier bias`
- `+ waiting age credit`
- `+ priority credit`
- `+ organization token balance / score scale`

The pieces are:

- `priority_tier` comes from `SchedulingParams.priority_tier`
- `organization_id` comes from `SchedulingParams.organization_id`
- waiting age credit grows with queue wait time and caps out
- each `(priority_tier, organization_id)` pair carries a token-balance state
- organizations that have consumed more recent scheduled tokens and active
  capacity get a more negative balance and are less likely to be activated next
  within that tier

In other words, queue-side balancing is now based on recent token service, not a
hand-tuned spike or long-decode penalty.

This is intentionally queue-side logic. It decides who gets activated, not who
ultimately runs once requests are already active.

## Request metadata path

The Python scheduler parameters are defined in `../../scheduling_params.py`:

- `priority_tier`
- `organization_id`

When a request becomes active, `../llm_request.py` copies that metadata onto the
bound `LlmRequest`:

- sets request `priority`
- stores `py_priority_tier`
- stores `py_organization_id`
- enables scheduler controls
- stores `scheduler_organization_hash`

The default C++ scheduler then reads the bound fields directly from
`LlmRequest`.

The relevant bound scheduler fields live on `LlmRequest` and are exposed through
nanobind:

- `scheduler_organization_hash`
- `scheduler_controls_enabled`
- `scheduler_credit`
- `scheduler_pause_count`
- `scheduler_score`
- `scheduler_age_credit`

The `scheduler_credit` and `scheduler_pause_count` fields are the important
request-local fairness state. Organization balancing lives in scheduler-owned
state in the capacity scheduler rather than on the request itself.

## C++ capacity scheduler algorithms

The top-level capacity scheduler is `batch_manager::CapacityScheduler`. It is a
policy wrapper that dispatches to one of four implementations.

### 1. `MaxRequestsScheduler`

Use case:

- no KV cache manager
- simple request-count limiting

Algorithm:

- scan the active list in order
- skip requests outside the schedulable state range
- schedule up to `maxNumRequests`
- never pause requests

This is the simplest policy and mainly acts as a fallback path.

### 2. `GuaranteedNoEvictScheduler`

Use case:

- keep already-started requests running
- do not pause started requests

Algorithm:

- reserve resources for already-running generation requests first
- track reserved KV blocks, cross-KV blocks, and PEFT pages
- collect pending context and disaggregated-generation-init requests
- greedily admit pending work while capacity remains
- if a pending request cannot fit KV capacity, stop trying later pending
  requests
- optionally skip some context requests if delaying them improves KV block reuse

This is the default capacity policy in `SchedulerConfig`.

### 3. `StaticBatchScheduler`

Use case:

- static-style batching behavior

Algorithm:

- same resource-reservation core as `GuaranteedNoEvictScheduler`
- but only admits pending requests when there are no currently active requests

This is more restrictive than `GuaranteedNoEvictScheduler`.

### 4. `MaxUtilizationScheduler`

Use case:

- maximize utilization, even if that means pausing started work

Algorithm:

- walk the active list in order
- try to fit each request using `MaxUtilizationScheduledBlocksManager`
- if the request fits, schedule it
- if it does not fit, find the last started request near the tail of the active
  vector
- simulate removing that request from scheduling
- mark it paused
- retry forward progress with the reduced active suffix

This is the policy that can pause started requests under pressure.

`TIER_AWARE_MAX_UTILIZATION` reuses this scheduler, but changes victim
selection: when a pause is required, it prefers the lowest started tier first,
then the lowest fairness score within that tier.

## Default-path C++ fairness logic

The default-path fairness logic also lives in `capacityScheduler.cpp`.

If any active request has `scheduler_controls_enabled=True`, the scheduler:

1. updates organization fairness state for the active set
2. computes a fairness score for each schedulable request
3. stable-sorts the active list by that score
4. runs the selected capacity policy on the reordered list
5. updates request credit and pause history after scheduling

The active fairness score is:

- request `priority() * basePriorityWeight`
- `+ schedulerCredit`
- `+ pauseCount * pauseProtection`
- `+ organizationTokenBalance / scoreScale`

The main ideas are:

- higher-tier traffic starts with higher base priority
- paused or skipped requests build credit over time
- organization balancing happens within a tier by charging scheduled token
  service back to the org
- orgs that consumed more recent scheduled tokens get a more negative token
  balance, so other orgs in the same tier move ahead

After each iteration:

- paused requests gain extra credit and increment pause count
- scheduled requests spend request credit
- scheduled requests also charge token service to their `(tier, org)` balance
- unscheduled-but-active requests slowly gain credit

This gives two layers of fairness:

- tiering across organizations
- token-based balancing within a tier

## C++ microbatch scheduler algorithm

The microbatch scheduler runs after capacity scheduling. It does not decide the
full fairness policy. It takes the already-fitting active set and turns it into
the exact batch for the next iteration.

Core algorithm:

- iterate the fitting active requests in order
- skip requests already inflight on another microbatch
- greedily add work until `maxBatchSize` or `maxNumTokens` would be exceeded
- schedule generation requests directly
- collect context requests for whole scheduling or chunking
- if needed, reduce context chunk sizes to fit the remaining token budget
- emit separate `contextRequests` and `generationRequests`

Important details:

- generation requests in the same batch must share beam width
- context requests may be chunked
- draft tokens may be dropped if they would force an extra chunk or exceed token
  capacity
- final request order is normalized by `utils::sortRequests(...)`

### Context chunking policies

These live in `microBatchScheduler.cpp`.

`EQUAL_PROGRESS`:

- grow all chunked context requests round-robin by `chunkUnitSize`
- tries to keep chunked requests making similar progress

`FIRST_COME_FIRST_SERVED`:

- give earlier context requests as much chunk space as possible first
- later requests get whatever budget remains

After chunk sizes are chosen, draft tokens that no longer fit are discarded.

## Final execution boundary

The final scheduler result before model execution is `ScheduledRequests`.

`PyExecutor._schedule()`:

- calls `self.scheduler.schedule_request(...)`
- optionally applies attention-DP balancing and batch waiting
- builds a `ScheduledRequests` object

`PyExecutor._forward_step()` then calls:

```python
self.model_engine.forward(scheduled_requests, ...)
```

That is the actual handoff from scheduling to execution.

The executor may still do small post-scheduler adjustments, but it does not
replace the main capacity or microbatch scheduling decisions after this point.

## Where to edit behavior

Use this rule of thumb:

- edit `waiting_queue.py` or `../scheduler_fairness.py` for queued-request
  admission order
- edit `capacityScheduler.cpp` for default active scheduling behavior
- edit `microBatchScheduler.cpp` for per-iteration batch formation after the
  fitting set is already correct
- edit `PyCapacityScheduler` only if you explicitly care about the optional
  non-default Python scheduler path

## Useful tests

Primary tests around this area:

- `cpp/tests/unit_tests/batch_manager/capacitySchedulerTest.cpp`
- `cpp/tests/unit_tests/batch_manager/microBatchSchedulerTest.cpp`
- `tests/unittest/_torch/executor/test_waiting_queue.py`
- `tests/unittest/_torch/executor/test_scheduler_fairness.py`
- `tests/unittest/_torch/executor/test_py_scheduler.py`
- `tests/unittest/executor/test_stats_serializer.py`

If you change default-path scheduling behavior, start with the C++ scheduler
tests.

For multi-tenant changes, also run the synthetic simulation matrix from the
repo root:

- `make -f Makefile.scheduler scheduler-sim-matrix`

That harness lives in `scripts/run_scheduler_simulation.py`. It compares a
baseline soft-tier policy against the current strict tier + org-balancing
policy across built-in noisy-neighbor and tier-pressure profiles, and emits:

- per-tier and per-org TTFT / latency tail metrics
- TTFT / latency SLA miss shares
- per-tier blocked ratios
- per-tier service share so lower-tier crowd-out is obvious
