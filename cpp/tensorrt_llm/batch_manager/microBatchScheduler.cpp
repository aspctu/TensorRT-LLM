/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"

#include <unordered_set>

namespace tensorrt_llm::batch_manager
{

using SizeType32 = MicroBatchScheduler::SizeType32;

MicroBatchScheduler::MicroBatchScheduler(std::optional<batch_scheduler::ContextChunkingConfig> ctxChunkConfig,
    std::optional<SizeType32> maxContextLength, LlmRequestState noScheduleUntilState,
    LlmRequestState noScheduleAfterState, bool decodeTokenBudgetEnabled, float decodeTokenBudgetScaleTokens,
    float decodeTokenBudgetMinRate, float decodeTokenBudgetMaxBalance)
    : mMaxContextLength(maxContextLength)
    , mCtxChunkConfig(ctxChunkConfig)
    , mNoScheduleUntilState(noScheduleUntilState)
    , mNoScheduleAfterState(noScheduleAfterState)
    , mDecodeTokenBudgetEnabled(decodeTokenBudgetEnabled)
    , mDecodeTokenBudgetScaleTokens(decodeTokenBudgetScaleTokens)
    , mDecodeTokenBudgetMinRate(decodeTokenBudgetMinRate)
    , mDecodeTokenBudgetMaxBalance(decodeTokenBudgetMaxBalance)
{
}

void MicroBatchScheduler::fitDraftTokens(RequestVector& contextsToBeChunked,
    std::optional<SizeType32> ctxTokensCapacity, SizeType32 const chunkUnitSize,
    std::optional<SizeType32> const& maxContextLength)
{
    // How many context tokens are in this batch already?
    SizeType32 numCtxTokens{0};
    for (auto const& llmReq : contextsToBeChunked)
    {
        numCtxTokens += llmReq->getContextChunkSize();
    }

    // Discard draft tokens that won't fit into the existing chunk unit, max
    // context length, or token capacity.
    for (auto& llmReq : contextsToBeChunked)
    {
        if (llmReq->isLastContextChunk() && llmReq->hasDraftTokens())
        {
            // How many more tokens could fit into this chunkUnit? (Round up to next multiple of chunkUnitSize)
            // Each chunkUnit requires an additional kvcache block, so we don't want to use an extra one just for draft
            // tokens.
            SizeType32 remainder = llmReq->getContextChunkSize() % chunkUnitSize;
            SizeType32 remainingSpaceForDraftTokens = remainder == 0 ? 0 : chunkUnitSize - remainder;

            if (maxContextLength)
            {
                // How much space is remaining before reaching maxContextLength?
                SizeType32 remainingContextLength = maxContextLength.value() - llmReq->getContextChunkSize();
                remainingSpaceForDraftTokens = std::min(remainingSpaceForDraftTokens, remainingContextLength);
            }
            if (ctxTokensCapacity)
            {
                // How much space is remaining before reaching ctxTokensCapacity?
                remainingSpaceForDraftTokens
                    = std::min(remainingSpaceForDraftTokens, ctxTokensCapacity.value() - numCtxTokens);
                numCtxTokens += remainingSpaceForDraftTokens;
            }
            // Discard draft tokens.
            SizeType32 const draftTokensToDiscard = llmReq->getNumDraftTokens() - remainingSpaceForDraftTokens;
            if (draftTokensToDiscard > 0)
            {
                TLLM_LOG_DEBUG("Discarding %d draft tokens", draftTokensToDiscard);
                llmReq->discardDraftTokens(draftTokensToDiscard);
            }
        }
    }
}

template <>
void MicroBatchScheduler::setCtxRequestsChunkSize<MicroBatchScheduler::ContextChunkingPolicy::kEQUAL_PROGRESS>(
    RequestVector& contextsToBeChunked, std::optional<SizeType32> ctxTokensCapacity, SizeType32 const chunkUnitSize,
    std::optional<SizeType32> const& maxContextLength)
{
    SizeType32 numCtxTokens{0};
    SizeType32 numTokensSingleLoop{1};

    while ((!ctxTokensCapacity || numCtxTokens < ctxTokensCapacity.value()) && numTokensSingleLoop)
    {
        numTokensSingleLoop = 0;
        for (auto& llmReq : contextsToBeChunked)
        {
            SizeType32 pastChunkSize = llmReq->getContextChunkSize();

            SizeType32 suggestedChunkSize = pastChunkSize + chunkUnitSize;
            llmReq->setContextChunkSize(suggestedChunkSize);

            SizeType32 actualChunkSize = llmReq->getContextChunkSize();
            SizeType32 actualIncrement = actualChunkSize - pastChunkSize;

            if ((ctxTokensCapacity && numCtxTokens + actualIncrement > ctxTokensCapacity.value())
                || (maxContextLength && actualChunkSize > maxContextLength.value()))
            {
                llmReq->setContextChunkSize(pastChunkSize);
                continue;
            }
            numCtxTokens += actualIncrement;
            numTokensSingleLoop += actualIncrement;
        }
    }
}

template <>
void MicroBatchScheduler::setCtxRequestsChunkSize<MicroBatchScheduler::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED>(
    RequestVector& contextsToBeChunked, std::optional<SizeType32> ctxTokensCapacity, SizeType32 const chunkUnitSize,
    std::optional<SizeType32> const& maxContextLength)
{
    for (auto& llmReq : contextsToBeChunked)
    {
        SizeType32 const suggestedChunkSize = llmReq->getContextRemainingLength();
        SizeType32 actualChunkSize = suggestedChunkSize;
        if (ctxTokensCapacity)
        {
            actualChunkSize = std::min(ctxTokensCapacity.value(), actualChunkSize);
        }
        if (maxContextLength)
        {
            actualChunkSize = std::min(maxContextLength.value(), actualChunkSize);
        }
        if (actualChunkSize != suggestedChunkSize)
        {
            actualChunkSize = actualChunkSize / chunkUnitSize * chunkUnitSize;
        }
        llmReq->setContextChunkSize(actualChunkSize);
        if (ctxTokensCapacity)
        {
            ctxTokensCapacity = ctxTokensCapacity.value() - actualChunkSize;
        }
    }
}

void MicroBatchScheduler::setCtxRequestsChunkSize(RequestVector& contextsToBeChunked,
    ContextChunkingPolicy const ctxChunkPolicy, std::optional<SizeType32> ctxTokensCapacity,
    SizeType32 const chunkUnitSize, std::optional<SizeType32> const& maxContextLength)
{
    for (auto& llmReq : contextsToBeChunked)
    {
        llmReq->setContextChunkSize(0);
    }
    switch (ctxChunkPolicy)
    {
    case ContextChunkingPolicy::kEQUAL_PROGRESS:
        setCtxRequestsChunkSize<MicroBatchScheduler::ContextChunkingPolicy::kEQUAL_PROGRESS>(
            contextsToBeChunked, ctxTokensCapacity, chunkUnitSize, maxContextLength);
        break;
    case ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED:
        setCtxRequestsChunkSize<MicroBatchScheduler::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED>(
            contextsToBeChunked, ctxTokensCapacity, chunkUnitSize, maxContextLength);
        break;
    default: TLLM_THROW("The chunked scheduling type `NO_CHUNKING` cannot be performed.");
    }

    // After scheduling chunk sizes, discard draft tokens that won't fit.
    fitDraftTokens(contextsToBeChunked, ctxTokensCapacity, chunkUnitSize, maxContextLength);
}

std::tuple<RequestVector, RequestVector> MicroBatchScheduler::operator()(RequestVector& activeRequests,
    ReqIdsSet const& inflightReqIds, SizeType32 maxBatchSizeRuntime,
    std::optional<SizeType32> maxNumTokensRuntime) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(microBatcherScheduleRequests);

    /* Decode token-budget scheduling
    * We observe that users who send us requests that generate many tokens (long output requests) 
    * cause significant performance degradation. By default, we run GUARANTEED_NO_EVICT scheduler which means that a 
    * request that generates many tokens can take up a batch slot for a while. Other requests, like shorter requests,  
    * spend time waiting in the queue for long-output to finish. The following ONLY applies when all batch slots are 
    * occupied.
    *
    * We solve this with a "decode-token budget". At each time step, every generation request earns credit. 
    *   credit_per_timestep = (max(min_credit, 1 / (1 + total generated tokens / generated token scale)))
    * The more tokens a request has already generated, the less "credit" that request is given. 
    * 
    * A request is eligible to be scheduled for decode when its credit balance reaches >= 1.0. 
    *   credit = sum(credit_per_timestep) over timesteps
    * 
    * Each scheduling iteration consumes 1 credit. If the scheduler has skipped many iterations on a given request, we cap how much
    * credit it can accmulate so that it doesn't dominate the next set of iterations. 
    *
    * Resulting behavior:
    *   - Under light load, behavior is unchanged (the credit mechanism is off).
    *   - Under overload, short / newly-started generations tend to be scheduled every step, while long-running decode
    *     requests skip some steps so they cannot continuously hog batch slots.
    *   - The long-running requests still make forward progress and keep their KV cache resident; they are just
    *     scheduled less frequently.
    * 
    * One risk is that if a long output request is not scheduled, it's KV cache may be evicted.
    */

    RequestVector contextRequests, generationRequests;
    SizeType32 batchNumTokens{0};
    SizeType32 scheduledReqSize{0};
    SizeType32 scheduledBeamWidth{0}; // 0 means no request is scheduled

    bool applyDecodeTokenBudget = false;
    std::unordered_set<LlmRequest::RequestIdType> seenDecodeIds;
    // We only apply the decode token budget when there are requests pending and the batch is full.
    if (mDecodeTokenBudgetEnabled && maxBatchSizeRuntime > 0
        && activeRequests.size() > static_cast<std::size_t>(maxBatchSizeRuntime))
    {
        applyDecodeTokenBudget = true;
        if (mDecodeTokenBudgetScaleTokens <= 0.0f)
        {
            applyDecodeTokenBudget = false;
        }
        if (mDecodeTokenBudgetMinRate <= 0.0f)
        {
            applyDecodeTokenBudget = false;
        }
        if (mDecodeTokenBudgetMaxBalance < 1.0f)
        {
            applyDecodeTokenBudget = false;
        }
    }

    if (applyDecodeTokenBudget)
    {
        for (auto const& llmReq : activeRequests)
        {
            if (!llmReq->isGenerationInProgressState())
            {
                continue;
            }
            auto const reqId = llmReq->mRequestId;
            seenDecodeIds.insert(reqId);
            // If the request is already in-flight on another micro-batch, we don't want to "re-credit" it here.
            if (inflightReqIds.find(reqId) != inflightReqIds.end())
            {
                continue;
            }
            auto const generated = static_cast<float>(llmReq->getMaxNumGeneratedTokens());
            auto rate = 1.0f / (1.0f + (generated / mDecodeTokenBudgetScaleTokens));
            rate = std::max(rate, mDecodeTokenBudgetMinRate);
            auto& credit = mDecodeTokenBudgetCredits[reqId];
            credit = std::min(credit + rate, mDecodeTokenBudgetMaxBalance);
        }

        for (auto it = mDecodeTokenBudgetCredits.begin(); it != mDecodeTokenBudgetCredits.end();)
        {
            if (seenDecodeIds.count(it->first) == 0)
            {
                it = mDecodeTokenBudgetCredits.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    RequestVector contextsToBeChunked;
    SizeType32 numChunkedTokens{0};
    bool allContextRequestsFit{true};
    std::shared_ptr<LlmRequest> firstGenCandidate;

    // 1. Select the generation phase requests that meet the criteria of total token size.
    //    If there is any remaining space, include the context requests and divide them into chunks.
    for (auto& llmReq : activeRequests)
    {
        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (!llmReq->hasReachedState(mNoScheduleUntilState) || llmReq->hasReachedState(mNoScheduleAfterState))
        {
            continue;
        }

        // if already in execution, skip
        if (inflightReqIds.find(llmReq->mRequestId) != inflightReqIds.end())
        {
            continue;
        }

        SizeType32 reqNumTokens = 0;
        if (llmReq->isEncoderInitState())
        {
            reqNumTokens = llmReq->getEncoderOutputLen();
            TLLM_CHECK_WITH_INFO(!mMaxContextLength || reqNumTokens <= mMaxContextLength.value(),
                "The number of encoder tokens (%d) exceeds the limit value (%d)", reqNumTokens,
                mMaxContextLength.value());
            if (maxNumTokensRuntime && batchNumTokens + reqNumTokens > maxNumTokensRuntime.value())
            {
                break;
            }
            TLLM_LOG_DEBUG("encoder request scheduled: ID %u", llmReq->mRequestId);
            contextRequests.emplace_back(llmReq);
            batchNumTokens += reqNumTokens;
        }
        else if (llmReq->isContextInitState())
        {
            if (!mCtxChunkConfig) // skip chunking
            {
                constexpr SizeType32 beam{0};
                reqNumTokens
                    = llmReq->getNumTokens(beam) + (llmReq->hasDraftTokens() ? llmReq->getNumDraftTokens() : 0);
                TLLM_CHECK_WITH_INFO(!mMaxContextLength || reqNumTokens <= mMaxContextLength.value(),
                    "The number of context tokens (%d) exceeds the limit value (%d)", reqNumTokens,
                    mMaxContextLength.value());
                if (maxNumTokensRuntime && batchNumTokens + reqNumTokens > maxNumTokensRuntime.value())
                {
                    break;
                }
                TLLM_LOG_DEBUG("context request scheduled: ID %u", llmReq->mRequestId);
                contextRequests.emplace_back(llmReq);
                batchNumTokens += reqNumTokens;
            }
            else
            {
                llmReq->setContextChunkSize(llmReq->getContextRemainingLength());
                auto const draftTokens
                    = (llmReq->isLastContextChunk() && llmReq->hasDraftTokens()) ? llmReq->getNumDraftTokens() : 0;
                reqNumTokens = llmReq->getContextChunkSize() + draftTokens;

                if (mMaxContextLength)
                {
                    if (mMaxContextLength.value() < reqNumTokens)
                    {
                        // The context exceeds the length limit, we need to try chunking later.
                        reqNumTokens = mMaxContextLength.value();
                        allContextRequestsFit = false;
                    }
                }
                contextsToBeChunked.emplace_back(llmReq);
                numChunkedTokens += reqNumTokens;
                TLLM_LOG_DEBUG("contexts-to-be-chunked request scheduled: ID %u", llmReq->mRequestId);
            }
        }
        else // (llmReq->isGenerationInProgressState())
        {
            auto const reqBeamWidth = llmReq->getBeamWidthByIter();
            reqNumTokens = reqBeamWidth + llmReq->getNumDraftTokens();
            if (maxNumTokensRuntime && batchNumTokens + reqNumTokens > maxNumTokensRuntime.value())
            {
                break;
            }
            if (scheduledBeamWidth == 0) // set `scheduledBeamWidth` when the first request is scheduled
            {
                scheduledBeamWidth = reqBeamWidth;
            }
            else if (scheduledBeamWidth != reqBeamWidth) // Skip request with different beam width
            {
                TLLM_LOG_DEBUG(
                    "generation request skipped: ID %u since its beam width (%d) is different from scheduled ones (%d)",
                    llmReq->mRequestId, reqBeamWidth, scheduledBeamWidth);
                continue;
            }
            if (!firstGenCandidate)
            {
                firstGenCandidate = llmReq;
            }
            if (applyDecodeTokenBudget)
            {
                auto const reqId = llmReq->mRequestId;
                auto it = mDecodeTokenBudgetCredits.find(reqId);
                if (it != mDecodeTokenBudgetCredits.end() && it->second < 1.0f)
                {
                    continue;
                }
                if (it != mDecodeTokenBudgetCredits.end())
                {
                    it->second -= 1.0f;
                }
            }
            TLLM_LOG_DEBUG("generation request scheduled: ID %u with beam width %d", llmReq->mRequestId, reqBeamWidth);
            generationRequests.emplace_back(llmReq);
            batchNumTokens += reqNumTokens;
        }

        if (++scheduledReqSize >= maxBatchSizeRuntime)
        {
            break;
        }
    }

    if (generationRequests.empty() && firstGenCandidate && contextsToBeChunked.empty() && contextRequests.empty())
    {
        auto const reqBeamWidth = firstGenCandidate->getBeamWidthByIter();
        auto const reqNumTokens = reqBeamWidth + firstGenCandidate->getNumDraftTokens();
        if (!maxNumTokensRuntime || reqNumTokens <= maxNumTokensRuntime.value())
        {
            generationRequests.emplace_back(firstGenCandidate);
        }
    }

    if (maxNumTokensRuntime && numChunkedTokens > maxNumTokensRuntime.value() - batchNumTokens)
    {
        allContextRequestsFit = false;
    }

    // 2. If not all contexts fit into the batch, the chunk size should be adjusted accordingly.
    if (!allContextRequestsFit)
    {
        TLLM_CHECK_WITH_INFO(mCtxChunkConfig, "If chunking is not enabled, context scheduling should be completed.");
        auto const ctxTokensCapacity
            = maxNumTokensRuntime ? std::make_optional(maxNumTokensRuntime.value() - batchNumTokens) : std::nullopt;
        setCtxRequestsChunkSize(contextsToBeChunked, mCtxChunkConfig.value().chunkingPolicy, ctxTokensCapacity,
            mCtxChunkConfig.value().chunkUnitSize, mMaxContextLength);
    }
    for (auto const& llmReq : contextsToBeChunked)
    {
        if (llmReq->getContextChunkSize() > 0)
        {
            contextRequests.emplace_back(llmReq);
            batchNumTokens += llmReq->getContextChunkSize();
            TLLM_LOG_DEBUG(
                "context request scheduled: ID %lu, chunk size %d", llmReq->mRequestId, llmReq->getContextChunkSize());
        }
    }

    utils::sortRequests(contextRequests, generationRequests, !allContextRequestsFit);

    TLLM_LOG_DEBUG(
        "batchSize (num ctx/enc requests + num gen requests): %u", contextRequests.size() + generationRequests.size());
    TLLM_LOG_DEBUG("batchNumTokens (num ctx/enc input tokens + num gen input tokens) / maxNumTokens: %d / %d",
        batchNumTokens, maxNumTokensRuntime.value_or(0));
    TLLM_LOG_DEBUG(
        "[Summary] Micro Batch scheduler schedules %d context/encoder requests, %d generation requests. "
        "%d requests inflight with the model already",
        contextRequests.size(), generationRequests.size(), inflightReqIds.size());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {std::move(contextRequests), std::move(generationRequests)};
}

} // namespace tensorrt_llm::batch_manager
