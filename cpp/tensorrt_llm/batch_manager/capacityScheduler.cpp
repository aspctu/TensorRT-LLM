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

#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/scheduledBlocksManager.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::batch_manager
{
using kv_cache_manager::VecUniqueTokens;
using kv_cache_manager::BlockKey;
using kv_cache_manager::BlockKeyHasher;

namespace
{

std::tuple<std::unordered_set<BlockKey, BlockKeyHasher>, std::unordered_set<BlockKey, BlockKeyHasher>>
prefillWithChunkedContextsAlreadyExecuting(RequestList const& activeRequests,
    kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager = std::nullopt)
{
    std::unordered_set<BlockKey, BlockKeyHasher> newlyContributedContextBlocks;
    std::unordered_set<BlockKey, BlockKeyHasher> newlyContributedCrossContextBlocks;
    for (auto const& req : activeRequests)
    {
        if (req->isContextInitState() && !req->isFirstContextChunk())
        {
            // Chunked context request already executing, but haven't completed all chunks yet.
            // Skipping is not an option, register it's contributed blocks
            if (kvCacheManager.isEnableBlockReuse())
            {
                auto uniqueTokens = req->getUniqueTokens(0);
                auto newContextBlockOpt = kvCacheManager.findNewContextBlock(uniqueTokens, *req);
                if (newContextBlockOpt.has_value())
                {
                    newlyContributedContextBlocks.insert(newContextBlockOpt.value());
                }
            }
            if (crossKvCacheManager && crossKvCacheManager->isEnableBlockReuse())
            {
                auto uniqueTokens = *(req->getEncoderUniqueTokens().value());
                auto newContextBlockOpt = crossKvCacheManager->findNewContextBlock(uniqueTokens, *req);
                if (newContextBlockOpt.has_value())
                {
                    newlyContributedCrossContextBlocks.insert(newContextBlockOpt.value());
                }
            }
        }
    }
    return {std::move(newlyContributedContextBlocks), std::move(newlyContributedCrossContextBlocks)};
}

bool oneManagerBeneficialToSkip(tensorrt_llm::batch_manager::kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
    VecUniqueTokens const& uniqueTokens, std::shared_ptr<LlmRequest> const& llmRequest,
    std::unordered_set<BlockKey, BlockKeyHasher>& newlyContributedContextBlocks)
{
    // Find first context block that isn't already in KV cache
    auto newContextBlockOpt = kvCacheManager.findNewContextBlock(uniqueTokens, *llmRequest);
    if (newContextBlockOpt.has_value())
    {
        auto const& newContextBlock = newContextBlockOpt.value();
        if (newlyContributedContextBlocks.count(newContextBlock) > 0)
        {
            // newContextBlock was contributed by earlier scheduled request.
            // Better to skip this step so we can reuse.
            return true;
        }

        // This request is contributing newContextBlock.
        newlyContributedContextBlocks.insert(newContextBlock);
    }
    // Either all context blocks are already in KV cache,
    // or no previously scheduled request has contributed newContextBlock.
    return false;
}

//! \brief Check if it is beneficial to skip this request rather than schedule it.
//! \details One condition that makes it beneficial is if this request can reuse kv cache block(s) contributed by
//! already scheduled context requests.
bool beneficialToSkip(std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest> const& req,
    kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
    std::unordered_set<BlockKey, BlockKeyHasher>& newlyContributedContextBlocks,
    std::unordered_set<BlockKey, BlockKeyHasher>& newlyContributedCrossContextBlocks)
{
    if (req->isContextInitState() && req->isFirstContextChunk())
    {
        if (kvCacheManager.isEnableBlockReuse())
        {
            auto uniqueTokens = req->getUniqueTokens(0);
            if (oneManagerBeneficialToSkip(kvCacheManager, uniqueTokens, req, newlyContributedContextBlocks))
            {
                return true;
            }
        }
        if (crossKvCacheManager && crossKvCacheManager->isEnableBlockReuse())
        {
            auto uniqueTokens = *(req->getEncoderUniqueTokens().value());
            if (oneManagerBeneficialToSkip(*crossKvCacheManager, uniqueTokens, req, newlyContributedCrossContextBlocks))
            {
                return true;
            }
        }
    }
    return false;
}

using OrgFairnessStates = std::unordered_map<std::uint64_t, double>;

struct RequestFairnessScore
{
    double total;
    double ageCredit;
    double orgTokenBalance;
};

constexpr double kSchedulerBasePriorityWeight = 1024.0;
constexpr double kSchedulerMaxCredit = 8.0;
constexpr double kSchedulerWaitCredit = 0.75;
constexpr double kSchedulerPauseBonus = 1.5;
constexpr double kSchedulerPauseProtection = 0.25;
constexpr double kSchedulerServiceCost = 0.75;

constexpr double kOrgTokenBalanceDecay = 0.85;
constexpr double kOrgTokenBalanceMin = -256.0;
constexpr double kOrgTokenScoreScale = 16.0;
constexpr double kAdmissionPromptTokenCostDivisor = 128.0;
constexpr double kAdmissionMaxNewTokenCostWeight = 0.25;
constexpr double kAdmissionMaxCharge = 96.0;
constexpr double kOrgActivePressureScale = 48.0;
constexpr double kOrgActivePressurePenaltyScale = 8.0;

[[nodiscard]] bool isSchedulableRequest(
    std::shared_ptr<LlmRequest> const& req, LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
{
    if (req->isDisaggGenerationInitState())
    {
        return true;
    }
    return req->hasReachedState(noScheduleUntilState) && !req->hasReachedState(noScheduleAfterState);
}

[[nodiscard]] bool hasSchedulerControlsEnabled(RequestList const& activeRequests)
{
    return std::any_of(activeRequests.begin(), activeRequests.end(),
        [](std::shared_ptr<LlmRequest> const& req) { return req->getSchedulerControlsEnabled(); });
}

[[nodiscard]] std::uint32_t getSchedulerTier(LlmRequest const& req)
{
    auto const priority = static_cast<double>(req.priority());
    return static_cast<std::uint32_t>(std::max(0.0, std::floor(priority)));
}

[[nodiscard]] std::uint64_t getOrgFairnessKey(LlmRequest const& req)
{
    auto const tier = static_cast<std::uint64_t>(getSchedulerTier(req));
    auto const organizationHash = req.getSchedulerOrganizationHash();
    return organizationHash ^ (tier + 0x9e3779b97f4a7c15ULL + (organizationHash << 6) + (organizationHash >> 2));
}

[[nodiscard]] std::unordered_map<std::uint32_t, SizeType32> allocateReservedTierSlots(
    std::vector<std::uint32_t> const& orderedTiers, SizeType32 totalSlots)
{
    if (totalSlots <= 0 || orderedTiers.empty())
    {
        return {};
    }

    std::unordered_map<std::uint32_t, SizeType32> reservedSlots;
    reservedSlots.reserve(orderedTiers.size());
    auto remainingSlots = totalSlots;
    auto remainingTiers = static_cast<SizeType32>(orderedTiers.size());
    for (auto const tier : orderedTiers)
    {
        if (remainingSlots <= 0)
        {
            reservedSlots.emplace(tier, 0);
        }
        else if (remainingTiers == 1)
        {
            reservedSlots.emplace(tier, remainingSlots);
        }
        else
        {
            reservedSlots.emplace(tier, std::max<SizeType32>(1, remainingSlots - (remainingTiers - 1)));
        }

        remainingSlots -= reservedSlots.at(tier);
        --remainingTiers;
    }

    return reservedSlots;
}

[[nodiscard]] double estimateScheduledTokens(LlmRequest const& req)
{
    if (req.isEncoderInitState())
    {
        return static_cast<double>(std::max<SizeType32>(1, req.getEncoderOutputLen()));
    }

    if (req.isContextInitState() || req.isDisaggGenerationInitState())
    {
        auto chunkSize = req.getContextChunkSize();
        if (chunkSize <= 0)
        {
            chunkSize = req.getNumTokens(0);
        }

        auto const draftTokens = (req.isLastContextChunk() && req.getNumDraftTokens() > 0) ? req.getNumDraftTokens() : 0;
        return static_cast<double>(std::max<SizeType32>(1, chunkSize + draftTokens));
    }

    return static_cast<double>(std::max<SizeType32>(1, 1 + req.getNumDraftTokens()));
}

[[nodiscard]] double estimateActivationCharge(LlmRequest const& req)
{
    auto const promptCost = static_cast<double>(req.getOrigPromptLen()) / kAdmissionPromptTokenCostDivisor;
    auto const decodeCost = static_cast<double>(req.getMaxNewTokens()) * kAdmissionMaxNewTokenCostWeight;
    return std::min(kAdmissionMaxCharge, promptCost + decodeCost);
}

[[nodiscard]] double estimateOrgPressure(LlmRequest const& req)
{
    return 1.0 + (estimateActivationCharge(req) / kOrgActivePressureScale);
}

void updateOrgFairnessStates(OrgFairnessStates& orgFairnessStates, RequestList const& activeRequests,
    LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
{
    for (auto& [orgKey, tokenBalance] : orgFairnessStates)
    {
        tokenBalance = std::max(kOrgTokenBalanceMin, tokenBalance * kOrgTokenBalanceDecay);
    }

    for (auto const& req : activeRequests)
    {
        if (!isSchedulableRequest(req, noScheduleUntilState, noScheduleAfterState))
        {
            continue;
        }
        auto const orgKey = getOrgFairnessKey(*req);
        if (orgFairnessStates.find(orgKey) == orgFairnessStates.end())
        {
            orgFairnessStates.emplace(orgKey, 0.0);
        }
    }
}

[[nodiscard]] RequestFairnessScore evaluateFairnessScore(
    std::shared_ptr<LlmRequest> const& req, OrgFairnessStates const& orgFairnessStates)
{
    double orgTokenBalance = 0.0;
    if (auto const it = orgFairnessStates.find(getOrgFairnessKey(*req)); it != orgFairnessStates.end())
    {
        orgTokenBalance = it->second;
    }

    auto const ageCredit = req->getSchedulerCredit();
    auto const total = req->priority() * kSchedulerBasePriorityWeight + ageCredit
        + static_cast<double>(req->getSchedulerPauseCount()) * kSchedulerPauseProtection
        + (orgTokenBalance / kOrgTokenScoreScale);
    return RequestFairnessScore{total, ageCredit, orgTokenBalance};
}

void assignFairnessMetrics(std::shared_ptr<LlmRequest> const& req, RequestFairnessScore const& score)
{
    req->setSchedulerScore(score.total);
    req->setSchedulerAgeCredit(score.ageCredit);
}

[[nodiscard]] double getVictimScore(std::shared_ptr<LlmRequest> const& req)
{
    if (req->getSchedulerControlsEnabled())
    {
        return req->getSchedulerScore();
    }
    return req->priority();
}

void sortActiveRequestsByFairness(RequestList& activeRequests, OrgFairnessStates const& orgFairnessStates,
    SizeType32 maxNumRequests, LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
{
    std::unordered_map<LlmRequest::RequestIdType, RequestFairnessScore> requestScores;
    requestScores.reserve(activeRequests.size());
    std::unordered_map<std::uint64_t, std::vector<std::shared_ptr<LlmRequest>>> requestsByOrg;
    requestsByOrg.reserve(activeRequests.size());
    for (auto const& req : activeRequests)
    {
        auto const score = evaluateFairnessScore(req, orgFairnessStates);
        assignFairnessMetrics(req, score);
        if (!isSchedulableRequest(req, noScheduleUntilState, noScheduleAfterState))
        {
            continue;
        }

        requestScores.emplace(req->mRequestId, score);
        requestsByOrg[getOrgFairnessKey(*req)].emplace_back(req);
    }

    struct OrganizationQueue
    {
        std::uint32_t tier;
        std::uint64_t orgKey;
        double orgTokenBalance;
        double orgPressure;
        std::vector<std::shared_ptr<LlmRequest>> requests;
        std::size_t nextRequestIndex{0};
        double emittedPressure{0.0};
        double emittedActivationDebt{0.0};
    };

    std::unordered_map<std::uint32_t, std::vector<OrganizationQueue>> organizationQueuesByTier;
    organizationQueuesByTier.reserve(requestsByOrg.size());
    for (auto& [orgKey, orgRequests] : requestsByOrg)
    {
        std::stable_sort(orgRequests.begin(), orgRequests.end(),
            [&requestScores](std::shared_ptr<LlmRequest> const& lhs, std::shared_ptr<LlmRequest> const& rhs)
            {
                return requestScores.at(lhs->mRequestId).total > requestScores.at(rhs->mRequestId).total;
            });

        auto const orgStateIt = orgFairnessStates.find(orgKey);
        auto const orgTokenBalance = orgStateIt != orgFairnessStates.end() ? orgStateIt->second : 0.0;
        auto const tier = getSchedulerTier(*orgRequests.front());
        auto const orgPressure = std::accumulate(orgRequests.begin(), orgRequests.end(), 0.0,
            [](double pressure, std::shared_ptr<LlmRequest> const& req)
            { return pressure + estimateOrgPressure(*req); });
        auto organizationQueue = OrganizationQueue{};
        organizationQueue.tier = tier;
        organizationQueue.orgKey = orgKey;
        organizationQueue.orgTokenBalance = orgTokenBalance;
        organizationQueue.orgPressure = orgPressure;
        organizationQueue.requests = std::move(orgRequests);
        organizationQueuesByTier[tier].emplace_back(std::move(organizationQueue));
    }

    std::vector<std::uint32_t> orderedTiers;
    orderedTiers.reserve(organizationQueuesByTier.size());
    for (auto const& [tier, _organizationQueues] : organizationQueuesByTier)
    {
        orderedTiers.emplace_back(tier);
    }
    std::sort(orderedTiers.begin(), orderedTiers.end(), std::greater<>());

    auto const quotaWindow = std::min<SizeType32>(maxNumRequests, requestScores.size());
    auto const quotasByTier = allocateReservedTierSlots(orderedTiers, quotaWindow);

    std::unordered_map<std::uint32_t, std::vector<std::shared_ptr<LlmRequest>>> orderedRequestsByTier;
    orderedRequestsByTier.reserve(orderedTiers.size());
    for (auto const tier : orderedTiers)
    {
        auto& organizationQueues = organizationQueuesByTier[tier];
        auto const tierFairPressureShare = std::accumulate(organizationQueues.begin(), organizationQueues.end(), 0.0,
            [](double totalPressure, OrganizationQueue const& queue) { return totalPressure + queue.orgPressure; })
            / static_cast<double>(organizationQueues.size());
        std::stable_sort(organizationQueues.begin(), organizationQueues.end(),
            [&requestScores, tierFairPressureShare](OrganizationQueue const& lhs, OrganizationQueue const& rhs)
            {
                auto const lhsTopReq = lhs.requests.front();
                auto const rhsTopReq = rhs.requests.front();
                auto const lhsTopScore = requestScores.at(lhsTopReq->mRequestId).total;
                auto const rhsTopScore = requestScores.at(rhsTopReq->mRequestId).total;
                auto const lhsPressureExcess = std::max(0.0, lhs.orgPressure - tierFairPressureShare);
                auto const rhsPressureExcess = std::max(0.0, rhs.orgPressure - tierFairPressureShare);
                auto const lhsSelectionScore = lhsTopScore - (lhsPressureExcess * kOrgActivePressurePenaltyScale);
                auto const rhsSelectionScore = rhsTopScore - (rhsPressureExcess * kOrgActivePressurePenaltyScale);
                if (lhsSelectionScore != rhsSelectionScore)
                {
                    return lhsSelectionScore > rhsSelectionScore;
                }

                if (lhsPressureExcess != rhsPressureExcess)
                {
                    return lhsPressureExcess < rhsPressureExcess;
                }

                if (lhs.orgPressure != rhs.orgPressure)
                {
                    return lhs.orgPressure < rhs.orgPressure;
                }

                if (lhs.orgTokenBalance != rhs.orgTokenBalance)
                {
                    return lhs.orgTokenBalance > rhs.orgTokenBalance;
                }

                return lhsTopReq->mRequestId < rhsTopReq->mRequestId;
            });

        auto& orderedTierRequests = orderedRequestsByTier[tier];
        auto const breadthLimit = std::min<std::size_t>(
            static_cast<std::size_t>(quotasByTier.at(tier)), organizationQueues.size());
        while (true)
        {
            if (orderedTierRequests.size() >= breadthLimit)
            {
                break;
            }
            double tierFairPressureShare = 0.0;
            SizeType32 activeOrganizationCount = 0;
            for (auto const& queue : organizationQueues)
            {
                if (queue.nextRequestIndex >= queue.requests.size())
                {
                    continue;
                }
                tierFairPressureShare += queue.orgPressure + queue.emittedPressure;
                ++activeOrganizationCount;
            }
            if (activeOrganizationCount == 0)
            {
                break;
            }
            tierFairPressureShare /= static_cast<double>(activeOrganizationCount);

            auto bestQueueIt = organizationQueues.end();
            for (auto queueIt = organizationQueues.begin(); queueIt != organizationQueues.end(); ++queueIt)
            {
                if (queueIt->nextRequestIndex >= queueIt->requests.size())
                {
                    continue;
                }

                if (bestQueueIt == organizationQueues.end())
                {
                    bestQueueIt = queueIt;
                    continue;
                }

                auto const& lhsTopReq = queueIt->requests[queueIt->nextRequestIndex];
                auto const& rhsTopReq = bestQueueIt->requests[bestQueueIt->nextRequestIndex];
                auto const lhsPressure = queueIt->orgPressure + queueIt->emittedPressure;
                auto const rhsPressure = bestQueueIt->orgPressure + bestQueueIt->emittedPressure;
                auto const lhsPressureExcess = std::max(0.0, lhsPressure - tierFairPressureShare);
                auto const rhsPressureExcess = std::max(0.0, rhsPressure - tierFairPressureShare);
                auto const lhsAdjustedScore = requestScores.at(lhsTopReq->mRequestId).total
                    - (queueIt->emittedActivationDebt / kOrgTokenScoreScale)
                    - (lhsPressureExcess * kOrgActivePressurePenaltyScale);
                auto const rhsAdjustedScore = requestScores.at(rhsTopReq->mRequestId).total
                    - (bestQueueIt->emittedActivationDebt / kOrgTokenScoreScale)
                    - (rhsPressureExcess * kOrgActivePressurePenaltyScale);

                if (lhsAdjustedScore != rhsAdjustedScore)
                {
                    if (lhsAdjustedScore > rhsAdjustedScore)
                    {
                        bestQueueIt = queueIt;
                    }
                    continue;
                }

                if (lhsPressureExcess != rhsPressureExcess)
                {
                    if (lhsPressureExcess < rhsPressureExcess)
                    {
                        bestQueueIt = queueIt;
                    }
                    continue;
                }

                if (lhsPressure != rhsPressure)
                {
                    if (lhsPressure < rhsPressure)
                    {
                        bestQueueIt = queueIt;
                    }
                    continue;
                }

                if (lhsTopReq->mRequestId < rhsTopReq->mRequestId)
                {
                    bestQueueIt = queueIt;
                }
            }

            if (bestQueueIt == organizationQueues.end())
            {
                break;
            }

            auto const& nextReq = bestQueueIt->requests[bestQueueIt->nextRequestIndex];
            orderedTierRequests.emplace_back(nextReq);
            bestQueueIt->emittedPressure += estimateOrgPressure(*nextReq);
            bestQueueIt->emittedActivationDebt += estimateActivationCharge(*nextReq);
            ++bestQueueIt->nextRequestIndex;
        }

        std::vector<std::shared_ptr<LlmRequest>> remainingTierRequests;
        for (auto const& queue : organizationQueues)
        {
            for (auto requestIndex = queue.nextRequestIndex; requestIndex < queue.requests.size(); ++requestIndex)
            {
                remainingTierRequests.emplace_back(queue.requests[requestIndex]);
            }
        }
        std::stable_sort(remainingTierRequests.begin(), remainingTierRequests.end(),
            [&requestScores](std::shared_ptr<LlmRequest> const& lhs, std::shared_ptr<LlmRequest> const& rhs)
            {
                auto const lhsScore = requestScores.at(lhs->mRequestId).total;
                auto const rhsScore = requestScores.at(rhs->mRequestId).total;
                if (lhsScore != rhsScore)
                {
                    return lhsScore > rhsScore;
                }
                return lhs->mRequestId < rhs->mRequestId;
            });
        orderedTierRequests.insert(
            orderedTierRequests.end(), remainingTierRequests.begin(), remainingTierRequests.end());
    }

    std::vector<std::shared_ptr<LlmRequest>> schedulableRequests;
    schedulableRequests.reserve(requestScores.size());

    for (auto const tier : orderedTiers)
    {
        auto& orderedTierRequests = orderedRequestsByTier[tier];
        auto const quota = std::min<SizeType32>(quotasByTier.at(tier), orderedTierRequests.size());
        for (SizeType32 index = 0; index < quota; ++index)
        {
            schedulableRequests.emplace_back(orderedTierRequests.front());
            orderedTierRequests.erase(orderedTierRequests.begin());
        }
    }

    auto const quotaWindowSize = static_cast<std::size_t>(quotaWindow);
    while (schedulableRequests.size() < quotaWindowSize)
    {
        bool addedRequest = false;
        for (auto const tier : orderedTiers)
        {
            auto& orderedTierRequests = orderedRequestsByTier[tier];
            if (orderedTierRequests.empty())
            {
                continue;
            }

            schedulableRequests.emplace_back(orderedTierRequests.front());
            orderedTierRequests.erase(orderedTierRequests.begin());
            addedRequest = true;
            if (schedulableRequests.size() >= quotaWindowSize)
            {
                break;
            }
        }
        if (!addedRequest)
        {
            break;
        }
    }

    while (schedulableRequests.size() < requestScores.size())
    {
        bool addedRequest = false;
        for (auto const tier : orderedTiers)
        {
            auto& orderedTierRequests = orderedRequestsByTier[tier];
            if (orderedTierRequests.empty())
            {
                continue;
            }

            schedulableRequests.emplace_back(orderedTierRequests.front());
            orderedTierRequests.erase(orderedTierRequests.begin());
            addedRequest = true;
        }
        if (!addedRequest)
        {
            break;
        }
    }

    auto schedulableReqIt = schedulableRequests.begin();
    for (auto& req : activeRequests)
    {
        if (!isSchedulableRequest(req, noScheduleUntilState, noScheduleAfterState))
        {
            continue;
        }

        req = *schedulableReqIt;
        ++schedulableReqIt;
    }
}

void updateRequestFairnessState(RequestList const& activeRequests, RequestVector const& scheduledRequests,
    RequestVector const& pausedRequests, OrgFairnessStates& orgFairnessStates, LlmRequestState noScheduleUntilState,
    LlmRequestState noScheduleAfterState)
{
    std::unordered_set<LlmRequest::RequestIdType> scheduledRequestIds;
    scheduledRequestIds.reserve(scheduledRequests.size());
    std::unordered_map<std::uint64_t, double> scheduledTokensByOrg;
    std::unordered_map<std::uint32_t, double> scheduledTokensByTier;
    for (auto const& req : scheduledRequests)
    {
        scheduledRequestIds.insert(req->mRequestId);
        auto const scheduledTokens = estimateScheduledTokens(*req);
        scheduledTokensByOrg[getOrgFairnessKey(*req)] += scheduledTokens;
        scheduledTokensByTier[getSchedulerTier(*req)] += scheduledTokens;
    }

    std::unordered_set<LlmRequest::RequestIdType> pausedRequestIds;
    pausedRequestIds.reserve(pausedRequests.size());
    for (auto const& req : pausedRequests)
    {
        pausedRequestIds.insert(req->mRequestId);
    }

    std::unordered_map<std::uint32_t, std::unordered_set<std::uint64_t>> activeOrgKeysByTier;

    for (auto const& req : activeRequests)
    {
        if (!isSchedulableRequest(req, noScheduleUntilState, noScheduleAfterState))
        {
            continue;
        }

        activeOrgKeysByTier[getSchedulerTier(*req)].insert(getOrgFairnessKey(*req));

        auto credit = req->getSchedulerCredit();
        auto pauseCount = req->getSchedulerPauseCount();

        if (pausedRequestIds.find(req->mRequestId) != pausedRequestIds.end())
        {
            req->setSchedulerPauseCount(pauseCount + 1);
            req->setSchedulerCredit(std::min(kSchedulerMaxCredit, credit + kSchedulerWaitCredit + kSchedulerPauseBonus));
            continue;
        }

        if (scheduledRequestIds.find(req->mRequestId) != scheduledRequestIds.end())
        {
            req->setSchedulerPauseCount(pauseCount > 0 ? pauseCount - 1 : 0);
            req->setSchedulerCredit(std::max(0.0, credit - kSchedulerServiceCost));
            continue;
        }

        req->setSchedulerCredit(std::min(kSchedulerMaxCredit, credit + kSchedulerWaitCredit));
    }

    for (auto const& [tier, orgKeys] : activeOrgKeysByTier)
    {
        if (orgKeys.empty())
        {
            continue;
        }

        auto const fairServiceGrant = scheduledTokensByTier[tier] / static_cast<double>(orgKeys.size());
        if (fairServiceGrant <= 0.0)
        {
            continue;
        }

        for (auto const orgKey : orgKeys)
        {
            orgFairnessStates[orgKey] += fairServiceGrant;
        }
    }

    for (auto const& [orgKey, scheduledTokens] : scheduledTokensByOrg)
    {
        auto& tokenBalance = orgFairnessStates[orgKey];
        tokenBalance = std::max(kOrgTokenBalanceMin, tokenBalance - scheduledTokens);
    }
}
} // namespace

MaxRequestsScheduler::MaxRequestsScheduler(
    SizeType32 maxNumRequests, LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
    : BaseCapacityScheduler(noScheduleUntilState, noScheduleAfterState)
    , mMaxNumRequests(maxNumRequests)
{
}

MaxUtilizationScheduler::MaxUtilizationScheduler(SizeType32 maxNumRequests, bool twoStepsLookAhead, bool tierAwareEviction,
    LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
    : BaseCapacityScheduler(noScheduleUntilState, noScheduleAfterState)
    , mMaxNumRequests(maxNumRequests)
    , mTwoStepsLookAhead{twoStepsLookAhead}
    , mTierAwareEviction{tierAwareEviction}
{
}

GuaranteedNoEvictScheduler::GuaranteedNoEvictScheduler(
    SizeType32 maxNumRequests, LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
    : BaseCapacityScheduler(noScheduleUntilState, noScheduleAfterState)
    , mMaxNumRequests(maxNumRequests)
{
}

StaticBatchScheduler::StaticBatchScheduler(
    SizeType32 maxNumRequests, LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
    : GuaranteedNoEvictScheduler(maxNumRequests, noScheduleUntilState, noScheduleAfterState)
{
}

std::tuple<RequestVector, RequestVector> MaxRequestsScheduler::operator()(RequestList const& activeRequests) const
{
    RequestVector scheduledRequests;
    for (auto const& req : activeRequests)
    {
        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (!req->hasReachedState(getNoScheduleUntilState()) || req->hasReachedState(getNoScheduleAfterState()))
        {
            continue;
        }

        if (scheduledRequests.size() >= static_cast<std::size_t>(mMaxNumRequests))
        {
            break;
        }

        if (req->isEncoderInitState() || req->isContextInitState() || req->isGenerationInProgressState())
        {
            scheduledRequests.emplace_back(req);
        }
    }
    return {std::move(scheduledRequests), RequestVector{}};
}

std::tuple<RequestVector, RequestVector> StaticBatchScheduler::operator()(
    kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const
{
    return this->impl<true>(kvCacheManager, crossKvCacheManager, peftCacheManager, activeRequests);
}

std::tuple<RequestVector, RequestVector> GuaranteedNoEvictScheduler::operator()(
    kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const
{
    return impl<false>(kvCacheManager, crossKvCacheManager, peftCacheManager, activeRequests);
}

template <bool StaticBatchScheduling>
std::tuple<RequestVector, RequestVector> GuaranteedNoEvictScheduler::impl(
    kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const
{
    RequestVector scheduledRequests;

    // Now check if we can add pending requests
    auto const maxPeftCachePages
        = peftCacheManager ? peftCacheManager->getMaxDevicePages() : std::numeric_limits<SizeType32>::max();

    // The optimization of delaying requests won't work for variable window attention
    bool skippingIsRelevant = (!kvCacheManager.getBlockManager().isVariableWindow())
        && (!crossKvCacheManager || !crossKvCacheManager->getBlockManager().isVariableWindow());

    // Keep track of blocks contributed by requests in context phase
    std::unordered_set<BlockKey, BlockKeyHasher> newlyContributedContextBlocks;
    std::unordered_set<BlockKey, BlockKeyHasher> newlyContributedCrossContextBlocks;
    if constexpr (!StaticBatchScheduling)
    {
        if (skippingIsRelevant)
        {
            std::tie(newlyContributedContextBlocks, newlyContributedCrossContextBlocks)
                = prefillWithChunkedContextsAlreadyExecuting(activeRequests, kvCacheManager, crossKvCacheManager);
        }
    }

    // If a request is already in progress, include it
    // If it's been allocated, it had resource to run to completion
    // Also keep track of blocks needed to drive all in-progress requests to completion
    auto reservedBlocks = kv_cache_manager::NoEvictScheduledBlocksManager(kvCacheManager);
    auto reservedCrossBlocks = crossKvCacheManager
        ? std::optional(kv_cache_manager::NoEvictScheduledBlocksManager(*crossKvCacheManager))
        : std::nullopt;
    SizeType32 claimedPeftPages{0};
    std::unordered_set<uint64_t> uniqTaskIds{};
    RequestVector pendingRequests;
    RequestVector pendingDisGenInitRequests;
    pendingRequests.reserve(activeRequests.size());
    pendingDisGenInitRequests.reserve(activeRequests.size());
    std::unordered_map<std::uint32_t, SizeType32> scheduledCountsByTier;
    for (auto const& req : activeRequests)
    {
        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (
            // Allow disagg_generation_init requests to be scheduled, so that we'll allocate their KV cache
            !req->isDisaggGenerationInitState()
            && (!req->hasReachedState(getNoScheduleUntilState()) || req->hasReachedState(getNoScheduleAfterState())))
        {
            continue;
        }

        if (scheduledRequests.size() >= static_cast<std::size_t>(mMaxNumRequests))
        {
            break;
        }

        if (req->isGenerationInProgressState())
        {
            scheduledRequests.emplace_back(req);
            ++scheduledCountsByTier[getSchedulerTier(*req)];
            reservedBlocks.decrementReservedBlocks(*req);
            if (reservedCrossBlocks)
                reservedCrossBlocks->decrementReservedBlocks(*req);
            bool const reqHasLora = req->getLoraTaskId().has_value();
            bool const isNewTask = reqHasLora && !uniqTaskIds.count(req->getLoraTaskId().value());
            if (isNewTask)
            {
                claimedPeftPages += peftCacheManager ? peftCacheManager->determineNumPages(req) : 0;
                uniqTaskIds.insert(req->getLoraTaskId().value());
            }
        }
        else if (req->isDisaggGenerationInitState())
        {
            pendingDisGenInitRequests.emplace_back(req);
        }
        else
        {
            pendingRequests.emplace_back(req);
        }
    }

    // If StaticBatchScheduling == true check if we can add pending requests only when no requests are active.
    // Otherwise, add just check that we can add pending requests.
    if (!StaticBatchScheduling || scheduledRequests.size() == 0)
    {
        // Now check if we can add pending requests
        auto availablePeftPages = maxPeftCachePages - claimedPeftPages;
        std::unordered_map<std::uint32_t, SizeType32> pendingBacklogByTier;
        for (auto const& req : pendingDisGenInitRequests)
        {
            ++pendingBacklogByTier[getSchedulerTier(*req)];
        }
        for (auto const& req : pendingRequests)
        {
            ++pendingBacklogByTier[getSchedulerTier(*req)];
        }
        std::vector<std::uint32_t> orderedPendingTiers;
        orderedPendingTiers.reserve(pendingBacklogByTier.size());
        for (auto const& [tier, _count] : pendingBacklogByTier)
        {
            orderedPendingTiers.emplace_back(tier);
        }
        std::sort(orderedPendingTiers.begin(), orderedPendingTiers.end(), std::greater<>());
        auto const tierReservedSlots = allocateReservedTierSlots(orderedPendingTiers, mMaxNumRequests);

        // Loop over pending requests and add them if they can be scheduled
        // Start by trying to include disagg generation init requests
        for (auto const& requests : {pendingDisGenInitRequests, pendingRequests})
        {
            for (auto const& req : requests)
            {
                auto const tier = getSchedulerTier(*req);
                SizeType32 remainingReservedSlots{0};
                for (auto const higherTier : orderedPendingTiers)
                {
                    if (higherTier <= tier)
                    {
                        continue;
                    }

                    auto const pendingBacklog = pendingBacklogByTier[higherTier];
                    auto const scheduledCount = scheduledCountsByTier[higherTier];
                    auto const reservedCount = tierReservedSlots.at(higherTier);
                    if (pendingBacklog > 0 && scheduledCount < reservedCount)
                    {
                        remainingReservedSlots += reservedCount - scheduledCount;
                    }
                }
                auto const futureScheduledCount = static_cast<SizeType32>(scheduledRequests.size() + 1);
                if (remainingReservedSlots > 0
                    && futureScheduledCount > (mMaxNumRequests - remainingReservedSlots))
                {
                    continue;
                }

                // if context request can reuse blocks contributed by another context request, skip
                if (!StaticBatchScheduling && skippingIsRelevant && !req->isDisaggGenerationInitState()
                    && beneficialToSkip(req, kvCacheManager, crossKvCacheManager, newlyContributedContextBlocks,
                        newlyContributedCrossContextBlocks))
                {
                    continue;
                }

                if (scheduledRequests.size() >= static_cast<std::size_t>(mMaxNumRequests))
                {
                    break;
                }

                if (req->isContextInitState() || req->isDisaggGenerationInitState())
                {
                    bool enoughBlocks = reservedBlocks.enoughAvailableBlocks(*req);
                    bool enoughCrossBlocks
                        = reservedCrossBlocks ? reservedCrossBlocks->enoughAvailableBlocks(*req) : true;
                    bool reqHasLora = req->getLoraTaskId().has_value();
                    bool isNewTask = reqHasLora && !uniqTaskIds.count(req->getLoraTaskId().value());
                    auto neededPeftPages = isNewTask && peftCacheManager ? peftCacheManager->determineNumPages(req) : 0;

                    if (enoughBlocks && enoughCrossBlocks && neededPeftPages <= availablePeftPages)
                    {
                        scheduledRequests.emplace_back(req);
                        ++scheduledCountsByTier[tier];
                        pendingBacklogByTier[tier] = std::max<SizeType32>(0, pendingBacklogByTier[tier] - 1);
                        reservedBlocks.decrementReservedBlocks(*req);
                        if (reservedCrossBlocks)
                            reservedCrossBlocks->decrementReservedBlocks(*req);
                        availablePeftPages -= neededPeftPages;
                        if (isNewTask)
                        {
                            uniqTaskIds.insert(req->getLoraTaskId().value());
                        }
                    }
                    else if (!enoughBlocks || !enoughCrossBlocks)
                    {
                        // If one requests fails to be scheduled, break
                        break;
                    }
                }
            }
        }
    }
    return {std::move(scheduledRequests), RequestVector{}};
}

// TODO(nhaber): remove forward declare and just keep the function here, right before the merge. I put it below just so
// the remote diff is easier to look at/rebase conflicts
bool trySchedulingRequestMaxUtilization(std::shared_ptr<LlmRequest> const& req, SizeType32 maxNumRequests,
    RequestVector& scheduledRequests, kv_cache_manager::MaxUtilizationScheduledBlocksManager& blocksManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, SizeType32& numScheduledPeftPages,
    std::unordered_set<uint64_t>& seenTaskIds);

std::tuple<RequestVector, RequestVector> MaxUtilizationScheduler::operator()(
    kv_cache_manager::BaseKVCacheManager& kvCacheManager, OptionalRef<BasePeftCacheManager const> peftCacheManager,
    RequestList const& activeRequests) const
{
    kvCacheManager.startScheduling();

    // The optimization of delaying requests won't work for variable window attention
    bool skippingIsRelevant = !kvCacheManager.getBlockManager().isVariableWindow();

    // Keep track of number of requests and block needed for the scheduled requests
    auto scheduledBlocksManager
        = kv_cache_manager::MaxUtilizationScheduledBlocksManager(kvCacheManager, mTwoStepsLookAhead);
    SizeType32 numScheduledPeftPages{0};
    std::unordered_set<uint64_t> seenTaskIds;

    // Keep track of blocks contributed by requests in context phase
    auto [newlyContributedContextBlocks, newlyContributedCrossContextBlocks]
        = prefillWithChunkedContextsAlreadyExecuting(activeRequests, kvCacheManager);

    // Find last active in case we need to evict
    auto startedReqLambda = [this](std::shared_ptr<LlmRequest> const& req)
    {
        return (req->hasReachedState(getNoScheduleUntilState()) && !req->hasReachedState(getNoScheduleAfterState())
            && ((req->isContextInitState() && !req->isFirstContextChunk()) || req->isGenerationInProgressState()));
    };

    RequestVector scheduledRequests;
    RequestVector pausedRequests;
    auto reqItEnd = std::end(activeRequests);
    for (auto reqIt = std::begin(activeRequests); reqIt != reqItEnd;)
    {
        auto const& req = *reqIt;
        TLLM_LOG_DEBUG("MaxUtilizationScheduler: scheduling request ID %lu", req->mRequestId);

        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (
            // Allow disagg_generation_init requests to be scheduled, so that we'll allocate their KV cache
            !req->isDisaggGenerationInitState()
            && (!req->hasReachedState(getNoScheduleUntilState()) || req->hasReachedState(getNoScheduleAfterState())))
        {
            TLLM_LOG_DEBUG("MaxUtilizationScheduler: request ID %lu cannot / should not be scheduled", req->mRequestId);
            reqIt++;
            continue;
        }

        // if context request can reuse blocks contributed by another context request, skip
        if (skippingIsRelevant
            && beneficialToSkip(
                req, kvCacheManager, std::nullopt, newlyContributedContextBlocks, newlyContributedCrossContextBlocks))
        {
            reqIt++;
            continue;
        }

        bool const wasScheduled = trySchedulingRequestMaxUtilization(req, mMaxNumRequests, scheduledRequests,
            scheduledBlocksManager, peftCacheManager, numScheduledPeftPages, seenTaskIds);
        if (wasScheduled)
        {
            TLLM_LOG_DEBUG("MaxUtilizationScheduler: request ID %lu -> start", req->mRequestId);
            reqIt++;
        }
        else
        {
            auto pauseVictimIt = reqItEnd;
            if (!mTierAwareEviction)
            {
                auto const rbegin = std::reverse_iterator(reqItEnd);
                auto const rend = std::reverse_iterator(reqIt);
                auto const lastStartedReqIt = std::find_if(rbegin, rend, startedReqLambda);
                if (lastStartedReqIt != rend)
                {
                    pauseVictimIt = std::prev(lastStartedReqIt.base());
                }
            }
            else
            {
                auto const currentTier = getSchedulerTier(*req);
                bool foundVictim = false;
                std::uint32_t victimTier = 0;
                double victimScore = 0.0;
                for (auto candidateIt = reqIt; candidateIt != reqItEnd; ++candidateIt)
                {
                    auto const& candidate = *candidateIt;
                    if (!startedReqLambda(candidate))
                    {
                        continue;
                    }

                    auto const candidateTier = getSchedulerTier(*candidate);
                    if (candidateTier > currentTier)
                    {
                        continue;
                    }

                    auto const candidateScore = getVictimScore(candidate);
                    if (!foundVictim || candidateTier < victimTier
                        || (candidateTier == victimTier && candidateScore < victimScore)
                        || (candidateTier == victimTier && candidateScore == victimScore))
                    {
                        pauseVictimIt = candidateIt;
                        victimTier = candidateTier;
                        victimScore = candidateScore;
                        foundVictim = true;
                    }
                }
            }

            if (pauseVictimIt != reqItEnd)
            {
                // If we can't allocate a request, free a started request and retry.
                kvCacheManager.schedulingRemoveSequence((*pauseVictimIt)->mRequestId);
                pausedRequests.emplace_back(*pauseVictimIt);
                TLLM_LOG_DEBUG("MaxUtilizationScheduler: request ID %lu -> pause", (*pauseVictimIt)->mRequestId);
                reqItEnd = pauseVictimIt;
            }
            else
            {
                break;
            }
        }
    }

    return {std::move(scheduledRequests), std::move(pausedRequests)};
}

bool trySchedulingRequestMaxUtilization(std::shared_ptr<LlmRequest> const& req, SizeType32 maxNumRequests,
    RequestVector& scheduledRequests, kv_cache_manager::MaxUtilizationScheduledBlocksManager& blocksManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, SizeType32& numScheduledPeftPages,
    std::unordered_set<uint64_t>& seenTaskIds)
{
    if (scheduledRequests.size() < static_cast<std::size_t>(maxNumRequests))
    {
        bool reqHasLora = req->getLoraTaskId().has_value();
        bool isNewTask = reqHasLora && !seenTaskIds.count(req->getLoraTaskId().value());
        SizeType32 numRequiredPeftPages
            = (isNewTask && peftCacheManager) ? peftCacheManager->determineNumPages(req) : 0;
        TLLM_LOG_DEBUG(
            "MaxUtilizationScheduler: request ID %lu required peft pages: %i", req->mRequestId, numRequiredPeftPages);
        auto const scheduledBlocksIfFitsKvCache = blocksManager.prepareNewNumberOfBlocksIfWeEndUpScheduling(*req);
        bool fitsPeft
            = (peftCacheManager ? numRequiredPeftPages + numScheduledPeftPages <= peftCacheManager->getMaxDevicePages()
                                : true);

        if (scheduledBlocksIfFitsKvCache && fitsPeft)
        {
            blocksManager.updateScheduledBlocks(scheduledBlocksIfFitsKvCache.value());
            numScheduledPeftPages += numRequiredPeftPages;
            TLLM_LOG_DEBUG("MaxUtilizationScheduler: scheduled peft pages: %i", numRequiredPeftPages);
            scheduledRequests.emplace_back(req);
            if (isNewTask)
            {
                seenTaskIds.insert(req->getLoraTaskId().value());
            }
            return true;
        }
    }
    return false;
}

CapacityScheduler::CapacityScheduler(SizeType32 maxNumRequests,
    executor::CapacitySchedulerPolicy capacitySchedulerPolicy, bool hasKvCacheManager, bool twoStepsLookAhead,
    LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
    : mMaxNumRequests(maxNumRequests)
{
    if (!hasKvCacheManager)
    {
        mScheduler = MaxRequestsScheduler{maxNumRequests, noScheduleUntilState, noScheduleAfterState};
    }
    else if (capacitySchedulerPolicy == executor::CapacitySchedulerPolicy::kMAX_UTILIZATION)
    {
        mScheduler
            = MaxUtilizationScheduler{
                maxNumRequests, twoStepsLookAhead, false, noScheduleUntilState, noScheduleAfterState};
    }
    else if (capacitySchedulerPolicy == executor::CapacitySchedulerPolicy::kTIER_AWARE_MAX_UTILIZATION)
    {
        mScheduler
            = MaxUtilizationScheduler{
                maxNumRequests, twoStepsLookAhead, true, noScheduleUntilState, noScheduleAfterState};
    }
    else if (capacitySchedulerPolicy == executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT)
    {
        mScheduler = GuaranteedNoEvictScheduler{maxNumRequests, noScheduleUntilState, noScheduleAfterState};
    }
    else if (capacitySchedulerPolicy == executor::CapacitySchedulerPolicy::kSTATIC_BATCH)
    {
        mScheduler = StaticBatchScheduler{maxNumRequests, noScheduleUntilState, noScheduleAfterState};
    }
    else
    {
        throw std::runtime_error("Unsupported capacity scheduler policy");
    }
}

std::tuple<RequestVector, RequestVector, RequestVector> CapacityScheduler::operator()(RequestList const& activeRequests,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> kvCacheManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager) const
{
    NVTX3_SCOPED_RANGE(capacitySchedulerScheduling);
    return std::visit(
        [this, &activeRequests, &kvCacheManager, &crossKvCacheManager, &peftCacheManager](
            auto const& scheduler) -> std::tuple<RequestVector, RequestVector, RequestVector>
        {
            RequestVector tmpFittingRequests;
            RequestVector pausedRequests;
            RequestList orderedActiveRequests(activeRequests.begin(), activeRequests.end());
            auto const fairnessEnabled = hasSchedulerControlsEnabled(orderedActiveRequests);

            if constexpr (std::is_same_v<std::decay_t<decltype(scheduler)>, MaxRequestsScheduler>)
            {
                if (fairnessEnabled)
                {
                    updateOrgFairnessStates(mSchedulerOrgFairnessStates, orderedActiveRequests,
                        scheduler.getNoScheduleUntilState(), scheduler.getNoScheduleAfterState());
                    sortActiveRequestsByFairness(orderedActiveRequests, mSchedulerOrgFairnessStates,
                        mMaxNumRequests, scheduler.getNoScheduleUntilState(), scheduler.getNoScheduleAfterState());
                }
                std::tie(tmpFittingRequests, pausedRequests) = scheduler(orderedActiveRequests);
                if (fairnessEnabled)
                {
                    updateRequestFairnessState(orderedActiveRequests, tmpFittingRequests, pausedRequests,
                        mSchedulerOrgFairnessStates,
                        scheduler.getNoScheduleUntilState(), scheduler.getNoScheduleAfterState());
                }
            }
            else if constexpr (std::is_same_v<std::decay_t<decltype(scheduler)>, MaxUtilizationScheduler>)
            {
                if (fairnessEnabled)
                {
                    updateOrgFairnessStates(mSchedulerOrgFairnessStates, orderedActiveRequests,
                        scheduler.getNoScheduleUntilState(), scheduler.getNoScheduleAfterState());
                    sortActiveRequestsByFairness(orderedActiveRequests, mSchedulerOrgFairnessStates,
                        mMaxNumRequests, scheduler.getNoScheduleUntilState(), scheduler.getNoScheduleAfterState());
                }
                std::tie(tmpFittingRequests, pausedRequests)
                    = scheduler(*kvCacheManager, peftCacheManager, orderedActiveRequests);
                if (fairnessEnabled)
                {
                    updateRequestFairnessState(orderedActiveRequests, tmpFittingRequests, pausedRequests,
                        mSchedulerOrgFairnessStates,
                        scheduler.getNoScheduleUntilState(), scheduler.getNoScheduleAfterState());
                }
            }
            else if constexpr (std::is_same_v<std::decay_t<decltype(scheduler)>, GuaranteedNoEvictScheduler>
                || std::is_same_v<std::decay_t<decltype(scheduler)>, StaticBatchScheduler>)
            {
                if (fairnessEnabled)
                {
                    updateOrgFairnessStates(mSchedulerOrgFairnessStates, orderedActiveRequests,
                        scheduler.getNoScheduleUntilState(), scheduler.getNoScheduleAfterState());
                    sortActiveRequestsByFairness(orderedActiveRequests, mSchedulerOrgFairnessStates,
                        mMaxNumRequests, scheduler.getNoScheduleUntilState(), scheduler.getNoScheduleAfterState());
                }
                std::tie(tmpFittingRequests, pausedRequests)
                    = scheduler(*kvCacheManager, crossKvCacheManager, peftCacheManager, orderedActiveRequests);
                if (fairnessEnabled)
                {
                    updateRequestFairnessState(orderedActiveRequests, tmpFittingRequests, pausedRequests,
                        mSchedulerOrgFairnessStates,
                        scheduler.getNoScheduleUntilState(), scheduler.getNoScheduleAfterState());
                }
            }
            else
            {
                throw std::runtime_error("Unsupported capacity scheduler policy");
            }
            TLLM_LOG_DEBUG("[Summary] Capacity scheduler allows %d requests, pauses %d requests",
                tmpFittingRequests.size(), pausedRequests.size());

            RequestVector fittingRequests;
            RequestVector fittingDisaggGenInitRequests;
            for (auto const& llmReq : tmpFittingRequests)
            {
                if (llmReq->isDisaggGenerationInitState())
                {
                    fittingDisaggGenInitRequests.push_back(llmReq);
                }
                else
                {
                    fittingRequests.push_back(llmReq);
                }
            }

            return {std::move(fittingRequests), std::move(fittingDisaggGenInitRequests), std::move(pausedRequests)};
        },
        mScheduler);
}

} // namespace tensorrt_llm::batch_manager
