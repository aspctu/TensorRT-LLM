import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import torch

from .interface import SpecConfig, SpeculativeDecodingMode
from .ngram import NGramPoolManager, NGramConfig
from .eagle3 import Eagle3Config, Eagle3OneModelWorker, Eagle3OneModelSpecMetadata

_MAX_NUM_REQUESTS = 128
_MAX_SEQUENCE_LENGTH = 2048

@dataclass
class HybridSpecConfig(SpecConfig):
    eagle_config: Optional[Eagle3Config] = None
    ngram_config: Optional[NGramConfig] = None
    draft_model_path: Optional[str] = None
    spec_dec_name: str = "HYBRID"
    max_eagle_potential_drafts: int = 3
    max_ngram_potential_drafts: int = 5
    max_draft_tokens: int = 8

    def __post_init__(self):
        if self.eagle_config is None:
            raise ValueError("HybridSpecConfig requires eagle_config to be set.")
        if self.ngram_config is None:
            raise ValueError("HybridSpecConfig requires ngram_config to be set.")

        self.draft_model_path = self.eagle_config.draft_model_path

        self.num_extra_kv_tokens = self.max_draft_tokens - 1 if self.max_draft_tokens > 0 else 0
        self.spec_dec_mode = SpeculativeDecodingMode.HYBRID

        self.eagle_config.max_draft_tokens = self.max_eagle_potential_drafts
        self.eagle_config.__post_init__()
        self.ngram_config.__post_init__()

    def update_from_model_config(self, model_config):
        self.eagle_config.update_from_model_config(model_config)
        self.ngram_config.update_from_model_config(model_config)
    
    def get_draft_model_prompt(self, prompt_tokens: List[int]) -> List[int]:
        return self.eagle_config.get_draft_model_prompt(prompt_tokens)

def merge_eagle_and_ngram(
        eagle: list[int] | None,
        ngram: list[int] | None,
        max_len: int,
        stop_on_first_mismatch: bool = True
    ) -> list[int]:
    print(f"Proposed Eagle draft: {eagle}")
    print(f"Proposed NGram draft: {ngram}")
    if not eagle:
        draft = ngram[:max_len] if ngram else []
        print(f"Returning the following as draft: {draft}") 
    if not ngram:
        draft = eagle[:max_len]
        print(f"Returning the following as draft: {draft}")

    if stop_on_first_mismatch:
        merged: list[int] = []
        for i in range(min(len(eagle), len(ngram))):
            if eagle[i] == ngram[i]:
                merged.append(eagle[i])
                if len(merged) == max_len:
                    break
            else:
                break
        return merged[:max_len]

class HybridSpeculativeDecodingWorker(Eagle3OneModelWorker):
    def __init__(self, spec_config: HybridSpecConfig, mapping):
        super().__init__(spec_config, mapping)
        self.spec_config = spec_config
        self._ngram_pool: NGramPoolManager | None = NGramPoolManager(
            spec_config.ngram_config,
            _MAX_NUM_REQUESTS
        )
        self._max_seq_len = _MAX_SEQUENCE_LENGTH
        self.eos_token_id = 128001 # TODO(Abu): Plumb this through correctly

    def forward(self, input_ids, position_ids, hidden_states, logits,
                attn_metadata, spec_metadata, draft_model):

        eagle_outputs = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model
        )

        eagle_draft_tokens_tensor = eagle_outputs['next_draft_tokens']
        batch_size = attn_metadata.num_seqs

        all_merged_drafts_list = []

        for b_idx in range(batch_size):
            num_accepted_in_verification = eagle_outputs['new_tokens_lens'][b_idx].item()
            prefix_for_ngram = eagle_outputs['new_tokens'][b_idx, :num_accepted_in_verification].tolist()
            end_id_for_ngram = self.eos_token_id

            ngram_candidates_for_req = None
            if self._ngram_pool is not None:
                ngram_candidates_for_req = self._ngram_pool._get_draft_tokens(
                    prefix=prefix_for_ngram,
                    request_id=spec_metadata.request_ids[b_idx],
                    end_id=end_id_for_ngram,
                    max_sequence_length=self._max_seq_len,
                )
                print(f"V1 Ngram draft: {ngram_candidates_for_req}")

            if ngram_candidates_for_req:
                ngram_candidates_for_req = [t for t in ngram_candidates_for_req if t != end_id_for_ngram]
                ngram_candidates_for_req = ngram_candidates_for_req[:self.spec_config.max_ngram_potential_drafts]
                print(f"V2 Ngram draft: {ngram_candidates_for_req}")

            current_eagle_draft_list = eagle_draft_tokens_tensor[b_idx].tolist()

            merged_for_req = merge_eagle_and_ngram(
                current_eagle_draft_list,
                ngram_candidates_for_req,
                max_len=self.spec_config.ngram_config.max_draft_tokens,
                stop_on_first_mismatch=True
            )
            all_merged_drafts_list.append(merged_for_req)

        device = eagle_draft_tokens_tensor.device
        K_h = self.spec_config.max_draft_tokens
        pad_token_id_for_draft = self.eos_token_id

        padded_merged_drafts_tensor = torch.full(
            (batch_size, K_h),
            fill_value=pad_token_id_for_draft,
            dtype=torch.int32,
            device=device,
        )

        for b_idx, merged_list in enumerate(all_merged_drafts_list):
            if merged_list:
                len_merged = len(merged_list)
                actual_len_to_copy = min(len_merged, K_h)
                padded_merged_drafts_tensor[b_idx, :actual_len_to_copy] = torch.tensor(
                    merged_list[:actual_len_to_copy], dtype=torch.int32, device=device
                )
        
        final_outputs = eagle_outputs.copy()
        final_outputs['next_draft_tokens'] = padded_merged_drafts_tensor
        
        if spec_metadata.batch_indices_cuda is not None and len(spec_metadata.batch_indices_cuda) >= batch_size:
            current_batch_indices = spec_metadata.batch_indices_cuda[:batch_size]
        else:
            current_batch_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

        last_accepted_token_indices = final_outputs['new_tokens_lens'] - 1
        last_accepted_token_indices = torch.clamp(last_accepted_token_indices, min=0)
        gathered_last_tokens = final_outputs['new_tokens'][
            current_batch_indices,
            last_accepted_token_indices
        ].unsqueeze(1)

        final_outputs['next_new_tokens'] = torch.cat(
            [gathered_last_tokens, padded_merged_drafts_tensor],
            dim=1
        )

        return final_outputs