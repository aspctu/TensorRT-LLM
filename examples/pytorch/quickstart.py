from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.llmapi import (EagleDecodingConfig, NGramDecodingConfig, KvCacheConfig) # HybridDecodingConfig
from transformers import AutoTokenizer

MODEL_NAME = "nvidia/Llama-3.3-70B-Instruct-FP8" # "meta-llama/Meta-Llama-3-8B-Instruct"
EAGLE_PATH = "/workspace/eagle-llama3-70b" # "workspace/eagle"

def main():
    # with open("code.txt", "r") as f:
    #    prompts = f.read()
    # prompts = [f"Here's some code : {prompts}\n What does the HybridDecodingConfig take in as arguments?"] 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    prompts = ["Hi, my name is"]

    sampling_params = SamplingParams(max_tokens=1028, end_id=128001)

    eagle_config = EagleDecodingConfig(
        max_draft_len=32,
        pytorch_eagle_weights_path=EAGLE_PATH,
    )

    ngram_config = NGramDecodingConfig(
        prompt_lookup_num_tokens=32,
        max_matching_ngram_size=32,
        is_keep_all=True,
        is_use_oldest=True,
        is_public_pool=True,
    )

    # spec_config = HybridDecodingConfig(
    #     eagle_config=eagle_config,
    #     ngram_config=ngram_config,
    #     max_eagle_potential_drafts=2,
    #     max_ngram_potential_drafts=32,
    # )
    # kv_cache_config = KvCacheConfig(enable_block_reuse=False)

    llm = LLM(
        model=MODEL_NAME,
        backend="pytorch",
        tensor_parallel_size=2,
        eagle_config=eagle_config,
        # speculative_config=spec_config,
        # kv_cache_config=kv_cache_config,
    )
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    main()
