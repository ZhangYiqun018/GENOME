import os
import glob
from typing import List

import time
from peft import LoraConfig
from transformers import AutoTokenizer
from loguru import logger
from safetensors.torch import load_file, save_file

def get_base_url(ports: List[int]) -> List[str]:
    """Build base URLs from ports, honoring optional VLLM_HOST env override."""
    host = os.getenv("VLLM_HOST")
    if not host:
        host = "localhost"
        logger.warning("VLLM_HOST not set; defaulting base URLs to localhost")
    base_url = [f"http://{host}:{port}/v1" for port in ports]
    logger.info(f"BASE URL: {base_url}")
    
    return base_url

def get_lora_pools(lora_dir: str) -> List[str]:
    """do not change this function, it is used for Model Swarms / Genome / GenomePlus."""
    lora_pools = []
    for dirname in [
        "code_alpaca", "gpt4_alpaca", "cot", "lima", "oasst1", "open_orca", "flan_v2", "science_literature", "wizardlm", "sharegpt",
    ]:
        path = os.path.join(lora_dir, dirname)
        if os.path.exists(path):
            lora_pools.append(path)
        
    return lora_pools

def load_lora_weight(lora_path: str) -> dict:
    """Load LoRA weights from path."""
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weight not found at {lora_path}")
    return load_file(os.path.join(lora_path, "adapter_model.safetensors"))

def save_lora_weight(lora_weight, lora_path: str, tokenizer: AutoTokenizer | str, config: LoraConfig | str):
    assert tokenizer is not None, "Tokenizer must be provided (for vllm evaluate)."
    assert config is not None, "LoraConfig must be provided."
    
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    if isinstance(config, str):
        config = LoraConfig.from_pretrained(config)
        
    tokenizer.save_pretrained(lora_path)
    config.save_pretrained(lora_path)

    save_file(lora_weight, filename=os.path.join(lora_path, "adapter_model.safetensors"))
    # wait for save completed.
    time.sleep(1)
    
def get_gemma_prompt(user_question):
    template = f"""
    <start_of_turn>user
    {user_question}
    <end_of_turn>
    <start_of_turn>model
    """
    return template

def get_llama3_1_prompt(user_question):
    template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>\n\n{user_question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n"""
    return template