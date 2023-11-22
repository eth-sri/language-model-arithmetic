import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import os
from loguru import logger
import json
from peft import PeftModel
from trl import AutoModelForCausalLMWithValueHead
from .utils import log
try:
    from auto_gptq import AutoGPTQForCausalLM
except ImportError:
    from transformers import AutoModelForCausalLM as AutoGPTQForCausalLM
    log(logger.warning, "Failed to import auto_gptq")

def load_tokenizer(dir_or_model):
    """
    This function is used to load the tokenizer for a specific pre-trained model.
    
    Args:
        dir_or_model: It can be either a directory containing the pre-training model configuration details or a pretrained model.
    
    Returns:
        It returns a tokenizer that can convert text to tokens for the specific model input.
    """
    log(logger.debug, f"Loading tokenizer for {dir_or_model}")

    is_lora_dir = os.path.isfile(os.path.join(dir_or_model, "adapter_config.json"))

    if is_lora_dir:
        loaded_json = json.load(open(os.path.join(dir_or_model, "adapter_config.json"), "r"))
        model_name = loaded_json["base_model_name_or_path"]
    else:
        model_name = dir_or_model
        
    if os.path.isfile(os.path.join(dir_or_model, "config.json")):
        loaded_json = json.load(open(os.path.join(dir_or_model, "config.json"), "r"))
        model_name = loaded_json["_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        log(logger.debug, "Setting pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer

def load_model(dir_or_model, classification=False, token_classification=False, return_tokenizer=False, dtype=torch.bfloat16, load_dtype=True, 
                rl=False, peft_config=None):
    """
    This function is used to load a model based on several parameters including the type of task it is targeted to perform.
    
    Args:
        dir_or_model: It can be either a directory containing the pre-training model configuration details or a pretrained model.

        classification (bool): If True, loads the model for sequence classification.

        token_classification (bool): If True, loads the model for token classification.

        return_tokenizer (bool): If True, returns the tokenizer along with the model.

        dtype: The data type that PyTorch should use internally to store the model’s parameters and do the computation.

        load_dtype (bool): If False, sets dtype as torch.float32 regardless of the passed dtype value.

        rl (bool): If True, loads model specifically designed to be used in reinforcement learning environment.

        peft_config: Configuration details for Peft models. 
    
    Returns:
        It returns a model for the required task along with its tokenizer, if specified.
    """
    log(logger.debug, f"Loading model for {dir_or_model} with {classification}, {dtype}, {load_dtype}")
    is_lora_dir = os.path.isfile(os.path.join(dir_or_model, "adapter_config.json"))

    if not load_dtype:
        dtype = torch.float32

    if is_lora_dir:
        loaded_json = json.load(open(os.path.join(dir_or_model, "adapter_config.json"), "r"))
        model_name = loaded_json["base_model_name_or_path"]
    else:
        model_name = dir_or_model

    original_model_name = model_name

    if classification:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype, use_auth_token=True, device_map="auto")  # to investigate: calling torch_dtype here fails.
    elif token_classification:
        model = AutoModelForTokenClassification.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype, use_auth_token=True, device_map="auto")
    elif rl:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype, use_auth_token=True, 
                                                                  peft_config=peft_config, device_map="auto")
    else:
        if model_name.endswith("GPTQ") or model_name.endswith("GGML"):
            model = AutoGPTQForCausalLM.from_quantized(model_name,
                                                        use_safetensors=True,
                                                        trust_remote_code=True,
                                                        # use_triton=True, # breaks currently, unfortunately generation time of the GPTQ model is quite slow
                                                        quantize_config=None, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype, use_auth_token=True, device_map="auto")

    if is_lora_dir:
        model = PeftModel.from_pretrained(model, dir_or_model)
        
    try:
        tokenizer = load_tokenizer(original_model_name)
        model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    if return_tokenizer:
        return model, load_tokenizer(original_model_name)
    return model
