import torch
from model_arithmetic import ModelArithmetic, Evaluation, PromptedLLM, Superseded, Autocomplete, enable_logging, Union
from transformers import set_seed
import pandas as pd
import os
import numpy as np

set_seed(42)

enable_logging()

def evaluate_formula(formula, do_non_spec, dataset, default_model, folder,
                    batch_size=4, temperature=1, top_p=1, top_k=0, model_name_fluency="meta-llama/Llama-2-7b-chat-hf", 
                    dtype=torch.bfloat16, preserve_memory=True, classifier_name="SkolkovoInstitute/roberta_toxicity_classifier", classification_with_input=False, 
                    dtype_faithfulness=torch.bfloat16, finetune_model=False, batch_size_faithfulness=8,
                    reload=False, reload_data=False, max_tokens=128, save_monitor=False):
    """
    Evaluates a formula using the specified parameters.

    Parameters:
    - formula (str): The formula to evaluate.
    - do_non_spec (bool): Whether to perform non-speculative evaluation along with the speculative evaluation.
    - dataset (str): The dataset to use for evaluation.
    - default_model (str): The default model to use for evaluation.
    - folder (str): The folder to store the evaluation results.
    - batch_size (int): The batch size for evaluation. Default is 4.
    - temperature (int): The temperature for sampling. Default is 1.
    - top_p (int): The top-p value for sampling. Default is 1.
    - top_k (int): The top-k value for sampling. Default is 0.
    - model_name_fluency (str): The model name for fluency evaluation. Default is "meta-llama/Llama-2-7b-chat-hf".
    - dtype (torch.dtype): The data type for evaluation. Default is torch.bfloat16.
    - preserve_memory (bool): Whether to preserve memory during evaluation. Default is True.
    - classifier_name (str): The name of the classifier model. Default is "SkolkovoInstitute/roberta_toxicity_classifier".
    - classification_with_input (bool): Whether to include input in classification. Default is False.
    - dtype_faithfulness (torch.dtype): The data type for faithfulness evaluation. Default is torch.bfloat16.
    - finetune_model (bool): Whether to finetune the model during evaluation. Default is False.
    - batch_size_faithfulness (int): The batch size for faithfulness evaluation. Default is 8.
    - reload (bool): Whether to reload the model during evaluation. Default is False.
    - reload_data (bool): Whether to reload the data during evaluation. Default is False.
    - max_tokens (int): The maximum number of tokens for evaluation. Default is 128.
    - save_monitor (bool): Whether to save the monitor file. Default is False.
    """
    arithmetic = ModelArithmetic(formula, default_model=default_model)
    formula_file = os.path.join(folder, "formula.txt")
    os.makedirs(os.path.dirname(formula_file), exist_ok=True)
    with open(formula_file, 'w') as outfile:
        outfile.write(str(formula))
        
    kwargs = {
        "batch_size":batch_size,
        "temperature":temperature,
        "top_p":top_p,
        "top_k":top_k,
        "model_name_fluency":model_name_fluency,
        "dtype":dtype,
        "preserve_memory":preserve_memory,
        "model_name":classifier_name,
        "classification_with_input":classification_with_input,
        "dtype_faithfulness":dtype_faithfulness,
        "finetune_model":finetune_model,
        "batch_size_faithfulness":batch_size_faithfulness,
        "reload":reload,
        "reload_data":reload_data,
        "do_perspective":False,
        "max_tokens":max_tokens
    }

    if do_non_spec:
        evaluator = Evaluation(arithmetic, dataset=dataset)

        evaluator.evaluate(
            **kwargs, 
            speculation=False,
            store_file=os.path.join(folder, "evaluation.json"),
            dataset_file=os.path.join(folder, "data.csv"),
        )
        store_file_monitor = os.path.join(folder, "monitor.json")
        
        if save_monitor:
            arithmetic.monitor.save(store_file_monitor)
        else:
            arithmetic.monitor.store(store_file_monitor)
        
        del arithmetic
        del evaluator
        torch.cuda.empty_cache()
        
    arithmetic = ModelArithmetic(formula, default_model=default_model)
    evaluator = Evaluation(arithmetic, dataset=dataset)
        
    evaluator.evaluate(
        **kwargs, 
        speculation=True,
        store_file=os.path.join(folder + "_spec", "evaluation.json"),
        dataset_file=os.path.join(folder + "_spec", "data.csv"),
    )
    store_file_monitor = os.path.join(folder + "_spec", "monitor.json")
    
    if save_monitor:
        arithmetic.monitor.save(store_file_monitor)
    else:
        arithmetic.monitor.store(store_file_monitor)

prompt_template = lambda formula_string, input_string: f"<s>[INST]<<SYS>>\n{formula_string}\n<</SYS>>\n\n{input_string} [/INST]"
normal = "You are a helpful assistant"
formal = "You are an assistant using formal and objective language to answer the user."
happy = "You are a happy assistant."
emotions = lambda emotion: f"You are {emotion} assistant."
easy = "You are a helpful assistant using very simple and straightforward language with short words to answer the user."
topical = lambda topic: f"You are a helpful assistant that answers the user in a way that is related to {topic}."

normal_prompt = lambda k=1, group=None, model_name=None: PromptedLLM(normal, speculative_factor=k, prompt_template=prompt_template, group=group, model=model_name)
formal_prompt = lambda k=1, group=None, model_name=None: PromptedLLM(formal, speculative_factor=k, prompt_template=prompt_template, group=group, model=model_name)
happy_prompt = lambda k=1, group=None, model_name=None: PromptedLLM(happy, speculative_factor=k, prompt_template=prompt_template, group=group, model=model_name)
emotion_prompts = {
    emotion: lambda k=1, group=None, model_name=None, e=emotion: PromptedLLM(emotions(e), speculative_factor=k, prompt_template=prompt_template, group=group, model=model_name)
    for emotion in ["an angry", "a sad", "a caring", "a loving"]
}
easy_prompt = lambda k=1, group=None, model_name=None: PromptedLLM(easy, speculative_factor=k, prompt_template=prompt_template, group=group, model=model_name)
topical_prompts = {
    topic: lambda k=1, group=None, model_name=None, t=topic: PromptedLLM(topical(t), speculative_factor=k, prompt_template=prompt_template, group=group, model=model_name)
    for topic in ["arts and culture", "business and entrepreneurs", "celebrity and pop culture", "diaries and daily life", "family", 
                  "fashion and style", "film, tv and video", "fitness & health", "food", "gaming", "learning",
                  "music", "news and social concern", "other hobbies", "relationships", "science and technology", "sports", "travel and adventure", 
                  "youth and student life"]
}

data = pd.read_csv("data/datasets/alpaca_processed.csv")
samples = data[:10000]["output"].tolist()
auto = Autocomplete(samples, store_corpus=False, model=f"meta-llama/Llama-2-13b-chat-hf")

formulas = [
    (Superseded(auto, normal_prompt(k=3)), True),
    (normal_prompt() + 0.2 * formal_prompt(k=10), True),
    (normal_prompt() + 0.4 * formal_prompt(k=7), False),
    (normal_prompt() + 0.6 * formal_prompt(k=6), False),
    (normal_prompt() + 0.8 * formal_prompt(k=5), False),
    (normal_prompt() + 1.0 * formal_prompt(k=5), False),
    (normal_prompt() + 0.2 * happy_prompt(k=10), True),
    (normal_prompt() + 0.4 * happy_prompt(k=8), False),
    (normal_prompt() + 0.6 * happy_prompt(k=6), False),
    (normal_prompt() + 0.8 * happy_prompt(k=6), False),
    (normal_prompt() + 1.0 * happy_prompt(k=5), False),
    (normal_prompt() + 0.2 * easy_prompt(k=9), True),
    (normal_prompt() + 0.4 * easy_prompt(k=7), False),
    (normal_prompt() + 0.6 * easy_prompt(k=6), False),
    (normal_prompt() + 0.8 * easy_prompt(k=5), False),
    (normal_prompt() + 1.0 * easy_prompt(k=5), False),
    (normal_prompt() + 0.2 * topical_prompts["sports"](k=10), True),
    (normal_prompt() + 0.4 * topical_prompts["sports"](k=6), False),
    (normal_prompt() + 0.6 * topical_prompts["sports"](k=5), False),
    (normal_prompt() + 0.8 * topical_prompts["sports"](k=5), False),
    (normal_prompt() + 1.0 * topical_prompts["sports"](k=4), False),
    (normal_prompt() + 0.2 * emotion_prompts["an angry"](k=8), True),
    (normal_prompt() + 0.4 * emotion_prompts["an angry"](k=5), False),
    (normal_prompt() + 0.6 * emotion_prompts["an angry"](k=4), False),
    (normal_prompt() + 0.8 * emotion_prompts["an angry"](k=3), False),
    (normal_prompt() + 1.0 * emotion_prompts["an angry"](k=3), False),
    (Superseded(auto, normal_prompt(k=3)) + 0.5 * formal_prompt(k=9), True),
    (Superseded(auto, normal_prompt(k=3)) + 0.5 * happy_prompt(k=10), True),
    (Superseded(auto, normal_prompt(k=3)) + 0.5 * easy_prompt(k=9), True),
    (Superseded(auto, normal_prompt(k=3)) + 0.5 * topical_prompts["sports"](k=8), True),
    (Superseded(auto, normal_prompt(k=3)) + 0.5 * emotion_prompts["an angry"](k=7), True),
    (Superseded(auto, normal_prompt(k=3)) + 0.2 * formal_prompt(k=14) + 0.5 * happy_prompt(k=10) + 0.05 * topical_prompts["sports"](k=30), True),
    (Superseded(auto, normal_prompt(k=3)) + 1.0 * formal_prompt(k=6) + 0.1 * emotion_prompts["an angry"](k=23) + 0.4 * topical_prompts["sports"](k=12), True),
    (normal_prompt() + 0.1 * formal_prompt(k=15), False),
    (normal_prompt() + 0.3 * formal_prompt(k=9), False),
    (normal_prompt() + 0.5 * formal_prompt(k=8), False),
    (normal_prompt() + 0.7 * formal_prompt(k=7), False),
    (normal_prompt() + 0.9 * formal_prompt(k=6), False),
    (normal_prompt() + 0.1 * happy_prompt(k=18), False),
    (normal_prompt() + 0.3 * happy_prompt(k=10), False),
    (normal_prompt() + 0.5 * happy_prompt(k=8), False),
    (normal_prompt() + 0.7 * happy_prompt(k=7), False),
    (normal_prompt() + 0.9 * happy_prompt(k=6), False),
    (normal_prompt() + 0.1 * easy_prompt(k=12), False),
    (normal_prompt() + 0.3 * easy_prompt(k=7), False),
    (normal_prompt() + 0.5 * easy_prompt(k=6), False),
    (normal_prompt() + 0.7 * easy_prompt(k=5), False),
    (normal_prompt() + 0.9 * easy_prompt(k=5), False),
    (normal_prompt() + 0.1 * topical_prompts["sports"](k=13), False),
    (normal_prompt() + 0.3 * topical_prompts["sports"](k=8), False),
    (normal_prompt() + 0.5 * topical_prompts["sports"](k=6), False),
    (normal_prompt() + 0.7 * topical_prompts["sports"](k=5), False),
    (normal_prompt() + 0.9 * topical_prompts["sports"](k=5), False),
    (normal_prompt() + 0.1 * emotion_prompts["an angry"](k=14), False),
    (normal_prompt() + 0.3 * emotion_prompts["an angry"](k=8), False),
    (normal_prompt() + 0.5 * emotion_prompts["an angry"](k=6), False),
    (normal_prompt() + 0.7 * emotion_prompts["an angry"](k=4), False),
    (normal_prompt() + 0.9 * emotion_prompts["an angry"](k=4), False),
]

classifiers = [
    "SamLowe/roberta-base-go_emotions",
    "cardiffnlp/tweet-topic-21-multi",
    "s-nlp/roberta-base-formality-ranker",
    "cardiffnlp/twitter-roberta-base-sentiment-latest", 
    lambda x: np.mean([len(word) for word in x.split(" ")])
]

dataset = pd.read_csv("data/datasets/alpaca_processed.csv")
# dataset_subset = dataset[10000:10010] # 10 random sentences for calibration, only done 32 tokens
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
dataset_subset = dataset[:1000]

for i, formula in enumerate(formulas):
    evaluate_formula(
        formula=formula[0],
        do_non_spec=formula[1],
        dataset=dataset_subset,
        default_model="meta-llama/Llama-2-13b-chat-hf",
        folder=f"eval/time/{i}",
        batch_size=1,
        temperature=1,
        top_p=1,
        top_k=0,
        model_name_fluency="meta-llama/Llama-2-13b-chat-hf",
        dtype=torch.bfloat16,
        preserve_memory=True,
        classifier_name=classifiers,
        classification_with_input=False,
        dtype_faithfulness=torch.bfloat16,
        finetune_model=False,
        batch_size_faithfulness=8,
        reload=False,
        reload_data=False,
        max_tokens=64
    )