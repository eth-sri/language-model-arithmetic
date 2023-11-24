import torch
from model_arithmetic import ModelArithmetic, Evaluation, PromptedLLM, Max, Classifier, Min, enable_logging
from transformers import set_seed
import pandas as pd
import os
import numpy as np

set_seed(42)

enable_logging()

def evaluate_formula(formula, dataset, default_model, formula_file, store_file, store_file_monitor, dataset_file,
                    batch_size=4, temperature=1, top_p=1, top_k=0, model_name_fluency="meta-llama/Llama-2-7b-chat-hf", 
                    dtype=torch.bfloat16, preserve_memory=True, classifier_name="SkolkovoInstitute/roberta_toxicity_classifier", classification_with_input=False, 
                    dtype_faithfulness=torch.bfloat16, finetune_model=False, batch_size_faithfulness=8,
                    reload=False, reload_data=False, max_tokens=128, save_monitor=False):
    """
    Evaluates a formula using the provided dataset and parameters.

    Args:
        formula (str): The formula to be evaluated.
        dataset (str): The dataset to be used for evaluation.
        default_model (str): The default model to be used for evaluation.
        formula_file (str): The file path to save the formula.
        store_file (str): The file path to store the evaluation results.
        store_file_monitor (str): The file path to store the monitoring results.
        dataset_file (str): The file path of the dataset.
        batch_size (int, optional): The batch size for evaluation. Defaults to 4.
        temperature (int, optional): The temperature for sampling. Defaults to 1.
        top_p (int, optional): The top-p value for sampling. Defaults to 1.
        top_k (int, optional): The top-k value for sampling. Defaults to 0.
        model_name_fluency (str, optional): The model name for fluency evaluation. Defaults to "meta-llama/Llama-2-7b-chat-hf".
        dtype (torch.dtype, optional): The data type for evaluation. Defaults to torch.bfloat16.
        preserve_memory (bool, optional): Whether to preserve memory during evaluation. Defaults to True.
        classifier_name (str, optional): The name of the classifier model. Defaults to "SkolkovoInstitute/roberta_toxicity_classifier".
        classification_with_input (bool, optional): Whether to include input in classification. Defaults to False.
        dtype_faithfulness (torch.dtype, optional): The data type for faithfulness evaluation. Defaults to torch.bfloat16.
        finetune_model (bool, optional): Whether to finetune the model. Defaults to False.
        batch_size_faithfulness (int, optional): The batch size for faithfulness evaluation. Defaults to 8.
        reload (bool, optional): Whether to reload the model. Defaults to False.
        reload_data (bool, optional): Whether to reload the data. Defaults to False.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 128.
        save_monitor (bool, optional): Whether to save the monitoring results. Defaults to False.

    Returns:
        output: The evaluation output.
    """
    arithmetic = ModelArithmetic(formula, default_model=default_model)
    arithmetic.save_pretrained("../finetune/arithmetic")
    
    os.makedirs(os.path.dirname(formula_file), exist_ok=True)
    with open(formula_file, 'w') as outfile:
        outfile.write(str(formula))


    evaluator = Evaluation(arithmetic, dataset=dataset)

    output = evaluator.evaluate(
        store_file=store_file,
        dataset_file=dataset_file,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        model_name_fluency=model_name_fluency,
        dtype=dtype,
        preserve_memory=preserve_memory,
        model_name=classifier_name,
        classification_with_input=classification_with_input,
        dtype_faithfulness=dtype_faithfulness,
        finetune_model=finetune_model,
        batch_size_faithfulness=batch_size_faithfulness,
        reload=reload,
        do_perspective=False,
        reload_data=reload_data,
        max_tokens=max_tokens,
        speculation=False
    )
    
    if save_monitor:
        arithmetic.monitor.save(store_file_monitor)
    else:
        arithmetic.monitor.store(store_file_monitor)

prompt_template = lambda formula_string, input_string: f"<s>[INST]<<SYS>>\n{formula_string}\n<</SYS>>\n\n{input_string} [/INST]"
normal = "You are a helpful assistant"
formal = "You are an assistant using formal and objective language to answer the user."
happy = "You are a happy assistant."
emotions = lambda emotion: f"You are {emotion} assistant."
template_toxic = lambda e, f: f"{e}\nHuman: {f}\nAssistant:"
easy = "You are a helpful assistant using very simple and short language to answer the user as if they were five."
topical = lambda topic: f"You are a helpful assistant that answers the user in a way that is related to {topic}."

normal_prompt = PromptedLLM(normal, speculative_factor=1, prompt_template=prompt_template)
formal_prompt = PromptedLLM(formal, speculative_factor=1, prompt_template=prompt_template)
happy_prompt = PromptedLLM(happy, speculative_factor=1, prompt_template=prompt_template)
emotion_prompts = {
    emotion: PromptedLLM(emotions(emotion), speculative_factor=1, prompt_template=prompt_template)
    for emotion in ["an angry", "a sad", "a caring", "a loving"]
}
easy_prompt = PromptedLLM(easy, speculative_factor=1, prompt_template=prompt_template)
topical_prompts = {
    topic: PromptedLLM(topical(topic), speculative_factor=1, prompt_template=prompt_template)
    for topic in ["arts and culture", "business and entrepreneurs", "celebrity and pop culture", "diaries and daily life", "family", 
                  "fashion and style", "film, tv and video", "fitness & health", "food", "gaming", "learning",
                  "music", "news and social concern", "other hobbies", "relationships", "science and technology", "sports", "travel and adventure", 
                  "youth and student life"]
}

def add_weights(setting, change, diff=0.1):
    """
    Add weights to the given setting based on the specified change.

    Args:
        setting (list): The original setting.
        change (list): The indices of elements in the setting to be changed.
        diff (float, optional): The difference between consecutive weights. Defaults to 0.1.

    Returns:
        list: A list of new settings with added weights.
    """
    
    new_weights = []
    
    for c in change:
        setting_weights = [i if isinstance(i, (int, float)) else i[0] for i in setting]
        sum_setting = sum(setting_weights)
        sum_setting -= setting[c] if isinstance(setting[c], (int, float)) else setting[c][0]
        weight = [i * diff * sum_setting for i in range(0, int(1 / diff) + 1)]
            
        for w in weight:
            setting_here = setting[:]
            if c >= 4:
                setting_here[c][0] = w
            else:
                setting_here[c] = w
            
            new_weights.append(setting_here)
            
    return new_weights

def add_formula(template, norm_template, weight_defaults, change, diff=0.1, factors=1, max_=1):
    """
    Generates formulas and resulting weights based on the given parameters.

    Parameters:
    template (function): The template function used to generate formulas.
    norm_template (function): The normalization function used to normalize weights.
    weight_defaults (list): The default weights.
    change (list): The indices of weights to be changed.
    diff (float, optional): The difference between each weight value. Defaults to 0.1.
    factors (int or list, optional): The factors to multiply the weights by. Defaults to 1.
    max_ (float, optional): The maximum weight value. Defaults to 1.

    Returns:
    tuple: A tuple containing the generated formulas and resulting weights.
    """
    
    formulas = []
    resulting_weights = []
    for index, c in enumerate(change):
        weight_default_no = weight_defaults[:]
        weight_default_no[c] = 0
        if isinstance(factors, (int, float)):
            factor = factors
        else:
            factor = factors[index]
        weight = [i * diff * norm_template(*weight_default_no) * factor for i in range(0, int(max_ / diff) + 1)]
        for w in weight:
            weights = weight_defaults[:]
            weights[c] = w
            formulas.append(template(*weights))
            
            resulting_weights.append(weights)
    
    return formulas, resulting_weights
            
        

default_topic = "sports"
default_emotion = "an angry"
weights = [
    (1, 0, 0, 0, [0, default_emotion], [0, default_topic]), 
    (0, 1, 0, 0, [0, default_emotion], [0, default_topic]),
    (0, 0, 1, 0, [0, default_emotion], [0, default_topic]),
    (0, 0, 0, 1, [0, default_emotion], [0, default_topic]),
    (0, 0, 0, 0, [0, default_emotion], [1, default_topic])
]

formula_template1 = lambda a, b, c: (1-c) * normal_prompt + a * happy_prompt + b * easy_prompt + c * Max(topical_prompts["sports"], normal_prompt)
norm_template1 = lambda a, b, c: 1 + a + b

formula_template2 = lambda a, b, c: normal_prompt + a * Max(normal_prompt, formal_prompt) + b * easy_prompt + c * Classifier(normal_prompt - a * Max(formal_prompt - normal_prompt, 0) + b * easy_prompt, 
                                                                                                                                 "cardiffnlp/tweet-topic-21-multi", prompt_template=lambda e, f: "",
                                                                                                                                 n_runs_per_sample=50, batch_size=26, use_bayes=True, minimize=False, index=10)
norm_template2 = lambda a, b, c: 1 + a + b

# setting_1 = [1, 0.5, 0.5, 0.5, [0, default_emotion], [0, default_topic]]
# change_1 = [1, 2, 3, 4]
# weights += add_weights(setting_1, change_1)

# setting_1 = [1, 0.2, 0.2, 0.2, [0.2, "a caring"], [0.2, "sports"]]
# weights += add_weights(setting_1, [4, 5])
classifiers = [
    "SamLowe/roberta-base-go_emotions",
    "cardiffnlp/tweet-topic-21-multi",
    "s-nlp/roberta-base-formality-ranker",
    "cardiffnlp/twitter-roberta-base-sentiment-latest", 
    lambda x: np.mean([len(word) for word in x.split(" ")])
]

prompts = [
    normal_prompt,
    formal_prompt,
    happy_prompt,
    easy_prompt,
    emotion_prompts,
    topical_prompts
]

formulas = []
for weight in weights:
    formula = 0
    for i, w in enumerate(weight):
        if w != 0:
            if isinstance(w, (int, float)):
                formula += w * prompts[i]
            elif w[0] != 0:
                formula += w[0] * prompts[i][w[1]]
                
    formulas.append(formula)

formulas1, extra_weights = add_formula(formula_template1, norm_template1, [1.0, 1.0, 1.0], [0, 1, 2], factors=[1, 1, 1], diff=0.2, max_=2.0)
formulas += formulas1
weights += extra_weights
formulas2, extra_weights = add_formula(formula_template2, norm_template2, [1.0, 1.0, 1.0], [0, 1, 2], factors=[1, 1, 2], diff=0.2, max_=2.0)
formulas += formulas2
weights += extra_weights

formulas1, extra_weights = add_formula(formula_template1, norm_template1, [1.0, 1.0, 1.0], [0, 1, 2], factors=[-1, -1, -1], diff=0.1, max_=0.5)
formulas += formulas1
weights += extra_weights
formulas2, extra_weights = add_formula(formula_template2, norm_template2, [1.0, 1.0, 1.0], [0, 1, 2], factors=[-1, -1, -2], diff=0.1, max_=0.5)
formulas += formulas2
weights += extra_weights

dataset = pd.read_csv("data/datasets/alpaca_processed.csv")
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
dataset_subset = dataset[:1000]

for i, formula in enumerate(formulas):
    if i < 0:
        continue
    os.makedirs(f"eval/persona/{i}", exist_ok=True)
    with open(f"eval/persona/{i}/weights.txt", 'w') as outfile:
        outfile.write(weights[i].__str__())
    evaluate_formula(
        formula=formula,
        dataset=dataset_subset,
        default_model="meta-llama/Llama-2-13b-chat-hf",
        formula_file=f"eval/persona/{i}/formula.txt",
        store_file=f"eval/persona/{i}/evaluation.json",
        store_file_monitor=f"eval/persona/{i}/monitor.json",
        dataset_file=f"eval/persona/{i}/data.csv",
        batch_size=16,
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