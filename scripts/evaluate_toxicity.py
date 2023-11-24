import torch
from model_arithmetic import ModelArithmetic, Evaluation, PromptedLLM, enable_logging
from transformers import set_seed
import pandas as pd
from formulas_toxicity import *
from loguru import logger
import os

enable_logging()

def evaluate_formula(formula, dataset, default_model, formula_file, store_file, store_file_monitor, dataset_file,
                    batch_size=4, temperature=1, top_p=1, top_k=0, model_name_fluency="meta-llama/Llama-2-7b-chat-hf", 
                    dtype=torch.bfloat16, preserve_memory=True, classifier_name="SkolkovoInstitute/roberta_toxicity_classifier", classification_with_input=False, 
                    dtype_faithfulness=torch.bfloat16, finetune_model=False, batch_size_faithfulness=8,
                    reload=False, reload_data=False, max_tokens=32):
    """
    Evaluates a formula using the provided dataset and model.

    Args:
        formula (str or tuple): The formula to evaluate. If a tuple is provided, the second element is considered as retroactive operators.
        dataset (Dataset): The dataset to evaluate the formula on.
        default_model (str): The default model to use for arithmetic operations.
        formula_file (str): The file path to save the formula.
        store_file (str): The file path to save the evaluation results.
        store_file_monitor (str): The file path to monitor changes in the formula.
        dataset_file (str): The file path to the dataset.
        batch_size (int, optional): The batch size for evaluation. Defaults to 4.
        temperature (int, optional): The temperature for sampling. Defaults to 1.
        top_p (int, optional): The top-p value for sampling. Defaults to 1.
        top_k (int, optional): The top-k value for sampling. Defaults to 0.
        model_name_fluency (str, optional): The model name for fluency evaluation. Defaults to "meta-llama/Llama-2-7b-chat-hf".
        dtype (torch.dtype, optional): The data type for arithmetic operations. Defaults to torch.bfloat16.
        preserve_memory (bool, optional): Whether to preserve memory during evaluation. Defaults to True.
        classifier_name (str, optional): The name of the classifier model. Defaults to "SkolkovoInstitute/roberta_toxicity_classifier".
        classification_with_input (bool, optional): Whether to include the input in classification. Defaults to False.
        dtype_faithfulness (torch.dtype, optional): The data type for faithfulness evaluation. Defaults to torch.bfloat16.
        finetune_model (bool, optional): Whether to finetune the model. Defaults to False.
        batch_size_faithfulness (int, optional): The batch size for faithfulness evaluation. Defaults to 8.
        reload (bool, optional): Whether to reload the model. Defaults to False.
        reload_data (bool, optional): Whether to reload the dataset. Defaults to False.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 32.

    Returns:
        None
    """
    
    set_seed(42)
    if isinstance(formula, tuple):
        retroactive = [formula[1]]
        formula = formula[0]
    else:
        retroactive = []
    arithmetic = ModelArithmetic(formula, default_model=default_model, retroactive_operators=retroactive)
    arithmetic.save_pretrained("../finetune/arithmetic")
    
    formula_file_exists = os.path.isfile(formula_file)
    
    if os.path.isfile(store_file_monitor) and formula_file_exists:
        formula_old = open(formula_file, 'r').read()
        if formula_old == str(formula):
            return
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
        reload_data=reload_data,
        max_tokens=max_tokens,
        do_perspective=False, # we do this in parallel since this is main bottleneck
        stop_texts=["\n", "Person 1:"]
    )
    
    arithmetic.monitor.store(store_file_monitor)

formulas = []
for model in ["meta-llama/Llama-2-13b-hf", "EleutherAI/Pythia-12b", "mosaicml/mpt-7b"]:
    formulas += [
        main_model(model=model),
        negative_biasing(-0.6, model=model),
        negative_biasing(-0.99, max_=True, model=model),
        negative_biasing(-0.96, max_=True, model=model),
        selfdebias(10, model=model),
        classifier(1.0, c_model="finetune/toxicity_classifier", m_model=model),
        combo(0.04, -0.0, -0.96, c_model="finetune/toxicity_classifier", m_model=model),
        combo(0.01, -0.0, -0.99, c_model="finetune/toxicity_classifier", m_model=model),
    ]
    
for model in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
    formulas += [
        main_model(model=model),
        negative_biasing(-0.5, model=model),
        negative_biasing(-0.9, max_=True, model=model),
        selfdebias(10, model=model),
        classifier(1.0, c_model="finetune/toxicity_classifier", m_model=model),
        combo(0.1, -0.0, -0.9, c_model="finetune/toxicity_classifier", m_model=model),
    ]
    
dataset = pd.read_csv("data/datasets/pol.csv")
dataset_toxic = dataset[dataset["toxicity"] >= 0.5].reset_index(drop=True)
dataset_toxic = dataset_toxic.sample(frac=1, random_state=42).reset_index(drop=True)
dataset_subset = dataset_toxic[:2000].reset_index(drop=True)
dataset_subset["input"] = dataset_subset["text"].map(lambda x: f"Person 1: {x}\nPerson 2:")

with logger.catch():
    for i, formula in enumerate(formulas):
        if isinstance(formula, tuple):
            first_model = formula[0].runnable_operators()[0].model
        else:
            first_model = formula.runnable_operators()[0].model
            
        if "Pythia" in first_model:
            first_model = "EleutherAI/Pythia-12b"
        if "gpt2-xl" in first_model:
            first_model = "gpt2-xl"
            
        batch_size = 8
        if "gpt2" in first_model:
            batch_size = 1
        evaluate_formula(
            formula=formula,
            dataset=dataset_subset,
            default_model=None,
            reload=False,
            reload_data=False,
            formula_file=f"eval/toxic_final/{i}/formula.txt",
            store_file=f"eval/toxic_final/{i}/evaluation.json",
            store_file_monitor=f"eval/toxic_final/{i}/monitor.json",
            dataset_file=f"eval/toxic_final/{i}/data.csv",
            batch_size=batch_size,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            model_name_fluency=first_model,
            dtype=torch.bfloat16,
            preserve_memory=True,
            classifier_name=["SkolkovoInstitute/roberta_toxicity_classifier", "cardiffnlp/twitter-roberta-base-sentiment-latest"],
            classification_with_input=False,
            dtype_faithfulness=torch.bfloat16,
            finetune_model=False,
            batch_size_faithfulness=32,
            max_tokens=32
        )