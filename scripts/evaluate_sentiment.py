import torch
from model_arithmetic import ModelArithmetic, Evaluation, PromptedLLM, enable_logging, load_tokenizer
from transformers import set_seed
import pandas as pd
from formulas_sentiment import *
from loguru import logger
import os

enable_logging()

def evaluate_formula(formula, dataset, default_model, formula_file, store_file, store_file_monitor, dataset_file,
                    batch_size=4, temperature=1, top_p=1, top_k=0, model_name_fluency="meta-llama/Llama-2-7b-chat-hf", 
                    dtype=torch.bfloat16, preserve_memory=True, classifier_name="SkolkovoInstitute/roberta_toxicity_classifier", 
                    classification_with_input=False, dtype_faithfulness=torch.bfloat16, finetune_model=False, batch_size_faithfulness=8,
                    reload=False, reload_data=False, max_tokens=32):
    """
    Evaluates a formula using the given dataset and parameters.

    Args:
        formula (str or tuple): The formula to evaluate. If a tuple is provided, the second element is considered as retroactive operators.
        dataset (pandas.DataFrame): The dataset to evaluate the formula on.
        default_model (str): The default model to use for evaluation.
        formula_file (str): The file path to store the formula.
        store_file (str): The file path to store the evaluation results.
        store_file_monitor (str): The file path to monitor changes in the formula.
        dataset_file (str): The file path to store the dataset.
        batch_size (int, optional): The batch size for evaluation. Defaults to 4.
        temperature (int, optional): The temperature for sampling. Defaults to 1.
        top_p (int, optional): The top-p value for sampling. Defaults to 1.
        top_k (int, optional): The top-k value for sampling. Defaults to 0.
        model_name_fluency (str, optional): The model name for fluency evaluation. Defaults to "meta-llama/Llama-2-7b-chat-hf".
        dtype (torch.dtype, optional): The data type for evaluation. Defaults to torch.bfloat16.
        preserve_memory (bool, optional): Whether to preserve memory during evaluation. Defaults to True.
        classifier_name (str, optional): The name of the classifier model. Defaults to "SkolkovoInstitute/roberta_toxicity_classifier".
        classification_with_input (bool, optional): Whether to include input text in classification. Defaults to False.
        dtype_faithfulness (torch.dtype, optional): The data type for faithfulness evaluation. Defaults to torch.bfloat16.
        finetune_model (bool, optional): Whether to finetune the model during evaluation. Defaults to False.
        batch_size_faithfulness (int, optional): The batch size for faithfulness evaluation. Defaults to 8.
        reload (bool, optional): Whether to reload the model during evaluation. Defaults to False.
        reload_data (bool, optional): Whether to reload the dataset during evaluation. Defaults to False.
        max_tokens (int, optional): The maximum number of tokens for input truncation. Defaults to 32.

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
    # for the default model, truncate the input text to 32 tokens
    tokenizer = load_tokenizer(model_name_fluency)
    dataset["input"] = dataset["input"].apply(lambda x: tokenizer.encode(x)[:32])
    # detokenize input again
    dataset['input'] = dataset['input'].apply(lambda x: tokenizer.decode(x, skip_special_tokens=True))
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
        do_perspective=False,
        stop_texts=['\n'],
    )
    
    arithmetic.monitor.store(store_file_monitor)

formulas = []
formulas_negative = []
for model in ["meta-llama/Llama-2-13b-hf", "EleutherAI/Pythia-12b", "mosaicml/mpt-7b"]:
    formulas += [
        main_model(model=model),
        main_model(sentence=positive_sentence, model=model) + 0.0 * main_model(sentence="", model=model),  # 0.0 needed for monitoring
        negative_biasing(-0.6, model=model, first_sentence=positive_sentence),
        negative_biasing(-0.96, max_=True, model=model, first_sentence=positive_sentence),
        selfdebias(10, model=model, first_sentence=positive_sentence, sentence=negative_sentence),
        classifier(1.0, c_model="finetune/sentiment_classifier", m_model=model, minimize=False, first_sentence=positive_sentence),
        combo(0.04, -0.0, -0.96, c_model="finetune/sentiment_classifier", m_model=model, minimize=False, first_sentence=positive_sentence),
    ]
    formulas_negative += [
        main_model(model=model),
        main_model(sentence=negative_sentence, model=model) + 0.0 * main_model(sentence="", model=model),
        negative_biasing(-0.6, model=model, sentence=positive_sentence, first_sentence=negative_sentence),
        negative_biasing(-0.96, max_=True, model=model, sentence=positive_sentence, first_sentence=negative_sentence),
        selfdebias(10, model=model, sentence=positive_sentence, first_sentence=negative_sentence),
        classifier(-1.0, c_model="finetune/sentiment_classifier", m_model=model, minimize=False, first_sentence=negative_sentence),
        combo(-0.04, -0.0, -0.96, c_model="finetune/sentiment_classifier", m_model=model, sentence=positive_sentence, minimize=False, first_sentence=negative_sentence),
    ]
    
dataset = pd.read_csv("data/datasets/IMDB_processed.csv")
dataset_positive = dataset[dataset["label"] == 1].reset_index(drop=True)
dataset_negative = dataset[dataset["label"] == 0].reset_index(drop=True)
dataset_positive = dataset_positive[:1000].reset_index(drop=True)
dataset_negative = dataset_negative[:1000].reset_index(drop=True)
dataset_positive["input"] = dataset_positive["text"]
dataset_negative["input"] = dataset_negative["text"]

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
            dataset=dataset_negative,
            default_model=None,
            reload=False,
            reload_data=False,
            formula_file=f"eval/sentiment_final/{i}/negtopos/formula.txt",
            store_file=f"eval/sentiment_final/{i}/negtopos/evaluation.json",
            store_file_monitor=f"eval/sentiment_final/{i}/negtopos/monitor.json",
            dataset_file=f"eval/sentiment_final/{i}/negtopos/data.csv",
            batch_size=batch_size,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            model_name_fluency=first_model,
            dtype=torch.bfloat16,
            preserve_memory=True,
            classifier_name=["cardiffnlp/twitter-roberta-base-sentiment-latest", 'finetune/sentiment_all'],
            classification_with_input=False,
            dtype_faithfulness=torch.bfloat16,
            finetune_model=False,
            batch_size_faithfulness=32,
            max_tokens=64
        )
        evaluate_formula(
            formula=formulas_negative[i],
            dataset=dataset_positive,
            default_model=None,
            reload=False,
            reload_data=False,
            formula_file=f"eval/sentiment_final/{i}/postoneg/formula.txt",
            store_file=f"eval/sentiment_final/{i}/postoneg/evaluation.json",
            store_file_monitor=f"eval/sentiment_final/{i}/postoneg/monitor.json",
            dataset_file=f"eval/sentiment_final/{i}/postoneg/data.csv",
            batch_size=batch_size,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            model_name_fluency=first_model,
            dtype=torch.bfloat16,
            preserve_memory=False,
            classifier_name=["cardiffnlp/twitter-roberta-base-sentiment-latest", 'finetune/sentiment_all'],
            classification_with_input=False,
            dtype_faithfulness=torch.bfloat16,
            finetune_model=False,
            batch_size_faithfulness=32,
            max_tokens=64
        )