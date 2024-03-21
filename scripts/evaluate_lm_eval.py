import os, sys
path = os.path.abspath(os.getcwd())
sys.path.append(path)
from model_arithmetic import Evaluation, ModelArithmetic, load_model, PromptedLLM, Max, KL_indicator, enable_logging
import torch
from loguru import logger
from transformers import set_seed

import tensorflow as tf

enable_logging()

# Necessary in order to avoid the small BLEURT model to take up all GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
        print(e)

BASE_EVAL_PATH = "eval/performance"

def evaluate(task_name, formula, save_path, default_model, num_fewshot=0, limit=None, no_cache=False, batch_size=1, dtype=torch.float16, output_folder=None):
    """
    Evaluate the LM-Eval model on a given task.

    Args:
        task_name (str): Name of the task to evaluate.
        formula (str or tuple): Formula to evaluate. If a tuple is provided, the second element is considered as a retroactive operator.
        save_path (str): Path to save the evaluation results.
        default_model (str): Default model to use for evaluation.
        num_fewshot (int, optional): Number of few-shot examples to use. Defaults to 0.
        limit (int, optional): Limit on the number of examples to evaluate. Defaults to None.
        no_cache (bool, optional): Whether to use the cache for evaluation. Defaults to False.
        batch_size (int, optional): Batch size for evaluation. Defaults to 1.
        dtype (torch.dtype, optional): Data type for evaluation. Defaults to torch.float16.
        output_folder (str, optional): Folder to save the output files. Defaults to None.
    """
    
    set_seed(42)

    model_args = None

    evaluator = Evaluation()
    
    if isinstance(formula, tuple):
        retroactive = [formula[1]]
        formula = formula[0]
    else:
        retroactive = []
    arithmetic = ModelArithmetic(formula, default_model=default_model, retroactive_operators=retroactive, 
                                 dtype=dtype, needs_input_tokens_lm_eval=True, lm_eval_task=task_name)

    evaluator.evaluate_lm_eval(model=arithmetic, model_args=model_args, task_name=task_name, batch_size=batch_size, 
                               num_fewshot=num_fewshot, limit=limit, write_out=True)
    evaluator.save(save_path)
    
def eval_multiple(formula, datasets, name, limit=None, num_fewshot=0, batch_size=1):
    """
    Evaluate multiple datasets using a given formula.

    Args:
        formula (str): The formula to evaluate.
        datasets (list): List of dataset names to evaluate.
        name (str): Name of the evaluation.
        limit (int, optional): Limit on the number of examples to evaluate. Defaults to None.
        num_fewshot (int, optional): Number of few-shot examples to include. Defaults to 0.
        batch_size (int, optional): Batch size for evaluation. Defaults to 1.
    """
    os.makedirs(os.path.join(BASE_EVAL_PATH, name), exist_ok=True)
    with open(os.path.join(BASE_EVAL_PATH, name, "formula.txt"), 'w') as outfile:
        outfile.write(str(formula))
    for dataset in datasets:
        evaluate(
            formula=formula,
            default_model="meta-llama/Llama-2-13b-hf",
            task_name=dataset,
            num_fewshot=num_fewshot,
            limit=limit,
            no_cache=True,
            save_path=os.path.join(BASE_EVAL_PATH, name, f"{dataset}_eval.json"),
            batch_size=batch_size,
            dtype=torch.bfloat16,
            output_folder=os.path.join(BASE_EVAL_PATH, name)
        )



if __name__ == "__main__":
    with logger.catch():
        gpt2xl = PromptedLLM("", prompt_template=lambda e, f: f"{f}", model="gpt2-xl")
        gpt2xl_no_context = PromptedLLM("", prompt_template=lambda e, f: f"", model="gpt2-xl")
        formulas = [
            gpt2xl,
            1.5 * gpt2xl - 0.5 * gpt2xl_no_context,
        ]

        for index, formula in enumerate(formulas):
            if index < 0:
                continue
            eval_multiple(
                formula=formula,
                datasets=["hellaswag", "lambada_openai", "winogrande", "arc_easy", "boolq", "arc_challenge", "piqa", "sciq"],
                # datasets=["crows_pairs_english"],
                name=str(index),
                limit=1000,
            )