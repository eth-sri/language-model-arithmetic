import dotenv
import os
import asyncio
import numpy as np
import json
import pandas as pd
from model_arithmetic import OpenAIQuery

dotenv.load_dotenv()

np.random.seed(42)

def run_queries(messages):
    """
    Runs queries using the given messages as input prompts and returns the generated outputs.

    Args:
        messages (list): A list of messages to be used as input prompts.

    Returns:
        list: A list of generated outputs.

    """
    system_prompt = """The user did an experiment where several language models were prompted to continue the start of a movie review. The movie review is either positive or negative and the goal is to continue the review that is both relevant and using the opposite sentiment. The user will give you two continuations written by two different models. Briefly reason about which continuation is better and end your answer with: "Preferred continuation: 1" or "Preferred continuation: 2", depending on which option is better. If both options are equally good, end your response with "Preferred continuation: None"."""
    model = "gpt-4"
    input_prompts = [
        [{"content": system_prompt, "role": "system"}, {"content": message, "role": "user"}] for message in messages
    ]
    querier = OpenAIQuery(model=model, tpm=30000, timeout=100, temperature=0, max_tokens=256, error_stop=10 ** 4)
    outputs = asyncio.run(querier.run_string_prompts(input_prompts))
    
    outputs = [output["message"]["content"] for output in outputs]
    
    return outputs

def process_output(output_message):
    """
    Process the output message from the model and returns the preferred model.

    Args:
        output_message (str): The output message from the model.

    Returns:
        float or None: Which model is preferred
    """
    try:
        split_ = output_message.split("Preferred continuation: ")[1]
    except Exception:
        return None
    if "1" in split_:
        return 0
    elif "2" in split_:
        return 1
    elif "None" in split_ or "none" in split_:
        return 0.5
    else:
        return None
    
def prepare_messages(df1, df2, sentiment):
    """
    Prepares messages for model comparison.

    Args:
        df1 (pandas.DataFrame): The first DataFrame containing generated outputs from Model 1.
        df2 (pandas.DataFrame): The second DataFrame containing generated outputs from Model 2.
        sentiment (str): The goal sentiment for the input reviews.

    Returns:
        list: A list of prompts, where each prompt is a list containing the input review, goal sentiment,
              Model 1 output, and Model 2 output, along with a flag indicating whether Model 1 and Model 2
              outputs were switched.
    """
    
    prompts = []
    # make sure df1 and df2 are aligned by input
    df = df1.merge(df2, on="input", how="inner", suffixes=("_1", "_2"))
    for index, row in df.iterrows():
        model1_output = row["generated_1"]
        model2_output = row["generated_2"]
        m1 = 0
        
        if np.random.uniform() < 0.5:
            # switch
            model1_output, model2_output = model2_output, model1_output
            m1 = 1
        
        prompt = f"Input Review: {row['input']}\nGoal Sentiment: {sentiment}\nModel 1:{model1_output}\nModel 2:{model2_output}"
        
        prompts.append([prompt, m1])
        
    return prompts
        
def interpret_outputs(prompts, outputs):
    """
    Interpret the outputs of the model and calculate scores based on the prompts.

    Args:
        prompts (list): List of prompts used for generating the outputs.
        outputs (list): List of model outputs.

    Returns:
        dict: A dictionary containing the scores calculated based on the outputs.
              The keys represent different score categories, and the values represent the count of each category.
    """
    processed_outputs = [process_output(output) for output in outputs]
    scores = {0: 0, 1:0, 0.5: 0, "None": 0}
    for output, prompt in zip(processed_outputs, prompts):
        if output is None:
            scores["None"] += 1
        elif output == 0.5:
            scores[0.5] += 1
        elif output == prompt[1]:
            scores[0] += 1
        elif output != prompt[1]:
            scores[1] += 1
    return scores

def run(df1, df2, goal_sentiment, save_file):
    """
    Runs the sentiment analysis on the given dataframes and returns the scores.
    
    Args:
        df1 (pandas.DataFrame): The first dataframe.
        df2 (pandas.DataFrame): The second dataframe.
        goal_sentiment (str): The desired sentiment.
        save_file (str): The file path to save the outputs and prompts.
        
    Returns:
        dict: The scores of the sentiment analysis.
    """
    
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    if os.path.isfile(save_file):
        with open(save_file, "r") as f:
            outputs, prompts = json.load(f)
    else:
        prompts = prepare_messages(df1, df2, sentiment=goal_sentiment)
        outputs = run_queries([prompt[0] for prompt in prompts])
        with open(save_file, "w") as f:
            json.dump([outputs, prompts], f, indent=4)
    
    scores = interpret_outputs(prompts, outputs)
    
    return scores

def process_scores(scores, folder_formula1, folder_formula2):
    """
    Process the scores dictionary and return a modified version with additional information.

    Args:
        scores (dict): A dictionary containing scores.
        folder_formula1 (str): The path to the folder containing formula1.
        folder_formula2 (str): The path to the folder containing formula2.

    Returns:
        dict: A modified version of the scores dictionary with additional information.
    """
    formula1 = open(os.path.join(folder_formula1, 'formula.txt'), "r").read()
    formula2 = open(os.path.join(folder_formula2, 'formula.txt'), "r").read()
    total = sum(scores.values()) - scores["None"]
    scores = {key: value / total for key, value in scores.items()}
    return {
        **scores,
        "formula1": formula1,
        "formula2": formula2,
    }

def compare_indices_single(index1, index2, goal_sentiment):
    """
    Compare the sentiment scores using gpt4 of two models for a given goal sentiment.

    Parameters:
    index1 (int): The first index to compare.
    index2 (int): The second index to compare.
    goal_sentiment (str): The goal sentiment to evaluate ('negtopos' or 'postoneg').

    Returns:
    dict: A dictionary containing the processed scores and the goal sentiment.
    """
    
    folder1 = f"eval/sentiment_final/{index1}/{goal_sentiment}"
    folder2 = f"eval/sentiment_final/{index2}/{goal_sentiment}"
    df1 = pd.read_csv(os.path.join(folder1, "data.csv"))
    df2 = pd.read_csv(os.path.join(folder2, "data.csv"))
    sentiment = 'positive' if goal_sentiment == 'negtopos' else 'negative'
    scores = run(df1, df2, sentiment, f"eval/gpt4_sentiment/{index1}_{index2}_{goal_sentiment}.json")
    processed_scores = process_scores(scores, folder1, folder2)
    processed_scores['goal_sentiment'] = sentiment
    
    return processed_scores

def compare_indices(index1, index2):
    """
    Compare the sentiment using gpt4 for the models at the two indices.
    
    Parameters:
    index1 (int): The first index.
    index2 (int): The second index.
    
    Returns:
    list: A list of sentiment scores for the two indices.
    """
    scores = []
    scores.append(compare_indices_single(index1, index2, 'negtopos'))
    scores.append(compare_indices_single(index1, index2, 'postoneg'))
    return scores

if __name__ == "__main__":
    all_scores = []
    
    os.makedirs("eval/gpt4_sentiment", exist_ok=True)
    # llama
    all_scores += compare_indices(3, 2)
    all_scores += compare_indices(3, 5)
    all_scores += compare_indices(6, 2)
    all_scores += compare_indices(6, 5)
    
    # pythia
    all_scores += compare_indices(10, 9)
    all_scores += compare_indices(10, 12)
    all_scores += compare_indices(13, 9)
    all_scores += compare_indices(13, 12)
    
    #mpt
    all_scores += compare_indices(17, 16)
    all_scores += compare_indices(17, 19)
    all_scores += compare_indices(20, 16)
    all_scores += compare_indices(20, 19)
    
    os.makedirs("eval/processed", exist_ok=True)
    data = pd.DataFrame(all_scores)
    data.to_csv("eval/processed/sentiment_gpt4_scores.csv", index=False)