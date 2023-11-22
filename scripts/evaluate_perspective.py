from model_arithmetic import Evaluation, enable_logging
import pandas as pd
import os
import json

enable_logging()


def run_folder(folder_name):
    """
    Run perspective evaluation on a folder.

    Args:
        folder_name (str): The path to the folder containing the evaluation files.

    Returns:
        None
    """
    
    if not os.path.exists(os.path.join(folder_name, "evaluation.json")):
        return
    data = pd.read_csv(os.path.join(folder_name, "data.csv"), escapechar='\\', lineterminator="\n")
    eval_ = json.load(open(os.path.join(folder_name, "evaluation.json")))
    if "perspective" in eval_["output"]:
        return
    evaluation = Evaluation(dataset=data)
    perspective = evaluation.perspective(classification_with_input=False)
    eval_ = json.load(open(os.path.join(folder_name, "evaluation.json")))
    eval_["output"]["perspective"] = perspective
    json.dump(eval_, open(os.path.join(folder_name, "evaluation.json"), "w"), indent=4)
    
    
if __name__ == "__main__":
    parent_folder = "eval/toxic_final"
    for folder in os.listdir(parent_folder):
        run_folder(os.path.join(parent_folder, folder))