import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
from matplotlib import cm


########################################
# Extract time results
########################################


def extract_interesting_data(folder, folder_formula=None):
    if folder_formula is None:
        folder_formula = folder
    formula = open(os.path.join(folder_formula, "formula.txt"), "r").read()
    evaluation = json.load(open(os.path.join(folder, "evaluation.json"), "r"))
    monitor = json.load(open(os.path.join(folder, "monitor.json"), "r"))
    return {
        "formula": formula,
        "n_tokens": monitor["length"]["total"],
        "time": monitor["total"],
        "models": [
            {
                "divergence": model["KL_divergence"]["mean"],
                "n_calls": model["n_calls"],
                "prompt_template": model["prompt_template"],
                "acceptance_prob": model.get("expected_acceptance_prob", {"mean": 0})["mean"],
            }
            for model in monitor["models_monitor"]
        ]
    }
    
base_folder = "eval/time"
all_data = []
for i in range(58):
    path = os.path.join(base_folder, str(i))
    try:
        interesting = extract_interesting_data(path)
        interesting["speculation"] = False
        all_data.append(    
            interesting
        )
    except Exception as e:
        pass
    
    path = os.path.join(base_folder, str(i) + "_spec")
    try:
        interesting = extract_interesting_data(path, os.path.join(base_folder, str(i)))
        interesting["speculation"] = True
        all_data.append(    
            interesting
        )
    except Exception as e:
        pass
    
for i, element in enumerate(all_data):
    element["time_per_token"] = element["time"] / element["n_tokens"] * 1000
    calls = sum([model["n_calls"] for model in element["models"] if "[INST]" in model["prompt_template"]])
    element["calls_per_token"] = calls / element["n_tokens"]
    del element["models"]
    
all_data_csv = pd.DataFrame(all_data)
os.makedirs("eval/processed", exist_ok=True)
all_data_csv.to_csv("eval/processed/time.csv")
    
    
character_indices = {
    "formal": [3, 7],
    "happy": [9, 13],
    "easy": [15, 19],
    "sports": [21, 25],
    "angry": [27, 31],
}

character_indices2 = {
    "formal": [46,50],
    "happy": [51, 55],
    "easy": [56, 60],
    "sports": [61, 65],
    "angry": [66, 70],
}


fig, ax = plt.subplots(figsize=(3.3,3.3))
ax.set_facecolor((0.95, 0.95, 0.95))
# set font size
plt.rcParams.update({'font.size': 16})

ax.title.set(x=0.15)
# set the title font to something fancy
ax.set_title("Calls per Token")
sns.despine(left=True, bottom=True)



base = 0.4
colors = [
    cm.get_cmap("Blues")(base),
    cm.get_cmap("RdPu")(base),
    cm.get_cmap("Greens")(base),
    cm.get_cmap("Oranges")(base),
    cm.get_cmap("Reds")(base),
    cm.get_cmap("Purples")(base),
    
]
colors = sns.color_palette("colorblind", len(character_indices))

for i, character in enumerate(character_indices):
    x1 = [all_data[j]["divergence"] for j in range(character_indices[character][0], character_indices[character][1] + 1)]
    x2 = [all_data[j]["divergence"] for j in range(character_indices2[character][0], character_indices2[character][1] + 1)]
    y1 = [all_data[j]["calls_per_token"] for j in range(character_indices[character][0], character_indices[character][1] + 1)]
    y2 = [all_data[j]["calls_per_token"] for j in range(character_indices2[character][0], character_indices2[character][1] + 1)]
    # alternatively concat y1 and y2 into y1[0], y2[0], y1[1], ...
    x = []
    y = []
    for j in range(len(x1)):
        x.append(x1[j])
        x.append(x2[j])
        y.append(y1[j])
        y.append(y2[j])
    # create line
    sns.lineplot(x=x, y=y, color=colors[i])
        

ax.set_xlabel("Divergence", fontsize=16)
ax.set_xscale("log")
fig.tight_layout()
os.makedirs("eval/plots", exist_ok=True)
fig.savefig("eval/plots/calls_per_token_divergence.pdf")

########################################
# Extract toxicity results
########################################

toxicity_results = []

def extract_KL(monitor):
    for model in monitor["models_monitor"]:
        if model["prompt_template"] == "Continue the following conversation.\n{{input_string}}":
            return model["KL_divergence"]["mean"]
        
    for model in monitor["models_monitor"]:
        if model["class"] == "PromptedLLM":
            return model["KL_divergence"]["mean"]

for index in range(48):
    try:
        formula = open(f"eval/toxic_final/{index}/formula.txt", "r").read()
        evaluation = json.load(open(f"eval/toxic_final/{index}/evaluation.json", "r"))
        monitor = json.load(open(f"eval/toxic_final/{index}/monitor.json", "r"))
        toxicity_results.append(
            {
                "formula": formula,
                "toxicity": evaluation["output"].get("perspective", None), 
                "perplexity": evaluation["output"]["perplexity"]["correct_perplexity"],
            }
        )
    except FileNotFoundError:
        pass
    
toxicity_results = pd.DataFrame(toxicity_results)
toxicity_results.to_csv("eval/processed/toxicity.csv")

########################################
# Extract sentiment results
########################################

sentiment_results = []

for index in range(21):
    for mode in ['negtopos', 'postoneg']:
        try:
            formula = open(f"eval/sentiment_final/{index}/{mode}/formula.txt", "r").read()
            evaluation = json.load(open(f"eval/sentiment_final/{index}/{mode}/evaluation.json", "r"))
            monitor = json.load(open(f"eval/sentiment_final/{index}/{mode}/monitor.json", "r"))
            index = 2 if mode == "negtopos" else 0
            sentiment_results.append(
                {
                    "formula": formula,
                    "sentiment": evaluation["output"]['faithfulness']["cardiffnlp/twitter-roberta-base-sentiment-latest"]['mean'][index], 
                    "perplexity": evaluation["output"]["perplexity"]["correct_perplexity"],
                    'mode': mode,
                }
            )
        except FileNotFoundError:
            pass
    
sentiment_results = pd.DataFrame(sentiment_results)
sentiment_results.to_csv("eval/processed/sentiment.csv")


########################################
# Extract personality results
########################################

results = []

for index in range(106):
    folder = "persona"
    formula = open(f"eval/{folder}/{index}/formula.txt", "r").read()
    weights = open(f"eval/{folder}/{index}/weights.txt", "r").read()
    numerical_re = re.compile(r"[-+]?\d*\.\d+|\d+")
    leave_only_numeric = lambda input_: numerical_re.findall(input_)[0]
    weights = [float(leave_only_numeric(weight + ",, 0")) for weight in weights.split(",")]
    evaluation = json.load(open(f"eval/{folder}/{index}/evaluation.json", "r"))
    monitor = json.load(open(f"eval/{folder}/{index}/monitor.json", "r"))
    divergence = extract_KL(monitor)
    shortness_function = [key for key in evaluation["output"]["faithfulness"] if key.startswith("<function")][0]
    if "Classifier" in formula:
        total_weight = weights[0] + 1 + weights[1]
    else:
        total_weight = weights[0] + 1 + weights[1]
        
    non_one_weight_index = np.where(np.array(weights) != 1)[0]
    if len(non_one_weight_index) == 0 and index > 0:
        weights_old = open(f"eval/{folder}/{index - 1}/weights.txt", "r").read()
        weights_old = [float(weights_old.replace("[", "").replace("]", "")) for weights_old in weights_old.split(",")]
        non_one_weight_index = np.where(np.array(weights) != 1)[0]
    non_one_weight_index = non_one_weight_index[0]
    
    results.append(
        {
            "formula": formula,
            "weights": weights,
            "divergence": divergence,
            "angry": evaluation["output"]["faithfulness"]["SamLowe/roberta-base-go_emotions"]["mean"][9],
            "sports": evaluation["output"]["faithfulness"]["cardiffnlp/tweet-topic-21-multi"]["mean"][16],
            "educational": evaluation["output"]["faithfulness"]["cardiffnlp/tweet-topic-21-multi"]["mean"][10],
            "formality": evaluation["output"]["faithfulness"]["s-nlp/roberta-base-formality-ranker"]["mean"][1],
            "sentiment": evaluation["output"]["faithfulness"]["cardiffnlp/twitter-roberta-base-sentiment-latest"]["mean"][2],
            "simple": evaluation["output"]["faithfulness"][shortness_function]["mean"],
            "perplexity": evaluation["output"]["perplexity"]["correct_perplexity"],
            "index": index,
            "sum_weights": total_weight,
            "relative_weight": weights[non_one_weight_index] / total_weight
        }
    )
results = pd.DataFrame(results)
    
formula1_indices = {
    "simple":[16,27],
    "sentiment": [5,16],
    "sports":[27,38],   
}

formula2_indices = {
    "simple": [49,60],
    "formality": [38,49],
    "educational": [60,71],
}

formula1_indices2 = {
    "simple":[78,83],
    "sentiment": [72,77],
    "sports":[83,88],   
}

formula2_indices2 = {
    "simple": [96,101],
    "formality": [89,95],
    "educational": [102,102],
}

order_first = {
    "formality": 1,
    "simple": 3, 
    "sports": 4, 
    "sentiment": 2,
    "educational": 0
}

fig, ax = plt.subplots(2, 3, figsize=(9, 4.5))

colors = sns.color_palette("colorblind", len(formula1_indices) + len(formula2_indices))


plt.rcParams.update({'font.size': 12})
sns.despine(left=True, bottom=True)

def description(character, formula_index):
    add_formula = r" ($F_" + formula_index + r"$)"
    if character != "simple":
        pass
    else:
        character = "simplicity"
        
    return character.capitalize() + add_formula

def plot(character, i, index0, index1, formula, ax, index2, index3):
    print(index0, index1)
    y = np.array([results[character][j] for j in range(index0, index1)])
    y2 = np.array([results[character][j] for j in range(index2, index3)])
    y = np.concatenate([y, y2])
    if character == "simple":
        y = (1 - y / 10)
    x_values = np.array([results["relative_weight"][j] for j in range(index0, index1)])
    x_2 = np.array([results["relative_weight"][j] for j in range(index2, index3)])
    x_values = np.concatenate([x_values, x_2])
    y_values = y
    sns.lineplot(
        x=x_values,
        y=y_values,
        color=colors[i],
        ax=ax
    )
    if character == "educational":
        return
    index_ = order_first[character]
    actual_height = results.iloc[index_][character]
    if character == "simple":
        actual_height = (1 - actual_height / 10)
    sns.lineplot(
        x=[min(x_values), max(x_values)],
        y=[actual_height, actual_height],
        color=colors[i],
        alpha=0.5,
        linestyle="--", 
        ax=ax
    )
    # sns.scatterplot(x=x_values, y=y_values, color=colors[i], edgecolor=None, s=20, alpha=0.5)

for i, character in enumerate(formula1_indices):
    ax[0][i].set_facecolor((0.95, 0.95, 0.95))
    ax[0][i].title.set(x=0.1)
    # set the title font to something fancy
    ax[0][i].title.set_fontfamily('sans-serif')
    ax[0][i].set_title(description(character, "1"))
    ax[0][i].set_xlabel("Relative Strength", fontsize=12)
    plot(character, i, formula1_indices[character][0], formula1_indices[character][1], "1", ax=ax[0][i], index2=formula1_indices2[character][0], index3=formula1_indices2[character][1])
    
for i, character in enumerate(formula2_indices):
    ax[1][i].set_facecolor((0.95, 0.95, 0.95))
    ax[1][i].title.set(x=0.1)
    # set the title font to something fancy
    ax[1][i].title.set_fontfamily('sans-serif')
    ax[1][i].set_title(description(character, "2"))
    ax[1][i].set_xlabel("Relative Strength", fontsize=12)
    plot(character, i + len(formula1_indices), formula2_indices[character][0], formula2_indices[character][1], "2", ax=ax[1][i], index2=formula2_indices2[character][0], index3=formula2_indices2[character][1])


fig.tight_layout()
fig.savefig("eval/plots/persona.pdf")
