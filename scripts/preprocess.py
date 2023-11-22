import pandas as pd
import json
import numpy as np
import os
import re

def preprocess_IMDB(dataset_location, save_location):
    data = pd.read_csv(dataset_location)
    data['label'] = data.apply(lambda row: 1 if row['sentiment'] == 'positive' else 0, axis=1)
    data['text'] = data['review']
    data = data[['text', 'label']]
    data.to_csv(save_location, index=False)

def preprocess_jigsaw(dataset_location, save_location, reproduction=False):
    data = pd.read_csv(dataset_location)
    dataset = data[np.array(data["toxicity"] == 0.0) | np.array(data["toxicity"] >= 0.5)]
    dataset["label"] = 1 - dataset["toxicity"].apply(lambda x: 1 if x >= 0.5 else 0)
    dataset["text"] = dataset["comment_text"]
    dataset = dataset[['text', 'label']]
    # unfortunately the original code to get to the balanced data got lost. We therefore map the indices manually, but note that this
    # just selects elements from the original dataset, such that it becomes a balanced dataset
    if reproduction:
        def read_indices(filename):
            with open(filename) as f:
                content = f.readlines()
            # remove whitespace characters like `\n` at the end of each line
            content = [int(x.strip()) for x in content] 
            return content

        indices = read_indices("mapping/jigsaw_balanced_indices.txt")
        # go from data to data_balanced by applying the indices
        data_balanced = dataset.iloc[indices]
    else:
        data_toxic = dataset[dataset["label"] > 0.5]
        data_non_toxic = dataset[dataset["label"] < 0.5]
        data_balanced = pd.concat([data_toxic, data_non_toxic.sample(len(data_toxic), random_state=42)])
        
    data_balanced.to_csv(save_location, index=False)

def preprocess_pol(dataset_location, save_location):
    # "../data/datasets/pol_062016-112019_labeled.ndjson"
    data = pd.read_json(dataset_location, 
                    lines=True, nrows=1000000, chunksize=100000)
    
    def contains_html(element):
        # return true if contains html or link
        return bool(re.search("<.*?>", element)) or bool(re.search("http", element))
    
    resulting_data = []
    while True:
        # break if no more data
        try:
            data1 = next(data)
        except:
            break
        for posts in data1["posts"]:
            post = posts[0]
            if "com" in post and not contains_html(post["com"]):
                resulting_data.append(
                    {
                        "text": post["com"],
                        "toxicity": post["perspectives"]["TOXICITY"]
                    }
                )
    data = pd.DataFrame(resulting_data)
    data.to_csv(save_location, index=False)
    
    

def preprocess_alpaca(dataset_location, save_location):
    json_data = json.load(open(dataset_location))
    resulting_data = []
    for element in json_data:
        input_ = element["instruction"] + "\n"
        if element["input"] != "":
            input_ += element["input"] + "\n"
        
        resulting_data.append(
            {
                "input": input_,
                "output": element["output"]
            }
        )
        
    data = pd.DataFrame(resulting_data)
    data.to_csv(save_location, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduction", action="store_true")
    args = parser.parse_args()
    
    preprocess_alpaca("../data/datasets/alpaca_data.json", "../data/datasets/alpaca_processed.csv")
    preprocess_jigsaw("../data/datasets/all_data.csv", "../data/datasets/jigsaw_balanced_processed.csv", reproduction=args.reproduction)
    preprocess_pol("../data/datasets/pol_062016-112019_labeled.ndjson", "../data/datasets/pol.csv")
    preprocess_IMDB("data/datasets/IMDB Dataset.csv", "data/datasets/IMDB_processed.csv")