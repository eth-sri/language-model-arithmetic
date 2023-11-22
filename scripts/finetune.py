from model_arithmetic import CustomDataset, load_model, load_tokenizer
import pandas as pd
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import torch
import os
from transformers import set_seed
from sklearn.metrics import accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

set_seed(42)

model_name = "SkolkovoInstitute/roberta_toxicity_classifier"
model = load_model(model_name, classification=True, dtype=torch.float32)
tokenizer = load_tokenizer(model_name)
data = pd.read_csv("data/datasets/jigsaw_balanced_processed.csv")
data["label"] = 1 - data["label"]
dataset = CustomDataset(tokenizer, data, random_cutoff=True)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, random_state=42)

training_args = TrainingArguments(
    output_dir="finetune/toxicity_classifier", 
    num_train_epochs=5,                        
    per_device_train_batch_size=64,    
    per_device_eval_batch_size=64,     
    warmup_ratio=0.05,                              
    weight_decay=0.01,                                    
    logging_steps=1000,
    learning_rate=1e-5,
    save_steps=50000,
    save_total_limit=1,
    eval_steps=50000,
    evaluation_strategy="steps",
    save_strategy="steps",
    bf16=False,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,  
    train_dataset=train_dataset,
    eval_dataset=test_dataset, 
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()
os.makedirs("finetune/toxicity_classifier", exist_ok=True)
# save the model
trainer.save_model("finetune/toxicity_classifier")

set_seed(42)

model_name = "roberta-base"
model = load_model(model_name, classification=True, dtype=torch.float32)
tokenizer = load_tokenizer(model_name)
data = pd.read_csv("data/datasets/IMDB_processed.csv")
data = data.sample(frac=1, random_state=42)
dataset = CustomDataset(tokenizer, data, random_cutoff=True)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, random_state=42)

training_args = TrainingArguments(
    output_dir="finetune/sentiment_classifier", 
    num_train_epochs=5,                        
    per_device_train_batch_size=64,    
    per_device_eval_batch_size=64,     
    warmup_ratio=0.05,                              
    weight_decay=0.01,                                    
    logging_steps=100,
    learning_rate=1e-5,
    save_steps=1000,
    save_total_limit=1,
    eval_steps=500,
    evaluation_strategy="steps",
    save_strategy="steps",
    bf16=False,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,  
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # compute accuracy as well
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()
os.makedirs("finetune/sentiment_classifier", exist_ok=True)
# save the model
trainer.save_model("finetune/sentiment_classifier")

set_seed(42)
model_name = "roberta-base"
model = load_model(model_name, classification=True, dtype=torch.float32)
tokenizer = load_tokenizer(model_name)
data = pd.read_csv("data/datasets/IMDB_processed.csv")
data = data.sample(frac=1, random_state=42)
dataset = CustomDataset(tokenizer, data, random_cutoff=False)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, random_state=42)

training_args = TrainingArguments(
    output_dir="finetune/sentiment_all", 
    num_train_epochs=5,                        
    per_device_train_batch_size=64,    
    per_device_eval_batch_size=64,     
    warmup_ratio=0.05,                              
    weight_decay=0.01,                                    
    logging_steps=100,
    learning_rate=1e-5,
    save_steps=1000,
    save_total_limit=1,
    eval_steps=500,
    evaluation_strategy="steps",
    save_strategy="steps",
    bf16=False,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,  
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # compute accuracy as well
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()
os.makedirs("finetune/sentiment_all", exist_ok=True)
# save the model
trainer.save_model("finetune/sentiment_all")