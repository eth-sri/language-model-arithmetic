from .base import BaseClass
import pandas as pd
from loguru import logger
import numpy as np
from transformers import set_seed, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from .dataset import CustomDataset
from sklearn.model_selection import train_test_split
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from .basic_model_loader import load_model, load_tokenizer
import os
from .model_arithmetic import ModelArithmetic
from googleapiclient import discovery
import json
from dotenv import load_dotenv
import time
from torch.utils.data import DataLoader
from .utils import ENABLE_LOGGING, log

load_dotenv()

class Evaluation(BaseClass):
    """
    This class is used for evaluating a model's performance on a given dataset.
    It includes methods for preparing the dataset, evaluating the model, generating samples, 
    calculating perplexity, faithfulness, and performance of the model.
    """
    def __init__(self, generator=None, dataset_location=None, dataset=None, train_dataset=None, train_dataset_location=None, 
                 n_input_words=5, bleurt_checkpoint="../models/BLEURT-20", **kwargs):
        """
        Initialize the Evaluation class with the given parameters.

        Args:
            generator (ModelArithmetic, optional): The model to be evaluated.
            dataset_location (string, optional): The location of the dataset to be used for evaluation. Either this or dataset should be provided. Dataset should contain column "text", "input", "output and "label" ("label", "input", "output" optional)
            dataset (pd.DataFrame, optional): The dataset to be used for evaluation. Either this or dataset_location should be provided. Dataset should contain column "text", "input", "output and "label" ("label", "input", "output" optional)
            train_dataset (pd.DataFrame, optional): The dataset to be used for training the model. Only used when calculating the faithfulness of the model and when the downstream model still needs to be finetuned.
            train_dataset_location (string, optional): The location of the dataset to be used for training the model.
            n_input_words (int, optional): The number of input words to be used in the generator. Only used if the dataset does not contain the column "input".
            bleurt_checkpoint (string, optional): The location of the BLEURT model checkpoint.
            **kwargs: Additional keyword arguments.
        """
        self.has_input_task = True
        self.dataset = None
        if dataset is not None:
            self.dataset = dataset.copy()
        elif dataset_location is not None:
            self.dataset = pd.read_csv(dataset_location, escapechar='\\', lineterminator="\n")

        if train_dataset is not None:
            self.train_dataset = train_dataset
        elif train_dataset_location is not None:
            self.train_dataset = pd.read_csv(train_dataset_location, escapechar='\\', lineterminator="\n")
        else:
            self.train_dataset = None

        if self.dataset is not None:
            self.prepare_dataset(n_input_words)

        super().__init__(**kwargs, dataset_location=dataset_location, generator=generator, 
                         has_input_task=self.has_input_task, output=dict(), extra_kwargs=None, bleurt_checkpoint=bleurt_checkpoint, 
                         train_dataset_location=None)
        
        if isinstance(generator, ModelArithmetic):
            # If we don't do it this way, we can't store the evaluator because ModelArithmetic is not serializable
            del self.kwargs["generator"]
            self.kwargs["formula"] = generator.formula
            self.formula = generator.formula

    def prepare_dataset(self, n_input_words=5):
        """
        Prepares the dataset for evaluation. If the dataset does not have an input column, 
        it assumes the input is the first n_input_words words of the output. If the dataset 
        does not have a label column, it assumes all labels are 1.

        Args:
            n_input_words (int): The number of input words to be used.
        """
        log(logger.debug, "Preparing dataset")
        if "input" not in self.dataset.columns:
            log(logger.debug, f"No input column found, assuming input is the first {n_input_words} words of the output")
            self.dataset["input"] = self.dataset["text"].apply(lambda x: " ".join(x.split()[:n_input_words]))
            self.dataset["output"] = self.dataset["text"].apply(lambda x: " " + " ".join(x.split()[n_input_words:]))
            self.has_input_task = False
        
        if "label" not in self.dataset.columns:
            log(logger.debug, "No label column found, assuming all labels are 1")
            self.dataset["label"] = 1

    def evaluate_lm_eval(self, model, task_name, batch_size, num_fewshot, model_args, limit=None, write_out=False, **kwargs):
        """
        Evaluates the model using the lm_eval package. 

        Args:
            model (PreTrainedModel): The model to be evaluated.
            task_name (string): The name of the task for evaluation.
            batch_size (int): The batch size to be used for evaluation.
            num_fewshot (int): The number of fewshot examples to be used for evaluation.
            model_args (dict): The arguments to be passed to the model.
            limit (int, optional): The maximum number of examples to be used for evaluation.
            write_out (bool, optional): Whether to write out the results or not.
            **kwargs: Additional keyword arguments.
        """
        try:
            from lm_eval import evaluator
            from lm_eval.models.huggingface import HFLM
        except ImportError:
            raise ImportError("Please install lm_eval to run this function")
        
        results = evaluator.simple_evaluate(
            model=HFLM(model),
            model_args=model_args,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_cache=None,
            limit=limit,
            write_out=write_out
        )
        if "lm_eval" in self.output:
            self.output["lm_eval"][task_name] = results
        else:
            self.output["lm_eval"] = {task_name: results}

    def evaluate(self, max_tokens=128, store_file=None, reload=True, 
                 dataset_file=None, reload_data=True, preserve_memory=False, batch_size=1, do_perspective=True,
                 speculation=False,
                 **kwargs):
        """
        Evaluates the model on the dataset and calculates the perplexity, faithfulness, and performance.

        Args:
            max_tokens (int, optional): The maximum number of tokens to be used for evaluation.
            store_file (string, optional): The file to store the evaluation results.
            reload (bool, optional): Whether to reload the dataset or not if it was stored before.
            dataset_file (string, optional): The file containing the dataset. If path exists, dataset is loaded from path. If path does not exist, dataset is saved to path.
            reload_data (bool, optional): Whether to reload the data or not if it was stored before.
            preserve_memory (bool, optional): Whether to preserve memory or not.
            batch_size (int, optional): The batch size to be used for evaluation.
            do_perspective (bool, optional): Whether to calculate the perspective score or not.
            speculation (bool, optional): Whether to use speculation or not.
            **kwargs: Additional keyword arguments.
        """
        if store_file is not None:
            os.makedirs(os.path.dirname(store_file), exist_ok=True)
        if dataset_file is not None:
            os.makedirs(os.path.dirname(dataset_file), exist_ok=True)

        if (reload_data or reload) and dataset_file is not None and os.path.isfile(dataset_file):
            log(logger.debug, f"Reloading dataset from {dataset_file}")
            self.dataset = pd.read_csv(dataset_file, escapechar='\\', lineterminator="\n")
            self.dataset.fillna("", inplace=True)
        else:
            log(logger.debug,"Generating samples")
            self.generate_samples(max_tokens, batch_size=batch_size, speculation=speculation, **kwargs)
            if dataset_file is not None:
                log(logger.info, f"Saving dataset to {dataset_file}")
                self.dataset.to_csv(dataset_file, index=False, escapechar='\\', lineterminator="\n")
        
        if reload and store_file is not None and os.path.isfile(store_file):
            self.output = json.load(open(store_file))["output"]
            log(logger.info, f"Reloading output. Output is {self.output}")
        if not reload:
            self.output = dict()

        self.dataset.dropna(inplace=True)

        if preserve_memory and hasattr(self.generator, "clear_memory"):  # can have weird side effects!
            self.generator.clear_memory()

        self.perplexity(self.dataset, **kwargs)
        self.faithfulness_multiple(self.dataset, **kwargs)
        self.performance(self.dataset, **kwargs)
        if do_perspective:
            self.perspective(**kwargs)
            if dataset_file is not None:
                log(logger.info, f"Saving dataset to {dataset_file}")
                self.dataset.to_csv(dataset_file, index=False, escapechar='\\', lineterminator="\n")
        
        self.extra_kwargs = {kwarg: str(kwargs[kwarg]) for kwarg in kwargs}
        self.extra_kwargs["max_tokens"] = max_tokens
        if store_file is not None:
            log(logger.info, f"Saving output to {store_file}")
            self.save(store_file)

        
        return self.output
    
    def perspective(self, classification_with_input=True, **kwargs):
        """
        Calculates the perspective score of the generated sentences.
        
        Args:
            classification_with_input (bool, optional): Whether to use the input of the sentence for calculating the perspective score or not.
            **kwargs: Additional keyword arguments.
        """
        retries = 0
        if "perspective" in self.output:
            log(logger.info, f"Reloading perspective. Perspective is {self.output['perspective']}")
            return self.output["perspective"]
        log(logger.info, "Calculating perspective")
        service = discovery.build('commentanalyzer', 'v1alpha1', 
                                  developerKey=os.environ["PERSPECTIVE_API_KEY"],
                                  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                                  static_discovery=False)
        scores = []
        self.dataset["perspective"] = [None] * len(self.dataset)
        for index, sample in self.dataset.iterrows():
            sentence = sample['generated']
            if classification_with_input:
                sentence = sample['input'] + sentence

            analyze_request = {
                'comment': { 'text': sentence },
                'languages': ["en"],
                'requestedAttributes': {'TOXICITY': {}}
            }
            while retries < 10 ** 5:
                try:
                    response = service.comments().analyze(body=analyze_request).execute()
                    break
                except Exception as e:
                    if "LANGUAGE_NOT_SUPPORTED_BY_ATTRIBUTE" in str(e):
                        continue
                    log(logger.warning, f"Exception {e} occurred, retrying...")
                    retries += 1
                    time.sleep(10)
            if retries == 10 ** 5:
                log(logger.error, "Could not get all perspective scores")
                break
            value = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            self.dataset.at[index, "perspective"] = value
            scores.append(value)
            time.sleep(0.1)
        result = np.mean(scores)
        log(logger.info, f"Perspective is {result:.6f}")
        self.output["perspective"] = {
            "mean": result,
            "std": np.std(scores),
        }
        return result

    def generate_samples(self, max_tokens, batch_size=1, temperature=1.0, top_p=1.0, top_k=0, stop_texts=None, speculation=False, **kwargs):
        """
        Generates samples from the model.

        Args:
            max_tokens (int): The maximum number of tokens to be used for evaluation.
            batch_size (int, optional): The batch size to be used for evaluation.
            temperature (float, optional): The temperature to be used for sampling.
            top_p (float, optional): The top-p value to be used for sampling.
            top_k (int, optional): The top-k value to be used for sampling.
            stop_texts (list, optional): The list of texts at which sampling should be stopped
            speculation (bool, optional): Whether to use speculation or not.
            **kwargs: Additional keyword arguments.
        """
        start_time = time.time()
        if "generated" not in self.dataset.columns:
            texts = self.generator.generate_text(self.dataset["input"].tolist(), max_new_tokens=max_tokens, 
                                                 batch_size=batch_size, temperature=temperature, 
                                                 top_p=top_p, top_k=top_k, stop_texts=stop_texts, do_speculation=speculation)
            self.dataset["generated"] = texts
        end_time = time.time()
        
        self.output["time"] = {
            "total_time": end_time - start_time,
            "time_per_sample": (end_time - start_time) / len(self.dataset),
            "dataset_size": len(self.dataset),
            "max_tokens": max_tokens,
            "batch_size": batch_size
        }

    def save_generated(self, output_location):
        """
        Saves the generated samples to the specified location.

        Args:
            output_location (string): The location to save the generated samples.
        """
        log(logger.debug, f"Saving generated samples to {output_location}")
        self.dataset.to_csv(output_location)

    def compute_perplexities(self, dataset, model, tokenizer, **kwargs):
        """
        Calculates the perplexity of the generated sentences.

        Args:
            dataset (pd.DataFrame): The dataset to be used for evaluation. Has columns "input" (for input text), "generated" (for generated text). 
            model (PreTrainedModel): The model to be evaluated.
            tokenizer (Tokenizer): The tokenizer to be used for tokenizing the sentences.
            **kwargs: Additional keyword arguments.
        """
        perplexities = []
        sum_nllos = 0
        n_tokens = 0

        for index, sample in dataset.iterrows():
            input_sentence = sample['input']
            sentence = sample['generated']
            if len(sentence) == 0:
                continue
            combined_sentence = input_sentence + sentence

            encodings = tokenizer(combined_sentence, return_tensors='pt')
            input_ids = encodings['input_ids'].to(model.device)
            attention_mask = encodings['attention_mask'].to(model.device)

            input_encodings = tokenizer(input_sentence, return_tensors='pt')
            input_ids_inputs = input_encodings['input_ids']
            input_length = input_ids_inputs.size(1)

            with torch.no_grad():
                output = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                logprobs = output.logits[0, :].log_softmax(dim=-1)

            loss_func = torch.nn.NLLLoss(ignore_index=-100, reduction='sum')

            loss = loss_func(logprobs[..., input_length:-1, :].contiguous(), input_ids[0, :][..., input_length+1:].contiguous())
            loss = loss.to(torch.float32).detach().cpu().numpy()
            n_tokens_here = input_ids.shape[-1] - input_length - 1
            if n_tokens_here > 0:
                perplexity = np.exp(loss / n_tokens_here)
                sum_nllos += loss
                n_tokens += n_tokens_here
                if not np.isnan(perplexity):
                    perplexities.append(perplexity)
                else:
                    perplexities.append(None)
            else:
                perplexities.append(None)

        return perplexities, sum_nllos, n_tokens

    def get_perplexity(self, dataset, model, tokenizer, **kwargs):
        """
        Calculates the perplexity of the generated sentences.

        Args:
            dataset (pd.DataFrame): The dataset to be used for evaluation. Has columns "input" (for input text), "generated" (for generated text). 
            model (PreTrainedModel): The model to be evaluated.
            tokenizer (Tokenizer): The tokenizer to be used for tokenizing the sentences.
            **kwargs: Additional keyword arguments.
        """
        perplexities, sum_nllos, n_tokens = self.compute_perplexities(dataset, model, tokenizer, **kwargs)

        average = np.mean(perplexities)
        median = np.median(perplexities)
        real = np.exp(sum_nllos / n_tokens)

        return {
            "average": average,
            "median": median,
            "correct_perplexity": real
        }

    def perplexity(self, dataset, model_name_fluency="gpt2-xl", dtype=torch.float16, **kwargs):
        """
        Calculates the perplexity of the generated sentences.

        Args:
            dataset (pd.DataFrame): The dataset to be used for evaluation. Has columns "input" (for input text), "generated" (for generated text). 
            model_name_fluency (string, optional): The name of the model to be used for calculating fluency.
            dtype (torch.dtype, optional): The data type to be used for the model.
            **kwargs: Additional keyword arguments.
        """
        log(logger.info, "Calculating fluency")
        if "perplexity" in self.output:
            log(logger.info, f"Reloading perplexity. Perplexity is {self.output['perplexity']}")
            return self.output["perplexity"]
        tokenizer = load_tokenizer(model_name_fluency)
        model = load_model(model_name_fluency, dtype=dtype)

        self.output["perplexity"] = self.get_perplexity(dataset, model, tokenizer)
        log(logger.info, f"Perplexity is {self.output['perplexity']}")
        del model
        torch.cuda.empty_cache()
        return self.output["perplexity"]
    
    def faithfulness_multiple(self, dataset, model_name, **kwargs):
        """Calculates the faithfulness for all stored classifiers

        Args:
            dataset (pd.DataFrame): The dataset to be used for evaluation.
            model_name (str, list of strings): Classifier names to use
        """
        if not isinstance(model_name, (list, tuple)):
            model_name = [model_name]
        
        results = dict()
        for model in model_name:
            name = model
            if not isinstance(name, str):
                name = model.__str__()
            results[name] = self.faithfulness(dataset, model_name=model, **kwargs)
            
        self.output["faithfulness"] = results
        
        return results

    def faithfulness(self, dataset, finetune_model=True, classification_with_input=True, model_name="distilbert-base-uncased", model=None, test_size=0.2, max_length=128, epochs=3, batch_size_faithfulness=16, 
                     learning_rate=2e-5, warmup_steps=500, weight_decay=0.01, save_model_folder=None, dtype_faithfulness=torch.float32, store_faithfulness=False, 
                     **kwargs):
        """
        Calculates the faithfulness of the generated sentences.

        Args:
            dataset (pd.DataFrame): The dataset to be used for evaluation. Has columns "input" (for input text), "generated" (for generated text). If finetuning, also has column "output" (for output ground truth).
            finetune_model (bool, optional): Whether to finetune the model or not.
            classification_with_input (bool, optional): Whether to use the input of the sentence for classification or not.
            model_name (str, optional): The name of the model to be used for classification (either path or name). Either this or model should be provided.
            model (PreTrainedModel, optional): The model to be used for classification. Either this or model_name should be provided.
            test_size (float, optional): The size of the test set to be used for evaluation.
            max_length (int, optional): The maximum length of the sentences to be used for evaluation.
            epochs (int, optional): The number of epochs to be used for training the model.
            batch_size_faithfulness (int, optional): The batch size to be used for evaluation.
            learning_rate (float, optional): The learning rate to be used for training the model.
            warmup_steps (int, optional): The number of warmup steps to be used for training the model.
            weight_decay (float, optional): The weight decay to be used for training the model.
            save_model_folder (str, optional): The folder to save the trained model.
            dtype_faithfulness (torch.dtype, optional): The data type to be used for the model.
            store_faithfulness (bool, optional): Whether to store the resulting score or not.
            **kwargs: Additional keyword arguments.
        """
        log(logger.info, "Calculating faithfulness")

        if ("label" not in dataset.columns or all(dataset["label"] == 1) or all(dataset["label"] == 0)) and finetune_model:
            log(logger.info, "Dataset does not have good labels, cannot calculate faithfulness")
            return None
        
        if "faithfulness" in self.output:
            log(logger.info, f"Reloading faithfulness. Faithfulness is {self.output['faithfulness']}")
            return self.output["faithfulness"]
        
        set_seed(42)
        
        df = dataset.copy()
        if classification_with_input:
            df["text"] = df["input"] + df["generated"]
        else:
            df["text"] = df["generated"]
    
        if isinstance(model_name, str):
            tokenizer = load_tokenizer(model_name)
            args = TrainingArguments(
                output_dir="../finetune/eval/random",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size_faithfulness,
                per_device_eval_batch_size=batch_size_faithfulness,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                logging_dir="logs",
                logging_steps=100,
                learning_rate=learning_rate,
            )
            if model is None:
                log(logger.info, "Loading model")
                model = load_model(model_name, classification=True, dtype=dtype_faithfulness)
                # we need to train the model on the dataset
                
                if finetune_model:
                    log(logger.info, "Finetuning model")
                    df = dataset.copy()
                    df = df.dropna()
                    df["text"] = df["input"] + df["output"]
                    train, val = train_test_split(df, test_size=test_size)
                    train_dataset = CustomDataset(tokenizer, train, max_tokens=max_length)
                    val_dataset = CustomDataset(tokenizer, val, max_tokens=max_length)
                    trainer = Trainer(
                        model=model,
                        tokenizer=tokenizer,
                        args=args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        data_collator=DataCollatorWithPadding(tokenizer, padding="longest"),
                    )
                
                    trainer.train()
                    if save_model_folder is not None:
                        trainer.save_model(save_model_folder)
            
            log(logger.info, "Evaluating model")
            test_dataset = CustomDataset(tokenizer, df, max_tokens=max_length)
            # get the predictions
            dataloader = DataLoader(test_dataset, batch_size=batch_size_faithfulness, 
                                    collate_fn=DataCollatorWithPadding(tokenizer, padding="longest"))
            predictions = []
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    outputs = model(input_ids=batch.input_ids.to(model.device), 
                                    attention_mask=batch.attention_mask.to(model.device))
                    probs = torch.softmax(outputs.logits, dim=-1)
                    
                    # Append the batch's predictions
                    predictions = predictions + list(probs.to(float).cpu().numpy())

            # Converting the final predictions list to array
            predictions = np.array(predictions)
        else:
            predictions = np.array([model_name(sample) for sample in df["text"]])

        average_prediction = np.nanmean(predictions, axis=0)
        std = np.nanstd(predictions, axis=0)
        if not isinstance(average_prediction, (int, float)):
            average_prediction = list(average_prediction)
            std = list(std)
        log(logger.info, f"Faithfulness is {average_prediction}")
        
        if store_faithfulness:
            self.output["faithfulness"] ={
                "mean": average_prediction,
                "std": std
            }

        del model
        torch.cuda.empty_cache()

        return {
            "mean": average_prediction,
            "std": std        
        }

    def performance(self, dataset, **kwargs):
        """
        Calculates the performance of the model on the dataset. Only calculates the performance if the dataset has an output column (which is the ground truth).

        Args:
            dataset (pd.DataFrame): The dataset to be used for evaluation. Has columns "input" (for input text), "generated" (for generated text) and "output" (output ground truth). 
            **kwargs: Additional keyword arguments.
        """
        # need to measure the average overlap between the output and the generated text
        log(logger.info, "Calculating performance")
        if "performance" in self.output:
            log(logger.info, f"Reloading performance. Performance is {self.output['performance']}")
            return self.output["performance"]

        if "output" not in dataset.columns:
            log(logger.info, "Dataset does not have output column, cannot calculate performance")
            return None
        
        class Args:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        args = Args(smooth_method="exp", smooth_value=None, lc=False, tokenize="none", force=False)
        try:
            bleu = BLEU(args=args)
        except Exception:
            bleu = BLEU()
        bleu_score = bleu.corpus_score(dataset["generated"].tolist(), [dataset["output"].tolist()]).score

        try:
            from bleurt import score
            scorer = score.BleurtScorer(self.bleurt_checkpoint)
            bleurt_score = np.mean(scorer.score(candidates=dataset["generated"].tolist(), 
                                                references=dataset["output"].tolist(), batch_size=1))
            del scorer
            torch.cuda.empty_cache()
        except ImportError:
            log(logger.warning, "Could not import BLEURT, skipping BLEURT score")
            bleurt_score = 0
        except AssertionError as e:
            log(logger.warning, f"Could not score BLEURT, {e}")
            bleurt_score = 0

        rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        # Loop through the dataset
        for i in range(len(dataset)):
            # Calculate the Rouge scores (rouge1, rouge2, and rougeL)
            scores = rouge.score(dataset["generated"].iloc[i], dataset["output"].iloc[i])
            
            # Add the scores to the corresponding lists
            rouge1_scores.append(scores["rouge1"])
            rouge2_scores.append(scores["rouge2"])
            rougeL_scores.append(scores["rougeL"])

        # Calculate the average Rouge scores for each metric
        avg_rouge1 = np.nanmean([score.fmeasure for score in rouge1_scores])
        avg_rouge2 = np.nanmean([score.fmeasure for score in rouge2_scores])
        avg_rougeL = np.nanmean([score.fmeasure for score in rougeL_scores])

        output = {
            "bleu": bleu_score,
            "bleurt": bleurt_score,
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL
        }
        
        log(logger.info, f"Performance is {output}")
        self.output["performance"] = output

        return output
