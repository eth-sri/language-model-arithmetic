from .operators import Operator
import numpy as np
import torch
from typing import Dict
import os
import json
from .basic_model_loader import load_model, load_tokenizer
from .utils import get_max_length, ENABLE_LOGGING, log
from loguru import logger
import dis


class RunnableOperator(Operator):
    def __init__(self, system_prompt="", model=None, speculative_factor=1, 
                 prompt_template = lambda system_prompt, input_string: system_prompt + input_string, run_priority=0, group=None, 
                 outputs_logprobs=True, **kwargs):
        # TODO: make save and load possible when filtering the input_string in the prompt_template, use byte64 and pickle
        """
        Initialize a runnable operator instance. A runnable operator is an operator that generates a probability distribution instead of modifies an existing one.
        
        Args:
            system_prompt (str): String to be used as a prompt. Only used in specific runnable operators
            model (optional): Model to be used for operation. If None, the model must be set later to the default model to be used.
            speculative_factor (int): Factor for speculative sampling.
            prompt_template (callable): Function for generating prompt. Takes two arguments: system_prompt and input_string. The operator will be run on prompt_template(..., ...) + continuation_tokens
            run_priority (int): Priority for running the operation. Higher priority means the operation will be run first, especially important for the classifier.
            group (optional): Group to which the operator belongs. This ensures that speculative sampling will not be tried when not all operators of a group are finished.
            outputs_logprobs (bool): Whether the operator outputs logprobs.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(speculative_factor=speculative_factor, model=model, system_prompt=system_prompt,
                         prompt_template=prompt_template, run_priority=run_priority, group=group, outputs_logprobs=outputs_logprobs, **kwargs)
        self.cache = None

    def get_model_name(self):
        return self.model

    def set_system_prompt(self, system_prompt):
        """
        Sets the prompt string for the operation.
        Args:
            system_prompt (str): String to be used as a prompt.
        """
        self.system_prompt = system_prompt
        
    def run_condition(self, new_tokens, trigger_end):
        """
        Determine if the run condition is met.
        
        Args:
            new_tokens (List[int]): Number of new tokens per sample in the batch
            trigger_end (List[bool]): Whether to trigger the end for each sample in the batch.
            
        Returns:
            bool: Whether the run condition is met.
        """
        new_tokens = [new_tokens[i] if not trigger_end[i]  or new_tokens[i] < 0 else max(new_tokens[i], self.speculative_factor) for i in range(len(new_tokens))]
        return np.mean(new_tokens) >= self.speculative_factor 
        # other possibility:
        # return np.max(new_tokens) + 1 >= speculative_factor
        
    def delete_cache(self, index=None, from_=None):
        """
        Delete the cache.
        """
        if from_ is None and index is None:
            self.cache = None
        
    def run(self, tokenized_inputs, **kwargs):
        """
        Run the operation. This method needs to be implemented by subclasses.
        
        Args:
            tokenized_inputs (torch.tensor): Inputs that have been tokenized.
            **kwargs: Arbitrary keyword arguments.
            
        Raises:
            NotImplementedError: This method needs to be implemented by subclasses.
        """
        raise NotImplementedError("This method needs to be implemented by subclasses.")
    
    def same_operator(self, other):
        """
        Determine if the other operator is the same as this one. This is important to avoid redundant runs of the same operator in a formula
        
        Args:
            other: Other operator to be compared.
            
        Returns:
            bool: Whether the other operator is the same as this one.
        """
        if isinstance(other, str):
            return self.id() == other
        elif isinstance(other, RunnableOperator):
            return self.id() == other.id()
        return False

    def norm(self, runnable_operator_outputs=None):
        """
        Compute the norm of the operator.
        
        Args:
            runnable_operator_outputs (optional): Outputs of runnable operators.
            
        Returns:
            int: The norm of the operator.
        """
        if runnable_operator_outputs is None or self.is_finished(runnable_operator_outputs):
            return 1
        return 0
    
    def is_finished(self, runnable_operator_outputs):
        """
        Determine if the operation is finished.
        
        Args:
            runnable_operator_outputs: Outputs of runnable operators.
            
        Returns:
            bool: Whether the operation is finished.
        """
        return any([self.same_operator(output) and runnable_operator_outputs[output] is not None for output in runnable_operator_outputs])
    
    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        """
        Evaluate the operation.
        
        Args:
            runnable_operator_outputs (Dict): Outputs of runnable operators.
            normalize (bool): Whether to normalize the evaluation.
            
        Returns:
            int: The evaluation of the operation.
        """
        for output in runnable_operator_outputs:
            if self.same_operator(output) and runnable_operator_outputs[output] is not None:
                return runnable_operator_outputs[output]
        return 0
    
    def generate_settings(self):
        """
        Generate settings for the operation.
        
        Returns:
            dict: Settings for the operation.
        """
        kwargs = super().generate_settings()
        kwargs["prompt_template"] = self.prompt_template("{{system_prompt}}", "{{input_string}}")
        return kwargs

    @staticmethod
    def load_from_settings(settings):
        """
        Load operator from settings.
        
        Args:
            settings (dict): Settings for the operation.
            
        Returns:
            Operator: Operator loaded from settings.
        """
        copy = settings["prompt_template"]
        prompt_template = lambda system_prompt, input_string: copy.replace("{{system_prompt}}", system_prompt).replace("{{input_string}}", input_string)
        settings["prompt_template"] = prompt_template
        return Operator.load_from_settings(settings)
    
    def get_prompt(self, input_string):
        """
        Get the prompt for the operation.
        
        Args:
            input_string (str): String to be used as input.
            
        Returns:
            callable: Function for generating prompt.
        """
        return self.prompt_template(self.system_prompt, input_string)
    
    def get_store_params(self):
        """
        Get parameters for storing the operation.
        
        Returns:
            dict: Parameters for storing the operation.
        """
        return {
            "class": self.__class__.__name__,
            "model": self.model,
            "speculative_factor": self.speculative_factor,
            "prompt_template": self.prompt_template(self.system_prompt, "{{input_string}}")
        }
        
    def id(self):
        """
        Get the ID of the operation.
        
        Returns:
            str: ID of the operation.
        """
        kwargs = self.kwargs.copy()
        return f"{self.__class__.__name__}(**{kwargs})"
    
    def load_model(self, dtype):
        """
        Load the model for the operation. Only needs to be overwritten when a model is necessary
        
        Args:
            dtype: Data type for the model.
            
        Returns:
            None
        """
        return None
    
    def initialize_after_model_set(self):
        """
        Initialize the operation after the model is set (to the default model if necessary).
        
        Raises:
            AssertionError: If the model is not set before initializing.
        """
        assert self.model is not None, "Model must be set before initializing."
        

class PromptedLLM(RunnableOperator):
    def __init__(self, system_prompt, model=None, speculative_factor=1, 
                 prompt_template = lambda system_prompt, input_string, : system_prompt + "\n" + input_string, dtype=None, group=None,
                 enable_cache=True, dim_keys_past=2, dim_values_past=2, run_eager=False, tokenizer=None, model_name=None, **kwargs):
        """
        Initializes an LLM Prompt. This is a runnable operator that uses a language model to generate a probability distribution.
        Args:
            system_prompt (str): String to be used as a prompt. Only used in specific runnable operators
            model (optional): Model to be used for operation. If None, the model must be set later to the default model to be used.
            speculative_factor (int): Factor for speculative sampling.
            prompt_template (callable): Function for generating prompt. Takes two arguments: system_prompt and input_string. The operator will be run on prompt_template(..., ...) + continuation_tokens
            run_priority (int): Priority for running the operation. Higher priority means the operation will be run first, especially important for the classifier.
            dtype (optional): Data type for the model.
            group (optional): Group to which the operator belongs. This ensures that speculative sampling will not be tried when not all operators of a group are finished.
            enable_cache (bool): Whether to enable the key-value cache.
            dim_keys_past (int): Dimension of the keys in the key-value cache. Usually 2, but for other models this can be different.
            dim_values_past (int): Dimension of the values in the key-value cache. Usually 2, but for other models this can be different.
            run_eager (bool): Whether to run the model in eager mode. This is necessary for some models, but incompatible with speculative sampling and some other features.
            tokenizer (Tokenizer): Tokenizer to be used for the operation. If None, the default tokenizer will be used.
            **kwargs: Arbitrary keyword arguments.
        """
        if dim_keys_past == 2 and dim_values_past == 2:
            # set the dims based on the model
            if model in ["tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct", "tiiuae/falcon-40b", "tiiuae/falcon-40b-instruct"]:
                dim_keys_past = 1
                dim_values_past = 1
        
        super().__init__(system_prompt=system_prompt, model=model, speculative_factor=speculative_factor, 
                         prompt_template=prompt_template, group=group, enable_cache=enable_cache, 
                         dim_keys_past=dim_keys_past, dim_values_past=dim_values_past, run_eager=run_eager, model_name=model_name)
        self.dtype = dtype
        self.tokenizer_length = None
        self.tokenizer = tokenizer
        self.previous_input_ids = None
        self.default_dim = 2
        if self.run_eager:
            log(logger.warning, "Eager mode is enabled. This will make several features, such as speculative sampling, inaccessible.")
        
    def load_model(self, dtype):
        """
        Loads the model for the operation.
        :param dtype: Data type for the model.
        """
        if not isinstance(self.model, str):
            return self.model
        if self.dtype is None:
            return load_model(self.model, dtype=dtype)
        return load_model(self.model, dtype=self.dtype)
    
    def initialize_after_model_set(self):
        if self.tokenizer is None:
            if isinstance(self.model, str):
                tokenizer = load_tokenizer(self.model)
            else:
                tokenizer = load_tokenizer(self.model.name_or_path)
            self.tokenizer_length = len(tokenizer)

    def get_model_name(self):
        if isinstance(self.model, str):
            return self.model
        elif self.model_name is not None:
            return self.model_name
        elif self.model is not None:
            return self.model.name_or_path
        
    def select_from_sample_cache(self, sample, from_=None, until=None):
        """Selects the cache from a sample that needs to be stored

        Args:
            sample (torch.tensor): Torch tensor, the samples key-value past as stored by the LLM
            from_ (int, optional): From which value to store the key-value past. Defaults to None.
            until (int, optional): Until which value to store the key-value past. Defaults to None.
        """
        for i in range(len(sample)):
            for j in range(len(sample[i])):
                sample[i][j] = sample[i][j][:, from_:until]
        
        return sample
    
    def swap_dimensions(self, sample):
        """Swaps dimensions in order to make the dimensions match the default dimensions. This is necessary because models do not use the same semantics for the key-value storage

        Args:
            sample (List[torch.tensor]): Key-value past as stored by the LLM
        """
        for i in range(len(sample)):
            # keys, values
            if self.default_dim != self.dim_keys_past:
                sample[i][0] = sample[i][0].transpose(self.default_dim - 1, self.dim_keys_past - 1)
            if self.default_dim != self.dim_values_past:
                sample[i][1] = sample[i][1].transpose(self.default_dim - 1, self.dim_values_past - 1)
        
        return sample
    
    def select_sample_cache(self, cache, sample_index):
        """Stores the key value past by selecting the sample index from the cache and storing them in a list

        Args:
            cache (List[torch.tensor]): Key-value cache as returned by the model
            sample_index (int): Which sample to select
        """
        sample = []
        for i in range(len(cache)):
            sample.append([
                cache[i][0][sample_index],
                cache[i][1][sample_index]
            ])
        sample = self.swap_dimensions(sample)
        return sample
    
    def pad_sample(self, sample, target_size):
        """Pads all samples key-value cache to a specific size

        Args:
            sample (torch.tensor): Key-value cache as stored by the LLM
            target_size (int): Target size
        """
        for i in range(len(sample)):
            for j in range(len(sample[i])):
                pad_size = target_size - sample[i][j].size(1)
                pad = (0, 0, pad_size, 0)
                if pad_size > 0:
                    sample[i][j] = torch.nn.functional.pad(sample[i][j], pad, "constant", 0)
                elif pad_size < 0:
                    sample[i][j] = sample[i][j][:, :target_size]
        return sample
    
    def stack_samples(self, samples):
        """Stacks the samples key-value cache by removing the List dimension and reordering to be appropriate for storing

        Args:
            samples (List[torch.tensor]): Key-value cache as returend by the model
        """
        stacked_samples = []
        for i in range(len(samples[0])):
            stacked_mult = []
            for j in range(len(samples[0][i])):
                stacked = torch.stack(
                    [samples[k][i][j] for k in range(len(samples))], dim=0
                )
                stacked_mult.append(stacked)
            stacked_samples.append(stacked_mult)
        return stacked_samples
        
    def store_cache(self, past_key_values, input_ids, lengths):
        """Stores the past key values returned by the model in an appropriate way

        Args:
            past_key_values (List[torch.tensor]): Tensor in which the key values where reutrned
            input_ids (torch.tensor): Input ids
            lengths (List[int]): Length of each sample in the batch
        """
        if self.run_eager:
            self.cache = past_key_values
            return
        self.cache = []
        self.previous_input_ids = []
        for i, length in enumerate(lengths):
            self.cache.append(
                self.select_from_sample_cache(self.select_sample_cache(past_key_values, i), from_=-length)
            )
            self.previous_input_ids.append(
                input_ids[i, -length:]
            )
    def common_starting_elements(self, t1, t2):
        """Check for the common starting elements in two tensors

        Args:
            t1 (torch.tensor): First Tensor
            t2 (torch.tensor): Second Tensor
        """
        min_length = min(t1.size(0), t2.size(0))
        eq = torch.eq(t1[:min_length], t2[:min_length])
        if not eq.any():
            return 0
        if eq.all():
            return min_length

        return torch.where(eq == 0)[0][0].item()
        
    def delete_previous_cache(self, new_input_ids, lengths):
        """Deletes previous cache by only keeping the common elements between the previous input ids and the new input ids

        Args:
            new_input_ids (torch.tensor): New input ids
            lengths (List[int]): List of lengths
        """
        if self.run_eager:
            return
        input_ids = [
            new_input_ids[i, -lengths[i]:] for i in range(len(lengths))
        ]
        elements = [self.common_starting_elements(input_ids[i], self.previous_input_ids[i]) for i in range(len(lengths))]
        self.cache = [
            self.select_from_sample_cache(self.cache[i], until=elements[i]) for i in range(len(lengths))
        ]
        
    
    def prepare_inputs(self, input_ids, attention_mask, n_new_tokens):
        """Prepares the inputs for the model

        Args:
            input_ids (torch.tensor): Input ids
            attention_mask (torch.tensor): Attention Mask
            n_new_tokens (int): Number of new tokens since last run
        """
        max_new_tokens = max(n_new_tokens)
        past_key_values = None
        if self.cache is not None and self.enable_cache:
            input_ids = input_ids[:, -max_new_tokens:]
            if self.run_eager:
                past_key_values = self.cache
            else:
                past_key_values = self.pad_cache(
                    [self.select_from_sample_cache(self.cache[i], until=-max_new_tokens + n_new_tokens[i]) if max_new_tokens > n_new_tokens[i] else self.cache[i]
                    for i in range(len(n_new_tokens))],
                    attention_mask.shape[1] - max_new_tokens
                )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
            "past_key_values": past_key_values
        }
    
    def pad_cache(self, cache, length):
        """Pads the cache and prepares them for the model

        Args:
            cache (torch.tensor): Key-value cache as stored by the LLM
            lengths (List[int]): List of lengths
        """
        for i in range(len(cache)):
            cache[i] = self.pad_sample(cache[i], length)
            cache[i] = self.swap_dimensions(cache[i])
        stacked_samples = self.stack_samples(cache)

        return stacked_samples
    
    def delete_cache(self, index=None, from_=None):
        """Deletes all cache

        Args:
            index (int, optional): _description_. Defaults to None.
            from_ (int, optional): _description_. Defaults to None.
        """
        # if index is not None and self.cache is not None:
        #     self.previous_input_ids = self.previous_input_ids[:index] + self.previous_input_ids[index + 1:]
        #     cache_shape = list(self.cache[0].shape)
        #     device = self.cache[0].device
        #     dtype = self.cache[0].dtype
        #     cache_shape[-2] = 0
        #     self.cache = self.cache[:index] + self.cache[index + 1:]
        #     self.previous_input_ids.append(torch.tensor([]))
        #     self.cache.append(torch.tensor([], device=device, dtype=dtype).reshape(cache_shape))
        #     return
        # else:
        self.previous_input_ids = None
        self.cache = None

    def run(self, tokenized_inputs, loaded_models, model_new_tokens, use_cache, **kwargs):
        """
        Runs the model on the tokenized inputs.
        Args:
            tokenized_inputs (torch.tensor): Inputs that have been tokenized.
            loaded_models (dict[PreTrainedModel]): Models that have been loaded. The model for this operation is in loaded_models[self.model]
            model_new_tokens (List[int]): Number of new tokens per sample in the batch
            use_cache (bool): Whether to use the key-value cache.
        """
        if isinstance(self.model, str):
            model = loaded_models[self.get_model_name()]
        else:
            model = self.model
        lengths = torch.sum(tokenized_inputs['attention_mask'], dim=-1)
        if self.cache is not None and self.enable_cache and use_cache:
            self.delete_previous_cache(tokenized_inputs['input_ids'], lengths)
                
        # if self.cache is not None:
        #     length_common_input_ids_per_sample = [
                
        #     ]
        actual_inputs = self.prepare_inputs(input_ids=tokenized_inputs['input_ids'].to(model.device),
                                            attention_mask=tokenized_inputs['attention_mask'].to(model.device),
                                            n_new_tokens=model_new_tokens)
        # run model 
        with torch.no_grad():
            try:
                model_output = model(**actual_inputs, return_dict=True)
            except RuntimeError as e:
                raise RuntimeError(f"Error thrown when running model. This is probably caused because the model handles the key-value cache differently. Consider setting dim_values_past and dim_keys_past values or disabling the key-value cache. Alternatively, you can set run_eager=True, but this feature is incompatible with speculative sampling and some other features.")
            logprobs = torch.log_softmax(model_output.logits[:, :, :self.tokenizer_length], dim=-1)
        
        if self.enable_cache and use_cache:
            self.store_cache(model_output.past_key_values, tokenized_inputs['input_ids'], lengths)
            
        logprobs = [logprobs[i, -model_new_tokens[i] : ].to(torch.float32) for i in range(logprobs.shape[0])]
        return logprobs

    def __str__(self):
        return f"PromptedLLM('{self.system_prompt}', model='{self.get_model_name()}')"

    def id(self):
        return f"PromptedLLM('{self.system_prompt}', model='{self.get_model_name()}')"
        
        
class Autocomplete(RunnableOperator):
    def __init__(self, corpus=None, speculative_factor=1, from_save_file=None, group=None, store_corpus=True, **kwargs):
        """
        Runnable operator that uses a corpus to generate a probability distribution that just predicts the most likely token based on the previous token
        
        Args:
            corpus (list[str]): Corpus to be used for training the autocomplete model
            speculative_factor (int): Factor for speculative sampling
            from_save_file (string): File from which to load the autocomplete model
            group (string): Group to which the operator belongs. This ensures that speculative sampling will not be tried when not all operators of a group are finished.
            store_corpus (bool): Whether to store the corpus when saving the model. This is necessary for loading the model from a file.
            kwargs: Arbitrary keyword arguments.
        """
        assert corpus is not None or (from_save_file is not None and os.path.isfile(from_save_file)), "Either corpus or from_save_file must be specified."
        self.tokenizer = None
        self.mapper = None
        self.unknown_mapper = None
        self.start_mapper = None
        super().__init__(speculative_factor=speculative_factor, corpus=corpus, from_save_file=from_save_file, group=group, **kwargs)
        if not store_corpus:
            log(logger.warning, "Not storing the corpus. This will make the model not loadable.")
            del self.kwargs["corpus"]
        
    def add_token(self, to_token, from_token=None):
        """
        Adds a token to the mapper, which keeps track of the most likely token to follow a given token.
        
        Args:
            to_token (int): Token to be added.
            from_token (int): Token from which to_token is most likely to follow. None if to_token is the first token in a sequence.
        """
        if from_token is None:
            self.start_mapper[to_token] = self.start_mapper.get(to_token, 0) + 1
        else:
            if from_token not in self.mapper:
                self.mapper[from_token] = dict()
            self.mapper[from_token][to_token] = self.mapper[from_token].get(to_token, 0) + 1
        self.unknown_mapper[to_token] = self.unknown_mapper.get(to_token, 0) + 1
    
    def return_max_token(self, dict_):
        """
        Returns the most likely token in a dictionary.
        
        Args
            dict_ (dict): Dictionary containing tokens and their counts.
        """
        max_token = None
        max_count = -1
        for token in dict_:
            if dict_[token] > max_count:
                max_token = token
                max_count = dict_[token]
        return int(max_token)
    
    def initialize_after_model_set(self):
        """
        Function that is run after the model is set (to the default model if necessary). Fits the autocomplete model or loads it if it in the save file.
        """
        self.tokenizer = load_tokenizer(self.model)
        
        if self.from_save_file is not None and os.path.isfile(self.from_save_file):
            self.load_from_file(self.from_save_file)
        else:
            self.corpus = [sentence for sentence in self.corpus if isinstance(sentence, str) and len(sentence) > 0]
            self.unknown_mapper = dict()
            self.start_mapper = dict()
            
            self.mapper = dict()
            
            tokenization = self.tokenizer(self.corpus)
            for sentence in tokenization.input_ids:
                for i in range(len(sentence)):
                    if i == 0:
                        self.add_token(to_token=sentence[i])
                    else:
                        self.add_token(to_token=sentence[i], from_token=sentence[i-1])
            
            self.array_mapper = torch.zeros(len(self.tokenizer), dtype=torch.int32)
            self.start_most_common = self.return_max_token(self.start_mapper)
            self.array_mapper = torch.tensor([
                self.return_max_token(self.mapper.get(i, self.unknown_mapper)) for i in range(len(self.tokenizer))
            ])
            
        if self.from_save_file:
            self.save_to_file(self.from_save_file)
            del self.kwargs["corpus"]
            del self.corpus
            
    def load_from_file(self, file):
        """
        Loads the autocomplete model from a file.
        
        Args:
            file (string): File from which to load the autocomplete model.
        """
        file_content = json.load(open(file, "r"))
        self.start_most_common = file_content["start_most_common"]
        self.array_mapper = file_content["array_mapper"]
        
    def save_to_file(self, file):
        """
        Saves the autocomplete model to a file.
        Args:
            file (string): File to which to save the autocomplete model.
        """
        json.dump({"start_most_common": self.start_most_common, 
                   "array_mapper": self.array_mapper.tolist()}, open(file, "w"))
        
    def run_on_token_past(self, tokens):
        """
        Outputs a probability distribution for the next token based on the previous tokens.
        
        Args:
            tokens (torch.tensor: Tokens to be used as input.
        """
        output = torch.zeros(len(self.tokenizer), dtype=torch.float32) - torch.inf
        if len(tokens) == 0:
            output[self.start_most_common] = 0.0
        else:
            output[self.array_mapper[tokens[-1]]] = 0.0
        
        return output
    
    def run_on_sample(self, tokens, n_new_tokens):
        """
        Runs the autocomplete model on a sample for all number of new tokens.
        
        Args:
            tokens (torch.tensor): Tokens to be used as input.
            n_new_tokens (int): Number of new tokens to be generated.
        """
        output = torch.zeros((n_new_tokens, len(self.tokenizer)))
        if n_new_tokens > 0:
            output[n_new_tokens - 1] = self.run_on_token_past(tokens)
        for i in range(2, n_new_tokens + 1):
            output[n_new_tokens - i] = self.run_on_token_past(tokens[:-i + 1])
        
        return output
    
    def run(self, tokenized_inputs, model_new_tokens, **kwargs):
        """
        Runs the autocomplete model on the tokenized inputs.
        
        Args:
            tokenized_inputs (torch.tensor): Inputs that have been tokenized.
            model_new_tokens (list[int]): Number of new tokens per sample in the batch
        """
        output = []
        for sample in range(tokenized_inputs.input_ids.shape[0]):
            output.append(self.run_on_sample(tokenized_inputs.input_ids[sample], 
                                             model_new_tokens[sample]))
        output = self.set_to_minimum(output)
        return output
    
    def __str__(self):
        return f"Autocomplete(model='{self.get_model_name()}', speculative_factor={self.speculative_factor})"
    
    def id(self):
        return f"Autocomplete(model='{self.get_model_name()}')"


class Classifier(RunnableOperator):
    def __init__(self, formula, model, n_runs_per_sample, batch_size=None, dtype=None, system_prompt="", 
                 prompt_template = lambda system_prompt, input_string: system_prompt +  input_string, minimize=False, group=None, use_bayes=True, index=1,
                 tokenizer=None, **kwargs):
        """
        Initializes the classifier operator. This is a runnable operator that uses a classifier to generate a probability distribution.
        Args:
            formula (Operator): Formula to be used for the classifier.
            model (string || PreTrainedModel): Model to be used for the classifier.
            n_runs_per_sample (int): Number of tokens to be used for classification. This is the approxmiation made, the most likely "n_runs_per_sample" tokens of the formula will be used to compute the distribution, the rest is approximated
            batch_size (int): Batch size to be used for classification. If None, the batch size is the same as for the generation process.
            dtype (torch.dtype): Data type for the model.
            system_prompt (string): String to be used as a prompt.
            prompt_template (function): Function for generating prompt. Takes two arguments: system_prompt and input_string. The operator will be run on prompt_template(..., ...) + continuation_tokens
            minimize (boolean): Whether to minimize the output of the classifier.
            group (string): Group to which the operator belongs. This ensures that speculative sampling will not be tried when not all operators of a group are finished.
            use_bayes (bool, optional): Whether to use bayes rule (and therefore use the minimization of the loss) when computing the distribution. In essence, this just subtracts the uniform distribution.
            index (int, 1): Index of the class to be used for classification. Only used when minimize is True.
            tokenizer (Tokenizer): Tokenizer to be used for the operation. If None, the default tokenizer will be used.
        """
        super().__init__(formula=formula, model=model, batch_size=batch_size, n_runs_per_sample=n_runs_per_sample, run_priority=-1, 
                         system_prompt=system_prompt, prompt_template=prompt_template, minimize=minimize, group=group, use_bayes=use_bayes, index=index)
        
        if tokenizer is None:
            self.tokenizer = load_tokenizer(self.model)
        else:
            self.tokenizer = tokenizer
        self.max_length = None
        self.dtype = dtype
        
    def load_model(self, dtype):
        """
        Loads the model for the operation.
        
        Args:
            dtype (torch.dtype): Data type for the model.
        """
        if not isinstance(self.model, str):
            return self.model
        if self.dtype is None:
            return load_model(self.model, dtype=dtype, classification=True)
        return load_model(self.model, dtype=self.dtype, classification=True)
    
    def run_single(self, tokenized_inputs, model, correct_prediction_history, other_tokenizer):
        """
        Runs the model on a single input and returns the log probabilities of the output tokens.

        Args:
            tokenized_inputs (torch.tensor): The tokenized input to run the model on.
            model (torch.nn.Module): The model to run.
            correct_prediction_history (List[Dict(model_name->output)]): For each element in the batch, the prediction history of the model.
            other_tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to decode the input samples.

        Returns:
            torch.Tensor: The log probabilities of the output tokens.
        """
    def run_single(self, input_ids, model, correct_prediction_history, other_tokenizer):
        
        output_formula = [
            self.formula.evaluate(correct_prediction_history[i], normalize=True) for i in range(len(correct_prediction_history))
        ]
        
        topk_tokens = [
            torch.topk(output_formula[i], k=self.n_runs_per_sample, dim=-1) for i in range(len(output_formula))
        ]
        
        input_samples = []
        for i in range(len(input_ids)):
            input_samples.append(other_tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=True))
            for token in topk_tokens[i].indices:
                input_samples.append(other_tokenizer.decode(input_ids[i].tolist() + [token], skip_special_tokens=True))
        
        if self.max_length is None:
            self.max_length = get_max_length(model.config)
        # -2 in max_length is because of weird behavior by the roberta model
        encoded_samples = self.tokenizer.batch_encode_plus(input_samples, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length - 2).to(model.device)
        
        if "token_type_ids" in encoded_samples:
            del encoded_samples["token_type_ids"]

        if self.batch_size is None:
            batch_size = len(input_ids)
        else:
            batch_size = self.batch_size

        model_outputs = []
        for i in range(0, len(encoded_samples['input_ids']), batch_size):
            batched_input_ids = encoded_samples['input_ids'][i:i+batch_size]
            max_len = max(len(seq) for seq in encoded_samples['input_ids'])
            batched_attention_mask = encoded_samples['attention_mask'][i:i+batch_size]
            with torch.no_grad():
                model_output = model(input_ids=batched_input_ids, attention_mask=batched_attention_mask)
            model_outputs.append(model_output.logits)
            
        model_outputs = torch.cat(model_outputs, dim=0)

        
        if not self.minimize:
            model_outputs_logprobs = torch.log_softmax(model_outputs, dim=-1)[:, self.index]
        elif self.minimize and model_outputs.shape[1] == 2:
            model_outputs_logprobs = torch.log_softmax(model_outputs, dim=-1)[:, 1 - self.index]
        else:
            model_output_probs = torch.softmax(model_outputs, dim=-1)[:, self.index]
            model_output_probs = 1 - model_output_probs
            model_outputs_logprobs = torch.log(torch.max(model_output_probs, torch.tensor([1e-12], device=model_output_probs.device)))
        
        output_logprobs = torch.zeros((len(input_ids), 1, len(other_tokenizer)), device=model_outputs_logprobs.device)
        
        for i in range(len(input_ids)):
            # change the topk tokens with factor * (model_token_output - model_no_token_output)
            output_model_sample = model_outputs_logprobs[i * (1 + self.n_runs_per_sample) : (i + 1) * (1 + self.n_runs_per_sample)]
            normal_logprob = output_model_sample[0]
            if not self.use_bayes:
                output_logprobs = output_logprobs + normal_logprob 
            
            for j in range(self.n_runs_per_sample):
                output_logprobs[i, -1, topk_tokens[i].indices[j]] = output_model_sample[j + 1]
                if self.use_bayes:
                    output_logprobs[i, -1, topk_tokens[i].indices[j]] = output_logprobs[i, -1, topk_tokens[i].indices[j]] - normal_logprob
        
        if not self.use_bayes:
            output_logprobs = torch.log_softmax(output_logprobs, dim=-1)
                
        
        return output_logprobs.to(torch.float32)
        
    
    def run(self, tokenized_inputs, loaded_models, model_new_tokens, new_prediction_history, other_tokenizer, **kwargs):
        """
        Runs the classifier on the tokenized inputs.
        
        Args:
            tokenized_inputs (torch.tensor): Inputs that have been tokenized.
            loaded_models (Dict[PreTrainedModel]): Models that have been loaded. The model for this operation is in loaded_models[self.model]
            model_new_tokens (List[int]): Number of new tokens per sample in the batch
            new_prediction_history (List(List(Dict[model_name -> output]))): Prediction History for the batch, this is used to determine the tokens to be used for classification.
            other_tokenizer (Tokenizer): Tokenizer to be used for the decoding of the input tokens. This is necessary in order to prepare the inputs for the classifier.
        """
        if not isinstance(tokenized_inputs, torch.Tensor):
            tokenized_inputs = tokenized_inputs.input_ids
        if all([tokens == 0 for tokens in model_new_tokens]):
            return [[] for _ in range(model_new_tokens)]

        if isinstance(self.model, str):
            model = loaded_models[self.model]
        else:
            model = self.model
            
        to_run_further = [tokens > 0 for tokens in model_new_tokens]
        
        output_logits = [[] for _ in range(len(model_new_tokens))]
        
        current_iteration = 0
        
        while any(to_run_further):
            indices = [i for i in range(len(to_run_further)) if to_run_further[i]]
            # only select the tokens that are not yet finished
            tokenized_inputs_ = [tokenized_inputs[i] for i in indices]
            # tokenized_inputs_ = tokenized_inputs[indices]
            if current_iteration > 0:
                tokenized_inputs_ = tokenized_inputs_[:, :-current_iteration]
                
            correct_prediction_history = [new_prediction_history[i][current_iteration] for i in indices]
            
            outputs = self.run_single(
                tokenized_inputs_, model, correct_prediction_history, other_tokenizer
            )
            
            for index in range(len(indices)):
                output_logits[indices[index]].append(outputs[index])
            
            current_iteration += 1
            to_run_further = [tokens > current_iteration for tokens in model_new_tokens]
        
        return [torch.cat(output_logits[i], dim=0) for i in range(len(output_logits))]

    def __str__(self):
        return f"Classifier('{self.system_prompt}', model='{self.get_model_name()}', formula='{self.formula}', n_runs_per_sample={self.n_runs_per_sample}, minimize={self.minimize}, bayes={self.use_bayes})"
    
    def get_operators(self, class_):
        return super().get_operators(class_) + self.formula.get_operators(class_)

    def norm(self, runnable_operator_outputs=None):
        """
        Returns the norm of the operator. Due to the "- normal_logprob" in the run function, this is always 0.
        """
        if self.use_bayes:
            return 0
        return 1
    
    def is_finished(self, runnable_operator_outputs):
        """
        Returns whether the operation is finished. This is the case if the formula is finished and the class itself has been run.
        """
        return super().is_finished(runnable_operator_outputs) and self.formula.is_finished(runnable_operator_outputs)
        

class ClassifierStrength(RunnableOperator):
    def __init__(self, model, batch_size=None, dtype=None, system_prompt="", 
                 prompt_template = lambda system_prompt, input_string: system_prompt +  input_string, group=None, only_input=False,
                 default_output=0.5, speculative_factor=1, tokenizer=None, **kwargs):
        """
        Initializes the classifier operator. This is a runnable operator that uses a classifier to generate a probability distribution.
        
        Args:
            model (PreTrainedModel || string): Model to be used for the classifier.
            dtype (torch.dtype): Data type for the model.
            system_prompt (string): String to be used as a prompt.
            prompt_template (function): Function for generating prompt. Takes two arguments: system_prompt and input_string. The operator will be run on prompt_template(..., ...) + continuation_tokens
            group (string): Group to which the operator belongs. This ensures that speculative sampling will not be tried when not all operators of a group are finished.
            only_input (bool): Whether to only use the input for classification (and not the continuation tokens)
            default_output (float): Default output of the classifier.
            speculative_factor (int): Factor for speculative sampling.
            tokenizer (Tokenizer): Tokenizer to be used for the operation. If None, the default tokenizer will be used.
        """
        super().__init__(model=model, batch_size=batch_size, run_priority=1, 
                         system_prompt=system_prompt, prompt_template=prompt_template, group=group, outputs_logprobs=False, only_input=only_input, default_output=default_output, 
                         speculative_factor=speculative_factor, **kwargs)
        if tokenizer is None:
            self.tokenizer = load_tokenizer(self.model)
        else:
            self.tokenizer = tokenizer
        self.max_length = None
        self.dtype = dtype
        
    def load_model(self, dtype):
        """
        Loads the model for the operation.
        
        Args:
            dtype (torch.dtype): Data type for the model.
        """
        if not isinstance(self.model, str):
            return self.model
        if self.dtype is None:
            return load_model(self.model, dtype=dtype, classification=True)
        return load_model(self.model, dtype=self.dtype, classification=True)
    
    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        """
        Evaluate the operation.
        
        Args:
            runnable_operator_outputs (Dict): Outputs of runnable operators.
            normalize (bool): Whether to normalize the evaluation.
            
        Returns:
            int: The evaluation of the operation.
        """
        for output in runnable_operator_outputs:
            if self.same_operator(output) and runnable_operator_outputs[output] is not None:
                # NOTE: float() might become a problem when batch post processing is implemented
                return float(runnable_operator_outputs[output])
        return self.default_output
    
    def run(self, tokenized_inputs, loaded_models, model_new_tokens, other_tokenizer, tokenized_only_input, **kwargs):
        """
        Runs the classifier on the tokenized inputs.
        
        Args:
            tokenized_inputs (torch.tensor): Inputs that have been tokenized.
            loaded_models (List[PreTrainedModel]): Models that have been loaded. The model for this operation is in loaded_models[self.model]
            model_new_tokens (List[Int]): Number of new tokens per sample in the batch
            other_tokenizer (Tokenizer): Tokenizer to be used for decoding. This is necessary in order to prepare the inputs for the classifier.
        """
        if isinstance(self.model, str):
            model = loaded_models[self.model]
        else:
            model = self.model

        if self.only_input:
            tokenized_inputs = tokenized_only_input

        input_samples = []
        for i in range(len(tokenized_inputs.input_ids)):
            for j in range(model_new_tokens[i]):
                input_samples.append(other_tokenizer.decode(tokenized_inputs.input_ids[i][:-j if j != 0 else None].tolist(), skip_special_tokens=True))

        if self.max_length is None:
            self.max_length = get_max_length(model.config)
        encoded_samples = self.tokenizer.batch_encode_plus(input_samples, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(model.device)

        if "token_type_ids" in encoded_samples:
            del encoded_samples["token_type_ids"]

        if self.batch_size is None:
            batch_size = len(tokenized_inputs)
        else:
            batch_size = self.batch_size

        model_outputs = []
        for i in range(0, len(encoded_samples['input_ids']), batch_size):
            batched_input_ids = encoded_samples['input_ids'][i:i+batch_size]
            batched_attention_mask = encoded_samples['attention_mask'][i:i+batch_size]
            with torch.no_grad():
                model_output = model(input_ids=batched_input_ids, attention_mask=batched_attention_mask, return_dict=True, output_hidden_states=True)
            model_outputs.append(model_output.logits)

        model_outputs = torch.cat(model_outputs, dim=0).to(torch.float32)
    
        # reshape the model outputs to the required format
        reshaped_model_outputs = []
        start_index = 0
        for tokens in model_new_tokens:
            reshaped_model_outputs.append(model_outputs[start_index:start_index+tokens])
            start_index += tokens

        reshaped_model_outputs_probs = [torch.softmax(output, dim=-1)[:, 1].unsqueeze(1) for output in reshaped_model_outputs]

        return reshaped_model_outputs_probs

    def __str__(self):
        return f"ClassifierStrength('{self.system_prompt}', model='{self.get_model_name()}')"

    def norm(self, runnable_operator_outputs=None):
        """
        Returns the norm of the operator. Due to the "- normal_logprob" in the run function, this is always 0.
        """
        for output in runnable_operator_outputs:
            if self.same_operator(output) and runnable_operator_outputs[output] is not None:
                return float(runnable_operator_outputs[output])
        return self.default_output