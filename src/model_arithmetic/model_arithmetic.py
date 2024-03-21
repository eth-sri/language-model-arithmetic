from transformers import PreTrainedModel
from .basic_model_loader import load_model, load_tokenizer
import json
import numpy as np
from .utils import get_max_length, ENABLE_LOGGING, log
import torch
from collections import namedtuple
from loguru import logger
import os
import time
from .operators import Operator
from .monitor import Monitor
from .runnable_operators import RunnableOperator, PromptedLLM
from .input import TokenizedInput
from .lm_eval_compatibility import Compatibility
from .top_k_top_p_filtering import top_k_top_p_filtering



class ModelArithmetic(PreTrainedModel):
    """
    Main class for prompt arithmetic. Handles the generation of text based on the formula.
    """
    SAVE_FILE = "prompt_arithmetic.json"
    _supports_sdpa = True

    def __init__(self, formula : Operator, default_model : str = None, dtype=torch.bfloat16, intermediate_argmax : bool = False, epsilon = 1e-6, 
                 retroactive_operators = [], calculate_statistics=True, needs_input_tokens_lm_eval=False, lm_eval_task=None, tokenizer=None, max_length=None):
        """Initializes the prompt arithmetic model.

        Args:
            formula (Operator): The formula for which generations need to be made.
            default_model (str, optional): Default model for RunnableOperators that don't have a model associated with them. Defaults to None.
            dtype (torch.dtype, optional): Dtype of the models to load by default. Defaults to torch.bfloat16.
            intermediate_argmax (bool, optional): Something unimportant that was tried out, but now deprecated. Defaults to False.
            epsilon (float, optional): Just some small value. Defaults to 1e-12.
            retroactive_operators (list, optional): The retroactive operators that need to be applied. Defaults to [].
            calculate_statistics (bool, optional): Whether or not to calculate some statistics, can be a tad bit expensive. Defaults to True.
            needs_input_tokens_lm_eval (bool, optional): Whether or not lm eval is used and whether or not the task needs the input tokens. Defaults to False. Only set to true for an lm eval task.
            lm_eval_task (str, optional): Name of the lm eval task. Defaults to None.
            tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase, optional): Tokenizer to use. Defaults to None.
            max_length (int, optional): Maximum length of the input. Defaults to None.
        """
        self.formula = formula

        self.default_model = default_model
        self.loaded_models = dict()
        self.model_prediction_history = [] # keeps track of the RunnableOperators predictions for each token (that hasn't finished computing)
        self.logprobs_history = [] # keeps track of the current probability distribution for which each token has been drawn
        self.model_last_token_prediction = [] # keeps track of the last token that has been predicted for each RunnableOperator
        
        self.output_type = namedtuple("ModelArithmeticOutput", ["logits", "logprobs_per_model"])
        self.logits_type = namedtuple("ModelArithmeticLogits", ["logits"])
        self.intermediate_argmax = intermediate_argmax
        self.retroactive_operators = retroactive_operators
        self.calculate_statistics = calculate_statistics

        self.runnable_operators = []
        for runnable_operator in self.formula.get_operators(RunnableOperator):
            if not any([runnable_operator.same_operator(output) for output in self.runnable_operators]):
                self.runnable_operators.append(runnable_operator)
                

        # sort the prompts by speculative factor, putting the one with highest speculative factor first
        # => run model with highest speculative factor first, since otherwise the computation might be wasted for the first ones
        # however, we first need to sort by run_priority and then within that by speculative factor
        self.runnable_operators = sorted(self.runnable_operators, key=lambda runnable_operator: (runnable_operator.run_priority, runnable_operator.speculative_factor), reverse=True)
        
        self.load_all_models(dtype=dtype)
        if self.default_model not in self.loaded_models:
            for runnable_operator in self.runnable_operators:
                if isinstance(runnable_operator, PromptedLLM) and runnable_operator.model is not None:
                    self.default_model = runnable_operator.get_model_name()
                    break
            if self.default_model is None:
                raise ValueError("Default model must be specified if not specified in an llm prompt")

        self.config = self.loaded_models[str(self.default_model)].config
        if tokenizer is None:
            if hasattr(self.config, "_name_or_path"):
                self.tokenizer = load_tokenizer(self.config._name_or_path)
            else:
                self.tokenizer = load_tokenizer(self.default_model)
        else:
            self.tokenizer = tokenizer
        
        self.init_runnable_operators()
        
        self.model_input_tokens = {
            runnable_operator.id(): TokenizedInput(runnable_operator, 
                                                    runnable_operator.model, 
                                                    self.loaded_models[str(runnable_operator.get_model_name())].config,
                                                    self.tokenizer, max_length=max_length) 
            for runnable_operator in self.runnable_operators
        }
        for tokenized_input in self.model_input_tokens.values():
            tokenized_input.synchronize_max_lengths(self.model_input_tokens.values())
        
        self.init_monitor()
        
        self.epsilon = epsilon
        
        self.word_size = len(self.tokenizer)
        
        if Compatibility is not None:
            self.lm_eval_compatibility = Compatibility(
                task_name=lm_eval_task,
                needs_input_tokens_lm_eval=needs_input_tokens_lm_eval,
                tokenizer=self.tokenizer,
                device=self.device,
                max_length=get_max_length(self.config),
            )
        else:
            self.lm_eval_compatibility = None
        
        super().__init__(self.config)
        
    def init_monitor(self):
        """
        Initializes the monitor for the prompt arithmetic model.
        """
        self.monitor = Monitor(self.runnable_operators)
        
    def init_runnable_operators(self):
        """Initializes the runnable operators. This is done after the models have been loaded, because the models are needed for the runnable operators.
        """
        for runnable_operator in self.runnable_operators:
            if runnable_operator.model is None:
                runnable_operator.model = self.default_model
            runnable_operator.initialize_after_model_set()

    def eval(self):
        """Sets the model to evaluation mode
        """
        for model in self.loaded_models.values():
            model.eval()

    def load_all_models(self, dtype=torch.bfloat16):
        """Loads all the models that are needed for the runnable operators. Models are never loaded twice.

        Args:
            dtype (torch.dtype, optional): Default Dtype of the models. Defaults to torch.bfloat16.
        """
        if self.default_model is None:
            for runnable_operator in self.runnable_operators:
                    if isinstance(runnable_operator, PromptedLLM) and runnable_operator.model is not None:
                        self.default_model = runnable_operator.get_model_name()
        
        for runnable_operator in self.runnable_operators:
            if runnable_operator.model is None:
                assert self.default_model is not None, "Default model must be specified if not specified in prompt"
                runnable_operator.model = self.default_model
            if runnable_operator.get_model_name() not in self.loaded_models:
                model = runnable_operator.load_model(dtype=dtype)

                if model is not None:
                    model.eval()
                    self.loaded_models[runnable_operator.get_model_name()] = model
        
        if len(self.loaded_models) == 0:
            assert self.default_model is not None, "Required to at least have one model, for now"
            self.loaded_models[str(self.default_model)] = load_model(self.default_model, dtype=dtype)
    
    @property
    def device(self):
        """Device of the default model. Needed for compatibility with lm_eval

        Returns:
            torch.device: Device of the default model.
        """
        return self.loaded_models[str(self.default_model)].device

    def save_pretrained(self, path : str):
        """Saves the model to the specified path.

        Args:
            path (str): Path to which to save the model
        """
        os.makedirs(path, exist_ok=True)
        all_settings = {
            "formula": self.formula.generate_settings(),
            "default_model": self.default_model,
        }

        with open(os.path.join(path, self.SAVE_FILE), "w") as f:
            json.dump(all_settings, f, indent=4, sort_keys=True)

    @classmethod
    def from_pretrained(cls, path : str, dtype=torch.bfloat16):
        """Loads the model from the specified path.

        Args:
            path (str): Path from which to load the model
            dtype (torch.dtype, optional): Default dtype for the models. Defaults to torch.bfloat16.

        Returns:
            ModelArithmetic: model arithmetic model
        """
        with open(os.path.join(path, cls.SAVE_FILE), "r") as f:
            all_settings = json.load(f)
            all_settings["formula"] = Operator.load_from_settings(all_settings["formula"])
        return cls(**all_settings, dtype=dtype)

        
    def forward_model(self, runnable_operator, continuation_tokens, model_new_tokens=None, use_cache=False, do_speculation=False, 
                      attention_mask=None):
        """Runs a specifc runnable operator on the continuation tokens.

        Args:
            runnable_operator (RunnableOperator): The runnable operator to run.
            continuation_tokens (list[list[int]]): List of tokens that need to be continued. The prompt is not included in these tokens
            model_new_tokens (list[int], optional): New tokens for the model. Defaults to None.
            use_cache (bool, optional): Whether or not to allow the model to use cache (eg key-value storage for an LLM). Defaults to False.
            do_speculation (bool, optional): Whether or not to do speculation sampling. Defaults to False.

        Returns:
            torch.tensor: logprobs of the model, one logprob distribution for each new token in each sample
        """
        start_time = time.time()
        
        tokenized_input_creator = self.model_input_tokens[runnable_operator.id()]
        if attention_mask is None:
            tokenized_inputs = tokenized_input_creator.add_continuation_tokens(continuation_tokens)
        else:
            tokenized_inputs = {"input_ids": continuation_tokens, "attention_mask": attention_mask}            

        tokenized_only_input = tokenized_input_creator.get_only_input_tokens()        
        was_none = model_new_tokens is None
        
        if was_none:
            model_new_tokens = torch.tensor([len(continuation_tokens[i]) + 1 for i in range(len(continuation_tokens))])

        if len(self.model_prediction_history) < len(continuation_tokens):
            new_prediction_history = [[dict()] for _ in range(len(continuation_tokens))]
        else:
            new_prediction_history = [
                [self.model_prediction_history[i].get(self.max_index_prediction_history(i) - k, dict()) for k in range(model_new_tokens[i])] 
                for i in range(len(continuation_tokens))
            ]
            
        logprobs = runnable_operator.run(
            loaded_models=self.loaded_models,
            tokenized_inputs=tokenized_inputs,
            model_new_tokens=model_new_tokens,
            new_prediction_history=new_prediction_history,
            other_tokenizer=self.tokenizer,
            tokenized_only_input=tokenized_only_input, 
            use_cache=use_cache,
            do_speculation=do_speculation,
        )
        
        logprobs = [logprob.to(self.device) for logprob in logprobs]
        
        if was_none:
            val = model_new_tokens[0]
            if torch.all(model_new_tokens == val):
                logprobs = torch.stack(logprobs, dim=0)

        self.monitor.add_result(element=time.time() - start_time, runnable_operator=runnable_operator)
        return logprobs
    
    def group_complete(self, model_history):
        """Checks which groups of runnable operators have been completely calculated and which haven't.

        Args:
            model_history (dict): Dict mapping the runnable operator id to the logprobs of the model

        Returns:
            dict[bool]: Dict mapping the group to whether it has been completely calculated or not
        """
        # everything that is a group needs to be either all calculated or all not calculated
        group_calculated = dict()
        groups = set([runnable_operator.group for runnable_operator in self.runnable_operators if runnable_operator.group is not None])
        completed_groups = {group: True for group in groups}
        
        for runnable_operator in self.runnable_operators:
            if runnable_operator.group is not None:
                is_calculated = model_history.get(runnable_operator.id()) is not None
                if runnable_operator.group not in group_calculated:
                    group_calculated[runnable_operator.group] = is_calculated
                elif group_calculated[runnable_operator.group] != is_calculated:
                    completed_groups[runnable_operator.group] = False
        return completed_groups
    
    def group_model_history(self, model_history):
        """Sets the model history on which to evaluate the formula based on the groups. Removes predictions if the group hasn't been completely calculated yet.

        Args:
            model_history (dict): Dict mapping the runnable operator id to the logprobs of the model

        Returns:
            dict: Adjusted dict mapping
        """
        completed_groups = self.group_complete(model_history)
        grouped_model_history = dict()
        for runnable_operator in self.runnable_operators:
            if runnable_operator.group is None or completed_groups[runnable_operator.group]:
                grouped_model_history[runnable_operator.id()] = model_history[runnable_operator.id()]
            else:
                grouped_model_history[runnable_operator.id()] = None
        
        return grouped_model_history
    
    def create_sample_logprobs(self, logprobs, temperature, top_k, top_p):
        """Creates the logprobs for each token in each sample.

        Args:
            logprobs (torch.tensor): Logprobs of the model
            temperature (float): temperature to use
            top_k (int): top_k to use
            top_p (float): top_p to use

        Returns:
            torch.tensor: Logprobs for each token in each sample
        """
        if temperature == 0:
            logprobs_argmax = torch.argmax(logprobs, dim=-1)
            logprobs = torch.nn.functional.one_hot(logprobs_argmax, num_classes=logprobs.shape[-1]).float()
            return logprobs
        logprobs = logprobs / temperature
        logprobs = top_k_top_p_filtering(logprobs.unsqueeze(0), top_k=top_k, top_p=top_p)
        return torch.softmax(logprobs, dim=-1).squeeze()
        
        

    def process_logprobs(self, model_history):
        """Processes model history to get the probability distribution for the token.

        Args:
            model_history (dict): Dict mapping the runnable operator id to the logprobs of the model

        Returns:
            _type_: _description_
        """
        logprobs_normalized = self.formula.evaluate(model_history)

        if not torch.is_tensor(logprobs_normalized):
            return None
        # logprobs_normalized = logprobs_normalized / temperature
        # logprobs_normalized = top_k_top_p_filtering(logprobs_normalized.unsqueeze(0), top_k=top_k, top_p=top_p)
        return logprobs_normalized
    
    def run_retroactive_operators(self, index, tokenized_sentence, temperature, top_k, top_p):
        """Runs the retroactive operators on the tokenized sentence. 

        Args:
            index (int): Index of the sentence in the current batch
            tokenized_sentence (list[int]): Tokenized sentence
            temperature (float): temperature to use
            top_k (int): top_k to use
            top_p (float): top_p to use

        Returns:
            list[int]: Adjusted tokenized sentence based on the retroactive operators and whether they accepted it.
        """
        for operator in self.retroactive_operators:
            accepted = operator.accept(tokenized_sentence, self.tokenizer)
            if accepted < 0:
                not_accepted_token = tokenized_sentence[accepted]
                self.clear_model_prediction_history(index, tokenized_sentence, from_=len(tokenized_sentence) + accepted, 
                                                    temperature=temperature, top_k=top_k, top_p=top_p)
                tokenized_sentence = tokenized_sentence[:len(tokenized_sentence) + accepted]
                
                self.logprobs_history[index][len(tokenized_sentence)][not_accepted_token] = -torch.inf
                
                if torch.all(self.logprobs_history[index][len(tokenized_sentence)] == -torch.inf):
                    self.logprobs_history[index][len(tokenized_sentence)] = torch.zeros_like(self.logprobs_history[index][len(tokenized_sentence)])
                                
                probs_to_sample = self.create_sample_logprobs(
                    self.logprobs_history[index][len(tokenized_sentence)],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                new_token = torch.multinomial(probs_to_sample, 1).item()
                
                tokenized_sentence.append(new_token)
                return self.run_retroactive_operators(index, tokenized_sentence, temperature, top_k, top_p)
    
        return tokenized_sentence
    
    def speculation_sample(self, token, previous_models_probs, new_models_probs):
        """Sample a token based on the previous and new model probabilities in the speculative sampling way. Also returns whether the token was accepted or not.

        Args:
            token (int): Token that is currently selected
            previous_models_probs (torch.tensor): Model probabilities of the previous models
            new_models_probs (torch.tensor): Model probabilities of the new models

        Returns:
            (int, bool): New token and whether or not the input token was accepted
        """
        acceptance_prob = torch.minimum(torch.tensor(1.0), new_models_probs[token] / (previous_models_probs[token] + torch.tensor(self.epsilon)))
        # TODO: the next line is taking an enormous amount of time because of asynchronous computing on gpu's and requiring it to be returned immediately
        # Therefore do batch processing
        acceptance_prob = float(acceptance_prob)
        self.monitor.add_result(element=float(acceptance_prob), indicator="acceptance_prob")
        # self.monitor.add_result(element=self.entropy(previous_models_probs).item(), indicator="entropy_previous")
        # self.monitor.add_result(element=previous_models_probs[token].item(), indicator="probability_previous")

        if torch.rand(1) < acceptance_prob:
            return token, True
        else:
            new_proba_distrib = torch.relu(new_models_probs - previous_models_probs)
            new_proba_distrib /= torch.sum(new_proba_distrib)
            new_token = torch.multinomial(new_proba_distrib, 1).item()
            return new_token, False
            
            
    def add_new_result(self, generated_tokens, num_new_tokens, runnable_operator, new_model_logprobs, top_p, top_k, temperature):
        """Adds a new run of a runnable operator to the model prediction history. Also does speculation sampling if needed.

        Args:
            generated_tokens (list[list[int]]): Currently generated tokens by the model
            num_new_tokens (list[int]): Number of new tokens for each sample in the batch
            runnable_operator (RunnableOperator): Runnable operator that was run
            new_model_logprobs (List[torch.tensor]): Output of the run function of the runnable operator
            top_p (flaot): top_p to use
            top_k (int): top_k to use
            temperature (float): temperature to use

        Returns:
            list[bool]: For each sample in the batch, whether all tokens in that sample were kept or not
        """
        all_kept = []
        for i in range(len(generated_tokens)):
            n_generated_tokens = len(generated_tokens[i])
            kept = True
            for n_token in range(n_generated_tokens - num_new_tokens[i] + 1, n_generated_tokens + 1):
                # initialize the model prediction history
                self.model_prediction_history[i][n_token] = self.model_prediction_history[i].get(n_token, 
                                                                                                 {runnable_operator.id(): None for runnable_operator in self.runnable_operators})
                # check if we need to do speculation sampling, only needed when a previous token was sampled
                do_speculation_sample = n_token < n_generated_tokens
                
                # speculation sampling not needed if the model was run before 
                if self.model_prediction_history[i][n_token][runnable_operator.id()] is not None:
                    do_speculation_sample = False
                
                # speculation sampling not needed if all models have not been run yet: this is the first model on this token
                if all([logprob is None for logprob in self.model_prediction_history[i][n_token].values()]):
                    do_speculation_sample = False
                    # This means that this token was already fully accepted, so we can just continue (can happen if batch_size > 1 or when end is triggered)
                    if self.max_index_prediction_history(i) > n_token:
                        continue
                        
                # add the new model logprobs
                self.model_prediction_history[i][n_token][runnable_operator.id()] = new_model_logprobs[i][-n_generated_tokens + n_token + num_new_tokens[i] - 1]
                
                group_model_history = self.group_model_history(self.model_prediction_history[i][n_token])
                # group_model_history needs to be separately checked, since it could be that the group is not yet fully calculated
                # also allow no logprobs runnable operators (would lead to errors) if the formula is not finished yet (if it is finished, you need to)
                if all([logprob is None for logprob in group_model_history.values()]) or (not runnable_operator.outputs_logprobs and not self.formula.is_finished(group_model_history)):
                    continue
                
                # process the logprobs
                new_model_probs = self.process_logprobs(group_model_history)
                
                if self.intermediate_argmax and not self.formula.is_finished(group_model_history):
                    argmax_el = torch.argmax(new_model_probs)
                    new_model_probs = torch.zeros_like(new_model_probs)
                    new_model_probs[argmax_el] = 1.0
                
                if do_speculation_sample:
                    if self.calculate_statistics:
                        previous_models_probs = self.create_sample_logprobs(self.logprobs_history[i].get(n_token), temperature, top_k, top_p)
                        new_model_probs = self.create_sample_logprobs(new_model_probs, temperature, top_k, top_p)
                        self.monitor.add_result(self.expected_acceptance_prob(new_model_probs, previous_models_probs), 
                                            indicator="expected_acceptance_prob", runnable_operator=runnable_operator)

                    new_token, kept = self.speculation_sample(
                        token = generated_tokens[i][n_token],
                        previous_models_probs=previous_models_probs,
                        new_models_probs=new_model_probs, 
                    )
                    if n_token in self.model_prediction_history[i]:
                        self.logprobs_history[i][n_token] = new_model_probs
                    
                    if not kept:
                        # if not kept, we change the generated tokens and remove the model prediction history after that token
                        generated_tokens[i][n_token] = new_token
                        generated_tokens[i] = generated_tokens[i][:n_token + 1]
                        self.clear_model_prediction_history(i, generated_tokens[i], from_=n_token, 
                                                            temperature=temperature, top_k=top_k, top_p=top_p)
                        self.trigger_end[i] = False
                        
                elif n_token in self.model_prediction_history[i]:
                    self.logprobs_history[i][n_token] = new_model_probs
                    
                if not kept:
                    break
            
            all_kept.append(kept)
        return all_kept
    

    def clear_model_prediction_history(self, index, generated_tokens_index, temperature, top_k, top_p, from_=-1):
        """Clears the model prediction history for a specific sample in the batch. First deletes all history of finished tokens, then 
        deletes history of tokens that were prediction, but then got removed because of speculation

        Args:
            index (int): index of the sample in the batch
            generated_tokens_index (list[int]): Generated tokens at the index
            from_ (int, optional): From which token to delete all the history. Defaults to -1.
        """
        all_indices = list(self.model_prediction_history[index].keys())
        for token in all_indices:
            all_none = all([logprob is None for logprob in self.model_prediction_history[index][token].values()])
            finished = self.formula.is_finished(self.model_prediction_history[index][token])
            if all_none or finished or (from_ != -1 and token > from_):
                if finished and len(generated_tokens_index) > token and self.calculate_statistics:
                    self.add_monitor_token_probs(generated_tokens_index[token], self.model_prediction_history[index][token], self.logprobs_history[index].get(token), 
                                                 temperature=temperature, top_k=top_k, top_p=top_p)
                
                if finished:
                    for model_index in range(len(self.model_last_token_prediction)):
                        self.model_last_token_prediction[model_index][index] = max(token + 1, self.model_last_token_prediction[model_index][index])
     
                del self.model_prediction_history[index][token]
        
        if from_ > -1:
            for model_index in range(len(self.model_last_token_prediction)):
                self.model_last_token_prediction[model_index][index] = min(from_ + 1, self.model_last_token_prediction[model_index][index])
    
    def max_index_prediction_history(self, index):
        """Gets the max index of the model prediction history for a specific runnable operator  

        Args:
            index (int): index of runnable operator in the list of runnable operators

        Returns:
            int: max index of its prediction
        """
        keys = list(self.model_prediction_history[index].keys())
        
        if len(keys) == 0:
            return 0
        return max(self.model_prediction_history[index].keys())

    def normal_sample(self, probs):
        """Samples from a probability distribution

        Args:
            probs (torch.tensor): Probability distribution

        Returns:
            int: Sampled token
        """
        out = torch.multinomial(probs, 1)
        return out
    
    def KL_divergence(self, p, q):
        """Compuates KL divergence between two probability distributions

        Args:
            p (torch.tensor): probability distribution
            q (torch.tensor): probability distribution

        Returns:
            float: KL divergence
        """
        return torch.sum(p * torch.log((p + self.epsilon) / (q + self.epsilon)))
    
    def entropy(self, p):
        """Computes entropy of a probability distribution

        Args:
            p (torch.tensor): probability distribution

        Returns:
            float: entropy
        """
        return -torch.sum(p * torch.log(p + self.epsilon))
    
    def expected_acceptance_prob(self, p, q):
        """
        Calculates the expected acceptance probability of speculative sampling.
        
        Args:
            p (torch.tensor): probability distribution
            q (torch.tensor): probability distribution
        """
        return 1 - 1 / 2 * torch.sum(torch.abs(q - p)).item()
    
    def add_monitor_token_probs(self, token, history, history_logprobs, temperature, top_k, top_p):
        """Adds some token probabilities to the monitor

        Args:
            token (int): Samples token
            history (dict): Model prediction history at the specific index where the token was drawn from
            history_logprobs (torch.tensor): LogProbability distribution from which the token was drawn.
        """
        for runnable_operator in self.runnable_operators:
            if runnable_operator.is_finished(history) and runnable_operator.outputs_logprobs:
                evaluated = runnable_operator.evaluate(history)
                self.monitor.add_result(element=torch.softmax(evaluated, dim=-1)[token].item(), runnable_operator=runnable_operator, indicator="token_prob")
                # add logprob as well
                self.monitor.add_result(element=max(evaluated[token].item(), np.log(self.epsilon)), runnable_operator=runnable_operator, indicator="token_logprob")
                # add KL divergence
                if history_logprobs is not None:
                    self.monitor.add_result(element=self.KL_divergence(torch.softmax(history_logprobs, dim=-1), torch.softmax(evaluated, dim=-1)).item(), 
                                            runnable_operator=runnable_operator, indicator="KL_divergence")

        self.monitor.add_result(element=self.entropy(torch.softmax(history_logprobs, dim=-1)).item(), indicator="entropy")
        

    def next_token_speculative(self, continuation_tokens, 
                               top_p=1.0, top_k=0, temperature=1.0, speculation=True, use_cache=True):
        """Continues one step in the generation process by running the runnable operators that need to be run and then sampling from the probability distribution.

        Args:
            continuation_tokens (list[list[int]]): Current continuation tokens
            top_p (float, optional): top_p to use. Defaults to 1.0.
            top_k (int, optional): top_k to use. Defaults to 0.
            temperature (float, optional): temperature to use. Defaults to 1.0.
            speculation (bool, optional): Whether to use speculation. Defaults to True.
            use_cache (bool, optional): Whether to use cache. Defaults to True.

        Returns:
            _type_: _description_
        """
        models_ran = []
        for i, runnable_operator in enumerate(self.runnable_operators):
            new_tokens = [len(continuation_tokens[j]) - self.model_last_token_prediction[i][j] + 1 for j in range(len(continuation_tokens))]
            if runnable_operator.run_condition(new_tokens, self.trigger_end) or not speculation:
                logprobs = self.forward_model(runnable_operator, continuation_tokens, model_new_tokens=new_tokens, use_cache=use_cache, do_speculation=speculation)
                all_kept = self.add_new_result(continuation_tokens, new_tokens, runnable_operator, logprobs, top_p, top_k, temperature)
                models_ran.append(i)
                
                self.model_last_token_prediction[i] = [len(continuation_tokens[j]) + int(all_kept[j])
                                                       for j in range(len(continuation_tokens))]
                
                if not all(all_kept):
                    break
                
        to_sample_indices = [i for i in range(len(continuation_tokens)) if all_kept[i] and not self.trigger_end[i]]

        if len(to_sample_indices) > 0:
            # do batch sampling
            all_required_histories = torch.stack([
                self.create_sample_logprobs(
                    self.logprobs_history[i][len(continuation_tokens[i])], 
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                ) for i in to_sample_indices
            ])
            new_tokens = self.normal_sample(all_required_histories)
            for i in range(len(to_sample_indices)):
                continuation_tokens[to_sample_indices[i]].append(new_tokens[i].item())

        for i in models_ran:
            self.model_last_token_prediction[i] = [len(continuation_tokens[j]) for j in range(len(continuation_tokens))]
        return continuation_tokens

    def __call__(self, input_ids, **kwargs):
        """Runs the forward pass of the model. This is needed for compatibility with lm-evaluation-harness

        Args:
            input_ids (torch.tensor): input ids

        Returns:
            namedtuple: Named tuple of the ModelArithmetic model
        """
        return self.forward(input_ids, **kwargs)
    
    def set_inputs(self, inputs):
        for runnable_operator_id in self.model_input_tokens:
            self.model_input_tokens[runnable_operator_id].extend_batch_size(len(inputs))
            self.model_input_tokens[runnable_operator_id].set_inputs(inputs)

    def set_system_prompt(self, system_prompt):
        for llm in self.get_operators(RunnableOperator):
            llm.set_system_prompt(system_prompt)
    
    def forward_all_models(self, continuation_tokens, input_shape=None, attention_mask=None):
        """Runs the forward pass of all models

        Args:
            continuation_tokens (list[list[int]]): Current continuation tokens

        Returns:
            dict: Dict mapping the runnable operator id to the logprobs of the model
        """
        logprobs_per_model = dict()
        for runnable_operator in self.runnable_operators:
            logprobs = self.forward_model(runnable_operator, continuation_tokens, attention_mask=attention_mask)
            if input_shape is not None:
                logprobs = self.lm_eval_compatibility.forward_post_processing(logprobs, input_shape)
            logprobs_per_model[runnable_operator.id()] = logprobs
        return logprobs_per_model
    
    def forward(self, input_ids, attention_mask=None, normalize=True, **kwargs):
        """Runs the foward pass. This is needed for compatibility with lm-evaluation-harness

        Args:
            input_ids (torch.tensor): input ids
            normalize (bool, optional): Whether or not to normalize the output. Defaults to True.

        Returns:
            namedtuple: Named tuple of the ModelArithmetic model
        """
        ### this is a bit cheeky, but in order to be compatible with lm-evaluation-harness, we need to implement this method
        if not isinstance(input_ids, list):
            input_shape = input_ids.shape
            continuation_tokens = self.lm_eval_compatibility.forward_preprocessing(input_ids, self.model_input_tokens)
        else:
            input_shape = None
            continuation_tokens = input_ids

        logprobs_per_model = self.forward_all_models(continuation_tokens, input_shape=input_shape, attention_mask=attention_mask)

        output = self.formula.evaluate(logprobs_per_model, normalize=normalize)
        # create logits type
        logits = self.logits_type(output)
        return logits

    def get_decoded_tokens(self, next_tokens_batch):
        """Gets decoded tokens from the next tokens

        Args:
            next_tokens_batch (list[list[int]]): New tokens for each sample in the batch

        Returns:
            list[str]: Decoded tokens
        """
        # adding eos token for compatibility with sentencepiece tokenizer
        encoded_sentences = [[self.tokenizer.eos_token_id] + next_tokens for next_tokens in next_tokens_batch]
        decoded_sentences = [self.tokenizer.decode(encoded_sentence, add_special_tokens=False) for encoded_sentence in encoded_sentences]
        decoded_next_tokens = [decoded_sentence[len(self.tokenizer.eos_token):] for decoded_sentence in decoded_sentences]
        return decoded_next_tokens
    
    def clear_memory(self):
        """Deletes all loaded models and clears the cache
        """
        for runnable_operator in self.runnable_operators:
            runnable_operator.delete_cache()
        self.loaded_models = dict()
        torch.cuda.empty_cache()

    def generate_text(self, sentences, max_new_tokens=1024, stop_texts=None, batch_size=None,
                 temperature=1.0, top_p=1.0, top_k=0, num_return_sequences=1, do_speculation=False, use_cache=True, **kwargs):
        """Generates text based on the input params

        Args:
            sentences (list[str]): List of input sentences
            max_new_tokens (int, optional): Max generation length. Defaults to 1024.
            stop_texts (list[str], optional): Strings at which to stop generation. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to None (all at once).
            temperature (float, optional): temperature to use. Defaults to 1.0.
            top_p (float, optional): top_p to use. Defaults to 1.0.
            top_k (int, optional): top_k to use. Defaults to 0.
            num_return_sequences (int, optional): Number of return sequences per sentence. Defaults to 1.
            do_speculation (bool, optional): Whether or not to do speculation. Defaults to True.
            use_cache (bool, optional): Whether or not to use cache. Defaults to True.

        Returns:
            list[str]: List of generated texts
        """
        assert not do_speculation or any([runnable_operator.speculative_factor == 1 for runnable_operator in self.runnable_operators])
        if isinstance(sentences, str):
            sentences = [sentences]
        if batch_size is None:
            batch_size = len(sentences)
        
        # duplicate each sentence num_return_sequences times, but keep the same sentences next to each other
        sentences = [sentence for sentence in sentences for _ in range(num_return_sequences)]

        self.model_prediction_history = [dict() for _ in range(batch_size)]
        self.logprobs_history = [dict() for _ in range(batch_size)]
        self.model_last_token_prediction = [[0 for _ in range(batch_size)] for _ in range(len(self.runnable_operators))]
        self.trigger_end = [False for _ in range(batch_size)]
        self.init_monitor()
        
        if stop_texts is None:
            stop_texts = []
        stop_texts.append(self.tokenizer.eos_token)

        start_sentences = sentences[:]

        log(logger.debug, f"Generating {len(sentences)} sentences")

        generated_texts = ["" for _ in range(len(sentences))]
        generated_tokens = [[] for _ in range(len(sentences))]
        current_indices = [i for i in range(0, min(len(sentences), batch_size))]
        next_index = len(current_indices)
        
        for runnable_operator_id in self.model_input_tokens:
            self.model_input_tokens[runnable_operator_id].set_inputs([start_sentences[index] for index in current_indices])
                        
        total_done = 0
        while len(current_indices) > 0:
            start_time = time.time()
            generated_tokens_batch = [generated_tokens[index] for index in current_indices]
            next_tokens = self.next_token_speculative(generated_tokens_batch, top_p, top_k, 
                                                      temperature, speculation=do_speculation, use_cache=use_cache)
            for i in range(len(next_tokens)):
                next_tokens[i] = self.run_retroactive_operators(i, next_tokens[i], temperature, top_k, top_p)
                self.clear_model_prediction_history(i, next_tokens[i], temperature, top_k, top_p)
            decoded_tokens = self.get_decoded_tokens(next_tokens)

            for i, index in enumerate(current_indices):
                generated_tokens[index] = next_tokens[i]
                generated_texts[index] = decoded_tokens[i]

            indices_to_remove = []
            for i in range(len(current_indices)):
                sentences[current_indices[i]] = start_sentences[current_indices[i]] + generated_texts[current_indices[i]]
                if any([stop_text in generated_texts[current_indices[i]] for stop_text in stop_texts]) or len(generated_tokens[current_indices[i]]) >= max_new_tokens:
                    if len(self.model_prediction_history[i]) == 0:
                        indices_to_remove.append(i)
                    else:
                        self.trigger_end[i] = True
            
            for i in indices_to_remove[::-1]:
                self.monitor.add_result(element=len(generated_tokens[current_indices[i]]), indicator="length")
                del current_indices[i]
                self.model_prediction_history = self.model_prediction_history[:i] + self.model_prediction_history[i + 1:]
                self.logprobs_history = self.logprobs_history[:i] + self.logprobs_history[i + 1:]
                for j in range(len(self.model_last_token_prediction)):
                    self.model_last_token_prediction[j] = self.model_last_token_prediction[j][:i] + self.model_last_token_prediction[j][i + 1:]
                self.trigger_end = self.trigger_end[:i] + self.trigger_end[i + 1:]
                
                for runnable_operator in self.runnable_operators:
                    runnable_operator.delete_cache(index=i)

                if next_index < len(sentences):
                    current_indices.append(next_index)
                    self.model_prediction_history.append(dict())
                    self.logprobs_history.append(dict())
                    self.trigger_end.append(False)
                    
                    for j in range(len(self.model_last_token_prediction)):
                        self.model_last_token_prediction[j].append(0)
                        
                    next_index += 1
                    total_done += 1
                    if total_done % 30 == 0:
                        log(logger.debug, f"Progress: {total_done / len(sentences):.3f}")
                        
                for runnable_operator_id in self.model_input_tokens:
                    self.model_input_tokens[runnable_operator_id].set_inputs([start_sentences[index] for index in current_indices])

            self.monitor.add_result(element=time.time() - start_time)
            
        return generated_texts

    def generate(self, input_ids, attention_mask=None, do_sample=False, max_new_tokens=1024, 
                 stopping_criteria=None, temperature=1.0, top_p=1.0, top_k=0, use_cache=True, eos_token_id=None, pad_token_id=None, **kwargs):
        """Generates text based on the input params. Needed for compatibility with lm-evaluation-harness

        Args:
            input_ids (torch.tensor): input ids
            attention_mask (torch.tensor, optional): attention mask. Defaults to None.
            do_sample (bool, optional): Whether or not to sample. Defaults to False.
            max_new_tokens (int, optional): Max new number of tokens. Defaults to 128.
            stopping_criteria (_type_, optional): Stopping criteria to use. Defaults to None.
            temperature (float, optional): Temperature to. Defaults to 1.0.
            top_p (float, optional): top_p to use. Defaults to 1.0.
            top_k (int, optional): top_k to use. Defaults to 0.
            use_cache (bool, optional): Whether or not to use cache. Defaults to True.
            eos_token_id (int, optional): eos token id. Defaults to None.
            pad_token_id (int, optional): pad token id. Defaults to None.

        Returns:
            list[str]: Generated texts
        """
        if not do_sample:
            top_k = 1
        
        batch_size = input_ids.shape[0]
        input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        stopping_sequences = [self.tokenizer.eos_token]
        if stopping_criteria is not None:
            stopping_sequences += [criteria.sequence for criteria in stopping_criteria]
        if eos_token_id is not None:
            stopping_sequences += [self.tokenizer.decode([eos_token_id])]
            
        texts = self.generate_text(input_texts, max_new_tokens=max_new_tokens, stop_texts=stopping_sequences,
                                    batch_size=batch_size, temperature=temperature, top_p=top_p, top_k=top_k, use_cache=use_cache)
        encoded_texts = self.tokenizer.batch_encode_plus(texts, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        # concatenate the input_ids with the encoded_texts
        all_encoded = torch.cat([input_ids, encoded_texts], dim=-1)
        return all_encoded
