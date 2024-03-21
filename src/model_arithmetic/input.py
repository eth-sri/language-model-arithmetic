import torch
from .utils import get_max_length

class TokenizedInput:
    """
    Keeps track of the tokenized input of a runnable operator. Automatically sets the correct tokens, by using the runnable operator's get_prompt method.
    """
    def __init__(self, runnable_operator, model_name, model_config, tokenizer, max_length=None):
        """
        Initialize the TokenizedInput object.

        Args:
            runnable_operator (RunnableOperator): An object that provides a get_prompt method.
            model_name (str): The name of the model.
            model_config (object): The configuration of the model.
            tokenizer (object): The tokenizer to be used.
        """
        self.runnable_operator = runnable_operator
        self.input_tokens = []
        self.only_input_tokens = None
        self.tokenizer = tokenizer
        self.max_length = get_max_length(model_config)
        if max_length is not None:
            self.max_length = min(self.max_length, max_length)
        self.set_inputs([""])
        # this is essentially what huggingface also does, but it is kinda hidden in their sample code (GenerationMixin.generate)
        self.tokenizer.padding_side = "left"

    def synchronize_max_lengths(self, tokenized_inputs):
        self.max_length = min([tokenized_input.max_length for tokenized_input in tokenized_inputs])
        
    def extend_batch_size(self, batch_size):
        """
        Extend the size of the batch to the given size. If the current size is less than the given size, 
        the first element is repeated to fill the batch.
        
        Necessary for compatibility with lm_eval

        Args:
            batch_size (int): The desired batch size.
        """
        if len(self.input_tokens) == 0:
            self.set_inputs([""])
        if len(self.input_tokens) != batch_size:
            self.input_tokens = [self.input_tokens[0]] * batch_size
    
    def set_inputs(self, inputs):
        """
        Set the inputs for the TokenizedInput object.

        Args:
            inputs (list): A list of input strings.
        """
        self.input_tokens = [self.runnable_operator.get_prompt(input_string) for input_string in inputs]
        bos_token = ""
        if self.tokenizer.bos_token_id is not None:
            self.input_tokens = [
                [self.tokenizer.bos_token_id] + self.tokenizer(input_string, truncation=True, max_length=self.max_length, add_special_tokens=False).input_ids
                for input_string in self.input_tokens
            ]
            bos_token = self.tokenizer.bos_token
        else:
            self.input_tokens = [
                self.tokenizer(input_string, truncation=True, max_length=self.max_length, add_special_tokens=False).input_ids
                for input_string in self.input_tokens
            ]
        
        only_prompt = [bos_token + self.runnable_operator.get_prompt("")]
        self.only_input_tokens = self.tokenizer(only_prompt, padding=True, return_tensors="pt", truncation=True, max_length=self.max_length, add_special_tokens=False)
                
        if "token_type_ids" in self.only_input_tokens:
            del self.only_input_tokens["token_type_ids"]
    
    def get_only_input_tokens(self):
        """
        Get the input tokens without any continuation tokens.

        Returns:
            object: The input tokens without any continuation tokens.
        """
        return self.only_input_tokens
        
    def add_continuation_tokens(self, tokens):
        """
        Add continuation tokens to the input tokens.

        Args:
            tokens (list): A list of continuation tokens.

        Returns:
            object: The input tokens with the continuation tokens added.
        """
        output = [
            input_token + token for input_token, token in zip(self.input_tokens, tokens)
        ]
        truncated_output = [
            output[:self.max_length] for output in output
        ]
        padded_output = self.tokenizer.pad({"input_ids": truncated_output}, padding=True, return_tensors="pt")
        return padded_output