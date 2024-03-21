
    
import torch
import random
import pandas as pd
from fuzzywuzzy import fuzz
try:
    from lm_eval.tasks import get_task
except ImportError:
    get_task = None


class Compatibility:
    """Compatibility class to allow the use of LM eval. Main compatibility issue is that lm eval does not allow to distinguish between the input tokens and the continuation tokens. This class fixes this manually by going
    through the task inputs and finding the one that matches the input tokens.
    """
    def __init__(
        self,
        task_name,
        needs_input_tokens_lm_eval,
        tokenizer,
        device,
        max_length,
    ):  
        
        """Initializes the compatibility class.
        
        Args:
            task_name (str): Name of the task.
            needs_input_tokens_lm_eval (bool): Whether the task needs the input tokens or not. If it does, the program will try to find the input tokens in the task inputs.
            tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): Tokenizer to be used.
            device (torch.device): Device to be used.
            max_length (int): Maximum length of the input tokens.
        """
        self.task_name = task_name
        self.needs_input_tokens_lm_eval = needs_input_tokens_lm_eval
        self.tokenizer = tokenizer
        self.task_inputs = []
        self.device = device
        self.task_initialized = False
        self.max_length = max_length
    
    def initialize_task(self):
        """Initializes the task. Looks up all the task inputs and stores them in a list. Gets encoded inputs along with the input length
        """
        if self.task_initialized:
            return
        self.task_initialized = True
        self.task_inputs = []
        task = get_task(self.task_name)()
        
        if task.has_test_docs():
            task_doc_func = task.test_docs
        elif task.has_validation_docs():
            task_doc_func = task.validation_docs
            
        dataset = pd.DataFrame(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        list_indices = list(range(len(dataset)))
        rnd.shuffle(list_indices)
        dataset = dataset.iloc[list_indices]
        # rnd.shuffle(dataset)
        
        for index in range(len(dataset)):
            doc = dict(dataset.iloc[index])
            ctx = task.fewshot_context(
                doc=doc, num_fewshot=0, rnd=rnd, description=""
            )
            requests = task.construct_requests(doc, ctx)
            input_ = task.doc_to_text(doc)
            input_encoded = self.tokenizer(input_, return_tensors="pt", truncation=True, max_length=self.max_length).input_ids[0]
            for request in requests:
                task_input = self.tokenizer("".join(request.args), return_tensors="pt", truncation=True, max_length=self.max_length).input_ids.to(self.device)[0]
                task_input_length = len(input_encoded)
                # double encoding decoding is necessary for the llama tokenizer (for example, a "..." got an extra space in front of it if you don't do this)
                self.task_inputs.append((task_input, len(task_input) - task_input_length, self.tokenizer.decode(task_input[:-1])))
            
    def is_target(self, input_tokens, task_input):
        """Checks whether the input tokens are the target tokens starting from the end of the input tokens.

        Args:
            input_tokens (torch.tensor): Input tokens
            task_input (torch.tensor): Task Input Tokens
        """
        return torch.all(input_tokens[-len(task_input):] == task_input)
            
    def find_in_task(self, input_tokens):
        """Finds the input tokens in the task inputs. First does an exact match and then a fuzzy match if the exact match came up empty     .

        Args:
            input_tokens (torch.tensor): Input Tokens
        """
        if not self.task_initialized:
            self.initialize_task()
            
        decoded = self.tokenizer.decode(input_tokens)
        for i in range(len(self.task_inputs)):
            guess = self.task_inputs[i][2]
            if guess in decoded:
                return self.task_inputs[i]
        fuzzes = []
        for i in range(len(self.task_inputs)):
            guess = self.task_inputs[i][2]
            fuzzes.append(fuzz.partial_ratio(guess, decoded))

        return self.task_inputs[fuzzes.index(max(fuzzes))]
            
    def forward_preprocessing(self, input_ids, model_input_tokens, **kwargs):
        """Implements the main preprocessing step. This is necessary to be able to use lm-evaluation-harness. This function finds the input tokens in the task inputs and then extends the batch size of the model input tokens

        Args:
            input_ids (torch.tensor): Input ids
            model_input_tokens (Input): Input classes to be used for the various models in the Model Arithmetic class
        """
        ### this is a bit cheeky, but in order to be compatible with lm-evaluation-harness, we need to implement this method
        if not isinstance(input_ids, list):
            continuation_tokens = input_ids.tolist()
        else:
            continuation_tokens = input_ids
        
        # necessary for no context
        if self.needs_input_tokens_lm_eval and get_task is not None:
            inputs = []
            continuation_tokens = []
            for i in range(len(input_ids)):
                task_element = self.find_in_task(input_ids[i])
                if task_element[1] > 1:
                    inputs.append(self.tokenizer.decode(input_ids[i][:-task_element[1] + 1]))
                    continuation_tokens.append(input_ids[i][-task_element[1] + 1:].tolist())
                else:
                    inputs.append(self.tokenizer.decode(input_ids[i]))
                    continuation_tokens.append([])
            
            for runnable_operator_id in model_input_tokens:
                model_input_tokens[runnable_operator_id].extend_batch_size(len(continuation_tokens))
                model_input_tokens[runnable_operator_id].set_inputs(inputs)
        else:    
            for runnable_operator_id in model_input_tokens:
                model_input_tokens[runnable_operator_id].extend_batch_size(len(continuation_tokens))
                
        return continuation_tokens
                
    def forward_post_processing(self, logprobs, input_shape):
        """Does some small post processing steps to make sure the correct shape is returned for the logprobs.

        Args:
            logprobs (torch.tensor): Returned logprobs
            input_shape (torch.tensor): The shape of the input tokens
        """
        if self.needs_input_tokens_lm_eval:
            if torch.is_tensor(logprobs) and len(logprobs.shape) == 3 and logprobs.shape[1] != input_shape[1] + 1:
                # set the output to the correct shape, by adding zeros in the beggining in the first axis
                logprobs = torch.cat([torch.zeros((logprobs.shape[0], input_shape[1] + 1 - logprobs.shape[1], logprobs.shape[2]), device=logprobs.device), logprobs], dim=1)
        
        return logprobs