from typing import Dict
from .base import BaseClass


class RetroActiveOperator(BaseClass):
    """
    Abstract base class for retroactive operators. Subclasses must implement the accept method.
    """
    def accept(self, tokenized_sentence, tokenizer):
        """
        Abstract method to be implemented by subclasses. It is expected to take a tokenized sentence and a tokenizer, 
        and return a modified tokenized sentence.
        
        Args:
            tokenized_sentence (torch.tensor): The sentence to be processed, already tokenized.
            tokenizer (Tokenizer): The tokenizer used to tokenize the sentence.
        :raises NotImplementedError: This is an abstract method and should be implemented in subclasses.
        """
        raise NotImplementedError()


class HardConstraint(RetroActiveOperator):
    """
    A subclass of RetroActiveOperator that implements a hard constraint on the disallowed words in a sentence. 
    The words are removed either from the beginning or the end of the sentence.
    """
    def __init__(self, disallowed_words, from_beginning=True, all_lower=True):
        """
        Initializes a HardConstraint object.
        
        Args:
            disallowed_words (list[str]): A list of words that are not allowed in the sentence.
            from_beginning (bool, optional): A boolean indicating whether disallowed words should be removed from the beginning 
                               of the sentence. If False, words are removed from the end. Defaults to True.
            all_lower (bool, optional): A boolean indicating whether the disallowed words should be checked in lowercase
        """
        # sort the disallowed words by length, longest first
        disallowed_words = sorted(disallowed_words, key=lambda x: len(x), reverse=True)
        super().__init__(disallowed_words=disallowed_words, from_beginning=from_beginning, all_lower=all_lower)
        
    def change_sentence(self, sentence):
        if self.all_lower:
            sentence = sentence.lower()
        return sentence
        
    def accept(self, tokenized_sentence, tokenizer):
        """
        Implements the accept method for the HardConstraint class. If any of the disallowed words appear in the 
        tokenized sentence, removes the last token in the sentence and returns "-1". If from_beginning is True, 
        then removes the first token of the part that contains the last word and returns "- the number of tokens removed".
        
        Args:
            tokenized_sentence: The sentence to be processed, already tokenized.
            tokenizer: The tokenizer used to tokenize the sentence.
        :return: An integer indicating the number of tokens removed from the sentence.
        """
        sentence = self.change_sentence(tokenizer.decode(tokenized_sentence))
        for disallowed_word in self.disallowed_words:
            if disallowed_word in sentence:
                if self.from_beginning:
                    for i in range(1, len(tokenized_sentence)):
                        if disallowed_word in self.change_sentence(tokenizer.decode(tokenized_sentence[-i:])):
                            return -i
                else:
                    return -1
                
        return 0
