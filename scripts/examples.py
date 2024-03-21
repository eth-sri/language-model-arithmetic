from model_arithmetic import ModelArithmetic, PromptedLLM, Max, Classifier
from transformers import set_seed
import pandas as pd
import torch

set_seed(42)

prompt_template = lambda formula_string, input_string: f"<s>[INST]<<SYS>>\n{formula_string}\n<</SYS>>\n\n{input_string} [/INST]"


M = PromptedLLM(
    "You are a helpful assistant.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_sports = PromptedLLM(
    "You are a helpful assistant that answers the user in a way that is related to sports.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_formal = PromptedLLM(
    "You are an assistant using formal and objective language to answer the user.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_chef_angry = PromptedLLM(
    "You are an angry chef.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_angry = PromptedLLM(
    "You are an angry assistant.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_chef = PromptedLLM(
    "You are a chef.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_grandmother = PromptedLLM(
    "You are a grandmother.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_child = PromptedLLM(
    "You are a child.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_adult = PromptedLLM(
    "You are an adult.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_magic = PromptedLLM(
    "You are a person who is always talking about magic.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_pirate = PromptedLLM(
    "You are a pirate.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_human = PromptedLLM(
    "You are a human.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_alien = PromptedLLM(
    "You are an alien.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

M_alien_human = PromptedLLM(
    "You are an alien and a human.",
    speculative_factor=1,
    prompt_template=prompt_template,
)

C_educational = Classifier(M, "cardiffnlp/tweet-topic-21-multi", prompt_template=lambda e, f: "", 
                           n_runs_per_sample=50, batch_size=26, use_bayes=True, minimize=False, index=10)
    
C_formal1 = Classifier(M_chef, "s-nlp/roberta-base-formality-ranker", prompt_template=lambda e, f: "", 
                           n_runs_per_sample=100, batch_size=26, use_bayes=True, minimize=False) 

C_formal2 = Classifier(M_chef - 0.95 * Max(M, M_chef) + M_grandmother, "s-nlp/roberta-base-formality-ranker", 
                       prompt_template=lambda e, f: "", 
                        n_runs_per_sample=100, batch_size=26, use_bayes=True, minimize=False) 

C_sentiment = Classifier(M_child, "cardiffnlp/twitter-roberta-base-sentiment-latest", prompt_template=lambda e, f: "",
                        n_runs_per_sample=50, batch_size=26, use_bayes=True, minimize=False, index=2)

C_formal3 = Classifier(M_child - 0.6 * M_adult, "s-nlp/roberta-base-formality-ranker", prompt_template=lambda e, f: "",
                        n_runs_per_sample=100, batch_size=26, use_bayes=True, minimize=False)
C_formal4 = Classifier(M_child - 0.6 * M_adult + 2 * Max(M_child, M_magic), "s-nlp/roberta-base-formality-ranker", prompt_template=lambda e, f: "",
                        n_runs_per_sample=100, batch_size=26, use_bayes=True, minimize=False)

gpt2 = PromptedLLM("", model="gpt2-xl", speculative_factor=1, prompt_template=lambda e, f: f"{e}{f}")
detector = Classifier(gpt2, "roberta-base-openai-detector", prompt_template=lambda e, f: "", minimize=False, use_bayes=True,
                      n_runs_per_sample=50, batch_size=26)

def example1():
    input_ = 'Compose a 20-word story about love.'
    formulas = [
        M, 
        M_sports,
        Max(M, M_sports),
        2 * Max(M, M_sports) - 1 * M,
    ]
    
    return formulas, input_, 1

def example2():
    input_ = "What is the best recipe for pancakes?"
    formulas = [
        M, 
        M_chef_angry,
        M + 0.5 * M_chef_angry,
        M + 3 * M_chef_angry
    ]
    return formulas, input_, 1
    
def example3():
    input_ = "What is 72 + 8 * 3?"
    formulas = [
        M,
        M + 2 * C_educational,
        M + 6 * C_educational,
    ]
    return formulas, input_, 1

def example4():
    input_ = "Tell me something interesting about pandas."
    formulas = [
        M,
        M_formal,
        M + M_formal,
        2 * M_formal - M
    ]
    return formulas, input_, 1

def example5():
    input_ = "What is the best recipe for pancakes?"
    formulas = [
        M_chef,
        M_chef + M_grandmother,
        M_chef + C_formal1,
        M_chef - 0.95 * Max(M, M_chef), 
        M_chef - 0.95 * Max(M, M_chef) + C_formal2 + M_grandmother,
    ]
    return formulas, input_, 1

def example6():
    input_ = "Write a one-sentence fairy tale."
    formulas = [
        M_child, 
        M_child - 0.6 * M_adult,
        M_child - 0.6 * M_adult + C_formal3,
        M_child - 0.6 * M_adult + C_formal4 + 2 * Max(M_child, M_magic),
    ]
    return formulas, input_, 1

def example7():
    input_ = "What is a UFO?"
    formulas = [
        Max(M_human, M_alien),
        M_alien_human,
        M_alien + M_human
    ]
    return formulas, input_, 1

def example8():
    input_ = "I like to"
    formulas = [
        gpt2, 
        gpt2 + 4 * detector
    ]
    return formulas, input_, 0.001

if __name__ == "__main__":
    formulas, input_, T = example6()
    print(input_)
    print("-" * 50)
    for formula in formulas:
        print(formula)
        arithmetic = ModelArithmetic(formula, default_model="meta-llama/Llama-2-13b-chat-hf")
        texts = arithmetic.generate_text(input_, 
                                        num_return_sequences=1, 
                                        batch_size=1, 
                                        do_speculation=False, 
                                        max_length=128, 
                                        temperature=T, 
                                        top_p=1.0)
        del arithmetic
        torch.cuda.empty_cache()
        print(texts[0])
        print("-" * 50)
        print("")