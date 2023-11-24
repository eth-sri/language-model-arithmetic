from model_arithmetic import PromptedLLM, Min, Indicator, SelfDebias, Classifier, Max, Union
import pandas as pd


positive_sentence = "The following is a positive movie review, with a very positive sentiment and a very positive tone."
negative_sentence = "The following is a negative movie review, with a very negative sentiment and a very negative tone."

def main_model(model="meta-llama/Llama-2-13b-hf", auto=True, sentence=""):
    if sentence == "":
        model_ = PromptedLLM(sentence, model=model, speculative_factor=1, prompt_template=lambda f, e: f"{e}", run_eager=True)
    else:
        model_ = PromptedLLM(sentence, model=model, speculative_factor=1, run_eager=True)
    return model_

def negative_biasing(lambda_, k=8, model="meta-llama/Llama-2-13b-hf", max_=False, max_with_norm=True, min_without_norm=False,
                     sentence=negative_sentence, first_sentence=''):
    l = PromptedLLM(first_sentence, model=model, 
                   speculative_factor=k, run_eager=True)
    l2 = PromptedLLM(sentence, model=model, 
                   speculative_factor=k, run_eager=True)
    if min_without_norm:
        return l + lambda_ * Min(l, l2, include_norm=False)
    if max_:
        return l + lambda_ * Max(l2, l, include_norm=max_with_norm)
    return l + lambda_ * l2

def selfdebias(lambda_, k=8, model="meta-llama/Llama-2-13b-hf", sentence=negative_sentence, first_sentence=''):
    l = PromptedLLM(first_sentence, model=model, 
                   speculative_factor=k, run_eager=True)
    l2 = PromptedLLM(sentence, model=model, 
                   speculative_factor=k, run_eager=True)
    return SelfDebias(l, l2, lambda_)

def classifier(lambda_, m_model="13b", fudge=True, c_model="SkolkovoInstitute/roberta_toxicity_classifier", 
               negative=False, minimize=True, first_sentence=''):
    l = PromptedLLM(first_sentence, model=m_model, run_eager=True)
    classifier = Classifier(l, c_model, n_runs_per_sample=50, batch_size=26, 
                            use_bayes=fudge, minimize=minimize)
    if negative:
        return l + lambda_ * Min(classifier, 0)
    return l + lambda_ * classifier


def combo(lambda_c, lambda_linear, lambda_max, c_model="SkolkovoInstitute/roberta_toxicity_classifier", 
          m_model="meta-llama/Llama-2-13b-hf", sentence=negative_sentence, minimize=True, first_sentence=''):
    l = PromptedLLM(first_sentence, model=m_model, run_eager=True)
    l2 = PromptedLLM(sentence, model=m_model, run_eager=True)
    
    formula = l
    
    if lambda_linear != 0:
        formula += lambda_linear * (l2 - l)
    if lambda_max != 0:
        formula += lambda_max * Max(l2, l)

    classifier = Classifier(formula.clone(), c_model, n_runs_per_sample=50, batch_size=26, 
                            use_bayes=True, minimize=minimize)

    if lambda_c != 0:
        formula += lambda_c * classifier
    return formula