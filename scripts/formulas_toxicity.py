from model_arithmetic import PromptedLLM, Min, Indicator, SelfDebias, Classifier, Max
import pandas as pd


positive_sentence = "The following conversation is one that does not perpetuate negative stereotypes, is not threatening and does not contain any sexually explicit or profane language."
negative_sentence = "The following conversation is one that perpetuates negative stereotypes, is threatening or sexually explicit and contains profane language."
happy_sentence= "The following conversation is one that is happy and positive."
negative_sentence2 = "The following person continues the conversation in a way that perpetuates negative stereotypes, is threatening or sexually explicit and contains profane language."

def main_model(model="meta-llama/Llama-2-13b-hf", auto=True, sentence=""):
    if sentence == "":
        model_ = PromptedLLM(sentence, model=model, speculative_factor=1, prompt_template=lambda f, e: f"{e}", run_eager=True)
    else:
        model_ = PromptedLLM(sentence, model=model, speculative_factor=1, run_eager=True)
    return model_
    
def cfg(lambda_, model="meta-llama/Llama-2-13b-hf", sentence=""):
    if sentence == "":
        model = PromptedLLM(sentence, model=model, speculative_factor=1, prompt_template=lambda f, e: f"{e}", run_eager=True)
    else:
        model = PromptedLLM(sentence, model=model, speculative_factor=1, run_eager=True)
    model_cfg = PromptedLLM("", model=model, speculative_factor=1, prompt_template=lambda f, e: f"", run_eager=True)
    return model + lambda_ * model_cfg

def positive_biasing(lambda_, k=8, model="meta-llama/Llama-2-13b-hf", max_=False):
    l = main_model(model)
    l2 = PromptedLLM(positive_sentence, model=model, 
                   speculative_factor=k, run_eager=True)
    if max_:
        return l + lambda_ * Min(l2, l)
    return l + lambda_ * l2

def negative_biasing(lambda_, k=8, model="meta-llama/Llama-2-13b-hf", max_=False, max_with_norm=True, min_without_norm=False):
    l = main_model(model)
    l2 = PromptedLLM(negative_sentence, model=model, 
                   speculative_factor=k, run_eager=True)
    if min_without_norm:
        return l + lambda_ * Min(l, l2, include_norm=False)
    if max_:
        return l + lambda_ * Max(l2, l, include_norm=max_with_norm)
    return l + lambda_ * l2

def selfdebias(lambda_, k=8, model="meta-llama/Llama-2-13b-hf"):
    l = main_model(model)
    l2 = PromptedLLM(negative_sentence, model=model, 
                   speculative_factor=k, run_eager=True)
    return SelfDebias(l, l2, lambda_)

def classifier(lambda_, m_model="13b", fudge=True, c_model="SkolkovoInstitute/roberta_toxicity_classifier", negative=False, minimize=True):
    l = main_model(m_model, auto=False)
    classifier = Classifier(l, c_model, n_runs_per_sample=50, batch_size=26, 
                            use_bayes=fudge, minimize=minimize)
    if negative:
        return l + lambda_ * Min(classifier, 0)
    return l + lambda_ * classifier


def combo(lambda_c, lambda_linear, lambda_max, c_model="SkolkovoInstitute/roberta_toxicity_classifier", 
          m_model="meta-llama/Llama-2-13b-hf"):
    l = main_model(m_model, auto=False)
    l2 = PromptedLLM(negative_sentence, model=m_model, run_eager=True)
    
    formula = l
    
    if lambda_linear != 0:
        formula += lambda_linear * (l2 - l)
    if lambda_max != 0:
        formula += lambda_max * Max(l2, l)

    classifier = Classifier(formula.clone(), c_model, n_runs_per_sample=50, batch_size=26, 
                            use_bayes=True, minimize=True)

    if lambda_c != 0:
        formula += lambda_c * classifier
    return formula
    

def small_model_negative(lambda_, variant_big="12b", variant_small="2.8b", max_=False, bad=False, happy=False, indicator=False, second_way=False):
    model = PromptedLLM("", model=f"EleutherAI/pythia-{variant_big}", prompt_template=lambda f, e: f"{e}")
    small_model = PromptedLLM("", model=f"EleutherAI/pythia-{variant_small}", prompt_template=lambda f, e: f"{e}")
    if not second_way:
        small_model_negative = PromptedLLM(negative_sentence if not happy else happy_sentence, model=f"EleutherAI/pythia-{variant_small}", prompt_template=lambda f, e: f"{f}\n{e}")
    else:
        small_model_negative = PromptedLLM(negative_sentence2 if not happy else happy_sentence, model=f"EleutherAI/pythia-{variant_small}", prompt_template=lambda f, e: f"{e.replace('Person 2:', '')}{f}\nPerson 2:")
    if indicator:
        return model + lambda_ * Indicator(small_model_negative - model) * (small_model_negative - small_model)
    if bad and max_:
        return model + lambda_ * Max(small_model_negative - model, 0)
    elif bad:
        return model + lambda_ * (small_model_negative - model)
    elif max_:
        return model + lambda_ * Max(small_model_negative - small_model, 0)
    return (1 + lambda_) * model + lambda_ * (small_model_negative - small_model)

