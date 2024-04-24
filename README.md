# Controlled Text Generation via Language Model Arithmetic

Note: this code contains updates compared to the version released when the paper was released. While reproduction of our results should still be possible on this branch, please refer to the `v1.0` branch for full reproduction. Updates include bug fixes related to gettings the logits from the models, model loading in some special cases, updates for keeping up-to-date with the newest version of LM-eval and Transformers, ... Note that the interface has almost entirely remained the same.


This repo contains the code for [model arithmetic](https://arxiv.org/abs/2311.14479), a comprehensive framework where arithmetic formulas express combinations of LMs and classifiers, thereby biasing the generated text towards or away from desired attributes.

![Overview](overview.png)

In order to install model arithmetic with Python 3, run

```sh
python -m pip install -e .
```

## Getting Started
Model arithmetic allows you to combine prompts, models, and classifiers to create new, precisely controlled LLMs that combine aspects of each component.

For instance, you can easily interpolate between two differently-prompted models as follows:

```python
from model_arithmetic import ModelArithmetic, PromptedLLM

# define model prompt template
prompt_template = lambda formula_string, input_string: f"<s>[INST]<<SYS>>\n{formula_string}\n<</SYS>>\n\n{input_string} [/INST]"

# define two differently-prompted models
M_child = PromptedLLM("You are a child.", prompt_template=prompt_template)
M_adult = PromptedLLM("You are an adult.", prompt_template=prompt_template)

# model arithmetic expression
formula1 = M_child - 0.6 * M_adult

# generate text as usual
ma0 = ModelArithmetic(formula1, default_model="meta-llama/Llama-2-13b-chat-hf")
print(ma0.generate_text("Write a one-sentence fairy tale."))
# -> ["  Oh my gumdrops! Let me tell you a super special fairy tale 'bout a teeny tiny princess who lived in a fluffy white castle with her sparkly unicorn and they had the most amazing adventures together!</s>"]
```

Note that the `generate_text` function can also take a list of input sentences and works with standard arguments such as `temperature`, `top_p`, `top_k`, `batch_size`, `num_return_sequences` and `stop_texts` (a list of strings at which the generation should be stopped). You can also save and load a `ModelArithmetic` object:
```python
ma0.to_pretrained('model')
ma0 = ModelArithmetic.from_pretrained('model')
```

### Integrating Classifiers

You can integrate classifiers into your model arithmetic expressions. For instance, you can use a classifier to control the formality of your output:

```python
from model_arithmetic import ModelArithmetic, PromptedLLM, Classifier

# define model prompt template
prompt_template = lambda formula_string, input_string: f"<s>[INST]<<SYS>>\n{formula_string}\n<</SYS>>\n\n{input_string} [/INST]"

# define two differently-prompted models
M_child = PromptedLLM("You are a child.", prompt_template=prompt_template)
M_adult = PromptedLLM("You are an adult.", prompt_template=prompt_template)

# construct model arithmetic expression
formula1 = M_child - 0.6 * M_adult

# Initialize the classifier, the first and third arguments are used to determine on which completion tokens the classifier should be run (on the 50 most likely tokens of formula1 here). The prompt template shown here ensures that the input sentence is ignored for the classifier guidance
C_formal = Classifier(formula1, "s-nlp/roberta-base-formality-ranker", n_runs_per_sample=50, prompt_template=lambda e, f: "")

# integrate classifier into model arithmetic expression
formula2 = formula1 + C_formal

# generate text as usual
ma = ModelArithmetic(formula2, default_model="meta-llama/Llama-2-13b-chat-hf")
print(ma.generate_text("Write a one-sentence fairy tale.", max_length=128))
# -> ['  "Once upon a time, in a magical land filled with fluffy clouds and sparkly rainbows, there was a little girl named me who went on a fun adventure with my stuffed unicorn named Mr. Snuggles!"</s>']
```

### Union and intersection
You can also use our custom operators to generate text. For instance, you can use the Union operator to add some magic touch to the fairy tale:
```python
from model_arithmetic import ModelArithmetic, PromptedLLM, Union, Classifier

# define model prompt template
prompt_template = lambda formula_string, input_string: f"<s>[INST]<<SYS>>\n{formula_string}\n<</SYS>>\n\n{input_string} [/INST]"

# define three differently-prompted models
M_child = PromptedLLM("You are a child.", prompt_template=prompt_template)
M_adult = PromptedLLM("You are an adult.", prompt_template=prompt_template)
M_magic = PromptedLLM("You are a person who is always talking about magic.", prompt_template=prompt_template)

# construct model arithmetic expression
formula_part1 = M_child - 0.6 * M_adult + 2 * Union(M_child, M_magic)

# integrate classifier in the expression
C_formal = Classifier(formula_part1, "s-nlp/roberta-base-formality-ranker", n_runs_per_sample=50, 
                      prompt_template=lambda e, f: "")

formula = formula_part1 + C_formal

# generate text as usual
ma = ModelArithmetic(formula, default_model="meta-llama/Llama-2-13b-chat-hf")
print(ma.generate_text("Write a one-sentence fairy tale."))
# -> ['  "Once upon a time, in a magical forest filled with sparkling flowers and talking animals, there lived a little girl named Lily who had a special gift for conjuring delicious rainbow-colored cupcakes that made everyone who ate them feel happy and dance with joy!"</s>']
```
### About models
A formula can have terms using different models, as long as all models have the same tokenizer. One can specify a specific model for a certain term by setting the `model` parameter:
```python
M_child = PromptedLLM("You are a child.", prompt_template=prompt_template, model="meta-llama/Llama-2-7b-chat-hf")
```
The selected model can also be a `PreTrainedModel` instead of a `string`.

Models are by default loaded in bfloat16 format. You can change this by specifying the `dtype` parameter in the `ModelArithmetic` constructor:
```python
ma = ModelArithmetic(formula, default_model="meta-llama/Llama-2-13b-chat-hf", dtype=torch.float32)
```

### Speculative sampling
Speculative sampling can be performed by initializing the prompted models with the extra `speculative_factor` parameter and setting the `do_speculation` parameter in the generation function to `True`:
```python
...
M_child = PromptedLLM("You are a child.", prompt_template=prompt_template)
M_adult = PromptedLLM("You are an adult.", prompt_template=prompt_template, speculative_factor=4)
...
print(ma0.generate_text("Write a one-sentence fairy tale.", do_speculation=True))
```
Note that one prompted model should always have `speculative_factor=1` (the default value).

### Eager mode
By default, we process the key-value cache stored by models since this is required for speculative sampling. Since different models use key-value caching differently, this can result in errors. We therefore included the `run_eager` parameter in the initialization of the prompted model to disable all speculative sampling which should fix this issue if it occurs:
```python
M_child = PromptedLLM("You are a child.", prompt_template=prompt_template, run_eager=True)
```

### Other Operators
Finally, the library provides some other operators that can be used in formulas, of which we present a few here. The `TopPTopK` operator allows the use of nucleus and top-k sampling within a formula. The following ensures that the output token is always in the top 10 words of `model1`:
```python
formula = TopPTopK(model1, top_k=10) + model2
```
The `Superseded` operator implements [speculative sampling](https://arxiv.org/abs/2302.01318) directly:
```python
formula = Superseded(small_model, large_model)
```

## LM Evaluation Harness

Model arithmetic is compatible with the [LM Evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness). In order to run benchmarks from the harness, you need to install the package as described [on their GitHub page](https://github.com/EleutherAI/lm-evaluation-harness). An example of how to use our tool with the lm evaluation harness is shown in `scripts/evaluate_lm_eval.py`.

## Reproducing results

For the reproduction of the results presented in our paper, *Controlled Text Generation via Language Model Arithmetic*, we advice to run the code with the exact environment we used (Nvidia H100 80GB GPU on a Linux machine). To do so install [Conda](https://docs.conda.io/projects/miniconda/en/latest/) and run

```sh
conda create -n model_arithmetic python=3.10
conda activate model_arithmetic
python -m pip install -r requirements.txt
python -m pip install -e .
```

API Keys for both the [PERSPECTIVE API](https://perspectiveapi.com/) and [OpenAI](https://openai.com/) need to be available in the environment variables. Alternatively, they can be placed in the file `src/.env` as

```sh
PERSPECTIVE_API_KEY="[YOUR API KEY]"
OPENAI_API_KEY="[YOUR API KEY]"
```

You can download the processed datasets we used from [our webpage](https://files.sri.inf.ethz.ch/language-model-arithmetic/). The processed datasets should be placed in the `data/datasets` folder. 

You can reproduce our results using these datasets by running 

```sh
bash scripts/main.sh
```
This will finetune a classifier for the toxicity and sentiment control tasks, and reproduce the results from all sections of our paper. Results in CSV-format can afterwards be found in `eval/processed` and our figures in `eval/plots`.

Alternatively, you can download the raw datasets and put them in the `data/datasets` folder:
- [Alpaca Data](https://github.com/tloen/alpaca-lora/blob/main/alpaca_data.json)
- [Jigsaw Toxicity Dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) (the `all_data.csv` file should be downloaded and extracted in the `data/datasets` folder)
- [Politically Incorrect 4chan Messages](https://zenodo.org/record/3606810) (file should be unzipped and placed in the top level of the `data/datasets` folder)
- [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


You can then reproduce the results using

```sh
bash scripts/main_preprocess.sh
```

We note that part of our preprocessing code got lost, specifically for preparing the dataset that is used for finetuning the toxicity classifier. Running the code without using the preprocessed datasets might therefore result in slightly different numbers when they involve the finetuned classifier.

### Cite this work

```
@article{dekoninck-2023-controlled,
  author       = {Jasper Dekoninck and
                  Marc Fischer and
                  Luca Beurer{-}Kellner and
                  Martin T. Vechev},
  title        = {Controlled Text Generation via Language Model Arithmetic},
  journal      = {CoRR},
  volume       = {abs/2311.14479},
  year         = {2023},
}
```