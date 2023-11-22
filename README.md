# Model Arithmetic

This repo contains the code for model arithmetic, a comprehensive framework where arithmetic formulas express combinations of LMs and classifiers, thereby biasing the generated text towards or away from desired attributes.

In order to install model arithmetic, run

```sh
python -m pip install -e .
```

## Library Usage
This library is designed for performing arithmetic operations using language models. The primary usage involves constructing a formula with multiple `Operator` objects and encapsulating it in the `ModelArithmetic` class. This class integrates seamlessly with the transformers library, resembling the usage of a `PreTrainedModel`. The preferred method for working with this library is through the `generate_text` function. Examples can be found in the `scr/examples.py` file.

### Key Components
#### Operators
The library provides a variety of operators, which can be employed in typical Python expressions like addition, multiplication, and using integers and floats. Unique operators such as `Max` (or `Union`) and `Min` (or `Intersection`) are also available, similar to Python's `max` and `min` functions but require importing from this library. All formulas adhere to the syntax detailed in the accompanying paper.

#### Specifically Notable Operators

1. `LLMPrompt` Operator:
   - Purpose: Implements a language model with a specific input prompt.
   - Arguments:
        - `model`: A `PreTrainedModel` or a model name string from the transformers library.
        - `prompt_template`: A function mapping a prompt and an input input string to an output. A typical example would be:
        ```python
        lambda prompt, input: f"###Instruction: {prompt} \n ###Input: {input} \n ###Output:"
        ```
        - `prompt_string`: Replaces the prompt in `prompt_template`.
        - `speculative_factor`: A factor for speculative sampling (optional).
2. `Classifier` Operator:
   - Purpose: Implements a classifier.
   - Arguments:
      - `model`: Similar to `LLMPrompt`.
      - `formula`: Operator for computing top k likely tokens. `Classifier` will only compute probabilities on the most likely tokens from this formula at this step, and use a simple approximation for all other tokens. Typically, this formula is set equal to the formula that is being used for inference without the `Classifier` itself.
      - `n_runs_per_sample`: The "k" value for top tokens.
      - `batch_size`: Batch size for processing; if None, the same batch size as supplied to the generate function is used.
      - `prompt_template`: Similar to `LLMPrompt`. Typically an empty lambda for classifiers.
      - `minimize`: Boolean indicating whether to minimize the classifier output.

3. `Superseded` Operator:

   - Purpose: Implements speculative sampling as detailed in our paper.
   - Usage: `Superseded(m, M)` where `m` and `M` are language models (`Operators`).

#### Other Operators

Operators like `TopPTopK`, `ClassifierStrength`, `HardConstraint`, and `Indicator` are also available, but the above-mentioned operators cover most use cases.

##### Model Arithmetic Class

The `ModelArithmetic` class offers various parameters for generating formulas:
- `formula`: The arithmetic formula.
- `default_model`: A default model for all `LLMPrompt` instances, if applicable.

Functionality:
- Saving/Loading: Use `from_pretrained` and `to_pretrained` for model persistence.
- Generation: Use `generate_text` for generating text. The main arguments are:
    - `sentences`: a list of strings, the input sentences
    - Well-known arguments like `batch_size`, `temperature`, `top_p`, `top_k`, and `max_length`
- The `generate` function offers similar functionality as `generate_text` but starts with `input_ids`, a list of tokenized input ids.

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

You also need to download all datasets and put them in the `data/datasets` folder:
- [Alpaca Data](https://github.com/tloen/alpaca-lora/blob/main/alpaca_data.json)
- [Jigsaw Toxicity Dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) (the `all_data.csv` file should be downloaded and extracted in the `data/datasets` folder)
- [Politically Incorrect 4chan Messages](https://zenodo.org/record/3606810) (file should be unzipped and placed in the top level of the `data/datasets` folder)
- [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

API Keys for both the [PERSPECTIVE API](https://perspectiveapi.com/) and [OpenAI](https://openai.com/) need to be available in the environment variables. Alternatively, they can be placed in the file `src/.env` as

```sh
PERSPECTIVE_API_KEY="[YOUR API KEY]"
OPENAI_API_KEY="[YOUR API KEY]"
```

Finally, you can reproduce the results using

```sh
bash scripts/main.sh
```
This will preprocess the data, finetune a classifier for toxicity & sentiment, and reproduce the results from all sections of our paper. Results in CSV-format can then be found in the `eval/processed` and our figures in `eval/plots`.

We note that part of our preprocessing code got lost, specifically for preparing the dataset that is used for finetuning the toxicity classifier. Running the code in this setting might therefore result in slightly different numbers when they involve the finetuned classifier. However, you can also adjust the `scripts/main.sh` such that the first line is

```sh
python scripts/preprocess.py --reproduction
```

This provides a manual fix that selects the exact samples in the exact order we use. 