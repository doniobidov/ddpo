# DDPO

Official repository for the AAAI 2026 paper **Dynamic Deep Prompt Optimization for Defending Against Jailbreak Attacks on LLMs**.

This repository contains the notebook demos and a Python runner to test DDPO on custom models and data.

The repository is organized by model. Each model directory contains notebooks for DDPO training and evaluation, and the `Data/` directory contains the input files.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{obidov2026ddpo,
  title={Dynamic Deep Prompt Optimization for Defending Against Jailbreak Attacks on LLMs},
  author={Obidov, Doniyorkhon and Yu, Honggang and Guo, Xiaolong and Yang, Kaichen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={42},
  pages={35742--35750},
  year={2026},
  doi={10.1609/aaai.v40i42.40887}
}
```

## Install

```bash
pip install -r requirements.txt
```

## What `run_ddpo.py` does

This script runs the full DDPO pipeline in one command:

1. load model and data
2. generate model training targets
3. select the best separation layer automatically
4. train DDPO
5. run jailbreak evaluation
6. optionally run MMLU evaluation
7. save outputs

## Current support scope

The code assumes a **Hugging Face causal LM** that exposes:

```text
model.model.embed_tokens
model.model.layers
model.model.norm
model.lm_head
```

This works for the model families already used in this repository. If your model does not match this structure, update these functions in `run_ddpo.py`:

- `resolve_llama_like_handles()`
- `forward_to()`
- `forward_from()`

## Important note about `--mode sys_prompt`

In `--mode sys_prompt`, the template's `{system_prompt}` slot is reserved for the **learned DDPO soft tokens**.

That means:

- leave `--system_prompt` empty when `--mode sys_prompt`
- if you want to use a **fixed textual system prompt**, use `--mode prefix` or `--mode suffix`
- the runner raises an error if you try to pass a non-empty `--system_prompt` together with `--mode sys_prompt`

## Before running

Before your first run, adjust these arguments:

1. `--model_name`
   - set this to your local model path or Hugging Face model name
2. `--data_dir`
   - set this if your data files are all in one folder
3. `--output_dir`
   - set this to the folder where you want outputs saved

If your files are not all inside one data directory, use the individual path arguments:

- `--train_bad_path`
- `--train_clean_path`
- `--test_bad_path`
- `--test_clean_path`
- `--mmlu_path`

## Standard run

If your model is at `./models/Meta-Llama-3-8B-Instruct` and your data is in `Data/`, run:

```bash
python run_ddpo.py \
  --model_name ./models/Meta-Llama-3-8B-Instruct \
  --data_dir Data \
  --output_dir outputs/llama3
```

This uses the default DDPO setup with `--mode sys_prompt`, which means the system-role slot is used for learned soft tokens and `--system_prompt` should stay empty.

## Run with a fixed textual system prompt

If you want a fixed system instruction plus DDPO in prefix or suffix mode, do this instead:

```bash
python run_ddpo.py \
  --model_name ./models/Meta-Llama-3-8B-Instruct \
  --data_dir Data \
  --output_dir outputs/llama3_prefix \
  --mode prefix \
  --system_prompt "You are a helpful assistant. Follow safety policies carefully."
```

## Run with custom data

To run with custom data, structure your data similarly to the provided examples and execute:

```bash
python run_ddpo.py \
  --model_name ./models/Meta-Llama-3-8B-Instruct \
  --train_bad_path /path/to/train_bad.csv \
  --train_clean_path /path/to/train_clean.csv \
  --test_bad_path /path/to/test_bad.csv \
  --test_clean_path /path/to/test_clean.csv \
  --mmlu_path /path/to/MMLU_data.json \
  --output_dir outputs/custom_run
```

## Output files

Each run saves:

```text
output_dir/
├── embed_generator.pt
├── evaluation_jailbreak.csv
├── evaluation_mmlu.csv                # only if MMLU is not skipped
├── layer_separation_scores.csv
├── metrics.json
├── run_config.json
├── selected_layer.json
├── training_history.csv
├── training_target_metadata.json
├── training_targets_ddpo.csv
└── training_targets_raw.csv
```

## Data format

### `train_bad.csv`
Harmful prompts used for DDPO training.

Required columns:

```text
prompt
```

### `train_clean.csv`
Benign prompts used for DDPO training.

Required columns:

```text
prompt
```

### `test_bad.csv`
Harmful evaluation prompts.

Required columns:

```text
prompt,attack
```

- `prompt`: the harmful input
- `attack`: attack name

### `test_clean.csv`
Benign evaluation prompts.

Required columns:

```text
prompt
```

### `MMLU_data.json`
Used for optional MMLU evaluation.

Expected structure:

```python
[
  [updated_input, question, subject, choices, answer],
  ...
]
```

## Running on a new model

For models that match one of the built-in template families, in many cases you only need to change `--model_name`.

Built-in template families in `soft_templates.py`:

- `llama3`
- `llama2`
- `mistral`
- `vicuna`
- `deepseek`
- `openchat`

### If your new model matches one of those formats

Use the matching family explicitly if auto-detection is not enough:

```bash
python run_ddpo.py \
  --model_name /path/to/your/model \
  --template_family llama3 \
  --data_dir Data \
  --output_dir outputs/new_model
```

### If your new model does not match one of those formats

Edit `soft_templates.py`.

1. Add a new template entry to `TEMPLATES`
2. Add a rule in `infer_template_family()` if you want auto-detection by model name
3. Rerun the script

If you do not add the template, the script will stop with an error.

## Useful options

```bash
--mode sys_prompt
--num_prompt_tokens 1
--epochs 40
--patience 3
--learning_rate 1e-3
--train_batch_size 8
--eval_batch_size 64
--layer_analysis_batch_size 8
--warmup_max_new_tokens 16
--eval_max_new_tokens 32
--mmlu_max_new_tokens 64
--skip_mmlu
```
