# DDPO

Official repository for the AAAI 2026 paper **Dynamic Deep Prompt Optimization for Defending Against Jailbreak Attacks on LLMs**.

This repository contains the original notebook demos used for the paper and a root-level Python runner for researchers who want to test DDPO on their own models and data.

=======
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

## Repository layout

```text
Data/
Deepseek/
Llama2/
Llama3/
Openchat/
Vicuna/
Results Summary/
README.md
```

<<<<<<< HEAD
Each model directory contains notebook demos. The `Data/` directory contains the example input files used by the notebooks and by the runner.

## Files to add at the repository root

Add these files at the repository root:

```text
run_ddpo.py
soft_templates.py
requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## What `run_ddpo.py` does

This script runs the full DDPO pipeline in one command:

1. load model and data
2. generate base-model training targets
3. select the best separation layer automatically
4. train DDPO
5. run jailbreak evaluation
6. optionally run MMLU evaluation
7. save outputs

The script prints each stage as it runs, shows progress bars, saves the selected layer automatically, and removes the need to edit multiple notebook cells.

## Current support scope

The root runner is intentionally aligned with the paper notebooks and currently assumes a **Llama-like Hugging Face causal LM** that exposes:

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
- the runner now raises a clear error if you try to pass a non-empty `--system_prompt` together with `--mode sys_prompt`

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

## Run with custom data paths

If your files are in different locations, run:

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

=======
>>>>>>> eeb32c83b5fb3e91de6384c7d8fe119c3ba8c5df
## Data format
File paths are relative by default. If your local layout is different, update the paths inside the notebooks.

Use the same file structure and columns as the provided example data.

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
- `attack`: attack name or category for reporting

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

If you do not add the template, the script will stop with a clear error telling you to define the model template in `soft_templates.py`.

## Running on new data

To test DDPO on a new dataset, keep the same file structure and column names.

<<<<<<< HEAD
Minimum requirements:

- harmful training file: `prompt`
- benign training file: `prompt`
- harmful test file: `prompt,attack`
- benign test file: `prompt`
- optional MMLU file: same JSON structure as `MMLU_data.json`

You can replace the provided files or point the runner to new ones with the path arguments shown above.

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

## Notebook demos

The notebooks in each model directory remain in the repository as demos.

Current notebook layout:

```text
<Model>/DDPO/
├── DDPO Optimization.ipynb
├── DDPO Testing.ipynb
└── DDPO MMLU Testing.ipynb
```

The notebooks are useful if you want to inspect the original paper workflow step by step.

### If you use the notebooks instead of `run_ddpo.py`

You will need to adjust paths manually inside the notebook cells.

At minimum:

- in `DDPO Optimization.ipynb`, set `model_name`
- in `DDPO Testing.ipynb`, set `model_name`
- in `DDPO Testing.ipynb`, set `generator_path`
- in `DDPO MMLU Testing.ipynb`, set `model_name`
- in `DDPO MMLU Testing.ipynb`, set `generator_path`
- make sure the CSV and JSON data files are reachable from the notebook working directory
- in the testing notebooks, set `target_layer_to_stop_at` to the layer selected during optimization

For other researchers, `run_ddpo.py` is the recommended entry point.
=======
1. Replace `train_bad.csv` and `train_clean.csv` with your training prompts.
2. Replace `test_bad.csv` and `test_clean.csv` with your evaluation prompts.
3. Keep the required columns:
   - training: `prompt`
   - harmful test: `prompt`, `attack`
   - benign test: `prompt`
4. If you want MMLU-style evaluation on a different benchmark, convert it to the same JSON structure used by `MMLU_data.json`.
5. Update output file names in the testing notebooks if you want dataset-specific result files.
