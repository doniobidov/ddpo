# ddpo

Official repository for the AAAI 2026 paper **Dynamic Deep Prompt Optimization for Defending Against Jailbreak Attacks on LLMs**.

The repository is organized by model. Each model directory contains notebooks for DDPO training and evaluation, and the `Data/` directory contains the shared input files.

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

- `Data/`
  - `train_bad.csv`
  - `train_clean.csv`
  - `test_bad.csv`
  - `test_clean.csv`
  - `MMLU_data.json`
- `<Model>/DDPO/`
  - `DDPO Optimization.ipynb`
  - `DDPO Testing.ipynb`
  - `DDPO MMLU Testing.ipynb`

## Requirements

The notebooks import:

- Python 3.9+
- `torch`
- `transformers`
- `bitsandbytes`
- `pandas`
- `numpy`
- `scikit-learn`
- `tqdm`
- Jupyter Notebook or JupyterLab

Example install:

```bash
pip install torch transformers bitsandbytes pandas numpy scikit-learn tqdm jupyter
```

Notes:
- The notebooks load the model with `load_in_8bit=True` and `torch_dtype=torch.bfloat16`.
- A CUDA GPU setup is recommended.
- File paths are relative by default. If your local layout is different, update the paths inside the notebooks.

## Data format

### Training data

`train_bad.csv` and `train_clean.csv` should contain at least:

```text
prompt
```

The provided files also include an `attack` column, but the optimization notebook reads only the `prompt` column for training.

### Test data

`test_bad.csv` should contain:

```text
prompt,attack
```

`test_clean.csv` should contain at least:

```text
prompt
```

If `test_clean.csv` also includes `attack`, it will be ignored by the testing notebook.

### MMLU data

`MMLU_data.json` is read as a list where each item has this structure:

```python
[
  updated_input,
  question,
  subject,
  choices,
  answer
]
```

## How to run DDPO

1. Open the notebook for your target model under `<Model>/DDPO/`.
2. Set `model_name` to your local path or Hugging Face model path.
3. Make sure the data files are in the notebook working directory, or update the file paths.
4. Run `DDPO Optimization.ipynb` from top to bottom.
5. The notebook will:
   - load the base model,
   - generate initial outputs for the training prompts,
   - estimate the best separation layer,
   - train the embedding generator,
   - save `embed_generator.pt`.
6. Copy the printed `best_separation_layer` into:
   - `target_layer_to_stop_at` in `DDPO Testing.ipynb`
   - `target_layer_to_stop_at` in `DDPO MMLU Testing.ipynb`
7. Set `generator_path = "embed_generator.pt"` in the testing notebooks if needed.
8. Run:
   - `DDPO Testing.ipynb` to create the jailbreak evaluation CSV
   - `DDPO MMLU Testing.ipynb` to create the MMLU evaluation CSV

## Outputs

In the Llama 3 DDPO notebooks, the default outputs are:

- `embed_generator.pt`
- `evaluation_llama3.csv`
- `evaluation_llama3_MMLU.csv`

For other models, update the output file names if you do not want Llama 3 names in your saved results.

## Running on a new model

To use DDPO with a new model:

1. Set `model_name` in all three notebooks.
2. Add the model's chat template:
   - in `format_prompt(...)` in `DDPO Optimization.ipynb`
   - in `embed_format(...)` in `DDPO Optimization.ipynb`
   - in `embed_format(...)` in `DDPO Testing.ipynb`
   - in `embed_format(...)` in `DDPO MMLU Testing.ipynb`
3. Make sure the model exposes the internals used by the notebooks:
   - `model.model.embed_tokens`
   - `model.model.layers`
   - `model.model.norm`
   - `model.lm_head`
4. Check tokenizer setup:
   - set `tokenizer.pad_token`
   - keep `tokenizer.padding_side = "left"` unless you intentionally adapt the padding logic
5. Run the optimization notebook first to discover the new `best_separation_layer`.
6. Copy that layer index into both testing notebooks before evaluation.

If your model uses a different internal structure, you will need to adapt `forward_to(...)`, `forward_from(...)`, and the embedding lookup code.

## Running on new data

To use a new jailbreak or benign dataset:

1. Replace `train_bad.csv` and `train_clean.csv` with your training prompts.
2. Replace `test_bad.csv` and `test_clean.csv` with your evaluation prompts.
3. Keep the required columns:
   - training: `prompt`
   - harmful test: `prompt`, `attack`
   - benign test: `prompt`
4. If you want MMLU-style evaluation on a different benchmark, convert it to the same JSON structure used by `MMLU_data.json`.
5. Update output file names in the testing notebooks if you want dataset-specific result files.

## Important code details

- The optimization notebook saves the trained generator to `embed_generator.pt`.
- The testing notebooks load that checkpoint with `generator_path`.
- The layer used during testing is not inferred automatically there; you must copy the selected layer from the optimization run.
- The notebooks are written around direct access to intermediate hidden states, so they work best with models whose Hugging Face implementation exposes layer modules in a Llama-style layout.
