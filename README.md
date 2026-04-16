# DDPO

Official repository for the AAAI 2026 paper **"Dynamic Deep Prompt Optimization for Defending Against Jailbreak Attacks on LLMs"**.

This repository contains Jupyter notebooks for reproducing the DDPO experiments reported in the paper, along with the data files used by the notebooks. The code is organized mainly by **target model** and **defense method**.

## Repository layout

```text
ddpo/
├── Data/
│   ├── train_bad.csv
│   ├── train_clean.csv
│   ├── test_bad.csv
│   ├── test_clean.csv
│   └── MMLU_data.json
├── Llama3/
│   ├── DDPO/
│   ├── DRO/
│   ├── PAT/
│   ├── RPO/
│   ├── No Defense/
│   └── Ablation Study/
├── Llama2/
├── Deepseek/
├── Vicuna/
├── Openchat/
└── Results Summary/
```

At the root level, the repository contains one folder per model (for example `Llama3`, `Llama2`, `Deepseek`, `Vicuna`, and `Openchat`), plus a shared `Data/` folder and a `Results Summary/` folder.

## What the DDPO notebooks do

For each model, the `DDPO/` folder typically contains three notebooks:

- `DDPO Optimization.ipynb`  
  Trains the DDPO embedding generator, automatically finds the best insertion layer, and saves the learned weights to `embed_generator.pt`.

- `DDPO Testing.ipynb`  
  Runs inference on jailbreak and benign evaluation prompts and saves generations to `evaluation_<model>.csv`.

- `DDPO MMLU Testing.ipynb`  
  Runs MMLU evaluation and saves generations to `evaluation_<model>_MMLU.csv`.

The notebooks load the base LLM with:

- `BitsAndBytesConfig(load_in_8bit=True)`
- `device_map="auto"`
- `torch_dtype=torch.bfloat16`

So in practice you should plan to run on a CUDA GPU with `bitsandbytes` available.

## Requirements

A minimal environment should include:

```bash
pip install torch transformers accelerate bitsandbytes pandas scikit-learn tqdm datasets jupyter ipywidgets
```

You also need local access to the target chat model weights (for example a local Hugging Face model directory).

## Data setup

The notebooks expect the following files:

- `train_bad.csv`
- `train_clean.csv`
- `test_bad.csv`
- `test_clean.csv`
- `MMLU_data.json`

By default, the notebooks load these files using plain relative paths such as:

```python
pd.read_csv("train_bad.csv")
pd.read_csv("test_bad.csv")
open("MMLU_data.json")
```

That means you should either:

1. copy the needed data files into the same working directory as the notebook you are running, or  
2. update the file paths in the notebook to point to `../../Data/...` or wherever your local copy lives.

## Quick start

### 1) Choose a model folder

Pick the target model you want to reproduce, for example:

```text
Llama3/DDPO/
```

### 2) Point the notebook to your local model

Set `model_name` to your local model path, for example:

```python
model_name = "./models/Meta-Llama-3-8B-Instruct"
```

### 3) Run `DDPO Optimization.ipynb`

This notebook does three important things:

1. loads the train split (`train_bad.csv` and `train_clean.csv`)
2. estimates the best separation layer by comparing harmful vs. clean hidden states
3. trains the `EmbedGenerator` MLP and saves the result as:

```python
save_path = "embed_generator.pt"
```

The optimization notebook also lets you choose the insertion mode:

```python
mode = "sys_prompt"   # or "prefix" / "suffix"
```

The submitted experiments use the DDPO mechanism implemented in this notebook; for the included model notebooks, the default setting is `sys_prompt`.

### 4) Copy the selected layer into the testing notebooks

After `DDPO Optimization.ipynb` finishes, note the printed value:

```python
print(f"Best separation layer: {best_separation_layer}")
```

Then open:

- `DDPO Testing.ipynb`
- `DDPO MMLU Testing.ipynb`

and set:

```python
target_layer_to_stop_at = <best layer>
start_layer_for_forward_from = target_layer_to_stop_at + 1
```

The testing notebooks do **not** recompute the best layer automatically; they expect this value to be filled in ahead of time. In the provided Llama 3 notebook, this value is already set to `23`.

### 5) Run evaluation notebooks

Running `DDPO Testing.ipynb` writes a CSV like:

```python
evaluation_llama3.csv
```

Running `DDPO MMLU Testing.ipynb` writes a CSV like:

```python
evaluation_llama3_MMLU.csv
```

The final cells in those notebooks compute quick summary metrics from the saved CSVs.

---

## Adding a new model

To use DDPO with a new chat model, you will usually need to update **four** things.

### A. Set the local model path

Update:

```python
model_name = "./models/<your-model>"
```

### B. Add the chat template

The notebooks hard-code prompt formatting by checking substrings in `model_name`.  
If your model is new, add a case in:

- `format_prompt(...)` in `DDPO Optimization.ipynb`
- `embed_format(...)` in `DDPO Optimization.ipynb`
- `embed_format(...)` in `DDPO Testing.ipynb`
- `embed_format(...)` in `DDPO MMLU Testing.ipynb`

If you do not add it, the notebook will print:

- `"A chat template for this model is not defined. Add the chat template in the format_prompt function."`
- `"A chat template for this model is not defined. Add the chat template in the embed_prompt function."`

The existing notebooks already include templates for:

- Vicuna
- Llama 3
- Llama 2 / Mistral-style `[INST]` format
- DeepSeek
- OpenChat

When adding a new model, make sure the template is consistent in **both** the plain text prompt builder and the embedding-format builder.

### C. Verify the model internals match the notebook assumptions

The DDPO notebooks assume the model exposes internals in this style:

- `model.model.layers`
- `model.model.norm`
- `model.model.embed_tokens`
- `model.lm_head`

This matches the included models, but some Hugging Face models use slightly different module names.  
If your model differs, you will need to adapt:

- `forward_to(...)`
- `forward_from(...)`
- `get_embeds(...)`
- any direct call to `model.model.embed_tokens(...)`

### D. Re-run layer selection for the new model

Do not reuse the layer chosen for a different model.  
Run `DDPO Optimization.ipynb` for the new model and use the printed `best_separation_layer`.

---

## Notes on how the implementation works

A few implementation details matter when reproducing results:

- The base LLM is frozen (`param.requires_grad = False` for all model parameters).
- Only the small `EmbedGenerator` MLP is trained.
- The current implementation uses `num_prompt_tokens = 1` by default.
- The placeholder prompt embedding is inserted as a zero vector and then replaced by the generated embedding after the partial forward pass.
- The tokenizer is set to left padding:
  ```python
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"
  ```

These details are important because the manual `forward_to(...)` / `forward_from(...)` logic depends on padding and token positions being handled consistently.

## Common gotchas

### File not found errors
The most common issue is that the notebook expects `train_bad.csv`, `test_bad.csv`, etc. in the current working directory.  
Fix by copying files locally or updating the paths.

### New model fails immediately
Most likely the chat template is missing. Add a new case in `format_prompt(...)` and `embed_format(...)`.

### New model loads but generation breaks
Check whether your model uses different internal attribute names than:

```python
model.model.layers
model.model.norm
model.model.embed_tokens
model.lm_head
```

### Out-of-memory issues
Reduce:

- `batch_size`
- `max_new_tokens`

You can also switch to a smaller model or a GPU with more memory.

### Wrong evaluation layer
Make sure `target_layer_to_stop_at` in the testing notebooks matches the layer found during optimization for the same model.

---

## Example workflow

For Llama 3, a typical workflow is:

1. set `model_name = "./models/Meta-Llama-3-8B-Instruct"`
2. make sure the CSV/JSON data files are accessible
3. run `Llama3/DDPO/DDPO Optimization.ipynb`
4. keep the saved `embed_generator.pt`
5. confirm `target_layer_to_stop_at`
6. run `Llama3/DDPO/DDPO Testing.ipynb`
7. run `Llama3/DDPO/DDPO MMLU Testing.ipynb`

---

## Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{obidov2026ddpo,
  title={Dynamic Deep Prompt Optimization for Defending Against Jailbreak Attacks on LLMs},
  author={Obidov, Doniyorkhon and Yu, Honggang and Guo, Xiaolong and Yang, Kaichen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

MIT
