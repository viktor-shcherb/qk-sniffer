# qk-sniffer

`qk-sniffer` instruments Hugging Face transformer models so each attention layer can stream sampled key/query vectors into Parquet shards that mirror Hugging Face dataset repos. It ships with Gemma 3/Llama/Qwen 3 hooks, a deterministic sampler, and a CLI (`sniff-qk`) that can pull/push full datasets from the Hub.

## Requirements & Installation
- Python 3.9+, recent PyTorch build matching your hardware.
- (Optional) create and activate a virtualenv.
- Install in editable mode so `models/` is importable:
  ```bash
  pip install --upgrade pip
  pip install -e .
  ```
- Put secrets (e.g., `HF_TOKEN`) in a `.env`; `python-dotenv` loads them automatically.

## Quick Start
1. **Instrument the model (if needed).** Copy the relevant `transformers` module into `models/<name>/modeling_<name>.py`, import `get_active_sniffer`/`compute_positions`, and call `sniffer.capture(...)` inside the attention block. Gemma 3, Llama, and Qwen 3 are already wired up.
2. **Create a config** (adapt the sample below):
   ```yaml
   dataset:
     path: viktoroo/example
     split: train
     text_column: text
   model:
     name: google/gemma-2-2b
     dtype: float16
   tokenizer:
     name: google/gemma-2-2b
     max_length: 4096
   inference:
     batch_size: 2
     autocast_dtype: float16
  capture:
    capture_queries: true
    capture_keys: true
    min_bucket_size: 128
    sampler:
      type: log_uniform       # log_uniform | uniform | all
      base_rate: 1.0
    capture_pre_rope: false
    capture_token_strings: false
   output:
     data_root: data/sniffed-qk
     readme_path: README.md
     hf_repo_id: viktoroo/sniffed-qk
   ```
3. **Run the CLI**:
   ```bash
   sniff-qk --config configs/sample_sniff.yaml
   # or
   PYTHONPATH=. python sniff.py --config configs/sample_sniff.yaml
   ```
   The CLI patches local modeling files into `transformers`, downloads the latest dataset snapshot (if `hf_repo_id` is set), runs inference, writes captures directly into that synced copy, then uploads the result.

## Key Configuration Notes
- `dataset.*` maps directly to `datasets.load_dataset`. Set `max_samples` for dry runs. Use `streaming: true` to stream without downloading the full split; when streaming, `max_samples` stops after the first *N* examples.
- `model.*`/`tokenizer.*` feed `AutoModelForCausalLM` and `AutoTokenizer`. `device_map=auto` works well for multi-GPU.
- `capture.*`
  - `layers`, `heads` accept Python-style integer lists; omit to capture every head.
  - `sampler.type=all` captures every token (no subsampling) while still bucketing by `min_bucket_size` for reporting. To capture the first *N* tokens, set `tokenizer.max_length` to *N*.
  - `min_bucket_size` drives bucketing (and sampling for `log_uniform`/`uniform`): `log_uniform` rounds it up to the next power of two and uses it as the minimum `2^i` bucket width (bucket IDs remain the exponent `i`, so `b{i}` → `[2^i, 2^{i+1})`). `uniform` and `all` store `floor(position / min_bucket_size)`.
  - `capture_pre_rope` captures Q/K before rotary position embedding is applied (default captures post-RoPE).
  - `capture_token_strings` stores an extra `token_str` column with the tokenizer string for each captured position.
- `sampler.type` controls the bucket definition automatically: `log_uniform` samples uniformly *within* each log bucket, `uniform` samples uniformly over every fixed-width bucket, and `all` disables subsampling while still reporting uniform buckets. All are deterministic per `(example, layer, head, kind)`.
- `output.*`
  - `data_root` is both your working directory and (optionally) the local clone of `hf_repo_id`. Each run pulls the repo into `data_root`, records captures there immediately, and pushes it back once inference finishes.
  - `readme_path` may be relative (inside `data_root`) or absolute. It is rewritten in place so the Hub copy stays in sync with the local metadata before uploading.

## Capture Output & Dataset Layout
- Each `(model split, layer, head, vector_kind)` writes to `data/<sanitized_model>/<lXXhYY{q|k}>/data.parquet`. Splits sanitize `[\W]` → `_` (e.g., `meta/llama3-8b` → `meta_llama3_8b`).
- `DatasetSaver` deduplicates `(example_id, position)` pairs per config by seeding an in-memory cache from existing Parquet shards, so reruns append only new tokens.
- The saver rewrites `output.readme_path` with Hugging Face front matter listing every config plus aggregate helpers such as `all`, `layer00`, `all_q`, etc., so `datasets.load_dataset` can point at the folder or Hub repo immediately.
- Columns:

  | Column           | Description                                                                                                                                                                                                                                                                                                                                    |
  |------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| | `bucket`         | Identifier of the bucket the token fell into. `log_uniform` stores the exponent `i` (clamped upward so the first bucket spans ≥ `min_bucket_size`), meaning bucket `b{i}` maps to `[2^i, 2^{i+1})`. `uniform` and `all` store `floor(position / min_bucket_size)` so every bucket covers exactly `min_bucket_size` tokens (last bucket may be smaller). |
  | `example_id`     | Batch ID from `set_active_example_ids` (or the implicit index).                                                                                                                                                                                                                                                                                |
  | `position`       | Token index after any cache offset.                                                                                                                                                                                                                                                                                                            |
  | `token_str`      | Optional string representation of the token at this position (if enabled).                                                                                                                                                                                                                                                                    |
  | `vector`         | Float32 list representing the captured query/key vector.                                                                                                                                                                                                                                                                                       |
  | `sliding_window` | Sliding-window size for local attention layers (`null` for full causal).                                                                                                                                                                                                                                                                       |

Loading example:
```python
from datasets import load_dataset

queries = load_dataset("viktoroo/sniffed-qk", "l00h00q", split="meta_llama3_8b")
layer0_all_heads = load_dataset("viktoroo/sniffed-qk", "layer00", split="meta_llama3_8b")
```

## Extending to New Models
1. Copy the upstream `transformers` module into `models/<model_name>/modeling_<model_name>.py`.
2. Import `compute_positions` and `get_active_sniffer` and call `sniffer.capture` inside the attention block (pass layer/head indices plus optional sliding-window size).
3. Run `pytest tests/test_models.py -k patch_modeling_modules` (or your custom test) to ensure the aliasing still works.
4. Include the new `model.name` in your YAML and run the CLI.

## Testing & Troubleshooting
- Run `pytest` (full suite) or scoped commands like `pytest tests/test_sniffer.py -k sampler`.
- If `transformers.models.<name>` fails to import, ensure `pip install -e .` has been run and `patch_modeling_modules()` executes before loading the model.
- If Hub pushes fail, verify you set `HF_TOKEN` (write access) and that `hf_repo_id` exists.
- Raise `capture.sampler.base_rate` if captures are too sparse; decrease `min_bucket_size` (power-of-two rounded) to bias toward early tokens.
