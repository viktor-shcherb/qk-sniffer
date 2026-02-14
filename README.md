# qk-sniffer

Capture sampled Q/K attention vectors from Hugging Face transformer models into Parquet datasets, one branch per model. Hooks are provided for Gemma 3, GLM, GLM4, Llama, Mllama, Ministral, Qwen 2, and Qwen 3.

## Setup

```bash
pip install -e .
```

Set `HF_TOKEN` in a `.env` file for Hub uploads.

## Usage

```bash
sniff-qk --config configs/sample.yaml
```

See [`configs/sample.yaml`](configs/sample.yaml) for a commented example. Configs in [`configs/attention-plasticity/`](configs/attention-plasticity/) are the production configs used for actual runs.

## How It Works

1. `patch_modeling_modules()` aliases local modeling files under `models/` into `transformers.models.*`, activating the capture hooks.
2. The model backbone is called directly (skipping the lm_head) with `use_cache=False`. The `attention_mask` is dropped to avoid O(n^2) 4D mask expansion — causal masking with right-side padding is sufficient.
3. Inside each attention layer, the hook calls `sniffer.capture()` with the Q/K states and position ids.
4. The sampler selects positions on CPU (no GPU sync), gathers vectors for all active heads at once as a dense `(n_heads, K, dim)` tensor, and transfers to CPU via a single async copy.
5. All captures accumulate in RAM. Parquet files are written only after inference completes, then optionally pushed to the Hub.

## Output Format

Each `(layer, head, kind)` produces `<data_root>/l{LL}h{HH}{q|k}/data.parquet`:

| Column | Type | Description |
|---|---|---|
| `bucket` | int32 | Position bucket id (log2 exponent for `log_uniform`, `floor(pos / bucket_size)` for `uniform`/`all`) |
| `example_id` | int32 | Dataset example index |
| `position` | int32 | Token position in the sequence |
| `vector` | list\<float32\> | The captured Q or K vector |
| `sliding_window` | int32 / null | Window size for sliding-attention layers, null for full attention |
| `token_str` | string / null | Token string (when `capture_token_strings: true`) |

Each model is published to its own branch (`hf_branch`). To load:

```python
from datasets import load_dataset

ds = load_dataset(
    "viktoroo/test-sniffer-qk",
    "l00h00q",
    revision="smollm2-135m-longbench-pro-128k-plus",
)
```

## Project Structure

```
sniff.py              CLI entry point, config loading, inference loop
sniffer/
  core.py             Sniffer class — capture hooks, in-memory accumulation, flush
  samplers.py         Position samplers (log_uniform, uniform, all)
saver/
  dataset.py          DatasetSaver — Parquet writing, metadata tracking
  readme.py           Auto-generated HF dataset card
models/
  <arch>/             Modified modeling files with sniffer.capture() calls
configs/
  sample.yaml         Minimal example config
  attention-plasticity/  Production configs
```

## Adding a New Model

1. Copy the upstream `transformers` modeling file into `models/<arch>/modeling_<arch>.py`.
2. In the attention forward, import `get_active_sniffer` and call `sniffer.capture(layer_idx, query_states, key_states, positions, sliding_window)`.
3. Verify with `pytest tests/ -k patch_modeling_modules`.

## Configuration Reference

Most fields are self-explanatory from the [sample config](configs/sample.yaml). Non-obvious options:

- **`capture.head_sampling`**: Randomly samples `count` query heads across all layers (deterministic via `seed`). Key heads are included automatically for GQA models.
- **`capture.full_attention_only`**: When `true`, skips sliding-window attention layers entirely.
- **`capture.sampler.type`**: `log_uniform` buckets by powers of two, `uniform` by fixed-width bins, `all` captures every position.
- **`capture.capture_pre_rope`**: Capture Q/K before rotary embedding (default: post-RoPE).
- **`inference.debug_logging`**: Per-batch timing breakdown of hooks, flush, and forward pass.
- **`output.hf_branch`**: One branch per model keeps the main branch clean.
