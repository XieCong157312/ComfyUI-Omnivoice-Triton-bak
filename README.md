# ComfyUI-Omnivoice-Triton

**Fast OmniVoice TTS for ComfyUI** — task-specific nodes wrapping [omnivoice-triton](https://github.com/newgrit1004/omnivoice-triton), accelerated with Triton kernel fusion and CUDA Graph capture. **3.48× end-to-end faster** than the unoptimised baseline at ~2 GB peak VRAM.

[![OmniVoice Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-k2--fsa%2FOmniVoice-blue)](https://huggingface.co/k2-fsa/OmniVoice)
[![arXiv](https://img.shields.io/badge/arXiv-2604.00688-b31b1b)](https://arxiv.org/abs/2604.00688)
[![Upstream](https://img.shields.io/badge/GitHub-omnivoice--triton-black)](https://github.com/newgrit1004/omnivoice-triton)
[![Sibling](https://img.shields.io/badge/Sibling-ComfyUI--Qwen3--TTS--Triton-black)](https://github.com/newgrit1004/ComfyUI-Qwen3-TTS-Triton)
[![Sibling](https://img.shields.io/badge/Sibling-ComfyUI--ZImage--Triton-black)](https://github.com/newgrit1004/ComfyUI-ZImage-Triton)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

## Features

- **3.48× end-to-end speedup** — `hybrid` (Triton + CUDA Graph) vs unoptimised `base` on a 57-char Korean sentence (RTX 5090, fp16). See [`benchmark/BENCHMARK.md`](benchmark/BENCHMARK.md) for the full table.
- **4 runner modes** — `base`, `triton`, `faster`, `hybrid`. Pick the trade-off between warmup cost and per-run latency.
- **Three dedicated nodes only** — `OmnivoiceTTSAuto`, `OmnivoiceTTSVoiceClone`, and `OmnivoiceTTSVoiceDesign`, matching OmniVoice's Auto TTS / Voice Cloning / Voice Design tasks.
- **Voice-clone AUDIO handoff** — reference clips are downmixed, resampled to 24 kHz, and passed in memory so the node avoids upstream file-loader round-trips.
- **~2 GB peak VRAM** across all modes. CUDA Graph adds ~40 MB, nothing else does.
- **Module-level runner cache** keyed by `(runner_mode, model_id, dtype, device)` — no redundant `load_model()` calls between runs with the same config.
- **Shape-keyed CUDA Graph cache** for `faster` / `hybrid` — capture once per input shape, replay on repeats.
- **Node-local vendor isolation** — optional `vendor/` installs are loaded before the shared ComfyUI environment to avoid cross-node dependency conflicts.
- **Torch-protective install flow** — `requirements.txt` is intentionally comments-only so ComfyUI Manager cannot install torch-sensitive packages before `install.py`.

## Installation

### Method 1: ComfyUI Manager (recommended)

Search for **"Omnivoice Triton TTS"** in ComfyUI Manager and click **Install**.

### Method 2: Manual install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/newgrit1004/ComfyUI-Omnivoice-Triton.git
cd ComfyUI-Omnivoice-Triton
python install.py          # runs from the ComfyUI venv
```

The `install.py` script installs `omnivoice-triton` and its transitive
dependencies **without touching your PyTorch install** (see next section).

`requirements.txt` is intentionally a no-op for ComfyUI Manager safety. Use
`python install.py` for dependency installation.

### Method 3: Vendor-isolated install (avoid cross-node dep conflicts)

```bash
python -m pip install --upgrade --no-deps \
  --target ComfyUI/custom_nodes/ComfyUI-Omnivoice-Triton/vendor \
  omnivoice==0.1.4 \
  omnivoice-triton==0.1.0 \
  "transformers>=5.3.0" \
  accelerate \
  huggingface-hub
```

The node prepends `vendor/` to `sys.path`, so these packages override the
shared ComfyUI environment only for this node.

### Why `--no-deps`?

`omnivoice-triton` depends on `omnivoice`, which in turn declares `torch>=2.4`
and `torchaudio>=2.4`. On some systems pip's resolver will happily install a
CPU-only `torch` wheel that satisfies this constraint, silently replacing
ComfyUI's CUDA-enabled build and breaking GPU acceleration.

`install.py` installs `omnivoice-triton` and `omnivoice` with `--no-deps` and
then pulls in the remaining torch-independent dependencies (`transformers`,
`triton`, `sageattention`, `soundfile`, `librosa`, `soxr`, etc.) explicitly. Your
ComfyUI `torch` is never modified.

If your PyTorch does get broken by another package, reinstall a CUDA wheel:

```bash
pip install --upgrade --force-reinstall torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

Or pick the matching index from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### Requirements

- **Python ≥ 3.12** (`omnivoice-triton` pins `requires-python>=3.12`)
- **PyTorch with CUDA 12.8+**, Blackwell / Ada / Hopper / Ampere GPU
- **ComfyUI launched with `--disable-cuda-malloc`** — required for `hybrid` mode. See *Running ComfyUI* below.

### Local checkout fallback (developer mode)

If you want to develop against a local `omnivoice-triton` checkout instead of
the PyPI wheel:

```bash
export COMFYUI_OMNIVOICE_TRITON_SRC=/path/to/omnivoice-triton
```

The env var can point at the repo root or its `src/` directory. The node
prepends that path to `sys.path` before importing `omnivoice_triton`.

The older sibling-repo pattern also still works: drop a symlink at `src/`
inside this repo:

```bash
ln -s /path/to/omnivoice-triton/src src
```

When `src/omnivoice_triton/` exists, `__init__.py` prepends it to `sys.path`
automatically, so `import omnivoice_triton` resolves to your local tree
instead of the PyPI install.

## Nodes

<details>
<summary><strong>Omnivoice TTS (Auto)</strong> — text → <code>AUDIO</code> (24 kHz mono)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | STRING, multiline | sample | Text to synthesize. |
| `runner_mode` | enum | `hybrid` | `base`, `triton`, `faster`, `hybrid`. See *Runner modes* below. |
| `num_step` | INT 4-64 | 32 | Iterative decoding steps. Lower = faster / lower quality. |
| `guidance_scale` | FLOAT 0-10 | 2.0 | Classifier-free guidance. Higher = more text-aligned. |
| `class_temperature` | FLOAT 0-2 | 0.0 | Token sampling temperature. 0 = greedy / deterministic. |
| `language` | enum | `auto` | `auto`, `ko`, `en`, `zh`. `auto` maps to `None` (model decides). |
| `model_id` | STRING | `k2-fsa/OmniVoice` | HuggingFace id or local path. |
| `dtype` | enum | `fp16` | `fp16`, `bf16`, `fp32`. |
| `device` | enum | `cuda` | `cuda`, `cpu`. |

**Output:** `audio` (AUDIO) — ComfyUI AUDIO dict `{"waveform": tensor[1, 1, T], "sample_rate": 24000}`.

</details>

<details>
<summary><strong>Omnivoice TTS (Voice Clone)</strong> — text + reference <code>AUDIO</code> → <code>AUDIO</code></summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | STRING, multiline | sample | Text to synthesize in the cloned voice. |
| `ref_audio` | AUDIO | — | Reference clip loaded with ComfyUI's `LoadAudio`. |
| `runner_mode` | enum | `hybrid` | Same runner modes as Auto. |
| `ref_text` | STRING, multiline | empty | Optional transcript for the reference clip. |
| `language` | enum | `auto` | `auto`, `ko`, `en`, `zh`. |
| `num_step` / `guidance_scale` / `class_temperature` | advanced | Auto defaults | Generation controls. |
| `model_id` / `dtype` / `device` | advanced | Auto defaults | Model/runtime controls. |

The node downmixes ComfyUI `AUDIO` to mono, resamples reference clips to the
OmniVoice 24 kHz rate with `soxr` when needed, and passes an in-memory
`(audio, sample_rate)` tuple to `runner.generate_voice_clone(...)`. This avoids
the upstream temporary-file / `librosa` loader path.

</details>

<details>
<summary><strong>Omnivoice TTS (Voice Design)</strong> — text + voice instruction → <code>AUDIO</code></summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | STRING, multiline | sample | Text to synthesize. |
| `instruct` | STRING, multiline | sample | Natural-language voice description. |
| `runner_mode` | enum | `hybrid` | Same runner modes as Auto. |
| `language` | enum | `auto` | `auto`, `ko`, `en`, `zh`. |
| `num_step` / `guidance_scale` / `class_temperature` | advanced | Auto defaults | Generation controls. |
| `model_id` / `dtype` / `device` | advanced | Auto defaults | Model/runtime controls. |

This calls `runner.generate_voice_design(...)`, so no reference audio is needed.
Use upstream-supported instruction items only, separated by comma + space; for
example: `female, young adult, high pitch`.

</details>

### Runner modes

| mode | what it does | typical latency (57-char text) |
|---|---|---|
| `base` | reference OmniVoice runner, no optimisation | 0.65 s |
| `triton` | fused Triton kernels (RMSNorm, SwiGLU, add+norm) | 0.63 s |
| `faster` | CUDA Graph capture, no Triton fusion | 0.26 s |
| `hybrid` | Triton fusion **and** CUDA Graph capture — **recommended** | **0.19 s** |

Short inputs like a 57-char sentence don't stress the attention path enough
for Triton fusion alone to help much (`triton` ≈ `base`). On longer text the
gap widens.

### Runner caching

The runner is cached at module scope, keyed by
`(runner_mode, model_id, dtype, device)`. A cache hit reuses the warm runner
(no `load_model()` cost). A key change evicts the previous runner via
`unload_model()` before building a new one, so VRAM never doubles during
mode swaps.

### CUDA Graph first-run cost (`faster` / `hybrid`)

`faster` and `hybrid` capture a CUDA Graph on the **first** inference for a
given input shape. Capture adds ~0.1-0.5 s to that first run. Subsequent runs
with the same shape replay the captured graph and are consistently fast.

If input text length changes significantly (producing a different token
sequence shape), a new graph is captured for that shape and cached for the
lifetime of the runner — no re-capture on repeat.

| Run | gen_s (hybrid) | Notes |
|-----|---------------|-------|
| 1st | ~0.48 s | includes CUDA Graph capture |
| 2nd+ | ~0.19 s | graph replay, stable |

## Running ComfyUI: `--disable-cuda-malloc` is required

```bash
python ComfyUI/main.py --listen 0.0.0.0 --port 8188 --disable-cuda-malloc
```

Without `--disable-cuda-malloc`, ComfyUI's default `cudaMallocAsync`
allocator conflicts with both `transformers`'s parallel shard loading and
CUDA Graph capture/teardown, causing fake OOMs and `hybrid`-mode crashes.

## Example workflows

Saved ComfyUI API workflows live in [`workflows/`](workflows/):

- [`tts_base.json`](workflows/tts_base.json)
- [`tts_triton.json`](workflows/tts_triton.json)
- [`tts_faster.json`](workflows/tts_faster.json)
- [`tts_hybrid.json`](workflows/tts_hybrid.json) (recommended)
- [`tts_voiceclone.json`](workflows/tts_voiceclone.json)
- [`tts_voice_design.json`](workflows/tts_voice_design.json)

All saved workflows use the task-specific node class names above.

KV-cache variants (`kv_cache*`) exist only on an unreleased local branch of
`omnivoice-triton` and are intentionally not shipped here — stick to the
four modes above on the current `omnivoice-triton==0.1.0` PyPI release.

## Benchmark

Latest numbers (RTX 5090, CUDA 12.8, PyTorch 2.11.0+cu128, Triton 3.6.0,
transformers 5.3.0, Python 3.12.3, WSL2):

| mode | gen_med | wall_med | peak VRAM | gen speedup | wall speedup |
|------|--------:|--------:|----------:|------------:|-------------:|
| base     | 0.650 s | 0.710 s | 2.01 GB | **1.00×** | **1.00×** |
| triton   | 0.630 s | 0.710 s | 2.01 GB | 1.03× | 1.00× |
| faster   | 0.260 s | 0.304 s | 2.04 GB | **2.50×** | **2.34×** |
| hybrid   | 0.190 s | 0.204 s | 2.05 GB | **3.42×** | **3.48×** |

Full methodology, caveats, and the 4-mode vs. text-length discussion are in
[`benchmark/BENCHMARK.md`](benchmark/BENCHMARK.md).

## Publishing to Comfy Registry

1. Bump `version` in `pyproject.toml`.
2. Verify `benchmark/BENCHMARK.md` reflects current numbers.
3. `git push origin main` and run `comfy node publish`.

## References

- **Wrapped package:** [omnivoice-triton](https://github.com/newgrit1004/omnivoice-triton)
- **Feature-rich OmniVoice node** (multi-speaker / longform tools): [Saganaki22/ComfyUI-OmniVoice-TTS](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS)
- **Sibling Triton-accelerated nodes:**
  - [ComfyUI-Qwen3-TTS-Triton](https://github.com/newgrit1004/ComfyUI-Qwen3-TTS-Triton)
  - [ComfyUI-ZImage-Triton](https://github.com/newgrit1004/ComfyUI-ZImage-Triton)
- **Paper:** [arXiv:2604.00688](https://arxiv.org/abs/2604.00688)

## License

Apache-2.0 — see [LICENSE](LICENSE).
