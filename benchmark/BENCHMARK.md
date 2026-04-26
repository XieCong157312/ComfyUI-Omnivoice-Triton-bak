# ComfyUI-Omnivoice-Triton Benchmark

End-to-end benchmark of `OmnivoiceTTSAuto` across the 4 primary runner modes,
driven through the ComfyUI HTTP API (`POST /prompt` → poll `/history`).

## Method

- Entrypoint: [`run_benchmark.py`](./run_benchmark.py)
- Core runner: [`bench_comfyui.py`](./bench_comfyui.py)
- Shared helpers:
  - [`comfyui/client.py`](./comfyui/client.py) — ComfyUI HTTP client
  - [`comfyui/vram_monitor.py`](./comfyui/vram_monitor.py) — optional pynvml VRAM polling helper
  - [`comfyui/workflow_builder.py`](./comfyui/workflow_builder.py) — mode별 saved workflow JSON loader + override helper
  - [`stats.py`](./stats.py) — aggregation + summary table
- Additional direct-runner tools:
  - [`bench_e2e.py`](./bench_e2e.py) — direct `omnivoice-triton` runner benchmark
  - [`bench_kernels.py`](./bench_kernels.py) — lightweight runner microbenchmark
  - [`eval_quality.py`](./eval_quality.py) — WAV pair quality/stat comparison
  - [`profiler.py`](./profiler.py) — torch profiler trace export
- Workflow source: [`workflows/tts_*.json`](../workflows/) — one saved ComfyUI API workflow per runner mode.
- Execution model: load workflow JSON for a mode, override text / filename prefix per run, then `POST /prompt` to ComfyUI.
- Text: 57-char Korean sentence (`"안녕하세요, OmniVoice Triton 벤치마크 테스트입니다. 오늘 날씨가 참 좋네요. 반갑습니다."`)
- Cache-bust: per-run unique trailing marker appended to text — ComfyUI caches TTS output by input hash; without this, runs after the first return in ~0.1s from cache.
- Warmup: 1 run per mode (covers model load / Triton JIT / CUDA Graph capture)
- Measured: 5 runs per mode
- Metric sources:
  - `gen_s` → from runner return dict (`result["time_s"]`), parsed from `[Omnivoice-Triton] TTS done: Xs` lines in `/tmp/comfyui.log`
  - `exec_s` → ComfyUI history `execution_start` → `execution_success` timestamp delta
  - `wall_s` → benchmark script wall clock (`POST /prompt` → `/history` completed)
  - `peak_vram_gb` → runner return dict
- Aggregation: median over the 5 measured runs

Mode transitions evict the previous runner (unload_model → free VRAM → create new runner → load_model), so the per-mode warmup pays a ~4s model-load cost.

## Code Layout

```text
benchmark/
├── run_benchmark.py          # thin CLI entrypoint
├── measure_vram.py           # nvidia-smi delta by mode
├── rerun_specific.py         # rerun selected modes with clean state
├── bench_comfyui.py          # ComfyUI workflow benchmark
├── bench_e2e.py              # direct runner benchmark
├── bench_kernels.py          # lightweight runner microbenchmark
├── eval_quality.py           # WAV stats comparison
├── profiler.py               # torch profiler trace export
├── stats.py                  # median/speedup reporting
├── assets/                   # sample texts and benchmark inputs
├── results/                  # extra benchmark outputs
└── comfyui/
    ├── client.py             # queue/history polling
    ├── vram_monitor.py       # optional VRAM polling helper
    └── workflow_builder.py   # load saved workflow templates from ../workflows/
```

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX 5090 (32 GB, SM 120 Blackwell) |
| CUDA | 12.8 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 |
| transformers | 5.3.0 |
| Python | 3.12.3 |
| OS | WSL2 (Ubuntu 24.04) |
| ComfyUI | 0.19.3 — launched with `--disable-cuda-malloc` |
| Model | `k2-fsa/OmniVoice` (fp16) |
| Node cache | one warm runner per `(mode, model_id, dtype, device)` |

## Results

| mode | gen_med (s) | exec_med (s) | wall_med (s) | peak VRAM | gen speedup | wall speedup |
|------|-------------|--------------|--------------|-----------|------------:|-------------:|
| base     | 0.650 | 0.658 | 0.710 | 2.01 GB | **1.00x** | **1.00x** |
| triton   | 0.630 | 0.636 | 0.710 | 2.01 GB | 1.03x | 1.00x |
| faster   | 0.260 | 0.270 | 0.304 | 2.04 GB | **2.50x** | **2.34x** |
| hybrid   | 0.190 | 0.194 | 0.204 | 2.05 GB | **3.42x** | **3.48x** |

Raw per-run data: [`benchmark_results.json`](./benchmark_results.json)

## Takeaways

1. **`hybrid` is the best mode** — 3.42x speedup on runner-internal generation time, 3.48x on end-to-end wall clock. This **exceeds** the 2.8x claim in `omnivoice-triton`'s README.
2. **`faster` (CUDA Graph alone) already delivers 2.50x** — the launch-overhead removal is the dominant contribution. On short inputs like this 57-char sentence, PyTorch op launch cost is a meaningful fraction of total latency.
3. **`triton` alone is nearly a no-op (1.03x) on short text** — Triton kernel fusion (RMSNorm, SwiGLU, fused add+norm) saves nothing meaningful when the per-kernel workload is tiny. The kernels help more as sequence length grows; this benchmark under-represents their contribution. Longer-text sweeps would show a different story.
4. **VRAM delta is essentially zero** — all four modes sit at 2.01-2.05 GB peak. The CUDA Graph memory pool adds ~40 MB. No mode is a memory trade-off.
5. **Wall-clock vs gen time gap** — `base` has wall 0.710s but gen 0.650s (≈60ms overhead for ComfyUI prompt dispatch + SaveAudio encoding + HTTP polling). `hybrid` has wall 0.204s vs gen 0.190s (≈14ms overhead) — the fixed overhead dominates less in absolute terms for faster modes, but ComfyUI/HTTP still adds ~14ms floor.
6. **First-run cost** — warmup is ~4-5s regardless of mode, dominated by `load_model` and (for Triton/hybrid) Triton kernel JIT and CUDA Graph capture. Amortizes fast.

## How to reproduce

```bash
# 1. Start ComfyUI with cudaMallocAsync disabled
cd services/video_generation
.venv/bin/python ComfyUI/main.py --listen 0.0.0.0 --port 8188 --disable-cuda-malloc

# 2. In another shell, run the benchmark
cd ComfyUI/custom_nodes/ComfyUI-Omnivoice-Triton
.venv/bin/python benchmark/run_benchmark.py

# Optional: override text / modes / log path
.venv/bin/python benchmark/run_benchmark.py \
  --modes base triton faster hybrid \
  --text "Hello from ComfyUI benchmark" \
  --log-path /tmp/comfyui.log
```

Results land in `benchmark/benchmark_results.json` and are printed as a table.

## Caveats

- **Short-text bias.** Only one 57-char Korean sentence. Longer inputs (e.g., 200+ chars) would stress the attention path more, favoring Triton kernel fusion. If you need to see `triton` shine alone, rerun with much longer text.
- **CUDA Graph capture is static-shape.** `faster`/`hybrid` capture graphs per shape, so inputs with varying length pay an extra capture cost on first sight. This benchmark uses near-identical text lengths so the graph is reused.
- **KV-cache variants are not in the current PyPI release.** The clean `pip install omnivoice-triton==0.1.0` environment used for this ComfyUI benchmark exposes only `base`, `triton`, `faster`, and `hybrid`. `kv_cache*` exists only in the unreleased local branch, so no ComfyUI workflows are shipped for it yet.
- **Peak VRAM is from the runner's internal tracking.** ComfyUI's own memory tracking may differ, but this is what users care about (the node's marginal VRAM cost above baseline).
- **One machine.** RTX 5090 only. Other Blackwell / Ada / Ampere / Hopper cards will shift the numbers.
