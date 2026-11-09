# ParoQuant on Ascend: Fused Rotation and Packed INT4 Execution

An Ascend 910B execution path for published [ParoQuant](https://github.com/z-lab/paroquant)
checkpoints, adding a fused eight-stage pairwise-rotation kernel in Triton Ascend and
native packed-INT4 (W4A16) matrix execution. This repository is a snapshot of the
`research/ascend-evaluation` branch prepared for review and reproduction; the draft
upstream contribution is [ParoQuant PR #57](https://github.com/z-lab/paroquant/pull/57).

## Overview

- **What:** the Ascend backend runs the published Qwen3-4B-PARO and Qwen3-8B-PARO
  checkpoints directly through the Transformers path. Learned rotations, channel
  scales, group scales, and zero points are unchanged, and the checkpoint
  serialization format is preserved (converted packed buffers are nonpersistent).
- **Fused rotation:** one Triton Ascend kernel fuses channel scaling and all eight
  group-128 pairwise rotation stages, rounding to FP16 between stages. Partner
  indices and sine/cosine coefficients are prepared once. Decode uses one input row
  per program; larger inputs use eight rows per program.
- **Packed INT4 execution:** a one-time conversion reorders AWQ nibbles
  (`[0, 2, 4, 6, 1, 3, 5, 7]` packing to consecutive signed nibbles, mapping
  `q → q − 8` with zero offset `8 − z`, so dequantization stays `(q − z) × scale`)
  and feeds the native `npu_weight_quant_batchmatmul` W4A16 operator (ND layout,
  high-precision mode). Full dequantized weights are never stored. A tiled Triton
  unpack-and-multiply path (vector reduction for at most four rows, Cube matrix
  operation for larger inputs) is included as an execution alternative.
- **Memory and speed** (three independent processes per configuration): compared
  with cached FP16 execution of the same quantized weights (dense fused baseline),
  the packed INT4 path reduces peak allocated memory by **38.9–42.1%**, with decode
  throughput **5.0–18.1% lower**. Against the naive expand-and-rotate path, mean
  decode throughput is **45.6–83.6×**. The original BF16 models are faster than
  either quantized path on this host.
- **Quality** (full test sets): WikiText-2 perplexity over all 299,077 next-token
  targets is 13.858716 (4B) and 9.881995 (8B) for packed INT4 execution, within
  0.00036 of the dense FP16 fused reference using identical weights. GSM8K (all
  1319 questions, greedy non-thinking decoding, ≤512 new tokens) accuracy is
  89.54% (1181/1319) for 4B and 87.11% (1149/1319) for 8B. All 252 quantized
  linear layers in each model validate against explicit FP16 dequantization with
  maximum relative L2 error 0.001092 (4B) / 0.002660 (8B); all 216 measured
  operator cases pass.

Measured at 2048 document tokens (2070 after chat formatting), eight generated
tokens per request, mean of three independent processes:

| Model | Path | TTFT (ms) | Decode (tokens/s) | Peak allocated (GiB) |
|---|---|---:|---:|---:|
| 4B | Expand + unfused (naive) | 3554.56 | 0.435 | 3.482 |
| 4B | Dense FP16 + fused | 280.60 | 23.854 | 8.142 |
| 4B | Original BF16 | 112.85 | 48.546 | 7.978 |
| 4B | Packed INT4 + fused | 284.61 | 22.656 | 4.857 |
| 8B | Expand + unfused (naive) | 5516.80 | 0.259 | 7.218 |
| 8B | Dense FP16 + fused | 401.17 | 24.059 | 15.911 |
| 8B | Original BF16 | 161.36 | 40.615 | 15.760 |
| 8B | Packed INT4 + fused | 395.31 | 21.687 | 9.714 |

Full tables (32/512/2048 document tokens with per-process ranges), the six-path
operator ablation, loading/preparation costs, and per-layer numerical errors are
in [`RESEARCH.md`](RESEARCH.md); summary figures are in `results/summary/`.

**Hardware and software:** Ascend 910B2C NPU with CANN 8.5.1; Python 3.11,
PyTorch 2.9, torch\_npu 2.9.0.post1, Transformers 4.57.6, Triton Ascend
3.2.0.dev20260322. Exact package versions are recorded in
[`environment_ascend.json`](environment_ascend.json).

## Links

- Technical report (methodology, measurements, numerical validation): [`RESEARCH.md`](RESEARCH.md)
- Environment and reproduction commands: [`ASCEND.md`](ASCEND.md)
- Chinese walkthroughs: [`research/中文讲解.md`](research/中文讲解.md), [`research/入门教学与面试问答.md`](research/入门教学与面试问答.md)
- Upstream project: <https://github.com/z-lab/paroquant> (base commit
  [`9ee635a`](https://github.com/z-lab/paroquant/commit/9ee635a9c2e7d02c46b1c0ac4071d69afbbfc1f1))
- Draft upstream contribution: <https://github.com/z-lab/paroquant/pull/57>
- Full development history: branch `research/ascend-evaluation` and related
  branches in the author's fork, <https://github.com/Gxj230958/paroquant>
- ParoQuant paper: <https://arxiv.org/abs/2511.10645>

## Repository contents

| Path | Contents |
|---|---|
| `paroquant/kernels/ascend/` | Fused-rotation Triton Ascend kernel and native W4A16 packed-INT4 wrappers (`linear.py`, `native.py`) |
| `paroquant/inference/backends/transformers/` | Transformers backend: quantized linear modules, quantizer integration, generation |
| `paroquant/` (rest) | Upstream ParoQuant package: CUDA/Metal kernels, MLX and vLLM backends, optimization CLI |
| `experiments/` | Correctness harness (`ascend_correctness.py`), layer and model benchmarks (`benchmark_layers.py`, `evaluate_ascend.py`), formal run scripts (`run_layers_formal.sh`, `run_models_formal.sh`, `run_quality.sh`), summarization and plotting (`summarize_ascend.py`, `plot_ascend.py`), dataset/checkpoint fetchers |
| `results/` | Raw and summary measurements as JSON/CSV plus standalone PNG/PDF figures (`results/summary/`) |
| `research/` | Chinese technical walkthroughs, application narrative, upstream PR text and review record, checkpoint/dataset provenance (`model_sources.json`), numerical/resume validation records |
| `data/` | `sources.json` recording dataset revisions (the JSONL data itself is fetched by `experiments/download_data.py`) |
| `docker/`, `.github/` | Upstream Dockerfile and CI workflows, retained from the fork |
| `RESEARCH.md`, `ASCEND.md`, `environment_ascend.json` | Report, reproduction guide, and recorded package versions |

Quantized and base model weights (~24 GB) and dataset JSONLs are intentionally not
included; use the fetch scripts described below.

## Reproduction

Requires an Ascend 910B NPU host with CANN 8.5.1 and the software versions listed
above (see [`ASCEND.md`](ASCEND.md) for the detailed environment notes, including
why eager execution uses the ND packed layout).

```bash
# 1. Install (no CUDA extensions needed for the NPU path)
pip install -e .       # plus torch, torch_npu, transformers, triton-ascend per ASCEND.md

# 2. Fetch the published checkpoints and the original baselines into models/
python experiments/download_models.py --model z-lab/Qwen3-4B-PARO
python experiments/download_models.py --model z-lab/Qwen3-8B-PARO
python experiments/download_models.py --model Qwen/Qwen3-4B
python experiments/download_models.py --model Qwen/Qwen3-8B

# 3. Fetch the evaluation datasets (records revisions in data/sources.json)
python experiments/download_data.py

# 4. Numerical correctness of the fused kernels (CPU references, several row counts)
python experiments/ascend_correctness.py

# 5. Operator ablation (six paths, q_proj/down_proj, 1/16/128 rows)
python experiments/benchmark_layers.py --model models/Qwen3-4B-PARO \
    --output results/layers_4b.json

# 6. Full-model timing, per-layer error against FP16 expansion, and quality
python experiments/evaluate_ascend.py --model models/Qwen3-4B-PARO \
    --mode optimized --new-tokens 128 --layer-errors results/layer_errors_4b.json \
    --output results/generation_4b.json
python experiments/evaluate_ascend.py --model models/Qwen3-4B-PARO \
    --mode optimized --task perplexity --data data/wikitext2_test.jsonl \
    --output results/perplexity_4b.json
python experiments/evaluate_ascend.py --model models/Qwen3-4B-PARO \
    --mode optimized --task gsm8k --data data/gsm8k_test.jsonl --new-tokens 512 \
    --output results/gsm8k_4b.json

# 7. The complete protocol in this repository's tables (three independent
#    processes per timing configuration; all 1319 GSM8K questions)
bash experiments/run_layers_formal.sh
bash experiments/run_models_formal.sh
bash experiments/run_quality.sh          # per-card/model selection via ASCEND_CARD/MODEL_SIZE
python experiments/summarize_ascend.py   # CSV/JSON summary tables
python experiments/plot_ascend.py        # standalone PDF/PNG figures
```

Quality jobs save progress after every batch or token segment and resume with the
same explicit configuration, skipping completed indices.

## Limitations

The supported checkpoints use FP16 activations, four-bit weights, and 128-channel
groups. Measurements describe a single Ascend 910B2C host with the listed software.
The native packed representation occupies one additional copy of the quantized
weights. On this host the native operator restricts packed INT4 FRACTAL\_NZ
execution to graph mode, so the eager path uses the ND layout. See
[`RESEARCH.md`](RESEARCH.md#limitations) for details.

## License and citation

[MIT](LICENSE), inherited from upstream ParoQuant. Please cite the ParoQuant
paper (ICLR 2026) when using this work; see `research/application.md` for a
summary of this Ascend contribution.
