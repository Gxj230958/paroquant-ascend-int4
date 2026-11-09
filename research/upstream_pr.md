Published ParoQuant checkpoints can run through the Transformers backend on Ascend NPU. This adds fused pairwise rotations and a packed W4A16 execution path while preserving the learned rotation parameters and checkpoint format.

The Triton Ascend rotation kernel fuses channel scaling and all eight group-128 rotation stages, preserving FP16 rounding between stages. A one-time conversion reorders AWQ nibbles and maps unsigned quantized values to the native signed INT4 representation with equivalent zero offsets. The production path uses `npu_weight_quant_batchmatmul`; a tiled Triton unpack/matrix path provides an independent comparison. Prepared rotation and packed-weight buffers are nonpersistent. CUDA extensions load only on the CUDA path.

The numerical checks compare rotation against independent CPU calculations and matrix products against explicitly dequantized weights. Row counts 1, 3, 17, and 64 cover both small-row reduction and matrix paths. The native path's largest relative L2 error is 0.000562. Actual Qwen3-4B-PARO and Qwen3-8B-PARO activations validate all 252 quantized linears in each model against materialized rotations and FP16 weights; maximum relative L2 errors are 0.001092 and 0.002660 respectively.


On an Ascend 910B2C, three independent full-model processes at each of three document lengths show 38.9–42.1% lower peak allocated memory than a fused dense FP16 reference using the same PARO weights. Decode throughput is 45.6–83.6× the naive per-call expansion path and 5.0–18.1% lower than the cached dense reference. Original BF16 models are faster. The dense reference releases packed storage after expansion; loading, initial preparation, and steady measurements are reported separately.

At 2048 document tokens (2070 after chat formatting), each request produces eight tokens: TTFT includes the first and decode speed uses the following seven.

| Model | Path | TTFT (ms) | Decode (tokens/s) | Peak allocated (GiB) |
|---|---|---:|---:|---:|
| 4B | Packed INT4 + fused | 284.61 | 22.656 | 4.857 |
| 4B | Dense FP16 + fused | 280.60 | 23.854 | 8.142 |
| 4B | Original BF16 | 112.85 | 48.546 | 7.978 |
| 8B | Packed INT4 + fused | 395.31 | 21.687 | 9.714 |
| 8B | Dense FP16 + fused | 401.17 | 24.059 | 15.911 |
| 8B | Original BF16 | 161.36 | 40.615 | 15.760 |

Six operator paths, two projections, and three row counts provide 216 measured cases across three process seeds. Rotation fusion improves complete-linear latency in every tested shape; the tiled Triton matrix alternative is slower than native. All measured paths pass their numerical comparison. The complete process ranges, preparation costs, and raw records accompany the reproduction documentation.

## Model quality

All six configurations evaluate the same full WikiText-2 raw test sequence
(299,077 next-token targets) and all 1319 GSM8K test questions. GSM8K uses
greedy non-thinking decoding, batches of eight, and at most 512 new tokens.
Each row reports the complete result; concurrent quality-job timings are not
used as performance evidence.

| Model | Execution | WikiText-2 perplexity | GSM8K correct / 1319 | GSM8K accuracy |
|---|---|---:|---:|---:|
| 4B | Packed INT4 + fused | 13.858716 | 1181 | 89.54% |
| 4B | Dense FP16 + fused | 13.859074 | 1185 | 89.84% |
| 4B | Original BF16 | 13.665473 | 1194 | 90.52% |
| 8B | Packed INT4 + fused | 9.881995 | 1149 | 87.11% |
| 8B | Dense FP16 + fused | 9.882060 | 1147 | 86.96% |
| 8B | Original BF16 | 9.728429 | 1152 | 87.34% |

Native and dense execution use identical learned PARO weights and rotations.
Their perplexities differ by less than 0.00036. The original BF16 checkpoints
have lower perplexity. GSM8K results use the stated 512-token non-thinking
protocol; the table separates execution effects from checkpoint quantization.

Validation commands:

```bash
python experiments/ascend_correctness.py
python experiments/evaluate_ascend.py --model models/Qwen3-4B-PARO \
  --mode optimized --layer-errors results/layer_errors_4b.json \
  --output results/layer_validation_4b.json
python experiments/evaluate_ascend.py --model models/Qwen3-8B-PARO \
  --mode optimized --layer-errors results/layer_errors_8b.json \
  --output results/layer_validation_8b.json
```

The supported checkpoint configuration uses FP16 activations, four-bit weights, and groups of 128 channels. Measurements target Ascend 910B2C with PyTorch 2.9, torch-npu 2.9.0.post1, Triton Ascend 3.2.0.dev20260322, Transformers 4.57.6, and CANN 8.5.1. The native eager path uses ND packed weights. Prepared weights occupy one additional packed copy; serialized checkpoints keep their original tensors.
