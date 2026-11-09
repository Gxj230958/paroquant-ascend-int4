# Ascend inference

The Ascend backend loads the published Qwen3-4B-PARO and Qwen3-8B-PARO
checkpoints without changing their rotation parameters or AWQ tensor format.
Eight pairwise rotation stages execute in one Triton Ascend kernel, with FP16
rounding between stages. The packed INT4 weights are reordered once for the
native Ascend W4A16 matrix operator. Full dequantized weights are not stored.
The converted buffers are excluded from the serialized state dictionary.

## Environment

The implementation uses Python 3.11, PyTorch 2.9, torch_npu 2.9.0.post1,
Transformers 4.57.6, Triton Ascend 3.2.0.dev20260322, and CANN 8.5.1 on
Ascend 910B2C. Install ParoQuant without CUDA extensions in an environment
containing these packages. The NPU path imports its kernels lazily.

```python
import torch
import torch_npu
from transformers import AutoModelForCausalLM
import paroquant.inference.backends.transformers.quantizer

model = AutoModelForCausalLM.from_pretrained(
    "models/Qwen3-4B-PARO", torch_dtype=torch.float16,
    device_map="npu", attn_implementation="sdpa")
```

The first forward pass prepares rotation coefficients and the native packed
layout. Subsequent forwards reuse these buffers. Each matrix operation receives
the original quantization scales and equivalent signed zero offsets.

The eager implementation uses packed INT4 weights in ND format and the native
operator's high-precision mode. The installed operator documentation restricts
packed INT4 FRACTAL_NZ execution to graph mode. Its per-group NZ configuration
also requires a transposed activation input; it is a distinct execution path.

## Experiments

Source the local CANN environment and verify that the selected NPU is healthy
and idle before running these commands outside the device-restricting sandbox.

```bash
python experiments/ascend_correctness.py
python experiments/benchmark_layers.py --model models/Qwen3-4B-PARO \
    --output results/layers_4b.json
python experiments/evaluate_ascend.py --model models/Qwen3-4B-PARO \
    --mode optimized --new-tokens 128 --layer-errors results/layer_errors_4b.json \
    --output results/generation_4b.json
python experiments/evaluate_ascend.py --model models/Qwen3-4B-PARO \
    --mode optimized --task perplexity --data data/wikitext2_test.jsonl \
    --output results/perplexity_4b.json
python experiments/evaluate_ascend.py --model models/Qwen3-4B-PARO \
    --mode optimized --task gsm8k --data data/gsm8k_test.jsonl --new-tokens 512 \
    --output results/gsm8k_4b.json
```

Layer comparisons include materialized per-stage rotations with full weight
dequantization, cached dense weights, fused rotations with cached dense weights,
the native matrix operator with unfused rotations, the complete optimized path,
and tiled Triton INT4 dequantization/matrix multiplication. Layer measurements
record layout preparation separately. Full-model first-forward time includes
lazy rotation and weight-layout preparation; steady execution reuses these
buffers. Generation reports loading, first-forward and prompt-specific warmup
separately from steady first-token and decode latency, with both cold and
steady peak allocation.
The dense full-model baseline releases packed weights, zero points, and scales
after expansion, and retains only the rotation representation used by its
forward path. Its loading peak includes the expansion; steady allocation
reflects the required FP16 representation.
Model loading and dataset evaluation use the same public checkpoints in every
execution mode. BF16 comparisons use the corresponding original Qwen model.

The complete experiment scripts run three independent processes for each
timing configuration and evaluate all 1319 GSM8K test examples:

```bash
bash experiments/run_layers_formal.sh
bash experiments/run_models_formal.sh
ASCEND_CARD=6 MODEL_SIZE=4 bash experiments/run_quality.sh
ASCEND_CARD=7 MODEL_SIZE=8 bash experiments/run_quality.sh
python experiments/summarize_ascend.py
```

A selected mode can run independently by passing it to `run_quality.sh`.
After the exclusive timing runs, the six quality jobs use this allocation:

| Card | Model | Mode | Command |
|---|---|---|---|
| 6 | Qwen3-4B-PARO | optimized | `ASCEND_CARD=6 MODEL_SIZE=4 bash experiments/run_quality.sh optimized` |
| 7 | Qwen3-8B-PARO | optimized | `ASCEND_CARD=7 MODEL_SIZE=8 bash experiments/run_quality.sh optimized` |
| 8 | Qwen3-4B-PARO | dense_fused | `ASCEND_CARD=8 MODEL_SIZE=4 bash experiments/run_quality.sh dense_fused` |
| 9 | Qwen3-8B-PARO | dense_fused | `ASCEND_CARD=9 MODEL_SIZE=8 bash experiments/run_quality.sh dense_fused` |
| 10 | Qwen3-4B | bf16 | `ASCEND_CARD=10 MODEL_SIZE=4 bash experiments/run_quality.sh bf16` |
| 11 | Qwen3-8B | bf16 | `ASCEND_CARD=11 MODEL_SIZE=8 bash experiments/run_quality.sh bf16` |

Each job evaluates perplexity before GSM8K and retains its existing result
paths. Concurrent quality-run elapsed times are excluded from performance
comparisons. `experiments/quality_jobs.json` records the allocation and protocol.

The quality jobs save every completed GSM8K batch and WikiText token segment.
Rerunning a job restores the same configuration, skips saved indices, and
continues the accumulated cross-entropy or answer count. Published optimized
WikiText evaluations are reused when present. All six model/mode quality jobs can run concurrently after exclusive timing
experiments finish.
Generation timing uses documents of 32, 512, and 2048 tokens followed by
exactly eight generated tokens; prompt formatting contributes to the reported
actual input length. Only the final token's vocabulary logits are computed
during generation. Perplexity evaluates all next-token logits.

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

## Limitations

The NPU backend supports the published FP16, four-bit, group-size-128 checkpoint
configuration. Measurements describe a single Ascend 910B2C and the listed
software versions. Cached native weights occupy an additional packed copy;
their memory is included in model-level allocation measurements.
