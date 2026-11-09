# Research experience

**ParoQuant inference on Ascend 910B2C**

- Ported published Qwen3-4B-PARO and Qwen3-8B-PARO checkpoints to Ascend
  without changing learned rotations or the checkpoint quantization format.
- Implemented fused eight-stage pairwise rotations in Triton Ascend and
  lossless AWQ-to-native INT4 packing for W4A16 matrix multiplication, with
  tilewise dequantization as an alternative execution path.
- Reduced complete-model peak allocated memory by 38.9–42.1% versus a fused
  FP16 reference using the same quantized weights; measured 45.6–83.6× decode
  throughput versus per-call weight expansion across 4B/8B and three context
  lengths. The packed path trades 5.0–18.1% lower decode throughput for memory
  savings against the strong cached FP16 baseline.
- Validated all 252 quantized linears in each model against explicit FP16
  dequantization and evaluated 299,077 WikiText-2 targets, obtaining
  perplexities of 13.8587 and 9.8820 for the 4B and 8B checkpoints. Full
  GSM8K evaluation (1319 questions, greedy non-thinking, 512-token limit)
  reaches 89.54% (1181 correct) and 87.11% (1149 correct), respectively.

Code and methodology: [`../RESEARCH.md`](../RESEARCH.md).

Draft contribution: [ParoQuant #57](https://github.com/z-lab/paroquant/pull/57).
