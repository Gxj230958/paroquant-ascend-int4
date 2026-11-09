"""Plot the independent Ascend model timing summary."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=Path("results/summary"))
args = parser.parse_args()
rows = json.loads((args.input / "generation.json").read_text())
modes = ("naive", "dense_fused", "bf16", "optimized")
labels = ("Expand + unfused", "Dense + fused", "Original BF16", "Packed INT4 + fused")
colors = ("#bbbbbb", "#e69f00", "#0072b2", "#009e73")
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for size, axis_row in zip((4, 8), axes):
    subset = [row for row in rows if f"{size}B" in row["model"]]
    for ax, field, ylabel in zip(
        axis_row,
        ("ttft_ms", "decode_tokens_per_second", "peak_mib"),
        ("First token (ms)", "Decode (tokens/s)", "Peak allocated (GiB)"),
    ):
        for index, (mode, label, color) in enumerate(zip(modes, labels, colors)):
            points = sorted((row for row in subset if row["mode"] == mode), key=lambda row: row["document_tokens"])
            centers = [row[field] / 1024 if field == "peak_mib" else row[field + "_mean"] for row in points]
            errors = (
                None
                if field == "peak_mib"
                else (
                    [value - row[field + "_min"] for row, value in zip(points, centers)],
                    [row[field + "_max"] - value for row, value in zip(points, centers)],
                )
            )
            ax.bar(
                [i + (index - 1.5) * 0.2 for i in range(len(points))],
                centers,
                width=0.19,
                yerr=errors,
                capsize=2,
                label=label,
                color=color,
            )
        ax.set_xticks(range(3), ["32", "512", "2048"])
        ax.set_xlabel("Document tokens")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Qwen3-{size}B")
        if field == "ttft_ms":
            ax.set_yscale("log")
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(args.input / "model_comparison.pdf")
fig.savefig(args.input / "model_comparison.png", dpi=180)

rows = json.loads((args.input / "layers.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
for size, axis_row in zip((4, 8), axes):
    for suffix, ax in zip(("self_attn.q_proj", "mlp.down_proj"), axis_row):
        for mode in ("naive", "dense", "dense_fused", "native_unfused", "native", "triton"):
            points = sorted(
                (
                    row
                    for row in rows
                    if f"{size}B" in row["model"] and row["module"].endswith(suffix) and row["mode"] == mode
                ),
                key=lambda row: int(row["shape"].split("x")[0]),
            )
            centers = [row["milliseconds_mean"] for row in points]
            errors = (
                [value - row["milliseconds_min"] for row, value in zip(points, centers)],
                [row["milliseconds_max"] - value for row, value in zip(points, centers)],
            )
            ax.errorbar(range(len(points)), centers, yerr=errors, marker="o", capsize=2, label=mode)
        ax.set_title(f"Qwen3-{size}B {suffix.rsplit('.', 1)[-1]}")
        ax.set_xticks(range(3), ["1", "16", "128"])
        ax.set_xlabel("Input rows")
        ax.set_ylabel("Complete linear (ms)")
        ax.set_yscale("log")
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(args.input / "layer_comparison.pdf")
fig.savefig(args.input / "layer_comparison.png", dpi=180)
