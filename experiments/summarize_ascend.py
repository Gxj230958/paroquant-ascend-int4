"""Summarize independent layer/model runs and complete held-out evaluations."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def write_table(directory, name, rows):
    if not rows:
        return
    with (directory / f"{name}.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (directory / f"{name}.json").write_text(json.dumps(rows, indent=2))


def interval(values, prefix):
    return {f"{prefix}_mean": statistics.mean(values), f"{prefix}_min": min(values), f"{prefix}_max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/summary"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = {"layers": [], "generation": [], "quality": []}

    groups = defaultdict(list)
    for path in sorted((args.results / "layers_formal").glob("*.json")):
        result = json.loads(path.read_text())
        raw["layers"].append(result)
        for row in result["records"]:
            key = (result["model"], row["module"], tuple(row["shape"]), row["mode"])
            groups[key].append((result["seed"], row))
    rows = []
    for (model, module, shape, mode), observations in groups.items():
        if len({seed for seed, _ in observations}) != 3:
            raise ValueError(f"Three independent layer processes required: {model} {module} {shape} {mode}")
        measurements = [row for _, row in observations]
        rows.append(
            {
                "model": model,
                "module": module,
                "shape": "x".join(map(str, shape)),
                "mode": mode,
                "independent_processes": len(observations),
                **interval([row["seconds"] * 1000 for row in measurements], "milliseconds"),
                "max_relative_l2": max(row["relative_l2"] for row in measurements),
                "max_absolute_error": max(row["max_error"] for row in measurements),
                "weight_layout_ms": statistics.mean(row["weight_layout_seconds"] * 1000 for row in measurements),
                "rotation_prepare_ms": statistics.mean(row["rotation_prepare_seconds"] * 1000 for row in measurements),
            }
        )
    write_table(args.output, "layers", rows)

    groups = defaultdict(list)
    for path in sorted((args.results / "models_formal").glob("*.json")):
        result = json.loads(path.read_text())
        if not result.get("complete"):
            raise ValueError(f"Generation run is incomplete: {path}")
        raw["generation"].append(result)
        for row in result["records"]:
            key = (result["model"], result["mode"], row["document_tokens"])
            groups[key].append(
                (
                    result["seed"],
                    {
                        **row,
                        "load_seconds": result["load_seconds"],
                        "initial_forward_seconds": result["warmup_seconds"],
                        "cold_peak_bytes": max(result["load_peak_bytes"], result["first_forward_peak_bytes"]),
                    },
                )
            )
    rows = []
    for (model, mode, document_tokens), observations in groups.items():
        if len({seed for seed, _ in observations}) != 3:
            raise ValueError(f"Three independent generation processes required: {model} {mode} {document_tokens}")
        measurements = [row for _, row in observations]
        rows.append(
            {
                "model": model,
                "mode": mode,
                "document_tokens": document_tokens,
                "prompt_tokens": measurements[0]["prompt_tokens"],
                "generated_tokens": measurements[0]["generated_tokens"],
                "independent_processes": len(observations),
                **interval([row["ttft_seconds"] * 1000 for row in measurements], "ttft_ms"),
                **interval([row["decode_tokens_per_second"] for row in measurements], "decode_tokens_per_second"),
                "peak_mib": max(row["peak_bytes"] for row in measurements) / 2**20,
                **interval([row["load_seconds"] for row in measurements], "load_seconds"),
                **interval([row["initial_forward_seconds"] for row in measurements], "initial_forward_seconds"),
                **interval([row["prompt_warmup_seconds"] for row in measurements], "prompt_warmup_seconds"),
                "cold_peak_mib": max(row["cold_peak_bytes"] for row in measurements) / 2**20,
            }
        )
    write_table(args.output, "generation", rows)

    rows = []
    for path in sorted((args.results / "quality").glob("*.json")):
        result = json.loads(path.read_text())
        if not result.get("complete"):
            raise ValueError(f"Quality evaluation is incomplete: {path}")
        raw["quality"].append(result)
        if result["task"] == "gsm8k" and result["evaluated_examples"] != 1319:
            raise ValueError(f"Full GSM8K evaluation required: {path}")
        rows.append(
            {
                "model": result["model"],
                "mode": result["mode"],
                "task": result["task"],
                "evaluated_tokens": result.get("evaluated_tokens"),
                "evaluated_examples": result.get("evaluated_examples"),
                "perplexity": result.get("perplexity"),
                "accuracy": result.get("accuracy"),
            }
        )
    write_table(args.output, "quality", rows)
    (args.output / "raw.json").write_text(json.dumps(raw, indent=2))


if __name__ == "__main__":
    main()
