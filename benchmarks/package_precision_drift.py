"""Keep a small drift summary beside deterministic gzip-compressed full samples.

Run only after the measurement finishes. --input and --output may name the same
JSON file: the full payload is compressed and verified before it is replaced.
No measurements or source hashes are changed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import precision_drift


POPULATION_FIELDS = ("mean", "std", "at_zero_fraction", "at_one_fraction")
STATE_FIELDS = precision_drift.FIELDS


def late_population_statistics(samples, steps):
    ticks = [sample["tick"] for sample in samples]
    start, end = steps * 0.8, steps
    if ticks[0] > start or ticks[-1] < end or end <= start:
        raise ValueError("samples must cover the full late 20 percent of the run")

    def average(values):
        total = 0.0
        for index in range(1, len(ticks)):
            lo, hi = max(start, ticks[index - 1]), min(end, ticks[index])
            if hi <= lo:
                continue
            duration = ticks[index] - ticks[index - 1]
            y0, y1 = values[index - 1], values[index]
            left = y0 + (y1 - y0) * ((lo - ticks[index - 1]) / duration)
            right = y0 + (y1 - y0) * ((hi - ticks[index - 1]) / duration)
            total += (left + right) * 0.5 * (hi - lo)
        return total / (end - start)

    def population_series(mode, field, name):
        if mode == "reference_fp32":
            return [sample["reference"][field][name] for sample in samples]
        if mode == "fp32_other_backend":
            return [sample[mode][field]["candidate_" + name] for sample in samples]
        return [sample["precisions"][mode][field]["candidate_" + name]
                for sample in samples]

    reference = {}
    candidates = {}
    modes = (*precision_drift.MODES[1:], "fp32_other_backend")
    for field in STATE_FIELDS:
        reference[field] = {name: average(population_series("reference_fp32", field, name))
                            for name in POPULATION_FIELDS}
        for mode in modes:
            stats = {name: average(population_series(mode, field, name))
                     for name in POPULATION_FIELDS}
            stats["difference_vs_fp32"] = {name: stats[name] - reference[field][name]
                                              for name in POPULATION_FIELDS}
            candidates.setdefault(mode, {})[field] = stats
    return {"window_ticks": {"start": start, "end": end},
            "weighting": "time-weighted average of sampled population statistics with linear interpolation at window boundaries",
            "reference_fp32": reference, "candidates": candidates}


def package(source: Path, output: Path):
    raw = source.read_bytes()
    payload = json.loads(raw)
    if "samples_file" in payload:
        raise ValueError("input is already a packaged summary; provide the full raw JSON")
    cases = payload["cases"]
    protocol = payload["protocol"]
    expected = {(scenario, seed, backend) for scenario in protocol["scenarios"]
                for seed in protocol["seeds"] for backend in ("native", "reference")}
    actual = {(case["scenario"], case["seed"], case["backend"]) for case in cases}
    if actual != expected or len(cases) != len(expected):
        raise ValueError("measurement is incomplete: cases do not cover the protocol")
    if any(not case.get("samples") or case["samples"][-1]["tick"] != protocol["steps"]
           for case in cases):
        raise ValueError("measurement is incomplete: a case has not reached its final tick")
    for case in cases:
        control_samples = [{"tick": sample["tick"],
                            "precisions": {"fp32_other_backend": sample["fp32_other_backend"]}}
                           for sample in case["samples"]]
        case["summary"]["fp32_backend_control"] = precision_drift.summarize(
            control_samples, protocol["steps"])["fp32_other_backend"]
        case["summary"]["late_population_statistics"] = late_population_statistics(
            case["samples"], protocol["steps"]
        )
    compressed_path = output.with_name(output.name + ".gz")
    if source.resolve() == compressed_path.resolve():
        raise ValueError("input cannot be the compressed output path")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0, compresslevel=9) as archive:
        archive.write(raw)
    compressed = buffer.getvalue()
    assert gzip.decompress(compressed) == raw
    summary = {key: value for key, value in payload.items() if key != "cases"}
    summary["cases"] = [{key: value for key, value in case.items() if key != "samples"}
                        for case in cases]
    summary["samples_file"] = compressed_path.name
    summary["samples_sha256"] = hashlib.sha256(compressed).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    compressed_path.write_bytes(compressed)
    assert hashlib.sha256(compressed_path.read_bytes()).hexdigest() == summary["samples_sha256"]
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return compressed_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="complete raw measurement JSON")
    parser.add_argument("--output", type=Path, required=True, help="compact summary JSON")
    args = parser.parse_args()
    compressed = package(args.input, args.output)
    print(f"wrote summary {args.output} and complete samples {compressed}")


if __name__ == "__main__":
    main()
