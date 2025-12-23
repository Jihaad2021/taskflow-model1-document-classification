# scripts/benchmark_onnx_model1.py

from __future__ import annotations

import sys
import argparse
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort

from src.models.onnx_export import export_sequence_classification_to_onnx
from src.models.onnx_inference import OnnxSequenceClassifier
from src.models.onnx_optimize import optimize_onnx_with_ort, quantize_onnx_dynamic_int8

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, PROJECT_ROOT.as_posix())

def _percentile(values_ms: Sequence[float], pct: float) -> float:
    """Compute a percentile for a list of latencies (in ms)."""
    if len(values_ms) == 0:
        raise ValueError("values_ms must be non-empty.")
    arr = np.array(values_ms, dtype=float)
    return float(np.percentile(arr, pct))


def _make_text_batch(batch_size: int) -> List[str]:
    """Create a deterministic batch of texts for benchmarking."""
    base = [
        "breaking news: market rallies after earnings report",
        "sports update: local team wins championship game",
        "technology: new smartphone model announced today",
        "business: company reports quarterly revenue growth",
        "world: diplomatic talks continue amid tensions",
    ]
    texts = []
    for i in range(batch_size):
        texts.append(base[i % len(base)] + f" #{i}")
    return texts


def _benchmark_model(
    name: str,
    clf: OnnxSequenceClassifier,
    texts: List[str],
    *,
    warmup_iters: int,
    bench_iters: int,
) -> Dict[str, float]:
    """Benchmark an ONNX model using the classifier predict() API."""
    # Warmup
    for _ in range(warmup_iters):
        _ = clf.predict(texts)

    # Timed iterations
    latencies_ms: List[float] = []
    start_all = time.perf_counter()

    for _ in range(bench_iters):
        t0 = time.perf_counter()
        _ = clf.predict(texts)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    end_all = time.perf_counter()
    total_s = end_all - start_all

    p50 = _percentile(latencies_ms, 50)
    p95 = _percentile(latencies_ms, 95)
    mean = statistics.mean(latencies_ms)
    batch_size = len(texts)
    throughput = (bench_iters * batch_size) / total_s

    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "mean_ms": mean,
        "throughput_samples_per_s": throughput,
        "batch_size": float(batch_size),
        "iters": float(bench_iters),
    }


def _print_table(rows: List[Tuple[str, Dict[str, float]]]) -> None:
    headers = [
        ("model", 28),
        ("batch", 7),
        ("iters", 7),
        ("p50_ms", 10),
        ("p95_ms", 10),
        ("mean_ms", 10),
        ("samples/s", 12),
    ]

    def fmt_cell(val: str, width: int) -> str:
        return val[:width].ljust(width)

    line = " | ".join(fmt_cell(h, w) for h, w in headers)
    sep = "-+-".join("-" * w for _, w in headers)

    print(line)
    print(sep)
    for model_name, metrics in rows:
        print(
            " | ".join(
                [
                    fmt_cell(model_name, 28),
                    fmt_cell(str(int(metrics["batch_size"])), 7),
                    fmt_cell(str(int(metrics["iters"])), 7),
                    fmt_cell(f'{metrics["p50_ms"]:.3f}', 10),
                    fmt_cell(f'{metrics["p95_ms"]:.3f}', 10),
                    fmt_cell(f'{metrics["mean_ms"]:.3f}', 10),
                    fmt_cell(f'{metrics["throughput_samples_per_s"]:.2f}', 12),
                ]
            )
        )


def _build_classifier(
    onnx_path: Path,
    tokenizer_name_or_path: str,
    max_length: int,
    threads: int,
) -> OnnxSequenceClassifier:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return OnnxSequenceClassifier.from_pretrained(
        onnx_path=onnx_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_length=max_length,
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ONNX variants for Model 1.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="hf-internal-testing/tiny-random-distilbert",
        help="HF model id or local checkpoint directory (used for tokenizer + export).",
    )
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/onnx_bench",
        help="Directory to write ONNX artifacts for benchmarking.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = out_dir / "baseline.onnx"
    optimized_path = out_dir / "optimized.onnx"
    int8_path = out_dir / "int8.onnx"

    print(f"Exporting baseline ONNX to: {baseline_path}")
    export_sequence_classification_to_onnx(
        model_name_or_path=args.model_name,
        output_path=baseline_path,
        max_length=args.max_length,
        opset=13,
    )

    print(f"Optimizing ONNX graph to: {optimized_path}")
    optimize_onnx_with_ort(baseline_path, optimized_path)

    print(f"Quantizing ONNX to INT8 (dynamic) to: {int8_path}")
    quantize_onnx_dynamic_int8(baseline_path, int8_path)

    texts = _make_text_batch(args.batch_size)

    rows: List[Tuple[str, Dict[str, float]]] = []

    print("\nLoading models for benchmarking...")
    clf_baseline = _build_classifier(baseline_path, args.model_name, args.max_length, args.threads)
    clf_optimized = _build_classifier(optimized_path, args.model_name, args.max_length, args.threads)
    clf_int8 = _build_classifier(int8_path, args.model_name, args.max_length, args.max_length,)

    # NOTE: int8 model load may be slower on first run; warmups cover that.

    print("\nRunning benchmarks...")
    rows.append(("baseline.onnx", _benchmark_model("baseline", clf_baseline, texts, warmup_iters=args.warmup_iters, bench_iters=args.bench_iters)))
    rows.append(("optimized.onnx", _benchmark_model("optimized", clf_optimized, texts, warmup_iters=args.warmup_iters, bench_iters=args.bench_iters)))
    # For int8, use separate session options (threads still applied)
    clf_int8 = _build_classifier(int8_path, args.model_name, args.max_length, args.threads)
    rows.append(("int8_dynamic.onnx", _benchmark_model("int8", clf_int8, texts, warmup_iters=args.warmup_iters, bench_iters=args.bench_iters)))

    print("\nResults (CPUExecutionProvider):")
    _print_table(rows)

    print(f"\nArtifacts written to: {out_dir.resolve()}")
    print("Recommended next step: use the best-performing ONNX variant in FastAPI/Docker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
