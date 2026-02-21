#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THREAD_ENV_KEYS = (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def run_cmd(args: list[str]) -> str:
    try:
        out = subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return ""


def sysctl_int(key: str) -> int | None:
    raw = run_cmd(["sysctl", "-n", key])
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def collect_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "chip": run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "model": run_cmd(["sysctl", "-n", "hw.model"]),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.replace("\n", " "),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    mem_bytes = sysctl_int("hw.memsize")
    info["mem_bytes"] = mem_bytes
    info["mem_gb"] = (mem_bytes / (1024**3)) if mem_bytes else None

    perf_cores = sysctl_int("hw.perflevel0.physicalcpu")
    eff_cores = sysctl_int("hw.perflevel1.physicalcpu")
    logical = sysctl_int("hw.logicalcpu")

    info["perf_cores"] = perf_cores
    info["eff_cores"] = eff_cores
    info["logical_cores"] = logical
    info["l1d_perf_bytes"] = sysctl_int("hw.perflevel0.l1dcachesize")
    info["l1d_eff_bytes"] = sysctl_int("hw.perflevel1.l1dcachesize")
    info["l2_perf_bytes"] = sysctl_int("hw.perflevel0.l2cachesize")
    info["l2_eff_bytes"] = sysctl_int("hw.perflevel1.l2cachesize")
    info["cacheline_bytes"] = sysctl_int("hw.cachelinesize")
    info["mem_usable_bytes"] = sysctl_int("hw.memsize_usable")

    return info


def anonymize_system_info(info: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(info)
    sanitized["model"] = "redacted"
    sanitized["os"] = "redacted"
    sanitized["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    sanitized["timestamp_utc"] = "redacted"
    sanitized["mem_usable_bytes"] = None
    return sanitized


def with_thread_env(base_env: dict[str, str], threads: int) -> dict[str, str]:
    env = base_env.copy()
    for key in THREAD_ENV_KEYS:
        env[key] = str(threads)
    return env


def lcg_worker(iters: int) -> int:
    x = 0x12345678
    acc = 0
    mask = 0xFFFFFFFF
    for _ in range(iters):
        x = (1664525 * x + 1013904223) & mask
        acc = (acc + x) & mask
    return acc


def bench_cpu_scaling(max_workers: int, iters_per_worker: int, repeats: int) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    results: list[dict[str, Any]] = []

    for workers in range(1, max_workers + 1):
        with ctx.Pool(processes=workers) as pool:
            pool.map(lcg_worker, [iters_per_worker] * workers)
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                _ = pool.map(lcg_worker, [iters_per_worker] * workers)
                times.append(time.perf_counter() - t0)

        elapsed = statistics.median(times)
        total_ops = workers * iters_per_worker
        throughput = total_ops / elapsed
        results.append(
            {
                "workers": workers,
                "iters_per_worker": iters_per_worker,
                "median_seconds": elapsed,
                "throughput_iters_per_sec": throughput,
            }
        )

    base = results[0]["throughput_iters_per_sec"]
    for row in results:
        row["speedup_vs_1"] = row["throughput_iters_per_sec"] / base
        row["efficiency"] = row["speedup_vs_1"] / row["workers"]

    return results


MATMUL_SUBPROCESS_CODE = r"""
import json
import os
import statistics
import time
import numpy as np

seed = int(os.environ.get('MM_SEED', '0'))
size = int(os.environ['MM_SIZE'])
repeat = int(os.environ['MM_REPEAT'])
warmup = int(os.environ['MM_WARMUP'])

rng = np.random.default_rng(seed)
a = rng.standard_normal((size, size), dtype=np.float64)
b = rng.standard_normal((size, size), dtype=np.float64)

for _ in range(warmup):
    _ = a @ b

times = []
for _ in range(repeat):
    t0 = time.perf_counter()
    _ = a @ b
    times.append(time.perf_counter() - t0)

out = {
    'times': times,
    'median_seconds': statistics.median(times),
}
print(json.dumps(out))
"""


def run_matmul_case(threads: int, size: int, repeat: int, warmup: int, seed: int) -> dict[str, Any]:
    env = with_thread_env(os.environ, threads)

    env["MM_SIZE"] = str(size)
    env["MM_REPEAT"] = str(repeat)
    env["MM_WARMUP"] = str(warmup)
    env["MM_SEED"] = str(seed)

    proc = subprocess.run(
        [sys.executable, "-c", MATMUL_SUBPROCESS_CODE],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(proc.stdout)
    sec = float(payload["median_seconds"])
    flops = 2.0 * (size**3)
    gflops = flops / sec / 1e9

    return {
        "threads": threads,
        "size": size,
        "median_seconds": sec,
        "gflops": gflops,
    }


def bench_matmul(thread_list: list[int], sizes: list[int], repeat: int, warmup: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threads in thread_list:
        for size in sizes:
            rows.append(run_matmul_case(threads, size, repeat=repeat, warmup=warmup, seed=7))
    return rows


MATMUL_SUSTAINED_SUBPROCESS_CODE = r"""
import json
import os
import statistics
import time
import numpy as np

size = int(os.environ['MS_SIZE'])
iters = int(os.environ['MS_ITERS'])
window = int(os.environ['MS_WINDOW'])
warmup = int(os.environ['MS_WARMUP'])
seed = int(os.environ.get('MS_SEED', '11'))

rng = np.random.default_rng(seed)
a = rng.standard_normal((size, size), dtype=np.float64)
b = rng.standard_normal((size, size), dtype=np.float64)

for _ in range(warmup):
    _ = a @ b

samples = []
for _ in range(iters):
    t0 = time.perf_counter()
    _ = a @ b
    samples.append(time.perf_counter() - t0)

chunk_medians = []
for i in range(0, len(samples), window):
    chunk = samples[i:i + window]
    if len(chunk) == window:
        chunk_medians.append(statistics.median(chunk))

chunk_gflops = [2.0 * (size ** 3) / sec / 1e9 for sec in chunk_medians]

out = {
    'size': size,
    'iters': iters,
    'window': window,
    'chunk_gflops': chunk_gflops,
}
print(json.dumps(out))
"""


def bench_matmul_sustained(threads: int, size: int, iters: int, window: int, warmup: int) -> dict[str, Any]:
    env = with_thread_env(os.environ, threads)

    env["MS_SIZE"] = str(size)
    env["MS_ITERS"] = str(iters)
    env["MS_WINDOW"] = str(window)
    env["MS_WARMUP"] = str(warmup)
    env["MS_SEED"] = "11"

    proc = subprocess.run(
        [sys.executable, "-c", MATMUL_SUSTAINED_SUBPROCESS_CODE],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(proc.stdout)
    chunk_gflops = payload["chunk_gflops"]
    first = chunk_gflops[0]
    last = chunk_gflops[-1]
    edge_count = max(1, min(3, len(chunk_gflops) // 3))
    first_edge = statistics.median(chunk_gflops[:edge_count])
    last_edge = statistics.median(chunk_gflops[-edge_count:])
    drift_pct = (last_edge / first_edge - 1.0) * 100.0

    return {
        "threads": threads,
        "size": size,
        "iters": iters,
        "window": window,
        "windows": len(chunk_gflops),
        "first_window_gflops": first,
        "last_window_gflops": last,
        "first_edge_gflops": first_edge,
        "last_edge_gflops": last_edge,
        "min_window_gflops": min(chunk_gflops),
        "max_window_gflops": max(chunk_gflops),
        "drift_pct": drift_pct,
        "chunk_gflops": chunk_gflops,
    }


HYBRID_MATMUL_SUBPROCESS_CODE = r"""
import json
import os
import time
import numpy as np

size = int(os.environ["HB_SIZE"])
iters = int(os.environ["HB_ITERS"])
warmup = int(os.environ["HB_WARMUP"])
seed = int(os.environ["HB_SEED"])

rng = np.random.default_rng(seed)
a = rng.standard_normal((size, size), dtype=np.float64)
b = rng.standard_normal((size, size), dtype=np.float64)

for _ in range(warmup):
    _ = a @ b

t0 = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    c = a @ b
    checksum += float(c[0, 0])
elapsed = time.perf_counter() - t0

print(json.dumps({"elapsed": elapsed, "checksum": checksum}))
"""


def run_hybrid_combo(
    processes: int,
    threads: int,
    logical_cores: int,
    size: int,
    iters_per_process: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    throughput_samples = []
    wall_samples = []
    worker_median_samples = []
    checksum_total = 0.0

    flops_per_process = iters_per_process * (2.0 * (size**3))

    for rep in range(repeats):
        procs: list[subprocess.Popen[str]] = []
        t0 = time.perf_counter()

        for rank in range(processes):
            env = with_thread_env(os.environ, threads)
            env["HB_SIZE"] = str(size)
            env["HB_ITERS"] = str(iters_per_process)
            env["HB_WARMUP"] = str(warmup)
            env["HB_SEED"] = str(1000 + rep * 97 + rank)
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-c", HYBRID_MATMUL_SUBPROCESS_CODE],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )

        worker_elapsed = []
        for proc in procs:
            out, err = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"hybrid subprocess failed: {err.strip()}")
            payload = json.loads(out)
            worker_elapsed.append(float(payload["elapsed"]))
            checksum_total += float(payload["checksum"])

        wall = time.perf_counter() - t0
        total_flops = flops_per_process * processes
        throughput_gflops = total_flops / wall / 1e9

        throughput_samples.append(throughput_gflops)
        wall_samples.append(wall)
        worker_median_samples.append(statistics.median(worker_elapsed))

    oversub_ratio = (processes * threads) / max(1, logical_cores)

    return {
        "processes": processes,
        "threads": threads,
        "size": size,
        "iters_per_process": iters_per_process,
        "oversub_ratio": oversub_ratio,
        "throughput_gflops": statistics.median(throughput_samples),
        "wall_seconds": statistics.median(wall_samples),
        "worker_elapsed_seconds": statistics.median(worker_median_samples),
        "throughput_samples_gflops": throughput_samples,
        "checksum": checksum_total,
    }


def bench_hybrid_matmul(
    logical_cores: int,
    quick: bool,
) -> dict[str, Any]:
    if quick:
        size = 1024
        iters_per_process = 12
        warmup = 1
        repeats = 1
    else:
        size = 1536
        iters_per_process = 18
        warmup = 1
        repeats = 2

    process_candidates = sorted({1, 2, 3, 4, 6, logical_cores})
    process_candidates = [p for p in process_candidates if p <= logical_cores and p > 0]

    thread_candidates = sorted({1, 2, 4, 6, logical_cores})
    thread_candidates = [t for t in thread_candidates if t <= logical_cores and t > 0]

    rows: list[dict[str, Any]] = []
    for processes in process_candidates:
        for threads in thread_candidates:
            if processes * threads > 2 * logical_cores:
                continue
            rows.append(
                run_hybrid_combo(
                    processes=processes,
                    threads=threads,
                    logical_cores=logical_cores,
                    size=size,
                    iters_per_process=iters_per_process,
                    warmup=warmup,
                    repeats=repeats,
                )
            )

    rows_sorted = sorted(rows, key=lambda r: r["throughput_gflops"], reverse=True)
    best = rows_sorted[0]

    def find_combo(p: int, t: int) -> dict[str, Any] | None:
        for row in rows:
            if row["processes"] == p and row["threads"] == t:
                return row
        return None

    thread_only = find_combo(1, logical_cores)
    process_only = find_combo(logical_cores, 1)
    balanced = find_combo(min(logical_cores, 3), min(logical_cores, 2))

    reference_rows = [r for r in rows if r["oversub_ratio"] <= 1.0]
    best_no_oversub = max(reference_rows, key=lambda r: r["throughput_gflops"]) if reference_rows else None

    return {
        "config": {
            "size": size,
            "iters_per_process": iters_per_process,
            "warmup": warmup,
            "repeats": repeats,
            "logical_cores": logical_cores,
            "process_candidates": process_candidates,
            "thread_candidates": thread_candidates,
            "max_oversub_ratio": 2.0,
        },
        "results": rows,
        "best": best,
        "best_no_oversub": best_no_oversub,
        "reference": {
            "thread_only": thread_only,
            "process_only": process_only,
            "balanced_p3_t2": balanced,
        },
    }


def bench_stream_single(np_module: Any, n: int, repeats: int) -> list[dict[str, Any]]:
    np = np_module
    scalar = 3.0

    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64) * 2.0
    c = np.zeros(n, dtype=np.float64)

    operations = [
        ("copy", 16 * n, lambda: b.__setitem__(slice(None), a)),
        ("scale", 16 * n, lambda: b.__setitem__(slice(None), scalar * a)),
        ("add", 24 * n, lambda: c.__setitem__(slice(None), a + b)),
        ("triad", 24 * n, lambda: a.__setitem__(slice(None), b + scalar * c)),
    ]

    rows: list[dict[str, Any]] = []
    for name, moved_bytes, fn in operations:
        fn()
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)

        sec = statistics.median(samples)
        bw = moved_bytes / sec / 1e9
        rows.append(
            {
                "op": name,
                "bytes_moved": moved_bytes,
                "median_seconds": sec,
                "bandwidth_gbps": bw,
            }
        )

    return rows


def stream_worker(payload: tuple[int, int]) -> float:
    n, loops = payload
    import numpy as np

    scalar = 3.0
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64) * 2.0
    c = np.zeros(n, dtype=np.float64)

    t0 = time.perf_counter()
    for _ in range(loops):
        a[:] = b + scalar * c
    elapsed = time.perf_counter() - t0

    moved_bytes = loops * 24 * n
    return moved_bytes / elapsed / 1e9


def bench_stream_scaling(max_workers: int, n_per_worker: int, loops: int, repeats: int) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    rows: list[dict[str, Any]] = []

    warmup_loops = max(1, loops // 4)

    for workers in range(1, max_workers + 1):
        with ctx.Pool(processes=workers) as pool:
            _ = pool.map(stream_worker, [(n_per_worker, warmup_loops)] * workers)
            samples = []
            for _ in range(repeats):
                parts = pool.map(stream_worker, [(n_per_worker, loops)] * workers)
                samples.append(sum(parts))

        total_bw = statistics.median(samples)
        rows.append(
            {
                "workers": workers,
                "n_per_worker": n_per_worker,
                "loops": loops,
                "total_bandwidth_gbps": total_bw,
            }
        )

    return rows


def bench_cache_sweep(
    np_module: Any,
    min_pow: int,
    max_pow: int,
    target_bytes: int,
    samples_per_size: int,
) -> list[dict[str, Any]]:
    np = np_module
    rng = np.random.default_rng(17)

    rows: list[dict[str, Any]] = []
    for p in range(min_pow, max_pow + 1):
        size_bytes = 2**p
        n = max(1, size_bytes // 8)
        arr = rng.random(n, dtype=np.float64)

        reps = target_bytes // size_bytes
        reps = max(8, reps)
        reps = min(2048, reps)

        sample_bandwidths = []
        checksum = 0.0
        for _ in range(samples_per_size):
            _ = float(arr.sum())
            t0 = time.perf_counter()
            local_checksum = 0.0
            for _ in range(reps):
                local_checksum += float(arr.sum())
            elapsed = time.perf_counter() - t0
            sample_bandwidths.append((size_bytes * reps) / elapsed / 1e9)
            checksum += local_checksum

        bw = statistics.median(sample_bandwidths)
        rows.append(
            {
                "size_bytes": size_bytes,
                "reps": reps,
                "bandwidth_gbps": bw,
                "bandwidth_samples_gbps": sample_bandwidths,
                "checksum": checksum,
            }
        )

    return rows


def derive_metrics(
    system: dict[str, Any],
    cpu_scaling: list[dict[str, Any]],
    stream_scaling: list[dict[str, Any]],
    matmul: list[dict[str, Any]],
    stream_single: list[dict[str, Any]],
    matmul_sustained: dict[str, Any],
    hybrid_matmul: dict[str, Any],
) -> dict[str, Any]:
    perf = system.get("perf_cores")
    eff = system.get("eff_cores")

    cpu_by_workers = {r["workers"]: r["throughput_iters_per_sec"] for r in cpu_scaling}

    per_p_core = None
    per_e_core = None
    e_vs_p = None

    p_reference_workers = None
    p_reference_thr = None
    if perf:
        p_candidates = []
        for workers in range(1, perf + 1):
            if workers in cpu_by_workers:
                p_candidates.append((workers, cpu_by_workers[workers]))
        if p_candidates:
            p_reference_workers, p_reference_thr = max(p_candidates, key=lambda x: x[1])
            per_p_core = p_reference_thr / p_reference_workers

    if perf and eff and p_reference_thr and p_reference_workers and eff > 0:
        e_active_rows = []
        for workers in range(perf + 1, perf + eff + 1):
            if workers in cpu_by_workers:
                e_active_rows.append((workers, cpu_by_workers[workers]))

        if e_active_rows:
            best_workers, best_thr = max(e_active_rows, key=lambda x: x[1])
            added = best_workers - p_reference_workers
            if added > 0:
                e_inc = max(0.0, best_thr - p_reference_thr)
                per_e_core = e_inc / added
                if per_p_core and per_p_core > 0:
                    e_vs_p = per_e_core / per_p_core

    boundary_worker = None
    if len(cpu_scaling) >= 3:
        deltas = []
        for idx in range(1, len(cpu_scaling)):
            d = cpu_scaling[idx]["throughput_iters_per_sec"] - cpu_scaling[idx - 1]["throughput_iters_per_sec"]
            deltas.append((cpu_scaling[idx]["workers"], d))

        ref = deltas[0][1] if deltas[0][1] > 0 else None
        if ref:
            for workers, d in deltas[1:]:
                if d < 0.45 * ref:
                    boundary_worker = workers
                    break

    saturation_worker = None
    max_bw = None
    if stream_scaling:
        max_bw = max(r["total_bandwidth_gbps"] for r in stream_scaling)
        threshold = 0.95 * max_bw
        for row in stream_scaling:
            if row["total_bandwidth_gbps"] >= threshold:
                saturation_worker = row["workers"]
                break

    peak_gflops = max((r["gflops"] for r in matmul), default=None)
    triad_bw = None
    for row in stream_single:
        if row["op"] == "triad":
            triad_bw = row["bandwidth_gbps"]
            break

    roofline_knee = None
    if peak_gflops and triad_bw and triad_bw > 0:
        roofline_knee = peak_gflops / triad_bw

    hybrid_best = hybrid_matmul.get("best")
    hybrid_ref = hybrid_matmul.get("reference", {})

    hybrid_thread_only_penalty_pct = None
    hybrid_process_only_penalty_pct = None
    hybrid_balanced_penalty_pct = None

    if hybrid_best and hybrid_best.get("throughput_gflops"):
        best_thr = hybrid_best["throughput_gflops"]
        thread_only = hybrid_ref.get("thread_only")
        process_only = hybrid_ref.get("process_only")
        balanced = hybrid_ref.get("balanced_p3_t2")

        if thread_only and thread_only.get("throughput_gflops"):
            hybrid_thread_only_penalty_pct = 100.0 * (thread_only["throughput_gflops"] / best_thr - 1.0)
        if process_only and process_only.get("throughput_gflops"):
            hybrid_process_only_penalty_pct = 100.0 * (process_only["throughput_gflops"] / best_thr - 1.0)
        if balanced and balanced.get("throughput_gflops"):
            hybrid_balanced_penalty_pct = 100.0 * (balanced["throughput_gflops"] / best_thr - 1.0)

    return {
        "cpu_throughput_per_p_core_iters_per_sec": per_p_core,
        "cpu_throughput_per_e_core_iters_per_sec": per_e_core,
        "e_core_vs_p_core_ratio": e_vs_p,
        "cpu_scaling_knee_worker": boundary_worker,
        "memory_bandwidth_saturation_worker": saturation_worker,
        "memory_bandwidth_peak_gbps": max_bw,
        "matmul_peak_gflops": peak_gflops,
        "triad_bandwidth_gbps": triad_bw,
        "roofline_knee_flop_per_byte": roofline_knee,
        "sustained_matmul_drift_pct": matmul_sustained.get("drift_pct"),
        "hybrid_best_processes": hybrid_best.get("processes") if hybrid_best else None,
        "hybrid_best_threads": hybrid_best.get("threads") if hybrid_best else None,
        "hybrid_best_oversub_ratio": hybrid_best.get("oversub_ratio") if hybrid_best else None,
        "hybrid_best_throughput_gflops": hybrid_best.get("throughput_gflops") if hybrid_best else None,
        "hybrid_thread_only_penalty_pct": hybrid_thread_only_penalty_pct,
        "hybrid_process_only_penalty_pct": hybrid_process_only_penalty_pct,
        "hybrid_balanced_penalty_pct": hybrid_balanced_penalty_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M-series Python benchmark probe")
    parser.add_argument("--output", default="results/m_series_probe.json", help="output json path")
    parser.add_argument("--quick", action="store_true", help="run shorter benchmark")
    parser.add_argument("--public", action="store_true", help="redact host-identifying metadata in output")
    args = parser.parse_args()

    try:
        import numpy as np
    except Exception as exc:
        raise SystemExit(f"numpy is required: {exc}")

    system = collect_system_info()
    system["numpy_version"] = np.__version__
    if args.public:
        system = anonymize_system_info(system)
    logical = system.get("logical_cores") or os.cpu_count() or 8

    if args.quick:
        cpu_iters = 8_000_000
        cpu_repeats = 2
        matmul_sizes = [1024, 2048]
        matmul_repeats = 3
        matmul_warmup = 1
        sustained_size = 2048
        sustained_iters = 160
        sustained_window = 16
        stream_n = 16_000_000
        stream_repeats = 3
        stream_mp_n = 1_500_000
        stream_mp_loops = 10
        stream_mp_repeats = 2
        cache_min_pow = 15
        cache_max_pow = 27
        cache_target_bytes = 512 * 1024 * 1024
        cache_samples = 3
    else:
        cpu_iters = 16_000_000
        cpu_repeats = 3
        matmul_sizes = [1024, 2048, 3072]
        matmul_repeats = 5
        matmul_warmup = 1
        sustained_size = 3072
        sustained_iters = 320
        sustained_window = 32
        stream_n = 24_000_000
        stream_repeats = 5
        stream_mp_n = 2_000_000
        stream_mp_loops = 14
        stream_mp_repeats = 3
        cache_min_pow = 15
        cache_max_pow = 29
        cache_target_bytes = 1024 * 1024 * 1024
        cache_samples = 3

    thread_candidates = sorted({1, 2, 4, 6, logical})
    thread_candidates = [t for t in thread_candidates if t <= logical]

    print("[1/7] cpu scaling", file=sys.stderr)
    cpu_scaling = bench_cpu_scaling(logical, cpu_iters, cpu_repeats)

    print("[2/7] matmul", file=sys.stderr)
    matmul = bench_matmul(thread_candidates, matmul_sizes, matmul_repeats, matmul_warmup)

    print("[3/7] matmul sustained", file=sys.stderr)
    matmul_sustained = bench_matmul_sustained(
        threads=logical,
        size=sustained_size,
        iters=sustained_iters,
        window=sustained_window,
        warmup=matmul_warmup,
    )

    print("[4/7] hybrid matmul", file=sys.stderr)
    hybrid_matmul = bench_hybrid_matmul(logical_cores=logical, quick=args.quick)

    print("[5/7] stream single", file=sys.stderr)
    stream_single = bench_stream_single(np, stream_n, stream_repeats)

    print("[6/7] stream multiprocessing", file=sys.stderr)
    stream_scaling = bench_stream_scaling(logical, stream_mp_n, stream_mp_loops, stream_mp_repeats)

    print("[7/7] cache sweep", file=sys.stderr)
    cache_sweep = bench_cache_sweep(np, cache_min_pow, cache_max_pow, cache_target_bytes, cache_samples)

    derived = derive_metrics(
        system,
        cpu_scaling,
        stream_scaling,
        matmul,
        stream_single,
        matmul_sustained,
        hybrid_matmul,
    )

    payload = {
        "system": system,
        "config": {
            "quick": args.quick,
            "public": args.public,
            "cpu_iters": cpu_iters,
            "cpu_repeats": cpu_repeats,
            "matmul_sizes": matmul_sizes,
            "matmul_repeats": matmul_repeats,
            "sustained_size": sustained_size,
            "sustained_iters": sustained_iters,
            "sustained_window": sustained_window,
            "stream_n": stream_n,
            "stream_repeats": stream_repeats,
            "stream_mp_n": stream_mp_n,
            "stream_mp_loops": stream_mp_loops,
            "stream_mp_repeats": stream_mp_repeats,
            "cache_min_pow": cache_min_pow,
            "cache_max_pow": cache_max_pow,
            "cache_target_bytes": cache_target_bytes,
            "cache_samples": cache_samples,
            "hybrid_matmul_config": hybrid_matmul["config"],
        },
        "benchmarks": {
            "cpu_scaling": cpu_scaling,
            "matmul": matmul,
            "matmul_sustained": matmul_sustained,
            "hybrid_matmul": hybrid_matmul,
            "stream_single": stream_single,
            "stream_scaling": stream_scaling,
            "cache_sweep": cache_sweep,
        },
        "derived": derived,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
