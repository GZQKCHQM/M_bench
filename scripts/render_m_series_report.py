#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}%"


def fmt_signed_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    if value > 0:
        return f"+{value:.{digits}f}%"
    return f"{value:.{digits}f}%"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def pick_by_workers(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(r["workers"]): r for r in rows}


def render_report(data: dict[str, Any], public: bool = False) -> str:
    system = data["system"]
    benches = data["benchmarks"]
    derived = data["derived"]

    cpu_scaling = benches["cpu_scaling"]
    stream_scaling = benches["stream_scaling"]
    stream_single = benches["stream_single"]
    matmul = benches["matmul"]
    sustained = benches["matmul_sustained"]
    cache = benches["cache_sweep"]
    hybrid = benches.get("hybrid_matmul", {})
    hybrid_rows = hybrid.get("results", [])
    hybrid_sorted = sorted(hybrid_rows, key=lambda r: r["throughput_gflops"], reverse=True)
    hybrid_best = hybrid_sorted[0] if hybrid_sorted else None
    hybrid_best_no_oversub = hybrid.get("best_no_oversub")
    hybrid_reference = hybrid.get("reference", {})
    hybrid_thread_only = hybrid_reference.get("thread_only")
    hybrid_process_only = hybrid_reference.get("process_only")
    hybrid_balanced = hybrid_reference.get("balanced_p3_t2")

    perf_cores = int(system.get("perf_cores") or 0)
    eff_cores = int(system.get("eff_cores") or 0)
    total_cores = int(system.get("logical_cores") or 0)

    cpu_by_w = pick_by_workers(cpu_scaling)
    p_thr = cpu_by_w.get(perf_cores, {}).get("throughput_iters_per_sec")
    all_thr = cpu_by_w.get(perf_cores + eff_cores, {}).get("throughput_iters_per_sec")

    e_gain_pct = None
    if p_thr and all_thr and p_thr > 0:
        e_gain_pct = 100.0 * (all_thr - p_thr) / p_thr

    sizes = sorted({int(r["size"]) for r in matmul})
    largest_size = max(sizes)
    largest_matmul = sorted([r for r in matmul if int(r["size"]) == largest_size], key=lambda r: int(r["threads"]))
    largest_peak = max(r["gflops"] for r in largest_matmul)

    near_peak_threads = None
    for row in largest_matmul:
        if row["gflops"] >= 0.98 * largest_peak:
            near_peak_threads = int(row["threads"])
            break

    matmul_2 = next((r["gflops"] for r in largest_matmul if int(r["threads"]) == 2), None)
    matmul_8 = next((r["gflops"] for r in largest_matmul if int(r["threads"]) == total_cores), None)
    matmul_gain_2_to_all = None
    if matmul_2 and matmul_8 and matmul_2 > 0:
        matmul_gain_2_to_all = 100.0 * (matmul_8 - matmul_2) / matmul_2

    peak_bw = max(r["total_bandwidth_gbps"] for r in stream_scaling)
    sat_worker = int(derived.get("memory_bandwidth_saturation_worker") or 0)

    cache_plateau = [r["bandwidth_gbps"] for r in cache if int(r["size_bytes"]) >= 16 * 1024 * 1024]
    cache_plateau_med = statistics.median(cache_plateau) if cache_plateau else None

    hybrid_table = md_table(
        ["Rank", "Proc", "Threads", "Oversub", "Throughput (GFLOPS)"],
        [
            [
                str(i + 1),
                str(row["processes"]),
                str(row["threads"]),
                fmt_float(row["oversub_ratio"], 2),
                fmt_float(row["throughput_gflops"], 1),
            ]
            for i, row in enumerate(hybrid_sorted[:12])
        ],
    )

    stream_single_table = md_table(
        ["Kernel", "Bandwidth (GB/s)"],
        [[r["op"], fmt_float(r["bandwidth_gbps"], 2)] for r in stream_single],
    )

    cpu_table = md_table(
        ["Workers", "Throughput (M it/s)", "Speedup", "Efficiency"],
        [
            [
                str(r["workers"]),
                fmt_float(r["throughput_iters_per_sec"] / 1e6, 2),
                fmt_float(r["speedup_vs_1"], 2),
                fmt_float(r["efficiency"], 2),
            ]
            for r in cpu_scaling
        ],
    )

    stream_scaling_table = md_table(
        ["Workers", "Total BW (GB/s)"],
        [[str(r["workers"]), fmt_float(r["total_bandwidth_gbps"], 2)] for r in stream_scaling],
    )

    matmul_rows = []
    for size in sizes:
        size_rows = sorted([r for r in matmul if int(r["size"]) == size], key=lambda r: int(r["threads"]))
        for r in size_rows:
            matmul_rows.append([
                str(size),
                str(r["threads"]),
                fmt_float(r["gflops"], 1),
                fmt_float(r["median_seconds"], 5),
            ])
    matmul_table = md_table(["N", "Threads", "GFLOPS", "Median sec"], matmul_rows)

    cache_focus_sizes = [
        32 * 1024,
        64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1024 * 1024,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
        512 * 1024 * 1024,
    ]
    cache_rows = []
    for size in cache_focus_sizes:
        row = next((r for r in cache if int(r["size_bytes"]) == size), None)
        if row is None:
            continue
        cache_rows.append([
            f"{size // 1024} KiB" if size < 1024 * 1024 else f"{size // (1024 * 1024)} MiB",
            fmt_float(row["bandwidth_gbps"], 2),
        ])
    cache_table = md_table(["Working set", "Read BW median (GB/s)"], cache_rows)

    generated = "public release (timestamp omitted)" if public else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_value = "redacted" if public else str(system.get("model", "-"))
    os_value = "redacted" if public else str(system.get("os", "-"))
    python_value = str(system.get("python", "-")).split(" ")[0]

    lines = [
        "# M1 Pro Reverse Engineering Report (Python Workloads)",
        "",
        f"Generated: {generated}",
        "",
        "## 0. 要約",
        "",
        f"- このM1 Proは`{perf_cores}P+{eff_cores}E`構成で、CPUスカラー処理は`{perf_cores}`ワーカーまで高い線形性を維持し、`{perf_cores + eff_cores}`ワーカー時の上積みは`{fmt_pct(e_gain_pct, 2)}`でした。",
        f"- Eコアの単コア寄与はPコア比`{fmt_pct((derived.get('e_core_vs_p_core_ratio') or 0) * 100.0, 2)}`で、混在時の実効増分は限定的です。",
        f"- メモリ帯域はプロセス並列で`{sat_worker}`ワーカー付近でほぼ飽和し、ピークは`{fmt_float(peak_bw, 2)} GB/s`でした。",
        f"- 大規模GEMM（N={largest_size}）は`{near_peak_threads}`スレッドで既にピークの98%に到達し、2スレッドから全コアまでの差は`{fmt_pct(matmul_gain_2_to_all, 2)}`でした。",
        f"- 連続負荷（N={sustained['size']}, {sustained['iters']} iter）でのドリフトは`{fmt_pct(sustained['drift_pct'], 3)}`で、持続性能の安定性は高いです。",
        f"- ハイブリッド並列（processes x threads）の最速は`{derived.get('hybrid_best_processes')} x {derived.get('hybrid_best_threads')}`で、thread-only比`{fmt_signed_pct(derived.get('hybrid_thread_only_penalty_pct'), 2)}`、process-only比`{fmt_signed_pct(derived.get('hybrid_process_only_penalty_pct'), 2)}`でした。",
        "",
        "## 1. 評価設計（Mシリーズ共通軸）",
        "",
        "このレポートは単機種の生値だけで終わらせず、Mシリーズ共通で比較できる軸を先に定義して計測しています。",
        "",
        "- CPU P-core unit: `P-core数時の総スループット / P-core数`",
        "- E/P ratio: `E-core単コア寄与 / P-core単コア寄与`",
        "- Memory saturation worker: メモリ帯域が95%ピークに達する最小ワーカー数",
        "- Roofline knee: `DGEMMピークGFLOPS / Triad帯域GB/s`",
        "- Sustained drift: 長時間連続計算の先頭窓と末尾窓の性能差",
        "",
        "## 2. 実行環境",
        "",
        md_table(
            ["Item", "Value"],
            [
                ["Chip", str(system.get("chip", "-"))],
                ["Model", model_value],
                ["Memory", f"{fmt_float(system.get('mem_gb'), 1)} GB"],
                ["Core layout", f"{perf_cores}P + {eff_cores}E (total {total_cores})"],
                ["OS", os_value],
                ["Python", python_value],
                ["NumPy", str(system.get("numpy_version", "-"))],
            ],
        ),
        "",
        "## 3. 計測結果",
        "",
        "### 3.1 CPUスケーリング（純Python, LCG）",
        "",
        cpu_table,
        "",
        "### 3.2 DGEMM（NumPy/Accelerate）",
        "",
        matmul_table,
        "",
        "### 3.3 STREAM系メモリカーネル（単一プロセス）",
        "",
        stream_single_table,
        "",
        "### 3.4 メモリ帯域スケーリング（マルチプロセス）",
        "",
        stream_scaling_table,
        "",
        "### 3.5 キャッシュ/ワーキングセット遷移（read sum）",
        "",
        cache_table,
        "",
        "### 3.6 ハイブリッド並列（processes x BLAS threads）",
        "",
        hybrid_table,
        "",
        "## 4. 逆解析で得られた知見",
        "",
        f"1. P/E混在の境界は`{derived.get('cpu_scaling_knee_worker')}`ワーカー付近で、`{perf_cores}`を超えると効率が急減します。",
        f"2. Eコアを有効化した増分は`{fmt_pct(e_gain_pct, 2)}`で、重い数値計算の主戦力はPコアです。",
        f"3. DGEMMはN={largest_size}で`{near_peak_threads}`スレッド時点で実質飽和し、過度なスレッド増加の利益は小さいです。",
        f"4. メモリ帯域は`{sat_worker}`ワーカーで95%ピーク到達、以降は競合で上下するためメモリ律速処理の並列数は抑える方が安定します。",
        f"5. 持続性能ドリフトが`{fmt_pct(sustained['drift_pct'], 3)}`に収まっており、長時間計算でも性能降下は観測されませんでした。",
        f"6. Roofline kneeは約`{fmt_float(derived.get('roofline_knee_flop_per_byte'), 2)} FLOP/byte`で、これ未満の演算密度は主に帯域制約を受けます。",
        f"7. 独立ジョブを同時実行する系では`{derived.get('hybrid_best_processes')} x {derived.get('hybrid_best_threads')}`が最速で、thread-only構成との差は`{fmt_signed_pct(derived.get('hybrid_thread_only_penalty_pct'), 2)}`でした。",
        "",
        "## 5. Pythonシミュレーションへの適用指針",
        "",
        f"1. 純Python主体の並列は`{perf_cores}`ワーカーを基本値にし、スループット最優先時のみ`{total_cores}`へ拡張する。",
        f"2. BLAS中心ワークロードは`VECLIB_MAXIMUM_THREADS={near_peak_threads}`から探索開始する。",
        f"3. メモリ律速処理は同時ワーカー数を`{sat_worker}`前後に抑え、プロセスを増やしすぎない。",
        "4. 1タスクの演算密度を上げるため、ループ融合・バッチ化・不要な中間配列削減を優先する。",
        "5. 連続運用時は性能低下よりもメモリ競合が支配的なので、まず並列度の最適化を行う。",
        f"6. 独立ジョブ並列では`{derived.get('hybrid_best_processes')} x {derived.get('hybrid_best_threads')}`を第一候補にする。",
        f"7. thread-only構成は最適構成比`{fmt_signed_pct(derived.get('hybrid_thread_only_penalty_pct'), 2)}`の差が出たため、常用設定にしない。",
        "",
        "## 6. このレポートが役立つ読者",
        "",
        md_table(
            ["読者", "使いどころ", "得られる意思決定"],
            [
                ["Pythonシミュレーション開発者", "NumPy/BLASの並列設定", "処理時間を縮める初期設定を即決できる"],
                ["Intel/Mac移行検討者", "移行前の実効性能見積もり", "CPU律速か帯域律速かで期待値を分けられる"],
                ["研究室・チーム運用者", "再現可能な性能ベースライン管理", "機材更新時の比較を同一指標で継続できる"],
            ],
        ),
        "",
        "## 7. 他Mチップへの横展開手順",
        "",
        "同じスクリプトをそのまま実行し、次の比較を取ると機種差を公平に評価できます。",
        "",
        "- `CPU P-core unit`の比較: 単コア群の純粋な世代差",
        "- `E/P ratio`の比較: Eコアの実効価値",
        "- `Memory saturation worker`と`peak BW`の比較: 帯域と並列競合耐性",
        "- `Roofline knee`の比較: どの演算密度でメモリ律速から脱出できるか",
        "- `Sustained drift`の比較: 長時間運用の安定性",
        "",
        "推定に使う近似式（実務での初期見積もり用）:",
        "",
        "- メモリ律速処理の速度比 `~ (target_peak_bw / current_peak_bw)`",
        "- CPU律速処理の速度比 `~ (target_p_core_unit / current_p_core_unit)`",
        "- 混合処理の速度比 `~ min(上記2つの上限)`",
        "",
        "## 8. 再現コマンド",
        "",
        "```bash",
        "python3 -m venv .venv",
        ". .venv/bin/activate",
        "python -m pip install -U pip setuptools wheel",
        "python -m pip install numpy psutil",
        "python scripts/m_series_probe.py --output results/m_series_probe_full.json",
        "python scripts/render_m_series_report.py --input results/m_series_probe_full.json --output reports/m1_pro_reverse_engineering_report.md",
        "```",
        "",
        "## 9. 主要指標サマリー",
        "",
        md_table(
            ["Metric", "Value"],
            [
                ["CPU P-core unit (M it/s)", fmt_float((derived.get("cpu_throughput_per_p_core_iters_per_sec") or 0) / 1e6, 2)],
                ["E/P ratio", fmt_float((derived.get("e_core_vs_p_core_ratio") or 0), 3)],
                ["CPU knee worker", str(derived.get("cpu_scaling_knee_worker", "-"))],
                ["Memory peak BW (GB/s)", fmt_float(derived.get("memory_bandwidth_peak_gbps"), 2)],
                ["Memory saturation worker", str(derived.get("memory_bandwidth_saturation_worker", "-"))],
                ["DGEMM peak (GFLOPS)", fmt_float(derived.get("matmul_peak_gflops"), 1)],
                ["Roofline knee (FLOP/byte)", fmt_float(derived.get("roofline_knee_flop_per_byte"), 2)],
                ["Sustained drift", fmt_pct(derived.get("sustained_matmul_drift_pct"), 3)],
                ["Hybrid best (proc x thread)", f"{derived.get('hybrid_best_processes')} x {derived.get('hybrid_best_threads')}"],
                ["Hybrid best throughput (GFLOPS)", fmt_float(derived.get("hybrid_best_throughput_gflops"), 1)],
                ["Thread-only delta vs best", fmt_signed_pct(derived.get("hybrid_thread_only_penalty_pct"), 2)],
                ["Process-only delta vs best", fmt_signed_pct(derived.get("hybrid_process_only_penalty_pct"), 2)],
                ["Cache plateau BW >=16MiB (GB/s)", fmt_float(cache_plateau_med, 2)],
            ],
        ),
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render markdown report from benchmark json")
    parser.add_argument("--input", default="results/m_series_probe_full.json", help="input json")
    parser.add_argument(
        "--output",
        default="reports/m1_pro_reverse_engineering_report.md",
        help="output markdown",
    )
    parser.add_argument("--public", action="store_true", help="omit timestamp and host-identifying metadata")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    text = render_report(data, public=args.public)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
