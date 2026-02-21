# 🍎 m-bench

> Know your Apple Silicon's true capabilities — measured, not guessed.

A benchmark suite tailored for Python/NumPy workloads, automatically generating **cross-machine comparable metrics** across the M-series lineup.  
"How many threads is optimal?" "Memory-bound or compute-bound?" — answered with real numbers.

---

## Motivation

Apple Silicon has a unique core architecture where conventional tuning wisdom doesn't always apply.  
The asymmetry between P-cores and E-cores, and the interference between BLAS threads and multiprocessing, can't be accurately understood without actual measurement.

This project exists to help you **stop guessing thread counts** and start deriving optimal settings from real hardware data.

---

## What You'll Learn

| Benchmark | Metrics | Use Case |
|---|---|---|
| `cpu_scaling` | Throughput / Speedup / Parallel efficiency | How far does scaling actually remain linear? |
| `matmul` | BLAS threads × matrix size → GFLOPS | Choosing the right `VECLIB_MAXIMUM_THREADS` |
| `stream_scaling` | Peak bandwidth / Saturation worker count | Upper limit on processes for memory-bound tasks |
| `matmul_sustained` | Performance drift over long runs | Stability validation for production workloads |
| `hybrid_matmul` | Full sweep of `processes × threads` | Optimal config for concurrent job execution |

---

## Benchmark Highlights (M1 Pro / 16GB)

```
Target: Apple M1 Pro (6P+2E), Python 3.13.12, NumPy 2.4.2

CPU
  P-cores:              Strong linear scaling up to 6 workers
  Gain at 8 workers:    +14.31%
  E/P core ratio:       42.94%

Memory
  Peak bandwidth:       63.63 GB/s
  95% saturation at:    3 workers

BLAS
  DGEMM (N=3072):       Nearly saturated at 2 threads

Stability
  Sustained drift:      0.844%

Hybrid Parallelism
  Best config:          3 × 2  (processes × threads)
  thread-only (1×8):    −20.46% vs. best
  process-only (8×1):   −11.69% vs. best
```

Full report → [`reports/m1_pro_reverse_engineering_report.md`](reports/m1_pro_reverse_engineering_report.md)

---

## Who This Is For

- **Developers running Python simulations with NumPy**  
  Replace gut-feel thread settings with evidence-based tuning.

- **Engineers migrating from Intel to Apple Silicon**  
  Quantify whether your workload is CPU-bound or memory-bound to accurately estimate migration gains.

- **Anyone running multiple compute jobs concurrently**  
  Find the `processes × threads` sweet spot to maximize overall throughput.

- **Labs or teams doing ongoing hardware comparisons**  
  Use the same metrics across generations to make consistent, data-driven upgrade decisions.

---

## Repository Structure

```
m_mac_chip/
├── scripts/
│   ├── m_series_probe.py          # Core benchmark runner
│   └── render_m_series_report.py  # JSON → Markdown report generator
├── results/
│   ├── m_series_probe_full.json   # Full measurement output (machine-readable)
│   └── m_series_probe_quick.json  # Quick measurement output
└── reports/
    └── m1_pro_reverse_engineering_report.md
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install numpy psutil
```

**Requirements:** macOS (Apple Silicon) / Python 3.10+

---

## Usage

### Full Benchmark

```bash
source .venv/bin/activate
python scripts/m_series_probe.py --public --output results/m_series_probe_full.json
```

### Generate Report

```bash
python scripts/render_m_series_report.py \
  --input  results/m_series_probe_full.json \
  --public \
  --output reports/m1_pro_reverse_engineering_report.md
```

### Quick Run (Sanity Check)

```bash
python scripts/m_series_probe.py --quick --public --output results/m_series_probe_quick.json
```

---

## Comparing Across Machines

Run the same scripts on any M-series machine and line up the numbers.

```bash
# Run on the target machine
python scripts/m_series_probe.py --public --output results/m_series_probe_m3max.json
python scripts/render_m_series_report.py --input results/m_series_probe_m3max.json \
  --public --output reports/m3max_report.md
```

**Key metrics to compare:**

| Metric | Description |
|---|---|
| `CPU P-core unit` | Baseline throughput of the P-core cluster |
| `E/P ratio` | Single-core contribution ratio: E-core vs. P-core |
| `Memory peak BW` | Peak memory bandwidth |
| `Memory saturation worker` | Minimum workers to reach 95% of peak bandwidth |
| `Roofline knee (FLOP/byte)` | Boundary between compute-bound and memory-bound regimes |
| `Sustained drift` | Performance variation over extended runs |
| `Hybrid best (proc × thread)` | Fastest config for concurrent workloads |

**Speed ratio approximations:**

```
Memory-bound workloads: speedup ≈ target_peak_bw / current_peak_bw
CPU-bound workloads:    speedup ≈ target_p_core_unit / current_p_core_unit
Mixed workloads:        speedup ≈ min(the two limits above)
```

---

## Practical Tuning Starting Points

Use your measurement results to tune in this order:

1. **Start BLAS thread exploration at `2`** — this is optimal or near-optimal in most cases
2. **For memory-bound workloads, begin adjusting around `workers ≈ 3`**
3. **For concurrent independent jobs, try `3 × 2` as your first candidate**
4. **Avoid locking into thread-only mode — always explore `processes × threads`**

---

## Reproducibility Checklist

- Run on AC power
- Minimize background processes
- Use the same Python / NumPy versions across machines
- Don't mix `--quick` and `full` results

---

## Limitations & Out of Scope

- Not designed for CUDA-based workload comparisons
- PyTorch / JAX-specific optimizations are not measured
- Absolute performance is affected by OS state, thermals, and concurrent load

---

## License

MIT

---

*If you have results from your own M-series machine, feel free to share them via a PR or Issue — more data makes the comparisons richer.*



# 🍎 m-bench

> Apple Silicon の「実力」を、推測ではなく実測で知るためのベンチマークスイート。

Python/NumPy ワークロードに特化し、M シリーズ間で横断比較できる **共通指標** を自動生成します。  
「スレッド数いくつが最適？」「メモリ律速？CPU律速？」—— そういった疑問に、数字で答えます。

---

## なぜ作ったのか

Apple Silicon はコア構成が独特で、一般的なチューニング知識がそのまま通用しないことがあります。  
特に P コアと E コアの非対称性、BLAS スレッドと multiprocessing の干渉は、実測しなければ正確に把握できません。

このプロジェクトは「感覚でスレッド数を決める」ことをやめ、**実機のデータから最適設定を導く** ためのツールです。

---

## 何がわかるか

| 測定項目 | 指標 | 活用シーン |
|---|---|---|
| `cpu_scaling` | スループット / スピードアップ / 並列効率 | 「何並列まで素直に伸びるか」の判断 |
| `matmul` | BLAS スレッド × 行列サイズ → GFLOPS | `VECLIB_MAXIMUM_THREADS` の初期値決定 |
| `stream_scaling` | 帯域ピーク / 飽和ワーカー数 | メモリ律速処理でのプロセス数上限 |
| `matmul_sustained` | 長時間実行中の性能ドリフト | 本番運用時の安定性確認 |
| `hybrid_matmul` | `processes × threads` の全探索 | 複数ジョブ同時実行の最適構成 |

---

## 実測ハイライト（M1 Pro / 16GB）

```
対象: Apple M1 Pro (6P+2E), Python 3.13.12, NumPy 2.4.2

CPU
  P コア群: 6 ワーカーまで高い線形スケーリング
  8 ワーカー時の上積み: +14.31%
  E/P コア寄与比: 42.94%

メモリ
  ピーク帯域:       63.63 GB/s
  95% 飽和ワーカー: 3

BLAS
  DGEMM (N=3072): 2 スレッドでほぼ飽和

安定性
  持続性能ドリフト: 0.844%

ハイブリッド並列
  最速構成:           3 × 2  (processes × threads)
  thread-only (1×8):  最適比 −20.46%
  process-only (8×1): 最適比 −11.69%
```

詳細レポート → [`reports/m1_pro_reverse_engineering_report.md`](reports/m1_pro_reverse_engineering_report.md)

---

## こんな人に刺さります

- **NumPy シミュレーションを書いている開発者**  
  並列設定を推測から実測ベースに切り替えられます。

- **Intel → Apple Silicon の移行を検討している人**  
  CPU 律速 / メモリ律速を切り分けて、移行効果の見積もりを定量化できます。

- **複数の計算ジョブを同時に走らせている人**  
  `processes × threads` の最適点を見つけ、スループットを最大化できます。

- **研究室・チームで機材を継続比較したい人**  
  世代をまたいで同じ指標で比較できるため、更新判断の根拠になります。

---

## リポジトリ構成

```
m_mac_chip/
├── scripts/
│   ├── m_series_probe.py          # ベンチマーク本体
│   └── render_m_series_report.py  # JSON → Markdown レポート生成
├── results/
│   ├── m_series_probe_full.json   # フル計測結果（機械処理向け）
│   └── m_series_probe_quick.json  # 短縮計測結果
└── reports/
    └── m1_pro_reverse_engineering_report.md
```

---

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install numpy psutil
```

**必要環境:** macOS (Apple Silicon) / Python 3.10+

---

## 使い方

### フル計測

```bash
source .venv/bin/activate
python scripts/m_series_probe.py --public --output results/m_series_probe_full.json
```

### レポート生成

```bash
python scripts/render_m_series_report.py \
  --input  results/m_series_probe_full.json \
  --public \
  --output reports/m1_pro_reverse_engineering_report.md
```

### 動作確認（短縮版）

```bash
python scripts/m_series_probe.py --quick --public --output results/m_series_probe_quick.json
```

---

## 機種間の比較方法

別の M シリーズ機で同じスクリプトを実行し、以下の指標を並べるだけです。

```bash
# 比較したい機種で実行
python scripts/m_series_probe.py --public --output results/m_series_probe_m3max.json
python scripts/render_m_series_report.py --input results/m_series_probe_m3max.json \
  --public --output reports/m3max_report.md
```

**比較する指標:**

| 指標 | 意味 |
|---|---|
| `CPU P-core unit` | P コア群の基礎スループット |
| `E/P ratio` | E コアと P コアの単コア性能比 |
| `Memory peak BW` | メモリ帯域ピーク |
| `Memory saturation worker` | 帯域 95% に到達する最小ワーカー数 |
| `Roofline knee (FLOP/byte)` | 計算律速と帯域律速の境界 |
| `Sustained drift` | 長時間実行時の性能変動率 |
| `Hybrid best (proc × thread)` | 同時実行時の最速構成 |

**速度比の近似式:**

```
メモリ律速処理: 速度比 ≈ target_peak_bw / current_peak_bw
CPU 律速処理:  速度比 ≈ target_p_core_unit / current_p_core_unit
混合処理:      速度比 ≈ min(上記 2 つの上限)
```

---

## 実務チューニングの出発点

計測結果をもとに、以下の順序で設定を詰めることを推奨します。

1. **BLAS スレッドは `2` から探索を始める**（大半のケースでこれが最適または近い）
2. **メモリ律速系は `workers ≈ 3` 前後から調整する**
3. **独立ジョブを並列実行する場合は `3 × 2` を第一候補にする**
4. **thread-only 固定は避け、`processes × threads` を必ず探索する**

---

## 計測の再現性を高めるために

- AC 電源に接続した状態で計測する
- バックグラウンドアプリを極力終了する
- Python / NumPy のバージョンを機種間で揃える
- `--quick` と `full` を混在させない

---

## 制約・スコープ外

- CUDA ワークロードの比較には対応していません
- PyTorch / JAX 固有の最適化は測定対象外です
- 絶対性能は OS 状態・温度・同時負荷の影響を受けます

---

## ライセンス

MIT

---

*同じ機種の結果をお持ちの方は、PR や Issue でシェアいただけると比較データが充実します。*



