# M1 Pro Reverse Engineering Report (Python Workloads)

Generated: public release (timestamp omitted)

## 0. 要約

- このM1 Proは`6P+2E`構成で、CPUスカラー処理は`6`ワーカーまで高い線形性を維持し、`8`ワーカー時の上積みは`5.06%`でした。
- Eコアの単コア寄与はPコア比`15.19%`で、混在時の実効増分は限定的です。
- メモリ帯域はプロセス並列で`5`ワーカー付近でほぼ飽和し、ピークは`60.86 GB/s`でした。
- 大規模GEMM（N=3072）は`2`スレッドで既にピークの98%に到達し、2スレッドから全コアまでの差は`-0.19%`でした。
- 連続負荷（N=3072, 320 iter）でのドリフトは`-0.155%`で、持続性能の安定性は高いです。
- ハイブリッド並列（processes x threads）の最速は`3 x 2`で、thread-only比`-20.98%`、process-only比`-11.78%`でした。

## 1. 評価設計（Mシリーズ共通軸）

このレポートは単機種の生値だけで終わらせず、Mシリーズ共通で比較できる軸を先に定義して計測しています。

- CPU P-core unit: `P-core数時の総スループット / P-core数`
- E/P ratio: `E-core単コア寄与 / P-core単コア寄与`
- Memory saturation worker: メモリ帯域が95%ピークに達する最小ワーカー数
- Roofline knee: `DGEMMピークGFLOPS / Triad帯域GB/s`
- Sustained drift: 長時間連続計算の先頭窓と末尾窓の性能差

## 2. 実行環境

| Item | Value |
| --- | --- |
| Chip | Apple M1 Pro |
| Model | redacted |
| Memory | 16.0 GB |
| Core layout | 6P + 2E (total 8) |
| OS | redacted |
| Python | 3.13 |
| NumPy | 2.4.2 |

## 3. 計測結果

### 3.1 CPUスケーリング（純Python, LCG）

| Workers | Throughput (M it/s) | Speedup | Efficiency |
| --- | --- | --- | --- |
| 1 | 8.50 | 1.00 | 1.00 |
| 2 | 17.24 | 2.03 | 1.01 |
| 3 | 25.45 | 2.99 | 1.00 |
| 4 | 33.00 | 3.88 | 0.97 |
| 5 | 40.74 | 4.79 | 0.96 |
| 6 | 48.21 | 5.67 | 0.94 |
| 7 | 49.23 | 5.79 | 0.83 |
| 8 | 50.65 | 5.96 | 0.74 |

### 3.2 DGEMM（NumPy/Accelerate）

| N | Threads | GFLOPS | Median sec |
| --- | --- | --- | --- |
| 1024 | 1 | 271.1 | 0.00792 |
| 1024 | 2 | 448.3 | 0.00479 |
| 1024 | 4 | 396.3 | 0.00542 |
| 1024 | 6 | 488.3 | 0.00440 |
| 1024 | 8 | 455.1 | 0.00472 |
| 2048 | 1 | 225.3 | 0.07625 |
| 2048 | 2 | 417.8 | 0.04112 |
| 2048 | 4 | 420.7 | 0.04084 |
| 2048 | 6 | 435.3 | 0.03946 |
| 2048 | 8 | 430.9 | 0.03987 |
| 3072 | 1 | 241.6 | 0.24000 |
| 3072 | 2 | 466.0 | 0.12441 |
| 3072 | 4 | 469.0 | 0.12364 |
| 3072 | 6 | 461.3 | 0.12569 |
| 3072 | 8 | 465.1 | 0.12466 |

### 3.3 STREAM系メモリカーネル（単一プロセス）

| Kernel | Bandwidth (GB/s) |
| --- | --- |
| copy | 83.07 |
| scale | 34.57 |
| add | 40.11 |
| triad | 30.80 |

### 3.4 メモリ帯域スケーリング（マルチプロセス）

| Workers | Total BW (GB/s) |
| --- | --- |
| 1 | 32.30 |
| 2 | 51.70 |
| 3 | 54.10 |
| 4 | 56.61 |
| 5 | 60.86 |
| 6 | 55.20 |
| 7 | 57.24 |
| 8 | 58.56 |

### 3.5 キャッシュ/ワーキングセット遷移（read sum）

| Working set | Read BW median (GB/s) |
| --- | --- |
| 32 KiB | 24.28 |
| 64 KiB | 31.23 |
| 128 KiB | 35.47 |
| 256 KiB | 37.45 |
| 512 KiB | 40.68 |
| 1 MiB | 41.34 |
| 2 MiB | 42.00 |
| 4 MiB | 42.98 |
| 8 MiB | 41.64 |
| 16 MiB | 41.12 |
| 64 MiB | 38.27 |
| 256 MiB | 40.84 |
| 512 MiB | 40.58 |

### 3.6 ハイブリッド並列（processes x BLAS threads）

| Rank | Proc | Threads | Oversub | Throughput (GFLOPS) |
| --- | --- | --- | --- | --- |
| 1 | 3 | 2 | 0.75 | 406.5 |
| 2 | 3 | 4 | 1.50 | 404.2 |
| 3 | 2 | 6 | 1.50 | 393.1 |
| 4 | 2 | 4 | 1.00 | 392.3 |
| 5 | 4 | 4 | 2.00 | 391.0 |
| 6 | 2 | 2 | 0.50 | 390.8 |
| 7 | 2 | 8 | 2.00 | 389.7 |
| 8 | 4 | 2 | 1.00 | 378.0 |
| 9 | 4 | 1 | 0.50 | 363.8 |
| 10 | 6 | 2 | 1.50 | 358.9 |
| 11 | 8 | 1 | 1.00 | 358.7 |
| 12 | 8 | 2 | 2.00 | 355.1 |

## 4. 逆解析で得られた知見

1. P/E混在の境界は`7`ワーカー付近で、`6`を超えると効率が急減します。
2. Eコアを有効化した増分は`5.06%`で、重い数値計算の主戦力はPコアです。
3. DGEMMはN=3072で`2`スレッド時点で実質飽和し、過度なスレッド増加の利益は小さいです。
4. メモリ帯域は`5`ワーカーで95%ピーク到達、以降は競合で上下するためメモリ律速処理の並列数は抑える方が安定します。
5. 持続性能ドリフトが`-0.155%`に収まっており、長時間計算でも性能降下は観測されませんでした。
6. Roofline kneeは約`15.86 FLOP/byte`で、これ未満の演算密度は主に帯域制約を受けます。
7. 独立ジョブを同時実行する系では`3 x 2`が最速で、thread-only構成との差は`-20.98%`でした。

## 5. Pythonシミュレーションへの適用指針

1. 純Python主体の並列は`6`ワーカーを基本値にし、スループット最優先時のみ`8`へ拡張する。
2. BLAS中心ワークロードは`VECLIB_MAXIMUM_THREADS=2`から探索開始する。
3. メモリ律速処理は同時ワーカー数を`5`前後に抑え、プロセスを増やしすぎない。
4. 1タスクの演算密度を上げるため、ループ融合・バッチ化・不要な中間配列削減を優先する。
5. 連続運用時は性能低下よりもメモリ競合が支配的なので、まず並列度の最適化を行う。
6. 独立ジョブ並列では`3 x 2`を第一候補にする。
7. thread-only構成は最適構成比`-20.98%`の差が出たため、常用設定にしない。

## 6. このレポートが役立つ読者

| 読者 | 使いどころ | 得られる意思決定 |
| --- | --- | --- |
| Pythonシミュレーション開発者 | NumPy/BLASの並列設定 | 処理時間を縮める初期設定を即決できる |
| Intel/Mac移行検討者 | 移行前の実効性能見積もり | CPU律速か帯域律速かで期待値を分けられる |
| 研究室・チーム運用者 | 再現可能な性能ベースライン管理 | 機材更新時の比較を同一指標で継続できる |

## 7. 他Mチップへの横展開手順

同じスクリプトをそのまま実行し、次の比較を取ると機種差を公平に評価できます。

- `CPU P-core unit`の比較: 単コア群の純粋な世代差
- `E/P ratio`の比較: Eコアの実効価値
- `Memory saturation worker`と`peak BW`の比較: 帯域と並列競合耐性
- `Roofline knee`の比較: どの演算密度でメモリ律速から脱出できるか
- `Sustained drift`の比較: 長時間運用の安定性

推定に使う近似式（実務での初期見積もり用）:

- メモリ律速処理の速度比 `~ (target_peak_bw / current_peak_bw)`
- CPU律速処理の速度比 `~ (target_p_core_unit / current_p_core_unit)`
- 混合処理の速度比 `~ min(上記2つの上限)`

## 8. 再現コマンド

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install numpy psutil
python scripts/m_series_probe.py --output results/m_series_probe_full.json
python scripts/render_m_series_report.py --input results/m_series_probe_full.json --output reports/m1_pro_reverse_engineering_report.md
```

## 9. 主要指標サマリー

| Metric | Value |
| --- | --- |
| CPU P-core unit (M it/s) | 8.03 |
| E/P ratio | 0.152 |
| CPU knee worker | 7 |
| Memory peak BW (GB/s) | 60.86 |
| Memory saturation worker | 5 |
| DGEMM peak (GFLOPS) | 488.3 |
| Roofline knee (FLOP/byte) | 15.86 |
| Sustained drift | -0.155% |
| Hybrid best (proc x thread) | 3 x 2 |
| Hybrid best throughput (GFLOPS) | 406.5 |
| Thread-only delta vs best | -20.98% |
| Process-only delta vs best | -11.78% |
| Cache plateau BW >=16MiB (GB/s) | 40.86 |
