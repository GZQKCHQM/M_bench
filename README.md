# m_mac_chip

Apple Silicon (Mシリーズ) 向けに、Python中心の計算ワークロードを定量評価するためのベンチマーク/レポート生成プロジェクトです。  
単純なベンチマーク結果の羅列ではなく、機種をまたいで比較しやすい共通指標を出力します。

## このプロジェクトで分かること

1. CPU並列の実効上限  
`cpu_scaling` から、`workers` ごとの `throughput/speedup/efficiency` を取得できます。  
「何並列までは素直に伸びるか」「どこから効率が落ちるか」を判断できます。

2. BLASスレッドの実用最適点  
`matmul` で行列サイズ別に `threads` と `GFLOPS` の関係を測定します。  
`VECLIB_MAXIMUM_THREADS` の初期値を決める根拠になります。

3. メモリ律速の判定と並列数の目安  
`stream_single` と `stream_scaling` から、帯域ピークと飽和ワーカー数を算出します。  
メモリ帯域が支配的な処理で、プロセスを増やしすぎる損失を避けられます。

4. 長時間運用時の安定性  
`matmul_sustained` で連続実行中の性能ドリフトを測定します。  
短時間ベンチだけでは見えない、実運用時の安定性を確認できます。

5. 複数ジョブ同時実行時の最適構成  
`hybrid_matmul` で `processes x BLAS threads` を探索します。  
thread-only / process-only との性能差を定量化して、同時実行設定の最適化に使えます。

## こんな人に役立ちます（具体）

- Pythonでシミュレーションを実装している開発者  
  NumPy計算の並列設定を、推測ではなく実測で決められます。

- Intel環境からMシリーズへの移行を検討している人  
  CPU律速かメモリ律速かを切り分けて、移行効果の見積もり精度を上げられます。

- 複数の計算ジョブを同時に回す運用をしている人  
  `processes x threads` の最適点を見つけ、処理待ち時間を短縮できます。

- 研究室/チームで機材比較を継続したい人  
  同じ指標で世代比較でき、更新判断を毎回同じ基準で行えます。

## 直近の実測ハイライト（このリポジトリの最新結果）

対象: Apple M1 Pro (6P+2E, 16GB), Python 3.13.12, NumPy 2.4.2

- CPUスカラー処理は6ワーカーまで高い線形性、8ワーカー時の上積みは `+14.31%`
- Eコア寄与はPコア比 `42.94%`（単コア寄与ベース）
- メモリ帯域ピーク `63.63 GB/s`、95%飽和は `3` ワーカー付近
- DGEMM (N=3072) は `2` スレッドでほぼ飽和
- 持続性能ドリフトは `0.844%`
- ハイブリッド並列の最速は `3 x 2`（processes x threads）
- thread-only (`1 x 8`) は最適構成比 `-20.46%`
- process-only (`8 x 1`) は最適構成比 `-11.69%`

詳細レポート: `reports/m1_pro_reverse_engineering_report.md`

## リポジトリ構成

```text
scripts/
  m_series_probe.py             # 実測ベンチマーク本体
  render_m_series_report.py     # JSON結果からMarkdownを生成
results/
  m_series_probe_full.json      # フル計測結果
  m_series_probe_quick.json     # 短縮計測結果
reports/
  m1_pro_reverse_engineering_report.md
```

## 評価指標（Mシリーズ共通で比較可能）

- `CPU P-core unit`  
  Pコア群の基礎スループット比較指標
- `E/P ratio`  
  Eコア単コア寄与 / Pコア単コア寄与
- `Memory saturation worker`  
  メモリ帯域が95%ピークに達する最小ワーカー数
- `Roofline knee (FLOP/byte)`  
  計算律速と帯域律速の境界目安
- `Sustained drift`  
  長時間連続実行での性能変動率
- `Hybrid best (proc x thread)`  
  複数ジョブ同時実行時の最速構成

## 必要環境

- macOS (Apple Silicon)
- Python 3.10+（本リポジトリの実測は 3.13.12）
- `venv` が利用可能であること

## セットアップ

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install numpy psutil
```

## 実行方法

### 1. フル計測

```bash
. .venv/bin/activate
python scripts/m_series_probe.py --public --output results/m_series_probe_full.json
```

### 2. レポート生成

```bash
. .venv/bin/activate
python scripts/render_m_series_report.py \
  --input results/m_series_probe_full.json \
  --public \
  --output reports/m1_pro_reverse_engineering_report.md
```

### 3. 短縮計測（動作確認向け）

```bash
. .venv/bin/activate
python scripts/m_series_probe.py --quick --public --output results/m_series_probe_quick.json
```

## 出力ファイルの見方

- `results/m_series_probe_full.json`  
  機械処理向けの生データ。CIでの比較や可視化に向いています。
- `reports/m1_pro_reverse_engineering_report.md`  
  人間向けの要約レポート。GitHub公開時の本文として使えます。

## Pythonシミュレーションでの実務的な使い方

1. まず `BLAS threads = 2` から探索を開始する  
2. メモリ律速系は `workers ≈ 3` 前後から調整する  
3. 独立ジョブを複数同時実行する場合は `3 x 2` を第一候補にする  
4. thread-only固定運用は避け、`processes x threads` を必ず探索する  

## 他のMシリーズ機で比較する手順

1. 比較したいMシリーズ機で、同じセットアップを実行する  
2. 各マシンでフル計測を実行し、機種ごとに別名で保存する  
`python scripts/m_series_probe.py --public --output results/m_series_probe_<machine>.json`
3. 各計測結果からレポートを生成する  
`python scripts/render_m_series_report.py --input results/m_series_probe_<machine>.json --public --output reports/<machine>_report.md`
4. 次の項目を機種ごとに並べて比較する  

- `CPU P-core unit`
- `E/P ratio`
- `Memory peak BW`
- `Memory saturation worker`
- `Roofline knee`
- `Sustained drift`
- `Hybrid best`

初期見積もりの近似式:

- メモリ律速処理の速度比 `~ (target_peak_bw / current_peak_bw)`
- CPU律速処理の速度比 `~ (target_p_core_unit / current_p_core_unit)`
- 混合処理の速度比 `~ min(上記2つの上限)`

## 再現性のためのチェックポイント

- 電源接続状態を固定する（AC推奨）
- バックグラウンド負荷を減らす
- 同一Python/NumPyバージョンで比較する
- 同一実行回数で比較する（`--quick` と `full` を混在させない）

## 制約

- CUDA前提ワークロードの比較には向きません
- 本測定はNumPy/Accelerate中心であり、PyTorch/JAX固有最適化は対象外です
- 絶対性能はOS状態・温度・同時負荷の影響を受けます
