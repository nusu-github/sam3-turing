# Windows / Turing 近似演算の実験結果

> **現状:** `tanh`・`fused` は不採用として削除した。`int8` は同じ出力の `int8_boundary`
> に置き換えたため削除した（[BOUNDARY_FUSION](BOUNDARY_FUSION.md)）。
> `sam3_approx_bench` も削除した。いずれも commit `7d7fddb` に残っている。

2026-09-24、RTX 2060 Max-Q、standalone LibTorch 2.10+cu130、base `908e476`。
通常の厳密演算がデフォルト。`SAM3_EXPERIMENT_MLP` はプロセス起動前に設定する実験用スイッチ。
モデル作成後に値を変更しない。今回は runtime / diagnostic のローカル変更のみでコミットしていない。

## 結果

| Vision MLP | 全体 wall 中央値 | p95 | 判断 |
|---|---:|---:|---|
| `exact`: 現行 FP16 + exact GELU | 535.24 ms | 553.29 ms | 基準 |
| `tanh`: 独立した近似 GELU | 543.61 ms | 565.89 ms | 改善を確認できず |
| `fused`: FP16 `_addmm_activation` | 541.67 ms | 554.76 ms | 改善を確認できず |
| `int8`: 全32ブロックの FC1/FC2 を W8A8 | **493.64 ms** | **513.86 ms** | **約7.77%短縮**、品質評価を続ける候補 |

各モード fresh process 2回、warmup 5、計測15回、計30サンプル。
実行順は exact/tanh/fused/int8/int8/fused/tanh/exact。各プロセスの中央値:

- exact: 531.30 / 547.40 ms
- tanh: 537.39 / 550.48 ms
- fused: 539.64 / 548.32 ms
- int8: 501.48 / 491.71 ms

固定クロックではなく、各run末尾の監視クロック中央値は1245–1335 MHz。
順序は対称化したが、温度・クロック差は残る。小さい差の有意性を主張しない。
INT8は両runとも両基準runより短かった。この環境・入力における探索結果であり、
他画像、他GPU、データセット全体での速度保証ではない。

入力は truck.ppm / truck、1008解像度、200 queries、閾値0.5。
テキストはキャッシュ、画像特徴は毎回計算。4 CPU threads、cuDNN benchmark off、TF32 off、
BLAS preference 0、CUBLAS_WORKSPACE_CONFIG=:4096:8。
CPU RGB転送、前処理、Vision、Detector、後処理を含み、モデルロード・ディスクIOを除く。
全run同一DLL、並行するGPU実験なし。

CUDAイベントでのVision平均は407.34→359.82 ms（約11.7%短縮）。
Detector平均は123.93→132.64 msで改善していない。各段平均をwall中央値に加算しない。

## INT8実装

- 実行経路は standalone C++ `at::_int_mm`。Pythonはプロセス起動・NVML監視のみ。
  子プロセスのPATHはWindowsシステムディレクトリのみで、依存DLLは実行ファイル横のLibTorch。
- 重みは出力チャネル別、入力はトークン別の対称INT8。スケールはmax(abs)/127。
- 重みの量子化はロード時、入力の量子化は各推論時。CUDAカーネルで行う。
- INT32 GEMM出力をCUDAで復元し、biasとFC1のerf GELUを適用してFP16へ戻す。
  FC1は元の「GELU前にFP16へ丸める」位置も変わる。
- LayerNorm統計、residual、attention、Detectorは現行の精度を維持。
- smoothing、校正データ、層別の精度選択は未実装。品質を改善する余地がある。

これはメモリ削減実装ではない。比較用FP16重みを残してINT8重みを追加している。
推論peak allocatedは2,375,946,240→2,687,193,088 bytes（**約296.83 MiB増**）。
モデルサイズやVRAMが小さくなったとは報告しない。

## 5ケースの出力比較

各モード別プロセスでcold inference + 1回実行。ここの時間は性能評価に使わない。
基準FP16とのマスクIoUを計算し、IoU合計を最大化するHungarian法で対応付けた。
これは正解ラベルに対するIoUやAPではなく、**参照モデルとの一致度**。

| ケース | 全モードの検出数 | tanh 最低IoU | fused 最低IoU | INT8 最低IoU |
|---|---:|---:|---:|---:|
| truck | 1 | 0.999967 | 0.999970 | 0.999670 |
| paper bag | 4 | 0.999883 | 0.999883 | 0.998122 |
| child | 6 | 0.999768 | 0.999710 | 0.996927 |
| wheel | 4 | 0.999558 | 0.999853 | 0.989715 |
| elephant / 不在 | 0 | — | — | — |

INT8の最大スコア差は0.0112305、最大box座標差は0.64856 px。
今回の閾値では件数が保たれたが、閾値近傍や別画像での検出消失・追加は未検証。
全出力は有限。exactのtruck出力は従来の融合ON基準とmask/box/score/queryすべてバイト一致。
5ケースのスモーク検証だけでは製品品質の採用判断をしない。

## 単体カーネル

モデルと同じ形状の合成入力を使い、各候補20回のCUDAイベント計測を正順/逆順で4巡。
重みの事前量子化を除き、動的入力量子化・INT32出力・復元・bias・FC1 GELUを含む。
`approx-micro-normal-1.json` の最初の巡では:

| 演算 | FP16 | INT8 GEMMのみ | INT8変換込み |
|---|---:|---:|---:|
| FC1 + GELU (5184×1024×4736) | 3.092 ms | 1.275 ms | 2.339 ms |
| FC2 (5184×4736×1024) | 2.647 ms | 1.025 ms | 1.932 ms |

GEMM単体の差をモデルの高速化率として使ってはいけない。
検証追加後の再実行も成功: 量子化結果をATen参照と完全一致で比較、小整数GEMMを
FP32参照と完全一致で比較、復元とGELUを参照とrtol/atol=1e-3で比較。
ゼロ行・定数行も含む。`approx-micro-normal-validated.json` を参照。

既存のCPU/CUDAテストは **48/48 passed**（143.99秒、実験モード未設定）。
ログ: `build/native-windows-cu130/windows-validation-approximation.log`。
これは既存機能の回帰確認であり、INT8のデータセット精度評価ではない。

## 除外した実行

最初はGPU制限で390 MHzだった。`approx-micro-2/3`、`approx-micro-unlocked-1` は
その条件の参考値であり、上の結論には使っていない。
`.cache/native-perf/approximation-power-transition` は中断、`approximation-v2` は
最初の基準だけ390 MHzなので系列全体を性能結論から除外した。
最初のdiagnosticはイベントのtiming指定不足で失敗し、修正後に再実行した。

## 次に狙う箇所

近似GELU単独は保留。INT8は速度差が出たが、暫定目標の15%短縮には届いていない。
次はINT32の大きな中間出力と量子化/復元コストを減らす。LayerNormからのINT8直接出力、
GEMM epilogueでの復元などを候補にする。ただしTuring cuBLASLtのepilogue制限があり、
対応する独自カーネルが必要になる可能性がある。
その前に校正用/評価用データを分けた品質測定を用意し、敏感なMLP層をFP16へ戻す比較を行う。

## 再現

- `native/tools/approx_bench.cpp` と `approx_kernels.cu`: 合成入力によるカーネル検証と性能測定。
- `experiments/run_approx_native.py timing` / `quality` と `summarize_approx_native.py [ROOT]`:
  同一DLLでモードを切替えて測定・集計するドライバー。
- ベンチとドライバーは実験終了後に削除した。当時のランタイムとともに commit `7d7fddb` に
  残っている（`git worktree add ../sam3-7d7fddb 7d7fddb`）。
- [全体の生データ・比較](approximation-results.json)、[測定DLL等のハッシュ](approximation-manifest.json)。
- 各runの出力・NVML監視・コマンドは `.cache/native-perf/approximation-v3/`。
