# INT8 / INT4 研究ループ

開始: 2026-09-24。ブランチ: `codex/int8-int4-calibration-research`。
開始点: `cfc27c6f971aab8680eba1ac2929b4c904c49259`。
文献: [2026-09-24 再調査](INT8_INT4_RESEARCH_20260924_JA.md)。

## ゴールと最初の達成条件

RTX 2060 Max-Q 6 GB / Windows / CUDA 13.0 / LibTorch 2.10 上で、現行の
選択的 INT8 より速く、独立評価画像でも FP16 との品質一致基準を満たす
再現可能な構成を確立する。まず SAM3 静止画推論を対象にする。
SAM3.1 動画への適用は、画像の成功から自動的に推定せず別に測る。

- 基準: INT8 attention / QKV + global 4 block MLP INT8、sequence attention output。
- 品質: finite、検出数一致、Hungarian 対応付け後の最小 mask IoU >= .98、
  最大 score 差 <= .02、最大 box 座標差 <= 1px。参照は FP16。
- 速度の採用目安: 同一の pixel / attention 設定で全体 median 5%以上短縮、
  p95が5%以上悪化しないこと。複数の fresh process で交互比較し、単一画像の偶然の差に依存しない。
- メモリ: 6 GB実機に収まり、peak allocated / whole-GPUを記録。
- 校正・候補選択・最終評価の画像を分離する。既存17例は回帰用。
- 上記はこの研究用に事前固定する採用条件。FP16一致は正解ラベルに対する
  segmentation accuracy と同じではない。ラベル付き指標を測った場合は別に記す。

## 実験の進め方

仮説 → 文献/ソース確認 → 最小の校正・実装 → operator検証 → 開発画像 →
end-to-end測定 → 採否と理由を記録 → 次の仮説。最終評価は候補を固定してから実行し、
その結果で再調整した場合は、そのセットを開発用へ移し新しい最終評価セットを用意する。
未達を成功扱いにせず、悪化・失敗・環境要因も保存する。

## Round 0 — 研究基盤

新規ブランチとゴールを作成。既存ビルド、GPU空き、FP16/INT8/INT4実験コードを確認。
COCOから校正・開発・最終評価用の画像/プロンプトを固定し、基準出力を保存する。
最初の変更は FC2 入力のチャネルscale/shift校正。production defaults は変更しない。

数値結果と再現コマンドは各round完了時に追記する。

### 固定データと検証方法

Hugging Face `detection-datasets/coco` の train 32枚を校正、val 16枚を開発、
別の val 32枚を最終評価用とした。各splitの Dataset Viewer 先頭100行をseed固定で
シャッフルした小さな便宜的標本であり、COCO全体を代表する精度評価ではない。
画像IDとSHA256の重複がないことを検査した。最大物体のカテゴリ、異なる小物体のカテゴリ、
一部で未注釈カテゴリをtext promptに使う。未注釈は実際の不在を保証しない。
画像・prompt・選択規則は [manifest](../../experiments/results/native_rtx2060/quant-research-data.json) に固定。

`prepare_quant_research_data.py` は既存manifestがあれば選択を変更せず、ローカル画像の
SHA256を照合する。`run_quant_research.py` は各画像/promptを別processで実行し、
exe/DLL/manifest/校正ファイルのSHA256、環境変数、出力一致、NVMLを記録する。
校正runの時間には統計収集の同期とI/Oが入る。品質runのcold/単発時間を速度主張に使わない。

## Round 1 — FC2チャネル変換

仮説: GELU後のチャネルごとの分布差を補正すれば、INT8 MLPの適用範囲を広げられる。
32層のFP16 FC2入力について各チャネルのmin/max/mean/RMSと重みmaxを記録する。
`x' = half(x/r-b)`、`W' = half(W*r)`、`bias' = half(bias + W' b)` とし、
実行時の変換をGELU/INT8量子化境界へ融合する。重み変換とbias補正はセットアップ時だけ。

これは [UQ-ViT公開コード](https://github.com/jiujiuwei/UQ-ViT/blob/3d7d4bb2e7eb06e6c858fd6e33f6f9adab47beed/classification/quant/quant_modules.py)
のscale/shift等価変換を参考にした独自の適用実験。公開実装のasymmetric observer、
clipping探索、精度結果をそのまま再現したものではない。最初はSmoothQuant型の振幅/重み
バランス係数と、shiftなし/mean/midrangeを比較する。全MLPへ適用した場合はFC1も既存INT8のため、
不合格からFC2だけを原因とは断定しない。

operator検証では入力幅1/31/32/257/1024/4736/8192を確認。
identity変換で既存量子化とbyte一致、非自明なscale/shiftで独立したCPUスカラー参照とbyte一致。
本番の5184x4736形状と非default CUDA streamを含む。モデル品質・速度はこれとは別に判定する。

### 最初のスクリーニング結果

校正32枚の観測process合計は214.24秒。FP16観測runの最大allocatedは2.307 GB、
NVMLのwhole-device標本最大は3.262 GB。後者はprocess単独の厳密なpeakではない。
最終評価用画像は推論に使用していない。開発用16枚・33promptのFP16参照を保存し、
候補選択はまず先頭4枚・9prompt（FP16非空8例）で行った。

| 構成（attention/QKV INT8を共通使用） | 合格 / 9 | 非空合格 / 8 | 検出数変化 | 最小mask IoU | 最大score差 |
|---|---:|---:|---:|---:|---:|
| 現行global4 MLP INT8 | 8 | 7 | 0 | .98403 | .03027 |
| 全32 MLP INT8 | 6 | 5 | 1 | .97468 | .03223 |
| 全32 MLP INT8 + identity校正 | 6 | 5 | 1 | .97468 | .03223 |
| 全32 MLP INT8 + alpha=.5、shiftなし | 8 | 7 | 0 | .98008 | .02686 |
| 全32 MLP INT8 + alpha=.5、mean shift | 7 | 6 | 0 | .97619 | .03467 |
| 全32 MLP INT8 + alpha=.5、midrange shift | 7 | 6 | 0 | .96056 | .02393 |

全層INT8のidentity校正は、9/9でmasks/scores/boxes/query IDsが元の全層INT8とbyte一致。
新しい変換経路を追加しただけで既存出力が変わったわけではない。
shiftなし校正で人物maskとノートPC検出数は改善したが、テレビのscore差が残る。
midrangeはテレビを改善する一方で他の2例を悪化させた。したがって**3候補とも採用条件は未達**。
既存17例で通ったglobal4も新規テレビ例ではscore基準を満たさなかった。

次の仮説: 残るテレビの誤差にはFC2以外の量子化も寄与する。
attention単独、attention+QKV、MLP単独と組合せで切り分ける。

数値: [screen summary](../../experiments/results/native_rtx2060/fc2-screen-summary.json)、
[identity一致](../../experiments/results/native_rtx2060/fc2-identity-model-check.json)、
[分布](../../experiments/results/native_rtx2060/fc2-channel-distributions.json)、
[校正パラメータ要約](../../experiments/results/native_rtx2060/fc2-calibration-candidates.json)。
個々の9例は同じresultsディレクトリの `screen-*.json` に保存。

### 再現の入口

既存のCUDA/LibTorch/CUTLASS/Kitchen構成で `sam3_image_latency` と
`sam3_boundary_bench` をビルドする。共有FP重みは変換せずそのまま読む。
以下はrepository rootから実行する（Windowsでは `python` に `.venv/Scripts/python.exe` を使用）。

```text
python experiments/prepare_quant_research_data.py
sam3_boundary_bench.exe boundary-check.json --check-only
python experiments/run_quant_research.py observe --role calibration --mode fp16 --name calibration-fp16
python experiments/run_quant_research.py quality --mode fp16 --name development-fp16
python experiments/make_fc2_calibration.py .cache/quant-research-20260924/runs/calibration-fp16 .cache/quant-research-20260924/calibrations/a050-none --alpha .5 --shift none
python experiments/run_quant_research.py quality --mode calibrated --calibration .cache/quant-research-20260924/calibrations/a050-none --name screen-a050-none --limit-images 4
```

`--shift mean/midrange` と `--identity` で他候補を生成。対照は `--mode selective/all`。
実行完了済みのrunは同じ署名でのみ再利用でき、途中失敗は自動上書きしない。
校正生成先も既存binaryを上書きしない。ビルドや条件を変えた再測定は別のrun名を使う。

## Round 2 — 構成間の誤差の切り分け

前roundの失敗例である画像4795の `tv` を固定して、構成を分離した。

| 量子化する箇所 | 最大score差 | 判定 |
|---|---:|---|
| attentionのみ | .01123 | Pass |
| attention + QKV、MLPはFP16 | .03271 | Fail |
| global4 MLPのみ | .00635 | Pass |
| 校正した全32 MLPのみ | .00293 | Pass（最大box差 .97687px） |
| 校正した全32 MLP + attention | .03271 | Fail |
| 校正した全32 MLP + QKV | .05713 | Fail |

全32 MLPの校正はalpha=.5、shiftなし。単独で通るattentionとMLPを組み合わせると
失敗するため、単層の誤差や単独構成だけで最終設定を決められない。
QKVだけが唯一の原因、またはFC2校正で全問題が解決したとは結論しない。
[切り分けの生結果](../../experiments/results/native_rtx2060/fc2-tv-isolation-summary.json)。

**attention/QKVをFP16、全32 MLPを校正INT8** にした構成は、続く同じ4枚9promptで
9/9（非空8/8）を通った。最小IoU .98207、最大score差 .00928、最大box差 .97687px。
これは開発用の小規模screenを通った段階であり、独立検証や速度条件の達成ではない。
[各例の結果](../../experiments/results/native_rtx2060/screen-cal-mlp-only.json)。

### 実装検証

新しいaffine境界のCompute Sanitizer memcheckは0 errors。
既存CTestは54/54、142.11秒で合格した。公開ABIと通常時の量子化設定は変更していない。
[検証要約](../../experiments/results/native_rtx2060/quant-research-validation.json)、
[memcheck log](../../experiments/results/native_rtx2060/boundary-affine-memcheck.log)、
[CTest log](../../experiments/results/native_rtx2060/quant-research-ctest.log)。
