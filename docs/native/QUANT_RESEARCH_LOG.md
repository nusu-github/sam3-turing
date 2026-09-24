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
- 速度の採用目安: pixel変換・attention出力layout等の共通最適化と入出力条件を揃え、
  固定した現行構成に対して全体 median 5%以上短縮、
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

### 開発16枚へ拡張した結果

校正全MLPのみの構成は33prompt中29合格（非空25例中22合格）。
検出数変化は2例。全体の最小IoU .94466、最大score差 .03711、最大box差1.35315px。
最初の9例の合格から一般化したと判断しない。

- 画像2261 `surfboard`: FP16は0検出、候補は1検出。
- 画像5992 `sheep`: 検出数8とmask基準は保つが、box差1.35315px。
- 画像9590 `person`: 検出数5→6、最小IoU .94466、score差 .03711。
- 同画像 `spoon`: IoU .97183、score差 .02246。

[全33例](../../experiments/results/native_rtx2060/development-cal-mlp-only.json)。
品質未達のため最終評価セットは使用しない。

同じ33promptで現行global4 + attention/QKV INT8を測ると31/33（非空23/25）だった。
テレビのscore差 .03027と羊のbox差1.12643pxが基準外。現行にも誤差はあるが、
校正全MLPのみが開発セット全体で上回ったとは言えない。
[同一条件での全件比較](../../experiments/results/native_rtx2060/fc2-development-summary.json)。

### 追加調査から次の仮説へ

Sciteで2025年以降のbias correction/ViT関連を追加検索し、
[Joint PTQ of ViTs（2026-02-21）](https://arxiv.org/html/2602.18861v1) を本文で確認した。
32枚はパラメータ初期化に用いる枚数で、その後は中間特徴と最終logitの蒸留を含む
全体最適化を24,000反復・batch32で行う。対象はImageNet分類のViT/DeiT/Swin。
SAM3で32枚の統計を取るだけで同じ結果になる研究ではない。HF papers APIは404で取得できなかった。
当面は層間相互作用と出力スコアを見ながら、より小さな再構成実験へ落とす。

次に調べるのは、(1) FC1とFC2の量子化の分離、(2) 実際に量子化した重みによる
平均出力誤差のbias補正、(3) global attentionだけを量子化する構成。
FC2の入力分布だけを改善しても、FC1やQKV由来の誤差は消えないためである。
まだ採用条件を満たした構成はなく、品質基準は変更しない。

bias correction自体は新規の発想ではない。
Sciteで [Nagel et al., ICCV 2019](https://arxiv.org/abs/1906.04721) の平均誤差補正も再確認した。
SAM3ではpost-GELU入力と実際の量子化重みに合わせて補正量を求め、
既存のscale/shiftに伴う代数的bias補正とは分けて効果を検証する。

### 速度・メモリと追加候補の採否

品質が通った開発用2画像（7574 / 1425の主prompt）で、現行と校正全MLPのみを比較した。
各画像・各構成2 fresh process、5 warmup + 15測定、2巡目は構成の順番を反転。
1構成60サンプル。これは品質不合格候補の診断用測定であり、採用の根拠にはしない。

| 構成 | pooled median ms | pooled p95 ms | 最大allocated GB | sampled whole GPU GB |
|---|---:|---:|---:|---:|
| 現行global4 + attention/QKV INT8 | 436.80 | 453.96 | 2.432 | 3.470 |
| 校正全MLP INT8、attention/QKV FP16 | 435.56 | 448.46 | 2.601 | 3.591 |

pooled medianの短縮は0.28%、画像別medianでは0.95% / 1.15%に留まり、5%条件未達。
process medianにも変動があり、NVML温度は測定開始63℃から最終77℃へ上がった。
この小さな差を安定した高速化と断定しない。次の有望候補では熱状態が安定してから再測定する。
GBは10^9 bytes。NVMLは20ms間隔の全GPU標本で厳密なprocess peakではない。
[全サンプル・process別結果・温度](../../experiments/results/native_rtx2060/fc2-mlp-only-paired-timing.json)。

再現: `run_quant_research.py timing --mode selective --image-id 7574 --primary-only --name <unique>` と、
`--mode calibrated --calibration <a050-none> --attention exact --projection exact` の候補を比較する。
画像1425でも同じ設定を使い、2巡目は候補を先にする。8 runのディレクトリを
`summarize_quant_timing.py <run1> ... <run8> --output <report.json>` に渡す。

さらに全MLP校正 + global attentionのみINT8 / QKV FP16を調べた。
テレビはPass、surfboardの検出数変化も解消したが、追加3画像5promptでは2/5合格。
羊のbox差1.70251px、人物の5→6検出とIoU .94018、spoonのIoU .97872 / score差 .02051でFail。
全33例への展開は行わず不採用とした。
[追加screen](../../experiments/results/native_rtx2060/fc2-global-attention-probe-summary.json)。
回転付きall attentionもテレビのbox差1.33914pxで不合格だった。

## 次の反復の開始点

ゴールは継続中。最終評価32枚は未使用、通常設定の採用変更なし。
まずFC1 FP16 / FC2 INT8を分離して、今回の失敗がFC1を含めた量子化から来るか確認する。
その後、校正データのみから平均誤差補正またはFC1のチャネル変換を作る。
FC2の品質・速度が成立した範囲でINT4へ同じ校正を移す。失敗例を隠したり基準を緩めたりせず、
screen → 開発33例 → 既存17例 → 候補固定 → 未使用holdout → 制御した速度測定、を続ける。

## Round 3 — FC1 / FC2の分離

前roundは実装・検証・反証データが増えたためprogressと判定して継続。
開始時は `ea8aeb4`、作業ツリーclean、RTX 2060 Max-QはGPU使用率0%、57℃を確認した。
新しい `SAM3_EXPERIMENT_MLP_PART=both|fc1|fc2` は、選んだLinearだけを量子化する。
未選択側のINT8重みは確保しない。既存のblock scopeと組み合わせられる。
初期実装ではpart分離とINT4の同時使用を拒否し、校正はFC2を量子化する場合だけ許可する。

FC2のみの場合はFP16 FC1出力からGELU → FP16丸め → optional affine → FP16丸め →
row INT8量子化を融合する。既存の両FC量子化経路はGELU前の復元値がFP32なので、
FC1をFP16に保つFC2単独との比較には、この丸め位置の差も含まれる。すべてが重み誤差だけの切り分けではない。

新GELU融合は7種類の幅、random/zero/constant入力、本番5184x4736、非default streamで
ATen GELU + 既存quantizerとbyte一致。affineありはCPUスカラー変換の参照とも一致。
1画像のFP16/従来両FC INT8は前ビルドと出力全ファイルがbyte一致。
[operator結果](../../experiments/results/native_rtx2060/round3-boundary-check.json)、
[既存経路の対照](../../experiments/results/native_rtx2060/round3-default-control.json)。

最初のscreenは画像1584、4795、2261、5992、9590の9prompt。
前roundの失敗と小さな人物maskを含む開発例であり、独立評価ではない。
attention/QKVをFP16に固定し、FC1のみINT8、FC2のみINT8、alpha=.5/shiftなし校正FC2の3条件を比較する。

### 分離screenの結果と平均誤差補正

FC1のみ / FC2のみはいずれも8/9合格。両方で9590 `person` の5→6検出が残った。
追加query 87のscoreはそれぞれ.54590 / .54395で、単に.50000付近の丸めだけとは言えない。
既存alpha=.5/shiftなしをFC2単独へ移すと5/9へ下がり、1584の人物mask、テレビscore、
laptop検出数、羊のboxが基準外になった。一方9590の人物数は正しくなるため、
校正を無条件に良いものとは扱わない。identity校正はテレビ画像の2promptとも
FC2未校正出力と全ファイルbyte一致した。
[集計と失敗の詳細](../../experiments/results/native_rtx2060/round3-isolation-summary.json)。

追加候補は校正32枚からFC2入力のchannel mean `mu` を保存し、実際のINT8重みを
復元した `Wq` に対して `bias_new = bias + W_original @ mu - Wq @ (mu/r-shift)` を
FP32で計算、FP16 biasとして保持する。起動時だけの処理で推論演算を増やさない。
既存の等価変換biasをこの値で置き換える。重み量子化と重み変換の丸めに対する平均補正であり、
活性の丸め・量子化や上流の誤差伝搬は含めない。`SAM3_EXPERIMENT_FC2_MEAN_BIAS=enabled`
およびrunnerの `--mean-bias` で明示的に有効にする。

元の校正データは変更せず、`make_fc2_calibration.py ... --mean-bias` によって
`identity-mean-bias` / `a050-none-mean-bias` を新規作成。
32層×4736のmeanファイルSHA256は
`67d2fc4002b37dc614219a18bcff803e5bd0f958cb68b7f47d9e1d1a3481f0aa`。
変換scale/shiftのSHAは以前のidentity / a050-noneと同じ。
CPU float64で独立に作った出力の標本平均と照合し、7幅で補正後の誤差は最大約2e-6。
FP16 bias化後の追加誤差もbias丸めの範囲内だった。
[演算検証](../../experiments/results/native_rtx2060/round3-boundary-mean-check.json)。

screenは同じ5画像9promptで、FC2単独 / 両FC、identity / alpha=.5の4候補。
パラメータ作成には開発画像を使わず、最終評価32枚も未使用のまま比較する。

### 今回の追加文献確認

Sciteで [UniQ-ViT（Neurocomputing 2026）](https://doi.org/10.1016/j.neucom.2025.132072)
の書誌を照合。著者所属機関の[公開抄録](https://research.birmingham.ac.uk/en/publications/uniq-vit-optimization-driven-uniform-quantization-for-vision-tran/)
では、uniform量子化の範囲初期化後、block単位で重みと量子化パラメータを調整し、
post-LayerNormのchannel差には2段階の再パラメータ化を使う。
本文取得はできず、抄録からSAM3/SM75の実測効果は判断しない。
GitHub connectorで[公開リポジトリ](https://github.com/Dexter-Yu/UniQ-ViT/tree/3d711f88975db8c322ff3b714dc9cd96558ec588)
を確認したところ、このrevisionには98-byte READMEだけがあり、再現実装はなかった。
「code available」という抄録の記述だけで再現可能とみなさない。

[Colinearity Decay（2026-05-02 v1）](https://arxiv.org/html/2605.01330v1) も本文確認。
Transformer内の行列の組に正則化を加える学習・fine-tuning手法で、
推論時のscale/shift校正だけで適用する方法ではない。Swin/ViTでのW4A4評価を報告しているが、
本プロジェクトのSAM3/SM75にそのまま当てはめる根拠はない。
HF papers APIの当該IDは404だったため、HF登録を裏付けにせずarXiv一次本文を参照した。

平均補正screenの結果:

| attention/QKV FP16の候補 | 補正前 | 平均補正あり |
|---|---:|---:|
| FC2のみ、identity | 8/9 | 7/9 |
| FC2のみ、alpha=.5 | 5/9 | 5/9 |
| 両FC、identity | 同条件の9例は未測定 | 8/9 |
| 両FC、alpha=.5 | 5/9 | 8/9 |

両FC/alpha=.5では、surfboardの0→1検出、羊のbox差、spoonのIoU/scoreが解消した。
9590人物は依然5→6検出、IoU .95339、score差 .02783でFail。
FC2単独のidentityでは小さな人物maskも基準外となり、補正がすべての構成で改善するわけではない。
[各候補の全結果・処理時間・メモリ](../../experiments/results/native_rtx2060/round3-mean-bias-summary.json)。
これらは品質用のcold+1回測定であり、latencyの比較には使用しない。

次のscreenは両FC/平均補正でidentityとalpha=.5の2条件を固定し、attention全層とQKVをINT8にする。
品質が成立するなら高速化の測定対象になるが、部分系の8/9から最終構成の品質を推定しない。

### attention/QKVとの組み合わせと今回の判断

全attention・QKVをINT8に戻すと、平均補正付き両FCはidentityが6/9、alpha=.5が5/9だった。
identityは小さな人物mask、テレビscore、9590人物検出数でFail。
alpha=.5は同じ3例に加えてspoonのIoU .97857もFail。
[全結果](../../experiments/results/native_rtx2060/round3-combined-summary.json)。
この2候補は現時点で不採用。開発33例や独立holdoutへは展開せず、速度改善も主張しない。

今回わかったことは、FC1/FC2を片方だけにしても累積誤差は残ること、
重みの平均誤差補正は一部の失敗を解消するがattention/QKVとの組み合わせには不十分なこと。
GELU丸め位置の既存コメントも実コードに合わせて修正した。
新しいnative演算はopt-inのままで、通常設定は変更しない。

次の反復では、最も良かった両FC/alpha=.5/平均補正を使い、まず既存のfirst24/last24などの
MLP scopeで誤差が集中する範囲を切り分ける。必要ならQKVにも層scopeを加えて同じ切り分けを行う。
これはモデル構成の開発選択なので、同じ開発画像での改善は独立した一般化の証拠にしない。
採用候補が固定されるまでholdout32枚は使わない。

### Round 3の検証・記録

最終ビルドで既存FP16 / 両FC INT8 / 平均補正なし校正の3対照を再確認し、
7574の主promptに対して保存済み出力と全ファイルbyte一致。
meanデータが校正フォルダに存在しても、flagなしでは従来出力を維持した。
CTestは **55/55合格、142.01秒**。新しいFP16 GELU境界、affine、平均補正検証を含む
operator全7幅のCompute Sanitizer memcheckは **0 errors**。
Python構文検査と `git diff --check` もPass。
[検証要約とバイナリSHA](../../experiments/results/native_rtx2060/round3-validation.json)、
[CTest](../../experiments/results/native_rtx2060/round3-ctest.log)、
[memcheck](../../experiments/results/native_rtx2060/round3-boundary-memcheck.log)。

モデル評価は対照を含む88回、開発画像6枚（難例5枚＋対照1枚）、51 run。
native子process時間の合計581.64秒。新しい32枚GPU校正取得は不要だった。
保存済み統計からmeanを生成するCPU処理の所要時間は今回は単独計測していない。
最大allocated 2.704GB、20ms sampled whole GPU 3.740GB（GB=10^9bytes）。
すべてcold+1回の品質評価で、これらから速度差を主張しない。

再現例: `make_fc2_calibration.py <calibration-fp16> <new-output> --alpha 0.5 --shift none --mean-bias`
でファイルを作り、`run_quant_research.py quality --mode calibrated --calibration <new-output>
--mlp-part both --mean-bias --attention exact --projection exact --image-id 9590 --name <unique>`。
FC2単独は `--mlp-part fc2`、全attention/QKV INT8との組み合わせはattention/projectionのoverrideを省く。
各run reportに環境、データ、モデル実行バイナリ、校正ファイルのhashを保存している。

今回も新しい切り分け・校正実装・検証・反証が増えたためprogressとして継続。
ゴール達成ではなく、採用候補は未確定。最終評価32枚は未使用。

## Round 4 — MLP / QKVの層ごとの混合精度

前roundは実装と検証・反証結果を増やしたprogressとして継続。
開始commit `ca99271`、作業ツリーclean、GPU使用率0%、56℃、使用382MiBを確認。
まず既存バイナリで、両FC/alpha=.5/平均補正・attention/QKV FP16を固定し、
MLP first24 / last24を同じ難例5枚9promptで比較する。
校正32枚と品質基準は変更せず、最終評価セットは未使用。

MLPとQKVを1層単位で切り分けるため、共有の層mask解釈を追加する。
`all/global/local` に加え、`firstN/lastN`（N=1..32）、`mask:0xHHHHHHHH` を許可。
bit0がblock0、bit31がblock31。mask0は切り分け用の全FP16指定。
MLPは既存 `SAM3_EXPERIMENT_MLP_SCOPE`、projectionは新規
`SAM3_EXPERIMENT_PROJECTION_SCOPE` で指定し、重み確保と実行を一致させる。
QKV fusionはINT8対象層だけに適用し、対象外は既存のFP16 linear/rotaryへ戻す。
入力画像に応じて層構成を変える仕組みではなく、process開始時に構成を固定する。

first24は8/9で、9590人物の5→6検出が残った。last24は7/9で人物は5検出に戻るが、
テレビのscore差 .02246とspoonのIoU .97163がFail。
[粗い範囲の比較](../../experiments/results/native_rtx2060/round4-coarse-mlp-summary.json)。
次はblock0〜7のそれぞれ1層だけをFP16に戻し、残り31層は同じ校正INT8とする。
まず9590の2promptで検出数とspoon品質のトレードオフを調べる。

共有mask parserはN=1..32の全境界、global/local、0/all/上位bit付きmask、
不正文字・範囲外・環境設定cacheをCPUで検証し、5/5 test Pass。
全層mask0+attention FP16はFP16参照とbyte一致、projection allは従来INT8とbyte一致、
MLP first24も改修前とbyte一致。maskの指定で重み準備と実行が食い違わないことを対照確認した。
[3対照](../../experiments/results/native_rtx2060/round4-control.json)。
`--scope` はcalibratedだけでなくallにも適用可能にした。selectiveは従来global4固定で、
無視されるscope指定はrunnerで拒否する。QKV範囲は `--projection-scope` で指定する。

block0〜7の1層だけを除外した比較では、人物の5→6検出が解消したのはblock0のみ。
block0除外時は人物IoU .99248、score差 .00293、box差 .40448pxでPass。
同じ画像のspoonはIoU .98571だがscore差 .02295でFail。
その他の7候補はすべて人物検出数が6のままだった。
[16評価の結果](../../experiments/results/native_rtx2060/round4-mlp-single-block-summary.json)。

次にMLP block0だけFP16を固定し、attention全層INT8の下でQKV範囲を
all / block0除外 / first24 / last24の4構成で比較する。
まずテレビ・laptopの4795と人物・spoonの9590（計4prompt）を使い、
部分系だけで改善したmaskを最終構成で採用できるか判断する。

QKVの粗い範囲比較はall 3/4、block0除外2/4、first24 3/4、last24 1/4。
[組み合わせ結果](../../experiments/results/native_rtx2060/round4-joint-coarse-summary.json)。
QKVをすべてFP16とする対照も3/4で、spoonのIoU .97183がFailだった。
QKV block0のINT8はこの組み合わせで人物の検出数を保つ側に作用する一方、
テレビの誤差には悪化側に働く。単純にFP16を増やす規則では解決しない。

テレビに対してQKV block1〜7を1層ずつ除外すると、block2 / block6がPass。
score差 .00830 / .01025、box差 .22986 / .13144px。
block7はscore差 .00195まで小さくなるがbox差1.74185pxでFailなので選ばない。
[QKV 7候補](../../experiments/results/native_rtx2060/round4-qkv-single-block-summary.json)。
MLP block0 FP16 + QKV block2 FP16（それ以外は従来の校正INT8/all attention INT8）と、
QKV block6に置き換えた2候補を同じ5画像9promptで比較する。

難例9件へ広げると、QKV block2 FP16候補は6/9、block6 FP16候補は7/9。
block6候補は検出数・score・boxが全件基準内だが、小さい人物mask .97890と
spoon .97872のIoUがFail。
[9件での比較](../../experiments/results/native_rtx2060/round4-candidate-summary.json)。

block6候補からさらにMLP末尾2 / 4 / 8層をFP16に戻したところ、
1584/9590の4promptはすべて2/4のまま。spoonのIoUは全候補で .97872。
後段MLPを戻すだけではこの失敗は解消しなかった。
[末尾MLP比較](../../experiments/results/native_rtx2060/round4-tail-summary.json)。

local attentionをFP16（global4のみINT8）にすると、小さい人物maskは .98523へ改善するが、
9590人物が再び5→6検出、spoonも .97163でFail、合計7/9。
[global attentionのみ](../../experiments/results/native_rtx2060/round4-global-attention-summary.json)。
いずれも採用不可。品質が通る前にholdoutへ移ったり、IoU .98の基準を変更したりしない。

速度上の余地を確認するため、block6候補のall attention / global attentionのみの2条件と
現行selectiveを診断比較する。`run_quant_scope_timing.py --name r4-scope-timing` は
現行構成で180 warmup画像＋5測定の事前加熱を行い、その後、開発7574/1425の主promptで
各構成2 fresh process、各5 warmup＋15測定を実行する。2巡目は構成と画像の順を逆にする。
1構成60測定、3構成計180測定。GPU clockは固定せず、NVML温度・clockを記録する。
候補は品質未達のままなので、速度が出ても採用とは扱わない。

比較条件の表現を明確化: 初期の「同一pixel/attention設定」は、無関係な実装最適化の差を
混ぜないための条件として記した。今回以降、attentionのINT8/FP16配分自体も混合精度の
候補変数として評価する。共通のpixel変換・layout指定・入力解像度・cached text/fresh image・
warmup/repeat条件は揃え、比較対象の現行global4 MLP + all attention/QKV INT8は固定する。
品質閾値、5% median改善、p95の上限、独立評価の要件は変更しない。

### 速度診断の結果

| 構成 | pooled median ms | pooled p95 ms | median短縮 | 最大allocated GB | sampled whole GPU GB |
|---|---:|---:|---:|---:|---:|
| 現行selective | 438.09 | 455.06 | — | 2.432 | 3.470 |
| MLP block0・QKV block6はFP16、他の対象LinearはINT8、all attention INT8 | 380.03 | 395.28 | 13.25% | 2.692 | 3.744 |
| 同MLP/QKV、global4 attentionのみINT8 | 386.91 | 400.17 | 11.68% | 2.692 | 3.744 |

all attention候補は画像別medianでも13.39% / 13.29%短縮。
global attention候補は11.17% / 12.36%短縮。両者とも各fresh processのmedianが現行より小さかった。
測定processの開始温度74〜76℃、終了75〜77℃で、Round 2より温度条件が揃った。
ただしclock固定や別日再測定はしていない。比較2画像では全timing processが品質gateを通るが、
難例screenではいずれも7/9なので **採用条件未達**。速度だけを成功としない。

[all attentionの全サンプル](../../experiments/results/native_rtx2060/r4-scope-timing-all-attention-summary.json)、
[globalのみ](../../experiments/results/native_rtx2060/r4-scope-timing-global-attention-summary.json)、
[事前加熱NVML・実行順](../../experiments/results/native_rtx2060/r4-scope-timing-protocol.json)。
対象はvision trunkのMLP/QKV/attention。FC2だけにscale/mean校正を適用し、FC1・QKVは
従来のrow INT8である。decoder等の精度は従来どおり。
GBは10^9bytes。whole GPUは標本による下限でprocess専有peakではない。
元のFP16重みを保持する研究実装なので、今回のINT8適用拡大はVRAM圧縮の主張ではない。

この結果から、MLPのINT8範囲を一律に大きく削るより、誤差を生むattention演算を
少数ずつFP16へ戻す方向にも速度上の余地がある。まず同じMLP0/QKV6構成で
attentionをすべてFP16とする9件の対照を追加し、次の層別attention実験の起点を確認する。

### 最後の対照・検証と次の反復

同じMLP0/QKV6構成でattentionをすべてFP16にしても7/9。
小さい人物maskは .98734へ改善するが、9590人物は5→6検出、spoonは .97872でFail。
[FP16 attention対照](../../experiments/results/native_rtx2060/round4-fp16-attention-summary.json)。
attentionの一律な精度変更だけで解決したとは言えず、QKV/MLPの組み合わせも再点検する。

全CTest **56/56、141.85秒**、selector専用test 5/5、3つのモデル対照はbyte一致。
今回の変更は層選択とdispatcherで、CUDA演算kernelは変更していない。
既存kernelへの新たなmemcheck結果を主張せず、Round 3の検査と今回の構成別モデル実行を区別する。
Python構文検査と `git diff --check` もPass。
[検証要約とバイナリSHA](../../experiments/results/native_rtx2060/round4-validation.json)、
[全CTest](../../experiments/results/native_rtx2060/round4-ctest.log)。

品質process112（対照を含む）＋timing process12、別途preheat1。
timingは3構成×60=180サンプル。使用画像は開発7枚、独立holdoutは未使用。
全候補は品質未達で採用しない。通常設定も変更しない。

前進した点は、1層単位のMLP/QKV切り分けを可能にし、検出数・score・小maskの
異なる感度を特定したこと、かつ品質調整へ使える11〜13%の速度差を同一条件で実測したこと。
次はMLP0 FP16＋QKV全層INT8＋all attention INT8の未測定3画像を補い、
QKV6をFP16にした際のmask悪化とattention側の誤差を切り分ける。
その結果に応じてattentionの層指定、または校正32枚からのQKV入力/LayerNorm校正へ進む。
開発例の偶然の誤差相殺だけを追わず、候補固定後の全33例・既存17例・独立評価と
再測定が揃うまで、ゴール完了とはしない。

## Round 5 — attentionの層ごとの切り分け

前roundは実装・速度診断・品質反証が増えたprogress。開始commit `a83c317`、
clean tree、GPU使用率0%、52℃、382MiBを確認した。
まずMLP block0 FP16 / QKV全層INT8 / attention全層INT8の未測定3画像を補完。
前roundの4promptと合わせて6/9で、小さい人物IoU .97769、羊box差1.02292px、
テレビscore/boxがFail。QKV6をFP16へ戻す前にも小さい人物maskの問題が存在する。
新規runは `r5-qall-1584/2261/5992`、既存は `r4-joint-qall-4795/9590`。

attentionに `SAM3_EXPERIMENT_ATTENTION_SCOPE` を追加する。
既存の共有mask parserを使い、対象外blockは既存FP16 SDPAへ戻す。
QKVの量子化とは独立で、デフォルトallは従来動作を維持する。
まずMLP0 / QKV6 FP16候補に対し、attention block0〜7をそれぞれ1層だけFP16に戻し、
1584/9590の4promptで切り分ける。閾値・校正データ・最終評価は変更しない。

新しい層選択のCPU testは6/6。all / global / mask0の各2promptを前roundの
all INT8 / global4のみINT8 / FP16 attentionと比較し、出力4ファイルが6/6 byte一致した。
[対照](../../experiments/results/native_rtx2060/round5-controls.json)。

追加の文献確認: [SageBwd（2026-03）](https://huggingface.co/papers/2603.02170)を
SciteとHF公式APIで照合した。今回はabstract/metadataまでの確認。
主題は訓練時の低bit attentionで、SAM3のPTQやSM75推論速度の実証として採用しない。
[SageAttention公式README](https://github.com/thu-ml/SageAttention/blob/main/README.md)
はAmpere/Ada/Hopper最適化、別系統のBlackwell FP4、量子化・smoothingを除いた
kernel TOPSを区別している。これらの倍率をRTX 2060の全体速度へ転用しない。
取得README blob SHAは `88a3fb78f881eb8253d842a31be91912dd5ede9b`。
手元のComfyKitchen adapterはQ/Kに加えてP/V側にもINT8を使い、Kはadaptive anchorで
中心化する設定。外部kernelにはfull-mean K smoothingも存在するが、現adapterでは未使用。
層選択で頭打ちになった場合の別仮説として残す。訓練におけるsmoothingの知見だけで
推論品質の改善を予断しない。

最初の8候補（attentionのblock0〜7を1層ずつFP16）は、4prompt中の合格数が
順に **4, 2, 2, 2, 4, 1, 4, 3**。
[全結果](../../experiments/results/native_rtx2060/r5-attn-exclude-summary.json)。
block0 / 4 / 6の3候補を残して、4795/2261/5992の5promptを補完する。
block3ではspoon IoU .99281でもscore差 .0200195のためFailのままとした。
小さな超過を許容するための閾値変更は行わない。

9promptまで広げた結果、attention block0 FP16は **9/9**、block4は **8/9**
（テレビscore差 .02734）、block6は **9/9**。
[9件の統合結果](../../experiments/results/native_rtx2060/round5-hard-summary.json)。
最小mask IoUの余裕がblock0 .98381 / block6 .98101のため、まずblock0候補を
開発16画像33promptへ広げる。MLPはblock0、QKVはblock6だけFP16、FC2校正は従来と同じ。
この時点で独立評価には進まない。

block0候補の全開発結果は **31/33**（非空23/25）。検出数とmask IoUは全件基準内だが、
7574 bowlのscore差 .029785、1425 bowlのbox差2.317856pxがFail。
[全33件](../../experiments/results/native_rtx2060/r5-a0-development.json)、
[要約](../../experiments/results/native_rtx2060/round5-a0-development-summary.json)。
FP16一致への改善は特定の難例に限られ、候補全体の採用基準には届かない。
次にattention block6 FP16候補を、この2画像5promptで比較する。

既存17例の再評価用に `run_quant_legacy_regression.py` を追加した。
保存済み候補の環境・バイナリ・校正hashを照合し、同じバイナリのfresh FP16と
候補を各例で連続実行する。旧5例＋追加12例の画像/prompt存在と構文は確認済み。
この時点ではまだモデル実行しておらず、17例の合格を主張しない。

attention block6 FP16候補では、7574 bowlのscore差は .014648へ収まるが、
1425 bowlのbox差1.740356pxが残った（追加5件中4件合格）。
[追加比較](../../experiments/results/native_rtx2060/r5-attn-bowls-summary.json)。
次にQKV6を含め全QKVをINT8へ戻し、attention block0 / 6 FP16の2候補を
1425/4795/9590の6promptで比較したところ **4/6 / 3/6**。
どちらも1425 bowlのbox差（1.458527 / 2.070831px）と9590人物の5→6検出がFail。
block6ではテレビscore差 .020508もFail。
[再組み合わせ](../../experiments/results/native_rtx2060/r5-qall-attn-summary.json)。

### 解釈と次の反復

層ごとの精度変更で小さいmaskの誤差と検出数の不一致を解消できた例はあるが、
9件の成功は33件への一般化を保証しなかった。31/33の合格数自体は現行selectiveと同じで、
失敗するpromptが入れ替わっている。今回の候補は採用しない。
新規の速度測定は行っていないため、Round 4の13.25%を今回の候補の実測値にしない。
独立holdout 32枚68promptは未使用のまま。

次は同じ失敗画像への層mask探索だけを増やさず、Kのfull-mean smoothingと、
校正32枚からのQKV入力チャネルscale/mean補正を順に切り分ける。
前者は現在anchor中心化を使うadapterの代替で、外部kernelのscratch/current stream/
determinismをoperator検証してからモデルへ適用する。後者は校正画像だけで統計を取得し、
元のFP16重みとの等価変換と量子化誤差を分離して確かめる。
いずれも新しい速度・精度の結果はまだなく、INT4やSAM3.1動画への有効性も未証明。

今回のモデル評価はquality 108 process / 40 run、開発16画像、子process時間合計702.64秒。
最初のQKV全層対照5件だけ前round DLL、それ以降の103件は新DLLを使用し、
各runのsignatureに区別して保存した。最大allocated 2.695GB、whole GPU標本最大3.761GB。
校正の再収集やtiming測定は行っていない。
[process集計](../../experiments/results/native_rtx2060/round5-process-summary.json)。

再現例（nameは新しい名前を指定）:

```powershell
.venv/Scripts/python.exe experiments/run_quant_attention_screen.py --name <unique> --scopes mask:0xfffffffe mask:0xffffffbf --images 1584 4795 2261 5992 9590
.venv/Scripts/python.exe experiments/run_quant_research.py quality --mode calibrated --scope mask:0xfffffffe --calibration .cache/quant-research-20260924/calibrations/a050-none-mean-bias --mean-bias --projection-scope mask:0xffffffbf --attention-scope mask:0xfffffffe --name <unique>
```

第1コマンドは今回9/9だった2候補の難例比較、第2コマンドはblock0候補の全33件。
通常のruntime defaultは変更していない。

### 検証と到達点

全CTest **57/57、141.36秒**、selector 6/6、従来3構成との6出力対照はbyte一致。
Python構文検査と `git diff --check` もPass。
新しいCUDA kernelは追加・変更していないため、新規memcheckの実施は主張しない。
重みmanifestのSHA256は研究開始時と一致した。
[検証要約](../../experiments/results/native_rtx2060/round5-validation.json)、
[CTest](../../experiments/results/native_rtx2060/round5-ctest.log)。

attentionの単層切り分け、108評価、33件での反証結果が増えたprogressとして継続する。
採用構成は未確定であり、ゴールは未達。準備した既存17例driverのモデル実行、
候補固定後の独立評価、速度の再測定、SAM3.1への別評価は引き続き必要。

## Round 6 — Kの平均中心化

前roundは層別実装と108評価・反証を追加したprogressとして継続。
開始commit `3e2b844`、clean tree、GPU使用率0%、56℃、382MiBを確認。
品質・速度閾値と校正/開発/holdoutの分離は維持する。

現在のComfyKitchen adapterはadaptive anchorによるK中心化を使う。
同じ固定版の公開launcherに存在するfull-mean K smoothingへ切り替える選択肢を追加した。
`SAM3_EXPERIMENT_KITCHEN_CENTER=anchor`（従来default）/ `mean` / `none`。
`mean`はFP32 `[B,H,64]` scratchとINT32 `[B,H]` counterを用い、launcherが初期化、
Kの平均計算、量子化を同じcurrent streamで実行する。外部kernel本体は変更しない。
全keyに共通のベクトルを引くとQKの各rowから同じ定数が引かれ、厳密なsoftmaxは不変。
有限精度の丸め・INT8量子化は別途検証する。

新規operator checkはN=1/65/129/576/5184、B=1/2/9、通常・大きなKオフセット・
token間で一定のK・全zero、rotation有無を含む40条件。
N<=129はCPU FP64の明示的なmatmul/softmaxと比較し、中心化前後の数学的同値も確認する。
productionサイズは既存SDPA、一定Kではmean(V)を参照とする。
非default CUDA streamで各条件を5回再実行して、byte一致と最大差を記録する。
誤差閾値は既存operator checkと同じmax abs .05 / RMSE .005。
中心化なしは大offset時の反証用で、誤差超過を記録する。モデル品質gateとは別である。

初回operator結果: mean 40/40、anchor 40/40、none 32/40。
最大abs/RMSEはmean .026245/.002494、anchor .023926/.002762、none 3.11328/.448709。
noneの大きな誤差はオフセットを含む入力の診断であり、通常SAM3入力の精度測定ではない。
meanのmemcheckは0 errors。
一方、N=5184のrandom/shifted（rotation有無）の4条件で、5回すべてにbyte差があった。
最大再実行差 .000259399。公開mean kernelは512rowごとのFP32部分和をatomic加算するため、
加算順による変動がある。単発の誤差基準Passだけでは再現性を満たしたと扱わない。

比較として `mean_half` を追加する。ATenのFP32 meanで中心を計算し、FP32で引いたKを
FP16へ丸めた後、中心化なしの既存INT8 quantizerへ渡す。
これは追加のK実体化・丸めを含む別経路で、fused meanとbyte同値を主張しない。
40条件の誤差と5回再実行、追加コストを別に評価してからモデルへ進む。

mean_half追加後（scope追加前）のビルドで4モードを再確認すると、mean_halfは40/40、max abs .026245、RMSE .002496。
40条件×5回の200再実行はすべてbyte一致。anchorも40/40・200回一致。
fused meanは誤差基準40/40だが19/200再実行が変化し、最大差は引き続き .000259399。
今回のモデル候補には再実行が安定したmean_halfを使い、fused meanは診断用に残す。
異なるGPU/LibTorch版や任意入力での決定性を保証したものではない。
[数値とこの検査に使ったバイナリSHA](../../experiments/results/native_rtx2060/round6-center-summary.json)。

mean_halfのmemcheckも0 errors。
[検査log](../../experiments/results/native_rtx2060/round6-mean-half-memcheck.log)。
新バイナリのanchorを前roundのblock0候補と比較し、1425/9590の4promptで4出力がbyte一致。
[既定値対照](../../experiments/results/native_rtx2060/round6-controls.json)。

同じMLP0/QKV6/attention0 FP16構成でcenterだけmean_halfへ変えると、
7574 bowlのscore差が .029785→.010742、1425 bowlのbox差が2.317856→.482269pxへ改善し、
2画像5promptが5/5合格した（非空4/4）。この結果を根拠に開発全33件へ広げる。
`r6-mean-half-a0-7574/1425` が初期screen、`r6-mean-half-development` が全開発評価。
独立holdoutはまだ使わない。

全33件では **31/33（非空23/25）**。bowl2件は改善したが、羊box差1.651917pxと
9590人物5→6検出がFailに入れ替わった。全mask IoUとscore差は基準内。
[全開発要約](../../experiments/results/native_rtx2060/round6-mean-half-development-summary.json)。
一律mean_halfは採用しない。

global attention（N=5184）とlocal window attention（N=576）では中心の推定対象が
異なるため、2群の切り分けを追加する。`SAM3_EXPERIMENT_KITCHEN_CENTER_SCOPE`
のall/default、local、globalでmeanを使う群を指定し、他方は従来anchorを使う。
精度bit数やMLP/QKV scopeは変えない。local/global指定はこの2つのSAM3形状に限定し、
他のtoken長は拒否する。任意の1層mask探索へは広げない。

候補を固定した後に速度を比較できるよう、`run_quant_candidate_timing.py` も追加した。
保存済みquality runの全precision引数・バイナリ・データ・校正hashを引き継ぎ、
現行selectiveの事前加熱後、既定4画像×2構成×2 fresh process、各5 warmup＋15測定を
順序反転して比較する。まだ実行しておらず、今回の速度改善を主張しない。

### 群別の切り分け結果

scope追加後のall設定を1425の2promptで再実行し、scope追加前のmean_halfと
全4出力ファイルがbyte一致した。
[対照](../../experiments/results/native_rtx2060/round6-scope-controls.json)。
7574/1425/5992/9590の8promptではlocalのみmean_halfが **7/8**、globalのみが **5/8**。
localでは羊box差が .730865pxへ収まり、残りは9590人物の5→6検出。
globalでは7574 bowl score差 .024902、1425 bowl box差1.280396px、
9590 spoon IoU .978571 / score差 .024902がFailとなった。
[8件比較](../../experiments/results/native_rtx2060/round6-center-scopes-summary.json)。

local候補を全開発33件へ広げると **31/33（非空23/25）**。
9590人物の検出数に加えて、1353人物のmask IoU .971460、box差2.583389pxがFailだった。
最大score差 .016602は基準内。local限定でも採用基準を満たさない。
[全開発要約](../../experiments/results/native_rtx2060/round6-mean-local-development-summary.json)。
一律meanとlocal meanは合格数が同じだが、失敗画像が異なる。この結果も開発データの
反証として残し、独立holdoutを使った成功率には数えない。

### 平均中心化のコスト

`run_kitchen_center_bench.py` を実行した。anchor→mean_half→mean_half→anchorの
4 fresh process、各演算100 warmup後20 CUDA event測定、4 pass/process。
以下は各方式8個のpass中央値の中央値であり、全標本をpoolした中央値やp95ではない。
native既存benchのhead-major出力を使う。モデルのsequence出力・end-to-end wallとは別。

| 形状（B×N、H=16、D=64） | anchor | mean_half | 増加 |
|---|---:|---:|---:|
| local 9×576 | 0.945456 ms | 1.500280 ms | +0.554824 ms（+58.68%） |
| global 1×5184 | 4.282195 ms | 4.796895 ms | +0.514700 ms（+12.02%） |

rotationなしの量子化・中心化を含むoperator時間。事前に熱平衡まで加熱しておらず、
whole processの温度は55〜73℃に変化した。診断上はATen meanとK実体化の追加コストが
見えるが、モデル全体の速度差や採用条件の達成をこの値から主張しない。
[数値と限界](../../experiments/results/native_rtx2060/round6-center-micro-summary.json)、
[実行順序・NVML・バイナリhash](../../experiments/results/native_rtx2060/r6-center-micro-protocol.json)。

今回のqualityは **93 process / 15 run / 16画像**、子process時間合計602.95秒。
最大allocated 2.692GB、whole GPU標本最大3.761GB。scope追加前42件、追加後51件を
各runのDLL hashで区別した。microbench 4 processはこの93件に含めない。
校正の再収集とモデル全体のtimingは未実施。
[process集計](../../experiments/results/native_rtx2060/round6-process-summary.json)。

再現例（nameは新しい名前を指定）:

```powershell
.venv/Scripts/python.exe experiments/run_quant_research.py quality --mode calibrated --scope mask:0xfffffffe --calibration .cache/quant-research-20260924/calibrations/a050-none-mean-bias --mean-bias --projection-scope mask:0xffffffbf --attention-scope mask:0xfffffffe --kitchen-center mean_half --kitchen-center-scope local --name <unique>
.venv/Scripts/python.exe experiments/run_kitchen_center_bench.py --name <unique>
```

### 次の反復

平均中心化だけでは固定gateを満たせず、今回の方式は採用しない。
次は校正32枚のFP16経路からQKV入力のチャネル統計を取り、入力scale/shiftと
重みmean bias補正を調べる。norm1のgamma/betaへ等価変換を折り込む案なら実行時の
追加affine演算を避けられるが、FP16の丸め位置が変わるため数学的同値だけで判断しない。
identity変換、FP64参照、実装のbias適用箇所、実モデルの品質を別々に検証する。
このQKV校正はまだ未実装・未測定である。

独立holdout 32枚68promptと既存17例の最終回帰は未使用のまま。
現在のmean_halfの結果をINT4やSAM3.1へ一般化しない。通常のruntime defaultは変更していない。

最終検証: 全CTest **60/60、144.01秒**、selector 9/9、対照6件byte一致。
Python構文検査・`git diff --check` はPass。データと重みmanifestのSHAも維持した。
40条件operator/memcheckはscope追加前のビルド、最終scopeビルドはselector・対照・
microbenchの数値/layout check・モデル51件・CTestで検証し、hashを混同せず記録した。
[検証要約](../../experiments/results/native_rtx2060/round6-validation.json)、
[CTest log](../../experiments/results/native_rtx2060/round6-ctest.log)。
研究上の反証と再現性検証が増えたprogressとして継続する。採用構成は未確定、ゴールは未達。

## Round 7 — QKV入力のチャネル校正

開始commit `5f72470`、clean tree、GPU利用率0%、56℃、382MiBを確認。
前roundはK中心化の93評価、operator costと再現性の検証を追加したprogressとして継続する。
固定品質gate、split、従来selective速度対照は維持する。

### 変換と実装

FP16経路のnorm1出力から、各blockのQKV入力チャネル1024個についてmin/max/mean/RMSと
重み列absmaxを記録する。観測は校正32枚限定、1画像1processで各blockの初回だけ。
開発画像やholdoutは統計に混ぜない。

`x' = x/r - shift`, `W' = W*r`, `b' = b + W'@shift` の等価変換を使う。
norm1のgammaをgamma/r、betaをbeta/r-shiftへsetup時に変更することで、
推論時の追加affine演算を避ける。INT8対象外のQKV blockはnorm/重み/biasとも変更しない。
QKV restore/RoPEのfused経路と、通常のquantized linearの両方に補正biasを渡す。
既定のFP16経路および校正未指定のINT8経路は従来の値を使う。

`r` はactivation/weightのチャネル最大値からalphaで分配し、幾何平均で正規化、
[1/16,16]にclipする。shiftなし／チャネル平均shiftを分離する。
norm affineはFP32、変換重みと補正biasはFP16保存。foldがHalfへのcastより先なので、
既にHalfへ丸めたxを変換する方法とのbit同値は仮定しない。
任意のmean biasは `b + W_original@mu - W_dequantized@(mu/r-shift)` をFP32で計算し、
Halfへ丸める。これは重み量子化の平均誤差だけを補正し、活性量子化や上流の誤差は扱わない。
元のFP16重みを保持しており、weight VRAM圧縮の実証ではない。

一次実装の再確認: [SmoothQuant公式smooth.py](https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py)
のblob SHA `c2a70e70fe645c69f970392940511da1465ed71d`をGitHub connectorで取得。
LayerNormと全消費先Linearのscale移動が確認できる。
[ICML 2023原論文](https://proceedings.mlr.press/v202/xiao23c.html)はLLMのW8A8の結果であり、
SAM3/SM75の品質・速度を保証する資料としては使わない。
Sciteで同論文を照合（支持0／反証0／言及80）、HF公式paper APIでもタイトルと
2022-11-18初公開日を確認した。引用件数は独立再現の証明にはしない。
shift、clip、FP32 norm foldingとmean biasは今回の実験上の選択として区別する。
[照合記録](../../experiments/results/native_rtx2060/round7-research-smoothquant.json)、
[HF metadata](../../experiments/results/native_rtx2060/round7-hf-smoothquant.json)。

### 基礎検証

専用CPU/CUDAテストを追加し、最初のbuildで2/2 Pass（2.54秒）。
独立FP64参照によるnorm+QKV等価変換の最大差7.55e-15、量子化重みの経験平均を
補正した差1.38e-6。恒等変換でnormパラメータとCUDA投影がbyte一致する。
global/local用のnorm出力（72×72 grid）はFP64参照とのmax差 .003822、RMSE .0002245。
残差から次blockのnormを準備する経路も、同じ残差和の直接normとbyte一致。
実QKV寸法5184×1024×3072のinteger GEMMを8行のCPU INT64内積で照合し完全一致、
bias復元はFP64参照との最大差 .0009752以下。非default CUDA streamで実行した。
これらはoperator検証であり、モデル品質合格や速度改善の主張ではない。
[詳細log](../../experiments/results/native_rtx2060/round7-qkv-operator-detail.log)。

校正32枚の収集は完了（子process合計295.85秒）。QKV observer追加後も、従来の
FP16校正出力と32/32件で全4ファイルがbyte一致した。
[観測要約](../../experiments/results/native_rtx2060/round7-qkv-observe-summary.json)、
[FP16対照](../../experiments/results/native_rtx2060/round7-observer-controls.json)。
1024観測ファイルのSHAと3種類の校正を
[provenance](../../experiments/results/native_rtx2060/round7-qkv-calibration-provenance.json)に記録。
scaleの全block範囲はshiftなし .079790〜2.890540、mean shiftあり .0625〜3.688550、
後者の最大abs shiftは6.651053。FP16校正入力の最大absは34.4375。

恒等校正を従来候補と対照した後、同じMLP0/QKV6/attention0 FP16の
anchor構成上で、mean biasのみ／scaleのみ／scale+mean bias／shift+scale+mean biasを
開発7画像15promptで比較する。holdoutは使わない。

モデルintegration対照を1425の2promptで実施した。新旧default、恒等校正、
QKV全層をFP16へ除外した際の校正有無、補正済みbiasを使うfused/unfused RoPEの
4組×2件、すべて全4出力ファイルがbyte一致した（fresh model 12 process）。
[対照結果](../../experiments/results/native_rtx2060/r7-qkv-controls-summary.json)。
scale+mean shift+mean biasでは1425 bowlのbox差が2.317856→.300903pxへ縮まった。
これは最初の1画像の観察であり、全開発評価を通過した意味ではない。

追加調査では[GCQ-ViT](https://journals.sagepub.com/doi/10.3233/FAIA250834)の
出版社HTML（手法・実験設定）とSciteを照合した。ページのonline日は2026-08-25だが
巻はECAI 2025、Scite年も2025であり、単に「2026年の新方式」とは数えない。
LayerNorm後のgroup化、logarithmic softmax、次元別の平均誤差補正を組み合わせる。
実験はViT/DeiT/SwinとSwin検出・segmentation、RTX 3090。確認した本文にGitHubリンクはなく、
SAM3/SM75のinteger kernelや全体latencyを検証できた資料ではない。
group scaleを現在の1つのINT32 GEMM結果へそのまま適用することもできない。

そこから辿った[Bias Compensation原論文](https://arxiv.org/abs/2404.01892)のabstractと
[公式実装](https://github.com/GongCheng1919/bias-compensation/blob/main/bias_compensation/quantizers/BiasCompensation.py)
（blob `709baaa9948e2535e743e66899c7ebc54d0d6ffc`）を確認。
校正時の浮動小数点出力と量子化出力の平均差を、出力チャネルごとのbiasへ蓄積している。
現在のweight-only mean補正が省いているactivation誤差を、実出力で測って補う次の仮説になる。
この方式のSAM3実装・実測はまだない。MTLQ-ViT（2026）もScite metadataで確認したが、
出版社本文取得に失敗したため、手法の採否は保留。
[追加照合記録](../../experiments/results/native_rtx2060/round7-bias-followup.json)。

### 4方式の比較

固定7画像15prompt（非空12）での結果は以下。

| QKV校正 | 合格 | 主なFail |
|---|---:|---|
| r=1、shift=0、mean biasのみ | 11/15 | 1584人物IoU、4795テレビscore、9590人物の検出数/IoU/score、spoon score |
| alpha .5、shiftなし、mean biasなし | 12/15 | 7574 bowl score、1425 bowl box、9590 spoon IoU/score |
| alpha .5、shiftなし、mean biasあり | 12/15 | 1584人物IoU/score、4795テレビscore、9590人物検出数 |
| alpha .5、mean shift、mean biasあり | 14/15 | 1584人物IoU .969636 |

[全60評価](../../experiments/results/native_rtx2060/r7-qkv-screen-summary.json)。
全方式ともこの時点では採用しない。最後の候補は検出数・score・boxが全15件基準内で、
9590 spoon IoUは1.0。残る1584人物maskの誤差がQKV校正とINT8 attentionの組み合わせに
依存するか確認するため、同じQKV/MLP構成でattentionをFP16にして同じ15件を評価する。
これは開発上の相互作用の切り分けであり、独立評価や速度の実証ではない。

attentionを全FP16にした結果は **12/15**。1584人物IoUは .986755へ戻ったが、
7574 bowl score差 .027832、9590人物5→6検出、spoon IoU .978723がFail。
[FP16 attention比較](../../experiments/results/native_rtx2060/r7-qkv-fp16-attention-summary.json)。
次にglobal4層だけINT8 attentionにした結果は **13/15**。1584人物は .989451だが、
1425 bowl box差1.722488pxと9590 spoon IoU .978571がFail。
[global attention比較](../../experiments/results/native_rtx2060/r7-qkv-global-attention-summary.json)。
丸めを減らせば全指標が単調に改善する状況ではなく、これらの構成も採用しない。

1584の失敗maskを個体ごとに調べると、FP16面積493pxの人物（query146）に対して
14/15候補は15px不一致、union494pxでIoU .969636。旧Round5候補は8px不一致だった。
比較は元解像度612×612で行い、maskの縮小やgateの緩和はしていない。
これはFP16との一致の分析であり、GTに対する正誤を意味しない。
[個体別診断](../../experiments/results/native_rtx2060/round7-small-mask-diagnostic.json)。

最良だった元のattention設定＋QKV scale/mean shift/mean biasを、旧Round5/6と
評価範囲を揃えるため全33件でも確認する。既知のFailがあるので採用・holdout解禁には使わず、
局所改善が他の画像で維持されるかの診断として扱う。

再現コマンド例（calibration出力は既存directoryを上書きしない。新規cacheへの作成時に使用）:

```powershell
.venv/Scripts/python.exe experiments/run_quant_research.py observe --role calibration --mode fp16 --observe-target qkv --name <observe-name>
.venv/Scripts/python.exe experiments/make_qkv_calibration.py .cache/quant-research-20260924/runs/<observe-name> .cache/quant-research-20260924/calibrations/qkv-identity --identity
.venv/Scripts/python.exe experiments/make_qkv_calibration.py .cache/quant-research-20260924/runs/<observe-name> .cache/quant-research-20260924/calibrations/qkv-a050-none
.venv/Scripts/python.exe experiments/make_qkv_calibration.py .cache/quant-research-20260924/runs/<observe-name> .cache/quant-research-20260924/calibrations/qkv-a050-mean --shift mean
.venv/Scripts/python.exe experiments/run_quant_qkv_controls.py --name <controls-name>
.venv/Scripts/python.exe experiments/run_quant_qkv_screen.py --name <screen-name>
.venv/Scripts/python.exe experiments/run_quant_qkv_screen.py --name <interaction-name> --variants shift_bias --attention kitchen
```

単一の全開発候補:

```powershell
.venv/Scripts/python.exe experiments/run_quant_research.py quality --mode calibrated --scope mask:0xfffffffe --calibration .cache/quant-research-20260924/calibrations/a050-none-mean-bias --mean-bias --projection-scope mask:0xffffffbf --attention-scope mask:0xfffffffe --qkv-calibration .cache/quant-research-20260924/calibrations/qkv-a050-mean --qkv-mean-bias --name <development-name>
```

### 全開発セットの結果と次の仮説

全33件の結果は **32/33**（非空24/25）。検出数は全件維持され、最大score差 .01708984、
最大box差 .974121px。残るFailは1584人物のmin mask IoU .96963563だけだった。
旧Round5/6の31/33から合格数は増えたが、固定gate未達のため採用しない。
[全開発集計](../../experiments/results/native_rtx2060/round7-qkv-shift-development-summary.json)。
先行screenと重なる15件は、別processでの全開発再実行と出力4ファイルがすべてbyte一致した。
[再実行対照](../../experiments/results/native_rtx2060/round7-repeat-controls.json)。

次はQKVのFP16出力と実INT8演算の出力との差を校正32枚で測り、出力チャネルごとの
平均誤差をbiasへ折り込む。現方式のweight-only補正で扱えない、norm foldingの丸めと
activation量子化の誤差を含める仮説である。FP16の上流を維持したshadow計算なら、
上流blockの量子化誤差までは補正していないと明記する。
校正は開発例や独立holdoutから採取せず、対象・式・評価条件を先に固定する。
学習・校正の追加コストと推論時の追加演算を区別して記録する。

次の実験の固定案: Round7のalpha .5/mean shiftとweight mean biasを初期値にする。
各FP16 blockへの残差入力から、fold済みnormと実INT8 QKVをshadow実行し、RoPE前の
`delta = mean(QKV_fp16 - QKV_int8)` を出力3072チャネルについて採取する。
同じtoken数の32校正画像を等重み平均し、`bias_new = Half(bias_old + mean_image(delta))`
をsetupで適用する。全INT8対象QKVに同じ規則を使い、開発例別の補正はしない。
shadow計算で本来のFP16出力が変わらないこと、独立した数値参照、補正なしの再現性、
fused/unfused経路を先に検証する。出力Halfへの再丸めにより平均誤差が厳密に0になる
保証はなく、最終判断は同じ開発gateと、その後の独立評価・速度評価で行う。

Round7では全体latencyを計測していない。以前の候補の速度をこの構成の実測値として
流用しない。独立holdout 32枚68promptと既存17例の最終回帰は未実行である。
INT4とSAM3.1への有効性も未確定で、研究ゴールは継続する。

### Round7の検証記録

全CTest **62/62、185.32秒**、QKV CUDAのcompute-sanitizer memcheck **0 errors**。
校正観測32/32、パラメータ・経路対照8/8、screen再実行15/15のbyte一致を確認した。
Python構文検査と`git diff --check`もPass。weight/data manifest SHAは変わらず、
全167モデルprocessで同じDLL SHA `c87c030a417a7e9f1276b3fd56cff076fc4c68cc1cd87b9cfa8bee30155d6dba`
を使用した。CUDA kernel本体と通常のruntime defaultは変更していない。
[検証要約](../../experiments/results/native_rtx2060/round7-validation.json)、
[CTest](../../experiments/results/native_rtx2060/round7-ctest.log)、
[memcheck](../../experiments/results/native_rtx2060/round7-qkv-memcheck.log)。

qualityは **135 process / 49 run / 16画像**、子process時間合計1,256.68秒。
最大cold allocated 2.695GB、whole GPU標本最大4.256GB。観測32 processは別に295.85秒、
最大allocated 2.291GB、whole GPU標本最大3.709GB。NVMLは20ms間隔のwhole device標本で、
厳密なprocess peakではない。推論時のbias折り込みに追加GEMMはないが、setup・校正は
別コストとして記録し、cold quality時間から速度の採否は判断しない。
[process集計](../../experiments/results/native_rtx2060/round7-process-summary.json)。

FC2と共通化したmean bias helperの回帰も、幅1/31/32/257/1024/4736/8192の7条件で確認。
境界量子化・affine・FP16 GELUの比較は全件一致、独立FP64参照に対するweight mean補正の
最大差は1.997e-6。これは数値検証のみでtimingは実行していない。
[FC2回帰結果](../../experiments/results/native_rtx2060/round7-fc2-boundary.json)。
