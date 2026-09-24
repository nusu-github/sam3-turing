# SAM3 / SM75 の INT8・INT4 再調査

調査日: **2026-09-24**。ローカル HEAD: `cfc27c6f971aab8680eba1ac2929b4c904c49259`。
対象は SAM3 / SAM3.1、Windows、RTX 2060 Max-Q 6 GB、C++/CUDA ランタイム。
これは文献・公開ソース・既存実験の照合であり、今回新しい量子化や速度測定は実行していない。

## 今回の判断

**次の優先候補は、校正したチャネル単位の変換と、層ごとの FP16 / W8A8 選択を組み合わせる方法。**
INT4 はその後に、GELU 後の活性誤差を抑えられた FC2 から試す。
一律 INT8、単純 absmax W4A4、回転だけの追加を繰り返す根拠は弱い。
この優先順位は下記の研究と手元の結果からの提案であり、SAM3 での効果は未検証。

特に更新すべき点は次の三つ。

- **Mix-QSAM3 の一次資料を全文確認できた。** 以前の調査では取得できなかった PDF を取得し、表1を画像でも確認した。SAM3 に直接対応する混合精度研究だが、SM75 の実時間を保証するものではない。
- **SAQ-SAM の再構成ありの公表精度には評価上の不具合がある。** 著者 README 自身が認めており、修正済みの再評価なしには4bit精度の裏付けに使えない。
- **現行 ComfyKitchen に SM75 の INT4 実装が存在する。** カーネルの有無と、SAM3 全体の精度・速度はそれぞれ確認する必要がある。

## 研究の比較

W は重み、A は活性のビット幅。W4A16 の重み圧縮、W4A8、W4A4 の整数演算は別の方式として扱う。

| 研究 | 公開時期・対象 | 参考になる方法 | このプロジェクトでの位置付け |
|---|---|---|---|
| [Mix-QSAM3](https://openaccess.thecvf.com/content/CVPR2026W/AIGENS/html/Ranjan_Mix-QSAM3_Mixed-Precision_Quantization_for_the_Segment_Anything_with_Concepts_Model_CVPRW_2026_paper.html) | CVPR Workshops 2026、SAM3 | KL に基づく情報保持と、隣接層の感度パターンを使うビット配分 | **選択的 INT8 の設計で最優先**。計算量の制約を手元の実測コストへ置き換えて検討 |
| [UQ-ViT](https://ojs.aaai.org/index.php/AAAI/article/view/39393) | AAAI 2026、ViT | NormQuant による GELU 後のチャネル差の補正、DeMax による Softmax 最大値の扱い | **FC2 の校正候補**。まず NormQuant 側を切り出す。SAM3 の実証ではない |
| [SAQ-SAM](https://ojs.aaai.org/index.php/AAAI/article/view/38249) | 2025年プレプリント / AAAI 2026、SAM | attention の意味を保つ clipping、画像・prompt を使う再構成 | アイデアは有用。ただし再構成ありの精度結果は下記の不具合のため保留 |
| [CAR-SAM](https://arxiv.org/abs/2605.16901) | 2026-05-16 プレプリント、SAM / SAM2 | MatMul の活性誤差を前段へ補償、cross-attention の共同再構成 | Decoder を低精度化する後段候補。SAM3 / SM75 で未実証 |
| [AHCPTQ](https://arxiv.org/abs/2503.03088) | ICCV 2025、SAM | post-GELU の log/uniform 混合量子化、分布が近いチャネルの grouping | 分布解析を参考にする。FPGA の速度を CUDA に転用しない |
| [ConvRot](https://huggingface.co/papers/2512.03673) | 2025-12、Diffusion Transformer | regular Hadamard で活性・重みを回転し W4A4 化 | 実装候補。SAM3 の精度保証ではなく、既存 H16/H64 実験との差分が必要 |

### Mix-QSAM3: 直接的な根拠と限界

[本文 Table 1、PDF p.6](https://openaccess.thecvf.com/content/CVPR2026W/AIGENS/papers/Ranjan_Mix-QSAM3_Mixed-Precision_Quantization_for_the_Segment_Anything_with_Concepts_Model_CVPRW_2026_paper.pdf) の SA-Co Gold 7 subset 平均:

| 設定 | cgF1 |
|---|---:|
| SAM3 FP32 | 54.1 |
| 一律 INT8 | 47.9 |
| Mix-QSAM3 MP8 | 52.4 |
| 一律 INT6 | 45.0 |
| Mix-QSAM3 MP6 | 48.8 |

MP8 は全層8bitという意味ではない。候補 `{2,4,6,8,16}` から、モデルサイズと BitOps を INT8 相当に制約して選ぶ。入力・出力層は FP16、校正は32枚、評価 GPU は RTX 3090 Ti。隣接依存は単層の感度曲線の JS 類似度で近似している。SM75 の end-to-end latency 表や、今回確認できた公式実装リンクはない。4bit が全体で成功した証拠にもならない。

適用案: 実装可能な `{FP16,W8A8,W4A4}` を候補にし、BitOps に加えて量子化・復元・境界変換を含む実測コストで選ぶ。単層選択後には累積誤差を再評価する。

### 公表された4bit精度の扱い

[SAQ-SAM 公式 README の固定版](https://github.com/jingjing0419/SAQ-SAM/blob/14b7055ee569f94640553347267494030f11f6c7/README.md#bug-in-baseline-framework) は、再構成ありの評価で `drop_prob` が1に戻らず、一部活性が非量子化のままだったと説明している。著者は自身の再構成結果も影響を受けると明記。PCC のみ・再構成なしは対象外としている。したがって「4bitでほぼ無劣化」を確立済みの前提にしない。これは [AHCPTQ 著者の指摘](https://github.com/Keio-CSG/AHCPTQ#4-bug-in-original-ptq4sam-framework) とも整合する。

[CAR-SAM 本文](https://arxiv.org/html/2605.16901v1) は SAM / SAM2 を評価し、正解 box prompt、32校正画像、140,000最適化ステップを使用する。SAM-B の W4A4 は39.3 mAPで、FPの55.8との差が残る。比較手法を上回ることと FP 品質を保つことは別。速度節の倍率は図の計算量から説明されており、SM75 の実測 latency として採用できない。公式実装は今回確認できなかった。

### UQ-ViT を試す意味

[公開コード](https://github.com/jiujiuwei/UQ-ViT/blob/3d7d4bb2e7eb06e6c858fd6e33f6f9adab47beed/classification/quant/quant_modules.py) の NormQuant は入力チャネルごとのスケール・シフトと、それに対応する重み・bias 補正を行う。任意の Linear 入力に適用でき、GELU 後の FC2 入力が今回の関心に合う。
一方、公開 `QuantLinear` は量子化した値を `F.linear` に渡す参照実装であり、SM75 整数カーネルの実証ではない。方法の移植、固定済み校正パラメータ、整数計算への落とし込みが必要。

等価変換の例は、行ベクトル入力で `x' = x / r - b`、`W' = W * r` とし、元の bias に `W' b` を加える形。量子化前の積は保存されるが、有限精度の丸めは別に検証する。FC2 入力への変換を GELU より前へ無条件に移動してはいけない。

## GitHub: 実際に実行するカーネルを確認

調査時の ComfyKitchen `main` は `f61028a7b0f4be4beb3595c64ee919e0c345f96d`。

- [SM75 INT4 GEMM](https://github.com/Comfy-Org/comfy-kitchen/blob/f61028a7b0f4be4beb3595c64ee919e0c345f96d/comfy_kitchen/backends/cuda/ops/turing_int4.cu): CUTLASS の packed signed INT4、スケール・bias 復元を含む。K整列等の条件がある。
- [実際の dispatch](https://github.com/Comfy-Org/comfy-kitchen/blob/f61028a7b0f4be4beb3595c64ee919e0c345f96d/comfy_kitchen/backends/cuda/__init__.py): SM、M、ビルドされた関数、fallback 設定で分岐する。INT4 ファイルを読み込んだだけで実行命令まで INT4 と断定しない。
- 同ソースの ConvRot は回転 group 16/64/256 を扱い、K の整除性を確認する。SAM3 FC2 の K=4736 は16/64で割れるが256では割れない。これは既存 H16/H64 実験と対応する。
- **回転の group size と量子化スケールの粒度は別**。公開 CUDA 経路は `rowwise` quantization と row/channel scale を使う。H64 に変えただけで K方向の64要素別スケールになるわけではない。
- 既存実験が固定した [PR #103](https://github.com/Comfy-Org/comfy-kitchen/pull/103) は2026-08-11に merge 済み。固定 head は `215cc5a8e52f163bf8d4631030f452767c737ab5`。現行 main との差分には後続の attention 最適化や Sol-attention 等があるが、すべてを同一の量子化変更として取り込まない。

[NVIDIA の Turing 資料](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#tensor-core-operations) も INT8 / INT4 入力と INT32 累算を確認できる。ハードウェア対応は既存実機実験とも一致する。

K方向に細分化したスケールを採る場合は、各 group の整数部分和をそれぞれ復元して加える必要がある。現在の GEMM 結果全体に一つの外積スケールを掛けるだけでは実現できない。精度改善と追加の部分和・復元コストを対で評価する。

## Hugging Face: 配布モデルの到達点

[honeysandhu/sam3.1-int8-int4-convrot](https://huggingface.co/honeysandhu/sam3.1-int8-int4-convrot) のモデルカードを確認。revision は `33937842a75da2c59dc7fe7553fb40920108b565`、最終更新は2026-08-23。

選択した230個の Linear に回転と MSE 最適化した重み clipping を適用し、活性は実行時に量子化。CLIP や敏感な層を FP16 に残す。配布容量は W8A8 1.19 GiB、W4A4 0.98 GiB、FP16参照1.63 GiB。
作者の RTX 3090 / ComfyUI / 2回 refinement の測定は1.933 / 1.647 / 1.396秒（FP16 / INT8 / INT4）。これは作者報告であり、SM75や当ランタイムでの再現結果ではない。品質確認も少数画像と union-mask 指標を含み、個別検出の一致や広範な benchmark を保証しない。

比較実験の参考として有用だが、配布重みを現在の共有 weight store にそのまま置換する提案ではない。まず変換条件・対象層・スケールを自分のランタイムで再現して比べる。

補足として [ONNX版](https://huggingface.co/danilobukvic/sam3-text-onnx) は INT4 `MatMulNBitsQuantizer` を使用する。これは W4A4 Tensor Core 演算と同義ではない。保存容量、実行 provider、実際の活性型を分けて読む。

## 手元の実験との照合

以下は今回の新規測定ではなく、既存レポートの再確認。

| 既存候補 | 実測・品質 | 今回の解釈 |
|---|---|---|
| 全32 FC2 の単純 W4A4 | 全体414.34→396.53 ms、FP16との品質gateは1/5 | 整数演算の速さはある。精度側の方法変更が必要 |
| W4/A8 と W8/A4 の誤差分離 | 元の5例で2/5と1/5 | この設定では活性4bitの影響が大きい。診断用INT8 GEMMなので混合bitの速度実証ではない |
| affine / MSE / regular H16,H64 | 改善しても全例合格せず | activation MSE・回転のみで解決済みとは言えない |
| INT8 attention のみ | FP16比17/17。独立のpaired系列で516.67→483.73 ms | 既存比較の基準として有用。未知データでの保証ではない |
| INT8 attention/QKV + global4ブロックのMLP INT8 | FP16比17/17。別paired系列でattention-only 478.63→453.31 ms | 混合精度の方向を支持。4ブロックの選び方は開発例に適合している可能性がある |

出典: [INT4_FC2](../../experiments/results/native_rtx2060/INT4_FC2.md)、
[CONVROT_KITCHEN](../../experiments/results/native_rtx2060/CONVROT_KITCHEN.md)、
[KITCHEN_EXTENDED](../../experiments/results/native_rtx2060/KITCHEN_EXTENDED.md)、
[PIXEL_AND_MLP_SCOPE](../../experiments/results/native_rtx2060/PIXEL_AND_MLP_SCOPE.md)。
17例には空検出4例と相関のある動画フレームを含む。系列が違う時間を足して合成速度を作らない。

## 次に行う実験の順序

1. **独立した校正・評価セットを用意する。** 既存17例は回帰用に残す。画像だけでなく複数の text / box prompt、不在概念、小物体を含める。SAM3 の cgF1・presence・mask品質と、現在の FP16一致gateを両方記録する。SAM3.1動画は別評価。
2. **INT8 FC1/FC2 の感度を測る。** QKV、attention、MLPを単独に切り分け、チャネルのscale/shift校正前後で比較する。既存の「global4だけ」を対照にし、層単位の精度維持と累積誤差で選び直す。
3. **FC2 の W4A4 に NormQuant型の校正を加える。** 元の absmax、既存 H16、校正変換のみ、校正＋回転を比較。変換順序と補正を固定し、GELU/quant境界へ融合できるか確認。融合の前に、数学的に対応する参照との整合性を検証する。
4. **必要なら group scale / 再構成へ進む。** まず小規模な校正で品質を見積もる。K方向group scaleは新しい部分和処理を要する。promptを含む最適化はSAM3の構造に合わせ、SAMのdecoderをそのまま移植しない。
5. **実カーネルの全体測定で選ぶ。** 丸めだけ模擬する fake quant の時間は整数性能に数えない。量子化、pack、rotation、復元、layout、fallback を含め、同一条件の fresh process 交互測定で median/p95/VRAM を出す。ファイルサイズ、常駐重み、実行時peakは別に記録する。

当面の最初の実装候補は **FC2入力の校正済みチャネルscale/shift + 選択的W8A8**。
既存 INT8 attention を対照に、まず精度を確認する。INT4 全面化やQAT開始を決めたものではない。

## 調査範囲と再確認方法

Scite で SAM / ViT quantization を2025年以降および2026年6月以降に絞って探索し、対象論文名でも照合した。SAQ-SAM、AHCPTQ、CAR-SAM、UQ-ViT を確認。Mix-QSAM3 は今回の Scite 検索に出ず、CVF の一次資料で補った。Scite上の新しい論文の支持・反証データは乏しく、独立再現が確立しているとは判定していない。arXiv記録の日付が年初に正規化されていたため、公開日はarXiv/会議の一次資料を優先した。

GitHub connector で公式 README、ソース、tree SHA、PR merge 状態、固定版との差分を確認。Hugging Face connector の `paper_search` は `Tool paper_search not found` を返したため、公式公開APIとモデルカードへ切り替えた。モデル本体のダウンロードは行っていない。

「最新」は調査日までに今回確認できた公開資料を指し、全分野の網羅やSOTA順位の断定ではない。取得した PDF・表画像・モデルカードの作業用控えは `.cache/quant-research-20260924/` に保存。旧調査 [APPROXIMATION_RESEARCH](../../experiments/results/native_rtx2060/APPROXIMATION_RESEARCH.md) は履歴として保持する。
