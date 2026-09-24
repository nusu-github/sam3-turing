# Python Turing パッチの開発メモ（要約）

2026-09-20〜21 に RTX 3090 のコンテナ（CPU は AMD EPYC 7763）で行った画像パッチの探索
（Round 1〜68、376 候補、有効な画像測定 373 件）の採否をまとめたもの。公開 API の使い方と
代表値は [docs/TURING_IMAGE_PATCH.md](../docs/TURING_IMAGE_PATCH.md)、全候補の数値表は
[results/README.md](results/README.md) と [results/summary.csv](results/summary.csv)、
各候補の生データは `results/*.json` にある。Round ごとの経緯を書いた元のメモ、sweep 設定
（`round*.json`）、候補実装は commit `080bec0`（`main`）に残っている。

測定条件: `truck.jpg` / `truck` の1画像の時間（warmup 後の中央値）と、truck / paper bag /
child / wheel / 空（elephant）の5条件での stock（BF16 MLP のまま）との一致。検出数
1/4/6/4/0 の一致、対応 mask の平均 IoU、変化画素数（全 18,038,400 画素）、score・box 最大差を
見る。正解ラベルに対する精度評価ではない。以下の ms は 3090 上の値で、Turing の速度として
外挿しない（RTX 2060 実機の値は [results/local_rtx2060](results/local_rtx2060/README.md)）。

## 公開パッチに採用したもの

| 機能（公開 API） | Round | 効果（代表値） |
|---|---|---|
| FP16 化・メモリ削減・テキスト cache（`apply_turing_patch`） | 1〜5 | stock 204 ms / 4.99 GiB → 約114 ms / 1.86 GiB、285 画素差 |
| grounding の一括 compile、出力4配列を Graph 外で clone（`compile=True`） | 13 | 123.12 → 115.36 ms |
| 4K mask の resize・sigmoid・閾値・bitpack 融合（`turing_masks`） | マスク出力 | 28.72 → 8.18 ms、FP16 cutoff 調整後 6.18 ms |
| 画像 MLP の動的 INT8 と Attention 射影 INT8（`apply_int8_patch(attention_projections=True)`） | 7, 14 | 92.26 → 88.61 ms、950 画素差 |
| INT8 GELU・再量子化の融合（`fused_mlp=True`） | 15, 17 | 85.46 ms、916 画素差 |
| GELU 後の非対称 INT8（`asymmetric_gelu=True`） | 25, 32 | 735 画素差 |
| 重み scale の二乗誤差探索＋2回調整（`optimize_weight_scales=True`） | 35, 44 | 697 画素差、score 差 0.0117 → 0.0039 |
| 重みのみ INT8、計算は FP16（`weight_only=True`） | 31, 34 | 115.57 ms、343 画素差（メモリ優先） |
| テキスト encoder の CPU 実行（`offload_text_encoder`） | 39, 42 | INT8 画像＋cache 済み 84.94 ms、0.793 GiB |
| CPU テキスト MLP の動的 INT8（`int8_mlp=True`） | 48, 51 | 新規語句 155.84 → 100.80 ms |
| CPU テキストの末尾 padding 省略（`trim_padding=True`） | 52, 56 | 新規語句 FP32 163.71 → 87.48 ms |
| 残差 FP16・decoder FFN FP16・mask head 再結合（`apply_image_refinements`） | 53b, 55 | 83.08 ms、673 画素差 |
| 4bit 重み（compact Gaussian group 16/32、非対称 group 16、`turing_int4`） | 38, 62, 63, 67 | Gaussian 32: 0.590 GiB、約116 ms、1995 画素差 |
| 低解像度 window の余白射影省略（`unpadded_projections=True`） | 49, 66, 68 | 784 入力・CPU 16 スレッドで 69.53 → 58.96 ms |

低解像度入力（Round 30、896〜560）は速度優先の選択肢として文書化したが、既定の 1008 は
変えていない（784 で平均 IoU 0.971、最小 IoU 0.885）。

## 採用しなかったもの

| 候補 | Round | 理由 |
|---|---|---|
| global Attention の K/V 2×2 pooling、MLP 入力の空間平均化 | 6, 8 | 近似。MLP 平均化は検出数 15 → 3 |
| 特徴分散で選ぶ MLP 共有 | 11 | 細部が崩れる（IoU 0.964）、速度利得なし |
| 重み統計による SmoothQuant 風 scale | 12 | 改善が小さい |
| MLP 前 LayerNorm と量子化の融合、LayerNorm 係数の重みへの前計算 | 17, 21 | 速度差が小さく、出力差またはメモリが増える |
| 重みのみ INT8 の独自 Triton GEMM（タイル・並べ替え・FP16 入力） | 16, 23, 33 | 最速でも FP16 の通常 GEMM より遅い |
| 入力正規化の融合、neck compile | 19 | 差が小さく Graph 境界が増える |
| decoder FFN の FP16 化のみ、ViT 残差の FP16 保持のみ | 20, 26 | 単独では改善が小さい（後に組合せで採用） |
| mask を作る query 数の固定削減 | 27 | 効果が小さく検出数に上限ができる |
| 複数 token をまとめた量子化 kernel | 28 | 遅くなる |
| MLP の省略、MLP チャネル削減、Attention head 削減、検出 decoder の層削減 | 22, 24, 36, 40 | 1〜3 ms の短縮に対して出力差が大きい |
| INT8 GEMM と後処理の融合（Triton） | 37, 41 | fc1 のみ融合は出力不変で速いが、SM75 で INT8 dot の lowering に失敗 |
| window サイズ変更、1008 からの小幅な解像度低下 | 46, 64 | 出力差が増える、または速度改善なし |
| 同一画像への複数 prompt の batch 化 | 複数 prompt | 逐次処理の方が速くメモリも少ない |
| INT8 重みの平均誤差を bias で補正 | 43（最終回） | 非対称 INT8 ＋ scale 調整の 697 画素に対し 769 画素 |

## SM75 / Turing に関する注意

- Triton 3.5.0 の sm75 オフライン compile で、公開パッチの 7 kernel（量子化、GELU、非対称
  GELU、FP16/FP32 mask resize+pack、pack、unpack）は通過した。独自 INT8 GEMM は
  `arith.extf` の lowering エラーになるため、Turing 向けには `torch._int_mm` を使う。
  オフライン compile は実機での動作・速度の確認ではない。
- GTX 16 系は Turing だが Tensor Core を持たない。3090 の INT8 / Tensor Core の測定値を
  Turing 全体へ外挿しない。
- CPU テキストの時間は CPU とスレッド数に強く依存する（EPYC では 8〜16 スレッドが速かった）。
  公開パッチはプロセスのスレッド数を変更しない。

## SAM 3.1 動画（実験のみ）

`sam3.1_multiplex.pt` で動画 0001 の先頭 24 フレーム・`person` を追跡した。FP16 化で
allocated 6.73 → 3.90 GiB、INT8 ＋ ViT compile で 195.40 ms/frame、CPU テキストの padding 省略
・検出器／tracker の compile を重ねて 139.90 ms/frame（stock 261.00 ms/frame から約 46% 短縮）。
24 フレーム×4 人・96 件の ID 対応は全構成で維持した。詳細は
[docs/TURING_VIDEO_EXPERIMENTS.md](../docs/TURING_VIDEO_EXPERIMENTS.md)。6 GB の Turing 実機では
検証していない。

## 終了時点

ユーザーの指示により 2026-09-21 の Round 43 で探索を終了した。準備済みで未実行の候補
（Round 47 の入力 clipping、60 の画像寄与による head 選択、61 の一部 FP16 復元、
65 の外れ値成分の FP16 分離など）は測定していない。公開パッチは検証済みの 5 モジュール
（`sam3/turing.py`、`turing_int8.py`、`turing_int4.py`、`turing_masks.py`、
`turing_refinements.py`）で、上流 checkout 向けの単独パッチは
[patches/turing-image.patch](../patches/turing-image.patch)。
