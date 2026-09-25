# 校正なしで常に速くする構造変換の調査

調査日: **2026-09-25**。対象: SAM3 画像推論（vision/neck は SAM3.1 動画と共通）、Windows、
RTX 2060 Max-Q 6 GB（SM75）、LibTorch 2.10 のネイティブランタイム。

GPU のないコンテナで行った一次資料の確認と CPU シミュレーションのまとめ。
**RTX 2060 での速度・品質はまだ測っていない。** 下記の倍率と短縮幅は、既存プロファイルと公開資料から出した上限の見込み。

## 今回の条件

1. 量子化は使ってよいが、校正などのチューニングは不可（教師画像しだいで品質が変わるため）。
2. 再学習はせず、公式の重みを使う。ただし重みの変換はよい。
3. 機能を制限する案、同じプロンプトや同じ画像のときだけ速くなる案は不可。常に速くなること。

## 結論: 計算の中身ではなく、累算器を替える

- GeForce 版 Turing の Tensor Core は、**FP16 で累算すると FP32 累算の 2 倍速**になる（公式資料と命令レベルの実測で確認）。
  現行の FP16 経路は、GEMM も attention もすべて FP32 累算の「半速レーン」で動いている。Vision の Linear はそのレーンの上限（約 18〜20 TFLOPS）にほぼ達している。
- **二段累算**（Tensor Core では FP16 で累算し、64〜128 要素ごとに FP32 レジスタへ足し込む）は、校正も重みの変更も要らず、入力にかかわらず常に効く。
  CPU シミュレーションでは、GEMM 1 回あたりの誤差は現行の約 2 倍、**公式 BF16 経路の約 1/4**、動的 INT8 の 1/20〜1/60 だった。
- 次の厳密な変換を組み合わせる: ネックの線形層の合成、GEMM エピローグへの融合、boxRPB を分離形のまま attention に渡す、デコーダ初層の定数化、ネック枝と検出器の並行実行。
  検出器のホスト待ち（下記 C2）を除いた上限見込みは、全体で **−20〜−35%（未測定）**。
  校正研究で最も速かった構成は −13%（品質基準は未達、[QUANT_RESEARCH_LOG](QUANT_RESEARCH_LOG.md) Round 4）。

## 1. 現状: 時間のほとんどが半速レーン

既存のプロファイル（[CURRENT_HOTSPOTS](../../experiments/results/native_rtx2060/CURRENT_HOTSPOTS.md)、FP16、1 推論 466 ms の GPU カーネル時間）を FLOP 数で割った値。

| 区分 | GPU 時間 | 実効性能 | 実行カーネル |
|---|---:|---:|---|
| Vision Linear（FC1/FC2/QKV/proj） | 249.7 ms | 18.1〜18.9 TFLOPS | `turing_fp16_s1688gemm_fp16_128x128_*`（`s` は FP32 累算） |
| Vision SDPA（global 4 層 + local 28 層） | 73.1 ms | 約 10.7 TFLOPS | `fmha_cutlassF_f16_aligned_64x64_rf_sm75`（FP32 累算） |
| 検出器エンコーダの SDPA（d=32） | 23.4 ms | 約 7.1 TFLOPS | 同上 |
| ネック | 16.4 ms | 約 13 TFLOPS | cuDNN `f16f16_f16f32`。ConvTranspose は dgrad の多段カーネルで、bias の加算（2.8 ms）は別のカーネル |
| 検出ヘッド | 29.4〜30.5 ms | — | 型とレイアウトのコピー 11.9 ms、cuDNN 畳み込み 4.0 ms、GroupNorm の統計 2.5 ms など |

カーネル単位で見ると `turing_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_tn` が 249.9 ms（54%）、`fmha_cutlassF` が 99.8 ms（22%）
（[hotspots-current.json](../../experiments/results/native_rtx2060/hotspots-current.json) の `exact-1`）。

RTX 2060 Max-Q の FP32 累算の上限は、30 SM × 256 FMA/clk × 2 = 15.4 TFLOPS/GHz。
測定中のクロック中央値 1.32 GHz では約 20 TFLOPS。
つまり Linear は半速レーンの上限近くで動いており、**同じレーンのまま** GEMM の設定を探しても伸びしろは小さい。
この結論は [GEMM_SEARCH](../../experiments/results/native_rtx2060/GEMM_SEARCH.md) の結果とも合う。

## 2. ハードウェア: レーンの速さは 1 : 2 : 4

| 資料 | FP16 入力・FP32 累算 | FP16 入力・FP16 累算 | INT8 |
|---|---:|---:|---:|
| [Turing ホワイトペーパー](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf) 付録 B、GeForce RTX 2070（TU106、Reference） | 29.9 TFLOPS | 59.7 TFLOPS | 119.4 TOPS |
| [Sun et al.「Dissecting Tensor Cores」](https://arxiv.org/abs/2206.02874) 表 5、RTX 2080 Ti の `mma.m16n8k8`（8 warp） | 255.1 FMA/clk/SM | 509.4 | 1012.6（m8n8k16） |

- 同じホワイトペーパーで、Quadro RTX と Tesla T4 は FP16 累算と FP32 累算が同じ値になっている。
  半速は **GeForce に固有**で、RTX 2060 Max-Q（TU106）も該当する。同論文の表 4 では Ampere 世代の GeForce（RTX 3070 Ti）も同じ比率だった。
- cuBLAS では、カーネル名が `turing_h1688gemm_*` なら FP16 累算、`turing_fp16_s1688gemm_*` なら FP32 累算。
  Turing（GPU の型番は書かれていないが、カーネル名から判断）での [llama.cpp と Candle の比較](https://github.com/huggingface/candle/issues/2139)では、前者（h1688）がプロンプト処理で約 1500 tok/s、後者（s1688）が約 1000 tok/s だった。
- 画像生成での先例:
  - PyTorch 2.7 以降に `torch.backends.cuda.matmul.allow_fp16_accumulation` がある。C++ では `at::globalContext().setAllowFP16AccumulationCuBLAS(true)`（[CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)、[PR #144441](https://github.com/pytorch/pytorch/pull/144441)）。
  - ComfyUI の `--fast fp16_accumulation`（[PR #6453](https://github.com/Comfy-Org/ComfyUI/pull/6453)、RTX 3090 で 4→5 it/s）。
  - [SageAttention](https://arxiv.org/html/2410.02367v2) は「FP16 累算の行列積は FP32 累算の 2 倍速で、PV では精度が落ちない」と報告している。

反対材料と注意点:

- ComfyUI の PR 作者は「効果は 3090 以降。古いカードでは *おそらく* 改善しない」と書いている。検証を伴わない見解だが、cuBLAS の Turing 用 h1688 カーネルが十分に最適化されているかは実測で確かめる必要がある。
- [Yan et al.（IPDPS 2020、RTX 2070/T4）](https://www.cse.ust.hk/~weiwa/papers/yan-ipdps20.pdf)によると、Turing で FP16 累算の HGEMM を回すとボトルネックは DRAM・L2・共有メモリへ移る。
  大きなタイル（256×128 など）が要り、当時の cuBLAS 10.1 はピークから遠かった。
  そのため実効の倍率は、上限の 2 倍ではなく **1.3〜1.8 倍** 程度を想定する（下記 A1 の見込みはこの範囲で計算）。
- Max-Q は電力制限が厳しい。Tensor Core の稼働率が上がるとクロックが下がり、利得の一部を打ち消しうる。NVML のクロックと電力を必ず同時に記録する。

## 3. 数値: 公式 BF16 経路の誤差範囲に収まる

[Fasi et al.（PeerJ CS 2021）](https://peerj.com/articles/cs-330/)（[プレプリント](https://eprints.maths.manchester.ac.uk/2761/1/fhms20.pdf)）が調べた Tensor Core の丸めの挙動:

- 積は厳密に計算される。
- ブロック内の加算は切り捨て（RZ）で行われる。
- 出力が binary16 のとき、FP16 への最終的な丸めは最近接偶数（RN）。

したがって FP16 累算の誤差は一方向に偏らず、ランダムウォーク的に増える。

これをモデル化した CPU シミュレーション（[`experiments/simulate_fp16_accumulation.py`](../../experiments/simulate_fp16_accumulation.py)、
[結果 JSON](evidence/fp16-accumulation-simulation-20260925.json)）の結果。
合成入力、64×64 の出力、4 seed。値は丸め前の float64 の積に対する相対 RMS 誤差。

| 方式 | QKV/FC1 型（K=1024、外れ値チャネル 8 本） | FC2 型（K=4736、GELU 後） | PV 型（K=5184） |
|---|---:|---:|---:|
| 現行（FP32 累算、FP16 出力） | 3.5e-4（×1.0） | 3.6e-4（×1.0） | 3.3e-4（×1.0） |
| 二段累算、64 要素ごとに FP32 へ | 6.9e-4（×2.0） | 7.1e-4（×2.0） | 6.2e-4（×1.9） |
| 二段累算、128 要素ごと | 9.0e-4（×2.6） | 9.2e-4（×2.6） | 7.8e-4（×2.4） |
| FP16 累算のみ（8 積ごとに丸め） | 1.7e-3（×4.7） | 3.5e-3（×9.9） | 3.4e-3（×10.4） |
| **公式 BF16 autocast**（BF16 入出力、FP32 累算） | **2.9e-3（×8.2）** | **2.9e-3（×8.1）** | **2.6e-3（×7.9）** |
| INT8（行 × 出力チャネル、動的） | 4.4e-2（×124） | 1.5e-2（×42） | 3.5e-2（×107） |
| INT8 + ブロック Hadamard（データ不要） | 1.1e-2（×31） | 1.3e-2（×35） | 1.7e-2（×53） |

- 公式の SAM3 予測器は `torch.autocast(dtype=torch.bfloat16)` で推論する（`sam3/model/sam3_base_predictor.py` の 223 行と 243 行）。
  二段累算の誤差はその経路の約 1/4（64 要素ごと）で、FP16 累算だけでも 0.6〜1.3 倍と同程度。一方 INT8 は、全次元の Hadamard 回転を加えても BF16 の約 3.6 倍になった（外れ値型の入力）。
- ネイティブの監査（[IMAGE_PRECISION_AUDIT](IMAGE_PRECISION_AUDIT.md)）では、SAM3 の mask AP は FP16 が BF16 より 0.17 ポイント低いだけだった。これが「公式経路の精度幅」の目安になる。
- 限界: 合成データで GEMM 1 回分の誤差しか見ていない。32 ブロックを通した誤差の伝搬、SAM3 固有の大きな活性値、FP16 のオーバーフロー（最大 65504）は実モデルで確かめる。

## 4. 提案一覧

「厳密」は、実数演算で等価（丸めの順序だけ変わる）という意味。いずれも公式の重みまたはその代数的な変換だけを使い、校正は行わない。

| # | 提案 | 対象（FP16 時の GPU 時間） | 数値 | 上限見込み |
|---|---|---|---|---:|
| A1 | 二段累算 GEMM | Vision Linear 249.7 ms | 近似（BF16 未満の誤差） | −60〜−110 ms |
| A2 | FP16 累算 attention（PV から、次に QK） | Vision SDPA 73.1 ms、検出器 23.4 ms | 近似 | −15〜−30 ms |
| A3 | 畳み込みの FP16 累算 | ネック 16.4 ms、検出ヘッドの畳み込み 4.0 ms | 近似 | −4〜−7 ms |
| B1 | ネックの線形層を合成 | ネック（215 → 155 GFLOP） | 厳密 | −3〜−5 ms（A3 と一部重複） |
| B2 | bias・GELU・RoPE を GEMM エピローグへ | GELU 13.9 ms、RoPE 6.4 ms | 厳密（丸め位置を維持） | −15〜−20 ms |
| B3 | boxRPB を分離形のまま attention へ | デコーダ RPB 5.8〜6.8 ms | 厳密 | −4〜−5 ms |
| B4 | 画像にもプロンプトにも依存しない部分の前計算 | デコーダ初層など | 厳密 | 約 −1 ms |
| C1 | ネックの高解像度枝を検出器と並行実行 | ネックの残り時間 | ビット一致 | 最大 −5〜−10 ms |
| C2 | 検出器の同期点の除去と CUDA Graph | 検出器のカーネル外 約 55 ms | ビット一致 | 要測定 |
| D1 | （INT8 レーンを使う場合）データ不要の残差回転 | INT8 化する層 | 厳密な変換 + RTN | — |

C2 を除いた合計は −107〜−188 ms で、約 520 ms に対して約 −20〜−35%。値は重複を含む粗い上限で、未測定。

## 5. 各提案

### A1: 二段累算 GEMM

同じ技法の先例:

- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) は、Tensor Core の精度が低い FP8 累算を 128 要素ごとに CUDA コアの FP32 へ移して足している（promotion）。
- SageAttention の `pv_accum_dtype="fp16+fp32"` も、短い区間を FP16 で累算し、FP32 のバッファへ足し込む。

SM75 向けの部品はそろっている:

- CUTLASS 2.x の `arch/mma_sm75.h` に `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16` がある。
- CuTe には `SM75_16x8x8_F16F16F16F16_TN` がある。

実装の方針:

- **最初の実験は 1 行で済む。** 環境変数で opt-in した場合だけ `at::globalContext().setAllowFP16AccumulationCuBLAS(true)` を呼ぶ。
  Nsight でカーネル名が `turing_h1688gemm_*` に変わることを確認し、速度と品質を測る。
  これは K=4736 まで FP16 だけで累算するため、**品質は最悪側、速度は cuBLAS しだい**の測定になる。
- 品質が足りない場合は、CUTLASS の SM75 用 `MmaPipelined` を改造する。
  - warp 単位の MMA を FP16 累算にし、64〜128 要素（threadblock の K を 2〜4 反復）ごとに FP32 フラグメントへ足してから FP16 側をゼロに戻す。
  - エピローグは FP32 の累算値から現行と同じ丸め位置で出力する。
  - FP16 と FP32 の両方の累算器を持つので、warp タイルは 64×32 程度にしてレジスタのスピルを避ける。
  - Yan et al. に従い、L2 の帯域が足りるようにタイルを 256×128 か 128×256 にする。
- K=1024 の QKV・FC1・proj は、FP16 累算だけでも BF16 より誤差が小さい（表の QKV/FC1 型で ×4.7。BF16 は ×8.2）。二段にするのは FC2（K=4736）だけという選択肢もある。
- オーバーフロー対策はデータ不要のまま行える。重みを 2 の冪で縮め、エピローグで戻す（FP16 では厳密）。
  attention の 1/√64 = 1/8 は 2 の冪なので、W_q に畳み込むこともできる。

### A2: FP16 累算の attention

- 以前試した外部カーネル（[ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing)、[TURING_ATTENTION](../../experiments/results/native_rtx2060/TURING_ATTENTION.md)）は、
  `kernel_traits.h` で `MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>`、つまり FP32 累算だった。それでも global attention の演算子は 9.45 ms → 7.56 ms になった。
- FlashAttention の出力の累算器は、KV タイルごとに FP32 で再スケールされる。
  PV だけを `SM75_16x8x8_F16F16F16F16_TN` にすれば、自然に「KV タイル長ごとの二段累算」になる（donor のタイルは 128。表の PV 型では 128 要素ごとで ×2.4、64 要素ごとで ×1.9）。
  QK は d=64 で 1 区間に収まる。
- donor の再配布条件は未解決のまま（TURING_ATTENTION の記録）。本番に入れるなら、CuTe（BSD-3）で自前に書く。
- 既存の ComfyKitchen INT8 attention も実行時の量子化だけで動き、校正は要らない。精度を上げたい層には A2、速度を優先する層には INT8、と層ごとに選べる。

### A3: 畳み込みの FP16 累算

cuDNN のカーネル名 `f16f16_f16f32` は FP32 累算を表す。選択肢は 3 つ。

- cuDNN の TRUE_HALF_CONFIG（畳み込み記述子の計算型を HALF にする、CC 5.3 以上）。ただし SM75 で Tensor Core を使うエンジンが選ばれるかは要確認。
- CUTLASS の implicit GEMM 畳み込みを FP16 累算で使う。
- B1 の変換で GEMM に置き換える。

### B1: ネックの線形層を合成する（重みの変換）

非線形の演算をはさまない線形層は、読み込み時に 1 つにまとめられる。

| 枝 | 現行 | 変換 | GFLOP |
|---|---|---|---:|
| 4 倍 | ConvT(1024→512) → GELU → ConvT(512→256) → 1×1 → 3×3 | ConvT(512→256) と 1×1 を 1 つの ConvT に合成し、さらに 3×3 と合わせて sub-pixel 畳み込み（144² 上の 4 位相 × 2×2 × 512→256）にする | 152.2 → 108.7 |
| 2 倍 | ConvT(1024→512) → 1×1(512→256) → 3×3 | ConvT と 1×1 を ConvT(1024→256) 1 つに合成 | 51.6 → 35.3 |
| 1 倍、0.5 倍 | 1×1 → 3×3 | 変更なし | 11.0 |
| 合計 | | | 214.9 → 155.1（−28%） |

- 合成した重みは `W'[i,o,kh,kw] = Σc W1x1[o,c] WT[i,c,kh,kw]`、バイアスは `b' = W1x1 bT + b1x1`。
  sub-pixel 形は [Shi et al. 2016](https://arxiv.org/abs/1609.07009) の等価性による。
  3×3 のゼロパディングは入力側のゼロパディングで表せる。ただし ConvT のバイアスの寄与は外周の 1 画素だけ変わるので、エピローグで補正する。
- ConvT(k=2, s=2) は `[5184,1024] × [1024, 4·Cout]` の GEMM と画素の並べ替えに等しい。
  同じ入力を読む 3 枝の最初の変換は、N=2048+1024+256 の 1 本の GEMM（35.3 GFLOP）にまとめられる。
  現在の cuDNN dgrad 経路（補助カーネル 4 段）とレイアウト変換も不要になる。
- 0.5 倍の枝（0.49 ms）は、画像の C API では計算した後に捨てられている（`c_api_image.cpp` 13 行の `pop_back()`、上流の `scalp=1` に相当）。
  追跡（`tracking_vision.cpp`）は 4 段すべてを使うので、必要な呼び出し元でだけ計算する。

### B2: エピローグへの融合

A1 を自前のカーネルで実装する場合、次をエピローグで行える。

- FC1: bias → FP16 へ丸め → erf GELU（FP32 で計算）→ FP16 へ丸め。
- QKV: bias → FP16 へ丸め → RoPE → head-major での書き込み。

丸めの位置を ATen と同じにでき（[PERFORMANCE](PERFORMANCE.md) の条件）、GELU 13.9 ms と RoPE 6.4 ms の読み書きがなくなる。
INT8 の研究で作った QKV の復元と RoPE の融合（[QKV_RESTORE_ROPE](../../experiments/results/native_rtx2060/QKV_RESTORE_ROPE.md)）がそのまま設計の参考になる。

### B3: boxRPB を分離形のまま attention へ

`decoder.py` の `_get_rpb_matrix` は、x 方向 `[200, 72, 8]` と y 方向 `[200, 72, 8]` のバイアスを足し、`[8, 200, 5184]` に展開して `contiguous()` する。
そのコピーが RPB 6.8 ms のうち 3.9 ms を占め、1 層で FP32 33 MB になる。

attention カーネルの中で `bias_y[q, key の行] + bias_x[q, key の列]` を足せば、展開は要らない。
PyTorch の [segment-anything-fast](https://pytorch.org/blog/accelerating-generative-ai/) は、SAM の分解された相対位置バイアス（rel_h, rel_w）で同じ融合をしている。

### B4: 画像にもプロンプトにも依存しない部分の前計算

- デコーダ初層の入力 `query_embed`、`reference_points`（sigmoid 後）、`presence_token` は学習済みの定数（`detector_decoder.cpp`）。
  したがって、初層の自己 attention ブロック（残差と norm を含む）、初層の `query_pos`、初層の boxRPB は読み込み時に計算しておける。
- 検出器エンコーダは q = k = LN(x) + pos で、pos は 72×72 の固定の正弦波。
  `W_q·pos` と `W_k·pos` をトークンごとのバイアスとして前計算すれば、Q・K・V を LN(x) からの 1 本の GEMM で出せる。

### C1: DAG の並べ替え（出力はビット一致）

ネックの 4 倍と 2 倍の枝（約 15 ms）を使うのは、最後のセグメンテーションヘッド（`detection_heads.cpp` 71 行からのループ）だけ。
検出器のエンコーダとデコーダが必要とするのは 72² の特徴（`grounding.cpp` の `pyramid.back()`）だけなので、この 2 枝を別の CUDA stream で検出器と並行に流せる。
検出器にはカーネルの外の待ち時間が約 55 ms ある（CURRENT_HOTSPOTS）ので、その GPU の空き時間を埋められる見込み。

### C2: 検出器の同期点と起動のオーバーヘッド

- ホストとの同期点が 3 か所ある: `grounding.cpp` 20 行と 34 行の `.all().item<bool>()`（ID の検証）、`detector_decoder.cpp` 64 行の `spatial_shapes.cpu()`。
  ID はホスト側で検証でき、空間形状は 72×72 の定数なので、どれも外せる。
- Windows の WDDM では、空のカーネルでも起動に 10〜80 µs（平均約 20 µs）かかる。Linux と TCC では約 5 µs
  （[NVIDIA フォーラム](https://forums.developer.nvidia.com/t/very-slow-kernel-launches/37345)、古い資料）。
  CUDA Graph は Blackwell/Linux の測定で不採用になった（[PERFORMANCE](PERFORMANCE.md)）が、Windows のノート PC で検出器だけを取り込む場合については改めて測る価値がある。
  - 形状を固定するため、テキストは 32 トークンまでパディングする（マスクするので意味は同じ）。
  - Hardware-accelerated GPU scheduling の設定も記録する。

### D1: INT8 レーンを使う場合のデータ不要の回転

SAM3 の ViT は `ln_post=False` で、ネックは正規化していない残差ストリームをそのまま読む。
そのため [SliceGPT](https://arxiv.org/abs/2401.15024) のように LayerNorm を RMSNorm へ変換する（平均を引く処理を重みに移す）方法は使いにくい。

代わりに、**Q·1 = 1 を満たす直交行列 Q** を使う。この Q は affine を外した LayerNorm と可換になる（LN(Qx) = Q·LN(x)）。
Hadamard 行列の後に、第 1 軸を 1/√d へ写す Householder 反射を掛ければ作れる。変換の手順:

1. norm1 と norm2 の γ・β を QKV と FC1 に畳み込む。
2. proj と FC2 の出力側に Q を、QKV・FC1・ネックの入力側に Qᵀ を畳み込む。
3. ln_pre の後で回転を 1 回かける。0.5 倍の枝の maxpool の前では回転を戻す。

[QuaRot](https://arxiv.org/abs/2404.00456) は、この種の回転と RTN で「6/8 bit では校正データなしでほぼ無損失」と報告している。
ただし上のシミュレーションでは、INT8 の誤差は回転後も公式 BF16 の 3.6 倍あった。FP16 累算を優先し、INT8 は一部の層に限るのが妥当。

## 6. 評価のしかた

- 既存の FP16 一致 gate（IoU 0.98、score 差 0.02、box 差 1 px）は、FP16 のノイズの大きさとほぼ同じ厳しさになっている。
  FP32 累算どうしの attention の置き換えでも child の box 差が 1.0154 px で Fail した（TURING_ATTENTION）。
  cuBLAS の split-K だけでも、要素の 27% が 1 ulp 変わる（GEMM_SEARCH）。
- 次の 2 つを追加することを提案する。どちらも data-free。
  1. FP32 参照からのずれを、公式 BF16 経路のずれと比べる（BF16 は Blackwell 開発機の `bf16_reference`）。
  2. 既存の COCO slice による mask AP の監査（IMAGE_PRECISION_AUDIT）。
- 既存の gate は回帰の検出用に残す。採用基準をどうするかは所有者が決める。

## 7. 着手の順番

1. **A1 の 1 行実験。** 環境変数で opt-in → Nsight でカーネル名を確認 → `sam3_image_latency` を別プロセスで交互に測定（NVML のクロックと電力も記録）→ 開発用 33 prompt、BF16 との比較、COCO slice。
   - 速く、品質も十分: 採用候補にする。
   - 速いが品質が足りない: 二段累算のカーネルを作る。
   - 速くならない: cuBLAS の Turing 用 h1688 が弱いと判断し、CUTLASS で自前に書く。
2. A1 を自前のカーネルにするなら、B2 を同じカーネルでまとめて行う。
3. A2 は PV から始める。QK の FP16 累算は、PV の後に別の候補として比べる。
4. B1、B3、B4、C1、C2 は厳密またはビット一致なので、A と独立に進められる。既存の byte 一致テストの枠組みで検証できる。

## 8. 検討して見送った案

| 案 | 見送った理由 |
|---|---|
| Ozaki 型の分割（FP16 相当の精度を INT8 の GEMM 複数本で出す） | GEMM が 3 本要る。INT8 GEMM は実測で FP16 の 2.24 倍しか速くない（CURRENT_HOTSPOTS）ので、FP16 累算に勝てない |
| Strassen | 加算を融合した CUTLASS 実装でも、1 段で最大 1.11 倍（[Huang et al.](https://arxiv.org/abs/1808.07984)、V100）。Tensor Core 上では追加の行列加算のメモリ往復が相対的に重くなり、誤差も増える。FP16 累算の 2 倍に比べて利得が小さい |
| 2:4 構造化スパース | SM80 以上が必要 |
| トークンのクラスタリング（[Expedite ViT](https://arxiv.org/abs/2210.01035)：ADE20K で FLOPs −30%、mIoU 99.5% を維持、fine-tuning なし） | 近似であり、層の位置やクラスタ数の選択が実質的にチューニングになる。小さい物体に不利。K/V の pooling はすでに不採用（NOTES） |
| 残差ストリームの FP16 化 | 近似で、効果も小さく、すでに不採用 |
| 重みの低ランク分解 | 近似。ViT の重みは低ランクではない |

## 調査範囲

- Web 検索と、一次資料（NVIDIA ホワイトペーパー、論文 PDF、GitHub のソースと PR、PyTorch のドキュメント）の本文を確認した。
- flash-attention-turing と CUTLASS はクローンして、MMA atom を確認した。
- CPU シミュレーションは Python 3.11.15、NumPy 2.4.6 で実行した。
- 未実施: GPU での実行、Windows でのビルド、速度・品質・メモリの測定。
