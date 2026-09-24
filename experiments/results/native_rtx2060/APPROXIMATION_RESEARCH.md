# ビット一致を緩和した SAM3 / Turing 最適化候補

調査日: 2026-09-24。対象: Windows、RTX 2060 Max-Q (SM75)、CUDA 13.0、
LibTorch 2.10、native `908e476`。この文書は研究・実験提案であり、実装や速度の実測結果ではない。
Scite で文献を探索し、本文未収録のものは著者公開版・会議公開版で補った。
Mix-QSAM3 は Scite 検索では見つからず、CVF の検索インデックスから要旨と結果表を確認した。
その PDF の直接取得は失敗しているため、実装詳細まで検証済みとは扱わない。

## 判断

最初の短い実験は **FC1 + bias + 近似 GELU の融合**。
より大きな改善の本命は **Vision MLP の選択的 W8A8 (重み・活性 INT8) 化**。
SageAttention の SM75 移植、トークン削減、エンコーダ蒸留は後段に置く。

現状の基準は [FUSION_AB.md](FUSION_AB.md): 全体中央値 540.91 ms、
Vision CUDA 平均 410.09 ms、Detector 127.29 ms。中央値と各段平均は単純加算しない。
旧プロファイルでは MLP が Vision の約48%、attention と投影が約40%。
主要 FP16 GEMM カーネルはモデル全体の累積 GPU カーネル時間の約50.6%だった。
この比率は最新融合後の厳密な内訳ではないが、次に行列積を狙う根拠になる。
SDPA はすでに SM75 用 memory-efficient CUTLASS カーネルを使っている。

## 読んだ文献と採否

| 文献 | 得られた知見 | この実装への判断 |
|---|---|---|
| [SmoothQuant, ICML 2023](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf) | 活性の外れ値を、チャネル単位の等価スケーリングで重み側へ移し、W8A8 を成立させる | MLP の量子化設計の出発点。LLM の精度・速度結果を SAM3 に流用はしない |
| [RepQ-ViT, ICCV 2023](https://arxiv.org/pdf/2212.08254) | LayerNorm 後のチャネル間分布差、Softmax 後の分布に別々の量子化設計を適用 | まず LayerNorm 後の分布調整を参考にする。Softmax の独自形式は初期実装に入れない |
| [PTQ4SAM, CVPR 2024](https://arxiv.org/html/2405.03144v1) | Key の二峰性への符号変換、Softmax の適応的対数量子化 | SAM 特有の敏感箇所を調べる指針。SAM3 で同じ分布があるかは測定が必要 |
| [Mix-QSAM3, CVPR Workshops 2026](https://openaccess.thecvf.com/content/CVPR2026W/AIGENS/html/Ranjan_Mix-QSAM3_Mixed-Precision_Quantization_for_the_Segment_Anything_with_Concepts_Model_CVPRW_2026_paper.html) | 層の情報保持・量子化感度・隣接層依存から精度を配分 | SAM3 に直接関連。全層一律 INT8 より、敏感な層を残す方針を支持 |
| [SageAttention](https://arxiv.org/html/2410.02367v2) | QK の INT8 化、Key の平均除去、PV の FP16 演算など | アルゴリズムは参考になるが公式実装は SM75 非対応。初手にはしない |
| [ToMe, ICLR 2023](https://arxiv.org/pdf/2210.09461) | 類似トークンを統合し計算量を減らす。統合サイズを attention に反映 | 分類 ViT の高速化を dense segmentation にそのまま適用できない |
| [StructSAM, arXiv 2026](https://arxiv.org/html/2603.07307v1) | 特徴勾配で境界を保護し、attention 内で merge → compute → unmerge | ToMe より用途に近い。ただし SAM3/RoPE への適用は未検証、MLP 削減の根拠にはならない |
| [EfficientViT-SAM, 2024](https://arxiv.org/html/2402.05008v2) | 軽いエンコーダへの蒸留と追加学習 | 大幅高速化の長期案。SAM3 のテキスト概念対応を保つ単純な重み交換ではない |

PTQ4SAM の「3.9倍」はビット幅で重み付けした計算量による**理論値**。
RTX 2060 上の FP16 基準に対するレイテンシ実測ではない。

Mix-QSAM3 の画像 SA-Co 平均 cgF1 は FP32 54.1、一律 INT8 47.9、MP8 52.4。
MP8 は対応する固定ビットモデルとサイズ・bit operations を合わせた混合精度設定であり、
ここで提案する INT8/FP16 選択方式と同一ではない。速度保証にも使わない。

## 実験 1: 近似 GELU を GEMM に融合

現在の `vision_encoder.cpp` は FP16 FC1 出力をいったん保存してから `gelu_(..., "none")`。
`FC1 + bias + GELU` を cuBLASLt epilogue にまとめる候補を既存 GEMM ベンチで測る。
[cuBLAS の仕様](https://docs.nvidia.com/cuda/archive/12.9.1/pdf/CUBLAS_Library.pdf)の GELU は tanh 近似。
現行の erf 版との差に加え、中間 FP16 丸め位置の変更も数値差になる。

比較は (A) 現行、(B) 独立 tanh GELU、(C) fused epilogue の3条件。
実形状 M=5184, K=1024, N=4736、実活性・実重みで確認する。
SM75 / CUDA13 で当該 epilogue と形状のアルゴリズムが返るかは未検証。
GEMM 本体の選択が悪化して融合の節約を打ち消す可能性も含めて測る。
単なる cuBLASLt への切替は既存実験で改善しなかったので、融合そのものの効果を分離する。
改善幅は未測定。単独で大幅高速化するという根拠はない。

## 実験 2: 選択的 W8A8 MLP

### 追補: ハードウェアと実装対応の確認

2026-09-24、ローカル PyTorch 2.10.0+cu130 / RTX 2060 Max-Q SM75 で
`torch._int_mm` を実行し、CUDA profiler で GPU カーネルを確認した。
256x256x256 に加え、FC1/FC2 の実形状 (M,K,N)=(5184,1024,4736)、
(5184,4736,1024) がいずれも成功。入力は小さな乱数 INT8、出力は CUDA 上の INT32。
`cutlass_75_tensorop_i8816gemm_s8_128x128_tn_align16` が実行され、
小整数を FP32 で積和した参照との完全一致も確認した。
[結果 JSON](int8-hardware-probe.json)。CPU フォールバックではない。
ただし Python 配布の ATen 経由の確認であり、standalone LibTorch バイナリでの
同じ演算のリンク・実行や、量子化を含む速度優位はまだ確認していない。

[CUDA13 cuBLAS 仕様](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublasltmatmul)
には Turing の IMMA 対応と専用レイアウト条件が記載されている。
同仕様の IMMA 専用レイアウトの INT8入力/INT32出力では non-default epilogue は非対応。
INT8 GEMM に bias/GELU を単に指定すれば融合できる、と仮定してはいけない。
別の復元カーネルか対応する独自 epilogue を含めた実測が必要。
一般の量子化 Linear API のバックエンド対応と GPU 命令対応は別で、未対応APIは
エラーになることもあり、自動 CPU フォールバックが保証されるわけではない。

[NVIDIA Turing ガイド](https://docs.nvidia.com/cuda/archive/11.5.0/turing-tuning-guide/index.html)
で INT8 入力・INT32 累算の Tensor Core 対応を確認。
SM75 向け cuBLASLt または CUTLASS の実カーネルで、量子化・レイアウト変換・復元も含めて測る。
fake quantization は品質検証用であり、その時間を INT8 実行性能として報告しない。

1. 各ブロックの FC1/FC2 入出力と外れ値を収集。校正画像と評価画像を分離する。
2. FC1 と FC2 を個別に量子化し、層ごとの出力誤差と最終検出・マスク変化を測る。
3. 重みは出力チャネル別スケール、活性は動的トークン別と校正済み静的スケールを比較する。
4. LayerNorm 後の smoothing を試す。FC2 前は GELU の非線形をまたぐため、
   同じスケーリングを無条件に FC1 重みへ折り込まない。必要なスケーリングを GELU 側に融合する。
5. 最初は residual・LayerNorm 統計・Softmax・Detector を現行精度に維持し、
   誤差に弱い MLP 層も FP16 に戻せる構成にする。
6. 勝ちが確認できてから QKV / output projection へ広げる。

FC1 の INT32 出力を巨大なテンソルとして書いてから復元する実装は、帯域コストで負け得る。
復元・bias・GELU または次の量子化を epilogue にまとめる設計を検討する。
重みだけの INT4 化はサイズ削減候補だが、この大きな M の行列積が速くなるとは限らない。

効果の目安は予測値ではなく感度計算で置く。仮に全体時間の半分を占める部分が
1.5倍速なら全体時間は約17%減、2倍速でも約25%減。変換コストを無視した計算であり、
541 ms が自動的に半分になるわけではない。

## 後段の実験

- **低精度 attention:** [公式 SageAttention のビルド定義](https://github.com/thu-ml/SageAttention/blob/main/setup.py)
  は SM80 以上。SM75 移植は別のカーネル開発になる。局所576トークンと大域5184トークンを
  分けて評価し、量子化準備コストまで含める。PV の FP16 累算の知見を MLP の長い K に
  無条件で適用しない。
- **StructSAM 型 token merging:** まず少数の大域 attention 層で低い削減率を試す。
  元の座標と統合サイズを保持し、RoPE の扱い、unmerge、window 境界を設計する必要がある。
  prompt に依存する統合は画像特徴の共有・再利用にも影響する。
- **蒸留 / 解像度低下:** 計算量を大きく削る別軸。ただし前者は学習、後者は位置埋め込みや
  window padding の調整を伴い、小物体・細線の品質を別に確認する。

## ビット一致の代わりの採用基準（提案）

現行 FP16 を保存し、変更を一つずつ比較する。既存5画像はスモークテストに残すが、
それだけで品質維持とは判定しない。校正に使わない画像・多様なテキスト・不在概念を含め、
小物体、細い境界、密集、低コントラストを評価する。

- ラベル付き評価: mask AP、SAM3 の概念検出を含む指標（cgF1 等）、境界品質。
- 参照モデルとの差: IoU/境界差、スコア変動、検出の消失・追加、閾値をまたいだ件数。
  出力順が変わり得るので query ID や配列順だけでマスクを対応付けない。
- 速度: fresh process の交互比較、十分な warmup、median と p95、GPU温度・クロック、
  ピーク VRAM。校正時間・初回時間・定常推論を分ける。
- 暫定目標: W8A8 は全体時間15%以上削減かつ mask AP 低下0.5ポイント以内を探索目標にする。
  これはユーザーが承認した品質閾値ではなく、実験の仮置き。概念検出や難例の悪化も別判定する。

最終的には速度と品質の曲線を出し、数値誤差の小ささだけで選ばない。
