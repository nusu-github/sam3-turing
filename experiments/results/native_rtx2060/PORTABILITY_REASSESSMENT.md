# 移植レビューの再評価（2026-09-24）

対象文書: `C:/Users/nusu-/Downloads/sam3_turing_portability_review_2026-09-24.md`。
現作業ツリー: `ba018a0` + ローカルのWindows対応・INT8 MLP境界融合・QKV量子化・計測コード。

今回は既存ログ、実装、donorの現在のソースを照合した。GPUベンチマーク・回帰テストの再実行や、推論実装の変更はしていない。確認済み事項の再測定を省略し、結論が変わる差分を抽出した。

添付文書は検討資料であり、その「非量子化FP16を第一群とする」等を新しいユーザー指示とは扱わない。このスレッドで許可された数値条件の緩和を維持し、FP16基準と検証済みINT8経路の両方を比較対象にする。

## 結論

次の大きな実験は **Vision d=64、まず4層のglobal attentionへの別forward導入**。文書の提案は、この部分では現在の実測と整合する。

一方、もう一方の柱を「FP16の4 GEMMを最初から再探索」に戻す優先度は低い。既にMLPはINT8境界融合で改善し、QKVもINT8化を測定している。追加候補は **QKV INT8 GEMMの復元epilogue、または復元とRoPEの融合**。ただしこれは未実装・未測定であり、成功を保証するものではない。

## 再実行を省略する項目と、その範囲

| 文書の論点 | 現在の証拠・判定 |
|---|---|
| Turingの命令制約 | 今回は再調査しない。SM75 INT8実行は実機kernel名でも確認済み。FP8/BF16等の不対応を再検証する必要はない |
| 最新fusionのWindows/Turing確認 | `ba018a0`まで取り込み済み。最新ログは51/51合格。古い41/41を理由に全試験を繰り返さない。ただしbatch 2・SAM3.1・動画全体の性能/品質まで完了したとはしない |
| Attention backendが不明 | 解消済み。実機traceは`fmha_cutlassF_f16_aligned_64x64_rf_sm75`。既にmemory-efficientなので、巨大なscore行列を新たに全量削減できるという説明は使わない |
| cuDNN/BLAS preferenceの切替 | 実測済み。単なる設定変更の再試験は不要 |
| cuBLASLt algorithm探索 | **一部実施済み**。FC1/FC2、FP32 accumulate、32 MiB workspace、3/7候補を実際に比較。単なるpreference確認ではない。FC1は不利、FC2は約1%の差にとどまった。同条件の再走査はしない |
| GEMM探索の未完部分 | QKV/outputの明示的FP16候補探索、独自CUTLASS kernel、全configuration、別workspace、batch 2、実重みの広範検証は残る。「全GEMMを尽くした」とは扱わない |
| GELU近似・組込みepilogue | tanhと`_addmm_activation`は実モデルで試験済みで、有利ではなかった。同じ候補は繰り返さない。独自の丸め維持epilogueは別案であり、未否定 |
| INT8境界融合・投影量子化 | 実装・計測・5ケース確認済み。MLP境界融合は従来INT8比約4.8%、QKV追加は同一シリーズ比較で約1.8–1.9%。output投影INT8と行単位復元は推奨できる改善なし |
| Graph / LUT / column-major | 文書の却下根拠の一部はBlackwell。Turingで普遍的に不利と断定しない。ただし新しい仮説なしに同じ案を再走査もしない |

参照: [GEMM探索](GEMM_SEARCH.md)、[実機backend](CUDNN_KERNELS.md)、[MLP融合](BOUNDARY_FUSION.md)、[追加最適化](NEXT_OPTIMIZATIONS.md)、`build/native-windows-cu130/windows-validation-projection.log`（51/51、135.81秒）。

## donor再確認で残った有効な指摘と補正

`ssiu/flash-attention-turing`をread-only調査用に`.cache/portability-review/flash-attention-turing`へ取得した。HEADは`9ef98fcb506bb1e2fe3cece50935e2935bf6b124`。traits / launcher / forward kernelのblobは添付の記録と一致した。現ランタイムへの組込みやパッケージインストールはしていない。

### streamと共有メモリ: 指摘は有効

launcherは256 threads、dynamic shared memory 65,536 bytes固定で、CUDA起動の第4引数にstreamを渡していない。attribute設定の戻り値確認もない。移植時はcurrent stream、device guard、CUDAエラー確認、非default stream試験が必要。

d=64、128×128のソースではQ/OとK/Vがそれぞれ同じ共有領域を使うため、配列寸法からは合計32 KiBになる。ただし今回CuTeの`cosize`をコンパイル評価したり、全kernelの資源量や境界を実行検証したりはしていない。**32 KiB化は試す価値がある仮説であり、2 CTA常駐・速度2倍の確定事項ではない。** registers、spill、必要alignmentも確認する。

一次ソース: [固定予約とlaunch](https://github.com/ssiu/flash-attention-turing/blob/9ef98fcb506bb1e2fe3cece50935e2935bf6b124/csrc/flash_attn/src/flash_fwd_launch_template.h)。

### layout: 全QKVをpackし直す必要はない

donorはtoken-major `[B,N,H,D]`のstrideを内部で構築する。現ランタイムのpaired RoPE成功経路は、Q/Kの`[B,H,N,64]` viewに対して **head stride=64、token stride=1024** を保証している。従ってQ/Kの実メモリは既にdonorに適合し、view変更で接続できる。

対してVはpacked QKV由来でtoken stride=3072。初期の素直な接続ではVのみpackするか、donorにVのrow/batch strideを渡す。B=1、5184tokenのVは10.125 MiBであり、これを全32層でpackすれば論理的read+writeは約648 MiB。実測時間はまだない。

出力もtoken-majorなら、そのままVision output projectionの入力として扱える可能性がある。一般の非連続tensor・fallbackにこの前提を広げず、対応条件でdispatchする。

参照: `native/src/rotary_pair.cpp`の`ordinary`条件、`native/src/vision_encoder.cpp`、[donorのstrideと共有領域](https://github.com/ssiu/flash-attention-turing/blob/9ef98fcb506bb1e2fe3cece50935e2935bf6b124/csrc/flash_attn/src/flash_fwd_kernel.h)。

### 64行tile: 単純なパラメータ変更ではない

forward traitsの`TiledMma`は`Layout<Shape<_8,_1,_1>>`、`Tile<_128,_32,_8>`に固定されている。`kNWarps_`や`kBlockM_`を変えるだけで64行tile・4warp版まで成立するとは言えない。warp配置、fragment、copy partitionを合わせて設計し直す必要がある。

最初はdonorの128×128と576/5184のtail経路を検証し、共有予約・stream・strideを個別に直す。64 tileはその後の独立候補とする。sliding-window mask未対応は、既に窓をbatch分割しているこのVision経路のblockerではない。

なお現在のLibTorch backendも実kernel名は64×64 tileである。64 tileという寸法だけを新しい高速化要因として数えず、donor側の仕事分割・データ再利用との差を測定する。

参照: [forward traits](https://github.com/ssiu/flash-attention-turing/blob/9ef98fcb506bb1e2fe3cece50935e2935bf6b124/csrc/flash_attn/src/kernel_traits.h)。

### ビルド・由来の追加確認点

donorの`setup.py`はWindowsでC++20と`/Zc:preprocessor`を使い、nvccには`--use_fast_math`も指定する。現在のnative CUDA C++17設定へ機械的にコピーせず、必要なコンパイル条件と数値差を切り分ける。Windows用の分岐があることは、このMSVC/CUDA 13.0組合せでの動作証明ではない。

今回取得した親リポジトリのtracked treeにはLICENSE/NOTICEファイルを見つけられず、検索対象の実装にも明示的なlicense表記を確認できなかった。直接コードを組み込む前に由来・利用条件の確認が残る。CUTLASS submoduleの利用条件を親実装へ自動的に当てはめない。submodule参照は`df18f5e4f5de76bed8be1de8e4c245f2f5ec3020`。今回は親ソースの調査のみで、submoduleの取得・実行はしていない。

## 現在の実測に基づく順序

1. **d=64 FP16 Attentionの限定比較。** global 4層で約40.6 ms、local 28層で約31.9 ms。まずglobalで、既存SDPAとpack込みに比較する。量子化を追加しなくてもINT8 MLP/QKV構成と併用できる。全Vision・推論全体に戻して比較するまで速度向上は未確定。
2. **QKV INT8の復元処理。** 現状はGEMM約26.5 ms、量子化5.3 ms、復元17.7 ms、別途RoPE約6 ms。GEMM epilogueでINT32→FP16復元を行う案、または復元＋RoPEが次の候補。INT8 accumulation、scale、bias、Half丸め、WindowsのRoPE FMA規約を維持して比較する。既に試した起動配置の変更を繰り返す案ではない。
3. **FP16 GEMMの未探索部分は限定して比較。** QKV/outputの未探索アルゴリズムや独自epilogueに具体的な仮説が立った時に実施。既存FC1/FC2 heuristic探索を最初からやり直す優先度は低い。
4. **Detectorの固定区間Graphは別候補。** 以前のtraceではDetector stageの約143 msに対してkernel sumは約86 msだった。差分すべてをCPU計算やGraphで消せる時間と扱わないが、host launch gapを別途調べる理由はある。全predictorのcaptureではなく、`.item()`等の同期・動的処理を除いた固定区間、別process比較、追加pool込みに限定する。Blackwellの否定結果だけでTuringのこの区間まで却下しない。

shape別kernel時間は[QKV trace](hotspots-qkv.json)を使用。理論FLOPs比85.48%やBlackwellのGEMM比率は、この順序の時間根拠には使わない。現INT8経路ではGEMM以外の量子化・復元も無視できない。

## 後回しにするが、否定はしない項目

- FlashInferのd=256 shared-memory修正PR #3526は、公開ページでも2026-07-22 merge済みと確認した。動画memory attentionの候補として有効だが、今回の画像推論の40 ms global Attentionとは別経路。SAM3.1、pointer tail、動画編集等の回帰は未完のまま明記する。[PR一次資料](https://github.com/flashinfer-ai/flashinfer/pull/3526)
- d=32→64 paddingはscale維持が必要で、無料の高速化ではない。Detectorのbias/mask付き経路へ無条件適用しない。
- conditional rescale / polynomial expは、動くAttention基準と対応するhardware counter取得後の段階。SFUが律速であるという新たな証拠は今回得ていない。
- 独自の丸め維持FP16 GELU epilogueは未検証。ただし既存のINT8境界融合を無視して第一候補に戻す理由はない。

今回の再評価は、全モデル・全batch・動画までの最速/品質保証を更新するものではない。確認済みの画像推論の証拠を使い、未確認の範囲を狭めた。
