# 継続最適化のメモ

元の50候補に続き、同じコンテナのPythonとRTX 3090で試す。各結果のJSONに設定・時間・メモリ・stock/FP16との差を残す。Turing実機の速度を推定値で置き換えない。

## Round 6

GPUイベントによる内訳では、コンパイル済み画像ViTが約93ms、検出デコーダーが約26ms。これを優先して試す。イベントでの内訳は入れ子の区間であり、別プロセスのA/B測定と単純に合算しない。

- `mha_sdpa`: PyTorch標準のMultiheadAttentionに不要なAttention重みを作らせない。
- `mha_half`: Linearだけの変換では残っていたAttentionのin-projection重みもFP16で保持する。
- `decoder_half_ffn`: FP32に保護していた小さい検出FFNをFP16化する。
- `vision_half_residual`: ViT全体をhalf化し、残差のFP32への昇格を防ぐ。
- `compile_encoder`, `compile_decoder`, `compile_text_uncached`: 未コンパイル部分に適用する。前二者のreduce-overheadはCUDA Graph出力の上書きエラーになった。失敗JSONを残し、Round 7でGraphを使わないコンパイルも試す。
- `global_kv_pool2`, `late_global_kv_pool2`: 全解像度のQueryを保ち、global AttentionのKey/Valueだけ2×2平均で減らす。近似なので標準パッチへの自動採用はしない。

トークン削減の方向は、Sciteで見つけた[StructSAM](https://arxiv.org/abs/2603.07307)を参考にした。今回のK/V平均化は独自の簡易候補であり、StructSAMの実装・再現ではない。GitHubの上流Issue検索も参照したが、外部の性能報告をこの環境での実測値として扱わない。

## Round 7

- `static_rpb`: 既知の特徴サイズをPython整数で渡し、box RPBのcache比較/assertでGPUスカラーを読む待ち時間をなくす。
- `compile_rpb`: RPBの小さい点ごとの演算をまとめる。
- `last_scores`: 最終層しか使わないeval時のスコアとbox計算を最終層だけにする。
- `native_mha`: 独自MHAから現行PyTorch functional MHAへの置換。
- `trim_text_*`: 因果Attentionの末尾paddingを8トークン単位で切り詰める。EOSまで残す。
- `embedding_half`: 単語埋め込み重みもFP16で保持する。
- `cpu_resize`: PILで先にリサイズしてからGPU転送。補間の結果差も測る。
- `*_compile_default`: CUDA Graphを使わないコンパイル。
- `int8_fc1`, `int8_mlp`: 再学習・較正なしのper-token動的INT8とper-output-channel重みINT8。Tritonで量子化し、torch._int_mmで計算する。

## マスク出力

`fused_masks_bench.py`は学習済み200 queryの低解像度logitを使い、4K出力の補間・sigmoid・閾値処理・bitpackを1つのTriton kernelに融合する候補を比較する。モデル全体の速度ではなく出力部分だけの測定。FP16の丸めを残すため、logit > 0への単純な置換は使わない。

## Round 8（組み合わせと追加仮説）

Round 7のdecoder default compile（137ms）を、FP16のAttention/単語埋め込みと組み合わせる。CUDA Graphを使う場合は、後続呼出しによる出力上書きを防ぐcloneを境界に置いて比較する。

ViTのMLPは、元の解像度の残差を残したまま2×2平均した特徴だけを処理して戻す案を試す。後半8層、4層ごとの8層、全層を比較する。2ブロックの省略も別候補にする。これらは精度との交換条件を調べる近似候補。

テキスト側は、文字列の前処理をコンパイル対象から外し、TransformerのTensor演算だけをコンパイルする案を追加する。文字列全体を含む版では`unicodedata.normalize`でGraph breakを確認した。

## 現時点の判断

- DecoderをCUDA Graphで動かし、その出力をcompileの外でcloneすると121.21ms。Graph出力の上書き失敗を解消できた。Round 9で他のメモリ削減と組み合わせる。
- 単語埋め込み + Attention重みのhalf保持はCUDA allocatedを2.125→1.878GiBへ削減。
- 画像MLPの両方をINT8計算にすると135.85ms、平均mask IoU 0.99739、検出数15で一致。任意の近似候補。
- テキストMLPのINT8はNVML 2.817GiB。ただし新規語句を毎回処理する速度は183.68msとなり、メモリ優先の候補。
- MLP全層の空間平均化は126.90msだが、検出数が15→3に減るため不採用。平均IoUの集計は対応したマスクのみなので、検出数も必ず見る。
- 全groundingのcompileはgeometry encoder内の`pin_memory`で失敗。Round 9では空の幾何promptに対する不要な計算を省く候補も試す。
- 4Kマスク融合は採用し、`d0bf52f`でpush済み。8.18ms・0.257GiB。従来経路も選べる。

## 次の探索候補

- 平均化するMLP tokenを一律に選ばず、2×2セル内の特徴分散が小さい領域だけをまとめる。高分散セルは全tokenを残し、計算量と境界保持の両方を見る。
- INT8のper-token最大値が一部のchannelに支配される場合に、channelごとのスケール調整を量子化kernelへ融合する。まず追加較正なしの重み統計による案から試す。
- 4K pack kernelはblock/warp数とFP16の二値化cutoffをさらに比較する。`tune_mask_pack.py`と`fused_mask_pack.py`の追加設定は次の測定用。

## Round 11：特徴分散で選ぶMLP共有

2×2セル内の特徴分散が小さい部分だけMLP入力を平均し、高分散部分は4トークンを残す。
MLPの後で元の配置に戻し、残差は元の解像度を保つ。`adaptive_mlp.py`に候補を保存した。

| 設定 | ms | 平均mask IoU vs stock | 検出数 | 判断 |
|---|---:|---:|---:|---|
| 採用FP16の再測定 | 120.57 | 0.99920 | 15 | 基準 |
| 75%のセルを全計算 | 121.06 | 0.96351 | 15 | 車輪などの細部が崩れる |
| 50%のセルを全計算 | 109.07 | 0.91718 | 16 | 検出数も変化 |
| 後半16層だけ75%を全計算 | 129.50 | 0.98806 | 15 | 遅くなる |
| 採用INT8の再測定 | 96.95 | 0.99742 | 15 | 基準 |
| INT8 + 75%全計算 | 100.51 | 0.96358 | 15 | 既存INT8より不利 |

今回は採用しない。分散による選別・並べ替えのコストと、小さい形状の変化が大きい。

## Round 12以降の候補

- Round 12：重みの列ごとの最大絶対値からスケールを作り、重みと入力に逆方向の
  スケールを掛けてからINT8化する。入力側は量子化kernelに融合する。
  追加の較正画像は使わない独自の重み統計ヒューリスティックであり、SmoothQuantの再現ではない。
  `alpha` 0.25 / 0.5 / -0.5、fc1だけ・fc2だけ・両方を比較する。
- Round 13：文字列promptのgrounding全体をコンパイルし、出力を必要な4配列に絞ってから
  Graph外でcloneする。box/point/mask promptは元の経路を使う試作。
- Round 14：画像MLPに加えてViTのQKV射影・出力射影・fusion encoderのFFNをINT8化する。
- Round 15：INT8の1段目の逆量子化→FP16 GELU→2段目の量子化を融合し、中間配列を省く。
  warp数とGELUのerf/tanh経路を比較する。

### Round 12の結果

| スケール設定 | ms | 平均mask IoU vs stock | 変化画素 |
|---|---:|---:|---:|
| INT8基準 | 100.30 | 0.99742 | 831 |
| alpha 0.25・両方 | 99.65 | 0.99788 | 752 |
| alpha 0.5・両方 | 99.25 | 0.99610 | 1125 |
| alpha 0.5・fc1のみ | 98.61 | 0.99732 | 808 |
| alpha 0.5・fc2のみ | 99.80 | 0.99691 | 1110 |
| alpha -0.5・両方 | 97.55 | 0.99779 | 812 |

全設定で検出数1/4/6/4/0は一致した。alpha 0.25に小幅な改善はあるが、
追加設定を標準INT8へ採用するほどの差ではないため、候補として保存する。
メモリもCUDA allocated 1.586〜1.588GiB、NVML 3.042〜3.062GiBでほぼ同じ。
### Round 13の結果

出力を4配列に絞ってGraph外でcloneする版は、FP16基準123.12msに対して115.36ms。
CUDA allocatedは1.875→1.862GiB、NVMLは3.104→3.097GiBだった。
5条件の検出数は一致し、変化画素は285、平均mask IoUは0.999205。
Graphを使わないdefault compileは114.15msだがNVMLは3.208GiB。
INT8との組合せは92.64ms。feature view・固定metadata・最終層だけのscore計算を追加しても
92.79msで、追加効果はなかった。

通常パッチではprocessor内だけでコンパイル結果を4配列に絞り、直接のmodel呼出しには
元の出力辞書を返す形に変更した。box/point/mask promptとearly_filterは従来の段ごとの
コンパイル経路を使う。INT8・packed maskを併用して、画像状態の再利用、box prompt、
空出力、固定語句、直接modelを呼ぶ場合の出力を確認し、採用した。
Round 14の公開API再測定はstock 210.81ms、FP16 114.31ms、INT8 92.26msだった。

Round 14以降はこの更新版を基準に測る。Round 16には、重みだけINT8で保持し、
入力はFP16のままTensor Core演算する別案も用意した。列ごとの重みスケールを
積和の後に掛けることで、完全なFP16重み配列の展開を避ける。

## Round 14：INT8を適用する場所

| 追加対象（MLPのINT8が基準） | ms | CUDA allocated GiB | NVML GiB | 平均mask IoU | 変化画素 |
|---|---:|---:|---:|---:|---:|
| 追加なし | 92.26 | 1.625 | 3.099 | 0.99740 | 836 |
| QKV射影 | 88.98 | 1.555 | 3.130 | 0.99735 | 891 |
| 出力射影 | 92.15 | 1.530 | 2.997 | 0.99721 | 885 |
| 両方 | 88.61 | 1.495 | 2.896 | 0.99694 | 950 |
| window Attentionの両射影だけ | 90.58 | 1.474 | 2.915 | 0.99706 | 942 |
| fusion encoderのFFN | 92.64 | 1.568 | 3.038 | 0.99734 | 836 |
| 両射影＋fusion FFN | 88.84 | 1.504 | 2.886 | 0.99690 | 948 |

全構成で1/4/6/4/0の検出数は一致。Attentionの両射影を追加する案を公開APIの
`attention_projections=True`として採用した。公開APIでは90.19ms・1.455GiB allocated・
2.854GiB NVML、平均IoU 0.996938で検出数は一致した。Attention本体のQK/AV積はFP16のまま。
fusion FFNまで追加する効果は小さいため、ここでは採用しない。

Round 17では、MLP前のLayerNormと量子化を融合する。Round 18は固定語句・新規語句・
efficient Attentionの更新後の測定。Round 19はuint8 resizeを保った入力正規化の融合と、
画像neckも含めたコンパイルを比較する。

## Round 15：GELUと再量子化の融合

MLPのINT8基準92.61ms・1.624GiB allocatedに対し、通常GELUの融合は
4 warpsが91.76ms、8 warpsが88.93ms。後者のallocatedは1.573GiB、NVMLは3.030GiB。
両warp設定は出力差も同じで、平均IoU 0.997491、変化840画素、検出数1/4/6/4/0。
tanh近似は89.17ms・1.628GiB allocated・平均IoU 0.997269だった。

通常GELUの8 warpsを優先し、Round 17のLayerNorm融合・Attention射影INT8との組合せを
先に実行する。Round 16の重みのみINT8は準備済みで、その後に比較する。

## Round 17：LayerNorm融合と組み合わせ

| 構成 | ms | CUDA allocated GiB | NVML GiB | 平均mask IoU |
|---|---:|---:|---:|---:|
| MLPのINT8基準 | 91.53 | 1.624 | 3.079 | 0.99740 |
| LayerNorm融合・4 warps | 93.72 | 1.629 | 3.103 | 0.99751 |
| LayerNorm融合・8 warps | 93.51 | 1.629 | 3.103 | 0.99740 |
| LayerNorm＋GELU融合 | 91.91 | 1.628 | 3.071 | 0.99745 |
| GELU融合＋Attention射影INT8 | 84.91 | 1.455 | 2.819 | 0.99742 |
| LayerNorm＋GELU融合＋Attention射影INT8 | 84.81 | 1.509 | 2.860 | 0.99738 |

全構成で1/4/6/4/0の検出数は一致。最後の2構成は約0.1msの差で、LayerNormを含む方の
メモリが大きいので、通常GELUの8 warps融合＋Attention射影を選ぶ。
この組合せはstock比916画素が変化し、score最大差0.0078125、box最大差0.495画素。

公開APIに`fused_mlp=True`を追加し、Round 18で公開版・固定語句・新規語句・
efficient Attentionを測定する。Round 16はその後に実行し、続いてRound 19の画像入力、
Round 20の一括コンパイル下でのdecoder FFNのFP16化を試す。

## Round 18：公開APIと用途ごとの比較

公開APIのGELU融合は90.43ms、Attention射影も含めると85.46ms・NVML 2.872GiB。
後者の出力差は試作版と同じ916画素、平均IoU 0.997421で、5条件の検出数は一致した。
固定語句FP16は114.52ms・1.256GiB allocated・2.411GiB NVML。
固定語句のINT8＋射影＋GELU融合は84.18ms・0.850GiB allocated・2.401GiB NVML。
語句キャッシュを無効にしてtext compileを使うFP16は116.32msだった。
テキストMLPもINT8化した融合版はキャッシュ有効で85.10ms・NVML 2.831GiB、
無効＋text compileで87.64ms・2.849GiB。後者の平均IoUは0.997450、変化909画素。
efficient Attentionに固定するとFP16が118.83ms、融合INT8が90.93ms。
全9構成で検出数1/4/6/4/0は一致した。Turing実機の速度ではなく3090上の別経路の確認。

次のRound 21はLayerNormのaffine係数をfc1へ前計算する案。
`W * (gamma * (x - mean) / std + beta) + b`を、`W * gamma`と`W * beta + b`に分け、
入力量子化では中心化のみを行って逆標準偏差を出力scaleへ含める。
量子化前の実数演算では同値だが、INT8では誤差の分布が変わるためA/Bで確認する。

Round 22はMLPだけを省略し、Attentionと元の残差を残す候補。
checkpointのfc1/fc2重みRMSの積は後半の層ほど小さかったため、末尾1・2・4層と先頭1層を比較する。
これは実タスクの重要度を測った値ではなく、候補選びの簡単なヒューリスティック。
順位は`results/mlp_weight_ranking.json`に保存した。

## Round 16：重みだけINT8

FP16の基準114.44msに対し、独自GEMMはM64/N64が168.58ms、M128/N64が150.46ms、
M64/N128が130.01ms。タイルで改善したが、現状では通常のFP16より遅い。
Round 23ではCTAをM方向の小さいグループで並べ替え、L2キャッシュで入力・重みを
再利用する案を試す。個々の積和の順序は同じで、タイルを実行する順序を変える。

## TuringのSKU差

GTX 1660系はTuringだがTensor Coreを搭載しない。
[NVIDIA公式比較表](https://www.nvidia.com/en-eu/geforce/graphics-cards/compare/)と
[NVIDIAのRTX/GTX解説](https://blogs.nvidia.com/blog/whats-the-difference-between-nvidia-rtx-and-gtx/)で確認した。
RTX 2060などとのハードウェア差を含め、3090上のINT8やTensor Core GEMMの測定値を
Turing全体へ外挿しない。ここでの実行環境は引き続き3090。

## Round 19：入力と画像neck

Attention射影INT8（GELU融合なし）の基準88.74msに対し、uint8 resize後の
正規化融合はFP16出力88.51ms、FP32出力87.72ms。FP16出力の5条件の差は基準と同じ。
neck全体のcompileは88.56ms、正規化FP16との組合せ86.87msだった。
後者はCUDA allocated 1.454GiB・NVML 2.874GiB、平均IoU 0.996951・947画素変化。
全設定で検出数1/4/6/4/0は一致。基準との速度差が小さく、
neckのcompileには新たなGraph境界が増えるため、今回は実験候補として保存する。

Round 16の重みのみINT8は全タイルで同じ出力差（平均IoU 0.999047・338画素変化）。
標準FP16より出力差は増えるが、動的INT8より小さい。速度がまだ不利なため、
Round 23のCTA並べ替えを先に実行して改善余地を見る。

## 追加候補の準備

Round 24はMLPの中間channelを重みノルムとbiasで順位付けし、残す幅を64の倍数に揃える。
98%・95%・90%・75%を残す全層版と、後半8層だけ75%にする版を比較する。
再学習や較正なしの近似であり、重要度を正解ラベルで確認したものではない。

Sciteで[PTQ4SAM](https://arxiv.org/abs/2405.03144)と
[SAQ-SAM](https://arxiv.org/abs/2503.06515)を見つけ、原著の概要を参照した。
HFのpaper_search connectorは利用できなかったため、既存の`hf papers search`も使用。
これらの手法を再現するのではなく、値の分布を考慮する簡単な候補を試す。
Round 25はGELU後だけを非対称INT8にする。tokenごとのmin/maxからscaleとzero pointを作り、
`INT8積 - zero_point * 重みの行和`で補正してから逆量子化する。
正側へ偏ったGELUの分布にINT8の範囲を多く割り当てる案で、追加学習は行わない。

Round 23の並べ替えは、[Triton公式GEMM資料](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)の
M方向のグループ化と同じ配置。上流コードもGitHub connectorで参照した。

## Round 23：重みINT8のGEMM並べ替え

グループ8・M64/N64が124.92ms、グループ8・M64/N128が117.71ms、
グループ4・M64/N128が118.74ms、グループ8・M128/N64が127.69ms。
全設定でallocated 1.575GiB・NVML 3.097GiB。
並べ替え前と同じ5条件・15マスク・338画素変化・平均IoU 0.999047で、
最速版は130.01→117.71msへ改善した。標準FP16の114.44msよりまだ遅いので、
公開パッチには追加せず、後で再利用できるkernel候補として保存する。

Round 26は、以前eagerで試したViT残差のFP16保持を現在の一括コンパイルと組み合わせる。
各blockの出力だけFP16へ丸め、INT8のscale bufferはFP32のまま残す。
画像block間の特徴メモリを減らす効果と、累積する丸め差を比較する。

`experiments/tune_weight_only.py`はMLPの実際の行列サイズ（5184×1024×4736と
5184×4736×1024）だけを使った短いkernel探索。入力・重みは乱数であり、
実画像の出力評価とは別に扱う。良いタイルだけ後続のモデルA/Bへ回す。

Round 24のchannel順位は、実行前にLayerNormの係数とbiasを含む形へ調整した。
正規化済み入力を独立な標準正規分布と仮定し、fc1の平均・分散を近似する。
同じ仮定で除去channelのGELU平均をfc2のbiasに足す95%・75%版も比較する。
この仮定の当否を較正で追い込まず、実画像A/Bで有用性を判断する。

Round 27はmaskを作る直前に最終scoreでqueryを8/16/32/64個へ絞る。
Decoder本体は全200 queryのまま、mask head以降だけ固定幅にするため、一括compileできる。
上限を超える検出を捨てる実験候補であり、一般入力で元と同じ検出数を保証するものではない。
最初は5条件で速度・差を確認し、効果があれば上限超過時の処理を検討する。

## Round 20：Decoder FFNをFP16へ

通常FP16構成では115.20→115.96ms、allocated 1.862→1.850GiB、NVML 3.095→3.157GiB。
INT8＋Attention射影＋GELU融合では86.61→85.45ms、allocated 1.456→1.443GiB、
NVML 2.880→2.804GiBだった。後者のstock比は916画素変化・平均IoU 0.997421で、
検出数は両系統とも1/4/6/4/0が一致。速度差が小さく、通常FP16の改善にはならないため、
公開標準のFFNは現状のFP32を保ち、候補を保存する。

Round 28は量子化kernelで複数tokenを1つのCTAへまとめる。入力の量子化を4/8行、
GELU＋再量子化を1/2/4行にして、CTA数とwarp数の組合せを比較する。
最大絶対値による対称INT8とFP16の丸めは元の式を保つ。

## Round 21：LayerNorm係数を重みへ前計算

GELU融合込みの前計算版は4 warps 89.31ms、8 warps 90.22msで、
いずれもallocated 1.573GiB・NVML 3.030GiB。stock比の平均IoUは0.99552/0.99520、
変化画素は1384/1438だった。Attention射影INT8との組合せは84.20ms、
allocated 1.454GiB・NVML 2.843GiB・平均IoU 0.995535・1380画素変化。
5条件の検出数は全構成で一致した。対応する通常融合版85.39ms・916画素変化と比べ、
速度差は小さく出力差が増えるため、ここでは採用しない。

次はRound 27のmask生成数を先に比較する。準備済みの22・24〜26・28も順次実行する。
