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

Round 29は添付実験のmask headの積の再結合を、現在の一括compileで再比較する。
最初のラウンドから保存している`static_reassociation` / `inplace_reassociation`を使い、
pixelごとの1×1射影をquery側へ移す。以前のeager比較では不採用だった候補。

## Round 27：mask生成query数を固定で削減

INT8融合版の基準85.42msに対し、上位8/16/32/64 queryだけmaskを作る版は
84.64/86.15/85.33/85.90msだった。全設定で5条件の検出数は一致し、stock比の差は
917画素・平均IoU 0.997421（基準916画素）。FP16の上位32個版も114.26msと従来とほぼ同じ。
検出数に上限を設ける割に効果が小さいため、公開パッチには取り込まない。
元の200 queryを処理する構成を維持する。

Round 30は現在のINT8・射影・GELU融合を896/840/784/672/560入力で測る。
以前の低解像度比較からコンパイル・量子化が変わったため、速度重視の任意設定の
時間と形状差を更新する。通常の1008設定を置き換えるものではない。

Round 31は重みをINT8で保存し、各Linear呼出しでFP16に戻して`F.linear`へ渡す案。
入力は量子化せず、独自行列積も使わない。MLPのみ・Attention射影も含む版を比較する。
compilerが展開した一時重みを再利用できるか、常駐メモリと速度の両面を見る。

## Round 22：MLPだけを省略

INT8融合の基準86.43ms・平均IoU 0.997421・916画素変化に対し、
末尾1層省略は84.51ms・0.993962・2113画素、2層は83.28ms・0.992465・2872画素、
4層は82.38ms・0.990636・3517画素だった。末尾側の検出数は1/4/6/4/0で一致。
先頭1層省略は85.27ms・平均IoU 0.976140・11277画素変化に加え、
車輪の検出数が4→3へ減った。平均IoUは対応した14マスクだけの値である。
NVMLは2.864〜2.880GiBとほぼ同じ。省略した演算量に対して形状差が大きいため、
公開パッチへの追加は見送り、全候補を保存する。

## Round 25：GELU後の非対称INT8

Attention射影INT8を含む基準85.00ms・916画素変化に対し、非対称4 warpsは
85.61ms・735画素・平均IoU 0.998051、8 warpsは87.11msで同じ出力だった。
box最大差は0.495→1.436画素、score最大差は0.00781→0.01172へ増えた。
マスクの差が減った中心はchildとwheelだが、box最大差が増えたのもchildだった。

Attention射影をFP16で残すMLPだけの非対称版は91.23ms、allocated 1.573GiB・
NVML 3.030GiB、544画素・平均IoU 0.998477、score最大差0.00586、box最大差0.475画素。
全構成で検出数1/4/6/4/0は一致。出力差を抑える選択肢として公開APIへ追加予定。
通常の速度優先構成と分けて`asymmetric_gelu=True`で指定し、4 warpsを使う。
Round 28終了後に公開コードを切り替え、Round 32の公開APIで比較する。

## Round 28：複数tokenをまとめた量子化

基準85.39msに対し、入力4行/GELU1行は86.91ms、8行/1行は86.29ms、
4行/2行は87.17ms、8行/2行は87.09ms、4行/4行（16 warps）は87.47msだった。
全設定で5条件の出力差は基準と同じ916画素・平均IoU 0.997421、検出数も一致。
allocatedはいずれも約1.455GiB。速度改善がないため採用しない。

Round 28の終了後、`asymmetric_gelu=True`を公開APIへ追加した。
GELU後のzero pointとfc2重み行和によるINT32補正を使い、4 warpsで実行する。
既存の対称GELU kernelはそのまま残した。Round 32で公開版を測定中。

## Round 32：非対称INT8の公開API

公開版の通常構成は85.16ms・1.454GiB allocated・2.884GiB NVMLで、出力差は従来と同じ。
非対称GELUのMLPのみ版は89.47ms・1.628GiB allocated・3.190GiB NVML、
平均IoU 0.998477・544画素変化・score最大差0.00586・box最大差0.475画素。
Attention射影もINT8にする版は85.01ms・1.456GiB allocated・2.851GiB NVML、
平均IoU 0.998051・735画素変化・score最大差0.01172・box最大差1.436画素だった。
全3構成で5条件の検出数は1/4/6/4/0で一致した。

`asymmetric_gelu=True`を任意設定として採用した。`fused_mlp=True`と組み合わせ、
vision MLPのGELU後だけにzero pointを導入する。重み行和は適用時に前計算する。
公開コードからの比較を保存し、単独パッチも再生成・逆適用チェック済み。
次はRound 31の一時FP16重み、kernelのタイル探索、残る構造変更候補を続ける。

重みだけINT8の独自GEMMには、入力を先にFP16へ変換する候補も追加した。
従来はkernel内でロードした値をFP16へ丸めていたため、LayerNormがFP32を返す場合に
同じ入力をタイルごとにFP32で読み直していた。先に変換すればLayerNormの出力変換との
融合や転送量削減が期待できる。演算そのもののFP16丸めは保ち、モデルA/Bで確かめる。
タイル探索は元からFP16入力を使うため、この候補に合わせて全体で再確認する。

## Round 31：一時的に重みをFP16へ展開

通常FP16は115.82ms・allocated 1.862GiB・NVML 3.095GiB。
MLPの重みのみINT8保存では116.78ms・1.574GiB・3.067GiB、343画素・平均IoU 0.999030。
Attention射影も含む版は116.22ms・1.455GiB・2.897GiB、389画素・平均IoU 0.998763。
後者のscore最大差0.00537、box最大差0.512画素。両方とも検出数1/4/6/4/0は一致した。
速度をほぼ保ち、活性を量子化する方式より差を小さくできるメモリ優先の候補。

## 重みINT8のkernelタイル探索

FP16乱数入力で各MLP形状30通り、計60通りを比較した。全kernelは有限値を返した。
展開したFP16重みの参照積との最大差は最速候補で0.001953125。
前半の通常FP16は0.798ms、M64/N128/K32・group16が0.741ms。
後半は通常FP16 0.708ms、同タイル・group4が0.737msだった。
これはkernelだけの乱数入力測定であり、実画像の比較とは分ける。
Round 33で入力FP16化と前半group16/後半group4の全体性能を確認する。
直前の通常FP16基準はRound 31の115.82msを使う。

## Round 33：独自GEMMのFP16入力と形状別グループ

FP16入力＋group8は119.07ms、FP16入力＋前半group16/後半group4は118.32ms、
元の入力dtype＋形状別groupは120.07msだった。全構成で検出数1/4/6/4/0、
stock比338画素・平均IoU 0.999047と、従来の独自GEMMと同じ差。
直前の通常FP16 115.82ms、および重み展開＋通常GEMM 116.78msを上回らないため採用しない。
単体のkernel探索が、必ずしもモデル全体の短縮にはつながらなかった。

Round 33後に、通常GEMMを使う`weight_only=True`を公開APIへ追加した。
動的INT8の`fused_mlp`とは併用しない。Round 34で公開版のMLPのみ・射影込みを測定する。

## Round 34：重みのみINT8の公開API

MLPのみは115.57ms・allocated 1.574GiB・NVML 3.030GiB、
平均IoU 0.999030・343画素変化・score最大差0.00391・box最大差0.509画素。
Attention射影も含む版は117.29ms・1.456GiB・2.862GiB、
平均IoU 0.998763・389画素変化・score最大差0.00537・box最大差0.512画素だった。
全5条件で検出数1/4/6/4/0は一致し、試作版と同じ出力差を確認した。

`weight_only=True`として採用した。選択したLinearはINT8重みを一時的にFP16へ展開し、
通常のFP16行列積を使う。動的INT8の`fused_mlp` / `asymmetric_gelu`とは併用しない。
単独パッチを再生成し、適用チェックも通した。続いてRound 26の残差FP16保持を測定する。

## Round 26：ViT残差をFP16で保持

通常FP16は114.06→112.33ms、allocated 1.862GiB・NVML 3.095GiBは同じ。
stock比285→298画素、平均IoU 0.999205→0.999132だった。
INT8融合は85.02→84.38msだが、allocated 1.455→1.510GiB、NVML 2.864→2.884GiB、
変化916→1009画素、平均IoU 0.997421→0.996831。全条件で検出数は一致。
改善は小さく、INT8版ではメモリと差が増えるため、公開パッチへの追加は見送る。

## Round 35の候補：重みだけで刻み幅を選ぶ

各出力行のINT8スケールをabsmaxの80〜100%から選び、重みの二乗誤差を最小化する。
追加で量子化値を固定した最小二乗スケール更新を2回行う案も用意する。
画像や活性による調整はせず、初期化時だけの探索で推論の演算は変えない。
対称・非対称GELUおよび重みのみINT8に組み合わせ、実画像の差が減るか測る。

## Round 36の候補：Attentionヘッドを少数削る

ViTの16ヘッドから出力エネルギーが小さそうな1〜2ヘッドを削る。
LayerNormの重みとV射影・出力射影から簡易的に順位を付け、Q/K/Vの対応する行と
出力射影の列を一緒に削除する。残ったヘッドの次元とRoPEは変えない。
削ったVの平均を出力バイアスに足す案も用意し、globalのみ・windowのみ・全層を比べる。
画像による較正や再学習はなく、この重み統計が実際の重要度を表すかは実測で判断する。

## Round 37の候補：INT8行列積の出力処理を融合

現在はINT8行列積のINT32出力を保存してから逆量子化・GELUを処理している。
Triton行列積の末尾でscale/bias/FP16丸め/GELUを計算し、FP16だけを書き出す案を試す。
後段の入力量子化は別kernelのままなので、行単位の最大値も従来通り使える。
fc2もFP16出力まで融合する版とfc1だけの版、4種類のタイルを比較する。
元のINT8積はINT32蓄積を保ち、GELU前後のFP16丸めも保つ。

## Round 38の候補：グループ単位の4bit重み保存

重みを16/32/64/128チャネルごとの4bitへ量子化し、2値を1バイトに保存する。
各グループにFP32 scale/offsetを保持し、Linearの直前にFP16重みへ展開する。
活性と行列積はFP16のまま。MLPとAttention射影を対象にするが、MLPのみの案も比較する。
対称±7とグループmin/maxの16水準も比較する。圧縮は重みとグループ情報を含めて評価する。

## Round 24：MLPのチャネル削減

| 保持率・範囲 | ms | 平均IoU vs stock | 変化画素 | 検出数合計 |
|---|---:|---:|---:|---:|
| 基準 | 85.67 | 0.997421 | 916 | 15 |
| 全層98% | 84.90 | 0.993689 | 2290 | 15 |
| 全層95% | 84.23 | 0.990927 | 3362 | 15 |
| 全層90% | 81.90 | 0.982748 | 5854 | 15 |
| 全層75% | 77.95 | 0.929640 | 22983 | 15 |
| 後半8層のみ75% | 83.56 | 0.991538 | 3268 | 15 |
| 平均出力を補正・95% | 84.22 | 0.983226 | 6606 | 15 |
| 平均出力を補正・75% | 79.11 | 0.649087 | 74545 | 5 |

保持数は64チャネル単位に切り下げている。補正75%は検出も減り、IoUは対応した5マスクだけ。
補正なし75%でもscore最大差0.347・box最大差54.1画素と大きかった。
小さく削る場合の短縮は1〜2ms程度で、全候補を不採用として保存する。
独立・標準正規という簡易仮定による平均出力補正は、今回はむしろ差を増やした。

## Round 39の候補：テキストエンコーダーをCPUへ

自由な語句を受け付けながらGPU常駐重みを減らすため、テキストエンコーダーをCPUの
FP32へ移す。返る特徴だけFP16でGPUへ戻し、既存のテキストLRU cacheはそのまま使う。
FP16画像版・INT8画像版それぞれで、cacheありとcache無効を分けて測定する。
cache無効の時間には語句をCPUで毎回エンコードする時間も含まれる。
画像品質は全5条件で確認し、CPU/GPUで演算精度が違うことによる差も記録する。

Round 37の初版には`tl.dot`の`out_dtype=tl.int32`指定漏れがあった。
直接実行では型assertに失敗し、torch.compile経路ではmutation解析失敗の後、
全条件で検出が0になった。59.73msという測定は無効と明記して表から除外した。
2候補目は中断し、未実行候補もまとめて`round37_retry.json`で比較し直す。
INT32を明示した修正版は、SAM3の2つのMLP形状を含む単体比較で有限値、
通常のINT8積＋FP16後処理との最大差0.0009765625、平均差1e-9未満を確認した。

## Round 40の候補：検出デコーダーの層数

検出デコーダーは各層でboxを更新し、中間特徴も返すため、6層を4〜5層へ減らす案を試す。
先頭から残す版と最終層を残す版を比較する。query数200とマスクheadは維持する。
重みを削って速度を優先する近似なので、検出数・score・box・maskの差を一緒に確認する。

比較用CLIの進捗表示にも各条件の検出数を追加し、変化画素数はstock比と明記した。
従来の進捗表示はFP16比を優先していたため、数値の参照先をJSONの集計表に統一する。
過去のJSONに保存されたstock/FP16の両比較は変更していない。

## Round 37修正版：INT8行列積と後処理の融合

基準86.25ms・916画素変化・平均IoU 0.997421。両行列積を融合した版は
M64/N128/K64/4 warpsで84.49ms、M128/N128/K64/8 warpsで83.97msだった。
M32/N128は94.69ms、M64/N64は91.56msへ遅くなった。
両段融合の全4設定で変化1021画素・平均IoU 0.997015・score最大差0.01270・
box最大差0.585画素。検出数は1/4/6/4/0で一致した。

fc1だけをM64/N128で融合し、fc2は既存処理に残す案は84.61msで、
5条件の全比較値は基準と一致した。allocated 1.457GiB・NVML 2.897GiB。
両段融合の最速版もallocated 1.456GiBで、モデル全体のピーク削減は見られなかった。
公開デフォルトは維持し、fc1のみ融合を出力差の増えない改善候補として保管する。
INT32出力型の指定漏れによる初版の不正出力・中断記録も残している。

## Round 41の候補：fc1融合のタイルと非対称GELU

Round 37で出力差が増えなかったfc1のみ融合を、大きいタイルでも比較する。
非対称GELUとの組み合わせは、fc1からFP16 GELUを出力してからmin/maxで量子化し、
fc2は既存のINT8積とINT32 zero-point補正を使う。
初期の両段融合の差を避け、非対称版の小さいマスク差と速度短縮を両立できるか確認する。

CPUテキストの最初の3結果は、FP16 cached 115.15ms・allocated 1.200GiB・NVML 2.370GiB、
FP16 uncached 154.58msで同じメモリ、INT8 cached 86.24ms・0.793GiB・2.543GiBだった。
いずれも5条件の検出数は一致している。公開用`offload_text_encoder`を別ファイルに準備し、
Round 39の終了後に公開コードへ移してRound 42で再測定する。
新しいCPU/GPU混在経路については既存のAPI smokeも一度実行する。

## Round 39：自由な語句を保ってテキストをCPUへ移す

| 構成 | ms | allocated GiB | NVML GiB | stock比画素差 |
|---|---:|---:|---:|---:|
| INT8画像・GPUテキストcached基準 | 85.62 | 1.455 | 2.831 | 916 |
| FP16画像・CPUテキストcached | 115.15 | 1.200 | 2.370 | 289 |
| FP16画像・CPUテキストuncached | 154.58 | 1.200 | 2.370 | 289 |
| INT8画像・CPUテキストcached | 86.24 | 0.793 | 2.543 | 915 |
| INT8画像・CPUテキストuncached | 146.64 | 0.794 | 2.390 | 915 |
| INT8画像・GPUテキストuncached | 104.83 | 1.455 | 2.866 | 916 |

CPUはAMD EPYC 7763、推論4スレッド。CPU側はFP32演算で、特徴をFP16でGPUへ戻す。
全構成で検出数1/4/6/4/0は一致。CPU版の平均IoUはFP16画像0.999190、INT8画像0.997419。
INT8 CPU版のscore最大差0.00830、box最大差0.511画素。
キャッシュ済み語句は速度を保ち、GPU常駐重みを減らせる。新規語句にはCPU時間がかかる。
`offload_text_encoder(processor)`を公開コードに追加し、Round 42で公開版を測定する。
テキスト側のINT8とGPUテキストcompileは併用不可として、適用時に明示的に拒否する。

## Round 42：CPUテキストの公開API

FP16画像版は113.39ms・allocated 1.202GiB・NVML 2.370GiB、289画素変化・平均IoU 0.999190。
INT8画像cachedは84.94ms・0.793GiB・2.333GiB、915画素・平均IoU 0.997419。
INT8画像uncachedは187.33ms・0.793GiB・2.333GiBで、同じ出力差だった。
全5条件の検出数1/4/6/4/0は一致した。単独パッチも公開関数を含む形で更新済み。

CPU uncachedは試作146.64msから公開測定187.33msへ変動し、後者の9回は147〜211msだった。
キャッシュ済みの画像推論は安定して速いが、CPUで語句を毎回処理する時間は変動する。
追加環境やCPU固定化の設定は導入せず、実測値とばらつきをそのまま残す。
API smokeは成功。状態A→B→Aの再利用、直接modelの元の辞書、box指定、空しきい値、
packedリセット、固定語句への切替と未登録語句の拒否、autocast復元を確認した。
非連続strideのbitpackも差0。CPU上のテキスト重みは約1.32GiBで、固定語句化時に
GPU割当がさらに減ることはない（記録値0 bytes）。

## Round 43の候補：重み量子化による平均ずれの補正

Sciteで見つけた[Post-Training Quantization for Vision Transformer](https://arxiv.org/abs/2106.14156)の
バイアス補正という方向を参考に、画像較正を使わない簡易案を用意した。論文の再現ではない。
LayerNorm直後の入力平均をbetaと近似し、`(元の重み−復元したINT8重み) @ 入力平均`を
Linearのbiasへ足す。MLP後段には正規分布を仮定したGELU平均、Attention出力射影には
V射影の平均を使う拡張も比べる。補正は初期化時だけで、推論の演算は増やさない。
Round 24で大きなチャネル削減の補正は失敗しているため、今回は小さい重み丸め誤差に限定する。

Round 35のMSE追加調整＋非対称GELUは、速度を保ちscore/boxの最大差も小さかった。
`optimize_weight_scales=True`として公開APIの任意設定を準備し、Round 29の後に
公開コードへ反映する。Round 44では通常版・最適化版・CPUテキスト併用版を比べる。
選択する重み以外の推論演算は増えず、既定設定のabsmax量子化も維持する。

## Round 35：重みの二乗誤差でINT8スケールを選ぶ

| 構成 | ms | 平均IoU vs stock | 変化画素 | score最大差 | box最大差 px |
|---|---:|---:|---:|---:|---:|
| 対称GELU基準 | 85.75 | 0.997421 | 916 | 0.00781 | 0.495 |
| スケール探索 | 85.54 | 0.997181 | 905 | 0.00586 | 0.550 |
| 探索＋2回調整 | 85.60 | 0.997130 | 907 | 0.00732 | 0.565 |
| 探索＋非対称GELU | 85.19 | 0.997787 | 752 | 0.00732 | 0.497 |
| 探索＋2回調整＋非対称GELU | 84.88 | 0.998074 | 697 | 0.00391 | 0.566 |
| 重みのみINT8＋探索 | 116.46 | 0.998794 | 404 | 0.00391 | 0.538 |
| 重みのみINT8＋探索＋2回調整 | 116.22 | 0.998694 | 416 | 0.00391 | 0.550 |

全7構成で検出数1/4/6/4/0は一致。allocated約1.455GiB、NVML 2.812〜2.851GiB。
重みの二乗誤差は探索で約2.45%、2回調整で約2.60%減った。
初期化は探索を含め約0.46〜0.51秒、2回調整込み約1.09〜1.11秒だった。
重みの誤差を減らしてもマスクIoUが改善しない組み合わせがある。
追加調整＋非対称GELUは、従来の非対称公開版（735画素、score差0.01172、box差1.436画素）
からこれらの出力差が改善したため、任意設定の公開候補としてRound 44へ進める。

## Round 45の候補：小さい改善を組み合わせる

CPUテキスト＋重みスケール調整＋非対称GELUを基準に、fc1融合、decoder FFNのhalf化、
ViT残差のhalf保持、画像正規化、neckのコンパイルを組み合わせる。
個別の1〜2msが積み重なるとは限らないため、単純な合算ではなく全体で測る。
half保持や正規化による出力差も含め、検出数と各差を基準と比較する。

## Round 29：マスクheadの射影を再結合

FP16基準115.93ms・285画素変化に対し、静的再結合116.52ms・280画素、
動的再結合116.02ms・284画素。allocated 1.862GiB・NVML 3.095GiBは同じだった。
INT8基準87.30ms・916画素に対し、静的85.67ms、動的85.71msで、両方919画素。
allocatedは約1.455GiB。検出数とscore/box最大差は各基準と同じ。
INT8で約1.6ms短縮した候補として残し、既定パッチには追加せず組み合わせ比較へ回す。

## Round 44：重みスケール調整の公開API

非対称INT8の基準85.67ms・735画素・平均IoU 0.998051に対し、
`optimize_weight_scales=True`は86.91ms・697画素・0.998074だった。
score最大差0.01172→0.00391、box最大差1.436→0.566画素。
CPUテキストも併用すると86.30ms・allocated 0.793GiB・NVML 2.368GiB、
698画素・IoU 0.998065・score最大差0.00391・box最大差0.574画素だった。
全3構成で検出数1/4/6/4/0は一致。非対称INT8と組み合わせる任意設定として採用した。
初期化時の処理だけなので、直前に成功したCPU混在のAPI smokeは繰り返さず、
公開コードからの全5条件比較を今回の確認とする。単独パッチも更新済み。

## Round 46の候補：window Attentionの範囲とpadding

通常の1008入力は72×72 tokenで、window 24を使う。18や12へ小さくすれば
MLPとglobal Attentionを維持しながら、local Attentionの組み合わせ数を減らせる。
局所的な情報共有の範囲が変わるため近似候補として比較する。
RoPEを新しいwindowに補間する版と、元の座標間隔を保つ版を用意する。
896入力は64×64 tokenでwindow 24にpaddingが必要なので、16または32で割り切れる
構成も比較する。global Attentionの設定は変えない。

672入力は48×48 tokenでwindow 24に割り切れ、Round 30では40.68msだった。
この結果を受け、window比較に840/window20、784/window28も追加する。
前者は60×60、後者は56×56 tokenでpaddingを省ける。
後者は通常の24より広いwindowで、局所Attentionの範囲を狭めずに余分な計算を減らす候補。

## Round 30：現在のINT8融合構成で解像度を比較

| 解像度 | ms | 平均IoU vs stock | 変化画素 | 最小IoU | box最大差 px |
|---|---:|---:|---:|---:|---:|
| 1008基準 | 86.78 | 0.997421 | 916 | 0.993191 | 0.49 |
| 896 | 76.04 | 0.973894 | 8644 | 0.899851 | 4.32 |
| 840 | 70.47 | 0.972801 | 8098 | 0.875769 | 7.89 |
| 784 | 63.84 | 0.970820 | 9336 | 0.885173 | 8.63 |
| 672 | 40.68 | 0.953963 | 13340 | 0.782442 | 17.62 |
| 560 | 35.23 | 0.932052 | 18663 | 0.614473 | 26.19 |

検出数は全設定1/4/6/4/0。速度優先の選択肢として文書の測定値を更新し、既定1008は維持する。
特に672はwindow 24に割り切れる。次はRound 46を先に実行し、784や840を維持したまま
windowを調整してpaddingを減らす案も比較する。これはAttentionの範囲も変える近似候補。

## Round 47の候補：INT8入力の裾を小さく切り詰める

各tokenの最大絶対値をそのまま量子化幅に使うと、少数の大きな値に刻み幅が支配される。
入力の最大値を0.995 / 0.99 / 0.98倍にし、範囲外を飽和させる候補を用意する。
GELU後は既存の非対称量子化の上端だけを0.995 / 0.99倍にする。
追加の画像較正は行わず、速度・5条件の出力差・検出数を比較する。
重みスケール最適化＋非対称GELUを基準とし、倍率1の置換確認も含める。
Sciteで参照した[FQ-ViT](https://arxiv.org/abs/2111.13824)にもViTでのclippingの扱いがあるが、
ここではpercentile較正や論文の量子化方式は使わず、各token最大値の倍率だけを試す。

## Round 48の候補：CPUテキストMLPの動的INT8

CPUオフロードでは新規語句の待ち時間が増えたため、CPUのMLPだけをPyTorchの動的INT8へ
置き換える。重みのper-tensor / per-channelと、それぞれの語句cacheあり・なしを比較する。
Attention・埋め込み・出力resizerはCPU FP32を維持する。GPU用INT8とは別の試作で、
公開APIのCPUテキストとGPUテキストINT8を併用しない制約は変えない。
CPUバックエンドと重み容量も記録し、画像側は公開の非対称INT8＋重み調整で固定する。

## Round 46：Attentionの窓とpaddingの比較

| 構成 | ms | allocated GiB | 平均IoU vs stock | 変化画素 | 検出数 |
|---|---:|---:|---:|---:|---:|
| r46_fused_attention_control | 85.40 | 1.456 | 0.997421 | 916 | 15 |
| window18_interpolated | 82.78 | 1.508 | 0.979056 | 5709 | 15 |
| window18_unit_rope | 83.07 | 1.455 | 0.979903 | 5362 | 15 |
| window12_interpolated | 83.25 | 1.507 | 0.980885 | 6677 | 8 |
| r46_resolution896_control | 76.38 | 1.420 | 0.973894 | 8644 | 15 |
| resolution896_window16 | 68.86 | 1.404 | 0.953107 | 10936 | 15 |
| resolution896_window32 | 74.18 | 1.409 | 0.965688 | 11066 | 15 |
| resolution840_window20 | 62.39 | 1.383 | 0.966463 | 8570 | 15 |
| resolution784_window28 | 57.91 | 1.400 | 0.970062 | 10150 | 15 |

通常1008で窓18にすると約2.6ms短縮するが、score最大差が0.166まで増える。窓12は
childを検出できず1/4/0/3/0となる。IoUは対応マスクだけの集計なので、この高めの値を
良好な結果とは扱わない。窓16の896は最小IoU 0.604と形状差が大きい。
784/window28は標準窓24の63.84ms→57.91msで、平均IoU 0.970820→0.970062、
最小IoU 0.88517→0.90596、box最大差8.63→7.99画素だった。検出数1/4/6/4/0を維持。
低解像度の速度候補として保存し、公開パッチの標準窓24は変更しない。
次は窓の範囲を維持したまま、padding領域のQKV・出力射影だけを省けるかを調べる。

## Round 49の候補：padding位置の射影を省く

window Attentionの窓24・RoPE・Key/Valueの数を維持し、QKVをwindow分割の前、
出力射影をwindow結合の後へ移す。低解像度で生じるpadding位置のLinearを省く狙い。
padding位置のQKVはゼロではなくQKV biasを補う。実tokenは元と同じ窓でAttentionを行う。
1008・896・784の基準と比較し、784ではQKV移動のみ・出力射影移動のみも分けて測る。
演算順とGEMM形状が変わるため、実際の出力差は計測して判断する。

## Round 50の候補：CPUテキストのスレッド数

新規語句を毎回処理するCPU INT8版について、既定4スレッドと1 / 2 / 8を比較する。
CPU初期化は引き続き1スレッド、推論時だけ変更する。CPUの設定は公開パッチからは変更せず、
このコンテナでの選択肢として時間と出力差を記録する。

## Round 48：CPUテキストMLPの動的INT8

| CPUテキスト | cache | ms | 平均IoU vs stock | 変化画素 |
|---|---|---:|---:|---:|
| FP32 | あり | 85.02 | 0.998065 | 698 |
| FP32 | なし | 155.84 | 0.998065 | 698 |
| per_tensor | あり | 85.78 | 0.997790 | 922 |
| per_tensor | なし | 107.00 | 0.997790 | 922 |
| per_channel | あり | 86.20 | 0.997815 | 888 |
| per_channel | なし | 118.68 | 0.997815 | 888 |

全設定で1/4/6/4/0。CUDA allocated 0.793〜0.795GiB、NVML 2.337〜2.537GiB。
CPUのx86 backend・4スレッドで48個のMLP Linearを動的INT8化した。
CPUの通常parameter＋展開した量子化重み/biasの容量は1,414,898,688→810,918,912 bytes。
プロセスRSSやpacked表現の管理領域を測った値ではない。
per-tensorは新規語句155.84→107.00ms。score最大差0.02051、box最大差0.523画素。
per-channelは118.68ms。score最大差0.02441、box最大差0.525画素だった。
マスク差の差は小さく、今回速かったper-tensorを任意の公開設定に選ぶ。
`offload_text_encoder(processor, int8_mlp=True)`として組み込み、Round 51で公開版を確認する。
既定のCPU FP32と、GPU用のtext=Trueは変更しない。

## Round 52の候補：CPUテキストのpadding tokenを省く

EOSまでのtokenを8の倍数または実長へ切り詰め、CPUテキストencoderを処理する。
元の32 tokenへゼロpaddingしてから返すため、GPU側のTensor形状とtoken maskは維持する。
実tokenは因果Attentionで後方paddingを参照しない。CPU FP32と動的INT8を別々に測る。
INT8では入力全体の量子化範囲も変わるため、マスク差を実測して判断する。
Round 7のtokenizer全体を短縮する案と違い、後段のGPU形状は変えない。

## Round 41：fc1融合のタイルと非対称GELU

| 構成 | ms | allocated GiB | NVML GiB | 変化画素 |
|---|---:|---:|---:|---:|
| r41_fused_attention_control | 86.78 | 1.455 | 2.831 | 916 |
| epilogue_fc1_large | 83.21 | 1.457 | 2.938 | 916 |
| epilogue_fc1_wide | 83.92 | 1.456 | 2.919 | 916 |
| r41_asymmetric_control | 85.75 | 1.455 | 2.831 | 735 |
| epilogue_fc1_asymmetric | 83.14 | 1.456 | 2.890 | 735 |
| epilogue_fc1_large_asymmetric | 83.64 | 1.455 | 2.901 | 735 |

対称版の大タイルは86.77→83.21ms、非対称版の小タイルは85.75→83.14ms。
4つの融合候補は、それぞれ対応する基準と全5条件の比較辞書が一致した。
マスク指標だけでなくscore・box・probability差も同じで、検出数は1/4/6/4/0。
新しいタイルの初回コンパイルには時間がかかる。大タイル対称版は今回約70秒だった。
GPUメモリはわずかに増えるため、Round 45で他の改善と組み合わせてから採用を判断する。

## sm75のオフラインcompileとRound 53

Triton 3.5.0でCUDA_VISIBLE_DEVICESを空にし、sm75向けのcompileだけを確認した。
独自INT8 GEMMの64×128 / 128×128タイルは、INT8 dotのloweringで`arith.extf`型エラーになる。
3090での速度・出力の測定は有効だが、この候補をTuring用の公開パッチへは入れない。
実験helperはsm80未満で理由を示して停止する。
Round 45の融合GEMMを含む組合せはAmpere向け候補として保留し、Round 53では
既存のtorch._int_mmを使い、decoder FP16・neck compile・正規化・head射影再結合・
残差half保持の組合せを測る。オフラインcompileはTuring実機の動作や速度の測定ではない。

## Round 51：CPUテキストINT8の公開版

公開APIからcacheあり84.79ms・NVML 2.368GiB、cacheなし100.80ms・NVML 2.345GiB。
allocatedはともに0.793GiBで、試作per-tensor版と5条件の比較辞書が一致した。
検出数1/4/6/4/0、平均IoU 0.997790、922画素、score差0.02051、box差0.523画素。
CPU FP32・新規語句155.84msに対し約35%短縮した。
API smokeは画像A→B→A、box、しきい値変更、空出力、固定語句への切替、直接model呼出し、
autocast復元を通過。48個のCPU INT8 Linearも確認し、非連続strideのbitpack差は0だった。
`int8_mlp=True`を任意設定として採用し、単独パッチも更新した。

sm75向けの追加compile確認では、公開の量子化・GELU/非対称GELU・FP16/FP32 mask resize+pack・
pack・unpackの7カーネルが通過した。独自INT8 GEMMの2タイルは失敗したままで、
RTX 3090の候補として保存する。sm75実機での実行確認は行っていない。

## Round 54の候補：4bit重みの非等間隔コード

Gaussianの分位点から対称な15段階（ゼロを含む）を作り、小さい重みに段階を多く割り当てる。
グループごとの最大絶対値と4bitコードを保存し、計算時はFP16重みに復元して通常のLinearを使う。
等間隔4bitのRound 38を先に測り、group 16 / 32 / 64 / 128の非等間隔版も比較する。
独自の簡易コードブックであり、特定のNF4実装や論文を再現したものではない。

Round 53bでは、5つの変更から入力正規化とneck compileを省き、残差FP16・decoder FFN FP16・
head再結合の3つだけで同程度の改善を得られるかを比べる。Round 55で画像の公開版、
Round 56でCPUテキストpadding省略の公開版を確認する。いずれも任意設定の候補。

## Round 53：標準カーネルによる組合せ

| 構成 | ms | allocated GiB | 平均IoU vs stock | 変化画素 | score最大差 | box最大差 px |
|---|---:|---:|---:|---:|---:|---:|
| r53_optimized_cpu_control | 86.68 | 0.794 | 0.998065 | 698 | 0.00391 | 0.574 |
| combined_standard_decoder | 86.00 | 0.781 | 0.998065 | 698 | 0.00391 | 0.580 |
| combined_standard_neck_normalize | 84.66 | 0.837 | 0.997977 | 711 | 0.00928 | 1.999 |
| combined_standard_static_head | 84.59 | 0.782 | 0.997978 | 710 | 0.00928 | 1.999 |
| combined_standard_half_residual | 83.78 | 0.782 | 0.998220 | 671 | 0.00342 | 0.629 |

すべて検出数1/4/6/4/0。5つを組み合わせた最後の候補は、基準86.68msから83.78msへ短縮し、
変化画素698→671、平均IoU 0.998065→0.998220となった。NVML 2.319GiB。
半精度残差を加える前のneck/正規化の組合せはbox差が約2画素へ増えていた。
効果は単純な足し算ではない。任意の追加パッチとして準備し、53bの簡素化と公開版55へ進む。

## Round 52：CPUテキストのpadding省略

| 構成 | ms | allocated GiB | 平均IoU vs stock | 変化画素 | score最大差 | box最大差 px |
|---|---:|---:|---:|---:|---:|---:|
| r52_cpu_fp32_uncached_control | 163.71 | 0.794 | 0.998065 | 698 | 0.00391 | 0.574 |
| cpu_fp32_trim8 | 91.74 | 0.794 | 0.998065 | 699 | 0.00391 | 0.571 |
| cpu_fp32_trim1 | 87.13 | 0.795 | 0.998073 | 698 | 0.00391 | 0.567 |
| r52_cpu_int8_uncached_control | 95.97 | 0.793 | 0.997790 | 922 | 0.02051 | 0.523 |
| cpu_int8_trim8 | 86.08 | 0.795 | 0.997855 | 970 | 0.01221 | 0.560 |
| cpu_int8_trim1 | 87.45 | 0.794 | 0.997809 | 904 | 0.01855 | 0.523 |

すべて新規語句を毎回処理し、検出数1/4/6/4/0は一致。FP32でEOS直後まで切ると
163.71→87.13ms、変化画素698で同数だった。比較辞書全体の完全一致ではなく、
score・box・IoUには小さい丸め差がある。INT8の実長版は87.45ms・904画素だった。
短い語句ではFP32とINT8の速度差は小さくなり、CPU重み容量と出力差で選べる。
FP32・INT8で同じ実長切り詰めを`trim_padding=True`の公開候補にし、56で確認する。
長い語句では省けるpaddingが少ないため、この短いtruck promptほどの短縮は期待しない。

## Round 53b：3変更へ絞る

残差FP16・decoder FFN FP16・head再結合だけで82.69ms・allocated 0.782GiB・NVML 2.362GiB。
5変更の83.78msより速く、正規化kernelとneck全体compileを追加せずに済んだ。
検出数1/4/6/4/0、673画素変化、平均IoU 0.997999、score最大差0.01074、box最大差0.556画素。
5変更はIoU 0.998220・score差0.00342だったため、指標のすべてで改善したわけではない。
速度と実装量を優先し、この3変更を`apply_image_refinements`の公開候補に選ぶ。
入力前処理は元のまま。新しい独自CUDA kernelも追加しない。公開版のRound 55とAPI smokeで確認する。

## Round 56：CPU padding省略の公開版

`trim_padding=True`の公開版で、CPU FP32は87.48ms、CPU INT8併用は87.64ms。
両方とも5条件の比較辞書が試作の実長版と一致した。FP32は698画素・IoU 0.998073、
INT8は904画素・IoU 0.997809。検出数1/4/6/4/0。
画像側の公開追加パッチを測定後、両方を有効にしたAPI smokeへ進む。

## Round 55：画像追加パッチの公開版

| 構成 | ms | allocated GiB | NVML GiB | 平均IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|---:|
| accepted_refined_int8_cpu | 83.08 | 0.837 | 2.382 | 0.997999 | 673 |
| accepted_refined_int8_gpu | 83.08 | 1.500 | 2.813 | 0.997987 | 675 |
| refined_fp16_cpu | 112.09 | 1.243 | 2.430 | 0.999144 | 293 |

すべて検出数1/4/6/4/0。INT8画像＋CPUテキストの全5条件の比較辞書は、3変更の試作と一致。
ただし公開版のピークallocatedは試作0.782GiBに対し0.837GiBだった。公開値を優先して記録する。
CPU/GPUテキストのINT8画像版はscore最大差0.01074、box最大差0.556/0.576画素。
追加変更は任意設定とし、素のFP16画像パッチやINT8の既定は変更しない。
Round 57でCPU padding省略との組合せを実測し、公開値を揃える。

## Round 57：画像追加パッチとCPU padding省略の併用

新規語句を毎回処理し、CPU FP32は85.37ms・allocated 0.782GiB・NVML 2.573GiB。
674画素・IoU 0.997997・score最大差0.01074・box最大差0.571画素だった。
CPU INT8も使う版は84.38ms・allocated 0.782GiB・NVML 2.321GiB。
822画素・IoU 0.997737・score最大差0.02783・box最大差0.552画素。
両方1/4/6/4/0。CPU INT8の速度差は小さく、FP32の方が出力差が小さかった。
CPU重み容量を減らしたい場合にINT8を選べる形にする。

画像追加パッチ・CPU INT8・padding省略・画像の非対称INT8＋重み調整を全て使った
API smokeが通過した。A→B→Aの状態再利用、直接modelの出力辞書、box、空出力、
固定語句への切替、未知語句拒否、autocast復元を確認。非連続strideのbitpack差は0。
CPU上の量子化Linearは48個。文書・比較図・単独パッチへ反映する。

## Round 50：CPUテキストINT8のスレッド数

| スレッド数 | ms | 変化画素 | 平均IoU vs stock |
|---|---:|---:|---:|
| 4 | 105.17 | 922 | 0.997790 |
| 1 | 224.40 | 922 | 0.997790 |
| 2 | 171.54 | 922 | 0.997790 |
| 8 | 87.90 | 922 | 0.997790 |

AMD EPYC 7763のこのコンテナでは8スレッドが87.90msで最速。4スレッドは105.17msだった。
この比較ではpadding省略を使わず、語句cacheも無効。全設定で1/4/6/4/0、922画素変化。
score/box/マスク指標は一致し、CPUで集計するprobability MAEの末尾にはごく小さい差がある。
公開パッチはプロセス全体のスレッド設定を変更しない。初期化後にtorch.set_num_threads(8)を
選ぶ余地があるが、他のCPUや他の処理との併用では個別に比べる。

## Round 58の候補：追加パッチをFlashAttentionなしで使う

画像追加パッチ・CPUテキストpadding省略・重み調整を組み合わせ、Attentionを
efficient backendに固定する。FP16画像、INT8画像、CPUテキストもINT8の3通りを比較する。
3090上での経路確認であり、Turing実機の速度推定には使わない。

## Round 49：padding位置の射影省略

| 入力・構成 | ms | allocated GiB | 平均IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|
| 1008 control | 86.57 | 1.454 | 0.997421 | 916 |
| 1008 QKV＋出力射影 | 85.08 | 1.456 | 0.997265 | 920 |
| 896 control | 75.93 | 1.420 | 0.973894 | 8644 |
| 896 QKV＋出力射影 | 72.72 | 1.405 | 0.974120 | 8534 |
| 784 control | 64.48 | 1.402 | 0.970820 | 9336 |
| 784 QKV＋出力射影 | 57.87 | 1.365 | 0.971077 | 9297 |
| 784 QKVのみ | 59.00 | 1.367 | 0.971077 | 9297 |
| 784 出力射影のみ | 64.44 | 1.368 | 0.970820 | 9336 |

すべて1/4/6/4/0。window 24を維持したまま784の余白の射影を省き、約10%短縮した。
主な効果はQKV側。896では約4%短縮。1008は72×72がwindow 24で割り切れるため、
padding省略による計算量の削減はない。1008の差は演算配置・丸め・実行変動を含む。
1008のscore最大差0.01270、box最大差1.139画素は基準より増えているため既定にはしない。
低解像度による大きな出力差は残る。784のIoU 0.971077は1008 stockとの比較。

Round 59では896/784で、重み調整・CPUテキストpadding省略・画像追加パッチと併用する。
試作のBlock.forward置換でも公開追加パッチのFP16出力を維持するようにした。
FP16画像の784でも比較し、INT8射影だけに有効な変更か調べる。

## SAM 3.1の短い動画比較の準備

facebook/sam3.1のsam3.1_multiplex.ptを取得（revision daa63191845a41281374e725f4c9e51c7a824460）。
既存Pythonで、同梱0001動画の先頭24フレームとperson promptを使うvideo_probe.pyを用意。
FA3・compile・TF32を無効、grounding batch 1、動画フレームCPU保存を両構成で揃える。
元のモデルと、BF16固定箇所をFP16へ置き換える試作を比較する。画像専用のneck削除は使わない。
add_prompt＋順方向追跡を計測し、フレーム読み込みは除く。動画の結果はまだ未測定。

## Round 60の候補：別画像1枚でheadの寄与を見る

Round 36の重みだけによるhead選択に加え、評価3画像とは別の同梱動画0001/0.jpgを1枚使い、
Attention出力を128 tokenまで抽出してheadごとの出力寄与を測る。二乗平均の小さいheadを削る案と、
平均寄与をbiasで補った上で分散の小さいheadを削る案を比較する。global・window・全層を分ける。
追加学習・勾配計算はしない。初期化時の1回の測定で、推論ごとの動的選別はしない。
Sciteで[Global Vision Transformer Pruning with Hessian-Aware Saliency](https://arxiv.org/abs/2110.04869)
を調べたが、今回の寄与二乗・分散は独自の簡易指標でありHessian法の再現ではない。

## Round 58：追加パッチをFlashAttentionなしで使用

| 構成 | ms | allocated GiB | NVML GiB | 平均IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|---:|
| r58_refined_trimmed_auto_control | 85.312 | 0.7822 | 2.2979 | 0.99799737 | 674 |
| refined_trimmed_cpu_int8_efficient | 90.013 | 0.7822 | 2.3506 | 0.99788641 | 814 |
| refined_trimmed_efficient | 90.605 | 0.7832 | 2.5491 | 0.99811614 | 689 |
| refined_fp16_trimmed_efficient | 119.423 | 1.1904 | 2.3408 | 0.99917998 | 285 |

すべて1/4/6/4/0。新規語句を毎回CPUで処理し、padding省略を使った。
FlashAttentionなしのINT8画像＋CPU FP32は90.60ms・689画素・IoU 0.998116、
CPU INT8も使うと90.01ms・814画素・IoU 0.997886だった。通常経路の基準は85.31ms。
FP16画像は119.42ms・285画素。これも3090上のAttention経路比較であり、Turing実機の測定ではない。

初回は実験用backend切替がCPUテキストにもCUDA専用SDPAを強制して失敗した。
2件のfailure JSONを保存し、attention_backend.pyでCUDA queryだけを切り替えるよう修正した。
通常経路の基準も再実行したため、このラウンドは4候補・7試行。

## 動画 Round 1：FP16化

24フレーム・person・batch 1・3回中央値。基準261.00ms/frame・allocated 6.726GiB・NVML 7.919GiB。
FP16は268.50ms/frame・3.902GiB・5.030GiB、efficient固定は284.79ms/frame・3.899GiB・5.110GiB。
全フレーム4人、96件のID対応を維持。平均IoU 0.998608 / 0.998610、変化画素4282 / 4270。
初回はinit_stateの未対応offload_state_to_cpu引数で停止。両構成へ同じAPI補正を入れて比較した。
性能と設定の詳細はdocs/TURING_VIDEO_EXPERIMENTS.md。画像向け公開パッチへは混ぜていない。

## 動画 Round 2：INT8・ViT compile・CPUテキスト

| 構成 | ms/frame | allocated GiB | NVML GiB | IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|---:|
| [video_fp16_int8_b1](results/video_fp16_int8_b1.json) | 265.94 | 3.496 | 4.632 | 0.998099 | 8151 |
| [video_fp16_int8_compile_b1](results/video_fp16_int8_compile_b1.json) | 195.40 | 3.509 | 5.003 | 0.998080 | 8313 |
| [video_fp16_int8_cpu_compile_b1](results/video_fp16_int8_cpu_compile_b1.json) | 226.79 | 2.849 | 4.485 | 0.998076 | 8319 |
| [video_fp16_int8_cpu_compile_efficient_b1](results/video_fp16_int8_cpu_compile_efficient_b1.json) | 226.65 | 2.838 | 4.505 | 0.997858 | 7567 |

全構成で24フレーム×4人・96件の人物ID対応を維持。INT8だけでは速度差が小さかったが、
ViT compileで265.94→195.40ms/frameへ短縮した。基準261.00ms/frameに対し約25%短縮。
CPUテキストはallocated 2.849GiBに下がる一方226.79ms/frameへ遅くなった。
CPUテキスト＋efficientでも226.65ms/frame・2.838GiB。
CPUの短い語句でpadding省略を使い、この待ち時間を減らす候補を追加する。
画像の公開パッチと同じ非対称INT8・重み調整を使うが、動画では別途そのまま同じ数値になるとは限らない。

## 同じ画像への複数語句の候補

truck画像を1回だけencodeし、truck / wheel / vehicle / elephantの4語句を処理する。
stockと公開パッチの逐次処理に加え、同じ画像特徴を参照する2語句・4語句batchを比較する。
画像batchは1のまま。FindStageの画像IDを繰り返し、テキストIDを分ける独立の試作。
画像前処理＋4語句＋マスク出力まで計測し、2 warmup・9回中央値。batch化のVRAM増加も比較する。
CPUテキストは短い語句のpadding省略を使う。公開APIには未採用で、実行結果はまだない。

## Round 36：重みから選んだheadの削減

| 構成 | ms | allocated GiB | 平均IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|
| prune_heads_all_14_comp | 82.181 | 1.4257 | 0.97205003 | 11178 |
| prune_heads_global_14_comp | 83.427 | 1.5077 | 0.99269399 | 2901 |
| prune_heads_all_15_comp | 84.516 | 1.4865 | 0.98760838 | 4468 |
| prune_heads_global_15 | 84.521 | 1.4513 | 0.99554614 | 1667 |
| prune_heads_window_15_comp | 84.524 | 1.4338 | 0.98692144 | 4590 |
| prune_heads_global_15_comp | 85.251 | 1.4513 | 0.99549641 | 1655 |
| r36_fused_attention_control | 85.513 | 1.4562 | 0.99742137 | 916 |

全構成で1/4/6/4/0を維持。global 1 head削減は約1msの短縮に対して916→1667画素へ差が増えた。
平均寄与のbias補正でも85.25ms・1655画素で改善は小さい。window側・全層へ広げると差が増える。
全層2 head削減は82.18ms・11178画素、NVMLも基準2.839GiBに対し3.024GiB。
この重みだけのhead選択は採用しない。別画像1枚の出力寄与で選ぶRound 60は独立に比較する。

## Round 61の候補：一部だけactivationをFP16に戻す

全層の重みは調整済みINT8のまま保存し、一部のLinearだけ重みを一時FP16へ復元して計算する。
QKVのglobal / 先頭8 / 全層、出力射影の全層、MLPの先頭4 / 末尾4を比較する。
全体をweight-onlyにするより小さい速度負担で、activation量子化による差を減らせるかを見る。
選択したMLPではGELU再量子化の融合も外す。選択以外の層・画像追加パッチ・CPU padding省略は維持する。

## 動画 Round 2b：CPUテキストpadding省略

| 構成 | ms/frame | allocated GiB | NVML GiB | IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|---:|
| [video_fp16_int8_cpu_trim_compile_b1](results/video_fp16_int8_cpu_trim_compile_b1.json) | 198.79 | 2.838 | 4.485 | 0.998079 | 8305 |
| [video_fp16_int8_cpu_trim_compile_efficient_b1](results/video_fp16_int8_cpu_trim_compile_efficient_b1.json) | 221.11 | 2.838 | 4.485 | 0.997859 | 7565 |

全フレーム4人・96件のID対応を維持。通常AttentionのCPUテキスト版はpadding省略で
226.79→198.79ms/frameへ短縮し、allocated 2.838GiB。score最大差0.00771。
テキスト計算は各runで1回、3 captionをまとめていた。省略後のCPU計算は約0.10〜0.14秒/run。
画像で採用した末尾padding省略が動画でも待ち時間の軽減に有効だった。

## Round 62の候補：4bitのscaleとoffsetも小さくする

group 16でFP32 scale/offsetを両方保存すると、そのメタデータだけで元の重み1要素あたり4bitになる。
scaleをFP16にし、対称量子化のoffsetは定数−8から計算、Gaussianの未使用offsetは保存しない候補を追加。
group 16 / 32の等間隔・Gaussianを比べる。計算前にscaleをFP32へ戻し、重みの復元はFP32→FP16。
重みの量子化後にscaleを丸めるため出力はわずかに変わり得る。既存の非compact構成は維持する。

## Round 38：4bit重み

| 構成 | ms | allocated GiB | NVML GiB | 平均IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|---:|
| r38_fp16_control | 113.68 | 1.8618 | 3.0947 | 0.99920526 | 285 |
| weight_int4_mlp_group32 | 114.384 | 1.5023 | 3.1494 | 0.99304115 | 1975 |
| weight_int4_group32_asym | 115.034 | 1.3936 | 2.7432 | 0.99246047 | 2206 |
| weight_int4_group128 | 116.186 | 1.2635 | 2.6455 | 0.98979964 | 3212 |
| weight_int4_group16 | 116.904 | 1.4728 | 2.8232 | 0.99409251 | 2150 |
| weight_int4_group64 | 116.942 | 1.2905 | 2.6807 | 0.99091942 | 3128 |
| r38_weight_int8_control | 117.36 | 1.4556 | 2.8213 | 0.99876291 | 389 |
| weight_int4_group32 | 117.817 | 1.3948 | 2.7861 | 0.99163274 | 2756 |

全構成で1/4/6/4/0。重みは4bitで保持するが、計算時はFP16へ復元するため速度はFP16並み。
group 16はscale/offsetがFP32のため保存量の削減が小さく、allocated 1.473GiBで重みINT8の1.456GiBを超えた。
group 128は1.263GiBまで減るが3212画素変化。group 32の非対称版は対称版より差が少なく、2206画素だった。
MLPだけの4bitは114.38ms・1.502GiB・1975画素で、Attentionも圧縮した場合よりメモリが増える。
次にRound 62のcompact metadataを優先する。小さいCPU配列でscaleの半精度保存・offset省略の有限値を確認し、
保存量も減ることを確認した。Gaussianの小配列は出力差0、等間隔は最大0.0009766だった。実モデルの品質は別に測る。

Round 54はcompact metadata実装に合わせて54bへ更新する。Gaussian group 16/32はRound 62で比較するため、
54bはcompact Gaussian group 64/128の2構成だけを追加する。未実行の古い54.jsonは候補履歴として残す。

## Round 62：compactな4bitメタデータ

| 構成 | ms | allocated GiB | NVML GiB | 平均IoU vs stock | 変化画素 | score最大差 | box最大差 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| int4_compact_group32 | 116.257 | 1.2614 | 2.6201 | 0.99162588 | 2761 | 0.0302734375 | 1.26904296875 |
| r62_int4_group32_control | 116.438 | 1.3414 | 2.7217 | 0.99163274 | 2756 | 0.0302734375 | 1.2672119140625 |
| int4_compact_group16_asym | 116.647 | 1.4529 | 2.835 | 0.9944606 | 1748 | 0.01416015625 | 0.64886474609375 |
| int4_compact_group32_gaussian | 116.806 | 1.2674 | 2.6553 | 0.99329326 | 1995 | 0.015625 | 1.07763671875 |
| int4_compact_group16_gaussian | 116.853 | 1.2906 | 2.6846 | 0.99389652 | 1943 | 0.0205078125 | 1.51263427734375 |
| int4_compact_group32_asym | 116.924 | 1.3996 | 2.751 | 0.99247757 | 2204 | 0.02001953125 | 1.371826171875 |
| int4_compact_group16 | 117.244 | 1.3478 | 2.7256 | 0.99410651 | 2149 | 0.0302734375 | 1.5836181640625 |

すべて1/4/6/4/0。対称group 32は今回の基準1.341→1.261GiB、2756→2761画素。
group 16の等間隔版は1.348GiB・2149画素、Gaussianは1.291GiB・1943画素だった。
両者のcompactメタデータ容量はほぼ同じなので、このピーク差を保存形式の差だけには帰属しない。
Gaussian group 32は1.267GiB・1995画素・IoU 0.993293。非対称group 16は1.453GiBだが
1748画素・IoU 0.994461・score最大差0.01416・box最大差0.649画素と、この4bit比較では差が少なめだった。
速度はいずれも約116〜117ms。4bitの既定採用はせず、GPUメモリを減らす候補として保持する。

## Round 63の候補：4bitをCPUテキスト・画像追加パッチと併用

compact Gaussian group 16/32、非対称group 16を、CPUテキストpadding省略・画像追加パッチと組み合わせる。
FP16画像の同じ構成を基準にする。Gaussian group 32はefficient Attentionでも比較する。
4bitの行列積はFP16の通常Linearであり、動的INT8のtorch._int_mmは使わない。
3090上の測定であり、GTX16やRTX20の速度を代用するものではない。

## Round 64の候補：1008から少しだけ解像度を下げる

896/784より小さい変更として980/952/924を試す。window 24は維持する。
Round 59の組み合わせでは余白の射影省略による速度改善が出なかったため、ここでは通常の射影を使う。
画像追加パッチ・画像の非対称INT8＋重み調整・CPUテキストpadding省略を使い、
1008の同じ構成と比較する。解像度による速度差とマスクの差の中間点を探す。

## Round 54b：compact Gaussianの大きいgroup幅

| 構成 | ms | allocated GiB | NVML GiB | IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|---:|
| int4_compact_group64_gaussian | 115.52 | 1.2485 | 2.8597 | 0.99153033 | 2550 |
| int4_compact_group128_gaussian | 115.594 | 1.2457 | 2.626 | 0.99206558 | 2603 |

両方1/4/6/4/0。group 64 / 128は2550 / 2603画素変化で、group 32の1995画素より増えた。
allocatedは1.249 / 1.246GiBで、group 32の1.267GiBからの削減は小さい。
続くCPUテキスト併用では、まずgroup 16/32の候補を使う。

## Round 59：余白の射影省略と画像追加パッチの併用

| 構成 | ms | allocated GiB | NVML GiB | IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|---:|
| r59_refined_resolution896_control | 79.052 | 0.733 | 2.4827 | 0.97368888 | 8507 |
| refined_unpadded_resolution896 | 88.025 | 0.7339 | 2.1982 | 0.97434917 | 8476 |
| r59_refined_resolution784_control | 77.668 | 0.7253 | 2.4144 | 0.97123186 | 9296 |
| refined_unpadded_resolution784 | 77.847 | 0.6916 | 2.165 | 0.97132407 | 9352 |
| r59_fp16_refined784_control | 87.089 | 1.1024 | 2.1768 | 0.97107591 | 9351 |
| fp16_refined_unpadded784 | 80.704 | 1.102 | 2.1943 | 0.97110031 | 9343 |

全構成1/4/6/4/0。INT8の896では79.05→88.02ms、784では77.67→77.85msで速度改善なし。
784のallocatedは0.725→0.692GiBへ減った。FP16の784では87.09→80.70ms、allocatedは約1.102GiBで同等。
CPUテキストを毎回計算する今回の構成では、Round 49のINT8速度改善をそのまま再現しなかった。
既定パッチは変えず、FP16・低解像度での選択肢として保持する。解像度低下によるマスク差はどの構成にも残る。
