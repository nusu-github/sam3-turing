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
