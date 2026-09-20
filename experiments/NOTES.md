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
