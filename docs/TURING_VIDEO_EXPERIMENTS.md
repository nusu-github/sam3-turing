# SAM 3.1の動画試作

![SAM 3.1 video benchmarks](images/video_benchmarks.png)

RTX 3090・同梱動画0001の先頭24フレーム・person prompt。既存コンテナのPythonを使用。
全構成でgrounding batch 1、フレームはCPU保存、FA3とTF32はOFF。最初の3構成はcompileなし。
cold run後の3回の中央値。add_prompt＋順方向追跡を計測し、フレーム読込み・モデル構築は除外。

同じ比較はリポジトリのルートから実行できる。重みが既にある場合は、そのパスを指定する。

```bash
hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir checkpoints/sam3.1
python experiments/video_sweep.py \
  --checkpoint checkpoints/sam3.1/sam3.1_multiplex.pt \
  --variants stock fp16_int8_cpu_trim_compile fp16_int8_cpu_trim_compile_efficient
```

stockの保存出力を後続候補の比較に使う。今回のstock推論は6GiBを超えており、この比較は3090で実施した。
新しい仮想環境は不要。各候補は同じPythonの別プロセスで順に実行する。

| 構成 | ms/frame | allocated GiB | GPU全体 NVML GiB | mask IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|---:|
| [video_stock_bf16_b1](../experiments/results/video_stock_bf16_b1.json) | 261.00 | 6.726 | 7.919 | 1.000000 | 0 |
| [video_fp16_b1](../experiments/results/video_fp16_b1.json) | 268.50 | 3.902 | 5.030 | 0.998608 | 4282 |
| [video_fp16_efficient_b1](../experiments/results/video_fp16_efficient_b1.json) | 284.79 | 3.899 | 5.110 | 0.998610 | 4270 |

全24フレームで4人、合計96個のマスク対応と人物ID対応を維持した。比較画素数88,473,600。
FP16はallocatedを約42%削減したが、速度は基準より少し遅かった。score最大差は0.00135。
IoUは素の出力との一致度であり、正解ラベルに対する精度評価ではない。Turing実機でも未確認。

現在は[実験用monkeypatch](../experiments/video_variants.py)。画像専用パッチのneck削除は動画には適用していない。
SAM 3.1内のBF16固定のcast・autocastをFP16へ置き換え、Linear/Conv/Attentionの重みをhalfで保持する。
FP32固定のdecoder FFNとnormは保持。FlashAttentionなしの候補はCUDAのSDPAをefficientに固定する。

共通APIとSAM 3.1のinit_stateに引数の不一致があったため、両構成とも未対応の既定値
offload_state_to_cpu=Falseを外す補正を入れた。元の失敗はfailure JSONに保存した。
offload_state_to_cpu=Trueには対応していない。通常のstockモデルにこのAPI補正と上記batch設定を加えた基準。

重み: facebook/sam3.1 / sam3.1_multiplex.pt、revision daa63191845a41281374e725f4c9e51c7a824460。
続くRound 2で非対称INT8＋重みスケール調整、ViT compile、CPUテキストを比較した。

## 動画 Round 2：INT8・ViT compile・CPUテキスト

| 構成 | ms/frame | allocated GiB | NVML GiB | IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|---:|
| [video_fp16_int8_b1](../experiments/results/video_fp16_int8_b1.json) | 265.94 | 3.496 | 4.632 | 0.998099 | 8151 |
| [video_fp16_int8_compile_b1](../experiments/results/video_fp16_int8_compile_b1.json) | 195.40 | 3.509 | 5.003 | 0.998080 | 8313 |
| [video_fp16_int8_cpu_compile_b1](../experiments/results/video_fp16_int8_cpu_compile_b1.json) | 226.79 | 2.849 | 4.485 | 0.998076 | 8319 |
| [video_fp16_int8_cpu_compile_efficient_b1](../experiments/results/video_fp16_int8_cpu_compile_efficient_b1.json) | 226.65 | 2.838 | 4.505 | 0.997858 | 7567 |

全構成で24フレーム×4人・96件の人物ID対応を維持。INT8だけでは速度差が小さかったが、
ViT compileで265.94→195.40ms/frameへ短縮した。基準261.00ms/frameに対し約25%短縮。
CPUテキストはallocated 2.849GiBに下がる一方226.79ms/frameへ遅くなった。
CPUテキスト＋efficientでも226.65ms/frame・2.838GiB。
CPUの短い語句でpadding省略を使い、この待ち時間を減らす候補を追加する。
画像の公開パッチと同じ非対称INT8・重み調整を使うが、動画では別途そのまま同じ数値になるとは限らない。

## 動画 Round 2b：CPUテキストpadding省略

| 構成 | ms/frame | allocated GiB | NVML GiB | IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|---:|
| [video_fp16_int8_cpu_trim_compile_b1](../experiments/results/video_fp16_int8_cpu_trim_compile_b1.json) | 198.79 | 2.838 | 4.485 | 0.998079 | 8305 |
| [video_fp16_int8_cpu_trim_compile_efficient_b1](../experiments/results/video_fp16_int8_cpu_trim_compile_efficient_b1.json) | 221.11 | 2.838 | 4.485 | 0.997859 | 7565 |

全フレーム4人・96件のID対応を維持。通常AttentionのCPUテキスト版はpadding省略で
226.79→198.79ms/frameへ短縮し、allocated 2.838GiB。score最大差0.00771。
テキスト計算は各runで1回、3 captionをまとめていた。省略後のCPU計算は約0.10〜0.14秒/run。
画像で採用した末尾padding省略が動画でも待ち時間の軽減に有効だった。

## 動画 Round 3：検出decoderのコンパイル

| 構成 | ms/frame | allocated GiB | NVML GiB | IoU vs stock | 変化画素 |
|---|---:|---:|---:|---:|---:|
| [video_fp16_int8_cpu_trim_compile_rpb_b1](../experiments/results/video_fp16_int8_cpu_trim_compile_rpb_b1.json) | 197.79 | 2.849 | 4.485 | 0.998079 | 8305 |
| [video_fp16_int8_cpu_trim_compile_decoder_b1](../experiments/results/video_fp16_int8_cpu_trim_compile_decoder_b1.json) | 162.73 | 2.835 | 4.427 | 0.998076 | 8331 |
| [video_fp16_int8_cpu_trim_compile_decoder_rpb_b1](../experiments/results/video_fp16_int8_cpu_trim_compile_decoder_rpb_b1.json) | 164.79 | 2.835 | 4.427 | 0.998076 | 8331 |

全フレーム4人・96件のID対応を維持。decoderコンパイルで198.79→162.73ms/frameへ短縮した。
RPB座標にPython整数を渡すだけでは197.79ms/frameで差は小さかった。
decoderとRPBの併用は164.79ms/frameで、decoder単独と出力比較値は同じだった。
decoder単独のcold runは今回のキャッシュで19.34秒。追加の保存出力はCUDA Graph外でcloneする。

```bash
python experiments/video_sweep.py \
  --checkpoint checkpoints/sam3.1/sam3.1_multiplex.pt \
  --variants fp16_int8_cpu_trim_compile_decoder
```
