# SAM 3.1の動画試作

RTX 3090・同梱動画0001の先頭24フレーム・person prompt。既存コンテナのPythonを使用。
両構成ともgrounding batch 1、フレームはCPU保存、FA3とTF32はOFF、compileなし。
cold run後の3回の中央値。add_prompt＋順方向追跡を計測し、フレーム読込み・モデル構築は除外。

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
次は画像側の非対称INT8＋重みスケール調整、ViT compile、CPUテキストを比較する。
