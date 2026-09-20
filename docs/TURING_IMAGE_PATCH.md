# SAM3-Turing 画像パッチ

通常の画像推論には `apply_turing_patch(processor, compile=True)` を使う。
起動の軽さを優先する場合は `compile=False`、長時間の連続処理には
`compile="max-autotune"` も選べる。元のモデルファイルを編集せず、実行時に適用する。

## RTX 3090での結果

このRunPodコンテナの既存Python / NVIDIA PyTorch 2.10.0a0 / CUDA 13.0で測定した。
専用の仮想環境は使用していない。Turing 6GB実機は未検証。

| 構成 | 1画像 ms | PyTorch allocated GiB | GPU全体 NVML GiB | マスク平均IoU（素の状態比） |
|---|---:|---:|---:|---:|
| 素の状態・BF16 | 204.13 | 4.994 | 6.017 | 1.0 |
| FP16共通基準 | 210.77 | 3.811 | 4.952 | 0.99918 |
| 最終パッチ・コンパイルなし | 174.94 | 2.249 | 3.405 | 0.99918 |
| 最終パッチ・コンパイルあり | **153.53** | **2.125** | **3.345** | **0.99920** |
| 同上・語句キャッシュ無効 | 176.90 | 2.126 | 3.345 | 0.99920 |
| max-autotune | 149.62 | 2.141 | 3.409 | 0.99921 |
| 固定語句＋コンパイル | 152.42 | 1.228 | **2.429** | 0.99920 |
| 解像度784＋コンパイル | 125.12 | 2.035 | 3.161 | 0.97111 |
| 解像度672＋コンパイル | 101.61 | 1.999 | 3.088 | 約0.954 |
| 解像度560＋コンパイル | 95.08 | 1.964 | 3.062 | 約0.934 |

コンパイル版は、素の状態から時間を約25%、GPU全体の使用量を約44%削減した。
同じ語句を使う画像ではテキスト特徴を再利用する。未知の語句を毎回処理する比較は
「語句キャッシュ無効」の行で、こちらも約13%短縮した。

速度は `truck.jpg` + `truck` の `set_image` + `set_text_prompt` 全体。
前処理とGPU転送を含み、画像ファイルの読込み・モデル構築は含めない。
2回ウォームアップ後、最終候補は9回の中央値。素の状態の最終測定は7回。
TF32は全構成でOFF。NVMLは5ms間隔のGPU全体の標本最大値。

結果差は3画像・5条件（truck、paper bag、child、wheel、空のelephant）で比較した。
検出数は標準版・最終コンパイル版とも **1 / 4 / 6 / 4 / 0**。
最終コンパイル版の差は **286 / 18,038,400画素（0.00159%）**、
score最大差 **0.00342**、box最大差 **0.50画素**。
boxの対応付け後にマスクを比較した。正解ラベルに対する精度評価ではない。

コンパイルの初回推論は、この一連の測定で約10秒、max-autotuneは約81秒だった。
コンパイラのキャッシュ状況で変わる。max-autotuneは約4msの追加改善なので、
起動のたびに使う標準設定にはしていない。

[全候補の表](../experiments/results/README.md) / [CSV](../experiments/results/summary.csv) /
[最終コンパイル版JSON](../experiments/results/final_compiled.json)

初回の画像推論探索は50候補・55試行（再測定込み）。初期の3失敗は修正して再実行済み。

## 使い方

```python
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.turing import apply_turing_patch

# このNGCコンテナではCPU初期化を1スレッドにすると安定する。
torch.set_num_threads(1)
model = build_sam3_image_model()  # checkpoint_path=... も指定可能
torch.set_num_threads(4)
processor = apply_turing_patch(Sam3Processor(model), compile=True)

state = processor.set_image(Image.open("assets/images/truck.jpg").convert("RGB"))
result = processor.set_text_prompt(prompt="truck", state=state)
masks, boxes, scores = result["masks"], result["boxes"], result["scores"]
```

画像・batch 1・推論専用。テキストと幾何box prompt、しきい値変更、空出力に対応する。
SAM 1形式のインスタンス操作・学習・SAM 3.1動画への適用は対象外。
パッチはモデル/processorの組に1回適用する。解除する場合はモデルを作り直す。

語句キャッシュは最大16件。`text_cache_size=0` で無効化できる。
固定語句だけを使う用途では、次を追加してテキストエンコーダの重みを解放できる。

```python
from sam3.turing import freeze_text_prompts
freeze_text_prompts(processor, ["truck", "wheel", "person", "visual"])
```

その後は未登録の語句がエラーになる。幾何boxだけの指定には `visual` を含める。

速度優先で解像度を下げる場合は、processor作成時に指定する。パッチがRoPEも調整する。
マスクの形状差が大きくなるため、標準値は1008のままにした。
560では最も差が大きいマスクのIoUが約0.62、box最大差は約26画素だった。

```python
processor = apply_turing_patch(Sam3Processor(model, resolution=784), compile=True)
```

## 取り込んだ変更

- BF16を強制するViT MLPを、FP16 autocastに従うLinear＋GELUへ置換。
- Linear / Convの重みを選択的にFP16化。decoderのFP32専用FFNとEmbedding等は維持。
- autocastの重みキャッシュを無効化し、FPNのclone、使わない段・位置キャッシュを削減。
- テキスト特徴をLRUキャッシュし、補間後のsigmoidをin-place化。
- 任意で画像エンコーダとセグメンテーションヘッドをコンパイル。

融合MLP、GELU近似、channels-last、PixelDecoderのin-place化、早期query選別を
個別・組み合わせで試したが、最終構成への追加効果は小さかった。
早期選別は `early_filter=True` として試せるが、既定では無効。
ヘッド射影の再結合、Attention固定、MLP分割、arena再利用、非コンパイルの実数RoPEは
今回の3090では標準採用に至らなかった。候補コードと測定値は実験ディレクトリに残した。

## 多数の二値マスクを小さく返す

`packed_masks=True` は、確率配列を保持せず、二値マスクを1画素1bitで返す追加オプション。
Tritonを使用する。通常の `masks` / `masks_logits` の代わりに
`masks_packed` と `mask_shape` を返す。

200 queryの実モデルlogitを4Kへ拡大する出力部品の比較では、dense二値出力が
24.10ms・6.243GiBに対し、8枚ずつ生成してbitpackすると **28.64ms・0.505GiB**。
出力は約198MiB。抽出した8マスク・66,355,200画素の比較では差0だった。
これは画像全体の速度ではなく、4K出力部品のストレス試験。
さらに一時メモリを減らす `mask_chunk_size=1` は46.06ms・0.288GiBだった。
従来経路のchunk既定値は8。[測定JSON](../experiments/results/packed_masks.json)

継続最適化では、この4段階を1つのTriton kernelに融合した。同じ4K出力部品の
比較で、従来のchunk 8は28.72ms・0.505GiB、新方式は **8.18ms・0.257GiB**。
200マスク全体の **1,658,880,000画素中1画素** が閾値付近の丸めで異なった。
奇数サイズ、縮小、閾値付近の値、1×1入力の追加比較は差0。
FP16/FP32ではこの方式を既定にした。`mask_chunk_size`は他のdtypeでの従来経路に使う。
直接 `resize_and_pack_masks(logits, size, fused=False)` を呼べば従来経路も選べる。
[追加測定JSON](../experiments/results/fused_masks.json)

```python
from sam3.turing_masks import unpack_masks

processor = apply_turing_patch(Sam3Processor(model), packed_masks=True)
state = processor.set_text_prompt("truck", processor.set_image(image))
h, w = state["mask_shape"][-2:]
first_mask = unpack_masks(state["masks_packed"][:1], (h, w))
```

## パッチと実験を再利用する

単独の適用ファイルは [patches/turing-image.patch](../patches/turing-image.patch)。
SAM3-Turingのこの変更を含まない上流チェックアウトで `git apply` し、上記APIを呼ぶ。
既にこのブランチを使っている場合は適用不要。

比較ループは [experiments/README.md](../experiments/README.md) を参照。
既存のPython環境で不足していた `timm`、`ftfy`、`iopath`、`portalocker` だけを追加した。
PyTorchやCUDAは入れ替えていない。

基準ソース: `2345a4ad109ac29c569da749c91d84f10dc08c40`。
重み: `facebook/sam3`、revision `3c879f39826c281e95690f02c7821c4de09afae7`。
入力資料は添付のFP16 / 20USD / GPU validationの3アーカイブ。
