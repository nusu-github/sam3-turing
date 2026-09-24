# CUDA / PyTorch ライブラリによる手動実装削減の検討

調査日: **2026-09-24**。対象: `native/` の CUDA カーネルと、それと対になる CPU 実装。
前提は [CUDA_LIBRARIES.md](CUDA_LIBRARIES.md) と同じ(Windows/MSVC 14.44、CUDA 13.0 Update 3、
同梱 CCCL 3.0.1、SM75、LibTorch 2.10.0+cu130、C++17)。既定の FP16 経路はビット一致を維持し、
SDK が同梱するのは LibTorch 由来のライブラリ(cuBLAS/cuBLASLt、cuDNN、NVRTC など)だけという制約も同じ。

本書は GPU を持たないコンテナで、一次資料の収集、LibTorch 2.10 / CUDA 13.0 の実ヘッダー確認、
nvcc 13.0 でのコンパイル確認、CPU 上の等価性確認を行ったもの。**RTX 2060 での実行、Windows ビルド、
速度測定はまだ行っていない**。どの案も、採用前に下記の [検証計画](#rtx-2060-での検証計画) を満たす必要がある。

## 結論

手動実装を大きく減らせるライブラリは見つからなかった。コード量の多い手書き部分
(vision fusion の Welford 405 行、INT8 境界カーネル、union-find、EDT、画素転置、Pillow 互換リサイズ)は、
ビット一致、SM75、Windows、6 GB のメモリ制約のどれかにライブラリが合わない。
一方、次の四つは保守性の改善が見込め、GPU 上で検証する価値がある。

| 優先 | 案 | 削減・効果 | リスク |
|---|---|---|---|
| 1 | [位置エンコーディングの CUDA カーネルを廃止](#r1-位置エンコーディングの-cuda-カーネルを廃止) | `vision_position_cuda.cu`(55 行)、重複した ATen 式、CMake オプション1個、Thrust デバイスラムダ | 本番の 72×72 で参照式(十数回の起動)に戻る分の時間。測定が必要 |
| 2 | [NMS の逐次処理をホストで行う](#r2-nms-の逐次処理をホストで行う) | `nms_kernel` と CPU/CUDA の分岐 | n×n bool の D2H コピー(n≤200 で 40 KB)。後続の `masked_select` はもともと同期する |
| 3 | [RoPE の丸め指定を1か所にまとめる](#r3-rope-の丸め指定を1か所にまとめる) | 3 か所の `_WIN32` 分岐(`rotary_cuda.cu`、`rotary_pair_cuda.cu`、`approx_kernels.cu`) | Windows と Linux の両方でビット一致を再確認する必要がある |
| 4 | [要素単位カーネルを `cub::DeviceFor` へ](#r4-要素単位カーネルを-cubdevicefor-へ) | 起動設定と添字分解の手書きを削除(`rotary_cuda.cu` 60→39 行、`components_cuda.cu` 90→81 行) | 起動形状の変化による速度差。行数の削減は小さい |
| 条件付き | [マスクの pack/unpack を ATen 演算へ](#c1-マスクの-packunpack-を-aten-演算へ) | R2 と合わせ、`ops_cuda.cu` を全廃し `ops.cpp` を 114→76 行 | pack に N×H×W バイトの中間テンソルが1つ増える。既存方針「中間テンソルを増やさない」に反する |

1 と C1 の両方を採用すると、`--extended-lambda` を使うソースがなくなり、CMake からこの指定を消せる。

## 前提と判定基準

- **ビット一致**: 既定経路の変更は、既存のビット単位テスト(`ops_test`、`rotary_test`、
  `rotary_pair_test`、`vision_position_test`、`vision_fusion_test`)と、実重みの parity
  スクリプトで出力が変わらないこと。
- **SM75 / Windows**: CUDA 13.0 は Maxwell、Pascal、Volta のオフラインコンパイルとライブラリサポートを削除し
  ([リリースノート](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html))、
  Turing が最も古い対象になった。新しいライブラリ機能は SM80 以上を要求するものが多い。
- **依存の追加**: LibTorch が同梱しないライブラリ(NPP、cuTENSOR など)を加えると、
  [BundleRuntime.cmake](../../native/cmake/BundleRuntime.cmake) で DLL を配布する必要がある。
  参考として PyPI の Windows wheel は圧縮状態で cuTENSOR 2.8.1 が 178.6 MB、NPP 13.2 が 245.1 MB。
- **API の安定性**: LibTorch のうち `ATen/native/*` は実装用ヘッダーで、
  [C++ ドキュメント](https://docs.pytorch.org/cppdocs/index.html)の公開 API には含まれない。
  CCCL は `cub::` と `thrust::` の ABI を保証しない([README](https://github.com/NVIDIA/cccl))。

## 収集した資料と要点

| 資料 | このランタイムに関係する記述 |
|---|---|
| [PyTorch C++ API](https://docs.pytorch.org/cppdocs/index.html) | ATen、CUDA 支援(`c10/cuda/CUDAStream.h`、`CUDAGuard.h`、`ATen/cuda/CUDAContext.h` の `getDeviceProperties()` と cuBLAS/cuBLASLt/cuSPARSE/cuSOLVER ハンドル)、`PackedTensorAccessor32/64`、`TORCH_LIBRARY` |
| [Torch Stable API](https://docs.pytorch.org/cppdocs/api/stable/index.html)、[stable ABI](https://docs.pytorch.org/docs/2.14/notes/libtorch_stable_abi.html) | `torch/csrc/stable`、`torch/headeronly`、C shim。演算は `empty`、`copy_`、`matmul`、`amax`、`sum`、ビット演算など(2.9/2.10 で追加)。拡張モジュールの再ビルドを不要にする仕組みで、LibTorch 全体の置き換えではないと明記されている |
| [CUDA 13.0 Update 3](https://docs.nvidia.com/cuda/archive/13.0.3/) | 同梱 Thrust/CUB/libcu++ は 3.0.1。NPP は `_Ctx` のない旧 API と `nppGetStreamContext()` を削除。cuSPARSE・cuFFT も Turing より前を削除 |
| [NPP 画像フィルタ](https://docs.nvidia.com/cuda/archive/13.0.3/npp/image_filtering_functions.html) | PBA+ による厳密 EDT(ROI 幅・高さは 64〜32767、サイトが1つ以上必要)。`LabelMarkersUF`(同値画素の連結領域、`nppiNormInf` で8近傍、ラベルは順序不定で欠番あり) |
| [cuBLAS 13.0.3](https://docs.nvidia.com/cuda/archive/13.0.3/cublas/index.html) | `CUBLASLT_EPILOGUE_GELU` は tanh 近似。IMMA(`CUBLAS_COMPUTE_32I`、INT8 入力)は既定以外のエピローグ非対応。行ごと×列ごとの外積スケーリングは CC 9.0 の FP8 限定、テンソル単位スケーリングも 8.9 以上 |
| [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/)(2026-09-02 更新) | SDPA は SM80 以上。RoPE 演算は SDPA との融合専用で非インターリーブ(前半・後半)回転。LayerNorm 融合は実行時コンパイルのエンジンで、全テンソルが同じレイアウトである必要がある |
| [cuTENSOR 2.8.1](https://docs.nvidia.com/cuda/cutensor/latest/index.html) | SM 7.5 以上、CUDA 12.x/13.0、Windows 対応。縮約、リダクション、要素単位演算、型変換付きの置換(FP16→FP32 を含む)。単項演算子一覧に GELU や erf はない |
| [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) | CSR/COO/BSR/Blocked-ELL/SELL の SpMV・SpMM・SpGEMM・SDDMM など。ラベリングや NMS に当たる機能はない |
| [cuSPARSELt 0.9.1](https://docs.nvidia.com/cuda/cusparselt/index.html) | 2:4 構造化疎行列積。対応は SM 8.0 以上で、SM75 は対象外 |
| [cuDSS](https://docs.nvidia.com/cuda/cudss/index.html) | 疎行列の直接法ソルバー(LU、Cholesky、LDLᵀ)。該当する処理がランタイムにない |
| [CCCL](https://github.com/NVIDIA/cccl) | 最新の CCCL は現行と一つ前の CTK メジャー系列で使える(古い CCCL を新しい CTK と組み合わせる前方互換はない)。C++17/20。`cub::`/`thrust::` は ABI 非保証 |

LibTorch 2.10 cu130 の wheel から取り出したヘッダーと、`nvidia-cuda-cccl==13.0.85`
(`CCCL_VERSION 3000001`)で次を確認した。

- `cub::DeviceFor::Bulk`、`ForEachN`、`ForEachInExtents` は CCCL 3.0.1 にある。
  方針は静的な 256 スレッド × 2 要素(ブロックあたり 512 要素)で、grid-stride ではない。
  `ForEachInExtents` は行優先で、添字を `fast_div_mod` で分解する。
- `cuda::ceil_div`(`<cuda/cmath>`)はある。`cuda/std/algorithm` はない(3.2 以降)。
- CUB の `WarpReduceShfl` はシャッフル距離を 1, 2, 4, 8, 16 の順に使う。PyTorch の
  LayerNorm 用 Welford は 16 から 1 の順なので、結合順序の違いで丸めが変わる。
- LibTorch 2.10 の ATen 演算のうち CUDA カーネルを持つもの: `_int_mm`、`_scaled_mm`、
  `_weight_int8pack_mm`、`_weight_int4pack_mm`、`_efficient_attention_forward`、`_cudnn_attention_forward` など。
  PyTorch v2.10.0 のソースでは、`_weight_int4pack_mm` の CUDA 本体は `__CUDA_ARCH__ >= 800` のときだけ
  コンパイルされる。`_weight_int8pack_mm` の CUDA 版は、1 スレッドが出力 1 要素を FP32 で計算する素朴なカーネル。
- `ATen/native/cpu/UpSampleKernelAVXAntialias.h` の uint8 リサイズは PIL-SIMD 由来で、int16 係数を使う。

## 手動実装の棚卸しと判定

| ソース | 処理 | 本番での用途 | 判定 |
|---|---|---|---|
| `src/vision_position_cuda.cu` | 位置テーブル + 展開 | 本番はレベル 2(72×72)だけ | **R1: 廃止を検証** |
| `src/ops_cuda.cu` `nms_kernel` | 貪欲 NMS | 動画の検出 NMS | **R2: ホスト処理へ** |
| `src/ops_cuda.cu` pack/unpack、`src/ops.cpp` の CPU ループ | マスクのビット詰め | 動画の出力キャッシュ(1 マスクずつ)。`resize_and_pack_masks` は公開演算とツールのみ | **C1: 条件付き** |
| `src/rotary_cuda.cu`、`src/rotary_pair_cuda.cu`、`tools/approx_kernels.cu` `restore_rope` | 複素 RoPE | memory attention、vision | **R3 + R4** |
| `src/roi_align_cuda.cu` + `roi_align_impl.h` | ROIAlign(torchvision 由来) | ボックスプロンプト | R4(任意)。torchvision C++ ライブラリの追加は不採用 |
| `src/components_cuda.cu` | 連結成分(atomic union-find) | 穴埋め・小領域除去 | R4(任意)。NPP は不採用 |
| `src/edt_cuda.cu` + `edt_impl.h` | 放物線包絡 EDT | **ランタイム内の呼び出しなし**(公開演算とテストのみ) | 維持。公開 API から外すかは所有者判断。NPP は不採用 |
| `src/vision_fusion_cuda.cu` | Welford LayerNorm + 窓分割 + 型変換 | 全 vision ブロック | 維持(PyTorch と同じ演算順序が必須) |
| `tools/pixel_transform.cu` | FP16 NHWC → FP32 NCHW | `SAM3_EXPERIMENT_PIXEL=fused_nchw` | 維持。cuTENSOR は依存に見合わない |
| `tools/approx_kernels.cu`、`approx_boundary.cu`、`int4_experiment.cu` | INT8/INT4 量子化・復元 | 実験(opt-in) | 維持。cuBLASLt では代替できない |
| `src/tracking_preprocess.cpp` | Pillow 12.2 互換リサイズ | 追跡の前処理 | 維持。ATen の uint8 AA は PIL-SIMD 由来 |

## 推奨案

### R1: 位置エンコーディングの CUDA カーネルを廃止

[PERFORMANCE.md](PERFORMANCE.md#position-maps) のとおり、検出と追跡が使う位置マップは
レベル 2(72×72)だけで、対話的デコードは使わない。専用カーネルの効果として記録されているのは
`[1,256,288,288]` FP16 で 1.267 ms / 215 MB → 0.125 ms / 43 MB で、これは 288×288 の値。
72×72 では要素数が 1/16 になり、ATen の参照式は十数回のカーネル起動が主なコストになる
(推定 0.1〜0.2 ms、未測定)。RTX 2060 の画像 1 枚は約 540 ms。

変更内容:

- `vision_position.cpp` を参照式だけにし、`vision_position_cuda.cu` と CMake の
  `SAM3_FUSE_VISION_POSITION` を削除する。
- `VisionEncoder::position` の `#else` にある同じ参照式を消し、`vision_position_encoding` を呼ぶ。
- 融合版はすでに参照式とビット一致を確認済み(54 ケース)なので、出力は変わらない想定。

キャッシュについて: 上流 Python の `PositionEmbeddingSine` は形状ごとにキャッシュするが、
`sam3/vision_position.h` は「呼び出しごとに独立した所有」と「永続キャッシュなし」を契約にしており、
テストも別名にならないことを確認している。位置は `VisionFeatures` として公開 API から返るため、
キャッシュを入れる場合は契約の変更として別途扱う。

検証: `vision_position_test cpu/cuda`、`sam3_vision_positions_benchmark`(本番と同じ
レベル選択で vision 全体を計測)、`sam3_image_latency`(5 ケース、交互実行)。

### R2: NMS の逐次処理をホストで行う

`nms_kernel` は 1 ブロックで行を順に処理するカーネルで、`generic_nms` はその直後に
`order.masked_select(keep)` を呼ぶ。`masked_select` は出力サイズのためにホストと同期するので、
抑制行列をホストへ移して CPU の既存ループを使っても、増えるのは n×n バイトのコピーだけになる。

```cpp
const auto suppression = ious.gt(threshold).index_select(0, order).index_select(1, order).cpu().contiguous();
auto keep = at::ones({n}, suppression.options());
// ... 既存の CPU ループ ...
return order.masked_select(keep.to(order.device()));
```

CPU/CUDA の分岐と CUDA 側の宣言が消える。bool の論理演算だけなので出力は同一。
検証: `ops_test cuda`、`video_detection_parity.py` / `video_frame_parity.py`、動画 1 フレームの時間。

### R3: RoPE の丸め指定を1か所にまとめる

複素積の虚部の FMA 順序が LibTorch と一致する必要があり(Linux は `fma(b,c,round(a*d))`、
Windows は `fma(a,d,round(b*c))`)、同じ `#ifdef _WIN32` が三つのファイルにある。
しかも `rotary_cuda.cu` の Linux 側だけは NVCC の自動縮約に任せ、残り二つは組み込み関数で順序を固定している。
`__device__` の共通ヘルパーを1つ作り、実部と虚部の両方を `__fmaf_rn`/`__fmul_rn` で固定すれば、
コンパイラやカーネル構成の変化に左右されなくなる。実部の順序はまだ記録がないため、
RTX 2060(Windows)と Linux の両方で `rotary_test`、`rotary_pair_test`、`sam3_qkv_rope_bench` の一致を確認して決める。

### R4: 要素単位カーネルを `cub::DeviceFor` へ

grid-stride ループ、`std::min(..., 4096)` のブロック数計算、`int64_t` の添字分解を CUB に任せる。
`ForEachInExtents` は多次元の添字を関数オブジェクトに直接渡すので、手書きの `%` と `/` の連鎖が消える。

```cpp
template <typename scalar_t> struct Rotate {
  const scalar_t* input; const c10::complex<float>* frequencies; scalar_t* output;
  int64_t in[4], out[4], half;
  __device__ void operator()(int64_t, int64_t b, int64_t h, int64_t n, int64_t d) const {
    const auto source = b * in[0] + h * in[1] + n * in[2] + d * 2 * in[3];
    // ... 既存の複素積と書き込み ...
  }
};
const cuda::std::dextents<int64_t,4> extents{value.size(0), value.size(1), value.size(2), value.size(3) / 2};
C10_CUDA_CHECK(cub::DeviceFor::ForEachInExtents(extents, op, c10::cuda::getCurrentCUDAStream()));
```

| ファイル | 対象 | 空行を除く行数(試作) |
|---|---|---|
| `rotary_cuda.cu` | `Layout` の分解ループ | 60 → 39 |
| `components_cuda.cu` | 3 カーネルを `DeviceFor::Bulk` へ | 90 → 81 |
| `roi_align_cuda.cu` | カーネルと起動計算 | 31 → 30 |

行数よりも、起動設定と添字計算の手書きがなくなることが利点。適さないものもある。

- **EDT**: 1 要素が 1 行全体の処理なので、1008 行ならブロック 2 個になり、30 SM の RTX 2060 をほとんど使わない。
  現行は 64 スレッド/ブロックで 16 ブロック。
- **NMS**(逐次)、**Welford/量子化**(ブロック協調)、**画素転置**(共有メモリのタイル)。

以前、全位置展開を `DeviceTransform` にして遅くなった例がある([CUDA_LIBRARIES.md](CUDA_LIBRARIES.md#rejected-substitutions))。
採用はファイルごとに、既存と同じ CUDA Graph の演算子計測で決める。
`rotary_cuda.cu` の Linux 側は自動縮約に依存しているため、R3 を先に入れてから置き換える。

## 条件付きの案

### C1: マスクの pack/unpack を ATen 演算へ

ビット詰めは ATen のシフトと和で書け、CPU と CUDA の実装を一つにできる。

```cpp
auto bits = masks.reshape({n, pixels}).view(at::kByte);
if (bytes * 8 != pixels) bits = at::constant_pad_nd(bits, {0, bytes * 8 - pixels});
return bits.view({n, bytes, 8}).bitwise_left_shift(at::arange(8, bits.options())).sum(-1, false, at::kByte);
// unpack
auto bits = packed.unsqueeze(-1).bitwise_right_shift(at::arange(8, packed.options())).bitwise_and_(1);
return bits.view({n, bytes * 8}).narrow(1, 0, pixels).view(at::kBool).reshape({n, 1, height, width});
```

この試作と R2 を組み込んだ `ops.cpp` を、元の `ops.cpp` と同時にリンクして CPU で比較し、
pack、unpack(末尾ビットがランダムな場合を含む)、`resize_and_pack_masks`(FP32/FP16)、
`generic_nms`(同点スコアを多く含む n = 0〜200)の 45 ケースがすべて一致した。

代償: pack はシフト結果として N×H×W バイトの中間テンソルを1つ作り、起動が 1 回から 2 回になる。
unpack は H×W が 8 の倍数でないとき reshape でコピーが1つ増える。現行の測定値は 1008×1008 で
pack 0.015 ms、unpack 0.027 ms。ランタイムで呼ぶのは `VideoMaskCache` だけで、マスクを 1 枚ずつ
詰めるため、中間テンソルは 4K 動画でも 1 枚あたり約 8.3 MB。公開演算の `resize_and_pack_masks` は
チャンク(既定 8 枚)単位なので、4K では約 66 MB になる。`video_mask_cache_benchmark` で
時間とピーク割り当て量を測ってから決める。

Python パッチの `sam3/turing_masks.py` の `_pack`/`_unpack`(Triton)も同じ式で置き換えられる。
`_resize_pack` の融合カーネルは対象外。

## 採用しない候補と根拠

| 候補 | 対象 | 根拠 |
|---|---|---|
| cuBLASLt エピローグ | INT8 の復元 + バイアス + GELU | IMMA は既定以外のエピローグ非対応。GELU は tanh 近似。行×列の外積スケーリングは CC 9.0 の FP8 限定 |
| `at::_scaled_mm`、`_weight_int4pack_mm`、`_weight_int8pack_mm` | INT8/INT4 GEMM | FP8(8.9 以上)、SM80 以上、FP32 の素朴な参照カーネル。SM75 の速度経路にならない |
| cuDNN SDPA / RoPE | vision attention | SDPA は SM80 以上。RoPE は SDPA 融合専用で、SAM3 の隣接ペア(複素)方式と回転の定義が違う |
| cuDNN LayerNorm 融合 | `vision_fusion_cuda.cu` | 全テンソル同一レイアウトの条件で窓分割出力を表せない。リダクション順序が `at::layer_norm` と異なり、既存のビット単位テストを満たさない。確認した LibTorch 2.10 cu130 の include には `cudnn.h` がなく、別途 cuDNN 開発パッケージが要る |
| CUB による Welford の置き換え | `vision_fusion_cuda.cu` | 結合順序が逆で丸めが変わる。PyTorch の実装は `.cu` にあり include できないため複製は残る。LibTorch 更新時の差分は `vision_fusion_test` が検出する |
| NPP `DistanceTransformPBA` | EDT | 64〜32767 とサイト1つ以上の制約で既存カーネルとフォールバックが残り、コードは減らない。EDT はランタイムで未使用 |
| NPP `LabelMarkersUF` | 連結成分 | 背景にもラベルが付き、ラベル番号が変わる(公開演算の出力が変わる)。サイズは出さないためヒストグラムは残る。CUDA 13 では `NppStreamContext` を自前で埋める必要がある |
| cuTENSOR 置換 | `pixel_transform.cu` | FP16→FP32 の置換自体は対応しているが、30 行のカーネルのために約 179 MB の依存を配布することになる。ATen のコピーは 256×288×288 で 11.07 ms、専用カーネルは 3.38 ms([PIXEL_AND_MLP_SCOPE](../../experiments/results/native_rtx2060/PIXEL_AND_MLP_SCOPE.md)) |
| cuSPARSE、cuSPARSELt、cuDSS | — | 疎行列演算、2:4 疎(SM80 以上)、疎ソルバー。対応する処理がない |
| Torch Stable API | ランタイム全体 | 専用ラッパーは `empty`、`copy_`、`matmul` などに限られ、conv2d、SDPA、layer_norm、補間は `torch_call_dispatcher` による汎用呼び出しになる。目的は拡張モジュールのバイナリ互換で、LibTorch を同梱して配布するこのランタイムには利点がない |
| `ATen/native/cuda/Loops.cuh`(`gpu_kernel`) | 復元などの要素単位カーネル | 公開 API ではない実装用ヘッダーで、LibTorch の更新ごとに追従が必要。置き換える対象は 10 行程度のカーネル |
| `TORCH_LIBRARY_IMPL` による CPU/CUDA 振り分け | `ops.cpp` などの `is_cuda()` 分岐 | C++ から関数を直接呼んでおり、ディスパッチャー経由にしても実装は減らない |
| ATen の uint8 AA リサイズ | Pillow 互換リサイズ | PIL-SIMD 由来の int16 係数で、Pillow 12.2 の固定 22 ビット係数と同一ではない |
| torchvision C++ ライブラリ | ROIAlign | 公式のビルド済み配布がなく、31 行のカーネルのために依存とビルドが増える |
| CUDA Graph | 全体 | 既に不採用([PERFORMANCE.md](PERFORMANCE.md#profile-after-fusion-and-rejected-candidates)) |
| 新しい CCCL(3.2 以降) | 全般 | CTK 13.x でも使えるが、LibTorch の CUB と ABI が混在しないよう `cub::`/`thrust::` 型を境界に出さない管理が要る。同梱 3.0.1 で足りている |

小さな API 置き換え(実験コードのみ): `int4_mm` と `kitchen_attention` が呼ぶたびに行う
`cudaGetDeviceProperties` は、LibTorch がキャッシュする `at::cuda::getDeviceProperties()` で置き換えられる。
グリッド計算の `(n + 255) / 256` は `cuda::ceil_div` で書ける。
`ATen/TensorUtils.h` の `checkAllSameGPU` などで引数検査を短くできるが、エラーメッセージが変わる。

## RTX 2060 での検証計画

1. 各案を別コミットにし、`SAM3_TEST_CUDA=ON` で CTest 全件(`ops_cuda`、`*_stream_test` を含む)。
2. 演算子: [CUDA_LIBRARIES.md](CUDA_LIBRARIES.md#adopted-changes-and-measurements) と同じ CUDA Graph の再生計測
   (30 バッチ × 20 回、old/new/new/old の順)と Compute Sanitizer memcheck。
3. モデル: `sam3_image_latency` の 5 ケースと、動画の `video_pipeline_parity.py` でバイト一致。
4. R3 と R4 の RoPE は Linux(Blackwell 開発機)と Windows(RTX 2060)の両方で一致を確認する。
5. C1 は `video_mask_cache_benchmark` で時間とピーク割り当て量を記録する。

## このコンテナで行った確認

証跡: [evidence/library-review-20260924.json](evidence/library-review-20260924.json)。

- nvcc 13.0.88(pip の `nvidia-cuda-nvcc` など)+ CCCL 3.0.1 + LibTorch 2.10.0+cu130 のヘッダーで、
  既存の CUDA ソース 11 本(`src/*.cu` 8 本、`tools/approx_kernels.cu`、`approx_boundary.cu`、
  `pixel_transform.cu`)が `-arch=sm_75` でコンパイルできた(Linux、g++ 13.3)。
- R4 の試作 3 本(`roi_align_cuda.cu`、`components_cuda.cu`、`rotary_cuda.cu`)も同条件でコンパイルできた。
  RoPE の Half カーネルの PTX は、元と試作で浮動小数点命令の構成が同じ(`fma.rn` 1、`mul` 3、`sub` 1、
  `cvt.rn.f16` 2)。ptxas 以降の縮約と実行結果は未確認。
- C1 + R2 の `ops.cpp` 試作は LibTorch 2.10 CPU にリンクし、元の実装と 45 ケースで一致した。
- 未実施: GPU 実行、Windows/MSVC ビルド、速度・メモリ測定。
