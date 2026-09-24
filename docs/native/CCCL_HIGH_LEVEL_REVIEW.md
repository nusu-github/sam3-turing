# libcu++ / Thrustで記述を短くする調査

2026-09-24。対象はWindows/MSVC、CUDA 13.0、同梱CCCL 3.0.1、SM75、C++17。
今回は本体を置き換えず、公式資料・導入済みヘッダー・単体のコンパイル／実行で調べた。

## 結論

記述短縮の中心は **Thrustの`tabulate`、`transform`、必要に応じた`for_each_n`**。
libcu++は **`span`／`mdspan`でポインター・サイズ・ストライドをまとめる**用途が有望。
既存のATenによる短いテンソル演算は維持し、手書きカーネルや専用functorの足場を減らす。

Thrustは範囲に対する並列アルゴリズム、libcu++はhost/deviceで使える型や標準ライブラリー機能、
CUBはその下で使う並列プリミティブ、と役割を分ける。libcu++の型を使うだけで演算が
GPU全体へ並列化されるわけではない。[Thrust概要](https://nvidia.github.io/cccl/unstable/thrust/)、
[libcu++概要](https://nvidia.github.io/cccl/unstable/libcudacxx/)

## この環境で実際に使えるもの

| API | 確認結果 | 用途 |
| --- | --- | --- |
| `thrust::tabulate` | C++17/SM75でコンパイル・実行・Graph再実行成功 | 添字から出力を作る。counting iteratorを呼び出し側から消せる |
| `thrust::transform` | 同上 | 入力値から出力値を作る。小さなlambdaで専用functorを省ける |
| `thrust::for_each_n` + zip iterator | 同上 | 複数配列をまとめて処理する |
| `cuda::std::span` | device lambda内で実行成功 | 所有権を持たない一次元ビュー |
| `cuda::std::mdspan` + `layout_stride` | C++17の`view(i,j)`でdevice内参照成功 | 多次元／ストライド付きビュー |
| `cuda::stream_ref` | ヘッダー存在を確認、実行検証はしていない | ストリームの非所有ラッパー。今回は既存c10管理を優先 |
| `cuda/std/algorithm`、`cuda/std/execution` | 導入済みヘッダーなし | 最新Web資料のコードをそのまま使えない |
| `cuda/algorithm`、`cuda/launch` | 導入済みヘッダーなし | 今回の採用対象外 |
| `cuda/experimental/stf.cuh` | 導入済みヘッダーなし | タスク実行基盤の導入は今回の短縮目的から外す |

公式の最新資料は`cuda/std/algorithm`をCCCL 3.2/CUDA 13.2、
`cuda/std/execution`をCCCL 3.4/CUDA 13.4以降としている。
インストール済み3.0.1と混同しない。今回の候補にSDK更新は不要。
[Algorithms Library](https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api/algorithms_library.html)

`mdspan`はデータの移動・転置・メモリ所有・自動境界チェックを提供するものではない。
既存Tensorのpointer/shape/strideを読みやすく表すために使う。
[Container Library](https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api/container_library.html)

## リポジトリの候補と優先順位

| 優先 | 対象 | 提案と短縮できるもの |
| --- | --- | --- |
| 1 | `src/ops_cuda.cu`の圧縮／展開 | `tabulate` + lambda。PackMask/UnpackMaskの専用型とcounting iteratorの記述をまとめる |
| 1 | `src/components_cuda.cu`のサイズ参照 | `transform` + 条件式lambda。GatherSize型を削除できる |
| 1 | `src/vision_position_cuda.cu`の軸テーブル | `tabulate` + lambda。AxisValue型と添字iteratorを省く |
| 2 | `src/roi_align_cuda.cu`の外側ループ | `tabulate`から既存`roi_sample`を呼ぶ。薄いカーネルと起動設定を除ける候補。性能未測定 |
| 2 | `src/rotary_pair_cuda.cu` | `for_each_n`でペアを処理。既存`rotate_pair`と明示FMA丸めは維持。zipを増やしすぎない |
| 2 | `src/rotary_cuda.cu`、ROIのアドレス計算 | `mdspan/layout_stride`を試す。shape/stride管理の共通化が実際に短くなる場合だけ採用 |
| 低 | `tools/int4_experiment.cu`の補正 | `for_each_n`で書けるが現在のカーネル自体が非常に短く、節約は小さい |
| 維持 | 正規化、行ごとの量子化、MSE選択 | Thrustの複数アルゴリズムへ分割すると起動回数・中間配列・加算順序が変わり得る。融合済みCUB処理を残す |
| 維持 | NMS／union-find／EDT | 依存関係が強く、単純なmap/scan/select置換では意味を保てない |
| 維持 | `image_results.cpp`、`ops.cpp`のATen選択・sort | 既に高レベルで短い。Thrustへ移すとdtype、出力形状、所有権の管理が増える |

前回、CUB化で遅くなった位置データの大量展開を、API名だけ変えて採用し直すことはしない。
Thrust版で再度性能を確認する必要がある。

## 短縮イメージ

以下は本体への適用案。実モデルでの性能・完全一致はまだ検証していない。
Tensorの検証・device guard・出力確保は既存のまま、演算部分を置き換える。

```cpp
auto policy = thrust::cuda::par_nosync.on(c10::cuda::getCurrentCUDAStream());
const auto* hist = histogram.const_data_ptr<int64_t>();
const auto* first = labels.const_data_ptr<int64_t>();
thrust::transform(policy, first, first + n, sizes.mutable_data_ptr<int64_t>(),
    [hist] __device__(int64_t label) { return label ? hist[label - 1] : 0; });
```

`GatherSize`の型定義が不要になる。`gather`だけに置き換えると背景ラベル0で`hist[-1]`を
参照してしまう。`gather_if`とゼロ初期化を足すより、この条件付きtransformの方が単純。

```cpp
thrust::tabulate(policy, dst, dst + count,
    [src, pixels, bytes] __device__(int64_t i) {
      const auto p = i % pixels;
      return bool((src[(i / pixels) * bytes + p / 8] >> (p % 8)) & 1);
    });
```

こちらはマスク展開。専用型、counting iterator、独自起動設定が不要になる。
長い処理はlambdaに押し込まず、既存の数学的ヘルパーを呼ぶ形にする。

## 同期・確保・実行経路

- Tensor所有権はATenに残す。`device_vector`へ移し替えるための確保・コピーは不要。
  生のdevice pointerと明示CUDA execution policyを使う。
- `par_nosync.on(current_stream)`を基本にする。これは不要な同期を省略できる方針であって、
  全アルゴリズムの非同期実行を保証するものではない。
- 特にhostへ値を返す`reduce`は、導入済み実装で一時領域を確保し、結果取得前に同期する。
  GPU内で使う行スケールをhostへ戻す置換には使わない。
- 動的な件数をhostへ返す選択処理やsortの作業領域も個別確認が必要。
  `par_nosync`だけで確保・同期の問題が消えるとは考えない。
- 素朴なtransformを何回も並べても、それらが自動的に一つのカーネルへ融合されるとは限らない。
  zip/transform iteratorは中間配列を省く候補だが、呼び出しを跨ぐ自動融合ではない。
- device lambdaにはraw pointer、ビュー、必要な数値だけを渡す。`at::Tensor`自体をcaptureしない。
  現在のdevice/streamとTensorの非同期寿命を維持する。
- Thrustは失敗を例外で報告する経路を持つ。既存の例外境界と非同期エラー検証は維持する。

ストリーム方針の公式例:
[explicit CUDA stream](https://github.com/NVIDIA/thrust/blob/main/examples/cuda/explicit_cuda_stream.cu)

導入済みソースの確認では、`thrust/system/cuda/detail/transform.h`はCUBのtransform dispatchへ
委譲し、optional synchronizationを行う。一方`tabulate.h`はCUDA backendのparallel_forへ
委譲する。**Thrust化しても、これまでのCUB呼び出しと同じカーネルになる保証はない**。
引数のアドレス安定性などでも選択経路が変わるため、同条件の測定が必要。

## 実行確認と次の実装単位

`experiments/cccl_highlevel_probe.cu`をCUDA 13.0 NVCC + MSVC、C++17、SM75、
`--extended-lambda`でコンパイルした。上記Thrust 3 API、zip、span、layout_stride mdspanを
同じ非デフォルトストリームで実行し、Graph capture後の3回再実行と128要素の期待値一致を確認。
これはAPI利用可能性の確認で、実モデルの性能・精度検証ではない。

ビルド／実行記録は`.cache/cccl-highlevel/build.log`、`run.log`。
本体・CMake・依存バージョンは今回変更していない。

次に実装するなら、優先1の4変換をひとまとまりにする。専用functorを短いlambdaへ置換し、
`check_cub_ops.py`と位置埋め込み54ケース、Graph再実行、旧CUB版との交互計測を再利用する。
ROIAlignとmdspanは、その結果を見て別の変更単位にする。


## 実装・比較結果（2026-09-24）

優先1の4処理をThrustへ置き換えて比較し、最終的に **unpackと位置埋め込みの軸テーブル生成** を採用した。
`thrust::tabulate`とdevice lambdaで、専用functor・明示的counting iteratorを削除。
Tensorの所有権、current CUDA stream、空入力の早期returnは維持した。
`--extended-lambda`は対象2ソースに限定してCMakeで指定する。CCCL更新・追加リンクライブラリは不要。

packはtabulate版に速度のばらつきがあり、中央値も悪化したため既存CUBを維持。
連結成分のサイズ参照はtransform版にしても行数が減らず、全体の性能同等性を確認できなかったためCUBを維持。
連結成分のrandom測定は旧版も二峰性であり、差をgatherカーネル単体の原因と断定しない。
libcu++のspan/mdspan、ROIAlign、RoPEは今回は変更していない。

### 測定

同じビルド条件のCUB版DLLを保存し、旧・新・新・旧、その逆順の計8プロセスで比較。
各値は30バッチ×20 Graph replayの中央値をさらに4プロセス間で中央値集計。
モデル全体のレイテンシではなく、対象op全体のGraph replay（出力割り当て除外）。
以下のThrust列は4候補を置き換えた試行版で、最終版では不採用2処理をCUBに戻している。

| 処理 | CUB ms | Thrust ms | 採否 |
|---|---:|---:|---|
| pack [1, 1008, 1008]  | 0.014996 | 0.016313 | CUB維持 |
| unpack [1, 1008, 1008]  | 0.032333 | 0.027035 | 採用 |
| components [1, 256, 256] random | 0.078688 | 0.099830 | CUB維持 |
| components [1, 256, 256] zero | 0.024308 | 0.022414 | CUB維持 |
| components [1, 256, 256] one | 0.402232 | 0.395639 | CUB維持 |
| position 5 72 | 0.042625 | 0.043293 | 採用 |
| position 5 144 | 0.135852 | 0.136459 | 採用 |
| position 5 288 | 0.471135 | 0.471398 | 採用 |
| position 6 72 | 0.066610 | 0.065824 | 採用 |
| position 6 144 | 0.150560 | 0.149453 | 採用 |
| position 6 288 | 0.493223 | 0.494112 | 採用 |

位置埋め込みのdtype 5はHalf、6はFloat。測定対象サイズでは差は約−1.2〜+1.6%。
unpack 1008²は約16.4%短縮。ただし処理単体の数µs差であり、モデル全体の高速化は主張しない。

### 検証記録

4候補の試行版でCTest 51/51通過（138.99秒）。CPUとの完全一致、空入力・byte境界・
strided入力、非デフォルトstream、入力変更後のCUDA Graph replayを確認。
位置埋め込み54ケースも完全一致。8プロセスのops出力SHA256はすべて
`2cdf93d7663962e687777a78fc912a6bd2c6a9e5d34e22bc702714c52836fa63`。

測定データは`experiments/results/native_rtx2060/thrust-*.json`、
`thrust-confirm-*.json`、`thrust-combined-summary.json`。
ビルド・比較ログと旧DLLは`.cache/thrust-pass/`に保存。

最終版（pack/gatherをCUBへ復帰）を再ビルドし、対象CTest 3/3、位置埋め込み54ケース、
opsのCPU完全一致・Graph再実行を再確認。Compute Sanitizer memcheckはopsと位置埋め込みの
両方で0 errors。ログは`thrust-final-tests.log`と`.cache/thrust-pass/memcheck-{ops,position}/process.log`。
今回の本体差分は3ファイル合計6行減（CMake指定は別）。大幅な行数削減ではなく、
採用2処理の専用型をなくして計算式を呼び出し位置へまとめた小規模な保守性改善。
