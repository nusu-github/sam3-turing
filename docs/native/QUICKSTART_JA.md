# Python 不要の SAM3 / SAM3.1 ネイティブランタイム

このブランチの `native/` は、LibTorch / ATen C++ と事前コンパイル CUDA による推論ライブラリです。画像・動画の owning predictor、C ABI 1、C++ API、tokenizer、前後処理、SAM3.1 multiplex を実装しています。推論時に Python / Triton を起動・リンクしません。学習 API の移植ではありません。

重みはコードに含めず、SAM3 と SAM3.1 の共通・固有モジュールを `native-weights-v1/` に分けています。同じ store を画像・動画で使い、用途別の全重みコピーや固定プロンプト、検出数削減は導入していません。元の checkpoint はバックアップであり、ネイティブ実行には不要です。

2026-09-23 の実装チェックポイントは `1ee344087a2d8c05dd18ae9254122d668b2a2a6d`。公開ブランチは [codex/native-onboarding](https://github.com/nusu-github/sam3-turing/tree/codex/native-onboarding)、重み・配布物・検証記録はユーザーの[非公開バケット](https://huggingface.co/buckets/RamRom/sam3-turing-native-20260922)に保存しています。全要件の検証完了を宣言するリリースではありません。

## 最初に取得するもの

バケットの `latest/README-ja.md` と `latest/manifest.json` が復元の入口です。

- Linux: `releases/20260923-1ee3440/` の CPU または CUDA 13.0 SDK **どちらか一つ**。依存ライブラリ・ヘッダ・CMake package・語彙・ライセンス・診断用実行ファイルをまとめています。過去の overlay を順番に適用する必要はありません。
- 両 OS 共通: `native-weights-v1/` 一式。17ファイル、6,900,865,193 bytes。SAM3 / SAM3.1 の両方を含む共有 store です。
- Windows: 同じ公開ソースを [Windows/Turing 手順](WINDOWS_TURING.md)でビルドします。Linux SDK は Windows 用バイナリではありません。

`hf` CLI は認証・取得時の道具です。取得後の C/C++ 実行環境に Python を残す必要はありません。バケット全体を同期すると、過去の検証データや元 checkpoint も取得してしまいます。現在の配布だけなら上記の範囲で足ります。

Linux の復元用スクリプトとチェックサムは `latest/` にあります。新しい空の配置先へ復元してください。

```sh
hf buckets sync hf://buckets/RamRom/sam3-turing-native-20260922/latest ./sam3-latest
bash ./sam3-latest/recover-linux.sh cuda /absolute/path/to/sam3-deployment
```

スクリプトは SDK と store を取得し、SHA-256 とネイティブ reader の全 tensor CRC を検証します。CPU 版は第1引数を `cpu` にします。Linux 配布物は x86_64、GLIBC 2.38 / GLIBCXX 3.4.32 / CXXABI 1.3.11 以上を必要とし、CUDA 版には互換 GPU driver が必要です。すべての Linux ディストリビューションに共通のバイナリではありません。

復元テストでは CPU 168項目／CUDA 207項目の全ファイル・symlink を照合し、復元した CUDA SDK の C API で153ファイル、追加プローブで243ファイルの出力一致を確認しました。重み17ファイルもクラウドから個別に取得し直して SHA-256 が一致しています。復元スクリプト自体はスペースを含む配置先と既存 store を使う第3引数付きで実行し、各 SDK で全3,088 tensor reference の CRC 検査を通しました。

## アプリケーションから使う

C の公開入口は [`native/include/sam3/c_api.h`](../../native/include/sam3/c_api.h)。C++ は [`VideoPredictor`](../../native/include/sam3/video_predictor.h) などを使います。

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_application LANGUAGES C)
find_package(Sam3Native CONFIG REQUIRED COMPONENTS C)
add_executable(my_application main.c)
target_link_libraries(my_application PRIVATE sam3::c)
sam3_copy_runtime(my_application)
```

`CMAKE_PREFIX_PATH` に展開した SDK の絶対パスを渡します。C client のビルドに Torch ヘッダ、C++ compiler、CUDA Toolkit、Python は不要です。C++ client は `COMPONENTS CPP` / `sam3::cpp` と、対応する standalone LibTorch **2.10.0** のヘッダ・ABI が必要です。

API の使用例は [C API](C_API.md)、[owning predictor の C API](PREDICTOR_C_API.md)、[C++ video predictor](VIDEO_PREDICTOR.md)にあります。任意の UTF-8 text、正負 box、point、mask、object ID の追加・削除、前後 propagation、途中編集、fetch、reset、cancel を実行時に指定します。画像モードは `image_only=1`、動画モードは frame provider と frame 数を設定します。CLI の `*_probe` は回帰検証用で、全機能を操作する汎用 UI ではありません。

Linux で Python wheel の LibTorch が `LD_LIBRARY_PATH` に入っている場合は、配布 SDK と混在させずに起動します。SDK 同梱の最小 C client ソースは `share/sam3-native/examples/c-client/` にあります。

```sh
env -u LD_LIBRARY_PATH cmake -S /absolute/path/to/sdk/share/sam3-native/examples/c-client \
  -B build-c-client -DCMAKE_PREFIX_PATH=/absolute/path/to/sdk
env -u LD_LIBRARY_PATH cmake --build build-c-client
```

## 融合・メモリ最適化

Vision の LayerNorm・window 配置・精度変換、residual と次 block の入力準備、Q/K RoPE を融合しました。MLP は丸め位置を保つ GELU の in-place 実行、位置 encoding は軸ごとの計算で重複した三角関数と中間配列を減らします。これらは入力画像やプロンプトの再利用を前提にしません。

Blackwell、FP16、batch 1、全 head・全 position の Vision 単体を、融合前 `3a58138` と交互測定した結果です。各モデル3組、各プロセス3回 warmup 後10回測定しています。

| モデル | 融合前 → 現版 | 時間短縮 |
|---|---:|---:|
| SAM3 | 57.922 → 50.745 ms | 12.39% |
| SAM3.1 | 59.858 → 52.606 ms | 12.12% |

前処理・text・detector・tracker・重みロードを含まない値です。Turing の速度や動画全体の速度へ読み替えないでください。batch 2、FP32、メモリ量と検証範囲は [POSITION_FUSION.md](POSITION_FUSION.md)にあります。FP16 parameter residency、全 CRC 検査の高速化、lossless mask cache はそれぞれ [COMPUTE_STORAGE.md](COMPUTE_STORAGE.md)、[WEIGHT_CRC.md](WEIGHT_CRC.md)、[OUTPUT_CACHE_DESIGN.md](OUTPUT_CACHE_DESIGN.md)を参照してください。

## 確認済みの範囲と残る課題

- Linux official standalone LibTorch: CPU 27 / CUDA 51 の CTest。最終成果物の復元後も C client の実画像推論と loader を検査しています。native regression では実重みの feature 340ファイル、C/C++ owner 19ケース・1,940ファイルの一致を確認しています。これは元 Python 実装との完全一致やデータセット品質評価を意味しません。
- Windows / RTX 2060 Max-Q: ユーザーが `19f8bf9` + 互換性パッチで **41/41** を確認。パッチは取り込み済みです。その後の融合実装は別途 Windows/Turing で検証してください。ユーザーの独自 LayerNorm 試作を含む42件とは区別しています。
- 元 Python 参照との FP16 mask 差は未解消です。4 frame の検証で単 rank 30 pixels、2 rank 26 pixels、公式 Python wheel の比較で32 pixelsが残りました。BF16 の logical 2 rank 参照では対象11出力が一致しています。条件・adapter・元実装の修正は [VIDEO_COLLECTIVE_REFERENCE.md](VIDEO_COLLECTIVE_REFERENCE.md)を参照してください。
- 物理的に異なる GPU 間の検証、幅広い動画・品質評価、長時間 CPU 実行の安定性は残っています。論理 rank の検証を物理 multi-GPU の証明として扱っていません。

添付資料に基づく方式選定は [ONBOARDING.md](ONBOARDING.md)、SDK のビルド・依存関係は [SDK.md](SDK.md)、各段階の記録は [PROGRESS.md](PROGRESS.md)にあります。AOTInductor PT2 を主経路には採用していません。GitHub Actions は使用していません。
