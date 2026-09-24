# Windows CUDA build

Tested on Windows 11 with an RTX 2060 Max-Q 6 GB (sm_75). The same CMake project
builds on Linux ([SDK.md](SDK.md)); Linux SDK binaries are not Windows binaries.

## Toolchain

| Component | Version |
|---|---|
| GPU / driver | RTX 2060 Max-Q 6 GB, sm_75 / 610.88 |
| CUDA Toolkit / nvcc | 13.0 / 13.0.88 |
| MSVC | 19.44.35228, toolset 14.44 (v143) |
| CMake / generator | 4.4.3 / Ninja |
| LibTorch | official standalone Release 2.10.0+cu130 |
| ICU / zlib (vcpkg) | 74.2#6 / 1.3.2#2 |

CUDA 13.0 rejects MSVC 19.51: select toolset 14.44 explicitly with
`vcvars64.bat -vcvars_ver=14.44`, even inside Visual Studio 2026 Build Tools.
Do not use `--allow-unsupported-compiler`.

The vcpkg manifest pins ICU 74.2#6, the Unicode 15 profile the tokenizer was
validated against. The `x64-windows-sam3` overlay triplet (dynamic CRT, dynamic
libraries, MSVC 14.44, Release) is used for both host and target so ICU host
tools are not built with a second compiler.

## Build and test

From an x64 developer prompt at the repository root, with `VCPKG_ROOT` pointing
to a bootstrapped vcpkg and `LIBTORCH_ROOT` to the extracted `libtorch`:

```bat
"%VCPKG_ROOT%\vcpkg.exe" install --x-manifest-root=native --x-install-root=build/vcpkg_installed --overlay-triplets=native/cmake/triplets --triplet=x64-windows-sam3 --host-triplet=x64-windows-sam3
set "TORCH_CUDA_ARCH_LIST=7.5"
cmake -S native -B build/native-windows-cu130 -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=%LIBTORCH_ROOT%" "-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake" "-DVCPKG_INSTALLED_DIR=%CD%/build/vcpkg_installed" "-DVCPKG_OVERLAY_TRIPLETS=%CD%/native/cmake/triplets" -DVCPKG_TARGET_TRIPLET=x64-windows-sam3 -DVCPKG_HOST_TRIPLET=x64-windows-sam3 -DVCPKG_MANIFEST_INSTALL=OFF -DSAM3_WITH_CUDA=ON -DSAM3_TEST_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/native-windows-cu130 --parallel 4
set "PATH=C:\Windows\System32;C:\Windows"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
ctest --test-dir build/native-windows-cu130 --output-on-failure
```

Both architecture settings are needed: `TORCH_CUDA_ARCH_LIST` also constrains
LibTorch's generated CUDA flags. Development tools copy the LibTorch DLLs beside
the executables, so tests run with only the Windows system directories on
`PATH`; no Python process is involved. Optional FFmpeg/libjpeg media and OpenCV
preprocessing stay disabled in this configuration ([MEDIA_IO.md](MEDIA_IO.md),
[VIDEO_PREPROCESS.md](VIDEO_PREPROCESS.md)).

The opt-in research experiments need external sources and are OFF by default:
`-DSAM3_EXPERIMENT_KITCHEN_ATTENTION=ON -DSAM3_KITCHEN_ROOT=...` (ComfyKitchen
INT8 attention) and `-DSAM3_EXPERIMENT_INT8_GEMM=ON -DSAM3_CUTLASS_ROOT=...`
(CUTLASS INT4 FC2). See [native/README.md](../../native/README.md#experiment-switches).

Troubleshooting:

- If CMake was configured before ICU was installed, add `-U "ICU_*"` when
  configuring again. Otherwise Windows SDK ICU import libraries can stay cached
  next to vcpkg's ICU 74 headers and cause unresolved symbols.
- An unresponsive MSYS2 mirror stalled the first dependency download. Pointing
  the local vcpkg download helper at `repo.msys2.org` fixed it with archive hash
  checks still enabled; this is a local environment change, not part of the
  repository.

## Windows-specific source changes

These changes came from a user-supplied compatibility patch (SHA-256
`3fd1b6bc667b21d2ce408d91ba02d73ac20ed00d7a8b8fed6781a5bce4472faf`, imported
in `c5e9dc6`):

- Generated tokenizer string literals are split every 512 source bytes to stay
  below MSVC's literal limit. All 13 frozen tables keep their byte content; the
  Unicode data was not regenerated.
- The two exported `MultiplexState` `constexpr` declarations are separated
  (MSVC C2487).
- `VideoMetadata::object_ids()` is exported for external C++ DLL consumers.
- The RoPE kernels evaluate the complex product's imaginary component as
  `fma(a,d,round(b*c))` on Windows, matching standalone LibTorch 2.10.0+cu130
  there; Linux keeps `fma(b,c,round(a*d))`. The rotary tests compute their
  expectation with the local LibTorch, so no tolerance was relaxed. Re-run the
  exact rotary and fusion tests after changing compiler, LibTorch, CUDA or GPU.

## Validation status

| Revision | Result |
|---|---|
| `19f8bf9` + compatibility patch | 41/41 CTest, run by the user |
| `908e476` (fusions imported) | 48/48 CTest; truck outputs byte-identical to the earlier native run |
| `ba018a0` + Windows experiments | 51/51 → 54/54 CTest after the maintenance refactor |
| `7d7fddb` (research) and after the cleanup | 62/62 CTest ([cleanup-parity.json](../../experiments/results/native_rtx2060/cleanup-parity.json)) |

`sam3_native.dll` contains sm_75 cubins and has no Python DLL dependency. The
image path has been measured on this GPU
([experiments/results/native_rtx2060](../../experiments/results/native_rtx2060/README.md));
video on Windows/Turing and broad quality evaluation have not been run.
