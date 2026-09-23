# Windows / Turing compatibility

The user supplied a compatibility patch and reported successful native execution
on an RTX 2060 Max-Q (6 GB, compute capability 7.5). Their **41/41 passing tests**
apply to commit `19f8bf9c8ee9dc37ea080a61d48d2bfbded378d6` plus this patch.
This is user-run physical Windows/Turing evidence, distinct from the Linux
Blackwell checks performed in this workspace. Later optimization commits still
need their own Windows/Turing regression runs. The user's separate 42-test
LayerNorm experiment is not included in this patch or counted here.

## Imported changes

- Split generated tokenizer literals every 512 source bytes, without regenerating
  Unicode data. All 13 string byte sequences and the remaining header text were
  checked against the previous source and the supplied SHA-256 manifest.
- Split the two exported `MultiplexState` constexpr declarations for MSVC C2487.
- Export `VideoMetadata::object_ids()` for external C++ DLL consumers.
- Match the observed Windows LibTorch complex-product imaginary-part rounding
  with `fma(a,d,round(b*c))`. Linux keeps its existing expression. The regression
  computes its expectation using the local LibTorch; no tolerance was relaxed.
- Pin ICU 74.2#6 and zlib through a vcpkg baseline and a Release x64/v143/14.44
  dynamic-runtime triplet.

The original patch SHA-256 is
`3fd1b6bc667b21d2ce408d91ba02d73ac20ed00d7a8b8fed6781a5bce4472faf`.
All eight imported file blobs match that patch. The supplied manifest and the
local import verification are retained alongside the private development evidence.

## User-tested configuration

| Component | Reported version |
|---|---|
| GPU | RTX 2060 Max-Q 6 GB, sm75 |
| Driver | 610.88 |
| CUDA Toolkit / nvcc | 13.0 / 13.0.88 |
| MSVC | 19.44.35228, toolset 14.44 / v143 |
| CMake / generator | 4.4.3 / Ninja |
| LibTorch | standalone Release 2.10.0+cu130 |
| ICU / zlib | 74.2#6 / 1.3.2#2 |

The user ran with only Windows system directories on PATH and native dependency
DLLs beside the executable. Python is not a runtime dependency. Optional
FFmpeg/JPEG/OpenCV integrations were disabled for these checks.

## Build

In an x64 Developer Command Prompt, select the supported toolset with
`vcvars64.bat -vcvars_ver=14.44`. A Visual Studio 2026 Build Tools installation can
contain this toolset; the user's default MSVC 19.51 was rejected by CUDA 13.0.
No `--allow-unsupported-compiler` bypass was used.

Set `VCPKG_ROOT` to bootstrapped vcpkg and `LIBTORCH_ROOT` to the extracted
standalone `libtorch` directory. Run from the repository root:

```bat
"%VCPKG_ROOT%\vcpkg.exe" install --x-manifest-root=native --x-install-root=build/vcpkg_installed --overlay-triplets=native/cmake/triplets --triplet=x64-windows-sam3 --host-triplet=x64-windows-sam3
set "TORCH_CUDA_ARCH_LIST=7.5"
cmake -S native -B build/native-windows-cu130 -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=%LIBTORCH_ROOT%" "-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake" "-DVCPKG_INSTALLED_DIR=%CD%/build/vcpkg_installed" "-DVCPKG_OVERLAY_TRIPLETS=%CD%/native/cmake/triplets" -DVCPKG_TARGET_TRIPLET=x64-windows-sam3 -DVCPKG_HOST_TRIPLET=x64-windows-sam3 -DVCPKG_MANIFEST_INSTALL=OFF -DSAM3_WITH_CUDA=ON -DSAM3_TEST_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/native-windows-cu130 --parallel 4
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
ctest --test-dir build/native-windows-cu130 --output-on-failure
```

Both architecture settings are intentional: `TORCH_CUDA_ARCH_LIST` also controls
LibTorch's CMake CUDA flags. Matching host and target triplets avoids building
ICU host tools with a different compiler. If an earlier CMake cache selected
Windows SDK ICU, add `-U "ICU_*"` when configuring and verify that both ICU
headers and libraries resolve under vcpkg.

The user changed a local vcpkg MSYS2 download mirror after a stalled download,
while retaining archive hash checks. That environment-specific helper change is
not part of the repository patch.

The explicit RoPE rounding rule was established for this LibTorch build and
hardware. Re-run the exact rotary and fusion tests when changing the compiler,
LibTorch distribution, CUDA version, or GPU.
