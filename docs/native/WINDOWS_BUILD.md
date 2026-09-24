# Windows CUDA build

Use the x64 MSVC 14.44 (VS 2022 / v143) toolset with CUDA Toolkit 13.0 and
the official **Release LibTorch 2.10.0+cu130** Windows distribution. CUDA 13.0
rejects MSVC 19.51; select 14.44 explicitly even when it is installed inside
Visual Studio 2026 Build Tools. Start an x64 developer command prompt with
`vcvars64.bat -vcvars_ver=14.44`.

The native vcpkg manifest pins ICU 74.2#6 for the tokenizer's Unicode profile.
The `x64-windows-sam3` overlay triplet uses the dynamic CRT and dynamic libraries,
MSVC 14.44, and Release builds. Use it for both host and target to avoid building
a second ICU tool package with another compiler.

From the repository root, with `VCPKG_ROOT` pointing to a bootstrapped vcpkg
checkout and `LIBTORCH_ROOT` to the extracted `libtorch` directory:

```bat
"%VCPKG_ROOT%\vcpkg.exe" install --x-manifest-root=native --x-install-root=build/vcpkg_installed --overlay-triplets=native/cmake/triplets --triplet=x64-windows-sam3 --host-triplet=x64-windows-sam3
set "TORCH_CUDA_ARCH_LIST=7.5"
cmake -S native -B build/native-windows-cu130 -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=%LIBTORCH_ROOT%" "-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake" "-DVCPKG_INSTALLED_DIR=%CD%/build/vcpkg_installed" "-DVCPKG_OVERLAY_TRIPLETS=%CD%/native/cmake/triplets" -DVCPKG_TARGET_TRIPLET=x64-windows-sam3 -DVCPKG_HOST_TRIPLET=x64-windows-sam3 -DVCPKG_MANIFEST_INSTALL=OFF -DSAM3_WITH_CUDA=ON -DSAM3_TEST_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/native-windows-cu130 --parallel 4
ctest --test-dir build/native-windows-cu130 --output-on-failure
```

Ninja, CMake, the chosen MSVC compiler and CUDA `nvcc` must be available in that
prompt. `TORCH_CUDA_ARCH_LIST` also constrains LibTorch's generated CUDA flags;
`CMAKE_CUDA_ARCHITECTURES` alone does not constrain those flags.

This configuration leaves optional FFmpeg/JPEG media decoding and OpenCV
preprocessing disabled. These require additional development dependencies.
The tests above do not establish actual-model accuracy or performance.

## RTX 2060 local validation, 2026-09-23

Built from `fe87ffb` plus the Windows compatibility changes in this working tree:

- Split generated tokenizer string literals into adjacent chunks below MSVC's
  literal limit. All 13 frozen string tables retain their original byte content;
  the Unicode/regex tables were not regenerated from a different environment.
- Separate the exported multiplex class's two `constexpr` declarations to avoid
  MSVC C2487.
- Export `VideoMetadata::object_ids()` so Windows C++ clients can link it.

Toolchain: CMake 4.4.3, Ninja, MSVC 19.44.35228, CUDA compiler 13.0.88,
standalone Release LibTorch 2.10.0+cu130, vcpkg ICU 74.2#6 and zlib 1.3.2#2.
The full default development-tool build succeeds. `sam3_native.dll` contains
five `sm_75` cubins and has no direct Python DLL dependency.

On an RTX 2060 Max-Q 6 GB (driver 610.88), CTest passes **40/40** tests:
all 22 CPU/non-CUDA checks and all 18 CUDA checks. Test processes used only
Windows system directories on PATH, with OMP/MKL thread counts set to four and
the native dependencies beside the executables. No Python process was needed.

Initially, `rotary_cuda` and `rotary_stream_cuda` failed the bitwise comparison
at `dtype=Float shape=[1, 16, 576, 64] layout=0`. A standalone diagnostic against
the same LibTorch binary compared one million random complex products. The
ordinary c10 expression matched the real component but differed on 330,340
imaginary components (maximum absolute difference 9.53674e-7). Explicitly
evaluating the imaginary component as `fma(a,d,round(b*c))` matched every result;
disabling fusion did not match the reference either.

The Windows fused rotary kernel now uses `__fmul_rn` followed by `__fmaf_rn` for
that imaginary component, preserving the reference's rounding order. It retains
one CUDA kernel and the existing layout handling. The non-Windows expression is
unchanged. The rotary tests pass 45 exact cases per device/stream, including a
new cancellation regression, all three dtypes, strided/broadcast inputs, special
values and fallbacks, plus two invalid-input checks. Tolerances are unchanged.
This establishes parity on the tested Windows/LibTorch/Turing configuration;
other platform/GPU combinations were not rerun in this local check.
Actual-model image inference was measured subsequently; see the
[initial RTX 2060 performance report](../../experiments/results/native_rtx2060/README.md).
Video and broader quality validation remain open.

If CMake was configured before ICU was installed, clear its `ICU_*` cache entries
when configuring again (`cmake -U "ICU_*" ...`). Otherwise Windows SDK ICU import
libraries can remain cached alongside vcpkg's ICU 74 headers and cause unresolved
symbols. Both ICU libraries must resolve beneath the selected vcpkg triplet.

The first dependency download encountered an unresponsive MSYS2 mirror. Using
the official `repo.msys2.org` source in the local vcpkg download helper resolved
it; vcpkg's archive hash checks remained enabled. This local mirror preference
is outside the project and does not change the pinned dependency source versions.
