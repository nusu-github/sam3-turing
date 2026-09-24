# Standalone LibTorch SDK (development release 0.1.0)

The native library installs as a relocatable CMake package with C and C++
interfaces. Weights stay in the external shared store ([WEIGHTS.md](WEIGHTS.md));
image/video and SAM3/SAM3.1 share one SDK and one store. Recovery of the
published private release is described in [QUICKSTART_JA.md](QUICKSTART_JA.md).

The SDK builds against the official standalone LibTorch distribution, pinned to
**LibTorch 2.10.0 CPU and CUDA 13.0** (archive URLs and hashes:
[sdk-libtorch-sources.json](evidence/sdk-libtorch-sources.json)). The
[LibTorch installation guide](https://docs.pytorch.org/cppdocs/installing.html)
describes the Windows DLL layout and release/debug ABI rules this follows.
Do not mix Python-wheel LibTorch libraries into the build's loader or linker
search paths.

## Build and install

Requirements: standalone LibTorch, ICU 72–74 (validated with 74.2), zlib, a
C++17 toolchain, and a matching CUDA toolkit when custom CUDA kernels are enabled.

```sh
cmake -S native -B build/sdk -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch \
  -DSAM3_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='75;120' \
  -DSAM3_BUILD_DEVELOPMENT_TOOLS=OFF
cmake --build build/sdk --parallel 4
cmake --install build/sdk --prefix /absolute/path/to/sdk
cmake -DSAM3_SDK_ROOT=/absolute/path/to/sdk \
  -DSAM3_TORCH_ROOT=/absolute/path/to/libtorch \
  -DSAM3_RUNTIME_NOTICES=/absolute/path/to/notices \
  -P native/cmake/BundleRuntime.cmake
```

For a CPU SDK use CPU-only LibTorch and `SAM3_WITH_CUDA=OFF`.
`SAM3_BUILD_DEVELOPMENT_TOOLS=OFF` leaves regression executables and tests out
of the build graph. The install contains the library, headers, CMake metadata,
the BPE vocabulary, licenses and a C client example; it does not contain
development probes or weights.

Windows uses the same project and bundling script with Windows paths, the
matching LibTorch/ICU/zlib packages and `--config Release`; see
[WINDOWS_BUILD.md](WINDOWS_BUILD.md). `SAM3_RUNTIME_SEARCH_DIRS` adds DLL
directories when needed.

`BundleRuntime.cmake` follows the shared-library dependency closure, including
dynamically loaded cuDNN/NVRTC/linalg plugins. It rejects Python and unexpected
development libraries, checks that core Torch libraries come from the supplied
distribution, accepts duplicate copies only when their SHA-256 values match, and
records file hashes. It excludes OS C/C++ runtimes and the NVIDIA driver. Pass
`SAM3_MEDIA_ROOT` / `SAM3_OPENCV_ROOT` to stage optional FFmpeg and OpenCV
runtimes ([MEDIA_IO.md](MEDIA_IO.md), [VIDEO_PREPROCESS.md](VIDEO_PREPROCESS.md)).
Linux defaults to `lib`; use `SAM3_LIBDIR=lib64` when that is the install
directory. Keep the upstream license/notice material with the runtime.

## C and C++ clients

```cmake
project(my_application LANGUAGES C)
find_package(Sam3Native CONFIG REQUIRED COMPONENTS C)
add_executable(my_application main.c)
target_link_libraries(my_application PRIVATE sam3::c)
sam3_copy_runtime(my_application) # stages bundled DLLs on Windows
```

A C consumer needs no Torch headers, C++ compiler, CUDA toolkit or Python; it
still needs the runtime libraries. C++ consumers request `COMPONENTS CPP` and
link `sam3::cpp`, which finds the matching LibTorch; put the standalone LibTorch
prefix before the SDK prefix in `CMAKE_PREFIX_PATH`.

The installed example `share/sam3-native/examples/c-client` builds as C11 and
can be installed into the SDK's `bin`. Without arguments it checks the ABI and
defaults; with `WEIGHTS VOCABULARY TEXT` it runs the text encoder. Linux builds
use relative runtime paths; Windows stages DLLs beside the executable.

On Linux, clear an `LD_LIBRARY_PATH` inherited from a Python environment
(`env -u LD_LIBRARY_PATH ...`): it made a consumer link against an older Torch
library. Configure consumers in a fresh build directory after moving an SDK, as
cached absolute Torch paths otherwise go stale.

Development tools can also be built against an installed SDK through
`native/eval` (see [IMAGE_PRECISION_AUDIT.md](IMAGE_PRECISION_AUDIT.md#reproduction)).

## Validation

- Official CPU LibTorch: all CPU CTests pass; official CUDA LibTorch: all CUDA
  CTests pass (27 / 51 at the `1ee3440` release).
- A C11 client of the CUDA SDK reproduces the retained C++ outputs exactly
  (10 image and 26 video checkpoints, including emission timing), and a client
  linked only to a moved CUDA SDK runs the full SAM3.1 image/edit lifecycle.
  Loader tracing shows all core and model runtime libraries come from the SDK.
- A C client in a moved CPU SDK runs the text encoder with `PATH=/nonexistent`
  and inside a chroot without Python.
- The recovered release (see [QUICKSTART_JA.md](QUICKSTART_JA.md)) verifies
  every file hash and symlink and reruns the C API checks.

The Linux binaries were built with GCC 13.3 and require GLIBC 2.38 /
GLIBCXX 3.4.32 / CXXABI 1.3.11; they are not universal Linux binaries. Per-library
requirements and sizes: [sdk-validation.json](evidence/sdk-validation.json).
