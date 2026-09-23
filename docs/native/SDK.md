# Standalone LibTorch SDK (development release 0.1.0)

The native library now installs as a relocatable CMake package with C and C++
interfaces. Weights stay in the external shared modular store. Image/video and
SAM3/SAM3.1 do not get duplicated weight packages.

The attached `cppdocs/installing.md` describes standalone LibTorch archives,
CMake use, Windows DLL copying, and release/debug ABI compatibility. This is the
platform basis for this route; it does not require a Linux-only model exporter.
The [official LibTorch installation guide](https://docs.pytorch.org/cppdocs/installing.html)
and [download selector](https://pytorch.org/get-started/locally/) provide the
upstream distribution route. This SDK validation pins **LibTorch 2.10.0 CPU and
CUDA 13.0**; it does not substitute an unrecorded latest build. Linux archive URLs,
hashes and build hashes are recorded in `sdk-libtorch-sources.json`. The matching
Windows CUDA archive responded successfully but was not downloaded or executed.

The prior NVIDIA development build remains useful as the original comparison
reference. Its MPI/UCX/external-MKL dependency tree is not used for these SDK builds.
The official CPU runtime has no such dynamic dependencies. Linux CUDA packages
still have their platform-specific LibTorch dependencies; the inference API is
ATen/CUDA, not an adoption of a Linux-only distributed communication API.

## Build and install

Use a standalone LibTorch archive and ICU 72–74 (validated here with 74.2), plus
zlib. Building the library needs a C++17 toolchain and a matching CUDA toolkit
when custom CUDA kernels are enabled. Do not mix Python-wheel LibTorch binaries
into this build's loader/linker search paths.

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

Windows uses the same CMake project and `BundleRuntime.cmake`, with Windows paths
and the matching Windows LibTorch/ICU/zlib packages. Use the Visual Studio developer
shell and `cmake --build ... --config Release` / `cmake --install ... --config Release`.
Keep debug and release C++ dependencies consistent, as the attached guide requires.
`SAM3_RUNTIME_SEARCH_DIRS` supplies additional DLL directories when necessary.
These Windows instructions are build plumbing, **not** a Windows execution result.

For a true CPU SDK use CPU-only LibTorch and `SAM3_WITH_CUDA=OFF`. This differs
from the earlier custom-CUDA-disabled build that still linked CUDA-capable Torch.
`SAM3_BUILD_DEVELOPMENT_TOOLS=OFF` excludes regression executables/tests from the
build graph. The SDK install itself contains the library, headers, CMake metadata,
vocabulary, license, and a C client example; it does not install the development
probes or model weights.

Runtime bundling follows the shared-library dependency closure and includes
cuDNN/NVRTC/linalg plugins that can be loaded dynamically. It rejects Python and
unexpected development libraries, checks that core Torch libraries originate
from the supplied distribution, and permits identical already-bundled copies.
It records file hashes and can copy supplied upstream notices. Preserve matching
upstream license/notice material with runtime distributions. The bundle excludes
OS C/C++ runtimes and the NVIDIA driver. Linux defaults to `lib`; use
`SAM3_LIBDIR=lib64` when that is the SDK's installation directory.

## C and C++ clients

A C-only application can use:

```cmake
project(my_application LANGUAGES C)
find_package(Sam3Native CONFIG REQUIRED COMPONENTS C)
add_executable(my_application main.c)
target_link_libraries(my_application PRIVATE sam3::c)
sam3_copy_runtime(my_application) # stages bundled DLLs on Windows
```

No Torch headers, C++ compiler, CUDA toolkit or Python discovery occurs for this
consumer configuration. Runtime libraries are still required. For direct C++
ATen-facing APIs request `COMPONENTS CPP` and link `sam3::cpp`; that component
finds the matching LibTorch version and needs its headers/toolchain ABI. Put the
standalone LibTorch prefix before the SDK prefix in `CMAKE_PREFIX_PATH` for C++
consumers, so Torch's own library searches consistently use that distribution.

The installed example lives at `share/sam3-native/examples/c-client`. It can be
built as C11, installed into the SDK's `bin`, and moved with the SDK. With no
arguments it checks ABI/defaults; with `WEIGHTS VOCABULARY TEXT` it executes the
real text encoder and checks finite output features. The Linux installed example
and native library use relative runtime paths; Windows stages DLLs beside the
executable. Consumers with their own layout must preserve that runtime relationship.

An incompatible `LD_LIBRARY_PATH` from the original Python environment caused a
consumer link to select its old Torch CUDA library even for this CPU SDK. Clear
that environment variable for these SDK builds/tests (`env -u LD_LIBRARY_PATH ...`
on Linux). The SDK's linker search hint is not a promise that arbitrary mixed
LibTorch versions in one environment/process are safe. The successful isolated build is retained. When moving an SDK, configure
consumers in a fresh build directory: cached absolute Torch library paths from
a previous location otherwise remain stale.

## Evidence and remaining scope

The following checks describe the initial SDK. Later component and SDK overlays
have their own evidence linked below.

- Official CPU LibTorch: all 16 CTests pass. Official CUDA LibTorch: all 28 pass.
- Official CUDA C11 client: all 10 image and 26 video checkpoints match the
  retained development-reference C++ outputs exactly, including video timing.
  Existing original-reference settings/adapters remain applicable.
- C and C++ consumers build from installed metadata. The C consumer discovers
  only the C compiler; the C++ consumer executes an ATen-facing mask operation.
- A C client installed into a moved CPU SDK executes the text encoder with
  `PATH=/nonexistent`. A separate Linux chroot contains no Python files and also
  executes that encoder. This is an isolation test, not a deployment requirement.
  The chroot lacks `/proc/cpuinfo`, so cpuinfo reports a diagnostic but inference
  completes; this test does not measure representative CPU performance.
- An external C11 client linked only to the moved CUDA SDK executes the full
  SAM3.1 image/edit lifecycle. All 10 outputs match. ELF loader tracing confirms
  its 23 loaded core/model-runtime libraries come from the SDK, not the original
  build or Python environment.

These Linux binaries were built on this host with GCC 13.3; their actual OS/ABI
requirements include GLIBC 2.38 and GLIBCXX 3.4.32; runtime file sizes and
per-library required symbols are recorded in `sdk-validation.json`. They
are not universal Linux binaries. Windows/Turing physical tests remain with the
user. sm75 code presence is build evidence only. No GitHub Actions were used.
The original CPU instability has not been diagnosed by this environment change;
finite official-runtime tests do not establish long-run CPU stability.

Native local media support and modular SDK overlays are described in
[MEDIA_IO.md](MEDIA_IO.md), random access in [MEDIA_SEEK.md](MEDIA_SEEK.md),
source input transforms in [VIDEO_PREPROCESS.md](VIDEO_PREPROCESS.md), and
tracking ranks in [VIDEO_MULTIDEVICE.md](VIDEO_MULTIDEVICE.md) and
[VIDEO_PARALLEL.md](VIDEO_PARALLEL.md). Broader quality/performance, codec coverage,
source FP16 precision and platform validation remain open.
The previously unresolved 120-pixel reverse-edit difference was subsequently
traced to a stale original pointer; see [REVERSE_EDIT_POINTER.md](REVERSE_EDIT_POINTER.md). This SDK is a tested development distribution, not a full-function
completion claim.

The lossless displayed-mask cache adds C/C++ configuration and statistics APIs;
see [OUTPUT_CACHE_DESIGN.md](OUTPUT_CACHE_DESIGN.md). Its CPU/CUDA SDK increment is
`output-cache-sdk-overlay/overlays.json` in the existing private bucket. Apply it
at the matching preceding SDK root described by `rotary-sdk-overlay/overlays.json`.
The recipe verifies its prerequisite hash and lists changed-file hashes; all
dependency and shared-weight layers are reused. Validation details are in
`output-cache-validation.json`.

The next increment, `cache-fetch-sdk-overlay/overlays.json`, is based on that
output-cache SDK and removes repeated packed-cache writes during reads. See
[CACHE_FETCH.md](CACHE_FETCH.md) for exact regression and measurement scope.
