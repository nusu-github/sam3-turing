# Native SAM3 building blocks

This is a Python-independent ATen C++/CUDA library, **not yet a SAM3 inference runtime**.
It implements little-endian mask packing/unpacking, chunked bilinear resize +
sigmoid + mask packing, and stable score-ordered generic NMS with no detection
count cap. CPU and CUDA implementations use the same public C++ API in
`include/sam3/ops.h`. Development-only dispatcher registration also permits
reference tests through `torch.ops.load_library`, without linking Python.

See [the onboarding report](../docs/native/ONBOARDING.md) and
[progress](../docs/native/PROGRESS.md) for constraints and remaining work.

Provide LibTorch (or compatible development PyTorch exposing TorchConfig.cmake):

```sh
cmake -S native -B build/native -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch -DCMAKE_BUILD_TYPE=Release -DSAM3_WITH_CUDA=ON -DSAM3_TEST_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=75;120"
cmake --build build/native --config Release --parallel 2
ctest --test-dir build/native -C Release --output-on-failure
```

Select architectures supported by your CUDA toolkit; 75 is the Turing target,
120 is used on the present Blackwell development GPU. Torch's own CMake package
may add additional architectures. GPU execution tests require real hardware for
the architecture being tested. No Turing runtime validation has been performed.

Use both `-DSAM3_WITH_CUDA=OFF -DSAM3_TEST_CUDA=OFF` for CPU-only builds. Windows
requires matching MSVC and LibTorch Release/Debug configurations. Tests fail on
unavailable requested devices rather than silently falling back. No Python
command is used by this CMake project or the C++ executables.

Development-only extensive reference comparisons (Python is allowed here):

```sh
python native/tests/parity.py build/native/libsam3_native.so --cuda
```

The C++ ABI currently follows LibTorch and is not a stable C ABI. An application
must distribute matching LibTorch/CUDA libraries, and this directory is not yet
a relocatable runtime package. Generic NMS materializes a boolean N-by-N matrix;
this is a correctness baseline with optimization still pending. Resize preserves
ATen's dtype and sigmoid rounding, and bounds temporaries by `chunk_size` without
limiting output count.
