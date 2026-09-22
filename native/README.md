# Native setup probe

This is an ATen CPU/CUDA matrix-product smoke test, **not a SAM3 inference runtime**.
See [the onboarding report](../docs/native/ONBOARDING.md) for constraints, evidence,
remaining work, private artifact locations, and restoration instructions.

Provide an installed LibTorch distribution (or a compatible development PyTorch
installation exposing TorchConfig.cmake):

```sh
cmake -S native -B build/native -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch -DCMAKE_BUILD_TYPE=Release -DSAM3_TEST_CUDA=ON
cmake --build build/native --config Release --parallel 2
ctest --test-dir build/native -C Release --output-on-failure
```

Use `-DSAM3_TEST_CUDA=OFF` for CPU-only machines. Windows requires matching MSVC
and LibTorch Release/Debug configurations. Tests fail on unavailable requested
devices rather than silently falling back. No Python command is used by this
CMake project or executable. Standalone deployment and Windows remain untested.
