# Native SAM3 building blocks

This is a Python-independent ATen C++/CUDA library, **not yet a SAM3 inference runtime**.
It implements little-endian mask packing/unpacking, chunked bilinear resize +
sigmoid + mask packing, stable score-ordered generic NMS with no detection count cap, 8-connected
component labeling/counts, and Euclidean distance transform. CPU and CUDA implementations use the same public C++ API in
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


Weight sharing inventory (development only, original checkpoints untouched):

```sh
python native/tools/inventory_weights.py --checkpoint sam3=/path/to/sam3.pt --checkpoint sam3.1=/path/to/sam3.1_multiplex.pt --output /private/path/inventory.json
```

Tensor identity includes dtype, shape and SHA256 of the contiguous logical bytes.
This is an inventory, not a deployable weight format. Components use equal-valued
nonzero pixels, 8-connectivity, root-index labels, and 64-bit labels/counts. EDT
returns float32 distances and uses upstream's finite `1e9` distance for masks
without any zero pixel. These operations preserve input shape (EDT: `[B,H,W]`,
components: `[B,H,W]` or `[B,1,H,W]`). GPU calls honor the current stream.

The native weight reader and full token-level VE text encoder are now available.
See [the store format](../docs/native/WEIGHTS.md). After exporting private weights,
a standalone C++ command accepts arbitrary token IDs:

```sh
build/native/sam3_text /private/native-weights-v1 sam3 cuda 49406 4629 49407
```

`sam3::TextEncoder` retains its loaded language module across calls and accepts
batched token tensors with variable sequence length up to the upstream context
of 32. It executes all 24 layers and returns padding mask, resized language
memory and input embeddings. Native Unicode/BPE tokenization is still pending;
this CLI is a token-level development probe, not a complete text-prompt product.
The Python `text_encode` test operator reloads weights per call; production C++
callers should retain the `TextEncoder` object for the needed lifetime.

Visual modules and RGB tensor preprocessing are available in `vision_encoder.h`
and `preprocess.h`. `VisionEncoder` returns the 32-layer trunk and all requested
SAM3 dual / SAM3.1 tri-neck features. Inputs are normalized `[B,3,1008,1008]`;
`preprocess_rgb` converts arbitrary-size RGB tensors using the upstream resize
and normalization. Move raw pixels to the execution device before preprocessing
to match that device's torchvision rounding.

```sh
build/native/sam3_vision /private/native-weights-v1 sam3.1 cuda fp16
```

This command uses synthetic pixels and prints feature shapes. It is a native
module probe, not complete segmentation. `fp32` and `fp16` use the unfused MLP
path; `bf16_reference` reproduces the original BF16-forcing MLP for comparison
on capable hardware and is not intended for Turing deployment. Vision calls
restore the caller's autocast state. The original model weights are not copied
into separate image/video variants or separate precision variants.

`GeometryEncoder` in `geometry_encoder.h` implements the detector's configured
point/box prompt encoder, including image pooling, label/position embeddings,
right-padded sequence concatenation, CLS and all three transformer layers.
Its input tensors preserve arbitrary prompt counts, positive/negative labels,
per-image padding and empty prompt sequences. This is distinct from the tracker
and interactive mask-prompt path, which remain to be ported.

```sh
build/native/sam3_geometry /private/native-weights-v1 sam3.1 cuda fp16
python native/tests/roi_align_parity.py build/native/libsam3_native.so --report /tmp/roi.json
python native/tests/geometry_parity.py build/native/libsam3_native.so /private/native-weights-v1 --checkpoint sam3=/private/sam3.pt --report /tmp/geometry.json
```

The geometry executable is an empty/mixed-prompt module probe using synthetic
features; applications supply their own feature/prompt tensors through C++.
The native ROIAlign kernel supports CPU/CUDA float32/float64/float16 and follows
torchvision's autocast policy. Its sampling implementation is adapted under
the [torchvision BSD license](third_party/torchvision/LICENSE); neither the
Python torchvision package nor its compiled library is required at runtime.
