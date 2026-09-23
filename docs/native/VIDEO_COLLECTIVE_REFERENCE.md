# Source replay for logical tracking ranks

The SAM3 BF16 two-rank outputs from `080038b` match the original predictor when
its neural sessions are partitioned into the same logical ranks: all 11 semantic
lifecycle outputs agree exactly, including IDs, probabilities, boxes, binary
masks and applicable emission timing. FP16 remains non-exact. This investigation
changes reference tests only; the deployable C/C++ SDK is unchanged.

The earlier [owning-device validation](VIDEO_MULTIDEVICE.md) observed 133 BF16
and 26 FP16 changed mask pixels between native one-rank and two-rank execution.
Collection promotion alone does not explain them. On this fixture original local
masks are already FP32, and adding an empty peer to original collection leaves
all saved reference arrays unchanged in both precisions. Partitioning original
neural sessions reproduces the BF16 rank-dependent result. Session partitioning
and global object order change together here; their individual effects have not
been isolated.

## Measurements

The four-frame fixture, weights and native outputs are reused from `080038b`.
All comparisons below check 11 outputs. IDs, probabilities, boxes and emission
timing agree exactly in every row; the counts refer to binary-mask pixels.

| Precision | Original reference execution | Native execution | Different pixels | Different outputs |
| --- | --- | --- | ---: | ---: |
| BF16 | Single rank | Single rank | 0 | 0 |
| BF16 | Single neural rank plus empty peer | Two ranks | 133 | 5 |
| BF16 | Partitioned neural sessions, two ranks | Two ranks | 0 | 0 |
| FP16 | Single rank | Single rank | 30 | 4 |
| FP16 | Single neural rank plus empty peer | Two ranks | 32 | 5 |
| FP16 | Partitioned neural sessions, two ranks | Two ranks | 26 | 4 |
| FP16 | Official Python 2.10.0+cu130, single rank | Single rank | 32 | 4 |

The first six rows use NVIDIA Python Torch `2.10.0a0+b558c986e8.nv25.11`;
native execution uses standalone LibTorch 2.10.0 cu130. The final row uses the
official Python wheel with torchvision 0.25.0+cu130 and NumPy 1.26.4. Its historical
evidence filename is `fp16-matched-control`, but loader tracing showed it loaded
the wheel's core libraries, whose hashes differ from standalone LibTorch. It is
**not a same-binary runtime comparison**. The isolated diagnostic loads cuBLAS,
cuDNN and cudart through standalone SDK aliases; cuSPARSE and nvrtc-builtins come
from system CUDA, and torchvision also loads its packaged cudart. Attempting to substitute the standalone
core failed at import due to a missing NVSHMEM binding symbol; the standalone
Python wrapper also failed under CPython 3.12 due to `_PyCode_SetExtra`. Neither
failed import produced inference results. Existing reference environments and
all deployed SDKs were retained.

FP16 uses the existing test-only vision-MLP adapter to match native FP16 policy;
the original model normally forces those MLP operations to FP32. TF32 is disabled
after model construction. No tolerance converts the residual differences into a
passing exact comparison. The source-versus-native FP16 issue remains open.

## Adapter scope and reproduction

`native/tests/video_collective_reference.py` provides two diagnostics:

- `install_empty_peer_replay` runs original local propagation unchanged, then
  executes the original collection method's bytecode with one populated rank
  and empty peers. A copied globals dictionary supplies synchronous tensor
  copies for `all_gather`. No global `torch.distributed` method is patched.
- `install_partitioned_rank_replay` executes original propagation and
  birth/removal separately on actual rank-local sessions, serially on CUDA 0.
  Original rank-zero planning sees all rank metadata. Original global memory
  update visits rank-concatenated sessions, preserving global visibility before
  local memory encoding. Process dispatch and transport are adapted explicitly.

The detector runs once. No neural phase is mocked. The partitioned adapter
covers this semantic add/replace, full propagation, fetch and reset fixture;
instance-edit distributed paths are not adapted. This is not unmodified process
orchestration or physical multi-GPU validation.

Run the reference harness from a development environment with original weights:

```bash
python native/tests/video_predictor_parity.py \
  --model sam3 --checkpoint /path/to/sam3.pt --mode bf16_reference \
  --collective-partition-ranks 2 \
  --native-output /path/to/two-ranks-sam3-bf16_reference \
  --reference-output /path/to/new-reference \
  --frames /path/to/0.ppm /path/to/1.ppm /path/to/2.ppm /path/to/3.ppm \
  --report /path/to/comparison.json --require-exact
```

Omit the collective option and select single-rank native output for the control;
use `--collective-empty-peers 1` for collection-only replay. Select `--mode fp16`
for that diagnostic; its exact check currently fails. Saved reference tensors can
be rechecked with `--reference-cache`, retaining the matching precision and
collective flags. Python is needed only for this original-model reference test,
not for deployed inference.

Three integrity tests pass, including 32 CPU/CUDA tensor subcases spanning
FP16/BF16/FP32/FP64, zero/two objects, noncontiguous inputs and two/three ranks.
They retain original ID-alignment assertions and verify source code/globals are
unchanged. The saved partitioned BF16 reference also passes the cache comparison.

[video-collective-reference-validation.json](video-collective-reference-validation.json)
records per-output differences, source hashes and artifact references. The private
bucket prefix `native-foundation/video-collective-reference/` retains reference
tensors, reports, logs and source. Existing native outputs, input frames, weights
and SDK layers are referenced instead of copied. No new distribution variants.

No Actions, Windows/Turing execution, physical multi-GPU result or performance
improvement is claimed. Parallel scheduling and broader validation remain open.
