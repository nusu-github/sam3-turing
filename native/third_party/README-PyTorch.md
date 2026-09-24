# PyTorch-derived normalization helpers

`native/src/vision_fusion_cuda.cu` adapts the Welford update, combination and
warp-reduction order from PyTorch `aten/src/ATen/native/cuda/layer_norm_kernel.cu`.
The source available in this development image had SHA-256
`76c658445ba6a20c9f0298295a00f822fc1429385af8de34a6519b566b8576c5`.
The license and contributor notices are preserved in `LICENSE-PyTorch`, which
is installed with the native SDK. The residual/layout/cast integration is local.

Numerical comparisons use the standalone LibTorch 2.10.0 distribution actually
linked into the SDK. The source-image hash is provenance, not a claim that the
NVIDIA development wheel and the standalone SDK share identical core binaries.
