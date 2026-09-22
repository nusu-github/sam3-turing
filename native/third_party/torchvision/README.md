The ROIAlign sampling implementation in `src/roi_align_impl.h` is adapted
from torchvision's CPU/CUDA ROIAlign forward kernels (commit `7a13ad0f`).
Source: https://github.com/pytorch/vision/tree/7a13ad0f/torchvision/csrc/ops
The BSD license is reproduced in this directory. Only inference sampling is
used; torchvision is not a native runtime dependency.
