"""Keep the uint8 resize, then fuse conversion and normalization."""

from unittest.mock import patch

import torch
import triton
import triton.language as tl
from torchvision.transforms.v2 import functional as VF


@triton.jit
def _normalize(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src + i, i < N, 0).to(tl.float32)
    y = (x * (1.0 / 255.0) - 0.5) * 2.0
    tl.store(dst + i, y, i < N)


def apply_fast_normalize(processor, dtype, stack):
    original = processor.transform
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[dtype]

    def transform(image):
        if image.dtype != torch.uint8:
            return original(image)
        image = VF.resize(image, [processor.resolution, processor.resolution])
        image = image.as_subclass(torch.Tensor).contiguous()
        result = torch.empty_like(image, dtype=dtype)
        _normalize[(triton.cdiv(image.numel(), 1024),)](
            image, result, image.numel(), 1024, enable_fp_fusion=False
        )
        return result

    stack.enter_context(patch.object(processor, "transform", transform))
