"""Force the CUDA attention backend without changing CPU text attention."""

from contextlib import contextmanager
from unittest.mock import patch

import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@contextmanager
def efficient_cuda_attention():
    original = F.scaled_dot_product_attention

    def attention(query, *args, **kwargs):
        if query.device.type != "cuda":
            return original(query, *args, **kwargs)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return original(query, *args, **kwargs)

    with patch.object(F, "scaled_dot_product_attention", attention):
        yield
