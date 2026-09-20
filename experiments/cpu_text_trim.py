"""Skip causal text padding on CPU, then restore the original GPU tensor shapes."""

from unittest.mock import patch

import torch.nn.functional as F


def apply_cpu_text_trim(processor, stack, alignment=8):
    if not getattr(processor, "_turing_cpu_text", False):
        raise ValueError("This candidate requires CPU text offload")
    if alignment < 1:
        raise ValueError("Alignment must be positive")
    encoder = processor.model.backbone.language_backbone.encoder
    if encoder.attn_mask is None:
        raise ValueError("This candidate requires the standard causal text encoder")
    original = encoder.forward

    def forward(tokens):
        # EOT has the highest CLIP token ID; token 0 can also occur in real text.
        length = int(tokens.argmax(1).max()) + 1
        length = min(
            tokens.shape[1], ((length + alignment - 1) // alignment) * alignment
        )
        pooled, memory = original(tokens[:, :length])
        padding = tokens.shape[1] - length
        if padding:
            memory = F.pad(memory, (0, 0, 0, padding))
            if pooled.ndim == 3:
                pooled = F.pad(pooled, (0, 0, 0, padding))
        # VETextEncoder still returns the original token mask/embeddings. Only
        # masked trailing memory differs, and the GPU grounding sees 32 tokens.
        return pooled, memory

    stack.enter_context(patch.object(encoder, "forward", forward))
