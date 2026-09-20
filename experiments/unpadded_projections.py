"""Move window QKV/projection linears across partitioning to skip padding rows."""

from unittest.mock import patch

import torch
import torch.nn.functional as F

from sam3.model.vitdet import window_partition, window_unpartition


def apply_unpadded_projections(
    model, side, stack, qkv=True, projection=True, half_output=False
):
    for block in model.backbone.vision_backbone.trunk.blocks:
        size = block.window_size
        if not size:
            continue
        attn = block.attn
        if attn.use_rel_pos or attn.cls_token or attn.use_fa3:
            raise ValueError(
                "This candidate expects the ordinary SAM3 window Attention"
            )
        valid, _ = window_partition(
            torch.ones(
                (1, side, side, 1), device=attn.qkv.bias.device, dtype=torch.bool
            ),
            size,
        )
        block.register_buffer("_unpadded_window_valid", valid)

        def forward(x, block=block, attn=attn, size=size):
            shortcut = x
            x = block.norm1(x)
            height, width = x.shape[1:3]
            if qkv:
                projected = attn.qkv(x)
                windows, padded = window_partition(projected, size)
                # The original zero input rows produce the QKV bias, not zero QKV.
                if height % size or width % size:
                    windows = torch.where(
                        block._unpadded_window_valid,
                        windows,
                        attn.qkv.bias.to(windows.dtype),
                    )
            else:
                windows, padded = window_partition(x, size)
                windows = attn.qkv(windows)
            batches = windows.shape[0]
            parts = windows.reshape(batches, size * size, 3, attn.num_heads, -1)
            query, key, value = parts.permute(2, 0, 3, 1, 4).unbind(0)
            query, key = attn._apply_rope(query, key)
            attended = F.scaled_dot_product_attention(query, key, value)
            attended = attended.transpose(1, 2).reshape(batches, size, size, -1)
            if projection:
                attended = window_unpartition(attended, size, padded, (height, width))
                attended = attn.proj(attended)
            else:
                attended = window_unpartition(
                    attn.proj(attended), size, padded, (height, width)
                )
            x = shortcut + block.dropout(block.drop_path(block.ls1(attended)))
            result = x + block.dropout(
                block.drop_path(block.ls2(block.mlp(block.norm2(x))))
            )
            return result.half() if half_output else result

        stack.enter_context(patch.object(block, "forward", forward))
