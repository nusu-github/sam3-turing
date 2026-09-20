"""Spatial MLP sharing only in low-variance 2x2 feature cells."""

from unittest.mock import patch
import torch


def apply_adaptive_mlp(model, keep_fraction, selected_layers, stack):
    if not 0 < keep_fraction < 1:
        raise ValueError("keep_fraction must be between zero and one")
    for index, block in enumerate(model.backbone.vision_backbone.trunk.blocks):
        if index not in selected_layers:
            continue
        original = block.mlp.forward

        def forward(x, original=original):
            b, h, w, c = x.shape
            if h % 2 or w % 2:
                return original(x)
            cells = (
                x.reshape(b, h // 2, 2, w // 2, 2, c)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(b, -1, 4, c)
            )
            groups = cells.shape[1]
            keep = max(1, min(groups - 1, int(groups * keep_fraction)))
            mean = cells.mean(2)
            variance = (cells.square().mean((2, 3)) - mean.square().mean(2)).clamp_min(
                0
            )
            order = variance.argsort(dim=1, descending=True)
            high, low = order[:, :keep], order[:, keep:]
            high_index = high[:, :, None, None].expand(-1, -1, 4, c)
            low_index = low[:, :, None, None].expand(-1, -1, 4, c)
            separate = cells.gather(1, high_index).reshape(b, keep * 4, c)
            shared = mean.gather(1, low[:, :, None].expand(-1, -1, c))
            projected = original(torch.cat((separate, shared), dim=1))
            result = projected.new_empty((b, groups, 4, c))
            result.scatter_(
                1, high_index, projected[:, : keep * 4].reshape(b, keep, 4, c)
            )
            result.scatter_(
                1, low_index, projected[:, keep * 4 :, None, :].expand(-1, -1, 4, -1)
            )
            return (
                result.reshape(b, h // 2, w // 2, 2, 2, c)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(b, h, w, c)
            )

        stack.enter_context(patch.object(block.mlp, "forward", forward))
