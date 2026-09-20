"""Experimental CPU inference: finish true GroupNorm channel groups separately.

No weight approximation. Real-arithmetic graph equivalence; FP32 bits can differ.
Keeps all 200 low-resolution masks and the semantic output, like the stock head.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'round2'))
from variants import build_head, make_inputs, stream_head


def group_stream(head, inputs, tile_rows=16, groups_per_pass=1):
    if torch.is_grad_enabled() or head.training or head.no_dec or head.aux_masks:
        raise ValueError('Inference only, with final-query masks')
    feats = inputs['backbone_feats']
    if len(feats) != 3 or feats[0].shape[0] != 1 or inputs['image_ids'].tolist() != [0]:
        raise ValueError('One image, three FPN levels')
    if head.presence_head is not None or head.pixel_decoder.shared_conv:
        raise ValueError('Unsupported head')
    if any(feats[i].shape[-2] != 2 * feats[i+1].shape[-2] or feats[i].shape[-1] != 2 * feats[i+1].shape[-1] for i in (0, 1)):
        raise ValueError('Exact 2x nearest upsampling required')
    enc = inputs['encoder_hidden_states']
    if head.cross_attend_prompt is not None:
        enc = enc + head.cross_attend_prompt(query=head.cross_attn_norm(enc), key=inputs['prompt'],
                                          value=inputs['prompt'], key_padding_mask=inputs['prompt_mask'])[0]
    enc_map = enc.permute(1, 2, 0).reshape_as(feats[-1])
    dec = head.pixel_decoder
    merged = feats[1] + F.interpolate(enc_map, size=feats[1].shape[-2:], mode='nearest')
    previous = F.relu(dec.norms[0](dec.conv_layers[0](merged)))
    del merged, enc_map, enc
    current = feats[0]
    conv, norm = dec.conv_layers[1], dec.norms[1]
    if conv.kernel_size != (3, 3) or conv.padding != (1, 1) or conv.stride != (1, 1) or conv.groups != 1:
        raise ValueError('Expected stock 3x3 convolution')
    if tile_rows < 1 or groups_per_pass < 1 or norm.num_groups % groups_per_pass:
        raise ValueError('Positive tiles and complete normalization groups required')
    _, channels, height, width = current.shape
    group_channels = channels // norm.num_groups
    block_channels = group_channels * groups_per_pass
    e = head.mask_predictor.mask_embed(inputs['obj_queries'][-1])
    ea = e @ head.instance_seg_head.weight[:, :, 0, 0]
    eb = e @ head.instance_seg_head.bias
    masks = current.new_zeros((1, e.shape[1], height, width))
    semantic = current.new_zeros((1, 1, height, width))
    sw = head.semantic_seg_head.weight[:, :, 0, 0]
    for cs in range(0, channels, block_channels):
        ce = cs + block_channels
        buffer = current.new_empty((1, block_channels, height, width))
        for start in range(0, height, tile_rows):
            end = min(height, start + tile_rows)
            lo, hi = max(0, start - 1), min(height, end + 1)
            rows = torch.arange(lo, hi, device=previous.device) // 2
            up = previous.index_select(-2, rows).repeat_interleave(2, dim=-1)
            merged = current[..., lo:hi, :] + up
            # Slice OUTPUT channels, retaining all 256 INPUT channels and halos.
            chunk = F.conv2d(merged, conv.weight[cs:ce], conv.bias[cs:ce], padding=1)
            buffer[..., start:end, :].copy_(chunk[..., start-lo:end-lo, :])
            del up, merged, chunk
        feature = F.group_norm(buffer, groups_per_pass, norm.weight[cs:ce], norm.bias[cs:ce], norm.eps)
        del buffer
        feature.relu_()
        for start in range(0, height, tile_rows):
            end = min(height, start + tile_rows)
            part = feature[..., start:end, :]
            masks[..., start:end, :].add_(torch.einsum('bqc,bchw->bqhw', ea[..., cs:ce], part))
            semantic[..., start:end, :].add_(torch.einsum('qc,bchw->bqhw', sw[:, cs:ce], part))
        del part, feature
    masks.add_(eb[..., None, None])
    semantic.add_(head.semantic_seg_head.bias[None, :, None, None])
    return dict(pred_masks=masks, semantic_seg=semantic, presence_logit=None)
