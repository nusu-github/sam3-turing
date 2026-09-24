"""SAM3.1 propagation decoder and demux/gating/pointer host reference tests."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from sam3.model.multiplex_mask_decoder import MultiplexMaskDecoder
from sam3.model.multiplex_utils import MultiplexController
from sam3.model.video_tracking_multiplex import VideoTrackingMultiplex
from sam3.sam.mask_decoder import MLP
from sam3.sam.prompt_encoder import PositionEmbeddingRandom
from sam3.sam.transformer import TwoWayTransformer

DECODER_KEYS = ['masks', 'iou_pred', 'sam_tokens_out', 'object_score_logits']
HEAD_KEYS = ['low_res_multimasks', 'high_res_multimasks', 'ious', 'low_res_masks',
             'high_res_masks', 'obj_ptr', 'object_score_logits']


def reference(weights, device):
    decoder = MultiplexMaskDecoder(
        transformer_dim=256, transformer=TwoWayTransformer(2, 256, 8, 2048),
        multiplex_count=16, num_multimask_outputs=3, multimask_outputs_only=True,
        use_high_res_features=True, iou_prediction_use_sigmoid=False,
        pred_obj_scores=True, pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True, dynamic_multimask_via_stability=False)
    host = SimpleNamespace(
        hidden_dim=256, sam_image_embedding_size=72, image_size=1008,
        add_output_suppression_embeddings=True, _maybe_clone=lambda x: x,
        pred_obj_scores=True, object_score_logit_threshold=0.,
        decode_mask_with_shared_tokens=False, stability_score_attentuation=False,
        use_obj_ptrs_in_encoder=True, use_no_obj_ptr=True, use_linear_no_obj_ptr=True)
    root = 'tracker.model.'
    for name, module in [('sam_mask_decoder', decoder), ('image_pe_layer', PositionEmbeddingRandom(128)),
                         ('obj_ptr_proj', MLP(256, 256, 256, 3)),
                         ('no_obj_ptr_linear', torch.nn.Linear(256, 256))]:
        prefix = root + name + '.'
        module.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}, strict=True)
        setattr(host, name, module.eval().to(device))
    for name in ['output_valid_embed', 'output_invalid_embed']:
        setattr(host, name, weights[root + name].to(device))
    host.get_propagation_dense_pe = MethodType(VideoTrackingMultiplex.get_propagation_dense_pe, host)
    host._forward_sam_heads = MethodType(VideoTrackingMultiplex._forward_sam_heads, host)
    return host


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('library', type=Path)
    parser.add_argument('store', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--modes', nargs='+', default=['fp32', 'fp16', 'bf16_reference'])
    parser.add_argument('--math', action='store_true', help='Test the precompiled SDPA math fallback')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.ops.load_library(str(args.library.resolve()))
    torch.set_num_threads(4)
    torch.manual_seed(930)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.math:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    host = reference(weights.get('model', weights), args.device)
    decoder = host.sam_mask_decoder
    results = []

    @contextmanager
    def reference_backend():
        # Original Attention.forward unconditionally re-enables fused backends.
        # For this fallback test only, preserve math-only policy around it.
        flash = torch.backends.cuda.enable_flash_sdp
        efficient = torch.backends.cuda.enable_mem_efficient_sdp
        try:
            if args.math:
                torch.backends.cuda.enable_flash_sdp = lambda enabled: flash(False)
                torch.backends.cuda.enable_mem_efficient_sdp = lambda enabled: efficient(False)
            yield
        finally:
            torch.backends.cuda.enable_flash_sdp = flash
            torch.backends.cuda.enable_mem_efficient_sdp = efficient

    def compare(actual, expected, **metadata):
        errors = {}
        assert len(actual) == len(expected)
        for name, value, target in zip(metadata['keys'], actual, expected):
            torch.testing.assert_close(value, target, rtol=0, atol=0)
            errors[name] = (value.float() - target.float()).abs().max().item()
        item = dict(**metadata, max_errors=errors, exact=True)
        results.append(item)
        print(json.dumps(item), flush=True)

    for mode in args.modes:
        dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16_reference': torch.bfloat16}[mode]

        def amp():
            return torch.autocast(args.device, enabled=mode != 'fp32',
                                  dtype=dtype if mode != 'fp32' else torch.bfloat16)

        # Diagnostic grids cover layout/shape behavior; full 72x72 grids below
        # retain the shipped 1008px model resolution and all 48 mask tokens.
        for batch, height, width, shared, extra_kind, channels_last in [
                (1, 8, 8, True, 'none', False), (2, 8, 12, True, 'per_bucket', True),
                (2, 12, 8, False, 'shared', False), (1, 72, 72, True, 'per_bucket', True),
                (2, 72, 72, False, 'none', True)]:
            image = torch.randn(batch, 256, height, width, device=args.device, dtype=dtype)
            high = [torch.randn(1 if shared else batch, c, height * scale, width * scale,
                                device=args.device, dtype=dtype) for c, scale in [(32, 4), (64, 2)]]
            if channels_last:
                image = image.contiguous(memory_format=torch.channels_last)
                high = [x.contiguous(memory_format=torch.channels_last) for x in high]
            position = torch.randn(1, 256, height, width, device=args.device)
            extra = None if extra_kind == 'none' else torch.randn(
                1 if extra_kind == 'shared' else batch, 16, 256, device=args.device)
            with amp(), reference_backend():
                expected = decoder(image, position, True, high, extra)
            actual = torch.ops.sam3_native.multiplex_decode(str(args.store), image, position, high, extra, mode)
            compare(actual, [expected[k] for k in DECODER_KEYS], keys=DECODER_KEYS, mode=mode, case='decoder',
                    batch=batch, grid=[height, width], shared_high=shared, extra=extra_kind, channels_last=channels_last)

        cases = [('one_object', 1, False, False, 0., False),
                 ('cross_bucket', 17, False, False, 0., False),
                 ('removed_and_added', 19, True, False, 0., False),
                 ('forced_present_projected_high', 3, False, True, -1e6, False),
                 ('forced_absent', 3, False, False, 1e6, False),
                 ('stability_selection', 3, False, False, 0., True)]
        for name, count, mutate, project, threshold, attenuate in cases:
            state = MultiplexController(16).eval().get_state(
                count, torch.device(args.device), torch.float32, random=True)
            if mutate:
                state.remove_objects([0, 3, 17])
                new = state.find_next_batch_of_available_indices(4, allow_new_buckets=True, prefer_new_buckets=True)
                state.add_objects(new, allow_new_buckets=True, prefer_new_buckets=True)
            image = torch.randn(state.num_buckets, 256, 72, 72, device=args.device, dtype=dtype)
            image = image.contiguous(memory_format=torch.channels_last)
            raw_high = [torch.randn(1, 256 if project else c, size, size, device=args.device, dtype=dtype)
                        .contiguous(memory_format=torch.channels_last) for c, size in [(32, 288), (64, 144)]]
            host.object_score_logit_threshold = threshold
            host.stability_score_attentuation = attenuate
            with amp(), reference_backend():
                high = [decoder.conv_s0(raw_high[0]), decoder.conv_s1(raw_high[1])] if project else raw_high
                expected = host._forward_sam_heads(image, propagation_high_res_features=high,
                                                   multimask_output=True, multiplex_state=state)
                expected_values = [expected[k] for k in HEAD_KEYS] + [host.get_propagation_dense_pe(), *high]
            actual = torch.ops.sam3_native.multiplex_propagation(
                str(args.store), image, raw_high, state.assignments, mode, threshold, attenuate, project)
            compare(actual, expected_values, keys=HEAD_KEYS + ['position', 'high0', 'high1'], mode=mode,
                    case=name, objects=state.total_valid_entries, buckets=state.num_buckets,
                    present=(actual[6] > threshold).flatten().tolist())
            del actual, expected, expected_values
    if args.math:
        assert torch.backends.cuda.math_sdp_enabled()
        assert not any([torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
                        torch.backends.cuda.cudnn_sdp_enabled()])
    args.report.write_text(json.dumps(dict(torch=torch.__version__, device=args.device, math_only=args.math,
        reference_backend_override='Prevent original Attention.forward from re-enabling Flash/memory-efficient SDPA' if args.math else None, cases=results,
        scope='Original shipped SAM3.1 propagation decoder plus suppression, demux, presence gating, candidate selection and pointers. Synthetic feature inputs; no temporal scheduling or real-video accuracy claim.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
