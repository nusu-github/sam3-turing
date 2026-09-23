"""Image-versus-single-frame-video reference for the owning native predictor.

The optional empty-points audit records unmodified upstream behavior. It does
not repair its fallback or claim that native restoration equals a failed call.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch


def native_result(root):
    meta = json.loads((root / 'semantic.json').read_text())
    ids = np.fromfile(root / 'semantic.ids.i64.bin', np.int64)
    n, h, w = len(ids), meta['height'], meta['width']
    packed = np.fromfile(root / 'semantic.masks.bin', np.uint8).reshape(n, h, (w + 7) // 8)
    return dict(out_obj_ids=ids,
                out_probs=np.fromfile(root / 'semantic.scores.f32.bin', np.float32),
                out_boxes_xywh=np.fromfile(root / 'semantic.boxes.f32.bin', np.float32).reshape(n, 4),
                out_binary_masks=np.unpackbits(packed, axis=-1, bitorder='little')[..., :w].astype(bool))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint')
    p.add_argument('--image', type=Path)
    p.add_argument('--prompt', default='person')
    p.add_argument('--kind', choices=['image', 'video'], default='image')
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--native', type=Path)
    p.add_argument('--cached', action='store_true')
    p.add_argument('--audit-empty-points', action='store_true')
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    if not a.cached:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        assert a.checkpoint and a.image
        torch.set_num_threads(1)
        torch.manual_seed(189)
        wrapper = build_sam3_multiplex_video_predictor(checkpoint_path=a.checkpoint, use_fa3=False,
            use_rope_real=False, compile=False, warm_up=False, async_loading_frames=False)
        model = wrapper.model
        model._warm_up_complete = True
        model.batched_grounding_batch_size = 1
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        info = dict(kind=a.kind, prompt=a.prompt, matmul_tf32=False, cudnn_tf32=False,
                    source_complex_rope=True, source_adapters=[], source_grounding_batch=1)
        with tempfile.TemporaryDirectory() as directory, torch.autocast('cuda', dtype=torch.bfloat16):
            shutil.copy2(a.image, Path(directory) / ('0' + a.image.suffix))
            resource = str(a.image) if a.kind == 'image' else directory
            state = model.init_state(resource, offload_video_to_cpu=True, async_loading_frames=False)
            assert state['is_image_only'] == (a.kind == 'image')
            _, value = model.add_prompt(state, 0, text_str=a.prompt)
            a.reference.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.reference / 'semantic.npz', **{k: np.asarray(v) for k, v in value.items() if k != 'frame_stats'})
            info['ids'] = value['out_obj_ids'].tolist()
            info['scores'] = value['out_probs'].tolist()
            if a.audit_empty_points and len(value['out_obj_ids']):
                obj_id = int(value['out_obj_ids'][0])
                tracker_states = model._get_sam2_inference_states_by_obj_ids(state, [obj_id])
                assert len(tracker_states) == 1
                original_mask = model._get_mask_input(tracker_states[0], 0, obj_id)
                assert original_mask is not None
                np.save(a.reference / 'original-mask.npy', original_mask.cpu().numpy())
                # Test both a direct empty edit and clearing after a real click.
                info['empty_points_audit'] = []
                for point_first in (False, True):
                    model.reset_state(state)
                    model.add_prompt(state, 0, text_str=a.prompt)
                    stage = 'real_point' if point_first else 'empty_points'
                    try:
                        if point_first:
                            model.add_prompt(state, 0, obj_id=obj_id, points=torch.tensor([[.45, .55]]), point_labels=torch.tensor([1]))
                            stage = 'empty_points'
                        _, out = model.add_prompt(state, 0, obj_id=obj_id, points=torch.empty(0, 2), point_labels=torch.empty(0, dtype=torch.long))
                        info['empty_points_audit'].append(dict(point_first=point_first, success=True, ids=out['out_obj_ids'].tolist()))
                        np.savez_compressed(a.reference / f'empty-{point_first}.npz', **{k: np.asarray(v) for k, v in out.items() if k != 'frame_stats'})
                    except Exception as exc:
                        info['empty_points_audit'].append(dict(point_first=point_first, success=False, stage=stage,
                                                              error_type=type(exc).__name__, error=str(exc)))
            assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        (a.reference / 'reference-info.json').write_text(json.dumps(info, indent=2) + '\n')
    info = json.loads((a.reference / 'reference-info.json').read_text())
    report = dict(reference=info, scope='Unmodified SAM3.1 single-frame semantic preview; empty-point audit is separate from native restoration invariants.')
    if a.native:
        actual = native_result(a.native)
        with np.load(a.reference / 'semantic.npz') as expected:
            report['exact'] = {k: bool(np.array_equal(v, expected[k])) for k, v in actual.items()}
            if actual['out_binary_masks'].shape == expected['out_binary_masks'].shape:
                report['mask_mismatches'] = int(np.count_nonzero(actual['out_binary_masks'] != expected['out_binary_masks']))
        report['all_exact'] = all(report['exact'].values())
        retained = a.reference / 'original-mask.npy'
        restored = a.native / 'restored-input.bool.bin'
        if retained.exists() and restored.exists():
            original = np.load(retained)
            native = np.fromfile(restored, np.bool_).reshape(original.shape)
            report['restored_input_equals_source_original'] = bool(np.array_equal(native, original))
            report['all_exact'] &= report['restored_input_equals_source_original']
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if a.native:
        assert report['all_exact'], 'image semantic output differs'


if __name__ == '__main__':
    main()
