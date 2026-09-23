"""Opt-in capture of actual original temporal inputs at the reverse-edit regression."""
import inspect
import torch


def capture_reverse_input(model, destination):
    tracker = model.tracker.model
    original = tracker._prepare_memory_conditioned_features
    signature = inspect.signature(original)
    encoder = tracker.transformer.encoder
    encoder_forward = encoder.forward
    active = [None]

    def save_encoder(*args, **kwargs):
        if active[0] is not None:
            for src, dst in [('memory', 'assembled_memory'), ('memory_pos', 'assembled_position'),
                             ('memory_image', 'assembled_image'), ('memory_image_pos', 'assembled_image_position')]:
                active[0][dst] = kwargs[src].detach().cpu().clone()
        return encoder_forward(*args, **kwargs)

    def trace(*args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        if values['frame_idx'] != 17 or not values['track_in_reverse']:
            return original(*args, **kwargs)
        fields = {}
        for group, frames in values['output_dict'].items():
            prefix = 'cond' if group == 'cond_frame_outputs' else 'tracked'
            for index, entry in frames.items():
                for src, dst in [('maskmem_features', 'memory'), ('maskmem_pos_enc', 'position'),
                                 ('obj_ptr', 'pointer'), ('image_features', 'image'), ('image_pos_enc', 'image_position')]:
                    x = entry.get(src)
                    if isinstance(x, list): x = x[-1]
                    if x is not None: fields[f'{prefix}{index}.{dst}'] = x.detach().cpu().clone()
        fields['source'] = values['current_vision_feats'][-1].detach().cpu().clone()
        fields['source_position'] = values['current_vision_pos_embeds'][-1].detach().cpu().clone()
        active[0] = fields
        try:
            result = original(*args, **kwargs)
            fields['conditioned'] = result.detach().cpu().clone()
        finally:
            active[0] = None
        torch.save(fields, destination / 'reverse-input.pt')
        return result

    encoder.forward = save_encoder
    tracker._prepare_memory_conditioned_features = trace

    # Keep unmodified neural point outputs so a stale consolidated pointer can
    # be distinguished from a native decoder mismatch.
    run_frame = tracker._run_single_frame_inference
    frame_signature = inspect.signature(run_frame)
    point_calls = []

    def trace_point(*args, **kwargs):
        values = frame_signature.bind(*args, **kwargs).arguments
        result = run_frame(*args, **kwargs)
        if values['frame_idx'] == 19 and values.get('point_inputs') is not None:
            point_calls.append({
                'initial': values['is_init_cond_frame'],
                'labels': values['point_inputs']['point_labels'].detach().cpu().clone(),
                'pointer': result[0]['obj_ptr'].detach().cpu().clone(),
                'pred_masks': result[0]['pred_masks'].detach().cpu().clone(),
            })
            torch.save(point_calls, destination / 'point19-neural.pt')
        return result

    tracker._run_single_frame_inference = trace_point
