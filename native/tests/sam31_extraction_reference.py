"""Explicit test-only correction for upstream dense-memory extraction loss.

Unmodified upstream tries slot demux on [bucket,256,72,72], catches its error,
and writes None. This optional adapter re-encodes from original effective inputs
with the original memory encoder. It never runs in production or silently alters
an unmodified-source comparison. Both references must be reported separately.
"""
import inspect
import json
import numpy as np
import torch


def preserve_extracted_memory(model):
    # The public tracker is an attribute-proxy wrapper; internal self-calls run
    # on its model, so hooks must be installed on that actual neural module.
    tracker=model.tracker.model
    original_encoder=tracker._encode_new_memory
    retained={};current_frame=[None]
    def capture(*args,**kwargs):
        inputs=inspect.signature(original_encoder).bind(*args,**kwargs).arguments
        result=original_encoder(*args,**kwargs)
        frame=current_frame[0]
        if frame is not None:
            for row,obj_id in enumerate(inputs['multiplex_state'].object_ids):
                retained[obj_id,frame]=(inputs['pred_masks_high_res'][row:row+1].detach().cpu().clone(),inputs['object_score_logits'][row:row+1].detach().cpu().clone())
        return result
    tracker._encode_new_memory=capture
    def with_frame(original):
        signature=inspect.signature(original)
        def wrapped(*args,**kwargs):
            frame=signature.bind(*args,**kwargs).arguments['frame_idx']
            previous=current_frame[0];current_frame[0]=frame
            try:return original(*args,**kwargs)
            finally:current_frame[0]=previous
        return wrapped
    tracker._run_memory_encoder=with_frame(tracker._run_memory_encoder)
    model._run_single_frame_inference=with_frame(model._run_single_frame_inference)
    original_extract=model._extract_object_to_singleton_state
    def extract(state,obj_id,rank):
        source=next(s for s in state['sam2_inference_states'] if obj_id in s['obj_ids'])
        if len(source['obj_ids'])<=1:return original_extract(state,obj_id,rank)
        old={f:retained[obj_id,f] for group in source['output_dict'].values() for f in group}
        original_extract(state,obj_id,rank)
        singleton=next(s for s in state['sam2_inference_states'] if s['obj_ids']==[obj_id])
        for group in singleton['output_dict'].values():
            for frame,value in group.items():
                high,scores=old[frame]
                assert value['maskmem_features'] is None,'upstream extraction behavior changed; review this adapter'
                image=value['image_features'].to(singleton['device'])
                features,position=original_encoder(image=None,current_vision_feats=[image],feat_sizes=[(72,72)],pred_masks_high_res=high.to(singleton['device']),object_score_logits=scores.to(singleton['device']),is_mask_from_pts=False,conditioning_objects=value['conditioning_objects'],multiplex_state=singleton['multiplex_state'])
                value['maskmem_features']=features.to(torch.bfloat16).to(singleton['storage_device'])
                value['maskmem_pos_enc']=[v.to(singleton['storage_device']) for v in position]
        print('CORRECTED_REFERENCE rebuilt',len(old),'dense singleton memories',flush=True)
    model._extract_object_to_singleton_state=extract


def compare_edits(a,model,state,dtype):
    selected=int(state['tracker_metadata']['obj_ids_all_gpu'][0]);rows=[]
    def check(tag,value):
        root=a.native_output;ids=np.fromfile(root/f'{tag}.ids.i64.bin',np.int64);n=len(ids);h,w=state['orig_height'],state['orig_width']
        actual=dict(out_obj_ids=ids,out_probs=np.fromfile(root/f'{tag}.scores.f32.bin',np.float32),out_boxes_xywh=np.fromfile(root/f'{tag}.boxes.f32.bin',np.float32).reshape(n,4))
        packed=np.fromfile(root/f'{tag}.masks.bin',np.uint8).reshape(n,h,(w+7)//8);actual['out_binary_masks']=np.unpackbits(packed,axis=-1,bitorder='little')[...,:w].astype(bool)
        equal={k:bool(np.array_equal(v,value[k])) for k,v in actual.items()};masks=actual['out_binary_masks'];expected=value['out_binary_masks']
        same=masks.shape==expected.shape;union=np.count_nonzero(masks|expected,axis=(1,2)) if same else None
        row=dict(tag=tag,exact=equal,mask_mismatches=int(np.count_nonzero(masks!=expected)) if same else -1,mask_iou=np.divide(np.count_nonzero(masks&expected,axis=(1,2)),union,out=np.ones(n),where=union>0).tolist() if same else None)
        rows.append(row);print(row,flush=True)
        if a.reference_output:np.savez_compressed(a.reference_output/f'{tag}.npz',**{k:np.asarray(v) for k,v in value.items() if k!='frame_stats'})
    with torch.autocast('cuda',enabled=a.mode!='fp32',dtype=dtype):
        points=torch.tensor([[.45,.55],[.8,.15]],device='cuda');labels=torch.tensor([1,0],device='cuda')
        _,value=model.add_prompt(state,18,points=points,point_labels=labels,obj_id=selected);check('18.point',value)
        for index,value in model.propagate_in_video(state,start_frame_idx=18,max_frame_num_to_track=2,reverse=False):check(f'{index}.track',value)
    if a.reference_output:(a.reference_output/'edit-reference.json').write_text(json.dumps(dict(corrected_dense_extraction=a.sam31_rebuild_extracted_memory))+'\n')
    exact=all(all(r['exact'].values()) for r in rows)
    a.report.with_suffix('.edit.json').write_text(json.dumps(dict(model=a.model,mode=a.mode,corrected_dense_extraction=a.sam31_rebuild_extracted_memory,scope='Original high-level first point refinement of a grouped object, singleton extraction, and three-frame partial propagation after full neural forward. Optional adapter explicitly re-encodes historical dense memories upstream loses; no neural computation mocked. This is not a quality benchmark.',cases=rows,exact=exact),indent=2)+'\n')
    if a.require_exact:assert exact,'SAM3.1 edited video differs; see report'
