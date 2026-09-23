"""Explicit test-only correction for upstream dense-memory extraction loss.

Unmodified upstream tries slot demux on [bucket,256,72,72], catches its error,
and writes None. This optional adapter re-encodes from original effective inputs
with the original memory encoder. It never runs in production or silently alters
an unmodified-source comparison. Both references must be reported separately.
"""
import inspect
from functools import wraps
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


def preserve_singleton_history(model):
    """Avoid an unnecessary mux/demux of already-singleton dense history.

    Original interaction extraction removes and re-adds the only object, then
    muxes dense spatial tensors as slot data. Preserve untouched frame tensors
    across that bookkeeping operation. Current-frame neural output stays intact.
    """
    tracker=model.tracker.model;snapshots={};differences=[]
    extract=tracker._extract_object_for_interaction
    merge=tracker._merge_singleton_interaction_result
    def capture(state,obj_id,frame_idx):
        assert state['obj_ids']==[obj_id],'this adapter only covers existing singleton sessions'
        snapshots[id(state)]=(frame_idx,{(group,f):{key:x.get(key) for key in ('maskmem_features','maskmem_pos_enc','obj_ptr')} for group,frames in state['output_dict'].items() for f,x in frames.items() if f!=frame_idx})
        return extract(state,obj_id,frame_idx)
    def restore(state,*args,**kwargs):
        result=merge(state,*args,**kwargs)
        frame_idx,old=snapshots.pop(id(state))
        for (group,f),fields in old.items():
            current=state['output_dict'][group][f]
            for key,original in fields.items():
                if original is None:continue
                value=current.get(key)
                if key=='maskmem_pos_enc' and value and original:
                    a,b=value[-1],original[-1]
                    if a is not None and b is not None:
                        if a.ndim==5:a=a[:,0]
                        if a.shape==b.shape: differences.append(dict(frame=f,edit_frame=frame_idx,max_position_error=float((a.float()-b.float()).abs().max()),before_dtype=str(b.dtype),after_dtype=str(a.dtype)))
                current[key]=original
        return result
    tracker._extract_object_for_interaction=capture;tracker._merge_singleton_interaction_result=restore
    return differences


def refresh_refined_memory(model):
    """Retain the latest encoded point edit when source cond/non-cond keys collide.

    Source computes new memory in non-cond output and then deletes that output
    because an older cond key exists. Its low masks have already been updated
    through aliases, so masks and memory disagree. Capture the *original* newly
    consolidated neural output and retain it under the conditioning key.
    """
    tracker=model.tracker.model;pending={}
    consolidate=tracker._consolidate_temp_output_across_obj
    signature=inspect.signature(consolidate)
    def capture(*args,**kwargs):
        values=signature.bind(*args,**kwargs).arguments
        result=consolidate(*args,**kwargs)
        pending[id(values['inference_state']),values['frame_idx'],values['is_cond']]=result
        return result
    tracker._consolidate_temp_output_across_obj=capture
    preflight=tracker.propagate_in_video_preflight
    def repair(state,*args,**kwargs):
        collisions={f for output in state['temp_output_dict_per_obj'].values() for f in output['non_cond_frame_outputs'] if f in state['output_dict']['cond_frame_outputs']}
        result=preflight(state,*args,**kwargs)
        for frame in collisions:
            latest=pending[id(state),frame,False]
            state['output_dict']['cond_frame_outputs'][frame]=latest
            tracker._add_output_per_object(state,frame,latest,'cond_frame_outputs')
            state['consolidated_frame_inds']['non_cond_frame_outputs'].discard(frame)
            state['consolidated_frame_inds']['cond_frame_outputs'].add(frame)
        pending.clear()
        return result
    tracker.propagate_in_video_preflight=repair


def refresh_refined_pointer(model):
    """Use the latest original pointer when old cond/new non-cond keys collide.

    The source consolidator always selects the conditioning entry for obj_ptr,
    even when consolidating newer non-conditioning point edits. Masks come from
    the latest temporary output, leaving that pointer one click behind. This
    opt-in reference repair retains the original newly computed pointer; it does
    not modify neural computations or consume native tensors.
    """
    tracker = model.tracker.model
    consolidate = tracker._consolidate_temp_output_across_obj
    signature = inspect.signature(consolidate)
    audit = []

    @wraps(consolidate)
    def repair(*args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        state, frame = values['inference_state'], values['frame_idx']
        previous = state['output_dict']['cond_frame_outputs'].get(frame)
        latest = state['output_dict']['non_cond_frame_outputs'].get(frame)
        result = consolidate(*args, **kwargs)
        if (not values['is_cond'] and previous is not None and latest is not None
                and any(frame in frames for frames in state['point_inputs_per_obj'].values())):
            pointer = latest.get('obj_ptr')
            if pointer is not None:
                old = result['obj_ptr']
                audit.append(dict(frame=frame,run_mem_encoder=values['run_mem_encoder'],
                                  changed_elements=int((old != pointer).sum()),
                                  max_pointer_error=float((old.float()-pointer.float()).abs().max())))
                result['obj_ptr'] = pointer
        return result

    tracker._consolidate_temp_output_across_obj = repair
    return audit


def edit_tags(extended=False):
    first=['18.point','18.track','19.track','20.track']
    return first+(['19.repeat1','19.repeat2','18.reverse','17.reverse','20.new','20.multi','21.multi','22.multi','20.remove','20.remove_again','22.stateless'] if extended else [])


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
        if a.reference_output and tag in ('19.repeat1','19.repeat2'):
            session=next(s for s in state['sam2_inference_states'] if s['obj_ids']==[selected]);fields={}
            for group,frames in session['output_dict'].items():
                for frame,x in frames.items():
                    if frame not in (0,18,19):continue
                    for name in ('pred_masks','maskmem_features'):
                        if x.get(name) is not None:fields[f'{group}/{frame}/{name}']=x[name].float().cpu().numpy()
            np.savez_compressed(a.reference_output/f'{tag}.state.npz',**fields)

        if a.reference_output:np.savez_compressed(a.reference_output/f'{tag}.npz',**{k:np.asarray(v) for k,v in value.items() if k!='frame_stats'})
    with torch.autocast('cuda',enabled=a.mode!='fp32',dtype=dtype):
        points=torch.tensor([[.45,.55],[.8,.15]],device='cuda');labels=torch.tensor([1,0],device='cuda')
        _,value=model.add_prompt(state,18,points=points,point_labels=labels,obj_id=selected);check('18.point',value)
        for index,value in model.propagate_in_video(state,start_frame_idx=18,max_frame_num_to_track=2,reverse=False):check(f'{index}.track',value)
        if a.extended_edit_output:
            assert len(a.frames)>22
            _,value=model.add_prompt(state,19,points=points,point_labels=labels,obj_id=selected);check('19.repeat1',value)
            extra=torch.tensor([[.5,.65]],device='cuda');positive=torch.tensor([1],device='cuda')
            _,value=model.add_prompt(state,19,points=extra,point_labels=positive,obj_id=selected,clear_old_points=False);check('19.repeat2',value)
            for index,value in model.propagate_in_video(state,start_frame_idx=19,max_frame_num_to_track=2,reverse=True):check(f'{index}.reverse',value)
            _,value=model.add_prompt(state,20,points=extra,point_labels=positive,obj_id=9000);check('20.new',value)
            for index,value in model.propagate_in_video(state,start_frame_idx=20,max_frame_num_to_track=2,reverse=False):check(f'{index}.multi',value)
            model.remove_object(state,9000,frame_idx=None,is_user_action=True)
            for index,value in model.propagate_in_video(state,start_frame_idx=20,max_frame_num_to_track=0,reverse=False):check('20.remove',value)
            model.remove_object(state,9000,frame_idx=None,is_user_action=True)
            for index,value in model.propagate_in_video(state,start_frame_idx=20,max_frame_num_to_track=0,reverse=False):check('20.remove_again',value)
            model.use_stateless_refinement=True;other=int(state['tracker_metadata']['obj_ids_all_gpu'][1])
            _,value=model.add_prompt(state,22,points=points,point_labels=labels,obj_id=other);check('22.stateless',value)

    if a.reference_output:(a.reference_output/'edit-reference.json').write_text(json.dumps(dict(corrected_dense_extraction=a.sam31_rebuild_extracted_memory,iteration_mask_enabled=a.sam31_enable_repeat_refinement,default_remove_frame=a.sam31_default_remove_frame,refresh_refined_memory=a.sam31_refresh_refined_memory,preserve_singleton_history=a.sam31_preserve_singleton_history,refresh_refined_pointer=a.sam31_refresh_refined_pointer))+'\n')
    exact=all(all(r['exact'].values()) for r in rows)
    a.report.with_suffix('.edit.json').write_text(json.dumps(dict(model=a.model,mode=a.mode,corrected_dense_extraction=a.sam31_rebuild_extracted_memory,iteration_mask_enabled=a.sam31_enable_repeat_refinement,default_remove_frame=a.sam31_default_remove_frame,refresh_refined_memory=a.sam31_refresh_refined_memory,preserve_singleton_history=a.sam31_preserve_singleton_history,refresh_refined_pointer=a.sam31_refresh_refined_pointer,extended_sequence=a.extended_edit_output,scope='Original high-level first point refinement of a grouped object, singleton extraction, and three-frame partial propagation after full neural forward. Optional adapter explicitly re-encodes historical dense memories upstream loses; no neural computation mocked. Extended sequence, when selected, adds repeated/accumulated points, reverse propagation, new ID, forward propagation, removal/fetch and stateless first refinement. This is not a quality benchmark.',cases=rows,exact=exact),indent=2)+'\n')
    if getattr(a,'pointer_refresh_audit',None):a.report.with_suffix('.pointer-refresh.json').write_text(json.dumps(a.pointer_refresh_audit,indent=2)+'\n')
    if getattr(a,'singleton_history_differences',None):a.report.with_suffix('.history-casts.json').write_text(json.dumps(a.singleton_history_differences,indent=2)+'\n')
    if a.require_exact:assert exact,'SAM3.1 edited video differs; see report'
