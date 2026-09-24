"""Compare integrated native planning/output assembly with original host methods.

Neural correction and memory execution are recorded, not replaced with a neural
reference. Session executors are exercised by separate actual-weight probes.
"""
import argparse,copy,hashlib,json,importlib
from pathlib import Path
import numpy as np
import torch
from sam3.model.sam3_video_base import Sam3VideoBase
from sam3.model.sam3_multiplex_base import Sam3MultiplexBase
from occlusion_parity import RecordingTracker,DTYPES
from temporal_memory_parity import source_transfer_device

FIELDS=[('first','obj_first_frame'),('unmatched','consecutive_unmatch_count'),('alive','trk_keep_alive'),('removed','removed_mask'),('overlap','overlap_pair_counts'),('occluded','last_occluded_tensor')]

def owner(mux,device,ints,floats,flags):
    cls=Sam3MultiplexBase if mux else Sam3VideoBase
    host=cls.__new__(cls);torch.nn.Module.__init__(host);host._device=torch.device(device)
    host.rank=0;host.world_size=1;host.is_multiplex=mux;host._warm_up_complete=flags[1]
    host.masklet_confirmation_enable=flags[2];host.suppress_det_close_to_boundary=flags[3]
    for name,value in zip(['hotstart_delay','hotstart_unmatch_thresh','hotstart_dup_thresh','init_trk_keep_alive','min_trk_keep_alive','max_trk_keep_alive','masklet_confirmation_consecutive_det_thresh','recondition_every_nth_frame','fill_hole_area','bucket_capacity'],ints):setattr(host,name,value)
    for name,value in zip(['new_det_thresh','trk_assoc_iou_thresh','assoc_iou_thresh','reconstruction_bbox_iou_thresh','reconstruction_bbox_det_score','suppress_overlapping_based_on_recent_occlusion_threshold'],floats):setattr(host,name,value)
    host.suppress_unmatched_only_within_hotstart=flags[4];host.decrease_trk_keep_alive_for_empty_masklets=flags[5];host.use_iom_recondition=flags[6];host.allow_unoccluded_to_suppress=flags[7]
    host.iom_thresh_recondition=.8;host.iou_thresh_recondition=.8;host.o2o_matching_masklets_enable=False;host.max_num_objects=0 if mux else 1000000000
    # SAM3.1 max_num_objects also controls optional padding. Keep it uncapped
    # through is_image_only=True; detector threshold stays explicitly equal.
    host.image_only_det_thresh=floats[0];host.sprinkle_removal_area=999
    host.tracker=RecordingTracker();host.tracker.input_mask_size=1152
    host._tracker_update_memories=lambda *args,**kwargs:None
    return host

def snapshot(out,root,meta,plan,current,host,frame):
    def save(k,v,dtype=None):out[root+k]=torch.as_tensor(v,dtype=dtype).detach().cpu().clone()
    save('ids',meta['obj_ids_all_gpu']);save('new_ids',plan['new_det_obj_ids']);save('new_ranks',plan['new_det_gpu_ids']);save('new_detections',plan['new_det_fa_inds']);save('removed',sorted(plan['obj_ids_newly_removed']),torch.long);save('unmatched',plan['unmatched_trk_obj_ids']);save('max_id',meta['max_obj_id']);save('tracking_masks',current)
    for r,ids in enumerate(meta['obj_ids_per_gpu']):save(f'rank/{r}',ids)
    for id,score in meta['obj_id_to_score'].items():save(f'score/{id}',float(score),torch.double)
    key='obj_id_to_sam2_score_frame_wise' if host.is_multiplex else 'obj_id_to_tracker_score_frame_wise'
    for f,scores in meta[key].items():
        for id,score in scores.items():save(f'frame_score/{f}/{id}',score)
    for id,f in meta['obj_id_to_last_occluded'].items():save(f'occluded/{id}',f)
    rank=meta['rank0_metadata']
    if host.masklet_confirmation_enable:
        save('confirmation/status',rank['masklet_confirmation']['status']);save('confirmation/count',rank['masklet_confirmation']['consecutive_det_num'])
    for field,key in [('first','obj_first_frame_idx'),('alive','trk_keep_alive'),('unmatched','unmatched_frame_inds')]:
        for id,value in rank[key].items():save(f'host/{field}/{id}',value,torch.long)
    for (a,b),values in rank['overlap_pair_to_frame_inds'].items():save(f'host/overlap/{a}/{b}',values,torch.long)
    save('host/removed',sorted(rank['removed_obj_ids']),torch.long)
    for f,values in rank['suppressed_obj_ids'].items():save(f'host/suppressed/{f}',sorted(values),torch.long)
    if host.is_multiplex:
        gpu=meta['gpu_metadata']
        for name,key in FIELDS:
            if key in gpu:save(f'device/{name}',gpu[key])
    calls=host.tracker.calls;save('corrections',[id for _,ids,_ in calls for id in ids],torch.long)
    save('correction_masks',torch.cat([mask for _,_,mask in calls]) if calls else torch.empty(0,1152,1152,dtype=torch.bool))
    if not host.is_multiplex:save('geometry',sorted(plan['reconditioned_obj_ids']),torch.long)

def masks_for(ids,step,device,dtype):
    masks=torch.full((len(ids),7,9),-2.,device=device,dtype=dtype)
    for row,id in enumerate(ids):
        col=(id%4)*2;masks[row,1:6,col:col+2]=2.
        if step%5==3 and row%2==0:masks[row]=-2.
    return masks

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--devices',nargs='+',default=['cpu','cuda']);p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Original CPU CCL stacks an empty list when a frame has no new detections.
    # Preserve its nonempty path; empty component images contain no labels/counts.
    cc=importlib.import_module('sam3.perflib.connected_components');original_cc=cc.connected_components
    def components(x):
        if x.size(0)==0:return torch.empty_like(x,dtype=torch.long),torch.empty_like(x,dtype=torch.long)
        return original_cc(x)
    cc.connected_components=components
    rows=[]
    for device in a.devices:
      for mode in a.modes:
       for mux in [False,True]:
        for reverse in [False,True]:
         for variant in range(4):
          ints=[5,2,2,2,-4,8,2,2 if variant==0 else -1,3,16]
          floats=[.7,.5,.1,.7 if variant==0 else 0.,.8,.5 if variant!=2 else 0.]
          flags=[reverse,variant!=3,variant!=2,variant==1,False,variant==1,variant==2,variant==1]
          host=owner(mux,device,ints,floats,flags);meta=host._initialize_metadata();inputs=[];frames=[];expected={};dtype=DTYPES[mode]
          for step in range(8):
            frame=7-step if reverse else step;frames.append(frame);ids=meta['obj_ids_all_gpu'];tracks=masks_for(ids,step,device,dtype)
            # Mix matched tracks, novel detections, duplicate regions and empty masks.
            novel=masks_for([step+101],0,device,dtype)
            det=torch.cat([tracks[::2].clone(),novel],0) if step%4!=3 else novel
            scores=torch.full((len(det),),.93,device=device,dtype=dtype)
            if step%3==2:scores[-1]=.65
            boxes=torch.tensor([[.2,.2,.8,.8]]*len(det),device=device,dtype=dtype);boxes[-1]=torch.tensor([0.,0.,.01,.01],device=device,dtype=dtype) if variant==1 else boxes[-1]
            keep=torch.ones(len(det),device=device,dtype=torch.bool)
            if mux and step%3==1:keep[-1]=False
            logits=torch.full((len(ids),),3.,device=device,dtype=dtype)
            if len(ids)>1:logits[1]=.7
            inputs.append([det,scores,boxes,keep,tracks.clone(),logits]);d=dict(mask=det,scores=scores,bbox=boxes)
            states=[dict(index=0,obj_ids=ids.tolist(),obj_idx_to_id=dict(enumerate(ids.tolist())))] if len(ids) else []
            host.tracker.calls=[];host.tracker.preflight=[]
            with source_transfer_device(device),torch.autocast(device,enabled=mode!='fp32',dtype=dtype):
                args=[frame,8,reverse,d]
                if mux:args.append(keep)
                plan,new=host.run_tracker_update_planning_phase(*args,tracks,logits,meta,states,is_image_only=True)
                snapshot(expected,f'{step}/planned/',new,plan,tracks,host,frame)
                kwargs={'sam2_update_plan' if mux else 'tracker_update_plan':plan}
                output=host.build_outputs(frame,8,reverse,d,tracks,logits,meta,orig_vid_height=37,orig_vid_width=53,reconditioned_obj_ids=plan['reconditioned_obj_ids'],**kwargs)
                for id,mask in output.items():expected[f'{step}/output/{id}']=mask.cpu().clone()
                score_key='obj_id_to_sam2_score_frame_wise' if mux else 'obj_id_to_tracker_score_frame_wise'
                if len(ids):new[score_key][frame].update(zip(ids,logits.sigmoid() if mux else logits.sigmoid().tolist()))
                snapshot(expected,f'{step}/final/',new,plan,tracks,host,frame)
            meta=new
          actual=torch.ops.sam3_native.video_update_sequence(inputs,frames,mux,1,ints,floats,flags,mode)
          for key,value in expected.items():
            try:torch.testing.assert_close(actual[key],value,rtol=0,atol=0,check_dtype=False)
            except (KeyError,AssertionError) as e:raise AssertionError(f'{device} {mode} mux={mux} reverse={reverse} variant={variant} {key}: {e}') from e
          rows.append(dict(device=device,mode=mode,multiplex=mux,reverse=reverse,variant=variant,frames=8,exact_comparisons=len(expected)));print(rows[-1],flush=True)
    a.report.write_text(json.dumps(dict(cases=rows,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('sam3/model/sam3_video_base.py'),Path('sam3/model/sam3_multiplex_base.py')]},scope='Integrated original planning and raw output assembly over multi-frame sequences. Neural correction/memory calls are recording/no-op fixtures; actual-weight execution requires separate tests. Uncapped source configuration via image-only planner flag with unchanged thresholds. Tensor-vs-Python-scalar score representation compared numerically with exact values. Native previous metadata is immutable; original multiplex shallow-copy behavior is not required. Original CPU connected-components empty-batch stack failure is adapted only by returning empty label/count tensors.'),indent=2)+'\n')
if __name__=='__main__':main()
