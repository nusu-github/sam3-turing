"""Original SAM3 track_step, including deferred encoding, offload and trimming."""
import argparse
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from sam3.model.sam3_tracker_base import Sam3TrackerBase
from sam3.model_builder import _create_tracker_transformer
from video_heads_parity import reference as heads_reference
from memory_encoder_parity import reference as memory_reference
from temporal_memory_parity import source_transfer_device


def reference(weights, device):
    host=heads_reference(weights,'sam3',device)
    _,memory=memory_reference(weights,'sam3',device)
    host.__dict__.update(memory.__dict__)
    encoder=_create_tracker_transformer().encoder.eval()
    prefix='tracker.transformer.encoder.'
    encoder.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
    encoder.to(device)
    if device=='cpu':
        for child in encoder.modules():
            if hasattr(child,'compute_cis'):child.freqs_cis=child.compute_cis(end_x=72,end_y=72,device='cpu')
    projection=torch.nn.Linear(256,64).eval();prefix='tracker.obj_ptr_tpos_proj.'
    projection.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
    host.obj_ptr_tpos_proj=projection.to(device);host.transformer=SimpleNamespace(encoder=encoder)
    for name in ['no_mem_embed','maskmem_tpos_enc','cond_frame_spatial_embedding','cond_frame_obj_ptr_embedding']:
        if 'tracker.'+name in weights:setattr(host,name,weights['tracker.'+name].to(device))
    for name in ['_forward_sam_heads','_use_mask_as_output','_encode_new_memory','_apply_non_overlapping_constraints',
                 '_prepare_memory_conditioned_features','_get_tpos_enc','frame_filter','cal_mem_score','track_step','_use_multimask']:
        setattr(host,name,MethodType(getattr(Sam3TrackerBase,name),host))
    host.iter_use_prev_mask_pred=True;host.keep_first_cond_frame=False;host.mem_dim=64;host.mf_threshold=.01
    return host


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('library',type=Path);parser.add_argument('store',type=Path);parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--device',default='cuda');parser.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();torch.ops.load_library(str(args.library.resolve()));torch.set_num_threads(4);torch.manual_seed(1481)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    weights=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True);weights=weights.get('model',weights)
    host=reference(weights,args.device);results=[]
    for mode in args.modes:
        dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
        cases=[dict(name='initial_one',initial=True,points=1),dict(name='initial_many',initial=True,points=17),
               dict(name='initial_empty',initial=True,points=0),dict(name='refine_previous',points=3,previous=True,use_previous=False),
               dict(name='forward'),dict(name='reverse',reverse=True),dict(name='direct_mask_1008',mask=1008),dict(name='direct_mask_1152',mask=1152),
               dict(name='deferred_memory',initial=True,points=2,encode=False),dict(name='disabled_memory',slots=0),
               dict(name='non_overlap_memory',initial=True,points=1,overlap=True),
               dict(name='offloaded_scores',initial=True,points=1,score=True,offload=True),
               dict(name='trim_low_score',initial=True,points=1,current=85,score=True,trim=True),
               dict(name='trim_high_score',initial=True,points=1,current=85,score=True,trim=True,high_score=True),
               dict(name='trim_offloaded',initial=True,points=1,current=85,score=True,trim=True,offload=True),
               dict(name='trim_reverse',reverse=True,trim=True),dict(name='single_output',initial=True,points=1,multi=False),
               dict(name='single_tracking',tracking_multi=False)]
        for case in cases:
            batch=2;current=case.get('current',12);slots=case.get('slots',3);pointer_limit=4;stride=1
            initial=case.get('initial',False);reverse=case.get('reverse',False);use_previous=case.get('use_previous',True)
            encode=case.get('encode',True);score=case.get('score',False);overlap=case.get('overlap',False);offload=case.get('offload',False);trim=case.get('trim',False)
            multi=case.get('multi',True);tracking_multi=case.get('tracking_multi',True)
            def rand(shape,device=args.device):return torch.randn(shape,device=device,dtype=dtype)
            # Expanded cached features match a session reusing one image for all objects.
            image=rand((1,256,72,72)).contiguous(memory_format=torch.channels_last).expand(batch,-1,-1,-1)
            position=rand((1,256,72,72)).expand(batch,-1,-1,-1)
            high=[rand((1,c,size,size)).contiguous(memory_format=torch.channels_last).expand(batch,-1,-1,-1) for c,size in [(32,288),(64,144)]]
            points=labels=mask=previous=None
            if 'points' in case:
                points=torch.rand(batch,case['points'],2,device=args.device)*1008
                labels=torch.randint(-1,4,(batch,case['points']),device=args.device,dtype=torch.int32)
            if case.get('mask'):
                mask=rand((batch,1,case['mask'],case['mask']))>0;mask[0].zero_()
            if case.get('previous'):previous=rand((batch,1,288,288)).float().clamp(-32,32)
            old=current-stride*slots;far=current-20*pointer_limit
            tracked=list(dict.fromkeys(i for i in [current-2,current-1,current+1,current+2,old,far] if i>0))
            ids=[0]+tracked;conditioning=[True]+[False]*len(tracked)
            arrays=[[] for _ in range(4)];output={'cond_frame_outputs':{},'non_cond_frame_outputs':{}}
            for index,is_cond in zip(ids,conditioning):
                storage='cpu' if index%2 else args.device
                memory=rand((batch,64,72,72),storage);memory_position=rand((batch,64,72,72),storage);pointer=rand((batch,256))
                confidence=torch.tensor(.9 if index!=old or case.get('high_score') else -.1,device=storage,dtype=dtype)
                entry=dict(maskmem_features=memory,maskmem_pos_enc=[memory_position],obj_ptr=pointer,eff_iou_score=confidence,
                           pred_masks=torch.ones(batch,1,1,1),pred_masks_high_res=torch.ones(batch,1,1,1),object_score_logits=torch.ones(batch,1),iou_score=torch.ones(batch))
                output['cond_frame_outputs' if is_cond else 'non_cond_frame_outputs'][index]=entry
                for array,tensor in zip(arrays,[memory,memory_position,pointer,confidence]):array.append(tensor)
            host.num_maskmem=slots;host.max_cond_frames_in_attn=2;host.max_obj_ptrs_in_encoder=pointer_limit;host.memory_temporal_stride_for_eval=stride
            host.use_memory_selection=score;host.non_overlap_masks_for_mem_enc=overlap;host.offload_output_to_cpu_for_eval=offload;host.trim_past_non_cond_mem_for_eval=trim
            host.multimask_output_in_sam=multi;host.multimask_output_for_tracking=tracking_multi;host.multimask_min_pt_num=0;host.multimask_max_pt_num=1
            pyramid=high+[image];sequence=[x.flatten(2).permute(2,0,1) for x in pyramid]
            positions=[None,None,position.flatten(2).permute(2,0,1)];sizes=[(288,288),(144,144),(72,72)]
            point_inputs=None if points is None else dict(point_coords=points,point_labels=labels)
            with source_transfer_device(args.device),torch.autocast(args.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                expected=host.track_step(current,initial,sequence,positions,sizes,None,point_inputs,mask,output,128,reverse,encode,previous,use_previous)
            settings=[current,128,slots,2,pointer_limit,stride,0,1]
            flags=[initial,reverse,use_previous,encode,score,overlap,offload,trim,multi,tracking_multi]
            actual=torch.ops.sam3_native.tracking_frame(str(args.store),image,position,high,points,labels,mask,previous,ids,conditioning,*arrays,settings,flags,mode)
            # Prompt metadata remains in TrackingFrameRequest/session storage;
            # compare the frame host's numerical outputs and state changes.
            targets={k:v for k,v in expected.items() if isinstance(v,torch.Tensor) and k not in ['point_inputs','mask_inputs']}
            if expected.get('maskmem_pos_enc') is not None:targets['maskmem_pos_enc']=expected['maskmem_pos_enc'][-1]
            history_status=[]
            for entries in output.values():
                for index,entry in entries.items():history_status.append([index]+[int(entry.get(k) is not None) for k in ['maskmem_features','maskmem_pos_enc','pred_masks_high_res','iou_score','eff_iou_score']])
            targets['history_status']=torch.tensor(history_status)
            assert actual.keys()==targets.keys(),(actual.keys(),targets.keys())
            errors={}
            for key,value in actual.items():
                torch.testing.assert_close(value,targets[key],rtol=0,atol=0)
                errors[key]=(value.float()-targets[key].float()).abs().max().item()
            row=dict(mode=mode,case=case['name'],mask_shape=list(actual['pred_masks'].shape),mask_device=str(actual['pred_masks'].device),
                     memory_encoded='maskmem_features' in actual,max_errors=errors,exact=True)
            results.append(row);print(json.dumps(row),flush=True)
    args.report.write_text(json.dumps(dict(torch=torch.__version__,device=args.device,cases=results,
        scope='Original SAM3 inference track_step: temporal conditioning, interactive/direct-mask heads, optional memory encoding, score outputs, CPU offload and history trimming. Synthetic cached image features; session prompt/consolidation policy is not tested. CPU source adapts CUDA transfers and rotary caches.'),indent=2)+'\n')


if __name__=='__main__':main()
