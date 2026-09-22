"""Original SAM3.1 inference frame routing, partial correction and memory writes."""
import argparse,json
from pathlib import Path
from types import MethodType,SimpleNamespace
import torch
from sam3.model.video_tracking_multiplex import VideoTrackingMultiplex
from sam3.model.multiplex_utils import MultiplexController
from sam3.model_builder import _create_multiplex_transformer
from video_heads_parity import reference as interactive_reference
from multiplex_decoder_parity import reference as propagation_reference
from memory_encoder_parity import reference as memory_reference
from memory_attention_parity import backend_context
from temporal_memory_parity import source_transfer_device

def reference(weights,device):
    host=interactive_reference(weights,'sam3.1',device)
    host.__dict__.update(propagation_reference(weights,device).__dict__)
    host.__dict__.update(memory_reference(weights,'sam3.1',device)[1].__dict__)
    encoder=_create_multiplex_transformer().encoder.eval();prefix='tracker.model.transformer.encoder.'
    encoder.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True);encoder.to(device)
    if device=='cpu':
        for module in encoder.modules():
            if hasattr(module,'compute_cis'):module.freqs_cis=module.compute_cis(end_x=72,end_y=72,device='cpu')
    projection=torch.nn.Linear(256,256).eval();prefix='tracker.model.obj_ptr_tpos_proj.'
    projection.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True);projection.to(device)
    host.transformer=SimpleNamespace(encoder=encoder);host.obj_ptr_tpos_proj=projection
    for name in ['interactivity_no_mem_embed','maskmem_tpos_enc']:setattr(host,name,weights['tracker.model.'+name].to(device))
    for name in ['_forward_sam_heads','_use_mask_as_output','_encode_new_memory','_apply_non_overlapping_constraints','get_propagation_dense_pe',
                 '_get_interactive_pix_mem','_prepare_memory_conditioned_features','_get_tpos_enc','frame_filter',
                 '_track_step_aux','_trim_output_and_memory','track_step','_use_multimask','cal_mem_score']:
        setattr(host,name,MethodType(getattr(VideoTrackingMultiplex,name),host))
    host.directly_add_no_mem_embed=True;host.use_mask_input_as_output_without_sam=True;host.iter_use_prev_mask_pred=True
    host.mem_dim=256;host.sincos_tpos_enc=True;host.proj_tpos_enc_in_obj_ptrs=True;host.mf_threshold=.01
    host.keep_first_cond_frame=False;host.only_obj_ptrs_in_the_past_for_eval=False;host.use_signed_tpos_enc_to_obj_ptrs=False
    host.use_maskmem_tpos_v2=True;host.add_tpos_enc_to_obj_ptrs=True;host.num_multimask_outputs=3
    return host

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('checkpoint',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--cases',nargs='+');p.add_argument('--math',action='store_true');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(1821)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    weights=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True);host=reference(weights.get('model',weights),a.device);rows=[]
    cases=[dict(name='initial_points',initial=True,points=1),dict(name='initial_17_points',initial=True,points=17),
           dict(name='initial_empty_points',initial=True,points=0),dict(name='initial_mask_1152',initial=True,mask=1152),
           dict(name='initial_mask_1008',initial=True,mask=1008),dict(name='refine_previous',points=2,previous=True),
           dict(name='forward',objects=17 if a.device=='cuda' else 2),dict(name='reverse',reverse=True),
           dict(name='propagate_correct_multi',points=1,partial=True),dict(name='propagate_correct_single',points=4,partial=True),
           dict(name='deferred',initial=True,points=1,encode=False),dict(name='interactive_only_features',initial=True,points=1,encode=False,save=False),
           dict(name='scores',score=True),dict(name='offloaded',score=True,offload=True),dict(name='overlap',initial=True,points=1,overlap=True),
           dict(name='stability',points=1,partial=True,stability=True),dict(name='disabled_memory',initial=True,slots=0),
           dict(name='trim_low',initial=True,points=1,trim=True,score=True,current=85),
           dict(name='trim_high',initial=True,points=1,trim=True,score=True,current=85,high_score=True),
           dict(name='trim_offload',initial=True,points=1,trim=True,score=True,current=85,offload=True),
           dict(name='trim_unfiltered',initial=True,points=1,trim=True,current=85)]
    for mode in a.modes:
        dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
        for case in cases:
            if a.cases and case['name'] not in a.cases:continue
            print(f'START {mode} {case["name"]}',flush=True)
            count=case.get('objects',3);current=case.get('current',12);slots=case.get('slots',3);ptr_limit=4
            if a.math and count>3:count=3 # keep the math fallback comparison within GPU memory
            buckets=MultiplexController(16).eval().get_state(count,torch.device(a.device),torch.float32,random=True);batch=buckets.num_buckets
            def rand(shape,device=a.device):return torch.randn(shape,device=device,dtype=dtype)
            features=[]
            for _ in range(2):features += [rand((1,256,72,72)).contiguous(memory_format=torch.channels_last),rand((1,256,72,72)),
                rand((1,32,288,288)).contiguous(memory_format=torch.channels_last),rand((1,64,144,144)).contiguous(memory_format=torch.channels_last)]
            points=labels=mask=previous=None;objects=None
            if 'points' in case:
                if case.get('partial'):objects=[2,0]
                elif case.get('previous'):objects=list(range(count))
                point_batch=len(objects) if objects is not None else count
                points=torch.rand(point_batch,case['points'],2,device=a.device)*1008;labels=torch.randint(-1,4,(point_batch,case['points']),device=a.device,dtype=torch.int32)
            if case.get('mask'):mask=rand((count,1,case['mask'],case['mask']))>0;mask[0].zero_()
            if case.get('previous'):previous=rand((count,1,288,288)).float().clamp(-32,32)
            initial=case.get('initial',False);reverse=case.get('reverse',False);encode=case.get('encode',True);score=case.get('score',False)
            offload=case.get('offload',False);trim=case.get('trim',False);save=case.get('save',True);overlap=case.get('overlap',False);stability=case.get('stability',False)
            ids=[0,current-1,current-slots,current-20*ptr_limit];ids=list(dict.fromkeys(i for i in ids if i>=0));conditioning=[i==0 for i in ids]
            arrays=[[] for _ in range(6)];output=dict(cond_frame_outputs={},non_cond_frame_outputs={})
            for index,is_cond in zip(ids,conditioning):
                storage='cpu' if index%2 else a.device
                memory=rand((batch,256,72,72),storage);position=rand((batch,256,72,72),storage);pointer=rand((batch,16,256))
                confidence=torch.tensor(.9 if index!=current-slots or case.get('high_score') else -.1,device=storage,dtype=dtype)
                image=rand((72*72,1,256),storage);image_pos=rand((72*72,1,256),storage)
                entry=dict(conditioning_objects=set(),maskmem_features=memory,maskmem_pos_enc=[position],obj_ptr=pointer,eff_iou_score=confidence,
                    image_features=image,image_pos_enc=image_pos,pred_masks=torch.ones(count,1,1,1),pred_masks_high_res=torch.ones(count,1,1,1),
                    object_score_logits=torch.ones(count,1),iou_score=torch.ones(count),multistep_point_inputs=[None],multistep_pred_multimasks=[torch.ones(count,1,1,1)])
                output['cond_frame_outputs' if is_cond else 'non_cond_frame_outputs'][index]=entry
                for array,value in zip(arrays,[memory,position,pointer,confidence,image,image_pos]):array.append(value)
            host.num_maskmem=slots;host.max_cond_frames_in_attn=2;host.max_obj_ptrs_in_encoder=ptr_limit;host.memory_temporal_stride_for_eval=1
            host.use_memory_selection=score;host.offload_output_to_cpu_for_eval=offload;host.trim_past_non_cond_mem_for_eval=trim;host.save_image_features=save
            host.non_overlap_masks_for_mem_enc=overlap;host.stability_score_attentuation=stability
            host.multimask_output_in_sam=True;host.multimask_output_for_tracking=True;host.multimask_min_pt_num=0;host.multimask_max_pt_num=1
            def backbone(offset):
                image,position,h0,h1=features[offset:offset+4]
                return dict(vision_feats=[x.flatten(2).permute(2,0,1) for x in [h0,h1,image]],vision_masks=[None]*3,
                    vision_pos_embeds=[None,None,position.flatten(2).permute(2,0,1)],feat_sizes=[(288,288),(144,144),(72,72)])
            with backend_context((a.device=='cuda' and mode=='fp32') or a.math,a.math),source_transfer_device(a.device):
                with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                    expected=host.track_step(frame_idx=current,is_init_cond_frame=initial,backbone_features_interactive=backbone(0),
                        backbone_features_propagation=backbone(4) if encode or save else None,image=None,
                        point_inputs=dict(point_coords=points,point_labels=labels) if points is not None else None,mask_inputs=mask,
                        gt_masks=None,frames_to_add_correction_pt=[],output_dict=output,num_frames=128,track_in_reverse=reverse,
                        run_mem_encoder=encode,prev_sam_mask_logits=previous,multiplex_state=buckets,objects_to_interact=objects)
                settings=[current,128,slots,2,ptr_limit,1,0,1];flags=[initial,reverse,encode,score,overlap,offload,trim,save,True,True,stability]
                actual=torch.ops.sam3_native.multiplex_frame(str(a.store),features,points,labels,mask,previous,objects,buckets.assignments,ids,conditioning,*arrays,settings,flags,0.,mode)
            targets={k:expected[k] for k in ['pred_masks','pred_masks_high_res','object_score_logits','obj_ptr','maskmem_features','image_features','image_pos_enc','iou_score','eff_iou_score'] if k in expected}
            if expected.get('maskmem_pos_enc') is not None:targets['maskmem_pos_enc']=expected['maskmem_pos_enc'][-1]
            for key in ['multistep_pred_multimasks','multistep_pred_multimasks_high_res','multistep_pred_ious']:
                if key in expected:targets[key]=expected[key][-1]
            targets['conditioning_objects']=torch.tensor(sorted(expected['conditioning_objects']),dtype=torch.int64)
            status=[]
            for group in output.values():
                for index,entry in group.items():status.append([index]+[int(entry.get(k) is not None) for k in ['maskmem_features','maskmem_pos_enc','image_features','image_pos_enc','pred_masks_high_res','iou_score','eff_iou_score','multistep_pred_multimasks']])
            targets['history_status']=torch.tensor(status,dtype=torch.int64)
            assert actual.keys()==targets.keys(),(set(actual)-set(targets),set(targets)-set(actual))
            for key,value in actual.items():
                try:torch.testing.assert_close(value,targets[key],rtol=0,atol=0)
                except AssertionError as error:raise AssertionError(f'{mode} {case["name"]} {key}: {error}') from error
            row=dict(mode=mode,case=case['name'],objects=count,buckets=batch,outputs=len(actual),exact=True)
            rows.append(row);print(json.dumps(row),flush=True)
    a.report.write_text(json.dumps(dict(torch=torch.__version__,device=a.device,cases=rows,math_fallback=a.math,
        scope='Original VideoTrackingMultiplex.track_step inference with no ground-truth training correction loop. Synthetic full-grid cached features; dynamic insert/recondition and demo session singleton extraction/merge remain separate.',
        adaptations='CUDA FP32 removes the source Flash-only context. CPU redirects CUDA transfers and rotary caches.'),indent=2)+'\n')
if __name__=='__main__':main()
