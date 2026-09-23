"""Compare complete interactive SAM3 operation sequences and stored state.

The original predictor methods are bound without its constructor's permanent
CUDA autocast context or visual backbone; both implementations receive identical
full 72/144/288 cached visual features. No session/temporal math is replaced.
"""
import argparse
import gc
import json
import random
from pathlib import Path
from types import MethodType,SimpleNamespace
import numpy as np
from sam3.model.sam3_video_base import Sam3VideoBase
import torch
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores
from tracking_frame_parity import reference
from temporal_memory_parity import source_transfer_device


def snapshot(state, prefix, out):
    def save(key, value):
        if value is not None:out[prefix+key]=value.cpu().clone()
    def frame(key, entry):
        for name in ['pred_masks','obj_ptr','object_score_logits','maskmem_features','maskmem_pos_enc','iou_score','eff_iou_score','pred_masks_video_res']:
            value=entry.get(name)
            if name=='maskmem_pos_enc' and value is not None:value=value[-1]
            save(key+name,value)
    def history(key, entries):
        for source,target in [('cond_frame_outputs','cond/'),('non_cond_frame_outputs','tracked/')]:
            save(key+target+'order',torch.tensor(list(entries[source]),dtype=torch.int64))
            for index,entry in entries[source].items():frame(key+target+str(index)+'/',entry)
    save('ids',torch.tensor(state['obj_ids'],dtype=torch.int64))
    save('status',torch.tensor([int(state['tracking_has_started']),state['first_ann_frame_idx'] if state['first_ann_frame_idx'] is not None else -1]))
    save('direction',torch.tensor([[i,int(v['reverse'])] for i,v in sorted(state['frames_already_tracked'].items())],dtype=torch.int64).reshape(-1,2))
    history('global/',state['output_dict'])
    for source,target in [('cond_frame_outputs','cond'),('non_cond_frame_outputs','tracked')]:
        save('consolidated_'+target,torch.tensor(sorted(state['consolidated_frame_inds'][source]),dtype=torch.int64))
    for i in range(len(state['obj_ids'])):
        root=f'obj{i}/'
        for index,point in state['point_inputs_per_obj'][i].items():
            save(root+f'points/{index}/coords',point['point_coords']);save(root+f'points/{index}/labels',point['point_labels'])
        for index,mask in state['mask_inputs_per_obj'][i].items():save(root+f'masks/{index}',mask)
        history(root,state['output_dict_per_obj'][i])
        for source,target in [('cond_frame_outputs','pending_cond/'),('non_cond_frame_outputs','pending_tracked/')]:
            for index,entry in state['temp_output_dict_per_obj'][i][source].items():frame(root+target+str(index)+'/',entry)


def run_reference(host, arrays, operations, payloads, settings, flags, mode, device):
    host.device=torch.device(device);host.input_mask_size=1152;host.low_res_mask_size=288
    for name in Sam3TrackerPredictor.__dict__:
        if name!='__init__' and callable(getattr(Sam3TrackerPredictor,name)):
            setattr(host,name,MethodType(getattr(Sam3TrackerPredictor,name),host))
    def features(self,state,index,batch):
        index%=len(arrays[0]);image,position,h0,h1=[x[index].expand(batch,-1,-1,-1) for x in arrays]
        return None,None,[x.flatten(2).permute(2,0,1) for x in [h0,h1,image]],[None,None,position.flatten(2).permute(2,0,1)],[(288,288),(144,144),(72,72)]
    host._get_image_feature=MethodType(features,host)
    encoder=host.transformer.encoder
    if not hasattr(encoder,'session_original_forward'):encoder.session_original_forward=encoder.forward
    def encoder_forward(**kwargs):
        # Original predictor keeps CUDA BF16 autocast permanently enabled.
        # Without autocast, its mandatory BF16 session storage causes a linear
        # dtype error. Restore only the compressed memory values for FP32 tests.
        if mode=='fp32':kwargs['prompt']=kwargs['prompt'].float()
        return encoder.session_original_forward(**kwargs)
    encoder.forward=encoder_forward
    original_run=host._run_single_frame_inference
    def run_ready(*args,**kwargs):
        result=original_run(*args,**kwargs)
        # Ensure source asynchronous D2H copies finish before CPU consolidation.
        if device=='cuda':torch.cuda.synchronize()
        return result
    host._run_single_frame_inference=run_ready
    host.non_overlap_masks_for_output=flags[1];host.clear_non_cond_mem_around_input=flags[2];host.clear_non_cond_mem_for_multi_obj=flags[3]
    host.add_all_frames_to_correct_as_cond=flags[4];host.always_start_from_first_ann_frame=flags[5];host.use_memory_selection=flags[6];host.non_overlap_masks_for_mem_enc=flags[7]
    host.max_point_num_in_prompt_enc=settings[3];host.fill_hole_area=settings[4];host.num_maskmem=settings[5];host.max_obj_ptrs_in_encoder=settings[6]
    host.max_cond_frames_in_attn=2;host.memory_temporal_stride_for_eval=1;host.offload_output_to_cpu_for_eval=False;host.trim_past_non_cond_mem_for_eval=False
    host.multimask_output_in_sam=True;host.multimask_output_for_tracking=True;host.multimask_min_pt_num=0;host.multimask_max_pt_num=1
    state=host.init_state(video_height=settings[1],video_width=settings[2],num_frames=settings[0],offload_state_to_cpu=flags[0])
    if device=='cpu':state['storage_device']=torch.device('cpu')
    out={}
    for i,(op,payload) in enumerate(zip(operations,payloads)):
        prefix=f'{i}/';count=0
        def emit(index,ids,low,masks,logits=None):
            nonlocal count
            key=prefix+f'out{count}/';count+=1
            out[key+'frame']=torch.tensor([index]);out[key+'ids']=torch.tensor(ids,dtype=torch.int64);out[key+'masks']=masks.cpu().clone()
            if low is not None:out[key+'low']=low.cpu().clone()
            if logits is not None:out[key+'logits']=logits.cpu().clone()
        if op[0] in [0,1]:
            kwargs=dict(points=payload[:,:2],labels=payload[:,2].int()) if op[0]==0 else dict(box=payload)
            emit(*host.add_new_points_or_box(state,op[1],op[2],clear_old_points=bool(op[3]),rel_coordinates=bool(op[4]),use_prev_mem_frame=bool(op[5]),**kwargs))
        elif op[0]==2:emit(*host.add_new_mask(state,op[1],op[2],payload))
        elif op[0]==8:
            Sam3VideoBase._recondition_masklets(SimpleNamespace(tracker=host),op[1],{'mask':payload.float().unsqueeze(0)},
                {op[2]:0},[state],{'obj_ids_all_gpu':np.array(state['obj_ids'])},torch.full((len(state['obj_ids']),),2.,device=device))
            out[prefix+'affected']=torch.tensor(sorted(state['obj_ids']),dtype=torch.long)
        elif op[0]==3:host.propagate_in_video_preflight(state,bool(op[3]))
        elif op[0]==4:
            iterator=host.propagate_in_video(state,None if op[1]<0 else op[1],None if op[2]<0 else op[2],bool(op[3]),tqdm_disable=True,run_mem_encoder=bool(op[4]),propagate_preflight=bool(op[5]))
            for value in iterator:
                emit(*value)
                if op[6]>0 and count>=op[6]:break
            iterator.close()
        elif op[0]==5:emit(*host.clear_all_points_in_frame(state,op[1],op[2]))
        elif op[0]==6:
            ids,updated=host.remove_object(state,op[2],strict=bool(op[3]))
            for index,masks in updated:emit(index,ids,None,masks)
        elif op[0]==7:host.clear_all_points_in_video(state)
        if device=='cuda':torch.cuda.synchronize()
        out[prefix+'outputs']=torch.tensor([count]);snapshot(state,prefix+'state/',out)
    return out


def scenarios(device):
    device="cpu" # Original box helper creates empty point tensors on CPU.
    point=torch.tensor([[.37,.49,1.]],device=device);more=torch.cat([torch.rand(17,2,device=device),torch.ones(17,1,device=device)],1)
    box=torch.tensor([.15,.2,.73,.84],device=device);mask=torch.zeros(37,53,device=device);mask[4:30,8:42]=1
    empty=torch.empty(0,device=device)
    def op(kind,frame=0,obj=0,a=0,b=0,c=0,d=0,e=0,payload=empty):return [kind,frame,obj,a,b,c,d,e],payload
    common=[op(0,0,101,1,1,payload=point),op(0,0,101,0,1,payload=more),op(1,1,202,1,1,payload=box),
            op(2,1,303,payload=mask),op(2,1,202,payload=mask.flip(0)),op(3,a=1),op(4,0,3,0,1),
            op(0,2,101,1,1,1,payload=point),op(4,3,3,1,1,1),op(5,2,101),op(6,obj=303),
            op(4,0,3,0,1,1,1,1),op(4,1,2,0,1,0,1),op(4,2,1,0,1),op(7),
            op(0,2,999,1,1,payload=point),op(4,2,0,0,1,1),op(6,obj=999)]
    # Sparse frame indices expose source set iteration order, affecting ordered
    # conditioning memory even when all individual prompt outputs match.
    sparse=[op(0,i,101,1,1,payload=point) for i in [17,8,1]]+[op(3,a=1),op(4,9,0,0,1)]
    basic=[op(0,0,101,1,1,payload=point),op(4,0,2,0,1,1),op(0,1,101,1,1,1,payload=point),op(4,2,2,1,1,1),op(5,0,101),op(7)]
    prepared=torch.nn.functional.interpolate(mask[None,None],(1152,1152),mode='bilinear',align_corners=False)[0,0]>.5
    recondition=[op(2,0,101,payload=mask),op(2,0,202,payload=mask.roll(8,1)),op(3,a=1),
                 op(4,0,1,0,1),op(8,1,202,payload=prepared),op(4,2,0,0,1),
                 op(8,2,101,payload=prepared.roll(79,0)),op(4,2,2,1,1)]
    return [('recondition_execute',recondition,[True,False,False,False,True,False,False,False],3),('multi_edit' ,common,[True,False,True,False,True,False,False,False],4),
            ('scores_and_noncond',basic,[False,True,True,True,False,False,True,True],3),
            ('sparse_order',sparse,[True,False,True,False,True,True,False,False],20)]


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('checkpoint',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--cases',nargs='+');p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(1519)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    rng=random.Random(1519);order_cases=0
    for dictionary in [False,True]:
        groups_to_test=[[[17,8,1]],[[0,32,64,96,128],[96,1,33,65]],
                       [list(range(51000,0,-1))],[[2**61-1,2**61,2**62,2**63-1,0,1]]]
        groups_to_test += [[[rng.randrange(5000) for _ in range(rng.randrange(80))] for _ in range(rng.randrange(1,6))] for _ in range(300)]
        for groups in groups_to_test:
            expected=set()
            for group in groups:
                mapping=dict.fromkeys(group)
                expected.update(mapping if dictionary else mapping.keys())
            # Source dictionaries cannot contain duplicate keys in one group.
            unique=[list(dict.fromkeys(group)) for group in groups]
            actual=torch.ops.sam3_native.tracking_frame_order(unique,dictionary)
            assert actual==list(expected),(dictionary,groups,actual,list(expected))
            order_cases+=1
    weights=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True);host=reference(weights.get('model',weights),a.device);rows=[]
    for mode in a.modes:
        dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
        arrays=[[torch.randn(1,c,s,s,device=a.device,dtype=dtype).contiguous(memory_format=torch.channels_last) for _ in range(2)] for c,s in [(256,72),(256,72),(32,288),(64,144)]]
        for name,sequence,flags,frames in scenarios(a.device):
            if a.cases and name not in a.cases:continue
            operations,payloads=map(list,zip(*sequence));settings=[frames,31,47,0,0,3,4]
            print(f'START {mode} {name}',flush=True)
            with source_transfer_device(a.device),torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                expected=run_reference(host,arrays,operations,payloads,settings,flags,mode,a.device)
            actual=torch.ops.sam3_native.tracking_session(str(a.store),*arrays,operations,payloads,settings,flags,mode)
            assert actual.keys()==expected.keys(),(set(actual)-set(expected),set(expected)-set(actual))
            for key,value in actual.items():
                try:torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
                except AssertionError as error:raise AssertionError(f'{mode} {name} {key}: {error}') from error
            row=dict(mode=mode,case=name,operations=len(operations),tensors=len(actual),exact=True);print(json.dumps(row),flush=True);rows.append(row)
            del actual,expected;gc.collect()
    # Independent postprocessing reference uses original CPU/skimage CCL, so
    # the expected result never calls the new native connected-component kernel.
    postprocess=0
    for shape in [(2,1,31,47),(1,1,32,48)]:
        for area in [0,1,5,20]:
            masks=torch.randn(shape);masks[...,3:5,4:6]=1
            for overlap in [False,True]:
                expected=torch.nn.functional.interpolate(masks,(35,49),mode='bilinear',align_corners=False)
                if overlap:expected=host._apply_non_overlapping_constraints(expected)
                if area:expected=fill_holes_in_mask_scores(expected,area)
                # Resampling CPU and CUDA has ordinary F32 rounding differences;
                # compare postprocess itself on identically pre-resampled logits.
                resized=torch.nn.functional.interpolate(masks,(35,49),mode='bilinear',align_corners=False)
                actual=torch.ops.sam3_native.tracking_postprocess(resized.to(a.device),35,49,overlap,area).cpu()
                torch.testing.assert_close(actual,expected,rtol=0,atol=0);postprocess+=1
    a.report.write_text(json.dumps(dict(device=a.device,torch=torch.__version__,cases=rows,postprocess_exact=postprocess,frame_order_exact=order_cases,
        adaptations=['FP32 restores BF16 stored memory to float at the memory encoder input; unmodified source fails without permanent autocast.',
                     'Wait for source D2H copies before CPU consolidation/snapshots to avoid timing-dependent reads.',
                     'CPU redirects original hard-coded CUDA transfers and rotary caches to CPU.'],
        scope='Original SAM3 low-level interactive tracker session, cached synthetic full-grid features; not the high-level text video tracker or real-video end-to-end validation.'),indent=2)+'\n')


if __name__=='__main__':main()
