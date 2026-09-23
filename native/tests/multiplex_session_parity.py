"""Adapted original SAM3.1 demo workflows through the native session API.

Dynamic multi-object layouts use the native explicit dense-history policy and
are exercised separately; this test does not imply full upstream demo parity.
"""
import argparse,json,hashlib
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import torch
from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo,Sam3VideoTrackingMultiplexDemo
from sam3.model.multiplex_utils import MultiplexController,MultiplexState
from sam3.model.video_tracking_multiplex import VideoTrackingDynamicMultiplex
from multiplex_frame_parity import reference
from memory_attention_parity import backend_context
from temporal_memory_parity import source_transfer_device

@contextmanager
def layout_staging(mode):
    # Source singleton merge applies a device-resident mux matrix directly to
    # offloaded BF16 history. Restore its compute device, and F32 without AMP.
    originals={name:getattr(MultiplexState,name) for name in ['mux','demux']}
    def wrap(function):
        def apply(self,value):
            value=value.to(self.device)
            if mode=='fp32':value=value.float()
            return function(self,value)
        return apply
    for name,function in originals.items():setattr(MultiplexState,name,wrap(function))
    try:yield
    finally:
        for name,function in originals.items():setattr(MultiplexState,name,function)

def restore_annotation_indices(state):
    # Removing the sole object during source extraction clears consolidated
    # indices. Merge restores its inputs/history but omits these indices, so
    # the next source preflight fails its own input-frame equality assertion.
    inputs=set()
    for key in ['point_inputs_per_obj','mask_inputs_per_obj']:
        for values in state[key].values():inputs.update(values)
    for key in ['cond_frame_outputs','non_cond_frame_outputs']:state['consolidated_frame_inds'][key].clear()
    for index in inputs:
        key='cond_frame_outputs' if index in state['output_dict']['cond_frame_outputs'] else 'non_cond_frame_outputs'
        assert index in state['output_dict'][key]
        state['consolidated_frame_inds'][key].add(index)

def session_reference(weights,device):
    host=reference(weights,device)
    for name in VideoTrackingMultiplexDemo.__dict__:
        if name!='__init__' and callable(getattr(VideoTrackingMultiplexDemo,name)):setattr(host,name,MethodType(getattr(VideoTrackingMultiplexDemo,name),host))
    for name in ['recondition_masks_in_existing_state','add_new_masks_to_existing_state']:
        setattr(host,name,MethodType(getattr(VideoTrackingDynamicMultiplex,name),host))
    host.init_state=MethodType(Sam3VideoTrackingMultiplexDemo.init_state,host);host.device=torch.device(device);host.image_size=1008;host.input_mask_size=1152;host.low_res_mask_size=288;host.is_dynamic_model=True
    original_init=host.init_state
    def init_state(*args,**kwargs):
        state=original_init(*args,**kwargs)
        if device=='cpu':state['device']=torch.device('cpu');state['storage_device']=torch.device('cpu')
        return state
    host.init_state=init_state
    host.multiplex_controller=MultiplexController(16).eval();host.clear_non_cond_mem_around_input=False;host.clear_non_cond_mem_for_multi_obj=False;host.add_all_frames_to_correct_as_cond=True;host.always_start_from_first_ann_frame=False;host.fill_hole_area=0
    host.num_maskmem=3;host.max_obj_ptrs_in_encoder=4;host.max_cond_frames_in_attn=2;host.memory_temporal_stride_for_eval=1;host.offload_output_to_cpu_for_eval=False;host.trim_past_non_cond_mem_for_eval=False;host.save_image_features=True
    host.multimask_output_in_sam=True;host.multimask_output_for_tracking=True;host.multimask_min_pt_num=0;host.multimask_max_pt_num=1;host.stability_score_attentuation=False
    original_run=host._run_single_frame_inference;original_memory=host._run_memory_encoder
    def ready(function):
        def invoke(*args,**kwargs):
            result=function(*args,**kwargs)
            if device=='cuda':torch.cuda.synchronize()
            return result
        return invoke
    host._run_single_frame_inference=ready(original_run);host._run_memory_encoder=ready(original_memory)
    return host

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--modes',nargs='+',default=['fp16','fp32','bf16_reference']);p.add_argument('--cases',nargs='+');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(2918);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weights=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=True);host=session_reference(weights.get('model',weights),a.device)
    encoder=host.transformer.encoder;original_encoder=encoder.forward
    rows=[]
    point=torch.tensor([[.4,.6,1.]]);many=torch.cat([torch.rand(17,2),torch.ones(17,1)],1);box=torch.tensor([.15,.2,.75,.8]);mask=torch.zeros(37,53);mask[5:31,9:44]=1;empty=torch.empty(0)
    def op(kind,index=0,obj=101,x=0,y=0,z=0,u=0,v=0,payload=empty):return [kind,index,obj,x,y,z,u,v],payload
    scenarios=[('points',[op(0,x=1,y=1,payload=point),op(0,x=0,y=1,payload=many),op(1,index=1,x=1,y=1,payload=box),op(3,x=1),op(4,index=0,obj=2,y=1,z=0),op(7)],False),
               ('mask',[op(2,payload=mask),op(3,x=1),op(4,index=0,obj=2,y=1,z=0),op(7)],True),
               ('mask_batch',[([8,0,101,202],torch.stack([mask,mask.roll(8,1)])),op(3,x=1),op(4,index=0,obj=2,y=1,z=0),op(7)],False),
               ('refine',[op(0,x=1,y=1,payload=point),op(3,x=1),op(4,index=0,obj=2,y=1,z=0),op(0,index=1,x=1,y=1,payload=point),op(3,x=1),op(4,index=2,obj=2,x=1,y=1,z=0)],False)]
    for score in [False,True]:
        scenarios.append(('recondition'+('_score' if score else ''),[
            ([8,0,101,202,303],torch.stack([mask,mask.roll(8,1),mask.roll(12,0)])),
            op(3,x=1),op(4,index=0,obj=1,y=1,z=0),
            ([9,1,303,101],torch.stack([mask.roll(4,0),mask.roll(7,1)])),
            op(3,x=1),op(4,index=2,obj=0,y=1,z=0),
            ([9,2,202],mask.unsqueeze(0)),op(3,x=1),op(4,index=2,obj=2,x=1,y=1,z=0)],score))
    scenarios.append(('recondition_repeat',[
        ([8,0,101,202,303],torch.stack([mask,mask.roll(8,1),mask.roll(12,0)])),op(3,x=1),
        ([9,0,303],mask.roll(3,0).unsqueeze(0)),([9,0,101],mask.roll(4,1).unsqueeze(0)),op(3,x=1),
        ([9,0,202],mask.roll(9,0).unsqueeze(0)),op(3,x=1),op(4,index=0,obj=2,y=1,z=0)],False))
    prepared=torch.nn.functional.interpolate(mask[None,None],(1152,1152),mode='bilinear',align_corners=False)[0,0]>.5
    scenarios.append(('recondition_execute',[
        ([8,0,101,202,303],torch.stack([mask,mask.roll(8,1),mask.roll(12,0)])),
        op(3,x=1),op(4,index=0,obj=1,y=1,z=0),
        ([10,1,303,101],torch.stack([prepared.roll(47,0),prepared.roll(103,1)])),
        op(4,index=2,obj=0,y=1,z=0),
        ([10,2,202],prepared.unsqueeze(0)),op(4,index=2,obj=2,x=1,y=1,z=0)],False))
    for mode in a.modes:
        dtype={'fp16':torch.float16,'fp32':torch.float32,'bf16_reference':torch.bfloat16}[mode]
        def attention(**kwargs):
            if mode=='fp32':kwargs['memory']=kwargs['memory'].float()
            return original_encoder(**kwargs)
        encoder.forward=attention
        arrays=[]
        for index in range(3):
            values=[]
            for neck in range(2):
                for shape in [(1,256,72,72),(1,256,72,72),(1,32,288,288),(1,64,144,144)]:values.append(torch.randn(shape,device=a.device,dtype=dtype))
            arrays.append(values)
        def features(self,state,index,batch):
            values=arrays[index%len(arrays)];out={}
            for name,offset in [('interactive',0),('sam2_backbone_out',4)]:
                image,position,h0,h1=values[offset:offset+4];out[name]=dict(vision_feats=[x.flatten(2).permute(2,0,1) for x in [h0,h1,image]],vision_masks=[None]*3,vision_pos_embeds=[None,None,position.flatten(2).permute(2,0,1)],feat_sizes=[(288,288),(144,144),(72,72)])
            return None,out
        host._get_image_feature=MethodType(features,host)
        for name,operations,score in scenarios:
            if a.cases and name not in a.cases:continue
            print('START',mode,name,flush=True);host.non_overlap_masks_for_output=False;host.use_memory_selection=score;host.non_overlap_masks_for_mem_enc=False
            expected={};offload=not name.startswith('recondition')
            with backend_context(a.device=='cuda' and mode=='fp32',False),source_transfer_device(a.device),layout_staging(mode):
                state=host.init_state(video_height=37,video_width=53,num_frames=4,offload_state_to_cpu=offload)
                state['device']=torch.device(a.device);state['storage_device']=torch.device('cpu' if offload else a.device)
                with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                    for i,(operation,payload) in enumerate(operations):
                        count=0
                        def emit(frame,ids,low,masks):
                            nonlocal count
                            key=f'{i}/out{count}/';count+=1;expected[key+'masks']=masks.cpu().clone();expected[key+'ids']=torch.tensor(ids);expected[key+'frame']=torch.tensor(frame)
                            if low is not None:expected[key+'low']=low.cpu().clone()
                        if operation[0] in [0,1]:
                            refining=operation[1] in state['frames_already_tracked']
                            pts=payload[:,:2] if operation[0]==0 else payload.reshape(2,2);labels=payload[:,2].int() if operation[0]==0 else torch.tensor([2,3],dtype=torch.int32)
                            emit(*host.add_new_points(state,operation[1],operation[2],pts,labels,bool(operation[3]),rel_coordinates=bool(operation[4])))
                            if refining:restore_annotation_indices(state)
                        elif operation[0]==2:emit(*host.add_new_masks(state,operation[1],[operation[2]],payload.unsqueeze(0)))
                        elif operation[0] in [8,9]:emit(*host.add_new_masks(state,operation[1],operation[2:],payload,reconditioning=operation[0]==9))
                        elif operation[0]==10:
                            host.add_new_masks(state,operation[1],operation[2:],payload>0,reconditioning=True)
                            host.propagate_in_video_preflight(state,True)
                            expected[f'{i}/affected']=torch.tensor(sorted(state['obj_ids']),dtype=torch.long)
                        elif operation[0]==3:host.propagate_in_video_preflight(state,bool(operation[3]))
                        elif operation[0]==4:
                            for value in host.propagate_in_video(state,operation[1],operation[2],bool(operation[3]),tqdm_disable=True,run_mem_encoder=bool(operation[4])):emit(*value)
                        elif operation[0]==7:host.clear_all_points_in_video(state)
                        expected[f'{i}/outputs']=torch.tensor(count)
                        if name.startswith('recondition'):
                            for source,target in [('cond_frame_outputs','cond'),('non_cond_frame_outputs','tracked')]:
                                for frame,entry in state['output_dict'][source].items():
                                    for key,field in [('low','pred_masks'),('memory','maskmem_features'),('pointer','obj_ptr'),('logits','object_score_logits')]:
                                        value=entry.get(field)
                                        if value is not None:expected[f'{i}/{target}/{frame}/{key}']=value.cpu().clone()
                                    value=entry.get('maskmem_pos_enc')
                                    if value is not None:expected[f'{i}/{target}/{frame}/position']=value[-1].cpu().clone()
                                    expected[f'{i}/{target}/{frame}/conditions']=torch.tensor(sorted(entry.get('conditioning_objects',set())),dtype=torch.long)
                    if a.device=='cuda':torch.cuda.synchronize()
                actual=torch.ops.sam3_native.multiplex_session(str(a.store),arrays,[item[0] for item in operations],[item[1] for item in operations],[4,37,53],[offload,False,True,score,False],mode)
            for i,(operation,_) in enumerate(operations):
                if operation[0] in [3,10]:
                    assert not any(key.startswith(f'{i}/video/') for key in actual), 'consolidated frame retained temporary previews'
            for key,value in expected.items():
                try:torch.testing.assert_close(actual[key].sort().values if key.endswith('/conditions') else actual[key],value,rtol=0,atol=0)
                except AssertionError as error:raise AssertionError(f'{mode} {name} {key}: {error}') from error
            rows.append(dict(mode=mode,case=name,operations=len(operations),outputs=len(expected),exact=True));print(json.dumps(rows[-1]),flush=True)
    a.report.write_text(json.dumps(dict(device=a.device,torch=torch.__version__,cases=rows,source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['sam3/model/video_tracking_multiplex.py','sam3/model/video_tracking_multiplex_demo.py']},scope='Adapted original demo neural outputs and, for reconditioning cases, stored masks/memory/pointers/scores/conditions after each operation. Includes three-object reordered/repeated correction, preflight, forward/reverse propagation and native executor. Reconditioning source comparisons keep state on the compute device: the original offloaded scatter fails with CPU/CUDA mismatch. Native resident/offloaded/paged equivalence is checked separately. Dynamic insertion/removal layout policy is tested separately.',adaptations='Synthetic projected features; source CPU CUDA transfers and every nested session constructor redirected to CPU, CUDA FP32 Flash-only context removed, FP32 compressed memory restored before projection, D2H synchronized. Source mux/demux explicitly stage offloaded tensors to the matrix device and restore F32 when outside AMP; unadapted CUDA refinement failed on CPU-vs-CUDA matmul. Source consolidated annotation indices are restored from the merged inputs/history after singleton refinement; without this, next preflight fails its equality assertion. These adaptations preserve neural equations.'),indent=2)+'\n')
if __name__=='__main__':main()
