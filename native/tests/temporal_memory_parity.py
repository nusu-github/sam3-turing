"""SAM3 frame selection, memory assembly and conditioned-feature parity."""
import argparse,json,random
from contextlib import contextmanager
from pathlib import Path
from types import MethodType,SimpleNamespace
import torch
from sam3.model.sam3_tracker_base import Sam3TrackerBase
from sam3.model.sam3_tracker_utils import select_closest_cond_frames
from sam3.model_builder import _create_tracker_transformer

@contextmanager
def source_transfer_device(device):
    # Original host hard-codes .cuda() for offloaded spatial memories. For CPU
    # comparison, adapt that transfer destination only, preserving all math.
    original=torch.Tensor.cuda
    if device=='cpu':torch.Tensor.cuda=lambda self,*args,**kwargs:self.to('cpu')
    try:yield
    finally:torch.Tensor.cuda=original

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('checkpoint',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(619);rng=random.Random(619)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    selection_cases=0
    for _ in range(600):
        current=rng.randrange(30);order=rng.sample(range(30),rng.randrange(0,18));limit=rng.choice([-1,2,3,4,8,32]);keep=rng.choice([False,True])
        expected=select_closest_cond_frames(current,{i:i for i in order},limit,keep)
        actual=torch.ops.sam3_native.select_conditioning_frames(current,order,limit,keep)
        assert tuple(map(list,expected))==actual,(current,order,limit,keep,expected,actual)
        selection_cases+=1
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True);state=state.get('model',state)
    encoder=_create_tracker_transformer().encoder.eval();prefix='tracker.transformer.encoder.'
    encoder.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);encoder.to(a.device)
    if a.device=='cpu':
        for child in encoder.modules():
            if hasattr(child,'compute_cis'):child.freqs_cis=child.compute_cis(end_x=72,end_y=72,device='cpu')
    projection=torch.nn.Linear(256,64).eval();prefix='tracker.obj_ptr_tpos_proj.'
    projection.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);projection.to(a.device)
    capture={}
    def record(**kwargs):
        capture['memory']=kwargs['prompt'].clone();capture['position']=kwargs['prompt_pos'].clone();capture['pointers']=kwargs['num_obj_ptr_tokens']
        return encoder(**kwargs)
    host=SimpleNamespace(hidden_dim=256,mem_dim=64,training=False,num_maskmem=7,max_obj_ptrs_in_encoder=16,
        max_cond_frames_in_attn=4,keep_first_cond_frame=False,memory_temporal_stride_for_eval=1,use_memory_selection=False,mf_threshold=.01,
        transformer=SimpleNamespace(encoder=record),obj_ptr_tpos_proj=projection,
        no_mem_embed=state['tracker.no_mem_embed'].to(a.device),maskmem_tpos_enc=state['tracker.maskmem_tpos_enc'].to(a.device))
    for method in ['_prepare_memory_conditioned_features','_get_tpos_enc','frame_filter','cal_mem_score']:
        setattr(host,method,MethodType(getattr(Sam3TrackerBase,method),host))
    for name in ['cond_frame_spatial_embedding','cond_frame_obj_ptr_embedding']:
        if 'tracker.'+name in state:setattr(host,name,state['tracker.'+name].to(a.device))
    results=[]
    for mode in a.modes:
        dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
        cases=[('forward',8,12,False,False,True,7,4,16,1,False,False),
               ('reverse_stride',8,12,True,False,True,7,4,16,3,True,False),
               ('score_filter',8,12,False,False,True,7,4,16,2,False,True),
               ('score_filter_reverse',8,12,True,False,True,7,2,8,2,True,True),
               ('near_start',8,1,False,False,True,7,4,16,3,False,False),
               ('at_start_filtered',8,0,False,False,True,7,4,16,1,False,True),
               ('at_end_filtered',8,23,True,False,True,7,4,16,1,False,True),
               ('unlimited_cond',8,12,False,False,True,7,-1,4,1,False,False),
               ('initial',8,12,False,True,True,7,4,16,1,False,False),
               ('skip_previous',8,12,False,False,False,7,4,16,1,False,False),
               ('memory_disabled',8,12,False,False,True,0,4,16,1,False,False),
               ('full_grid',72,12,False,False,True,3,2,4,1,False,False)]
        for name,side,current,reverse,initial,use_previous,slots,limit,pointer_limit,stride,keep,filter_scores in cases:
            batch=1 if side==72 else 2
            def rand(shape,device=a.device):return torch.randn(*shape,device=device,dtype=dtype)
            source=rand((batch,256,side,side)).flatten(2).permute(2,0,1);source_pos=rand((batch,256,side,side)).flatten(2).permute(2,0,1)
            cond_ids=[17,0,8,23,14,3,12];tracked_ids=[i for i in range(24) if i not in cond_ids and i%5!=0]
            rng.shuffle(tracked_ids);ids=cond_ids+tracked_ids;is_cond=[True]*len(cond_ids)+[False]*len(tracked_ids)
            features=[];positions=[];pointers=[];scores=[];output_dict={'cond_frame_outputs':{},'non_cond_frame_outputs':{}}
            for index,conditioning in zip(ids,is_cond):
                storage='cpu' if index%2 else a.device
                feature=rand((batch,64,side,side),storage);position=rand((batch,64,side,side),storage);pointer=rand((batch,256))
                value=float('nan') if index==7 else .01 if index%3==0 else .7 if index%2 else -.1
                score=torch.empty(0,device=storage,dtype=dtype) if index%4==0 else torch.tensor(value,device=storage,dtype=dtype)
                features.append(feature);positions.append(position);pointers.append(pointer);scores.append(score)
                entry=dict(maskmem_features=feature,maskmem_pos_enc=[position],obj_ptr=pointer)
                if score.numel():entry['eff_iou_score']=score
                output_dict['cond_frame_outputs' if conditioning else 'non_cond_frame_outputs'][index]=entry
            host.num_maskmem=slots;host.max_cond_frames_in_attn=limit;host.max_obj_ptrs_in_encoder=pointer_limit
            host.memory_temporal_stride_for_eval=stride;host.keep_first_cond_frame=keep;host.use_memory_selection=filter_scores
            capture.clear()
            with source_transfer_device(a.device),torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                expected=host._prepare_memory_conditioned_features(current,initial,[source],[source_pos],[(side,side)],output_dict,24,reverse,use_previous)
            actual,memory,counts=torch.ops.sam3_native.sam3_temporal(str(a.store),source,source_pos,side,side,ids,is_cond,features,positions,pointers,scores,
                current,24,initial,reverse,use_previous,slots,limit,pointer_limit,stride,keep,filter_scores,.01,mode)
            errors={};torch.testing.assert_close(actual,expected,rtol=0,atol=0);errors['conditioned_features']=(actual.float()-expected.float()).abs().max().item()
            if capture:
                assert len(memory)==2 and counts==[capture['pointers']]
                for key,value in zip(['memory','position'],memory):
                    torch.testing.assert_close(value,capture[key],rtol=0,atol=0);errors[key]=(value.float()-capture[key].float()).abs().max().item()
            else:assert memory==[] and counts==[0]
            row=dict(mode=mode,case=name,grid=side,reverse=reverse,pointer_tokens=counts[0],max_errors=errors)
            results.append(row);print(json.dumps(row),flush=True)
        logits=torch.tensor([[-2.],[0.],[2.]],device=a.device,dtype=dtype)
        for shape in [(3,3),(3,)]:
            iou=torch.rand(shape,device=a.device,dtype=dtype)
            expected=host.cal_mem_score(logits,iou);actual=torch.ops.sam3_native.memory_confidence(logits,iou)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            results.append(dict(mode=mode,case='memory_confidence',iou_shape=list(shape),max_errors={'score':0.}))
    a.report.write_text(json.dumps(dict(torch=torch.__version__,device=a.device,selection_cases=selection_cases,cases=results,
        scope='Original SAM3 inference frame selection, memory/pointer assembly and attention. CPU reference redirects only hard-coded spatial .cuda() transfers and computes rotary caches on CPU. SAM3.1 temporal host not tested.'),indent=2)+'\n')
if __name__=='__main__':main()
