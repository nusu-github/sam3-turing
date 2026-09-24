"""Original memory neural module and tracker-host parity; no temporal attention."""
import argparse,json
from pathlib import Path
from types import MethodType,SimpleNamespace
import torch
from sam3.model.memory import SimpleMaskEncoder,SimpleMaskDownSampler,SimpleFuser,CXBlock
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_tracker_base import Sam3TrackerBase
from sam3.model.video_tracking_multiplex import VideoTrackingMultiplex
from sam3.model.multiplex_utils import MultiplexState

def reference(state,model,device):
    multiplex=model=='sam3.1';root='tracker.model.' if multiplex else 'tracker.'
    module=SimpleMaskEncoder(out_dim=256 if multiplex else 64,
        position_encoding=PositionEmbeddingSine(256 if multiplex else 64,precompute_resolution=1008 if device=='cuda' else None),
        mask_downsampler=SimpleMaskDownSampler(kernel_size=3,stride=2,padding=1,interpol_size=[1152,1152],
            multiplex_count=16 if multiplex else 1,starting_out_chan=4 if multiplex else 1,input_channel_multiplier=2 if multiplex else 1),
        fuser=SimpleFuser(CXBlock(256,kernel_size=7,padding=3,layer_scale_init_value=1e-6,use_dwconv=True),2))
    prefix=root+'maskmem_backbone.'
    module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);module.eval().to(device)
    cls=VideoTrackingMultiplex if multiplex else Sam3TrackerBase
    host=SimpleNamespace(hidden_dim=256,training=False,non_overlap_masks_for_mem_enc=False,
        sigmoid_scale_for_mem_enc=2. if multiplex else 20.,sigmoid_bias_for_mem_enc=-1. if multiplex else -10.,
        maskmem_backbone=module,_maybe_clone=lambda x:x,no_obj_embed_spatial=state[root+'no_obj_embed_spatial'].to(device),
        apply_sigmoid_to_mask_logits_for_mem_enc=True,binarize_mask_from_pts_for_mem_enc=False,
        add_object_conditional_embeddings=False,condition_as_mask_input=True,
        condition_as_mask_input_fg=1.,condition_as_mask_input_bg=0.,object_score_logit_threshold=0.)
    host._encode_new_memory=MethodType(cls._encode_new_memory,host)
    host._apply_non_overlapping_constraints=MethodType(cls._apply_non_overlapping_constraints,host)
    return module,host

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(482)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    results=[]
    def compare(actual,expected,**metadata):
        errors={}
        for key,value,target in zip(['features','position'],actual,expected):
            torch.testing.assert_close(value,target,atol=0,rtol=0)
            if key=='position' and a.device=='cuda':
                assert value.stride()==target.stride(),(value.stride(),target.stride())
            errors[key]=(value.float()-target.float()).abs().max().item()
        row=dict(**metadata,shape=list(actual[0].shape),max_errors=errors)
        results.append(row);print(json.dumps(row),flush=True)
    for checkpoint in a.checkpoint:
        model,path=checkpoint.split('=',1);state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);state=state.get('model',state)
        module,host=reference(state,model,a.device);multiplex=model=='sam3.1'
        for mode in a.modes:
            dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
            module_cases=[('resize_logits',(213,319),False,False),('full_probabilities',(1152,1152),True,True),
                          ('shared_image',(213,319),True,True)]
            if a.device=='cuda':module_cases.append(('cpu_staged_image',(213,319),True,True))
            for name,size,skip,channels_last in module_cases:
                shared=name in ['shared_image','cpu_staged_image']
                image=torch.randn(1 if shared else 2,256,72,72,device='cpu' if name=='cpu_staged_image' else a.device,dtype=dtype)
                masks=torch.randn(2,32 if multiplex else 1,*size,device=a.device,dtype=dtype)
                if channels_last:
                    image=image.contiguous(memory_format=torch.channels_last);masks=masks.contiguous(memory_format=torch.channels_last)
                with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                    out=module(image,masks,skip_mask_sigmoid=skip)
                actual=torch.ops.sam3_native.memory_encode(str(a.store),model,image,masks,skip,mode)
                compare(actual,(out['vision_features'],out['vision_pos_enc'][0]),model=model,mode=mode,case=name)
            cases=[('predictions',False,False,None,0,0.),('from_points',True,False,[],0,0.),
                   ('overlap',False,True,[0,2],0,0.)]
            if multiplex:cases += [('scores_short',False,False,[1],-1,0.),('scores_long',True,True,[0,1,2],2,.5)]
            for name,from_points,overlap,conditions,score_delta,threshold in cases:
                objects=3;buckets=2 if multiplex else objects
                assignments=[[2,-1,0]+[-1]*13,[1]+[-1]*15]
                mux=MultiplexState(assignments,torch.device(a.device),dtype,16) if multiplex else None
                raw=torch.randn(buckets,256,72,72,device=a.device,dtype=dtype).contiguous(memory_format=torch.channels_last)
                sequence=raw.flatten(2).permute(2,0,1);image=sequence.permute(1,2,0).view(buckets,256,72,72)
                masks=torch.randn(objects,1,1008,1008,device=a.device,dtype=dtype)
                scores=torch.tensor([-2.,0.,2.,3.,-1.][:objects+score_delta],device=a.device,dtype=dtype).unsqueeze(1)
                host.non_overlap_masks_for_mem_enc=overlap;host.object_score_logit_threshold=threshold
                with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                    kwargs=dict(conditioning_objects=conditions,multiplex_state=mux) if multiplex else {}
                    out=host._encode_new_memory(None,[sequence],[(72,72)],masks,scores,from_points,**kwargs)
                condition_tensor=torch.tensor(conditions,device=a.device,dtype=torch.int64) if conditions is not None else None
                actual=torch.ops.sam3_native.memory_frame(str(a.store),model,image,masks,scores,from_points,overlap,threshold,
                    mux.mux_matrix if multiplex else None,condition_tensor,mode)
                compare(actual,(out[0],out[1][0]),model=model,mode=mode,case=name)
            if multiplex and mode==('fp16' if a.device=='cuda' else 'fp32'):
                # More than one bucket's capacity; permute object assignments.
                objects=37;buckets=3;order=torch.randperm(objects).tolist()+[-1]*11
                mux=MultiplexState([order[i:i+16] for i in range(0,48,16)],torch.device(a.device),dtype,16)
                image=torch.randn(1,256,72,72,device=a.device,dtype=dtype)
                sequence=image.flatten(2).permute(2,0,1);masks=torch.randn(objects,1,173,211,device=a.device,dtype=dtype)
                scores=torch.randn(objects,1,device=a.device,dtype=dtype);conditions=[0,16,36]
                host.non_overlap_masks_for_mem_enc=False;host.object_score_logit_threshold=0.
                with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                    out=host._encode_new_memory(None,[sequence],[(72,72)],masks,scores,False,conditioning_objects=conditions,multiplex_state=mux)
                actual=torch.ops.sam3_native.memory_frame(str(a.store),model,image,masks,scores,False,False,0.,mux.mux_matrix,
                    torch.tensor(conditions,device=a.device),mode)
                compare(actual,(out[0],out[1][0]),model=model,mode=mode,case='37_objects_3_buckets')
    a.report.write_text(json.dumps(dict(torch=torch.__version__,device=a.device,cases=results,
        scope='Original memory encoder and frame-memory host behavior for shipped SAM3/SAM3.1 configurations. CPU reference computes positions on CPU; CUDA reference retains builder positional cache. Temporal attention/scheduling not tested.'),indent=2)+'\n')
if __name__=='__main__':main()
