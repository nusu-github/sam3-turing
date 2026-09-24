"""Temporal attention parity, with source backend adaptations explicitly recorded."""
import argparse,json
from contextlib import contextmanager,nullcontext
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
import sam3.model.decoder as decoder_module
from sam3.model_builder import _create_tracker_transformer,_create_multiplex_transformer

@contextmanager
def backend_context(adapt_source,math_only):
    original=decoder_module.sdpa_kernel
    if adapt_source:decoder_module.sdpa_kernel=lambda *args,**kwargs:nullcontext()
    try:
        with sdpa_kernel(SDPBackend.MATH) if math_only else nullcontext():
            flash=torch.backends.cuda.enable_flash_sdp;efficient=torch.backends.cuda.enable_mem_efficient_sdp
            if math_only:
                # SAM3 RoPEAttention explicitly enables these in forward.
                # Keep the requested fallback active in the reference too.
                torch.backends.cuda.enable_flash_sdp=lambda enabled:flash(False)
                torch.backends.cuda.enable_mem_efficient_sdp=lambda enabled:efficient(False)
            try:yield
            finally:
                torch.backends.cuda.enable_flash_sdp=flash
                torch.backends.cuda.enable_mem_efficient_sdp=efficient
    finally:decoder_module.sdpa_kernel=original

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(547)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    results=[]
    for checkpoint in a.checkpoint:
        model,path=checkpoint.split('=',1);state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);state=state.get('model',state)
        multiplex=model=='sam3.1';module=(_create_multiplex_transformer() if multiplex else _create_tracker_transformer()).encoder.eval()
        prefix='tracker.model.transformer.encoder.' if multiplex else 'tracker.transformer.encoder.'
        module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);module.to(a.device)
        if a.device=='cpu':
            # RoPE caches are not registered buffers; reproduce construction on
            # a CPU-only host, rather than leaving CUDA caches on this GPU host.
            for child in module.modules():
                if hasattr(child,'compute_cis'):child.freqs_cis=child.compute_cis(end_x=72,end_y=72,device='cpu')
        reference_trace=[]
        handles=[layer.register_forward_hook(lambda mod,inp,out:reference_trace.append((out[1] if multiplex else out).transpose(0,1).clone())) for layer in module.layers]
        for mode in a.modes:
            dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
            cases=[('one_memory',8,2,1,0,False,False),('two_memories_pointers',8,2,2,5,False,False),
                   ('only_pointers',8,2,0,3,False,False),('full_grid',72,1,2,4,False,False),
                   ('shared_positions',8,2,1,3,False,False)]
            if multiplex:cases += [('batched_image_stream',8,2,1,0,True,False),('pre_padded_image_pointers',8,2,1,3,True,True),
                                  ('batched_image_shared_positions',8,2,1,3,True,False)]
            if a.device=='cuda' and mode=='fp16':cases += [('math_fallback_full_grid',72,1,1,4,False,False)]
            if mode!='fp32':cases += [('mixed_precision_streams',8,2,2,5,False,False)]
            for name,side,batch,frames,pointers,batched_image,prepad in cases:
                count=side*side;length=count*frames+pointers;channels=256 if multiplex else 64
                def random(shape):return torch.randn(*shape,device=a.device,dtype=dtype)
                source=random((batch,count,256)).transpose(0,1)
                source_pos=random((batch,count,256)).transpose(0,1)
                memory=random((batch,length,channels)).transpose(0,1);memory_pos=random((batch,length,channels)).transpose(0,1)
                image=memory_image=memory_image_pos=None
                if multiplex:
                    image_batch=batch if batched_image else 1
                    image=random((image_batch,count,256)).transpose(0,1)
                    image_length=length if prepad else length-pointers
                    memory_image=random((image_batch,image_length,256)).transpose(0,1);memory_image_pos=random((image_batch,image_length,256)).transpose(0,1)
                if name in ['shared_positions','batched_image_shared_positions']:
                    source_pos=source_pos[:,:1];memory_pos=memory_pos[:,:1]
                    if multiplex:memory_image_pos=memory_image_pos[:,:1]
                if name=='mixed_precision_streams':
                    source=source.float();memory_pos=memory_pos.float()
                    if multiplex:image=image.float();memory_image_pos=memory_image_pos.float()
                math_only=name.startswith('math_fallback')
                adapted=multiplex and ((a.device=='cuda' and mode=='fp32') or math_only)
                reference_trace.clear()
                with backend_context(adapted,math_only):
                    with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                        if multiplex:expected=module(image,source,memory_image,memory,src_pos=source_pos,memory_image_pos=memory_image_pos,memory_pos=memory_pos,num_obj_ptr_tokens=pointers)['memory']
                        else:expected=module(source,memory,src_pos=source_pos,prompt_pos=memory_pos,num_obj_ptr_tokens=pointers)['memory']
                    state_before=[torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.cudnn_sdp_enabled(),torch.backends.cuda.math_sdp_enabled()]
                    actual=torch.ops.sam3_native.memory_attention(str(a.store),model,source,source_pos,memory,memory_pos,pointers,mode,image,memory_image,memory_image_pos)
                    assert state_before==[torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.cudnn_sdp_enabled(),torch.backends.cuda.math_sdp_enabled()]
                    if math_only:
                        assert torch.backends.cuda.math_sdp_enabled()
                        assert not any([torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.cudnn_sdp_enabled()])
                errors={}
                assert len(actual)==5 and len(reference_trace)==4
                for key,value,target in zip(['layer_0','layer_1','layer_2','layer_3','output'],actual,reference_trace+[expected]):
                    torch.testing.assert_close(value,target,rtol=0,atol=0)
                    errors[key]=(value.float()-target.float()).abs().max().item()
                row=dict(model=model,mode=mode,case=name,grid=side,batch=batch,memory_frames=frames,pointer_tokens=pointers,
                    source_flash_only_context_removed=adapted,math_only_verified=math_only,max_errors=errors)
                results.append(row);print(json.dumps(row),flush=True)
        for handle in handles:handle.remove()
    a.report.write_text(json.dumps(dict(torch=torch.__version__,device=a.device,cases=results,
        scope='Four original temporal attention layers and final norm. SAM3.1 CUDA FP32/explicit math reference removes only its Flash-only context; equations and weights unchanged. CPU caches computed on CPU as on a CPU-only host. No frame selection/scheduler tested.'),indent=2)+'\n')
if __name__=='__main__':main()
