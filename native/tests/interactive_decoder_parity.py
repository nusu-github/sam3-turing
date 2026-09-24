"""Interactive two-way decoder, all candidates and stability selection parity."""
import argparse,json
from pathlib import Path
import torch
from sam3.sam.mask_decoder import MaskDecoder
from sam3.sam.transformer import TwoWayTransformer
from sam3.sam.prompt_encoder import PromptEncoder

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(231)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    results=[]
    for spec in a.checkpoint:
        model,path=spec.split('=',1);state=torch.load(path,map_location='cpu',mmap=True,weights_only=True);state=state.get('model',state)
        root='tracker.' if model=='sam3' else 'tracker.model.interactive_'
        ref=MaskDecoder(transformer_dim=256,transformer=TwoWayTransformer(2,256,8,2048),num_multimask_outputs=3,
            use_high_res_features=True,iou_prediction_use_sigmoid=model=='sam3',pred_obj_scores=True,pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,dynamic_multimask_via_stability=True).eval()
        prefix=root+'sam_mask_decoder.'
        ref.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);ref.to(a.device)
        for mode in a.modes:
            dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
            for case,batch,h,w,repeat,n in [('full',1,72,72,False,3),('repeat',3,11,17,True,7),('batched',2,13,9,False,0)]:
                pe=PromptEncoder(256,(h,w),(h*14,w*14),16).eval()
                prefix=root+'sam_prompt_encoder.'
                pe.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);pe.to(a.device)
                image=torch.randn(1 if repeat else batch,256,h,w,device=a.device).to(dtype)
                high=[torch.randn(image.shape[0],256,h*s,w*s,device=a.device).to(dtype) for s in [4,2]]
                if case=='batched':
                    image=image.contiguous(memory_format=torch.channels_last)
                    high=[x.contiguous(memory_format=torch.channels_last) for x in high]
                points=torch.rand(batch,n,2,device=a.device)*min(h,w)*14;labels=torch.arange(batch*n,device=a.device).view(batch,n)%5-1
                masks=torch.randn(batch,1,h*4,w*4,device=a.device) if case!='full' else None
                with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                    sparse,dense=pe((points,labels),None,masks);position=pe.get_dense_pe()
                    projected=[ref.conv_s0(high[0]),ref.conv_s1(high[1])]
                    raw=ref.predict_masks(image,position,sparse,dense,repeat,projected)
                for selection,multi,dynamic,threshold in [('multi',True,True,.98),('stability',False,True,.98),
                                                         ('force_fallback',False,True,1.1),('stable',False,True,-.1),('single',False,False,.98)]:
                    ref.dynamic_multimask_via_stability=dynamic;ref.dynamic_multimask_stability_thresh=threshold
                    # Select from original outputs without repeating the expensive
                    # transformer. Exercise its exact source selection implementation.
                    rm,ri,rt,ro=raw
                    if multi: expected=(rm[:,1:],ri[:,1:],rt[:,1:],ro)
                    elif dynamic:
                        em,ei=ref._dynamic_multimask_via_stability(rm,ri);expected=(em,ei,rt[:,:1],ro)
                    else: expected=(rm[:,:1],ri[:,:1],rt[:,:1],ro)
                    expected=(*expected,rm,ri,rt)
                    with torch.autocast(a.device,dtype=torch.bfloat16):
                        actual=torch.ops.sam3_native.interactive_decode(str(a.store),model,image,sparse,dense,position,
                            high,True,multi,repeat,mode,dynamic,.05,threshold)
                        assert torch.is_autocast_enabled(a.device) and torch.get_autocast_dtype(a.device)==torch.bfloat16
                    errors={}
                    for key,value,target in zip(['masks','iou','tokens','object_logits','all_masks','all_iou','all_tokens'],actual,expected):
                        torch.testing.assert_close(value,target,rtol=0,atol=0)
                        errors[key]=(value-target).abs().max().item()
                    item=dict(model=model,mode=mode,case=case,selection=selection,batch=batch,grid=[h,w],max_errors=errors)
                    results.append(item);print(json.dumps(item),flush=True)
    a.report.write_text(json.dumps(dict(torch=torch.__version__,device=a.device,cases=results),indent=2)+'\n')
if __name__=='__main__':main()
