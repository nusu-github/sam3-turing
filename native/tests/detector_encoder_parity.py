"""Compare all six fusion layers and decoder metadata with original models."""
import argparse,json
from pathlib import Path
import torch
from sam3.model_builder import _create_transformer_encoder
from sam3.model.position_encoding import PositionEmbeddingSine

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--reference',type=Path);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(176)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results=[]
    for spec in a.checkpoint:
        model,path=spec.split('=',1);ref=_create_transformer_encoder().eval()
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);state=state.get('model',state)
        prefix='detector.transformer.encoder.'
        ref.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);ref.to(a.device)
        for mode in a.modes:
            dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
            for case,b,h,w,length in [('full',1,72,72,33),('batch_padding',3,15,21,17),('geometry_only',2,13,19,1),('actual_vision',1,72,72,33)]:
                if case=='actual_vision':
                    if a.reference is None or model!='sam3':continue
                    saved=torch.load(a.reference/'vision.pt',map_location=a.device,weights_only=True)
                    image=saved['backbone_fpn'][-1].to(dtype);pos=saved['vision_pos_enc'][-1]
                    saved_text=torch.load(a.reference/'text-0.pt',map_location=a.device,weights_only=True)
                    points=torch.empty(0,b,2,device=a.device);labels=torch.empty(0,b,dtype=torch.long,device=a.device);empty=torch.empty(b,0,dtype=torch.bool,device=a.device)
                    geo,gm=torch.ops.sam3_native.geometry_encode(str(a.store),model,image,pos,points,labels,empty,torch.empty(0,b,4,device=a.device),labels,empty,mode)
                    prompt=torch.cat([saved_text['language_features'],geo],0);mask=torch.cat([saved_text['language_mask'],gm],1)
                else:
                    image=torch.randn(b,256,h,w,device=a.device).to(dtype);pos=PositionEmbeddingSine(256,normalize=True)(image)
                    prompt=torch.randn(length,b,256,device=a.device);mask=torch.zeros(b,length,dtype=torch.bool,device=a.device)
                imask=None
                if case=='batch_padding':
                    imask=torch.zeros(b,h,w,dtype=torch.bool,device=a.device)
                    for i in range(b):
                        mask[i,length-i-2:]=True;imask[i,h-i-1:,:]=True;imask[i,:,w-i-2:]=True
                with torch.autocast(a.device,dtype=dtype if mode!='fp32' else torch.bfloat16,enabled=mode!='fp32'):
                    expected=ref(src=[image.flatten(2).permute(2,0,1)],src_pos=[pos.flatten(2).permute(2,0,1)],prompt=prompt,
                        src_key_padding_mask=[imask.flatten(1).t()] if imask is not None else None,prompt_key_padding_mask=mask,feat_sizes=[(h,w)])
                with torch.autocast(a.device,dtype=torch.bfloat16):
                    actual=torch.ops.sam3_native.detector_encode(str(a.store),model,image,pos,prompt,mask,imask,mode)
                    assert torch.is_autocast_enabled(a.device) and torch.get_autocast_dtype(a.device)==torch.bfloat16
                errors={}
                for k,v in expected.items():
                    if v is None:assert k not in actual;continue
                    tolerance={'fp32':5e-5,'fp16':.003,'bf16_reference':.02}[mode] if v.is_floating_point() else 0
                    torch.testing.assert_close(actual[k],v,rtol=tolerance,atol=tolerance)
                    errors[k]=(actual[k].float()-v.float()).abs().max().item()
                item=dict(model=model,mode=mode,case=case,memory_shape=list(actual['memory'].shape),dtype=str(actual['memory'].dtype),max_errors=errors)
                results.append(item);print(json.dumps(item),flush=True)
    a.report.write_text(json.dumps(dict(torch=torch.__version__,device=a.device,cases=results),indent=2)+'\n')
if __name__=='__main__':main()
