"""Full 3-layer geometry encoder parity with real SAM3/SAM3.1 weights."""
import argparse
import json
from pathlib import Path
import torch
from sam3.model_builder import _create_geometry_encoder
from sam3.model.geometry_encoders import Prompt
from sam3.model.position_encoding import PositionEmbeddingSine

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('library',type=Path)
    p.add_argument('store',type=Path)
    p.add_argument('--checkpoint',action='append',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference'])
    p.add_argument('--vision-reference',type=Path)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    torch.ops.load_library(str(args.library.resolve()))
    torch.set_num_threads(4)
    torch.manual_seed(791)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    results=[]
    for spec in args.checkpoint:
        model,filename=spec.split('=',1)
        reference=_create_geometry_encoder().eval()
        state=torch.load(filename,map_location='cpu',weights_only=True,mmap=True)
        state=state.get('model',state)
        prefix='detector.geometry_encoder.'
        reference.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
        reference.to(args.device)
        for mode in args.modes:
            amp_dtype=torch.float16 if mode=='fp16' else torch.bfloat16
            for case,batch,np,nb,h,w in [('empty',1,0,0,72,72),('points',1,4,0,72,72),('boxes',1,0,5,72,72),
                ('mixed_padding',3,5,4,13,19),('many_prompts',2,37,29,17,11),('all_padded',2,3,2,9,15),('actual_vision',1,2,3,72,72)]:
                if case=='actual_vision':
                    if not args.vision_reference or model!='sam3': continue
                    saved=torch.load(args.vision_reference,map_location=args.device,weights_only=True)
                    image=saved['backbone_fpn'][-1].float()
                    position=saved['vision_pos_enc'][-1].float()
                else:
                    image=torch.randn(batch,256,h,w,device=args.device)
                    position=PositionEmbeddingSine(256,normalize=True)(image)
                # Real AMP necks emit low-precision features; norm/grid sampling
                # and ROIAlign must reproduce their mixed-dtype behavior too.
                if mode!='fp32': image=image.to(amp_dtype)
                points=torch.rand(np,batch,2,device=args.device)*1.2-.1
                boxes=torch.rand(nb,batch,4,device=args.device)
                if nb:
                    boxes[0,:,:]=torch.tensor([.5,.5,.8,.8],device=args.device)
                    if nb>1: boxes[1,:,2:]=0
                pl=torch.arange(np*batch,device=args.device).reshape(np,batch)%2
                bl=(torch.arange(nb*batch,device=args.device).reshape(nb,batch)%2).bool()
                pm=torch.zeros(batch,np,dtype=torch.bool,device=args.device)
                bm=torch.zeros(batch,nb,dtype=torch.bool,device=args.device)
                if 'padding' in case:
                    for i in range(batch):
                        pm[i,max(np-i-1,0):]=True
                        bm[i,max(nb-2*i,0):]=True
                if case=='all_padded': pm[:]=True;bm[:]=True
                prompt=Prompt(point_embeddings=points,point_labels=pl,point_mask=pm,box_embeddings=boxes,box_labels=bl,box_mask=bm)
                with torch.autocast(args.device,enabled=mode!='fp32',dtype=amp_dtype):
                    expected,mask=reference(prompt,[image.flatten(2).permute(2,0,1)],[(h,w)],[position.flatten(2).permute(2,0,1)])
                # An outer, different context must be restored even for fp32.
                with torch.autocast(args.device,dtype=torch.bfloat16):
                    actual,amask=torch.ops.sam3_native.geometry_encode(str(args.store),model,image,position,points,pl,pm,boxes,bl,bm,mode)
                    assert torch.is_autocast_enabled(args.device) and torch.get_autocast_dtype(args.device)==torch.bfloat16
                error=(actual-expected).abs().max().item()
                tolerance={'fp32':3e-5,'fp16':.003,'bf16_reference':.02}[mode]
                torch.testing.assert_close(actual,expected,rtol=tolerance,atol=tolerance)
                torch.testing.assert_close(amask,mask,rtol=0,atol=0)
                result=dict(model=model,mode=mode,case=case,shape=list(actual.shape),dtype=str(actual.dtype),max_abs_error=error)
                results.append(result);print(json.dumps(result),flush=True)
    args.report.write_text(json.dumps(dict(torch=torch.__version__,device=args.device,cases=results),indent=2)+'\n')
if __name__=='__main__': main()
