"""Interactive point/box/mask embeddings and dense Fourier position parity."""
import argparse,json
from pathlib import Path
import torch
from sam3.sam.prompt_encoder import PromptEncoder

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(218)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    results=[]
    for spec in a.checkpoint:
        model,path=spec.split('=',1);state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);state=state.get('model',state)
        prefix='tracker.sam_prompt_encoder.' if model=='sam3' else 'tracker.model.interactive_sam_prompt_encoder.'
        weights={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        for grid,size in [((72,72),(1008,1008)),((11,17),(154,238))]:
            ref=PromptEncoder(256,grid,size,16).eval();ref.load_state_dict(weights,strict=True);ref.to(a.device)
            for mode in a.modes:
                dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
                for case,batch,n,use_points,use_boxes,use_masks in [
                    ('none',1,0,False,False,False),('empty_points',2,0,True,False,False),
                    ('labels',3,9,True,False,False),('box',2,0,False,True,False),
                    ('mask',3,0,False,False,True),('all',2,6,True,True,True),
                    ('box_mask',2,0,False,True,True),('many_points',1,317,True,False,True)]:
                    points=labels=boxes=masks=None
                    if use_points:
                        points=torch.rand(batch,n,2,device=a.device)*max(size)*1.2-10
                        labels=torch.arange(batch*n,device=a.device).view(batch,n)%6-1
                        if case=='all':points=points.to(dtype);labels=labels.int()
                        else:points=points.transpose(1,2).contiguous().transpose(1,2)
                    if use_boxes:boxes=torch.rand(batch,4,device=a.device)*max(size)
                    if use_masks:
                        masks=torch.randn(batch,1,4*grid[0],4*grid[1],device=a.device).to(dtype)
                        if case=='all':masks=masks.transpose(2,3).contiguous().transpose(2,3)
                    before=[x.clone() if x is not None else None for x in [points,labels,boxes,masks]]
                    with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                        sparse,dense=ref((points,labels) if use_points else None,boxes,masks);pos=ref.get_dense_pe()
                    with torch.autocast(a.device,dtype=torch.bfloat16):
                        actual=torch.ops.sam3_native.interactive_prompt(str(a.store),model,a.device,points,labels,boxes,masks,list(grid),list(size),mode)
                        assert torch.is_autocast_enabled(a.device) and torch.get_autocast_dtype(a.device)==torch.bfloat16
                    errors={}
                    for key,value,expected in zip(['sparse','dense','position'],actual,[sparse,dense,pos]):
                        torch.testing.assert_close(value,expected,rtol=0,atol=0)
                        errors[key]=(value-expected).abs().max().item() if value.numel() else 0
                    for x,y in zip([points,labels,boxes,masks],before):
                        if x is not None:torch.testing.assert_close(x,y,rtol=0,atol=0)
                    item=dict(model=model,mode=mode,case=case,grid=list(grid),batch=batch,points=n,max_errors=errors)
                    results.append(item);print(json.dumps(item),flush=True)
    a.report.write_text(json.dumps(dict(torch=torch.__version__,device=a.device,cases=results),indent=2)+'\n')
if __name__=='__main__':main()
