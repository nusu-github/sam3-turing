"""Compare the connected native detector against previously captured image results."""
import argparse,json
from pathlib import Path
import torch

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path)
    p.add_argument('reference',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    saved=torch.load(a.reference/'vision.pt',map_location='cuda',weights_only=True)
    ids=torch.zeros(1,device='cuda',dtype=torch.long);labels=torch.empty(0,1,device='cuda',dtype=torch.long)
    padding=torch.empty(1,0,device='cuda',dtype=torch.bool);points=torch.empty(0,1,2,device='cuda');boxes=torch.empty(0,1,4,device='cuda')
    results=[]
    for index in range(3):
        text=torch.load(a.reference/f'text-{index}.pt',map_location='cuda',weights_only=True)
        expected=torch.load(a.reference/f'text-{index}-grounding.pt',map_location='cuda',weights_only=True)['raw']
        actual=torch.ops.sam3_native.grounding(str(a.store),'sam3',saved['backbone_fpn'],saved['vision_pos_enc'][-1],ids,ids,
            text['language_features'],text['language_mask'],points,labels,padding,boxes,labels,padding,None,None,None,True,False,'bf16_reference')
        errors={}
        for key,value in expected.items():
            torch.testing.assert_close(actual[key],value,rtol=0,atol=0)
            errors[key]=(actual[key]-value).abs().max().item()
        result=dict(case=f'text-{index}',queries=actual['pred_logits'].shape[1],mask_shape=list(actual['pred_masks'].shape),max_errors=errors)
        results.append(result);print(json.dumps(result),flush=True)
    a.report.write_text(json.dumps(dict(torch=torch.__version__,precision='bf16_reference',scope='Connected detector from saved visual/text features; excludes native vision/text execution.',cases=results),indent=2)+'\n')
if __name__=='__main__':main()
