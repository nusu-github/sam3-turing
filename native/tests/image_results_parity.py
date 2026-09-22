"""Image thresholding, resizing and uncapped ragged batch result parity."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.data_misc import interpolate

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--reference',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(19);results=[]
    width,height=Image.open(a.reference/'input.png').size
    for file in sorted(a.reference.glob('*.pt')):
        saved=torch.load(file,map_location='cuda',weights_only=True)
        if not isinstance(saved,dict) or 'raw' not in saved:continue
        raw=saved['raw'];threshold=.25 if file.stem=='box-threshold' else .5
        actual=torch.ops.sam3_native.postprocess_image(raw['pred_boxes'],raw['pred_logits'],raw['pred_masks'],raw['presence_logit_dec'],[height],[width],threshold,True,3,"bf16_reference")
        for key,index in [('boxes',0),('scores',1),('masks_logits',2),('masks',3)]:
            torch.testing.assert_close(actual[index][0],saved['result'][key],rtol=0,atol=0)
        results.append(dict(case=file.stem,device='cuda',detections=len(actual[1][0]),max_abs_error=0))
    for device in ['cpu','cuda']:
        for dtype in [torch.float32,torch.float16,torch.bfloat16]:
            boxes=torch.rand(2,200,4,device=device);logits=torch.randn(2,200,1,device=device).to(dtype)
            masks=torch.randn(2,200,3,5,device=device).to(dtype);presence=torch.zeros(2,1,device=device).to(dtype)
            for threshold in [-1.,.25,2.]:
                expected=[]
                scores=(logits.sigmoid()*presence.sigmoid().unsqueeze(1)).squeeze(-1)
                for b,(h,w) in enumerate([(7,11),(13,9)]):
                    keep=scores[b]>threshold;prob=interpolate(masks[b][keep,None],(h,w),mode='bilinear',align_corners=False).sigmoid()
                    expected.append((box_cxcywh_to_xyxy(boxes[b][keep])*torch.tensor([w,h,w,h],device=device),scores[b][keep],prob,prob>.5,keep.nonzero().squeeze(1)))
                for chunk in [1,13]:
                    actual=torch.ops.sam3_native.postprocess_image(boxes,logits,masks,presence,[7,13],[11,9],threshold,True,chunk)
                    for b,item in enumerate(expected):
                        for i,value in enumerate(item):torch.testing.assert_close(actual[i][b],value,rtol=0,atol=0)
                    results.append(dict(case='ragged',device=device,dtype=str(dtype),threshold=threshold,chunk=chunk,counts=[len(x) for x in actual[1]],max_abs_error=0))
    a.report.write_text(json.dumps(dict(cases=results),indent=2)+'\n');print(json.dumps(dict(cases=len(results),max_abs_error=0)),flush=True)
if __name__=='__main__':main()
