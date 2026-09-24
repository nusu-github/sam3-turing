"""Source NMS policies and video-specific logit/keep/query ordering."""
import argparse,json
from pathlib import Path
import torch
from sam3 import perflib
from sam3.perflib.nms import nms_masks as sam3_nms
import sam3.model.sam3_multiplex_detector_utils as mux
from sam3.model.sam3_video_base import Sam3VideoBase
DTYPES={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}

def source(scores,masks,nms,iom):
 if nms==0:
  if not iom:return sam3_nms(scores,masks,.4,.3)
  before=perflib.is_enabled;perflib.is_enabled=False
  try:return mux.nms_masks(scores,masks,.4,.3,True)
  finally:perflib.is_enabled=before
 if nms==1:
  before=perflib.is_enabled;perflib.is_enabled=True
  try:return mux.nms_masks(scores,masks,.4,.3,iom)
  finally:perflib.is_enabled=before
 return mux.nms_masks(scores[None],masks[None],.4,.3,iom)[0]

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--devices',nargs='+',default=['cpu','cuda']);p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True);a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(680);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;rows=[]
 for device in a.devices:
  for mode in a.modes:
   dtype=DTYPES[mode];checks=0
   for n,h,w in [(0,7,9),(1,7,9),(7,7,9),(257,7,9),(3,288,288)]:
    masks=torch.randn(n,w,h,device=device,dtype=dtype).transpose(1,2);scores=torch.linspace(.001,.999,n,device=device)
    if n==3:masks.fill_(1)
    if n>3:masks[1]=masks[0];masks[2]=-1
    for nms in range(3):
     for iom in [False,True]:
      if not n and nms in [1,2]:continue # source reshape(N,-1) cannot infer an empty N
      with torch.autocast(device,enabled=mode!='fp32',dtype=dtype):expected=source(scores,masks,nms,iom)
      actual=torch.ops.sam3_native.video_mask_nms(scores,masks,nms,.4,.3,iom,mode)
      try:torch.testing.assert_close(actual,expected,rtol=0,atol=0)
      except AssertionError as e:raise AssertionError(f'{device} {mode} n={n} nms={nms} iom={iom}: {e}') from e
      checks+=1
   # Source single-frame perflib behavior: a suppressed/invalid earlier row
   # may suppress later rows. Include ties and a chain, not just identical masks.
   masks=torch.tensor([[[1,1,0,0]],[[0,1,1,0]],[[0,0,1,1]],[[0,0,0,0]]],device=device,dtype=dtype)*2-1
   for scores in [torch.tensor([.95,.8,.6,.5],device=device),torch.full((4,),.7,device=device),torch.tensor([.2,.8,.6,.1],device=device)]:
    for nms in [1,2]:
     for iom in [False,True]:
      with torch.autocast(device,enabled=mode!='fp32',dtype=dtype):expected=source(scores,masks,nms,iom)
      actual=torch.ops.sam3_native.video_mask_nms(scores,masks,nms,.4,.3,iom,mode);torch.testing.assert_close(actual,expected,rtol=0,atol=0);checks+=1
   # Both prompts keep all 257 model-output slots in multiplex mode. SAM3
   # compacts selected detections without a cap; input logits are immutable.
   logits=torch.linspace(-3,3,257,device=device,dtype=dtype).view(1,257,1).repeat(2,1,1);masks=torch.randn(2,257,5,7,device=device,dtype=dtype);boxes=torch.rand(2,257,4,device=device,dtype=dtype);before=logits.clone()
   for multiplex,nms in [(False,0),(True,1),(True,2)]:
    for threshold in [0.,.3]:
     actual=torch.ops.sam3_native.video_detections(logits,masks,boxes,multiplex,nms,.4,threshold,False,True,True,mode)
     for b,values in enumerate(actual):
      reference=logits[b,:,0].clone()
      if threshold>0:
       with torch.autocast(device,enabled=mode!='fp32',dtype=dtype):kept=source(reference.sigmoid(),masks[b],nms,False)
       reference-=1e4*(~kept).float()
      scores=reference.sigmoid();keep=scores>.4
      if multiplex:keep&=Sam3VideoBase._suppress_detections_close_to_boundary(None,boxes[b])
      indices=keep.argsort(descending=True) if multiplex else torch.where(keep)[0]
      for value,expected in zip(values,[masks[b,indices],scores[indices],boxes[b,indices],keep[indices]]):torch.testing.assert_close(value,expected,rtol=0,atol=0);checks+=1
      if multiplex:assert values[0].shape[0]==257
   torch.testing.assert_close(logits,before,rtol=0,atol=0);checks+=1
   rows.append(dict(device=device,mode=mode,comparisons=checks));print(rows[-1],flush=True)
 a.report.write_text(json.dumps(dict(cases=rows,scope='Actual source SAM3 stable GPU greedy / unique-score CPU fallback, SAM3.1 single perflib and batched policies including ties, suppression chains and IoM row-area behavior. Explicit same-mode arithmetic including overflowing FP16 full-grid counts; production defaults FP32 policy math. Video penalty/filter/boundary/query ordering mirrors source host operations. No query cap (257-slot fixtures). Original perflib empty NMS cannot reshape N=0; native returns empty keep flags.'),indent=2)+'\n')
if __name__=='__main__':main()
