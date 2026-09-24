"""Actual source global shrinkage policy and explicit-ID local assignment."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--devices',nargs='+',default=['cpu','cuda']);p.add_argument('--large',action='store_true');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4)
    host=Sam3TrackerPredictor.__new__(Sam3TrackerPredictor);torch.nn.Module.__init__(host)
    g=torch.Generator().manual_seed(1233)
    cases=[torch.full((1,3,5),-.2),torch.ones(1,3,5),torch.zeros(3,3,5),torch.ones(3,3,5),torch.randn(7,9,5,generator=g).transpose(1,2)]
    tie=torch.ones(2,1,10);tie[1,:,0:3]=2;cases += [tie,tie.flip(0)]
    nonfinite=torch.tensor([[[float('nan'),1.]],[[1.,float('inf')]],[[float('-inf'),-1.]]]);cases.append(nonfinite)
    rows=[]
    for device in a.devices:
      for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        count=0
        for case in cases:
          low=case.to(device,dtype=dtype)
          for mux,warmup in [(False,False),(False,True),(True,False),(True,True)]:
            expected=F.interpolate(low[:,None],(1152,1152),mode='bilinear',align_corners=False)
            if mux:expected=Sam3VideoTrackingMultiplexDemo._suppress_object_pw_area_shrinkage(expected)
            elif warmup:expected=host._suppress_object_pw_area_shrinkage(expected)
            scores=torch.where((expected>0).any(dim=(-1,-2)),10.,-10.)
            actual=torch.ops.sam3_native.prepare_video_memory(low,mux,warmup)
            for x,y in zip(actual,[expected,scores]):torch.testing.assert_close(x,y,rtol=0,atol=0,equal_nan=True);count+=1
        rows.append(dict(device=device,dtype=str(dtype),exact_tensor_comparisons=count));print(rows[-1],flush=True)
    large=[]
    if a.large:
      for device in a.devices:
        low=torch.ones(201,2,3,device=device)
        expected=Sam3VideoTrackingMultiplexDemo._suppress_object_pw_area_shrinkage(F.interpolate(low[:,None],(1152,1152),mode='bilinear',align_corners=False))
        scores=torch.where((expected>0).any(dim=(-1,-2)),10.,-10.)
        actual=torch.ops.sam3_native.prepare_video_memory(low,True,True)
        for x,y in zip(actual,[expected,scores]):torch.testing.assert_close(x,y,rtol=0,atol=0)
        large.append(dict(device=device,objects=201,exact_tensor_comparisons=2));print('large',large[-1],flush=True)
        del expected,scores,actual,low,x,y
    # An explicit global ID list avoids assumptions about rank offsets or
    # sorted state IDs after backfilling. Missing/duplicate IDs are errors.
    plans=[([30,10,50,20],[[20,30],[50],[],[10]]),([],[]),([99],[[],[99]]),([1,2,3],[[3,1],[2,3]])]
    for global_ids,states in plans:
        assert torch.ops.sam3_native.video_memory_rows(global_ids,states)==[[global_ids.index(x) for x in s] for s in states]
    for global_ids,states in [([1,1],[[1]]),([1],[[2]]),([1],[[1,1]])]:
        try:torch.ops.sam3_native.video_memory_rows(global_ids,states)
        except RuntimeError:pass
        else:raise AssertionError('invalid assignment accepted')
    report=dict(cases=rows,large=large,assignment_cases=len(plans),assignment_rejections=3,source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['sam3/model/sam3_tracking_predictor.py','sam3/model/video_tracking_multiplex_demo.py']},scope='Actual global shrinkage policy at 1152 resolution, warmup/model-specific singleton behavior, ties/empty masks/nonfinite/strided logits. Explicit global-ID mapping tested independently of source rank-order assumptions. Neural memory/session updates are separate tests.')
    a.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
