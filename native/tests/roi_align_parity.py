"""Sampling, dtype, autocast, layout and boundary parity with torchvision."""
import argparse
import json
from pathlib import Path
import torch
from torchvision.ops import roi_align

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('library',type=Path)
    p.add_argument('--report',type=Path,required=True)
    args = p.parse_args()
    torch.ops.load_library(str(args.library.resolve()))
    torch.set_num_threads(4)
    torch.manual_seed(113)
    results = []
    for device in ['cpu','cuda']:
        for dtype in [torch.float32,torch.float64,torch.float16]:
            x = torch.randn(3,5,13,19,device=device,dtype=dtype).transpose(2,3)
            boxes = torch.tensor([[0,0,0,12,18],[2,-2,-1,14,20],[1,2,3,2,3],
                                  [0,7.1,2.5,11.7,14.3],[1,4,8,3,5],[2,-20,-20,-8,-8]],device=device,dtype=dtype)
            for aligned in [False,True]:
                for ratio in [-1,1,3]:
                    for scale in [1.,.3]:
                        a = roi_align(x,boxes,(7,5),scale,ratio,aligned)
                        b = torch.ops.sam3_native.roi_align(x,boxes,scale,7,5,ratio,aligned)
                        error = (a-b).abs().max().item()
                        tolerance = 0.003 if dtype == torch.float16 else 2e-6 if dtype == torch.float32 else 1e-14
                        torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance)
                        results.append(dict(device=device,dtype=str(dtype),aligned=aligned,ratio=ratio,scale=scale,max_abs_error=error))
            for k in [0,6]:
                a = roi_align(x,boxes[:k],(1,1))
                b = torch.ops.sam3_native.roi_align(x,boxes[:k],1.,1,1,-1,False)
                torch.testing.assert_close(a,b,rtol=.003 if dtype==torch.float16 else 2e-6,atol=.003 if dtype==torch.float16 else 2e-6)
                results.append(dict(device=device,dtype=str(dtype),rois=k,pool=1,max_abs_error=(a-b).abs().max().item() if k else 0))
        for dtype in [torch.float16,torch.bfloat16]:
            x = torch.randn(2,4,17,21,device=device).to(dtype)
            boxes = torch.tensor([[0,-1,-1,20,16],[1,3.1,4.7,8.9,11.2]],device=device)
            with torch.autocast(device,dtype=dtype):
                a = roi_align(x,boxes,(7,7))
                b = torch.ops.sam3_native.roi_align(x,boxes,1.,7,7,-1,False)
                assert torch.is_autocast_enabled(device) and torch.get_autocast_dtype(device)==dtype
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            results.append(dict(device=device,autocast=str(dtype),max_abs_error=(a-b).abs().max().item()))
        for invalid in [[2,0,0,3,3],[-1,0,0,3,3],[.5,0,0,3,3],[0,float('nan'),0,3,3]]:
            try:
                torch.ops.sam3_native.roi_align(x.float(),torch.tensor([invalid],device=device),1.,7,7,-1,False)
            except RuntimeError:
                pass
            else:
                raise AssertionError('invalid ROI accepted')
    report = dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),cases=results)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(results),max_abs_error=max(x['max_abs_error'] for x in results))))

if __name__=='__main__': main()
