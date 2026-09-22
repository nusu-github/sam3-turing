"""Isolated source/native tracking box extraction; no model throughput claim."""
import argparse
import gc
import json
import statistics
from pathlib import Path
import torch
from sam3.model.sam3_tracker_utils import mask_to_box

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--objects',type=int,default=200);p.add_argument('--size',type=int,default=288);p.add_argument('--iterations',type=int,default=30);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    assert a.objects>0 and a.size>0 and a.iterations>0
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4)
    g=torch.Generator(device='cuda').manual_seed(807)
    masks=torch.rand(a.objects,1,a.size,a.size,device='cuda',generator=g)>.7
    masks[0]=False;masks[1:2]=True
    methods={'source':lambda:mask_to_box(masks),'native':lambda:torch.ops.sam3_native.tracking_mask_boxes(masks)}
    torch.testing.assert_close(methods['source'](),methods['native'](),rtol=0,atol=0)
    def measure(fn):
        for _ in range(5): result=fn();del result
        torch.cuda.synchronize();gc.collect();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(a.iterations):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();result=fn();end.record();end.synchronize();times.append(start.elapsed_time(end));del result
        return dict(median_cuda_event_ms=statistics.median(times),min_cuda_event_ms=min(times),max_cuda_event_ms=max(times),peak_allocated_delta_bytes=torch.cuda.max_memory_allocated()-baseline,baseline_allocated_bytes=baseline,samples_ms=times)
    runs=[{name:measure(methods[name]) for name in order} for order in [('source','native'),('native','source')]]
    report=dict(objects=a.objects,height=a.size,width=a.size,iterations=a.iterations,warmup=5,gpu=torch.cuda.get_device_name(),torch=torch.__version__,runs=runs,scope='Synthetic mask-to-box only: same inclusive int32 boxes including empty masks, GPU-event time includes stream work and launch gaps. Peak PyTorch allocated bytes above baseline excludes host RSS and reserved memory; not end-to-end video or Turing performance.')
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:{f:v for f,v in row.items() if f!='samples_ms'} for k,row in run.items()} for run in runs],indent=2))
if __name__=='__main__':main()
