"""Isolated GPU hotstart update: original outer-product reduction vs native."""
import argparse,gc,json,statistics,types
from pathlib import Path
import torch
import sam3.model.sam3_multiplex_base as mux
from hotstart_parity import CONFIGS,FIELDS,compare_state,options

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--objects',type=int,default=512);p.add_argument('--detections',type=int,default=200);p.add_argument('--iterations',type=int,default=30);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    assert a.objects>0 and a.detections>=0 and a.iterations>0
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    g=torch.Generator(device='cuda').manual_seed(634);match=torch.rand(a.detections,a.objects,device='cuda',generator=g)>.6;unmatched=torch.rand(a.objects,device='cuda',generator=g)>.6;nonempty=torch.rand(a.objects,device='cuda',generator=g)>.3
    config=[50,8,8,0,-4,8,0,1];native=torch.ops.sam3_native.hotstart_extend({},a.objects,0,0,'cuda');native['obj_first_frame']=torch.arange(a.objects,device='cuda')%32
    source={key:(value.item() if key=='N_obj' else value) for key,value in native.items()};adt=types.SimpleNamespace(trk_is_unmatched=unmatched,trk_is_nonempty=nonempty,im_mask=match);host=options(config)
    def original():return mux.Sam3MultiplexBase._process_hotstart_gpu(host,40,False,adt,{},source)
    def candidate():return torch.ops.sam3_native.hotstart_device(native,unmatched,nonempty,match,40,False,config)
    remove,suppress,expected=original();actual=candidate();compare_state(actual,expected);torch.testing.assert_close(actual['remove'],remove,rtol=0,atol=0);torch.testing.assert_close(actual['suppress'],suppress,rtol=0,atol=0)
    del remove,suppress,expected,actual;gc.collect();torch.cuda.synchronize()
    def measure(function):
        for _ in range(5):result=function();del result
        torch.cuda.synchronize();gc.collect();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(a.iterations):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();result=function();end.record();end.synchronize();times.append(start.elapsed_time(end));del result
        return dict(median_cuda_event_ms=statistics.median(times),min_cuda_event_ms=min(times),max_cuda_event_ms=max(times),peak_allocated_delta_bytes=torch.cuda.max_memory_allocated()-baseline,baseline_allocated_bytes=baseline,samples_ms=times)
    # Alternate ordering to expose warmup/order effects rather than one best run.
    runs=[]
    for order in [('original','native'),('native','original')]:
        runs.append({key:measure(original if key=='original' else candidate) for key in order})
    report=dict(objects=a.objects,detections=a.detections,iterations=a.iterations,warmup=5,gpu=torch.cuda.get_device_name(),torch=torch.__version__,source_outer_product_bytes=a.detections*a.objects*a.objects*4,native_pair_matrix_bytes=a.objects*a.objects*4,runs=runs,scope='Isolated hotstart update on synthetic association decisions, same previous state for each iteration. CUDA event time includes stream work and launch gaps; peak is PyTorch allocated bytes above baseline, not total device memory/host RSS. No inference/video speedup claim. Native and original full state/decision outputs compare exactly before timing.')
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:report[key] for key in ['objects','detections','source_outer_product_bytes','native_pair_matrix_bytes']}));print(json.dumps([{k:{f:v for f,v in data.items() if f!='samples_ms'} for k,data in run.items()} for run in runs],indent=2))
if __name__=='__main__':main()
