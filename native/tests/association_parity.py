"""Native association versus the repository's actual SAM3/SAM3.1 methods."""
import argparse,json,types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sam3.model.sam3_video_base as base
import sam3.model.sam3_multiplex_base as mux

NAMES=['unmatched','nonempty','is_new','best_track','high_confidence','high_overlap','keep','matches']
def metadata(value):
    if isinstance(value,base.LazyAssociateDetTrkResult):raise TypeError('realize first')
    if isinstance(value,base.RealizedAssociateDetTrkresult):value=(value.new_det_fa_inds,value.unmatched_trk_obj_ids,value.det_to_matched_trk_obj_ids,value.trk_id_to_max_iou_high_conf_det,value.empty_trk_obj_ids)
    return dict(new_detections=np.asarray(value[0]).tolist(),unmatched_tracks=np.asarray(value[1]).tolist(),empty_tracks=np.asarray(value[4]).tolist(),matches={int(k):np.asarray(v).tolist() for k,v in value[2].items()},recondition={int(k):int(v) for k,v in value[3].items()})
def actual_metadata(out):
    offsets=out['match_offsets'].tolist();keys=out['detection_keys'].tolist();ids=out['matched_ids'].tolist()
    return dict(new_detections=out['new_detections'].tolist(),unmatched_tracks=out['unmatched_tracks'].tolist(),empty_tracks=out['empty_tracks'].tolist(),matches={k:ids[offsets[i]:offsets[i+1]] for i,k in enumerate(keys)},recondition=dict(zip(out['recondition_ids'].tolist(),out['recondition_detections'].tolist())))

def cases():
    g=torch.Generator().manual_seed(7443)
    for n,m,h,w,ht,wt in [(0,0,3,5,7,9),(5,0,3,5,7,9),(0,4,3,5,7,9),(1,1,3,5,3,5),(5,7,7,9,3,5),(5,7,3,5,7,9),(5,7,3,10,5,6),(17,33,11,13,11,13),(257,513,3,7,3,7)]:
        det=torch.randn(n,h,w,generator=g);trk=torch.randn(m,ht,wt,generator=g);scores=torch.rand(n,generator=g);keep=torch.rand(n,generator=g)>.25
        if m:trk[-1].fill_(-1)
        if n:det[-1].fill_(float('nan'));scores[0]=.8
        yield f'random-{n}-{m}-{h}-{w}-{ht}-{wt}',det,scores,trk,keep
    yield 'full-288-mask',torch.ones(1,288,288),torch.tensor([.9]),torch.ones(1,288,288),torch.ones(1,dtype=torch.bool)
    yield 'no-tracks-keep-false',torch.zeros(3,2,2),torch.tensor([.9,.6,.7]),torch.empty(0,1,3),torch.tensor([False,True,True])
    # Duplicates exercise ambiguity clearing, argmax ties and dictionary overwrite.
    det=torch.tensor([[[1.,-1.,-1.,-1.]],[[1.,-1.,-1.,-1.]],[[-1.,1.,-1.,-1.]],[[-1.,-1.,1.,1.]]])
    trk=torch.tensor([[[1.,-1.,-1.,-1.]],[[1.,-1.,-1.,-1.]],[[-1.,1.,1.,-1.]],[[-1.,-1.,-1.,-1.]]])
    yield 'ambiguous-boundaries',det,torch.tensor([.7,.8,.8,.65]),trk,torch.tensor([True,True,True,False])
    # Non-contiguous views, non-finite logits/scores and empty tracks.
    det=torch.randn(9,13,7,generator=g).transpose(1,2);trk=torch.randn(6,13,7,generator=g).transpose(1,2);trk[-1].zero_()
    scores=torch.tensor([.7,.8,float('nan'),float('inf'),float('-inf'),.5,.0,1.,.65]);keep=torch.ones(18,dtype=torch.bool)[::2]
    yield 'strides-and-nonfinite',det,scores,trk,keep

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--devices',nargs='+',default=['cpu','cuda']);p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True);p.add_argument('--real-fixtures',type=Path);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs=list(cases())
    if a.real_fixtures:
        from c_api_parity import read
        ground=read(a.real_fixtures/'sam3-fp16-ground','ground');video=read(a.real_fixtures/'sam3-fp16-video','3-1');det=ground['raw/masks'][0].float();trk=video['low_masks'].squeeze(1).float()
        inputs.append(('real-neural-masks',det,ground['0/scores'].float(),trk.repeat(9,1,1),torch.ones(det.size(0),dtype=torch.bool)))
    rows=[]
    for device in a.devices:
      for mode in a.modes:
        dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
        for multiplex in [False,True]:
          count=0;workflows=0
          for name,det,scores,trk,keep in inputs:
            for iom,threshold,padding in [(False,.5,0),(True,.8,0),(False,0.,19),(True,0.,19),(False,1.,19)]:
              if name=='real-neural-masks' and (padding or iom):continue
              if not multiplex:padding=0
              d,t=det.to(device,dtype=dtype),trk.to(device,dtype=dtype);s=scores.to(device);k=keep.to(device) if multiplex else torch.ones_like(keep,device=device)
              ids=np.arange(t.size(0),dtype=np.int64)*17-20000000000
              host=types.SimpleNamespace(assoc_iou_thresh=threshold,trk_assoc_iou_thresh=threshold,new_det_thresh=.7,use_iom_recondition=iom,o2o_matching_masklets_enable=False,iom_thresh_recondition=threshold,iou_thresh_recondition=threshold,max_num_objects=padding)
              captured=[]
              original=base._associate_det_trk_compilable
              def capture(*args,**kwargs):
                result=original(*args,**kwargs);captured.append(result);return result
              if not multiplex:base._associate_det_trk_compilable=capture
              try:
                with torch.autocast(device_type=device,dtype=dtype,enabled=mode!='fp32'):
                  if multiplex:
                    expected=mux.Sam3MultiplexBase._associate_det_trk(host,d,s,k,t,ids)
                    raw=[getattr(expected,key) for key in ['trk_is_unmatched','trk_is_nonempty','is_new_det','det_to_max_iou_trk_idx','det_is_high_conf','det_is_high_iou','det_keep','im_mask']]
                  else:
                    expected=base.Sam3VideoBase._associate_det_trk(host,d,s.cpu().numpy(),t,ids);raw=list(captured[-1]) if captured else None
                actual=torch.ops.sam3_native.associate_tracking(d,s,t,k,ids.tolist(),multiplex,.7,threshold,threshold,.8,iom,threshold,threshold,padding,mode)
                if raw:
                  for key,value in zip(NAMES,raw):torch.testing.assert_close(actual[key],value,rtol=0,atol=0);count+=1
                if multiplex:expected=base.realize_adt_result(expected,{'obj_ids_all_gpu':ids},d)
                assert actual_metadata(actual)==metadata(expected),(name,device,mode,multiplex,iom,threshold,padding,actual_metadata(actual),metadata(expected));count+=5;workflows+=1
              finally:base._associate_det_trk_compilable=original
          rows.append(dict(device=device,mode=mode,model='sam3.1' if multiplex else 'sam3',workflows=workflows,exact_tensor_and_metadata_comparisons=count));print(json.dumps(rows[-1]),flush=True)
    placements=0
    for workloads in [[0],[1,0],[2,2,2],[10,0,2,0]]:
      for n in [0,1,3,17,200,1025]:
        for capacity in [1,3,16]:
          host=types.SimpleNamespace(is_multiplex=capacity!=1,bucket_capacity=capacity)
          expected=mux.Sam3MultiplexBase._assign_new_det_to_gpus(host,n,np.array(workloads,np.int64)).tolist();actual=torch.ops.sam3_native.assign_detection_devices(n,workloads,capacity);assert actual==expected;placements+=1
    boundaries=0
    for device in a.devices:
      for dtype in [torch.float16,torch.bfloat16,torch.float32,torch.float64]:
        boxes=torch.tensor([[0,.4,.05,.6],[.95,.4,1,.6],[.4,0,.6,.05],[.4,.95,.6,1],[.4,.4,.6,.6],[0,0,1,1],[float('nan'),0,1,1]],device=device,dtype=dtype)
        for margin in [0.,.025,.5,-.1]:
          expected=base.Sam3VideoBase._suppress_detections_close_to_boundary(None,boxes,margin);actual=torch.ops.sam3_native.detection_boundary_keep(boxes,margin);torch.testing.assert_close(actual,expected,rtol=0,atol=0);boundaries+=1
    report=dict(real_neural_mask_fixtures=bool(a.real_fixtures),fixture_names=[value[0] for value in inputs],cases=rows,placement_comparisons=placements,boundary_comparisons=boundaries,torch=torch.__version__,gpu=torch.cuda.get_device_name() if 'cuda' in a.devices else None,scope='Actual repository association wrappers/compilable function and metadata realization, source-specific empty branches and optional zero padding, no detection/track cap. Includes 257 detections/513 tracks and optional stored real neural masks. Placement is a plan comparison, not multi-GPU execution. No Hungarian path: upstream explicitly asserts it disabled.')
    a.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
