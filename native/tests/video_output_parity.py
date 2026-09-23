"""Actual source output generators/postprocessors with synthetic raw neural outputs.
Neural execution is replaced only at the raw-frame boundary; temporal buffering,
filtering, cache exclusions, boxes, overlap resolution and centers run unmodified.
"""
import argparse,json,types
from pathlib import Path
import numpy as np
import torch
from sam3 import perflib
from sam3.model.sam3_video_inference import Sam3VideoInference
from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTracking
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--no-perflib',action='store_true');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 perflib.is_enabled=not a.no_perflib
 torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(1);torch.manual_seed(471);rows=[]
 for mux in (False,True):
  cls=Sam3MultiplexTracking if mux else Sam3VideoInference
  tracker=object.__new__(Sam3VideoTrackingMultiplexDemo if mux else Sam3TrackerPredictor)
  for reverse in (False,True):
   for delay in (0,1,3,15):
    for batch in ([1,4,16] if mux else [1]):
     for threshold in (1,3,20):
      h,w=13,19;frames=list(range(11))[::(-1 if reverse else 1)];n=7;ids=[90,2,400,8,22,5,99];inputs=[];removed=[];suppressed=[];unconfirmed=[];raws={}
      for step,f in enumerate(frames):
       current=[] if step in (0,6) else ids
       masks=torch.rand(len(current),h,w,device=a.device)>.7
       if current:
        masks[0]=masks[1];masks[2]=False;masks[3]=False;masks[3,2,4]=True
       score=torch.linspace(.1,.9,len(current),device=a.device);tracker_score=torch.tensor([.5,.5,0.,-1.,.8,0.,1.][:len(current)],device=a.device)
       inputs.append([masks,score,tracker_score]);removed.append([90] if step>=5 else []);suppressed.append([22] if step%3==0 else []);unconfirmed.append([8,400] if step<7 else [])
       raws[f]=dict(obj_id_to_mask={id:masks[j:j+1] for j,id in enumerate(current)},obj_id_to_score={id:float(score[j]) for j,id in enumerate(current)},removed_obj_ids=set(removed[-1]),suppressed_obj_ids=suppressed[-1],unconfirmed_obj_ids=unconfirmed[-1],frame_stats={'num_obj_tracked':len(current),'num_obj_dropped':0})
       raws[f]['obj_id_to_sam2_score' if mux else 'obj_id_to_tracker_score']={id:tracker_score[j] if mux else float(tracker_score[j]) for j,id in enumerate(current)}
      fixture_ids=[list(raws[f]['obj_id_to_mask']) for f in frames]
      actual=torch.ops.sam3_native.video_output_sequence(inputs,fixture_ids,removed,suppressed,unconfirmed,frames,[11,frames[-1],h,w,delay,threshold,batch],reverse,mux)
      obj=types.SimpleNamespace(rank=0,hotstart_delay=delay,masklet_confirmation_consecutive_det_thresh=threshold,postprocess_batch_size=batch,running_in_prod=mux,is_multiplex=False,max_num_objects=1000,tracker=tracker)
      obj._compile_model=lambda:None;obj._get_processing_order=lambda *args,**kwargs:(frames,frames[-1]);latest=[None];cached={};hidden={}
      def raw_frame(state,f,*args,**kwargs):latest[0]=f;return raws[f]
      def cache(state,f,masks,**kwargs):
       cls._cache_frame_outputs(obj,state,f,masks,**kwargs)
       cached[f]=state['cached_frame_outputs'][f];hidden[f]=set().union(*(set(v) for v in kwargs.values() if v is not None))
      obj._run_single_frame_inference=raw_frame;obj._cache_frame_outputs=cache;obj._postprocess_output=types.MethodType(cls._postprocess_output,obj)
      if mux:obj._postprocess_output_batched=types.MethodType(cls._postprocess_output_batched,obj)
      state=dict(feature_cache={},cached_frame_outputs={},orig_height=h,orig_width=w,num_frames=11);checks=0
      for f,out in cls.propagate_in_video(obj,state,reverse=reverse):
       root=f'{f}/';assert actual[root+'emitted_at'].item()==latest[0],(mux,reverse,delay,batch,threshold,f,'emission');checks+=1
       for key,source in [('ids','out_obj_ids'),('scores','out_probs'),('boxes','out_boxes_xywh'),('masks','out_binary_masks'),('centers','out_centers')]:
        if source not in out:continue
        np.testing.assert_array_equal(actual[root+key].numpy(),out[source],err_msg=str((mux,reverse,delay,batch,threshold,f,key)));checks+=1
       assert actual[root+'hidden'].tolist()==sorted(hidden[f]);checks+=1
       assert {int(key.split('/')[-1]) for key in actual if key.startswith(root+'cache/')}==set(cached[f]);checks+=1
       for id,value in cached[f].items():torch.testing.assert_close(actual[root+f'cache/{id}'],value.cpu(),rtol=0,atol=0);checks+=1
      assert actual['pending'].item()==0
      rows.append(dict(model='sam3.1' if mux else 'sam3',reverse=reverse,delay=delay,batch=batch,confirmation_threshold=threshold,checks=checks))
 print('workflows',len(rows),'exact checks',sum(r['checks'] for r in rows),flush=True)
 a.report.write_text(json.dumps(dict(device=a.device,source_perflib=perflib.is_enabled,scope=__doc__,cases=rows),indent=2)+'\n')
if __name__=='__main__':main()
