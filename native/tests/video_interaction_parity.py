"""Compare action routing, ranges and partial cache merges to actual source methods."""
import argparse,json,random,types
from pathlib import Path
import numpy as np
import torch
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity as Sam3
from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity as Sam31
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--report',type=Path,required=True);a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(1);torch.manual_seed(189);rng=random.Random(283);names=['add','remove','refine','propagation_full','propagation_partial','propagation_fetch','propagation_cancel'];routing=merging=rejected=0
 for mux,cls in [(False,Sam3),(True,Sam31)]:
  obj=types.SimpleNamespace(tracker=object.__new__(Sam3VideoTrackingMultiplexDemo if mux else Sam3TrackerPredictor),running_in_prod=False)
  if mux:obj._parse_action_history_for_propagation=types.MethodType(cls._parse_action_history_for_propagation,obj)
  for case in range(500):
   actions=[]
   for _ in range(rng.randrange(13)):
    t=rng.randrange(7 if mux else 6);actions.append(dict(type=names[t],frame_idx=rng.choice([None,0,3,9]),obj_ids=([rng.choice([1,17,500,9000]) for _ in range(rng.randrange(5))] if t in [0,1,2,4] else None)))
   state=dict(action_history=actions,num_frames=10,previous_stages_out=[None]*10);state['previous_stages_out'][2]='ready';state['previous_stages_out'][7]='ready';start=rng.choice([None,0,4,9]);steps=rng.choice([None,0,1,4,30]);reverse=bool(case%2)
   args=([names.index(x['type']) for x in actions],[-1 if x['frame_idx'] is None else x['frame_idx'] for x in actions],[x['obj_ids'] or [] for x in actions],[x['obj_ids'] is not None for x in actions],10,mux,[2,7],-1 if start is None else start,-1 if steps is None else steps,reverse)
   try:expected=cls.parse_action_history_for_propagation(obj,state)
   except IndexError:
    try:torch.ops.sam3_native.video_action_route(*args)
    except RuntimeError:rejected+=1;continue
    raise AssertionError('source-invalid cancel history accepted')
   actual=torch.ops.sam3_native.video_action_route(*args);assert names[actual['type'].item()]==expected[0];assert actual['has_ids'].item()==(expected[1] is not None);assert sorted(actual['ids'].tolist())==sorted(expected[1] or [])
   order,end=cls._get_processing_order(obj,state,start,steps,reverse);first,last,step,empty=actual['range'].tolist();assert list(order)==([] if empty else list(range(first,last+step,step)));assert last==end;routing+=1
  for cached in (False,True):
   for n in (0,1,7,257):
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
     h,w=17,23;ids=list(range(n));masks=torch.rand(n,h,w,device=a.device)>.7;scores=torch.rand(n,device=a.device);trackers=torch.randn(n,device=a.device);ref_ids=[0,n+5];low=torch.randn(2,11,13,device=a.device,dtype=dtype);ref_scores=torch.tensor([-.2,2.5],device=a.device,dtype=dtype);suppressed=[1,17]
     actual=torch.ops.sam3_native.video_refinement_merge(masks,ids,scores,trackers,low,ref_ids,ref_scores,suppressed,mux,cached)
     state=dict(orig_height=h,orig_width=w,cached_frame_outputs={1:{id:masks[j:j+1] for j,id in enumerate(ids)}} if cached else {})
     converted={id:cls._convert_low_res_mask_to_video_res(obj,low[j],state) for j,id in enumerate(ref_ids)}
     merged=(cls._build_sam2_output if mux else cls._build_tracker_output)(obj,state,1,converted)
     all_scores={id:float(scores[j]) for j,id in enumerate(ids)};all_scores.update({id:1. for id in ref_ids});all_trackers={id:trackers[j] if mux else float(trackers[j]) for j,id in enumerate(ids)};all_trackers.update({id:ref_scores[j] if mux else float(ref_scores[j]) for j,id in enumerate(ref_ids)})
     raw=dict(obj_id_to_mask=merged,obj_id_to_score=all_scores);raw['obj_id_to_sam2_score' if mux else 'obj_id_to_tracker_score']=all_trackers
     expected=cls._postprocess_output(obj,state,raw,suppressed_obj_ids=suppressed);cls._cache_frame_outputs(obj,state,1,merged,suppressed_obj_ids=suppressed)
     for key,source in [('ids','out_obj_ids'),('scores','out_probs'),('boxes','out_boxes_xywh'),('masks','out_binary_masks')]:np.testing.assert_array_equal(actual[key].numpy(),expected[source]);merging+=1
     assert {int(k.split('/')[1]) for k in actual if k.startswith('cache/')}==set(state['cached_frame_outputs'][1]);merging+=1
     for id,value in state['cached_frame_outputs'][1].items():torch.testing.assert_close(actual[f'cache/{id}'],value.cpu(),rtol=0,atol=0);merging+=1
     for id,value in all_trackers.items():assert actual[f'tracker/{id}'].item()==float(value);merging+=1
 print(dict(routing_cases=routing,rejected_cancel_histories=rejected,merge_exact_checks=merging),flush=True)
 a.report.write_text(json.dumps(dict(device=a.device,routing_cases=routing,rejected_cancel_histories=rejected,merge_exact_checks=merging,scope=__doc__),indent=2)+'\n')
if __name__=='__main__':main()
