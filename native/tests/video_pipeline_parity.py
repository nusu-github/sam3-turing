"""Compare standalone coherent raw video outputs with the original neural pipeline.

This deliberately calls the source frame engine before predictor temporal filtering.
No source neural phase is mocked. The runtime fixture prompt/frame paths are supplied.
"""
import argparse,json,shutil,tempfile
from pathlib import Path
import numpy as np
import torch
from sam3.model_builder import build_sam3_video_model,build_sam3_multiplex_video_predictor

def compare_frame(root,i,ids,expected,low,source_scores):
 meta=json.loads((root/f'{i}.json').read_text());assert ids==meta['ids'],(ids,meta['ids']);h,w=meta['height'],meta['width'];n=len(ids)
 packed=np.fromfile(root/f'{i}.masks.bin',np.uint8).reshape(n,h,(w+7)//8);actual=np.unpackbits(packed,axis=-1,bitorder='little')[...,:w].astype(bool)
 native_low=np.fromfile(root/f'{i}.tracker_low.f32.bin',np.float32).reshape(low.shape)
 mismatch=np.count_nonzero(actual!=expected,axis=(1,2));union=np.count_nonzero(actual|expected,axis=(1,2));intersection=np.count_nonzero(actual&expected,axis=(1,2))
 return dict(frame=i,ids=ids,mask_mismatches=mismatch.tolist(),mask_iou=np.divide(intersection,union,out=np.ones(n),where=union>0).tolist(),max_tracker_error=float(np.max(np.abs(low-native_low))) if low.size else 0.,native_scores=meta['scores'],source_scores=source_scores)

def save_report(a,rows):
 exact=all(not any(r['mask_mismatches']) and r['max_tracker_error']==0 and np.allclose(r['native_scores'],r['source_scores'],rtol=0,atol=1e-8) for r in rows)
 a.report.write_text(json.dumps(dict(model=a.model,mode=a.mode,reference_mode=a.reference_mode or a.mode,source_grounding_batch=a.source_grounding_batch,source_complex_rope=a.source_complex_rope,cases=rows,exact=exact,scope='Original raw frame engine with actual neural detection, propagation and updates versus standalone C++ probe. Original feature casts retained; batching and RoPE comparison configuration recorded explicitly. Predictor temporal/output filtering and interactive actions are outside this comparison. FP16/FP32 source MLP uses requested precision adaptation; BF16 source neural execution is unchanged. TF32 disabled, one CPU initialization thread. JSON scores tolerate 1e-8 serialization rounding. Cached references, when used, contain the original neural outputs.'),indent=2)+'\n')
 if a.require_exact:assert exact,'raw pipeline differs; see report'

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['sam3','sam3.1'],required=True);p.add_argument('--checkpoint');p.add_argument('--native-output',type=Path,required=True);p.add_argument('--frames',nargs='+',type=Path,required=True);p.add_argument('--prompt',default='person');p.add_argument('--mode',choices=['fp16','fp32','bf16_reference'],default='fp16');p.add_argument('--reference-output',type=Path);p.add_argument('--edit-output',action='store_true');p.add_argument('--extended-edit-output',action='store_true');p.add_argument('--sam31-preserve-singleton-history',action='store_true');p.add_argument('--sam31-refresh-refined-pointer',action='store_true');p.add_argument('--sam31-refresh-refined-memory',action='store_true');p.add_argument('--sam31-default-remove-frame',action='store_true');p.add_argument('--sam31-enable-repeat-refinement',action='store_true',help='Explicitly enable original iter_use_prev_mask_pred, required by its repeated-point path');p.add_argument('--sam31-rebuild-extracted-memory',action='store_true',help='Explicit corrected-source comparison: re-encode dense singleton history lost by upstream slot demux');p.add_argument('--partial-output',action='store_true');p.add_argument('--final-output',action='store_true',help='Replay actual raw source results through its original output generator and compare native final output');p.add_argument('--trace-edit-state',action='store_true');p.add_argument('--trace-state',action='store_true');p.add_argument('--trace-frames',nargs='+',type=int,help='Save internal traces only for these frame indices; all frames still run');p.add_argument('--reference-cache',type=Path);p.add_argument('--reference-mode',choices=['fp16','fp32','bf16_reference']);p.add_argument('--require-exact',action='store_true');p.add_argument('--source-grounding-batch',type=int,default=16);p.add_argument('--source-complex-rope',action='store_true');p.add_argument('--report',type=Path,required=True);a=p.parse_args();a.edit_output=a.edit_output or a.extended_edit_output
 if a.reference_cache:
  rows=[]
  for i in range(len(a.frames)):
   with np.load(a.reference_cache/f'{i}.npz') as r:rows.append(compare_frame(a.native_output,i,r['ids'].tolist(),r['masks'],r['low'],r['scores'].tolist()))
  save_report(a,rows)
  tasks=[]
  if a.final_output:tasks.append(('final',[f'{i}.final' for i in range(len(a.frames))]))
  if a.partial_output:tasks.append(('partial',[f'{i}.partial' for i in (17,16,15)]))
  if a.edit_output and a.model=='sam3.1':
   from sam31_extraction_reference import edit_tags
   tasks.append(('edit',edit_tags(a.extended_edit_output)))
  if a.edit_output and a.model=='sam3':tasks.append(('edit',['18.edit_point','20.edit_mask_new','21.edit_mask_existing']+[f'{i}.edit_track' for i in range(18,23)]+['20.edit_remove','22.edit_stateless']))
  for kind,tags in tasks:
   checked=[]
   for tag in tags:
    with np.load(a.reference_cache/f'{tag}.npz') as ref:
     masks=ref['out_binary_masks'];h,w=masks.shape[-2:];ids=np.fromfile(a.native_output/f'{tag}.ids.i64.bin',np.int64);n=len(ids)
     actual=dict(out_obj_ids=ids,out_probs=np.fromfile(a.native_output/f'{tag}.scores.f32.bin',np.float32),out_boxes_xywh=np.fromfile(a.native_output/f'{tag}.boxes.f32.bin',np.float32).reshape(n,4));packed=np.fromfile(a.native_output/f'{tag}.masks.bin',np.uint8).reshape(n,h,(w+7)//8);actual['out_binary_masks']=np.unpackbits(packed,axis=-1,bitorder='little')[...,:w].astype(bool)
     equal={key:bool(np.array_equal(value,ref[key])) for key,value in actual.items()}
     if kind=='final':equal['emission']=json.loads((a.native_output/f'{tag}.json').read_text())['emitted_at']==int(ref['emitted_at'])
     checked.append(dict(tag=tag,exact=equal,mask_mismatches=int(np.count_nonzero(actual['out_binary_masks']!=masks)) if actual['out_binary_masks'].shape==masks.shape else -1))
   reference_metadata=json.loads((a.reference_cache/'edit-reference.json').read_text()) if kind=='edit' and (a.reference_cache/'edit-reference.json').exists() else {}
   exact=all(all(row['exact'].values()) for row in checked);a.report.with_suffix(f'.{kind}.json').write_text(json.dumps(dict(model=a.model,mode=a.mode,reference_mode=a.reference_mode or a.mode,scope='Compare latest standalone outputs with retained neural/predictor reference tensors. No reference tensors regenerated in this cached check. Any corrected-source extraction adapter is recorded separately.',reference_metadata=reference_metadata,cases=checked,exact=exact),indent=2)+'\n')
   if a.require_exact:assert exact,f'{kind} cached outputs differ'
  return
 assert not a.reference_mode or a.reference_mode==a.mode,'reference mode override only applies to cached comparisons'
 assert a.checkpoint,'checkpoint is required when generating original outputs'
 torch.set_num_threads(1);torch.manual_seed(189);tri=a.model=='sam3.1'
 if tri:wrapper=build_sam3_multiplex_video_predictor(checkpoint_path=a.checkpoint,use_fa3=False,use_rope_real=not a.source_complex_rope,compile=False,warm_up=False,async_loading_frames=False);model=wrapper.model
 else:model=build_sam3_video_model(checkpoint_path=a.checkpoint,load_from_HF=False,compile=False).eval()
 if tri:model.batched_grounding_batch_size=a.source_grounding_batch
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;model._warm_up_complete=True
 for block in model.detector.backbone.vision_backbone.trunk.blocks if hasattr(model.detector.backbone,'vision_backbone') else model.detector.backbone.visual.trunk.blocks:
  if a.mode!='bf16_reference':block.mlp.forward=lambda x,m=block.mlp:m.fc2(m.act(m.fc1(x)))
 raw_outputs={};final_rows=[]
 dtype={'fp16':torch.float16,'fp32':torch.float32,'bf16_reference':torch.bfloat16}[a.mode];rows=[];captured={}
 original=model.run_tracker_propagation
 def capture(*args,**kwargs):
  result=original(*args,**kwargs);captured['low']=result[0].detach().clone();return result
 model.run_tracker_propagation=capture
 if a.trace_edit_state:
  assert tri and a.reference_output and a.extended_edit_output
  from sam31_reverse_trace import capture_reverse_input
  capture_reverse_input(model,a.reference_output)
 if a.sam31_preserve_singleton_history:
  assert tri and a.extended_edit_output
  from sam31_extraction_reference import preserve_singleton_history
  a.singleton_history_differences=preserve_singleton_history(model)
 if a.sam31_refresh_refined_pointer:
  assert tri and a.extended_edit_output
  from sam31_extraction_reference import refresh_refined_pointer
  a.pointer_refresh_audit=refresh_refined_pointer(model)
 if a.sam31_refresh_refined_memory:
  assert tri and a.extended_edit_output
  from sam31_extraction_reference import refresh_refined_memory
  refresh_refined_memory(model)
 if a.sam31_default_remove_frame:
  assert tri and a.extended_edit_output
  original_remove=model.remove_object
  model.remove_object=lambda state,obj_id,frame_idx=None,is_user_action=False:original_remove(state,obj_id,frame_idx,is_user_action)
 if a.sam31_enable_repeat_refinement:
  assert tri and a.extended_edit_output
  model.tracker.model.iter_use_prev_mask_pred=True
 if a.sam31_rebuild_extracted_memory:
  assert tri and a.edit_output
  from sam31_extraction_reference import preserve_extracted_memory
  preserve_extracted_memory(model)
 if tri and a.trace_state:
  memory_encoder=model.tracker._run_memory_encoder
  def capture_memory(state,index,batch,high,scores,*args,**kwargs):
   captured['memory_inputs']=(index,high.detach().clone(),scores.detach().clone())
   return memory_encoder(state,index,batch,high,scores,*args,**kwargs)
  model.tracker._run_memory_encoder=capture_memory
  def capture_encoder(module,args,kwargs):
   captured['encoder_inputs']=[x.detach().clone() for x in args[:2]]
   captured['encoder_layout']=[dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype)) for x in args[:2]]
  model.tracker.maskmem_backbone.register_forward_pre_hook(capture_encoder,with_kwargs=True)
 with tempfile.TemporaryDirectory() as directory,torch.autocast('cuda',enabled=a.mode!='fp32',dtype=dtype):
  for i,path in enumerate(a.frames):shutil.copy2(path,Path(directory)/f'{i}.jpg')
  state=model.init_state(directory,offload_video_to_cpu=True,async_loading_frames=False)
  state['text_prompt']=a.prompt;state['input_batch'].find_text_batch[0]=a.prompt
  if tri:state['backbone_out']=model.detector.backbone.forward_text(state['input_batch'].find_text_batch,device='cuda')
  for i in range(len(a.frames)):
   print('FRAME',a.model,a.mode,i,flush=True);out=model._run_single_frame_inference(state,i,False)
   if a.final_output:raw_outputs[i]=out
   ids=list(map(int,sorted(out['obj_id_to_mask'])));h,w=state['orig_height'],state['orig_width']
   expected=torch.cat([out['obj_id_to_mask'][k] for k in ids]).cpu().numpy() if ids else np.empty((0,h,w),bool)
   low=captured['low'].float().cpu().numpy();scores=[float(out['obj_id_to_score'][k]) for k in ids]
   if a.reference_output:
    a.reference_output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.reference_output/f'{i}.npz',masks=expected,low=low,ids=ids,scores=scores)
   if tri and a.reference_output and a.trace_state and (a.trace_frames is None or i in a.trace_frames):
    for si,session in enumerate(state['sam2_inference_states']):
     for history in session['output_dict'].values():
      if i not in history:continue
      v=history[i];fields={}
      for name,key in [('memory','maskmem_features'),('position','maskmem_pos_enc'),('image','image_features'),('image_position','image_pos_enc'),('pointer','obj_ptr'),('low','pred_masks')]:
       value=v.get(key)
       if isinstance(value,list):value=value[-1]
       if isinstance(value,torch.Tensor):fields[name]=value.float().cpu().numpy()
      if captured.get('memory_inputs',(None,))[0]==i:
       fields['high_input']=captured['memory_inputs'][1].float().cpu().numpy();fields['proxy']=captured['memory_inputs'][2].float().cpu().numpy()
      fields['conditions']=np.array(sorted(v['conditioning_objects']),dtype=np.int64)
      for ei,value in enumerate(captured.get('encoder_inputs',[])):fields[f'encoder_input{ei}']=value.float().cpu().numpy()
      (a.reference_output/f'{i}.layout.json').write_text(json.dumps(captured.get('encoder_layout',[])))
      np.savez_compressed(a.reference_output/f'{i}.state{si}.npz',**fields)
   row=compare_frame(a.native_output,i,ids,expected,low,scores);rows.append(row);print(row,flush=True)
 if a.final_output:
  # Replay only the generator boundary: these are outputs of the actual neural
  # engine above. Original buffering, caching and postprocessing remain intact.
  latest=[None];original_raw=model._run_single_frame_inference
  def replay(state,index,*args,**kwargs):latest[0]=index;return raw_outputs[index]
  model._run_single_frame_inference=replay
  from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTracking
  from sam3.model.sam3_video_inference import Sam3VideoInference
  generator=Sam3MultiplexTracking.propagate_in_video if tri else Sam3VideoInference.propagate_in_video
  for index,value in generator(model,state,start_frame_idx=0,max_frame_num_to_track=len(a.frames)-1,reverse=False):
   root=a.native_output;meta=json.loads((root/f'{index}.final.json').read_text());n=meta['count'];h,w=state['orig_height'],state['orig_width']
   actual=dict(out_obj_ids=np.fromfile(root/f'{index}.final.ids.i64.bin',np.int64),out_probs=np.fromfile(root/f'{index}.final.scores.f32.bin',np.float32),out_boxes_xywh=np.fromfile(root/f'{index}.final.boxes.f32.bin',np.float32).reshape(n,4))
   packed=np.fromfile(root/f'{index}.final.masks.bin',np.uint8).reshape(n,h,(w+7)//8);actual['out_binary_masks']=np.unpackbits(packed,axis=-1,bitorder='little')[...,:w].astype(bool)
   equal={key:bool(np.array_equal(v,value[key])) for key,v in actual.items()};equal['emission']=meta['emitted_at']==latest[0];final_rows.append(dict(frame=index,emitted_at=latest[0],objects=n,exact=equal))
   if a.reference_output:np.savez_compressed(a.reference_output/f'{index}.final.npz',**{k:np.asarray(v) for k,v in value.items() if k!='frame_stats'},emitted_at=latest[0])
  model._run_single_frame_inference=original_raw
  a.report.with_suffix('.final.json').write_text(json.dumps(dict(model=a.model,mode=a.mode,scope='Actual source neural raw outputs replayed through original predictor generator, cache and final postprocessing. No neural outputs mocked. Temporal and final outputs compared with standalone C++; interactive actions remain outside this fixture.',cases=final_rows,exact=all(all(r['exact'].values()) for r in final_rows)),indent=2)+'\n')
  if a.require_exact:assert final_rows and all(all(r['exact'].values()) for r in final_rows),'final predictor outputs differ'
 save_report(a,rows)
 if a.partial_output:
  assert a.final_output and len(a.frames)>18,'partial regression needs final-output and at least 19 frames'
  selected=int(state['tracker_metadata']['obj_ids_all_gpu'][0]);model.add_action_history(state,'refine',frame_idx=18,obj_ids=[selected]);partial_rows=[]
  with torch.autocast('cuda',enabled=a.mode!='fp32',dtype=dtype):
   for index,value in model.propagate_in_video(state,start_frame_idx=18,max_frame_num_to_track=3,reverse=True):
    root=a.native_output;ids=np.fromfile(root/f'{index}.partial.ids.i64.bin',np.int64);n=len(ids);h,w=state['orig_height'],state['orig_width']
    actual=dict(out_obj_ids=ids,out_probs=np.fromfile(root/f'{index}.partial.scores.f32.bin',np.float32),out_boxes_xywh=np.fromfile(root/f'{index}.partial.boxes.f32.bin',np.float32).reshape(n,4))
    packed=np.fromfile(root/f'{index}.partial.masks.bin',np.uint8).reshape(n,h,(w+7)//8);actual['out_binary_masks']=np.unpackbits(packed,axis=-1,bitorder='little')[...,:w].astype(bool)
    equal={key:bool(np.array_equal(v,value[key])) for key,v in actual.items()};partial_rows.append(dict(frame=index,selected=selected,objects=n,exact=equal,mask_mismatches=int(np.count_nonzero(actual['out_binary_masks']!=value['out_binary_masks'])) if actual['out_binary_masks'].shape==value['out_binary_masks'].shape else -1))
    if a.reference_output:np.savez_compressed(a.reference_output/f'{index}.partial.npz',**{k:np.asarray(v) for k,v in value.items() if k!='frame_stats'})
  a.report.with_suffix('.partial.json').write_text(json.dumps(dict(model=a.model,mode=a.mode,scope='Full real-neural forward pass and original base output stage, followed by a refine action selecting the first tracked ID and original high-level partial reverse propagation over17,16,15. No new point/mask edit is applied in this regression. Standalone C++ selects actual sessions, encodes memories and merges only selected IDs into prior cached outputs.',cases=partial_rows,exact=all(all(r['exact'].values()) for r in partial_rows)),indent=2)+'\n')
  if a.require_exact:assert len(partial_rows)==3 and all(all(r['exact'].values()) for r in partial_rows),'partial outputs differ'
 if a.edit_output and tri:
  assert a.final_output and not a.partial_output and len(a.frames)>20
  from sam31_extraction_reference import compare_edits
  compare_edits(a,model,state,dtype)
 if a.edit_output and not tri:
  assert a.final_output and not tri and not a.partial_output and len(a.frames)>22,'edit regression requires SAM3 final-output,23frames and no preceding partial regression'
  selected,other=map(int,state['tracker_metadata']['obj_ids_all_gpu'][:2]);edit_rows=[];h,w=state['orig_height'],state['orig_width']
  def compare_edit(tag,value):
   root=a.native_output;ids=np.fromfile(root/f'{tag}.ids.i64.bin',np.int64);n=len(ids)
   actual=dict(out_obj_ids=ids,out_probs=np.fromfile(root/f'{tag}.scores.f32.bin',np.float32),out_boxes_xywh=np.fromfile(root/f'{tag}.boxes.f32.bin',np.float32).reshape(n,4));packed=np.fromfile(root/f'{tag}.masks.bin',np.uint8).reshape(n,h,(w+7)//8);actual['out_binary_masks']=np.unpackbits(packed,axis=-1,bitorder='little')[...,:w].astype(bool)
   equal={key:bool(np.array_equal(v,value[key])) for key,v in actual.items()};row=dict(tag=tag,exact=equal,mask_mismatches=int(np.count_nonzero(actual['out_binary_masks']!=value['out_binary_masks'])) if actual['out_binary_masks'].shape==value['out_binary_masks'].shape else -1);edit_rows.append(row);print(row,flush=True)
   if a.reference_output:np.savez_compressed(a.reference_output/f'{tag}.npz',**{k:np.asarray(v) for k,v in value.items() if k!='frame_stats'})
  with torch.autocast('cuda',enabled=a.mode!='fp32',dtype=dtype):
   points=torch.tensor([[.45,.55],[.8,.15]],device='cuda');labels=torch.tensor([1,0],device='cuda')
   _,value=model.add_prompt(state,18,points=points,point_labels=labels,obj_id=selected);compare_edit('18.edit_point',value)
   mask=torch.zeros(h,w,device='cuda');mask[h//4:3*h//4,w//3:2*w//3]=1
   _,value=model.add_tracker_new_mask(state,20,9000,mask);compare_edit('20.edit_mask_new',value)
   _,value=model.add_tracker_new_mask(state,21,selected,mask.roll((h//10,w//12),(0,1)));compare_edit('21.edit_mask_existing',value)
   for index,value in model.propagate_in_video(state,start_frame_idx=18,max_frame_num_to_track=4,reverse=False):compare_edit(f'{index}.edit_track',value)
   model.remove_object(state,9000,is_user_action=True)
   for index,value in model.propagate_in_video(state,start_frame_idx=20,max_frame_num_to_track=0,reverse=False):compare_edit('20.edit_remove',value)
   model.use_stateless_refinement=True
   _,value=model.add_prompt(state,22,points=points,point_labels=labels,obj_id=other);compare_edit('22.edit_stateless',value)
  a.report.with_suffix('.edit.json').write_text(json.dumps(dict(model=a.model,mode=a.mode,scope='Actual original high-level SAM3 point edit, new and existing exact-mask edits, five-frame partial propagation, user removal/fetch and stateless first point refinement after34real-neural forward frames. Standalone C++ controller uses the same shared neural cores and cache.',cases=edit_rows,exact=all(all(r['exact'].values()) for r in edit_rows)),indent=2)+'\n')
  if a.require_exact:assert len(edit_rows)==10 and all(all(r['exact'].values()) for r in edit_rows),'edited video outputs differ'
if __name__=='__main__':main()
