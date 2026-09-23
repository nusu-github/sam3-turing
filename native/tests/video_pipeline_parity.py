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
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['sam3','sam3.1'],required=True);p.add_argument('--checkpoint');p.add_argument('--native-output',type=Path,required=True);p.add_argument('--frames',nargs='+',type=Path,required=True);p.add_argument('--prompt',default='person');p.add_argument('--mode',choices=['fp16','fp32','bf16_reference'],default='fp16');p.add_argument('--reference-output',type=Path);p.add_argument('--trace-state',action='store_true');p.add_argument('--trace-frames',nargs='+',type=int,help='Save internal traces only for these frame indices; all frames still run');p.add_argument('--reference-cache',type=Path);p.add_argument('--reference-mode',choices=['fp16','fp32','bf16_reference']);p.add_argument('--require-exact',action='store_true');p.add_argument('--source-grounding-batch',type=int,default=16);p.add_argument('--source-complex-rope',action='store_true');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 if a.reference_cache:
  rows=[]
  for i in range(len(a.frames)):
   with np.load(a.reference_cache/f'{i}.npz') as r:rows.append(compare_frame(a.native_output,i,r['ids'].tolist(),r['masks'],r['low'],r['scores'].tolist()))
  save_report(a,rows);return
 assert not a.reference_mode or a.reference_mode==a.mode,'reference mode override only applies to cached comparisons'
 assert a.checkpoint,'checkpoint is required when generating original outputs'
 torch.set_num_threads(1);torch.manual_seed(189);tri=a.model=='sam3.1'
 if tri:wrapper=build_sam3_multiplex_video_predictor(checkpoint_path=a.checkpoint,use_fa3=False,use_rope_real=not a.source_complex_rope,compile=False,warm_up=False,async_loading_frames=False);model=wrapper.model
 else:model=build_sam3_video_model(checkpoint_path=a.checkpoint,load_from_HF=False,compile=False).eval()
 if tri:model.batched_grounding_batch_size=a.source_grounding_batch
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;model._warm_up_complete=True
 for block in model.detector.backbone.vision_backbone.trunk.blocks if hasattr(model.detector.backbone,'vision_backbone') else model.detector.backbone.visual.trunk.blocks:
  if a.mode!='bf16_reference':block.mlp.forward=lambda x,m=block.mlp:m.fc2(m.act(m.fc1(x)))
 dtype={'fp16':torch.float16,'fp32':torch.float32,'bf16_reference':torch.bfloat16}[a.mode];rows=[];captured={}
 original=model.run_tracker_propagation
 def capture(*args,**kwargs):
  result=original(*args,**kwargs);captured['low']=result[0].detach().clone();return result
 model.run_tracker_propagation=capture
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
 save_report(a,rows)
if __name__=='__main__':main()
