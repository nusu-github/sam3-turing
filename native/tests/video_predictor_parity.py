"""Actual original semantic replacement/reset versus the owning C++ video predictor."""
import argparse,json,shutil,tempfile
from pathlib import Path
import numpy as np
import torch
from sam3.model_builder import build_sam3_video_model,build_sam3_multiplex_video_predictor

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['sam3','sam3.1'],required=True);p.add_argument('--checkpoint');p.add_argument('--sam31-inclusive-grounding-bound',action='store_true',help='Test-only repair: source generator includes its last frame but batched detector excludes it');p.add_argument('--reference-cache',type=Path);p.add_argument('--native-output',type=Path,required=True);p.add_argument('--reference-output',type=Path,required=True);p.add_argument('--frames',nargs='+',type=Path,required=True);p.add_argument('--require-exact',action='store_true');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    if a.reference_cache:
        rows=[]
        info=a.reference_cache/'reference-info.json';reference_metadata=json.loads(info.read_text()) if info.exists() else {'provenance':'legacy cache; consult original live comparison report'}
        for path in sorted(a.reference_cache.glob('*.npz')):
            tag=path.stem;meta=json.loads((a.native_output/f'{tag}.json').read_text());ids=np.fromfile(a.native_output/f'{tag}.ids.i64.bin',np.int64);n=len(ids);h,w=meta['height'],meta['width']
            with np.load(path) as value:
                actual=dict(out_obj_ids=ids,out_probs=np.fromfile(a.native_output/f'{tag}.scores.f32.bin',np.float32),out_boxes_xywh=np.fromfile(a.native_output/f'{tag}.boxes.f32.bin',np.float32).reshape(n,4));packed=np.fromfile(a.native_output/f'{tag}.masks.bin',np.uint8).reshape(n,h,(w+7)//8);actual['out_binary_masks']=np.unpackbits(packed,axis=-1,bitorder='little')[...,:w].astype(bool)
                equal={k:bool(np.array_equal(v,value[k])) for k,v in actual.items()}
                if 'emitted_at' in value and int(value['emitted_at'])>=0:equal['emission']=meta['emitted_at']==int(value['emitted_at'])
                rows.append(dict(tag=tag,exact=equal))
        assert len(rows)==11,'expected all semantic lifecycle checkpoints'
        exact=all(all(r['exact'].values()) for r in rows);a.report.write_text(json.dumps(dict(model=a.model,cases=rows,exact=exact,reference_metadata=reference_metadata,scope='Cached original semantic lifecycle tensors versus latest standalone outputs. Emission timing is checked only when retained reference metadata includes it. No reference tensors regenerated.'),indent=2)+'\n')
        if a.require_exact:assert exact,'cached owning video outputs differ'
        return
    assert a.checkpoint,'original checkpoint required without reference cache'
    torch.set_num_threads(1);torch.manual_seed(189);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    mux=a.model=='sam3.1'
    if mux:
        wrapper=build_sam3_multiplex_video_predictor(checkpoint_path=a.checkpoint,use_fa3=False,use_rope_real=False,compile=False,warm_up=False,async_loading_frames=False);model=wrapper.model;model.batched_grounding_batch_size=1
    else:model=build_sam3_video_model(checkpoint_path=a.checkpoint,load_from_HF=False,compile=False).eval()
    if a.sam31_inclusive_grounding_bound:
        assert mux,'grounding-bound adapter is only for SAM3.1'
        batched=model.detector.forward_video_grounding_batched_multigpu
        def inclusive_grounding(*args,**kwargs):
            count=kwargs.get('max_frame_num_to_track')
            if count is not None and not kwargs.get('track_in_reverse',False):kwargs['max_frame_num_to_track']=count+1
            return batched(*args,**kwargs)
        model.detector.forward_video_grounding_batched_multigpu=inclusive_grounding
    model._warm_up_complete=True;rows=[];latest=[-1];original=model._run_single_frame_inference
    def capture(state,frame,*args,**kwargs):latest[0]=frame;return original(state,frame,*args,**kwargs)
    model._run_single_frame_inference=capture;a.reference_output.mkdir(parents=True,exist_ok=True)
    (a.reference_output/'reference-info.json').write_text(json.dumps(dict(model=a.model,mode='bf16_reference',source_grounding_batch=1,source_complex_rope=True,source_inclusive_grounding_bound_adapter=a.sam31_inclusive_grounding_bound),indent=2)+'\n')
    def check(tag,value,timing=False):
        meta=json.loads((a.native_output/f'{tag}.json').read_text());ids=np.fromfile(a.native_output/f'{tag}.ids.i64.bin',np.int64);n=len(ids);h,w=meta['height'],meta['width']
        actual=dict(out_obj_ids=ids,out_probs=np.fromfile(a.native_output/f'{tag}.scores.f32.bin',np.float32),out_boxes_xywh=np.fromfile(a.native_output/f'{tag}.boxes.f32.bin',np.float32).reshape(n,4));packed=np.fromfile(a.native_output/f'{tag}.masks.bin',np.uint8).reshape(n,h,(w+7)//8);actual['out_binary_masks']=np.unpackbits(packed,axis=-1,bitorder='little')[...,:w].astype(bool)
        equal={k:bool(np.array_equal(v,value[k])) for k,v in actual.items()}
        if timing:equal['emission']=meta['emitted_at']==latest[0]
        same=actual['out_binary_masks'].shape==value['out_binary_masks'].shape
        row=dict(tag=tag,native_ids=ids.tolist(),source_ids=value['out_obj_ids'].tolist(),exact=equal,mask_mismatches=int(np.count_nonzero(actual['out_binary_masks']!=value['out_binary_masks'])) if same else -1)
        rows.append(row);print(row,flush=True);np.savez_compressed(a.reference_output/f'{tag}.npz',**{k:np.asarray(v) for k,v in value.items() if k!='frame_stats'},emitted_at=latest[0] if timing else -1)
    with tempfile.TemporaryDirectory() as directory,torch.autocast('cuda',dtype=torch.bfloat16):
        for i,path in enumerate(a.frames):shutil.copy2(path,Path(directory)/f'{i}.jpg')
        state=model.init_state(directory,offload_video_to_cpu=True,async_loading_frames=False)
        _,value=model.add_prompt(state,0,text_str='person');check('0.person',value)
        for i,value in model.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=3,reverse=False):check(f'{i}.person_track',value,True)
        boxes=torch.tensor([[.30,.15,.35,.70]]);labels=torch.tensor([1])
        _,value=model.add_prompt(state,1,boxes_xywh=boxes,box_labels=labels);check('1.box',value)
        for i,value in model.propagate_in_video(state,start_frame_idx=1,max_frame_num_to_track=1,reverse=False):check(f'{i}.box_track',value,True)
        _,value=model.add_prompt(state,2,text_str='person',boxes_xywh=boxes,box_labels=torch.tensor([0]));check('2.text_box',value)
        if mux:_,value=model.fetch_and_process_single_frame_results(state,2)
        else:
            data=dict(obj_id_to_mask=state['cached_frame_outputs'][2],obj_id_to_score=state['tracker_metadata']['obj_id_to_score'],obj_id_to_tracker_score=state['tracker_metadata']['obj_id_to_tracker_score_frame_wise'][2]);value=model._postprocess_output(state,data)
        check('2.fetch',value)
        model.reset_state(state);_,value=model.add_prompt(state,0,text_str='person');check('0.reset_person',value)
    exact=all(all(r['exact'].values()) for r in rows);a.report.write_text(json.dumps(dict(model=a.model,mode='bf16_reference',source_grounding_batch=1,source_complex_rope=True,source_inclusive_grounding_bound_adapter=a.sam31_inclusive_grounding_bound,cases=rows,exact=exact,scope='Original high-level semantic add_prompt, full propagation with output scheduling, text/box-only/combined replacement, fetch and reset versus owning C++ API. No neural phases mocked. Optional SAM3.1 forward grounding-bound adapter is recorded separately; the source generator range/output scheduling is unchanged. Fixed sequence is a fixture, not API input limits. Native additionally checks invalid-input state preservation, point-only cache initialization and callback cancellation; those extra native invariants are not compared here.'),indent=2)+'\n')
    if a.require_exact:assert exact,'owning video predictor differs; see report'
if __name__=='__main__':main()
