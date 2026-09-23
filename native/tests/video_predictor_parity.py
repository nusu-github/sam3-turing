"""Actual original semantic replacement/reset versus the owning C++ video predictor."""
import argparse,json,shutil,tempfile
from pathlib import Path
import numpy as np
import torch
import torchvision
from sam3.model_builder import build_sam3_video_model,build_sam3_multiplex_video_predictor

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['sam3','sam3.1'],required=True);p.add_argument('--checkpoint');p.add_argument('--mode',choices=['fp16','bf16_reference'],default='bf16_reference');p.add_argument('--collective-empty-peers',type=int,default=0,help='Test-only replay of original multi-rank collection with one neural rank and explicit empty peers');p.add_argument('--collective-partition-ranks',type=int,default=0,help='Test-only SAM3 rank-local neural replay on one GPU');p.add_argument('--box-steps',type=int,default=1);p.add_argument('--sam31-inclusive-grounding-bound',action='store_true',help='Test-only repair: source generator includes its last frame but batched detector excludes it');p.add_argument('--sam31-preserve-batched-geometry',action='store_true',help='Test-only repair: use stored per-frame semantic geometry when source batched detector builds its prompt');p.add_argument('--reference-cache',type=Path);p.add_argument('--native-output',type=Path,required=True);p.add_argument('--reference-output',type=Path,required=True);p.add_argument('--frames',nargs='+',type=Path,required=True);p.add_argument('--require-exact',action='store_true');p.add_argument('--report',type=Path,required=True);a=p.parse_args();assert 0<=a.box_steps<=len(a.frames)-2,'regression box range exceeds fixture'
    assert a.collective_empty_peers>=0 and a.collective_partition_ranks>=0 and not(a.collective_empty_peers and a.collective_partition_ranks),'choose one collective replay'
    if a.reference_cache:
        rows=[]
        info=a.reference_cache/'reference-info.json';reference_metadata=json.loads(info.read_text()) if info.exists() else {'provenance':'legacy cache; consult original live comparison report'}
        assert reference_metadata.get('mode','bf16_reference')==a.mode,'cached precision differs'
        assert reference_metadata.get('collective_empty_peers',0)==a.collective_empty_peers and reference_metadata.get('collective_partition_ranks',0)==a.collective_partition_ranks,'cached collective replay differs'
        for path in sorted(a.reference_cache.glob('*.npz')):
            tag=path.stem;meta=json.loads((a.native_output/f'{tag}.json').read_text());ids=np.fromfile(a.native_output/f'{tag}.ids.i64.bin',np.int64);n=len(ids);h,w=meta['height'],meta['width']
            with np.load(path) as value:
                actual=dict(out_obj_ids=ids,out_probs=np.fromfile(a.native_output/f'{tag}.scores.f32.bin',np.float32),out_boxes_xywh=np.fromfile(a.native_output/f'{tag}.boxes.f32.bin',np.float32).reshape(n,4));packed=np.fromfile(a.native_output/f'{tag}.masks.bin',np.uint8).reshape(n,h,(w+7)//8);actual['out_binary_masks']=np.unpackbits(packed,axis=-1,bitorder='little')[...,:w].astype(bool)
                equal={k:bool(np.array_equal(v,value[k])) for k,v in actual.items()}
                if 'emitted_at' in value and int(value['emitted_at'])>=0:equal['emission']=meta['emitted_at']==int(value['emitted_at'])
                rows.append(dict(tag=tag,exact=equal,mask_mismatches=int(np.count_nonzero(actual['out_binary_masks']!=value['out_binary_masks'])) if actual['out_binary_masks'].shape==value['out_binary_masks'].shape else -1))
        assert len(rows)==10+a.box_steps,'expected all semantic lifecycle checkpoints'
        exact=all(all(r['exact'].values()) for r in rows);a.report.write_text(json.dumps(dict(model=a.model,cases=rows,exact=exact,reference_metadata=reference_metadata,scope='Cached original semantic lifecycle tensors versus latest standalone outputs. Emission timing is checked only when retained reference metadata includes it. No reference tensors regenerated.'),indent=2)+'\n')
        if a.require_exact:assert exact,'cached owning video outputs differ'
        return
    assert a.checkpoint,'original checkpoint required without reference cache'
    torch.set_num_threads(1);torch.manual_seed(189);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    mux=a.model=='sam3.1'
    if mux:
        wrapper=build_sam3_multiplex_video_predictor(checkpoint_path=a.checkpoint,use_fa3=False,use_rope_real=False,compile=False,warm_up=False,async_loading_frames=False);model=wrapper.model;model.batched_grounding_batch_size=1
    else:model=build_sam3_video_model(checkpoint_path=a.checkpoint,load_from_HF=False,compile=False).eval()
    # Match the existing native FP16 policy; source vision MLP normally forces FP32.
    if a.mode=='fp16':
        for block in model.detector.backbone.vision_backbone.trunk.blocks if hasattr(model.detector.backbone,'vision_backbone') else model.detector.backbone.visual.trunk.blocks:
            block.mlp.forward=lambda x,m=block.mlp:m.fc2(m.act(m.fc1(x)))
    collective_records=[]
    if a.collective_empty_peers:
        assert not mux and a.collective_empty_peers>0,'this diagnostic is scoped to SAM3'
        from video_collective_reference import install_empty_peer_replay
        collective_records=install_empty_peer_replay(model,1+a.collective_empty_peers)
    if a.collective_partition_ranks:
        assert not mux,'partition replay is scoped to SAM3'
        from video_collective_reference import install_partitioned_rank_replay
        collective_records=install_partitioned_rank_replay(model,a.collective_partition_ranks)
    # SAM3.1 wrapper construction unconditionally re-enables TF32. Set the
    # comparison policy AFTER construction, then record the effective flags.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    precision=dict(matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),torch_version=torch.__version__,torchvision_version=torchvision.__version__,numpy_version=np.__version__,cuda_version=torch.version.cuda,torch_build=torch.__config__.show(),fp16_reduced_precision_reduction=bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),bf16_reduced_precision_reduction=bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
    if a.sam31_inclusive_grounding_bound:
        assert mux,'grounding-bound adapter is only for SAM3.1'
        batched=model.detector.forward_video_grounding_batched_multigpu
        def inclusive_grounding(*args,**kwargs):
            count=kwargs.get('max_frame_num_to_track')
            if count is not None and not kwargs.get('track_in_reverse',False):kwargs['max_frame_num_to_track']=count+1
            return batched(*args,**kwargs)
        model.detector.forward_video_grounding_batched_multigpu=inclusive_grounding
    geometry_state=[None];geometry_uses=[]
    if a.sam31_preserve_batched_geometry:
        assert mux,'geometry adapter is only for SAM3.1'
        original_geometry=model.detector._get_geo_prompt_from_find_input
        def per_frame_geometry(find_input):
            frame=int(find_input.img_ids[0]);state=geometry_state[0]
            prompt=None if state is None else state['per_frame_geometric_prompt'][frame]
            if prompt is not None:
                geometry_uses.append(dict(frame=frame,boxes=int(prompt.box_embeddings.size(0))))
                return prompt
            return original_geometry(find_input)
        model.detector._get_geo_prompt_from_find_input=per_frame_geometry
    model._warm_up_complete=True;rows=[];latest=[-1];original=model._run_single_frame_inference
    def capture(state,frame,*args,**kwargs):
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,'reference precision changed after construction'
        latest[0]=frame;geometry_state[0]=state;return original(state,frame,*args,**kwargs)
    model._run_single_frame_inference=capture;a.reference_output.mkdir(parents=True,exist_ok=True)
    (a.reference_output/'reference-info.json').write_text(json.dumps(dict(model=a.model,mode=a.mode,source_fp16_vision_mlp_adapter=a.mode=='fp16',collective_empty_peers=a.collective_empty_peers,collective_partition_ranks=a.collective_partition_ranks,source_grounding_batch=1,source_complex_rope=True,box_steps=a.box_steps,source_inclusive_grounding_bound_adapter=a.sam31_inclusive_grounding_bound,effective_precision=precision,source_preserve_batched_geometry_adapter=a.sam31_preserve_batched_geometry),indent=2)+'\n')
    def check(tag,value,timing=False):
        meta=json.loads((a.native_output/f'{tag}.json').read_text());ids=np.fromfile(a.native_output/f'{tag}.ids.i64.bin',np.int64);n=len(ids);h,w=meta['height'],meta['width']
        actual=dict(out_obj_ids=ids,out_probs=np.fromfile(a.native_output/f'{tag}.scores.f32.bin',np.float32),out_boxes_xywh=np.fromfile(a.native_output/f'{tag}.boxes.f32.bin',np.float32).reshape(n,4));packed=np.fromfile(a.native_output/f'{tag}.masks.bin',np.uint8).reshape(n,h,(w+7)//8);actual['out_binary_masks']=np.unpackbits(packed,axis=-1,bitorder='little')[...,:w].astype(bool)
        equal={k:bool(np.array_equal(v,value[k])) for k,v in actual.items()}
        if timing:equal['emission']=meta['emitted_at']==latest[0]
        same=actual['out_binary_masks'].shape==value['out_binary_masks'].shape
        row=dict(tag=tag,native_ids=ids.tolist(),source_ids=value['out_obj_ids'].tolist(),exact=equal,mask_mismatches=int(np.count_nonzero(actual['out_binary_masks']!=value['out_binary_masks'])) if same else -1)
        if mux and tag=='2.box_track':
            meta_state=state['tracker_metadata'];row['metadata_at_last_processed_frame']=latest[0];row['source_keep_alive']=meta_state['gpu_metadata']['trk_keep_alive'].cpu().tolist()
            row['source_cache_areas']={str(i):int(mask.sum()) for i,mask in state['cached_frame_outputs'][2].items()}
        rows.append(row);print(row,flush=True);np.savez_compressed(a.reference_output/f'{tag}.npz',**{k:np.asarray(v) for k,v in value.items() if k!='frame_stats'},emitted_at=latest[0] if timing else -1)
    with tempfile.TemporaryDirectory() as directory,torch.autocast('cuda',dtype=torch.float16 if a.mode=='fp16' else torch.bfloat16):
        for i,path in enumerate(a.frames):shutil.copy2(path,Path(directory)/f'{i}.jpg')
        state=model.init_state(directory,offload_video_to_cpu=True,async_loading_frames=False)
        _,value=model.add_prompt(state,0,text_str='person');check('0.person',value)
        for i,value in model.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=3,reverse=False):check(f'{i}.person_track',value,True)
        boxes=torch.tensor([[.30,.15,.35,.70]]);labels=torch.tensor([1])
        _,value=model.add_prompt(state,1,boxes_xywh=boxes,box_labels=labels);check('1.box',value)
        for i,value in model.propagate_in_video(state,start_frame_idx=1,max_frame_num_to_track=a.box_steps,reverse=False):check(f'{i}.box_track',value,True)
        _,value=model.add_prompt(state,2,text_str='person',boxes_xywh=boxes,box_labels=torch.tensor([0]));check('2.text_box',value)
        if mux:_,value=model.fetch_and_process_single_frame_results(state,2)
        else:
            data=dict(obj_id_to_mask=state['cached_frame_outputs'][2],obj_id_to_score=state['tracker_metadata']['obj_id_to_score'],obj_id_to_tracker_score=state['tracker_metadata']['obj_id_to_tracker_score_frame_wise'][2]);value=model._postprocess_output(state,data)
        check('2.fetch',value)
        model.reset_state(state);_,value=model.add_prompt(state,0,text_str='person');check('0.reset_person',value)
    if a.sam31_preserve_batched_geometry:assert any(row['frame']==1 and row['boxes']==1 for row in geometry_uses),'stored box geometry adapter was not exercised'
    exact=all(all(r['exact'].values()) for r in rows);a.report.write_text(json.dumps(dict(model=a.model,mode=a.mode,source_fp16_vision_mlp_adapter=a.mode=='fp16',collective_empty_peers=a.collective_empty_peers,collective_partition_ranks=a.collective_partition_ranks,source_grounding_batch=1,source_complex_rope=True,box_steps=a.box_steps,source_inclusive_grounding_bound_adapter=a.sam31_inclusive_grounding_bound,effective_precision=precision,source_preserve_batched_geometry_adapter=a.sam31_preserve_batched_geometry,source_geometry_adapter_uses=geometry_uses,collective_replay_calls=collective_records,cases=rows,exact=exact,scope='Original high-level semantic add_prompt, full propagation with output scheduling, text/box-only/combined replacement, fetch and reset versus owning C++ API. Optional collection replay replaces transport with synchronous copies; partition replay additionally dispatches original neural rank-local sessions serially on one GPU and visits their memory states in rank-concatenated order. It does not test physical distributed execution or instance edits. FP16 explicitly adapts the vision MLP precision; no neural phase is mocked. Optional SAM3.1 forward grounding-bound and stored-geometry adapters are recorded separately; the source generator range/output scheduling is unchanged. Fixed sequence is a fixture, not API input limits. Native additionally checks invalid-input state preservation, point-only cache initialization and callback cancellation; those extra native invariants are not compared here.'),indent=2)+'\n')
    if a.collective_empty_peers:assert collective_records and any(row['objects'] for row in collective_records),'collection replay was not exercised'
    if a.collective_partition_ranks:assert any(row.get('phase')=='propagation' and all(row['ids_per_rank']) for row in collective_records),'rank partition replay was not exercised'
    if a.require_exact:assert exact,'owning video predictor differs; see report'
if __name__=='__main__':main()
