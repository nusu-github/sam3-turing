"""Original removal parity plus ID/memory conservation across layout changes."""
import argparse,copy,json
from pathlib import Path
from types import SimpleNamespace,MethodType
import torch
from sam3.model.multiplex_utils import MultiplexController,MultiplexState
from sam3.model.video_tracking_multiplex import VideoTrackingMultiplex
from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo

def source_remove(state,tensors,conditions,removed):
    ids=state.object_ids.copy();count=len(ids)
    frame=dict(pred_masks=tensors['low'].clone(),object_score_logits=tensors['logits'].clone(),obj_ptr=tensors['pointer'].clone(),
        maskmem_features=tensors['memory'].clone(),maskmem_pos_enc=[tensors['position'].clone()],iou_score=tensors['iou'].clone(),
        eff_iou_score=tensors['confidence'].clone(),conditioning_objects=set(conditions),local_obj_id_to_idx={value:i for i,value in enumerate(ids)})
    inference=dict(obj_ids=ids.copy(),obj_id_to_idx={value:i for i,value in enumerate(ids)},obj_idx_to_id=dict(enumerate(ids)),
        point_inputs_per_obj={i:{} for i in range(count)},mask_inputs_per_obj={i:{} for i in range(count)},
        output_dict_per_obj={i:{} for i in range(count)},temp_output_dict_per_obj={i:{} for i in range(count)},
        output_dict=dict(cond_frame_outputs={0:frame},non_cond_frame_outputs={}),multiplex_state=state,constants={})
    host=SimpleNamespace(use_memory_selection=True,_add_output_per_object=lambda *args:None)
    host._get_maskmem_pos_enc=MethodType(VideoTrackingMultiplexDemo._get_maskmem_pos_enc,host)
    host.cal_mem_score=MethodType(VideoTrackingMultiplex.cal_mem_score,host)
    VideoTrackingMultiplexDemo.remove_objects(host,inference,[ids[i] for i in removed],strict=True,need_output=False)
    return frame

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(20231);rows=[]
    modes=['fp32','fp16','bf16_reference'] if a.device=='cuda' else ['fp32']
    for mode in modes:
        dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
        for count,removed in [(3,[1]),(17,[16]),(17,list(range(16))),(33,[0,3,17]),(33,list(range(16,32)))]:
            ids=[1000+i*31 for i in range(count)];state=MultiplexController(16).eval().get_state(count,torch.device(a.device),torch.float32,random=False,object_ids=ids)
            assignments=copy.deepcopy(state.assignments);batch=state.num_buckets
            rand=lambda shape:torch.randn(shape,device=a.device,dtype=dtype)
            values=dict(low=rand((count,1,5,7)),high=rand((count,1,10,14)),logits=rand((count,1)),pointer=rand((batch,16,9)),
                memory=rand((batch,256,4,5)).cpu(),position=rand((1,256,4,5)).cpu().expand(batch,-1,-1,-1),iou=rand((count,)),confidence=rand((count,count)),image=rand((20,1,256)))
            conditions=list(range(0,count,2));old_ids=ids.copy()
            with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                expected=source_remove(state,values,conditions,removed)
                actual=torch.ops.sam3_native.multiplex_history(assignments,old_ids,state.assignments,state.object_ids,values,conditions,None,None,mode)
            mapping=dict(low='pred_masks',logits='object_score_logits',pointer='obj_ptr',memory='maskmem_features',iou='iou_score',confidence='eff_iou_score')
            for key,target in mapping.items():torch.testing.assert_close(actual[key],expected[target],rtol=0,atol=0)
            torch.testing.assert_close(actual['position'],expected['maskmem_pos_enc'][-1],rtol=0,atol=0)
            assert set(actual['conditions'].tolist())==expected['conditioning_objects'];assert actual['rebuild_calls'].item()==0
            keep=[i for i in range(count) if i not in removed];torch.testing.assert_close(actual['high'],values['high'][keep],rtol=0,atol=0)
            rows.append(dict(mode=mode,operation='remove',objects=count,removed=removed,source_outputs=8,exact=True))
        # Reordering IDs without changing their physical slots must preserve dense memory.
        old_ids=[71,82,93];old_slots=[[0,1,2]+[-1]*13];new_ids=[93,71,82];new_slots=[[1,2,0]+[-1]*13]
        rand=lambda shape:torch.randn(shape,device=a.device,dtype=dtype)
        values=dict(low=rand((3,1,5,7)),high=rand((3,1,10,14)),logits=rand((3,1)),pointer=rand((1,16,9)),memory=rand((1,256,4,5)).cpu(),position=rand((1,256,4,5)).cpu(),image=rand((20,1,256)))
        out=torch.ops.sam3_native.multiplex_history(old_slots,old_ids,new_slots,new_ids,values,[0,2],None,None,mode)
        for key in ['memory','position','pointer','image']:assert torch.equal(out[key],values[key])
        for key in ['low','high','logits']:assert torch.equal(out[key],values[key][[2,0,1]])
        assert out['conditions'].tolist()==[1,0] and out['rebuild_calls'].item()==0
        rows.append(dict(mode=mode,operation='reorder_IDs_preserve_slots',exact=True))
        # New buckets require absent historical objects; unchanged buckets stay byte-exact.
        fresh_slots=old_slots+[[3]+[-1]*15];fresh_ids=old_ids+[104]
        rebuilt=torch.full((2,256,4,5),777.,device=a.device,dtype=dtype)
        out=torch.ops.sam3_native.multiplex_history(old_slots,old_ids,fresh_slots,fresh_ids,values,[0,2],rebuilt,rebuilt+1,mode)
        assert out['rebuild_calls'].item()==1
        for key in ['low','high','logits']:
            assert torch.equal(out[key][:3],values[key]);assert (out[key][3]==-1024).all()
        assert torch.equal(out['pointer'][0],values['pointer'][0]) and (out['pointer'][1]==0).all()
        assert torch.equal(out['memory'][0],values['memory'][0]) and (out['memory'][1]==777).all()
        assert torch.equal(out['position'][0],values['position'][0]) and (out['position'][1]==778).all()
        rows.append(dict(mode=mode,operation='grow_preserve_old_bucket',exact=True))
        # Moving one object into a singleton changes a joint bucket: require re-encoding.
        single=[[0]+[-1]*15]
        try:torch.ops.sam3_native.multiplex_history(old_slots,old_ids,single,[93],values,[0,2],None,None,mode)
        except RuntimeError as error:assert 'require a rebuilder' in str(error)
        else:raise AssertionError('joint memory was silently sliced without rebuilding')
        out=torch.ops.sam3_native.multiplex_history(old_slots,old_ids,single,[93],values,[0,2],rebuilt[:1],rebuilt[:1]+1,mode)
        assert torch.equal(out['pointer'][0,0],values['pointer'][0,2]) and (out['pointer'][0,1:]==0).all()
        assert torch.equal(out['low'],values['low'][2:3]) and (out['memory']==777).all() and out['conditions'].tolist()==[0]
        rows.append(dict(mode=mode,operation='extract_requires_rebuild',exact=True))
    a.report.write_text(json.dumps(dict(device=a.device,torch=torch.__version__,cases=rows,scope='Original demo remove_objects core tensor parity (input clearing/per-object view rebuilding excluded); independent ID, pointer and dense-memory conservation invariants for growth/reorder/extraction. No claim of full demo-session parity.'),indent=2)+'\n')
    print(json.dumps(dict(passed=len(rows))),flush=True)
if __name__=='__main__':main()
