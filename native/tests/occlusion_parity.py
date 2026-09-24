"""Compare native policies with actual source methods and extracted planning blocks.

The fake tracker records mask-edit recipes; it does not execute neural updates.
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import textwrap
import types
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import sam3.model.sam3_video_base as base
import sam3.model.sam3_multiplex_base as mux
from sam3.model.sam3_tracker_utils import mask_to_box, fill_holes_in_mask_scores
from sam3.model.box_ops import fast_diag_box_iou
from hotstart_parity import CONFIGS, FIELDS, options, source_state_helpers

CHECKS = Counter()

def equal(a, b, category):
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    CHECKS[category] += 1

def same(a, b, category):
    assert a == b, (category, a, b)
    CHECKS[category] += 1

def context(device, mode):
    return torch.autocast(device, dtype=DTYPES[mode], enabled=mode != 'fp32')

DTYPES = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16_reference': torch.bfloat16}

def host(cls, threshold=.5, allow=False):
    out = types.SimpleNamespace(rank=1,
        suppress_overlapping_based_on_recent_occlusion_threshold=threshold,
        allow_unoccluded_to_suppress=allow)
    out._get_objects_to_suppress_based_on_most_recently_occluded = types.MethodType(
        cls._get_objects_to_suppress_based_on_most_recently_occluded, out)
    return out

def source_gate(cls):
    # Execute the original inline gate, including its first-pair/any-score rules.
    node = ast.parse(textwrap.dedent(inspect.getsource(cls.run_tracker_update_planning_phase))).body[0]
    def assigned(stmt, name):
        return isinstance(stmt, ast.Assign) and any(isinstance(x, ast.Name) and x.id == name for x in stmt.targets)
    first = next(i for i, x in enumerate(node.body) if assigned(x, 'should_recondition_iou'))
    last = next(i for i, x in enumerate(node.body) if assigned(x, 'should_recondition_periodic'))
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(node.body[first:last+1]), type_ignores=[]))
    return compile(module, inspect.getfile(cls), 'exec')

GATES = {False: source_gate(base.Sam3VideoBase), True: source_gate(mux.Sam3MultiplexBase)}

def occlusion_cases():
    g = torch.Generator().manual_seed(508)
    for n, h, w in [(0,3,5),(1,3,5),(2,1,1),(9,7,11),(201,5,7),(3,288,288)]:
        masks = torch.randn(n,h,w,generator=g)
        if n > 1: masks[1] = masks[0]
        if n > 2: masks[2] = -1
        if h == 288: masks[:] = 1
        yield masks
    masks = torch.randn(9,11,7,generator=g).transpose(1,2)
    masks[0] = float('nan'); masks[1] = 1; masks[2] = masks[1]
    yield masks

def check_occlusion(device, mode):
    for case, template in enumerate(occlusion_cases()):
      masks = template.to(device, dtype=DTYPES[mode]); n = len(masks)
      ids = np.arange(n, dtype=np.int64)*17-20000000000
      last = torch.tensor(([-1,0,3,3,100000,100001,-1]*((n+6)//7))[:n], device=device)
      if n == 0: last = last.long()
      removed = torch.arange(n,device=device)%3 == 1
      known = torch.arange(n,device=device)%4 != 0
      for multiplex, allow in [(False,False),(True,False),(True,True)]:
        cls = mux.Sam3MultiplexBase if multiplex else base.Sam3VideoBase
        for reverse in [False,True]:
          for threshold in [0.,.5,1.]:
            frame = 100002 if case%2 else 5
            owner = host(cls,threshold,allow)
            prior = torch.where(removed,100000,last) if multiplex else torch.where(known,last,torch.where(removed,100000,-1))
            with context(device,mode):
                suppressed = owner._get_objects_to_suppress_based_on_most_recently_occluded(masks>0,prior,ids,frame,reverse)
            expected = masks.clone(); expected[suppressed] = -10
            updated = prior.clone(); updated[(~(masks>0).any(dim=(-1,-2)))|suppressed] = frame
            original = masks.clone()
            actual = torch.ops.sam3_native.occlusion(masks,last,removed,known,frame,reverse,multiplex,threshold,allow,mode)
            for a,b in zip(actual,[expected,suppressed,updated]): equal(a,b,'occlusion')
            equal(masks,original,'input_immutability')
            if not multiplex:
                history = {int(ids[i]):last[i:i+1] for i in range(n) if known[i]}
                # Empty batch must retain unrelated history, as source does.
                history[777] = torch.tensor([3],device=device)
                previous = {'obj_ids_all_gpu':ids,'obj_id_to_last_occluded':history}
                new = {'obj_id_to_last_occluded':dict(history)}
                with context(device,mode):
                    result = cls._suppress_overlapping_based_on_recent_occlusion(owner,frame,masks.clone(),previous,new,set(ids[removed.cpu().numpy()]),reverse)
                out = torch.ops.sam3_native.occlusion_host(history,ids.tolist(),masks,ids[removed.cpu().numpy()].tolist(),frame,reverse,threshold,mode)
                equal(out['masks'],result,'host_occlusion')
                native_history = dict(zip(out['history_ids'].tolist(),out['history_values'].tolist()))
                same(native_history,{int(k):v.item() for k,v in new['obj_id_to_last_occluded'].items()},'host_occlusion')


def check_state_pipeline(device, mode):
    helpers,_ = source_state_helpers(); config = CONFIGS[0]; rng = np.random.default_rng(673)
    for reverse in [False,True]:
        native = torch.ops.sam3_native.hotstart_extend({},5,0,0,device)
        source = {k:(v.item() if k=='N_obj' else v.clone()) for k,v in native.items()}
        owner = host(mux.Sam3MultiplexBase,.3,True)
        for step in range(16):
            frame = 31-step if reverse else step; n = source['N_obj']
            masks = torch.tensor(rng.normal(size=(n,7,9)),device=device,dtype=DTYPES[mode])
            if n>1: masks[1] = masks[0]
            if n>2: masks[2] = -1
            match = torch.tensor(rng.random((7,n))>.7,device=device)
            nonempty = (masks>0).any(dim=(-1,-2)); unmatched = ~match.any(0)
            adt = types.SimpleNamespace(trk_is_unmatched=unmatched,trk_is_nonempty=nonempty,im_mask=match)
            with context(device,mode):
                remove,_,expected = mux.Sam3MultiplexBase._process_hotstart_gpu(options(config),frame,reverse,adt,{},source)
                new = {'gpu_metadata':expected}
                out = mux.Sam3MultiplexBase._suppress_overlapping_based_on_recent_occlusion(owner,frame,masks.clone(),{'obj_ids_all_gpu':np.arange(n)},new,remove,reverse)
            native = torch.ops.sam3_native.hotstart_device(native,unmatched,nonempty,match,frame,reverse,config)
            before = {k:v.clone() for k,v in native.items()}
            current = torch.ops.sam3_native.occlusion_hotstart(native,masks,native['remove'],frame,reverse,.3,True,mode)
            equal(current['masks'],out,'state_pipeline')
            for k in FIELDS: equal(current[k],expected[k],'state_pipeline')
            for k,v in before.items(): equal(native[k],v,'input_immutability')
            source = helpers['compact_gpu_metadata'](dict(expected),0,frame,0,device)
            native = torch.ops.sam3_native.hotstart_compact(current)
            source = helpers['extend_gpu_metadata_for_new_objects'](dict(source),step%3,frame,0,device)
            native = torch.ops.sam3_native.hotstart_extend(native,step%3,frame,0,device)
            for k in FIELDS: equal(native[k],source[k],'state_pipeline')


def check_geometry_cleanup(device, mode):
    g = torch.Generator().manual_seed(610)
    for n,h,w in [(0,3,5),(1,1,1),(7,7,9),(9,32,16),(201,3,5)]:
        masks = torch.randn(n,w,h,generator=g).transpose(1,2).to(device,dtype=DTYPES[mode])
        if n>1: masks[0]=-1; masks[1]=1
        if n>2: masks[2]=-1; masks[2,0,0]=1
        equal(torch.ops.sam3_native.tracking_mask_boxes(masks.unsqueeze(1)>0),mask_to_box(masks.unsqueeze(1)>0),'mask_boxes')
        if not n: continue  # upstream CC empty-batch handling is not required here
        for area in [0,1,4,16]:
          for holes,sprinkles in [(False,False),(True,False),(False,True),(True,True)]:
            # Source CUDA CCL requires contiguous storage; retain the strided
            # native input and compare the same values made contiguous for source.
            expected = fill_holes_in_mask_scores(masks.unsqueeze(1).contiguous(),fill_hole_area=area,sprinkle_removal_area=999,fill_holes=holes,remove_sprinkles=sprinkles)
            actual = torch.ops.sam3_native.clean_video_masks(masks.unsqueeze(1),area,holes,sprinkles)
            equal(actual,expected,'cleanup')
    for dtype in [torch.float16,torch.bfloat16,torch.float32,torch.float64]:
        a = torch.tensor([[0,0,0,0],[0,0,1,1],[1,2,0,0],[0,0,float('nan'),1]],device=device,dtype=dtype)
        b = a.flip(0)
        with context(device,mode): expected = fast_diag_box_iou(a,b)
        equal(torch.ops.sam3_native.diagonal_box_iou(a,b,mode),expected,'diagonal_iou')


def check_gates(device, mode):
    masks = torch.full((3,5,7),-1.,device=device,dtype=DTYPES[mode]); masks[0,1:4,1:5]=1; masks[1]=1
    ids = [30,10,20]
    boxes = torch.tensor([[1/7,1/5,4/7,3/5],[0,0,.1,.1],[0,0,0,0]],device=device,dtype=DTYPES[mode])
    scores = torch.tensor([.2,.9,.8],device=device,dtype=DTYPES[mode])
    for multiplex in [False,True]:
      for ci,cd in [([],[]),([30,10],[0,1]),([10,30],[1,0]),([20,10],[2,1]),([30],[1])]:
        for threshold in [0.,.5,1.1]:
          for period,frame in [(-1,0),(3,3),(3,4)]:
            owner = types.SimpleNamespace(reconstruction_bbox_iou_thresh=threshold,reconstruction_bbox_det_score=.8,recondition_every_nth_frame=period)
            candidates = dict(zip(ci,cd))
            scope = dict(torch=torch,mask_to_box=mask_to_box,fast_diag_box_iou=fast_diag_box_iou,
                self=owner,trk_id_to_max_iou_high_conf_det=candidates,det_out={'bbox':boxes,'scores':scores},
                tracker_metadata_prev={'obj_ids_all_gpu':np.array(ids)},tracker_low_res_masks_global=masks,
                frame_idx=frame,reconditioned_obj_ids=set(),det_mask_preds=None,
                adt_result=types.SimpleNamespace(trk_id_to_max_iou_high_conf_det=candidates),
                realize_adt_result=lambda value,*args:value)
            with context(device,mode): exec(GATES[multiplex],scope)
            actual = torch.ops.sam3_native.recondition_gate(ci,cd,ids,boxes,scores,masks,frame,multiplex,period,threshold,.8,mode)
            same(tuple(actual),(bool(scope['should_recondition_periodic']),bool(scope['should_recondition_iou']),sorted(scope['reconditioned_obj_ids'])),'gates')


class RecordingTracker:
    input_mask_size = 12
    def __init__(self): self.calls=[]; self.preflight=[]
    def add_new_mask(self,inference_state,frame_idx,obj_id,mask):
        self.calls.append((inference_state['index'],[obj_id],mask.unsqueeze(0).clone()))
    def add_new_masks(self,inference_state,frame_idx,obj_ids,masks,reconditioning):
        assert reconditioning
        self.calls.append((inference_state['index'],obj_ids,masks.clone()))
    def propagate_in_video_preflight(self,state,run_mem_encoder):
        assert run_mem_encoder
        self.preflight.append(state['index'])


def check_recipes(device, mode):
    g = torch.Generator().manual_seed(800)
    detections = torch.randn(5,7,9,generator=g).to(device,dtype=DTYPES[mode])
    masks = torch.randn(4,7,9,generator=g).to(device,dtype=DTYPES[mode])
    ids = [30,10,20,50]
    for multiplex,grouped in [(False,False),(True,False),(True,True)]:
      cls = mux.Sam3MultiplexBase if multiplex else base.Sam3VideoBase
      for ci,cd in [([],[]),([30,10,20,50],[2,0,4,1]),([50,20,10,30],[1,4,0,2])]:
        for score_values in [[.8,.8001,1,2],[5,5,5,5],[-5,-5,-5,-5],[1.38629436,1.387,.79,3]]:
          scores = torch.tensor(score_values,device=device,dtype=DTYPES[mode])
          for area in [0,3,16]:
            for state_ids in [[[10,50],[20,30]],[[50,10],[30,50]],[[30,10,20,50]]]:
              states = [dict(index=i,obj_ids=row,obj_idx_to_id=dict(enumerate(row))) for i,row in enumerate(state_ids)]
              recorder = RecordingTracker()
              owner = types.SimpleNamespace(tracker=recorder,is_multiplex=grouped,fill_hole_area=area,sprinkle_removal_area=999)
              current = masks.clone()
              args = [owner,5,{'mask':detections},dict(zip(ci,cd)),states,{'obj_ids_all_gpu':np.array(ids,dtype=np.int64)},scores]
              if multiplex: args.append(current)
              with context(device,mode): cls._recondition_masklets(*args)
              actual = torch.ops.sam3_native.recondition_prepare(ci,cd,ids,detections,scores,masks,12,12,multiplex,area,state_ids,grouped,mode)
              equal(actual['low_masks'],current,'recipes')
              same(actual['batch_states'].tolist(),[s for s,_,_ in recorder.calls],'recipes')
              same(actual['batch_ids'].tolist(),[id for _,row,_ in recorder.calls for id in row],'recipes')
              offsets=[0]
              for _,row,_ in recorder.calls: offsets.append(offsets[-1]+len(row))
              same(actual['batch_offsets'].tolist(),offsets,'recipes')
              expected = torch.cat([x for _,_,x in recorder.calls]) if recorder.calls else torch.empty((0,12,12),device=device,dtype=torch.bool)
              equal(actual['batch_masks'],expected,'recipes')


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--devices',nargs='+',default=['cpu','cuda']);p.add_argument('--modes',nargs='+',default=list(DTYPES));p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[]
    for device in a.devices:
      for mode in a.modes:
        CHECKS.clear()
        for fn in [check_occlusion,check_state_pipeline,check_geometry_cleanup,check_gates,check_recipes]:
            fn(device,mode);print(device,mode,fn.__name__,dict(CHECKS),flush=True)
        rows.append(dict(device=device,mode=mode,exact_comparisons=dict(CHECKS)))
    sources=['sam3/model/sam3_video_base.py','sam3/model/sam3_multiplex_base.py','sam3/model/sam3_tracker_utils.py','sam3/model/box_ops.py','sam3/perflib/masks_ops.py']
    report=dict(cases=rows,source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources},torch=torch.__version__,gpu=torch.cuda.get_device_name() if 'cuda' in a.devices else None,scope='Actual source occlusion and cleanup methods, AST-extracted reconditioning gates, actual masklet methods with a recording tracker (no neural edits), and hotstart->occlusion->compact->extend state transitions. FP16/BF16 modes reproduce source autocast arithmetic including count overflow; default runtime policy uses FP32 counts. Turing and Windows not executed.')
    a.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
