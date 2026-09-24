"""Compare native host/device state policies with actual upstream method bodies."""
import argparse,ast,copy,hashlib,json,types
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import sam3.model.sam3_video_base as base
import sam3.model.sam3_multiplex_base as mux

FIELDS=['obj_first_frame','consecutive_unmatch_count','trk_keep_alive','removed_mask','overlap_pair_counts','last_occluded_tensor']
CONFIGS=[[0,3,3,0,-4,8,1,0],[0,3,3,2,-4,8,0,1],[5,2,3,0,-4,8,1,0],[5,2,3,0,-4,8,0,1],[5,0,0,0,-2,2,0,0],[5,5,5,-1,-3,4,0,1]]
def options(x):
    return types.SimpleNamespace(**dict(zip(['hotstart_delay','hotstart_unmatch_thresh','hotstart_dup_thresh','init_trk_keep_alive','min_trk_keep_alive','max_trk_keep_alive','suppress_unmatched_only_within_hotstart','decrease_trk_keep_alive_for_empty_masklets'],x)))
def host_empty():
    return dict(obj_first_frame_idx={},trk_keep_alive={},unmatched_frame_inds=defaultdict(list),overlap_pair_to_frame_inds=defaultdict(list),removed_obj_ids=set(),suppressed_obj_ids=defaultdict(set))
def host_unpack(out):
    value={}
    for key,name in [('first_frame','obj_first_frame_idx'),('keep_alive','trk_keep_alive')]:value[name]=dict(out[key].tolist())
    for prefix,name in [('unmatched','unmatched_frame_inds'),('overlap','overlap_pair_to_frame_inds'),('suppressed','suppressed_obj_ids')]:
        keys=out[prefix+'_keys'].tolist();offsets=out[prefix+'_offsets'].tolist();items=out[prefix+'_values'].tolist();result={}
        for i,key in enumerate(keys):
            key=tuple(key) if prefix=='overlap' else key[0];result[key]=items[offsets[i]:offsets[i+1]]
            if prefix=='suppressed':result[key]=set(result[key])
        value[name]=result
    value['removed_obj_ids']=set(out['removed'].tolist());return value

def source_state_helpers():
    # Extract the original inline planning blocks, rather than duplicating their
    # algorithms in the test. The surrounding model/communication need not run.
    path=Path('sam3/model/sam3_multiplex_base.py');tree=ast.parse(path.read_text());blocks={}
    for node in ast.walk(tree):
        if isinstance(node,ast.With):
            for item in node.items:
                call=item.context_expr
                if isinstance(call,ast.Call) and call.args and isinstance(call.args[0],ast.Constant) and call.args[0].value in ['compact_gpu_metadata','extend_gpu_metadata_for_new_objects']:blocks[call.args[0].value]=node
    prefix="tracker_metadata_new={'gpu_metadata':state}\ndet_scores=torch.empty(0,device=device)\nself=types.SimpleNamespace(init_trk_keep_alive=initial)\n"
    functions={}
    for name,node in blocks.items():
        code=ast.parse('def run(state,num_new,frame_idx,initial,device):\n    pass\n').body[0]
        code.body=ast.parse(prefix).body+[copy.deepcopy(node)]+ast.parse("return tracker_metadata_new['gpu_metadata']").body
        module=ast.fix_missing_locations(ast.Module(body=[code],type_ignores=[]));scope={'torch':torch,'types':types};exec(compile(module,str(path),'exec'),scope);functions[name]=scope['run']
    assert len(functions)==2;return functions,hashlib.sha256(path.read_bytes()).hexdigest()
def compare_state(actual,expected):
    assert actual['N_obj'].item()==expected['N_obj'];count=1
    for name in FIELDS:torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0);count+=1
    return count

def host_workflows():
    rows=[]
    for cls in [base.Sam3VideoBase,mux.Sam3MultiplexBase]:
      for ci,config in enumerate(CONFIGS):
        for reverse in [False,True]:
          rng=np.random.default_rng(573+ci);source=host_empty();native={};next_id=10000000000;checks=0
          for step in range(48):
            frame=47-step if reverse else step
            new=list(range(next_id,next_id+(5 if step%6==0 else 0)));next_id+=len(new)
            ids=[x for x in source['obj_first_frame_idx'] if x not in source['removed_obj_ids']]+new
            matches={d:np.array(rng.choice(ids,size=min(len(ids),d+1),replace=False),np.int64) for d in range(3)}
            unmatched=np.array([x for x in ids if rng.random()<.3],np.int64);empty=np.array([x for x in ids if rng.random()<.2],np.int64)
            before={k:v.clone() for k,v in native.items()}
            out=torch.ops.sam3_native.hotstart_host(native,list(matches),[x.tolist() for x in matches.values()],unmatched.tolist(),empty.tolist(),new,frame,reverse,config)
            removed,source=cls._process_hotstart(options(config),frame,48,reverse,matches,np.array(new,np.int64),empty,unmatched,source,{})
            unpacked=host_unpack(out)
            for key,value in source.items():assert unpacked[key]==value,(cls.__name__,ci,reverse,step,key);checks+=1
            assert set(out['newly_removed'].tolist())==removed;checks+=1
            for key,value in before.items():torch.testing.assert_close(native[key],value,rtol=0,atol=0)
            native=out
          rows.append(dict(source=cls.__name__,config=ci,reverse=reverse,steps=48,state_field_comparisons=checks))
    print('host workflows',len(rows),flush=True);return rows

def confirmation_workflows():
    count=0;rng=np.random.default_rng(342)
    for cls in [base.Sam3VideoBase,mux.Sam3MultiplexBase]:
      for threshold in [0,1,3,7]:
        metadata={'masklet_confirmation':{'status':np.empty(0,np.int64),'consecutive_det_num':np.empty(0,np.int64)}};ids=np.empty(0,np.int64);status=[];consecutive=[];next_id=-20000000000
        for step in range(60):
          new=np.arange(next_id,next_id+(3 if step%3==0 else 0),dtype=np.int64);next_id+=len(new)
          updated=np.concatenate([ids[rng.random(len(ids))>.15],new]);rng.shuffle(updated)
          matched={0:updated[rng.random(len(updated))>.3],1:updated[rng.random(len(updated))>.7]}
          actual=torch.ops.sam3_native.confirmation(status,consecutive,ids.tolist(),updated.tolist(),list(matched),[x.tolist() for x in matched.values()],new.tolist(),threshold)
          metadata=cls.update_masklet_confirmation_status(types.SimpleNamespace(masklet_confirmation_consecutive_det_thresh=threshold),metadata,ids,updated,matched,new)
          for got,key in zip(actual,['status','consecutive_det_num']):np.testing.assert_array_equal(got.numpy(),metadata['masklet_confirmation'][key]);count+=1
          status,consecutive=(x.tolist() for x in actual);ids=updated
    print('confirmation comparisons',count,flush=True);return count

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--devices',nargs='+',default=['cpu','cuda']);p.add_argument('--report',type=Path,required=True);p.add_argument('--large',action='store_true');a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    host=host_workflows();confirmation=confirmation_workflows();helpers,source_hash=source_state_helpers();rows=[]
    for device in a.devices:
      for mode in ['fp32','fp16','bf16_reference']:
        dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode];checks=0;steps=0
        for ci,config in enumerate(CONFIGS):
          for reverse in [False,True]:
            rng=np.random.default_rng(800+ci);source={'N_obj':0};native={}
            for step in range(20):
              frame=19-step if reverse else step;n=source['N_obj'];d=[0,1,3,17,200][step%5]
              # Transposed match views ensure non-contiguous inputs are covered.
              match=torch.tensor(rng.random((n,d))>.6,device=device).t();unmatched=torch.tensor(rng.random(n)>.65,device=device);nonempty=torch.tensor(rng.random(n)>.3,device=device)
              adt=types.SimpleNamespace(trk_is_unmatched=unmatched,trk_is_nonempty=nonempty,im_mask=match)
              before={key:value.clone() for key,value in native.items()}
              with torch.autocast(device_type=device,dtype=dtype,enabled=mode!='fp32'):
                remove,suppress,expected=mux.Sam3MultiplexBase._process_hotstart_gpu(options(config),frame,reverse,adt,{},source)
                actual=torch.ops.sam3_native.hotstart_device(native,unmatched,nonempty,match,frame,reverse,config)
              checks+=compare_state(actual,expected)
              for key,value in [('remove',remove),('suppress',suppress)]:torch.testing.assert_close(actual[key],value,rtol=0,atol=0);checks+=1
              for key,value in before.items():torch.testing.assert_close(native[key],value,rtol=0,atol=0)
              indices=torch.nonzero(~expected['removed_mask']).flatten();source=helpers['compact_gpu_metadata'](dict(expected),0,frame,config[3],device);native=torch.ops.sam3_native.hotstart_compact(actual);checks+=compare_state(native,source);torch.testing.assert_close(native['indices'],indices,rtol=0,atol=0);checks+=1
              new=[0,1,3,5][step%4];source=helpers['extend_gpu_metadata_for_new_objects'](dict(source),new,frame,config[3],device);native=torch.ops.sam3_native.hotstart_extend(native,new,frame,config[3],device);checks+=compare_state(native,source)
              if step%7==6:
                order=torch.tensor(rng.permutation(source['N_obj']).copy(),device=device)
                source={**source,**{key:value[order] for key,value in source.items() if key in FIELDS and key!='overlap_pair_counts'},'overlap_pair_counts':source['overlap_pair_counts'][order][:,order]}
                native=torch.ops.sam3_native.hotstart_select(native,order);checks+=compare_state(native,source)
              steps+=1
        row=dict(device=device,mode=mode,steps=steps,exact_tensor_and_scalar_comparisons=checks);rows.append(row);print(json.dumps(row),flush=True)
    large=[]
    if a.large:
      for device,n in [(device,n) for device in a.devices for n in [16777216,16777217]]:
        # Check both sides of the exact-FP32-integer boundary without a cap.
        match=torch.ones(n,2,dtype=torch.bool,device=device);config=CONFIGS[0]
        native=torch.ops.sam3_native.hotstart_extend({},2,0,0,device);source={key:(value.item() if key=='N_obj' else value) for key,value in native.items()}
        unmatched=torch.zeros(2,dtype=torch.bool,device=device);nonempty=~unmatched;adt=types.SimpleNamespace(trk_is_unmatched=unmatched,trk_is_nonempty=nonempty,im_mask=match)
        remove,suppress,expected=mux.Sam3MultiplexBase._process_hotstart_gpu(options(config),1,False,adt,{},source)
        actual=torch.ops.sam3_native.hotstart_device(native,unmatched,nonempty,match,1,False,config);checks=compare_state(actual,expected)
        torch.testing.assert_close(actual['remove'],remove,rtol=0,atol=0);torch.testing.assert_close(actual['suppress'],suppress,rtol=0,atol=0);large.append(dict(device=device,detections=n,objects=2,comparisons=checks+2));print('large reduction passed',device,n,flush=True)
        del match,adt,actual,native,source,expected
    report=dict(host=host,confirmation_comparisons=confirmation,device=rows,large=large,source_sha256=source_hash,torch=torch.__version__,gpu=torch.cuda.get_device_name() if 'cuda' in a.devices else None,scope='Actual CPU hotstart and confirmation methods from both source classes, actual SAM3.1 GPU hotstart, and AST-extracted original compaction/extension blocks. Device policy and host policy are intentionally distinct. Per-frame metadata/decisions and previous-state immutability checked. No full high-level neural video integration claimed.')
    a.report.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
