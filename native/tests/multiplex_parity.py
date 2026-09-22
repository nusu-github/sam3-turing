"""Compare inference multiplex allocation, mutation and tensor transforms."""
import argparse,copy,json,random
from pathlib import Path
import torch
from sam3.model.multiplex_utils import MultiplexController,MultiplexState

def snapshot(out,prefix,state,probe):
    if state.assignments is None:
        out[prefix+'counts']=torch.tensor([0,0,state.multiplex_count,state.allowed_bucket_capacity,0,0,0])
        out[prefix+'assignments']=torch.empty(0,state.multiplex_count,dtype=torch.int64)
        if state.object_ids is not None:out[prefix+'ids']=torch.tensor(state.object_ids,dtype=torch.int64)
        return
    out[prefix+'counts']=torch.tensor([1,state.num_buckets,state.multiplex_count,state.allowed_bucket_capacity,state.total_valid_entries,state.total_non_padding_entries,state.available_slots])
    out[prefix+'assignments']=torch.tensor(state.assignments)
    if state.object_ids is not None:out[prefix+'ids']=torch.tensor(state.object_ids,dtype=torch.int64)
    out[prefix+'mux_matrix']=state.mux_matrix.clone();out[prefix+'demux_matrix']=state.demux_matrix.clone();out[prefix+'valid_mask']=state.get_valid_object_mask()
    if state.total_valid_entries:
        muxed=state.mux(probe[:state.total_valid_entries]);out[prefix+'mux']=muxed;out[prefix+'demux']=state.demux(muxed)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--device',default='cuda')
    p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(733)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results=[]
    def compare(actual,expected,**metadata):
        assert actual.keys()==expected.keys(),(actual.keys(),expected.keys())
        for key,value in actual.items():
            torch.testing.assert_close(value,expected[key],rtol=0,atol=0,equal_nan=True)
            if key.endswith('matrix'):assert value.stride()==expected[key].stride()
        results.append(dict(**metadata,tensors=len(expected),exact=True));print(json.dumps(results[-1]),flush=True)
    for mode in a.modes:
        dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
        # Float32 input with low-precision matrices exercises original autocast
        # rounding, including noncontiguous tensors and zero-length dimensions.
        probe=torch.randn(256,5,3,device=a.device).transpose(1,2)
        for width in [1,4,16]:
            policies=[(width,False,False),(width,False,True),(width,True,False),(width,True,True),(max(1,width//2),False,True)]
            for index,(capacity,full,shuffle) in enumerate(policies):
                for objects in sorted(set([1,width+1,37])):
                    ids=[1000+7*i for i in range(objects)] if index%2 else None;seed=100+width+objects+index
                    controller=MultiplexController(width,full_shuffle=full,eval_multiplex_count=capacity).eval()
                    torch.manual_seed(seed);state=controller.get_state(objects,torch.device(a.device),dtype,random=shuffle,object_ids=copy.deepcopy(ids))
                    with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                        expected={};snapshot(expected,'',state,probe)
                    torch.manual_seed(seed)
                    actual=torch.ops.sam3_native.multiplex_controller(probe,objects,width,capacity,full,shuffle,ids,mode)
                    compare(actual,expected,mode=mode,case='controller',width=width,capacity=capacity,objects=objects,full_shuffle=full,random=shuffle,external_ids=ids is not None)
        for width,capacity,objects,external in [(4,4,3,False),(16,16,37,True),(16,8,19,True),(4,2,5,False)]:
            controller=MultiplexController(width,eval_multiplex_count=capacity).eval();torch.manual_seed(811+objects)
            ids=list(range(2000,2000+objects)) if external else None
            state=controller.get_state(objects,torch.device(a.device),dtype,random=True,object_ids=copy.deepcopy(ids))
            assignments=copy.deepcopy(state.assignments);operations=[];indices=[];added_ids=[];allow=[];prefer=[];strict=[];expected={}
            rng=random.Random(811+objects);next_id=9000
            with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                snapshot(expected,'0.',state,probe)
                for step in range(14):
                    n=state.total_valid_entries;new_ids=[];allow_new=True;prefer_new=False;is_strict=True
                    if step==13:op='remove';items=list(range(n))
                    elif step%4==0:
                        op='next';items=[3];prefer_new=True
                    elif step%4 in [1,3]:
                        op='add';count=capacity+1 if step==1 else 3;items=list(range(n,n+count));prefer_new=step%4==3
                        new_ids=list(range(next_id,next_id+count));next_id+=count
                    else:
                        op='remove';items=rng.sample(range(n),max(1,n//3))
                        if step==6:items += [987654];is_strict=False
                    prefix=str(step+1)+'.'
                    if op=='next':expected[prefix+'next']=torch.tensor(state.find_next_batch_of_available_indices(items[0],allow_new_buckets=allow_new,prefer_new_buckets=prefer_new))
                    elif op=='add':state.add_objects(items,object_ids=copy.deepcopy(new_ids) if external else None,allow_new_buckets=allow_new,prefer_new_buckets=prefer_new)
                    else:expected[prefix+'kept']=torch.tensor(state.remove_objects(items,strict=is_strict),dtype=torch.int64)
                    operations.append(op);indices.append(items);added_ids.append(new_ids);allow.append(allow_new);prefer.append(prefer_new);strict.append(is_strict)
                    snapshot(expected,prefix,state,probe)
            actual=torch.ops.sam3_native.multiplex_state(probe,assignments,capacity,ids,operations,indices,added_ids,allow,prefer,strict,mode)
            compare(actual,expected,mode=mode,case='state_mutations',width=width,capacity=capacity,initial_objects=objects,steps=len(operations),external_ids=external)
        for shape in [(4,),(4,0,3),(4,2,3)]:
            value=torch.randn(shape,device=a.device);state=MultiplexState([[2,-1,0,-1],[1,-1,3,-1]],torch.device(a.device),dtype,4)
            with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                expected={};snapshot(expected,'0.',state,value)
            actual=torch.ops.sam3_native.multiplex_state(value,state.assignments,4,None,[],[],[],[],[],[],mode)
            compare(actual,expected,mode=mode,case='tensor_shape',shape=list(shape))
    a.report.write_text(json.dumps(dict(torch=torch.__version__,device=a.device,cases=results,
        scope='Original inference multiplex controller/state, valid allocation/removal paths and mux/demux. All-removed state is normalized to invalid with zero active counts; source stale metadata is not a supported active state. Training not tested.'),indent=2)+'\n')
if __name__=='__main__':main()
