"""Cancellation/resume equivalence and dynamic-session data conservation."""
import argparse,json
from pathlib import Path
import torch

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--mode',default='fp16');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(6803);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dtype={'fp16':torch.float16,'fp32':torch.float32,'bf16_reference':torch.bfloat16}[a.mode];arrays=[]
    for _ in range(3):arrays.append([torch.randn(shape,device=a.device,dtype=dtype) for _ in range(2) for shape in [(1,256,72,72),(1,256,72,72),(1,32,288,288),(1,64,144,144)]])
    point=torch.tensor([[.4,.6,1.]]);many=torch.cat([torch.rand(17,2),torch.ones(17,1)],1);mask=torch.zeros(37,53);mask[5:31,9:44]=1;empty=torch.empty(0)
    def op(kind,index=0,obj=101,x=0,y=0,z=0,u=0,v=0,payload=empty):return [kind,index,obj,x,y,z,u,v],payload
    def run(operations,all_conditioning=True):return torch.ops.sam3_native.multiplex_session(str(a.store),arrays,[x[0] for x in operations],[x[1] for x in operations],[5,37,53],[True,True,all_conditioning,False,False],a.mode)
    prefix=[op(0,x=1,y=1,payload=many),op(2,obj=202,payload=mask),op(3,x=1)]
    full=run(prefix+[op(4,index=0,obj=3,y=1,z=0)])
    resumed=run(prefix+[op(4,index=0,obj=3,y=1,z=0,u=2,v=1),op(4,index=2,obj=1,y=1,z=0)])
    compared=0
    for frame in range(4):
        target=f'3/out{frame}/';actual=f'{3 if frame<2 else 4}/out{frame if frame<2 else frame-2}/'
        for key in ['frame','ids','masks','low','logits']:torch.testing.assert_close(full[target+key],resumed[actual+key],rtol=0,atol=0);compared+=1
    for key,value in full.items():
        if key.startswith(('3/cond/','3/tracked/')):torch.testing.assert_close(value,resumed['4/'+key[2:]],rtol=0,atol=0);compared+=1
    dynamic=prefix+[op(4,index=0,obj=2,y=1,z=0),op(0,index=2,obj=303,x=1,y=1,payload=point),op(0,index=1,x=1,y=1,payload=point),op(0,index=1,x=0,y=1,payload=many),op(4,index=3,obj=3,x=1,y=1,z=1),op(6,obj=202,x=1),op(5,index=1),op(7)]
    changed=run(dynamic)
    assert changed['4/ids'].tolist()==[101,202,303] and changed['8/ids'].tolist()==[101,303] and changed['10/ids'].numel()==0
    assert changed['6/points/101/1'].shape[1]==18
    torch.testing.assert_close(changed['4/cond/0/memory'][0],changed['3/cond/0/memory'][0],rtol=0,atol=0)
    torch.testing.assert_close(changed['4/cond/0/pointer'][0],changed['3/cond/0/pointer'][0],rtol=0,atol=0)
    assert (changed['4/cond/0/low'][2]==-1024).all()
    # Remaining non-conditioning edits must survive clearing the last initial prompt.
    keep=run([op(0,x=1,y=1,payload=point),op(4,index=0,obj=1,y=1,z=1),op(0,index=1,x=1,y=1,payload=point),op(5,index=0),op(4,index=1,obj=0,y=1,z=1)],False)
    assert keep['4/outputs'].item()==1 and keep['4/ids'].tolist()==[101] and '4/cond/1/memory' in keep
    report=dict(device=a.device,mode=a.mode,cancellation_exact_tensor_comparisons=compared,dynamic_operations=len(dynamic),retained_points=18,checks=['cancel/resume equals uninterrupted outputs and history','midstream ID growth and removal','unchanged old bucket memory/pointers','historical absence for new object','reverse propagation','point accumulation','reset','remaining annotation promoted after last conditioning input cleared'],scope='Native policy invariants with full-grid synthetic features; not upstream multi-object demo parity.')
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
