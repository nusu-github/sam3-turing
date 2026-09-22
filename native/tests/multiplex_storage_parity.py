"""Lossless paged vs resident sessions, including dynamic history reconstruction."""
import argparse,json,tempfile
from pathlib import Path
import torch

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--modes',nargs='+',default=['fp16','bf16_reference','fp32']);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(1957);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[];point=torch.tensor([[.4,.6,1.]]);many=torch.cat([torch.rand(17,2),torch.ones(17,1)],1);mask=torch.zeros(37,53);mask[5:31,9:44]=1;empty=torch.empty(0)
    def op(kind,index=0,obj=101,x=0,y=0,z=0,u=0,v=0,payload=empty):return [kind,index,obj,x,y,z,u,v],payload
    prefix=[op(0,x=1,y=1,payload=many),op(2,obj=202,payload=mask),op(3,x=1),op(4,index=0,obj=2,y=1,z=0)]
    # New point bucket, same-bucket brush insertion, repeated refinement,
    # reverse/cancel/resume, removal, clear, reset and batched masks.
    operations=prefix+[op(0,index=2,obj=303,x=1,y=1,payload=point),op(2,index=2,obj=404,payload=mask.roll(6,1)),
        op(0,index=1,x=1,y=1,payload=point),op(0,index=1,x=0,y=1,payload=many),op(4,index=3,obj=3,x=1,y=1,z=1,u=2,v=1),
        op(4,index=1,obj=1,x=1,y=1,z=0),op(6,obj=202,x=1),op(5,index=1),op(7),
        ([8,0,700,800],torch.stack([mask,mask.roll(8,1)])),op(4,index=0,obj=2,y=1,z=1),op(7)]
    for mode in a.modes:
        dtype={'fp16':torch.float16,'fp32':torch.float32,'bf16_reference':torch.bfloat16}[mode]
        arrays=[[torch.randn(shape,device=a.device,dtype=dtype).contiguous(memory_format=torch.channels_last) for _ in range(2) for shape in [(1,256,72,72),(1,256,72,72),(1,32,288,288),(1,64,144,144)]] for _ in range(3)]
        for score in [False,True]:
            print('START',mode,'score',score,flush=True)
            def run(directory):return torch.ops.sam3_native.multiplex_session(str(a.store),arrays,[x[0] for x in operations],[x[1] for x in operations],[5,37,53],[True,True,True,score,False],mode,directory)
            resident=run('')
            with tempfile.TemporaryDirectory(prefix='sam3-history-parity-') as temp:
                paged=run(temp);assert not list(Path(temp).iterdir()),'session archives leaked after destruction'
            count=0
            for key,value in resident.items():
                if key.endswith(('resident_history_bytes','archived_history_bytes')):continue
                torch.testing.assert_close(paged[key],value,rtol=0,atol=0);count+=1
            resident_peak=max(v.item() for k,v in resident.items() if k.endswith('resident_history_bytes'))
            disk_peak=max(v.item() for k,v in paged.items() if k.endswith('archived_history_bytes'))
            assert resident_peak>0 and disk_peak>0
            assert all(v.item()==0 for k,v in paged.items() if k.endswith('resident_history_bytes'))
            row=dict(device=a.device,mode=mode,score_selection=score,operations=len(operations),exact_tensor_comparisons=count,resident_history_peak_logical_bytes=resident_peak,paged_history_peak_disk_bytes=disk_peak,paged_resident_history_tensor_bytes=0)
            rows.append(row);print(json.dumps(row),flush=True)
    a.report.write_text(json.dumps(dict(cases=rows,scope='Paged vs resident native sessions with synthetic full-grid features. Bytes cover archived frame payloads, not model, selected-frame working sets, feature cache, input prompts, metadata or test snapshots. Archives preserve values, dtypes and strides; no upstream-policy or ground-truth quality claim.'),indent=2)+'\n')
if __name__=='__main__':main()
