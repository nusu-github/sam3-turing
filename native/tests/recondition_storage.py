"""Reconditioned neural sessions: resident, host-offloaded and paged equality."""
import argparse,json,tempfile
from pathlib import Path
import torch

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--modes',nargs='+',default=['fp16','fp32','bf16_reference']);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(221)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[]
    for mode in a.modes:
        dtype={'fp16':torch.float16,'fp32':torch.float32,'bf16_reference':torch.bfloat16}[mode]
        arrays=[[torch.randn(shape,device=a.device,dtype=dtype) for _ in range(2) for shape in [(1,256,72,72),(1,256,72,72),(1,32,288,288),(1,64,144,144)]] for frame in range(3)]
        mask=torch.zeros(37,53);mask[3:30,7:40]=1;empty=torch.empty(0)
        operations=[[8,0,17,91,300],[3,0,0,1,0,0,0,0],[4,0,1,0,1,0,0,0],[10,1,300,17],[4,2,0,0,1,0,0,0],[9,2,91],[3,0,0,1,0,0,0,0],[4,2,2,1,1,0,0,0]]
        payloads=[torch.stack([mask,mask.roll(7,0),mask.roll(8,1)]),empty,empty,torch.stack([mask.roll(4,0),mask.roll(7,1)]),empty,mask.unsqueeze(0),empty,empty]
        def run(offload,directory=''):
            return torch.ops.sam3_native.multiplex_session(str(a.store),arrays,operations,payloads,[3,37,53],[offload,False,True,True,False],mode,directory)
        expected=run(False);checks=0
        for offload,paged in [(True,False),(True,True)]:
            with tempfile.TemporaryDirectory(prefix='sam3-recondition-') as directory:
                actual=run(offload,directory if paged else '')
                assert actual.keys()==expected.keys()
                for k,v in expected.items():
                    if k.endswith(('_history_bytes')):continue
                    torch.testing.assert_close(actual[k],v,rtol=0,atol=0);checks+=1
                if paged:
                    assert actual['7/archived_history_bytes'].item()>0
                    assert not list(Path(directory).iterdir()),'destroyed session retained archive files'
        for kind in ['memory','pointer','low']:
            torch.testing.assert_close(expected[f'2/cond/0/{kind}'],expected[f'3/cond/0/{kind}'],rtol=0,atol=0);checks+=1
        rows.append(dict(mode=mode,exact_tensor_comparisons=checks,operations=len(operations),scope='Same neural workflows with resident/offloaded/paged state; old conditioning frame unchanged and temporary archives cleaned after session destruction.'))
        print(rows[-1],flush=True)
    a.report.write_text(json.dumps(dict(device=a.device,torch=torch.__version__,cases=rows),indent=2)+'\n')
if __name__=='__main__':main()
