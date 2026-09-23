"""Repeated point edits must update conditioning memory, even with detector edits non-cond."""
import argparse,json
from pathlib import Path
import torch

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('--mode',default='bf16_reference');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(1);torch.manual_seed(836);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dtype={'fp16':torch.float16,'bf16_reference':torch.bfloat16,'fp32':torch.float32}[a.mode]
    features=[[torch.randn(shape,device='cuda',dtype=dtype) for _ in range(2) for shape in [(1,256,72,72),(1,256,72,72),(1,32,288,288),(1,64,144,144)]]]
    mask=torch.zeros(37,53);mask[5:31,8:42]=1;points=torch.tensor([[.4,.5,1.],[.8,.8,0.]]);extra=torch.tensor([[.2,.3,1.]])
    ops=[[2,0,101,0,0,0,0,0],[3,0,0,1,0,0,0,0],[4,1,1,0,1,0,0,0],[0,1,101,1,1,0,0,0],[3,0,0,1,0,0,0,0],[0,1,101,0,1,0,0,0],[3,0,0,1,0,0,0,0]]
    result=torch.ops.sam3_native.multiplex_session(str(a.store),features,ops,[mask,torch.empty(0),torch.empty(0),points,torch.empty(0),extra,torch.empty(0)],[4,37,53],[True,True,False,False,False],a.mode)
    assert '2/tracked/1/memory' in result
    assert '4/cond/1/memory' in result and '4/tracked/1/memory' not in result
    assert '6/cond/1/memory' in result and '6/tracked/1/memory' not in result
    assert result['6/points/101/1'].shape==(1,3,2)
    assert not torch.equal(result['4/cond/1/low'],result['6/cond/1/low'])
    assert not torch.equal(result['4/cond/1/memory'],result['6/cond/1/memory'])
    torch.testing.assert_close(result['4/cond/0/memory'],result['6/cond/0/memory'],rtol=0,atol=0)
    a.report.write_text(json.dumps(dict(device='cuda',mode=a.mode,operations=len(ops),checks=['tracked frame becomes point conditioning with all_edits_conditioning false','appended click retained','latest low mask changes','encoded memory changes with latest point','prior frame memory unchanged'],scope='Full-grid synthetic features with actual native frame and memory cores. No upstream neural parity claim.'),indent=2)+'\n')
    print('point conditioning and latest-memory invariants passed')
if __name__=='__main__':main()
