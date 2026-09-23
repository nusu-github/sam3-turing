"""Global memory replacement survives storage and layout changes losslessly."""
import argparse,json,tempfile
from pathlib import Path
import torch
from sam3.model.multiplex_utils import MultiplexController
from memory_encoder_parity import reference

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--modes',nargs='+',default=['fp16','fp32','bf16_reference']);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(971)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weights=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=True);_,host=reference(weights.get('model',weights),'sam3.1',a.device)
    rows=[]
    for mode in a.modes:
        dtype={'fp16':torch.float16,'fp32':torch.float32,'bf16_reference':torch.bfloat16}[mode]
        arrays=[[torch.randn(shape,device=a.device,dtype=dtype) for _ in range(2) for shape in [(1,256,72,72),(1,256,72,72),(1,32,288,288),(1,64,144,144)]] for _ in range(3)]
        mask=torch.zeros(37,53);mask[3:30,7:40]=1;empty=torch.empty(0)
        operations=[[8,0,101,202,303],[3,0,0,1,0,0,0,0],[11,0,0,1,0,0,0,0],[2,1,404,0,0,0,0,0],[3,0,0,1,0,0,0,0],[4,2,0,0,1,0,0,0],[11,2,0,1,0,0,0,0],[9,2,202],[3,0,0,1,0,0,0,0],[4,2,2,1,1,0,0,0]]
        payloads=[torch.stack([mask,mask.roll(7,0),mask.roll(8,1)]),empty,torch.ones(3,5,7),mask,empty,empty,-torch.ones(4,7,9),mask.unsqueeze(0),empty,empty]
        def run(offload,directory=''):return torch.ops.sam3_native.multiplex_session(str(a.store),arrays,operations,payloads,[3,37,53],[offload,False,True,False,False],mode,directory)
        expected=run(False);checks=0
        for offload,paged in [(True,False),(True,True)]:
            with tempfile.TemporaryDirectory(prefix='sam3-memory-') as directory:
                actual=run(offload,directory if paged else '')
                assert actual.keys()==expected.keys()
                for k,v in expected.items():
                    if k.endswith('_history_bytes'):continue
                    torch.testing.assert_close(actual[k],v,rtol=0,atol=0);checks+=1
                if paged:assert actual['9/archived_history_bytes'].item()>0 and not list(Path(directory).iterdir())
        # Global memory rewrite must not replace predicted masks/scores.
        for key in ['low','logits']:
            torch.testing.assert_close(expected['1/cond/0/'+key],expected['2/cond/0/'+key],rtol=0,atol=0);checks+=1
        high=expected['3/cond/0/memory_masks'].to(a.device);scores=expected['3/cond/0/memory_scores'].to(a.device)
        assert (high[-1]==-1024).all() and scores[-1].item()==-1024
        assert '8/cond/2/memory_masks' not in expected and '8/cond/2/memory_scores' not in expected,'preflight retained obsolete memory override'
        # Rebuild the changed bucket with the actual source neural encoder,
        # using the retained effective masks/proxies, not predicted masks.
        state=MultiplexController(16).eval().get_state(4,torch.device(a.device),torch.float32,random=False,object_ids=[101,202,303,404])
        with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype):
            memory,position=host._encode_new_memory(None,[arrays[0][4].flatten(2).permute(2,0,1)],[(72,72)],high,scores,False,conditioning_objects={0,1,2},multiplex_state=state)
        torch.testing.assert_close(expected['3/cond/0/memory'],memory.to(torch.bfloat16).cpu(),rtol=0,atol=0);checks+=1
        torch.testing.assert_close(expected['3/cond/0/position'],position[-1].cpu(),rtol=0,atol=0);checks+=1
        rows.append(dict(mode=mode,exact_tensor_comparisons=checks,operations=len(operations)));print(rows[-1],flush=True)
    a.report.write_text(json.dumps(dict(cases=rows,device=a.device,scope='Native global memory write -> object insertion/history rebuild -> forward/reverse propagation -> correction/preflight, resident/offloaded/paged equivalence, plus actual source encoder comparison for rebuilt bucket. Predicted low masks/logits unchanged by memory rewrite; obsolete overrides cleared on preflight.'),indent=2)+'\n')
if __name__=='__main__':main()
