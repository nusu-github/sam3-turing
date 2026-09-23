"""Dense singleton extraction retains history across resident/offloaded/paged state."""
import argparse,json,tempfile
from pathlib import Path
import torch

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--mode',default='bf16_reference');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(1);torch.manual_seed(9329);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dtype={'fp16':torch.float16,'bf16_reference':torch.bfloat16,'fp32':torch.float32}[a.mode]
    features=[[torch.randn(shape,device=a.device,dtype=dtype) for _ in range(2) for shape in [(1,256,72,72),(1,256,72,72),(1,32,288,288),(1,64,144,144)]]]
    masks=torch.zeros(2,37,53);masks[0,4:20,5:30]=1;masks[1,21:33,29:49]=1;empty=torch.empty(0)
    operations=[[8,0,101,202],[3,0,0,1,0,0,0,0],[4,0,2,0,1,0,0,0],[12,0,202,0,0,0,0,0]]
    rows=[];reference=None
    with tempfile.TemporaryDirectory() as directory:
        for kind,offload,paging in [('resident',False,''),('offloaded',True,''),('paged',True,directory)]:
            result=torch.ops.sam3_native.multiplex_session(str(a.store),features,operations,[masks,empty,empty,empty],[4,37,53],[offload,True,False,False,False],a.mode,paging)
            assert result['3/ids'].tolist()==[101] and result['3/extracted/ids'].tolist()==[202]
            assert result['3/extracted/tracked_count'].item()==0 and result['3/extracted/annotations_after_discard'].item()==0
            compared=0
            for group,frame in [('cond',0),('tracked',1),('tracked',2)]:
                before=f'2/{group}/{frame}/';source=f'3/{group}/{frame}/';fork=f'3/extracted/{group}/{frame}/'
                assert result[fork+'memory'].shape==(1,256,72,72) and result[fork+'position'].shape==(1,256,72,72)
                assert torch.isfinite(result[fork+'memory']).all()
                torch.testing.assert_close(result[fork+'low'],result[before+'low'][1:2],rtol=0,atol=0)
                torch.testing.assert_close(result[source+'memory'],result[before+'memory'],rtol=0,atol=0)
                torch.testing.assert_close(result[fork+'pointer'][:,0],result[before+'pointer'][:,1],rtol=0,atol=0)
                assert torch.count_nonzero(result[fork+'pointer'][:,1:])==0
                compared+=3
            if reference is not None:
                for key,value in result.items():
                    if key.startswith('3/') and not key.endswith('_history_bytes'):
                        torch.testing.assert_close(value,reference[key],rtol=0,atol=0);compared+=1
            else:reference=result
            rows.append(dict(storage=kind,exact_tensor_comparisons=compared,retained_spatial_frames=3))
    a.report.write_text(json.dumps(dict(device=a.device,mode=a.mode,cases=rows,scope='Full-grid synthetic features, actual memory encoder and frame core; extraction from second slot into fresh singleton. Tests preserved object masks/pointers, retained dense memory, unchanged source bucket, annotation discard, restart flags and exact resident/offloaded/paged results. Not upstream accuracy.'),indent=2)+'\n');print(rows)
if __name__=='__main__':main()
