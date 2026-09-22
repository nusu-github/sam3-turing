"""Retained-image history rebuilding against the original memory encoder host."""
import argparse,json
from pathlib import Path
import torch
from sam3.model.multiplex_utils import MultiplexState
from memory_encoder_parity import reference

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(5719);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weights=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=True);_,host=reference(weights.get('model',weights),'sam3.1',a.device);rows=[]
    for mode in (['fp16','fp32','bf16_reference'] if a.device=='cuda' else ['fp32']):
        dtype={'fp16':torch.float16,'fp32':torch.float32,'bf16_reference':torch.bfloat16}[mode]
        for name,slots,count,conditions,overlap in [('grow',[list(range(16)),[16,17]+[-1]*14],18,[0,2],False),('extract',[[0]+[-1]*15],1,[0],False),('reassign',[[2,0,-1,1]+[-1]*12],3,[0,2],True)]:
            image=torch.randn(1,256,72,72,device=a.device,dtype=dtype).contiguous(memory_format=torch.channels_last).flatten(2).permute(2,0,1).cpu()
            masks=torch.randn(count,1,1008,1008);scores=torch.randn(count,1)
            if count>3:masks[3:]=-1024;scores[3:]=-1024
            host.non_overlap_masks_for_mem_enc=overlap;state=MultiplexState(slots,torch.device(a.device),torch.float32,16)
            with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                expected=host._encode_new_memory(None,[image.to(a.device)],[(72,72)],masks.to(a.device),scores.to(a.device),False,conditioning_objects=conditions,multiplex_state=state)
            actual=torch.ops.sam3_native.multiplex_history_encode(str(a.store),image,masks,scores,slots,conditions,overlap,a.device,mode)
            for got,wanted in zip(actual,[expected[0],expected[1][0]]):torch.testing.assert_close(got,wanted,rtol=0,atol=0)
            rows.append(dict(mode=mode,case=name,objects=count,buckets=len(slots),outputs=2,exact=True));print(json.dumps(rows[-1]),flush=True)
    a.report.write_text(json.dumps(dict(device=a.device,torch=torch.__version__,cases=rows,scope='CPU-stored shared image sequence and full-resolution masks rebuilt through original SAM3.1 memory host, synthetic features.'),indent=2)+'\n')
if __name__=='__main__':main()
