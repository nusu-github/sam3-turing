"""C ABI batch positions versus the original-validated C++ module composition."""
import argparse,gc,json
from pathlib import Path
import numpy as np
import torch
from c_api_parity import read

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('fixtures',type=Path);p.add_argument('--models',nargs='+',default=['sam3','sam3.1']);p.add_argument('--modes',nargs='+',default=['fp16','bf16_reference','fp32']);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    h,w=map(int,(a.fixtures/'inputs/image/size.txt').read_text().split());rgb=torch.from_numpy(np.fromfile(a.fixtures/'inputs/image/0.rgb',dtype=np.uint8).reshape(h,w,3).copy()).permute(2,0,1).contiguous().cuda()
    normalized=torch.ops.sam3_native.preprocess_rgb(rgb).squeeze(0);inputs=torch.stack([normalized,normalized]);rows=[]
    boxes=torch.tensor([[.1,.1,.9,.9],[.2,.2,.7,.8]],device='cuda')
    for model in a.models:
        for mode in a.modes:
            print('START',model,mode,flush=True);root=a.fixtures/f'{model}-{mode}-image';head='sam2_convs' if model=='sam3' else 'interactive_convs'
            vision=torch.ops.sam3_native.vision_encode(str(a.store),model,inputs,mode,[head]);pyramid=[vision[f'{head}.{i}'] for i in range(3)];actuals=[];count=0
            for i,name in enumerate(['batch','batch-1']):
                actual=read(root,name);expected=torch.ops.sam3_native.interactive_image(str(a.store),model,pyramid,[h,h],[w,w],i,None,None,boxes,None,False,True,False,0.,0.,0.,mode)
                for key,value in zip(['masks','iou','low_res_logits'],expected):torch.testing.assert_close(actual[key],value.cpu(),rtol=0,atol=0);count+=1
                actuals.append(actual);del expected
            differences={key:float((actuals[0][key].float()-actuals[1][key].float()).abs().max()) for key in actuals[0]}
            row=dict(model=model,mode=mode,exact_tensor_comparisons=count,batch_position_max_abs_difference=differences,batch_position_differing_values={key:int((actuals[0][key]!=actuals[1][key]).sum()) for key in actuals[0]});rows.append(row);print(json.dumps(row),flush=True)
            del vision,pyramid,actuals;gc.collect();torch.cuda.empty_cache()
    a.report.write_text(json.dumps(dict(cases=rows,scope='Both batch positions compared independently at zero tolerance to the previously original-validated C++ composition. Identical image values in different batch positions need not produce bit-identical floating point results.'),indent=2)+'\n')
if __name__=='__main__':main()
