"""C ABI grounding/text outputs versus the already verified C++ module path."""
import argparse,gc,json
from pathlib import Path
import numpy as np
import torch
from sam3.model.tokenizer_ve import SimpleTokenizer
from c_api_parity import read

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('fixtures',type=Path);p.add_argument('--models',nargs='+',default=['sam3','sam3.1']);p.add_argument('--modes',nargs='+',default=['fp16','bf16_reference','fp32']);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    h,w=map(int,(a.fixtures/'inputs/ground/size.txt').read_text().split());rgb=torch.from_numpy(np.fromfile(a.fixtures/'inputs/ground/0.rgb',dtype=np.uint8).reshape(h,w,3).copy()).permute(2,0,1).contiguous().cuda()
    image=torch.ops.sam3_native.preprocess_rgb(rgb);tokenizer=SimpleTokenizer('sam3/assets/bpe_simple_vocab_16e6.txt.gz');tokens=tokenizer(['a truck','person'],context_length=32);rows=[]
    mapping={'raw/logits':'pred_logits','raw/boxes':'pred_boxes','raw/masks':'pred_masks','raw/semantic':'semantic_seg','raw/presence_logits':'presence_logit_dec','raw/presence':'presence_feats','raw/queries':'queries'}
    for model in a.models:
        for mode in a.modes:
            print('START',model,mode,flush=True);root=a.fixtures/f'{model}-{mode}-ground';text=read(root,'text');count=0
            torch.testing.assert_close(text['tokens'],tokens,rtol=0,atol=0);count+=1
            torch.testing.assert_close(read(root,'nul-tokenize')['tokens'],tokenizer(['a\0 truck'],context_length=32),rtol=0,atol=0);count+=1
            padding,features,embeddings=torch.ops.sam3_native.text_encode(str(a.store),model,tokens.cuda(),mode)
            for name,value in [('padding',padding),('features',features),('embeddings',embeddings)]:torch.testing.assert_close(text[name],value.cpu(),rtol=0,atol=0);count+=1
            vision=torch.ops.sam3_native.vision_encode(str(a.store),model,image,mode,['convs']);pyramid=[vision[f'convs.{i}'] for i in range(3)]
            for case in ['ground','geometry','visual']:
                batch=1 if case=='ground' else 2;ids=torch.zeros(batch,device='cuda',dtype=torch.long);text_ids=ids if batch==1 else torch.tensor([1,0],device='cuda')
                points=torch.empty(0,batch,2,device='cuda');point_labels=torch.empty(0,batch,device='cuda',dtype=torch.long);point_padding=torch.empty(batch,0,device='cuda',dtype=torch.bool)
                if batch==1:boxes=torch.empty(0,1,4,device='cuda');labels=point_labels;box_padding=point_padding
                else:boxes=torch.tensor([[[.5,.5,.6,.7],[.4,.4,.3,.3]]],device='cuda');labels=torch.tensor([[1,0]],device='cuda');box_padding=torch.zeros(2,1,device='cuda',dtype=torch.bool)
                visual=torch.full((3,2,256),.125,device='cuda') if case=='visual' else None
                visual_padding=torch.tensor([[False,False,True],[False,True,True]],device='cuda') if case=='visual' else None
                previous=torch.full((5184,2,256),.01,device='cuda') if case=='visual' else None
                raw=torch.ops.sam3_native.grounding(str(a.store),model,pyramid,vision['position.2'],ids,text_ids,features,padding,points,point_labels,point_padding,boxes,labels,box_padding,visual,visual_padding,previous,case!='visual',model=='sam3.1',mode)
                actual=read(root,case)
                for key,source in mapping.items():torch.testing.assert_close(actual[key],raw[source].cpu(),rtol=0,atol=0);count+=1
                result=torch.ops.sam3_native.postprocess_image(raw['pred_boxes'],raw['pred_logits'],raw['pred_masks'],raw['presence_logit_dec'],[h]*batch,[w]*batch,-1. if batch==1 else 1.,model=='sam3',8,mode)
                assert actual['batch_count'].item()==batch;count+=1
                for i in range(batch):
                    for field,value in zip(['boxes','scores','mask_probabilities','masks','query_indices'],result):torch.testing.assert_close(actual[f'{i}/{field}'],value[i].cpu(),rtol=0,atol=0);count+=1
                del raw,actual,result
            rows.append(dict(model=model,mode=mode,exact_tensor_comparisons=count,uncapped_detections=200));print(json.dumps(rows[-1]),flush=True)
            del vision,pyramid,padding,features,embeddings,text;gc.collect();torch.cuda.empty_cache()
    a.report.write_text(json.dumps(dict(cases=rows,scope='C11 standalone outputs match the previously original-validated C++ components at zero tolerance: text, embedded-NUL tokenization, full 200 detections, repeated image/reordered text IDs, positive/negative boxes, visual prompts and previous-mask features with text disabled. This validates the ABI composition, not a new independent original neural comparison.'),indent=2)+'\n')
if __name__=='__main__':main()
