"""Score/box/mask parity using original score update and segmentation modules."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from sam3.model_builder import _create_dot_product_scoring,_create_segmentation_head
from sam3.model.model_misc import MLP
from sam3.model.sam3_image import Sam3Image

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(183)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    results=[]
    for spec in a.checkpoint:
        model,path=spec.split('=',1);score=_create_dot_product_scoring().eval();seg=_create_segmentation_head().eval();box=MLP(256,256,4,3).eval()
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);state=state.get('model',state)
        for prefix,module in [('detector.dot_prod_scoring.',score),('detector.segmentation_head.',seg),('detector.transformer.decoder.bbox_embed.',box)]:
            module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);module.to(a.device)
        host=SimpleNamespace(training=False,use_dot_prod_scoring=True,dot_prod_scoring=score,detach_presence_in_joint_score=False,
            transformer=SimpleNamespace(decoder=SimpleNamespace(dac=True,bbox_embed=box)))
        for mode in a.modes:
            dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
            for case,sources,ids,h,w in [('full',1,[0],72,72),('repeat_image',1,[0,0,0],11,17),('remap_batch',3,[2,0,2,1],9,13)]:
                batch=len(ids);image_ids=torch.tensor(ids,device=a.device)
                pyramid=[torch.randn(sources,256,h*s,w*s,device=a.device).to(dtype) for s in [4,2,1]]
                # Exercise channels-last convolution layouts and image gathering.
                if case=='remap_batch':pyramid=[x.contiguous(memory_format=torch.channels_last) for x in pyramid]
                memory=torch.randn(h*w,batch,256,device=a.device).to(dtype)
                prompt=torch.randn(23,batch,256,device=a.device);pm=torch.zeros(batch,23,dtype=torch.bool,device=a.device)
                for i in range(batch):pm[i,21-i:]=True
                hidden=torch.randn(6,200,batch,256,device=a.device);refs=torch.rand(6,200,batch,4,device=a.device)
                pl=torch.randn(6,1,batch,device=a.device).to(dtype)*5;presence=torch.randn(1,batch,256,device=a.device)
                for joint in [False,True]:
                    host.supervise_joint_box_scores=joint
                    with torch.autocast(a.device,dtype=dtype if mode!='fp32' else torch.bfloat16,enabled=mode!='fp32'):
                        expected={'presence_feats':presence}
                        Sam3Image._update_scores_and_boxes(host,expected,hidden.transpose(1,2),refs.transpose(1,2),prompt,pm,pl.transpose(1,2))
                        expected.update(seg(backbone_feats=pyramid,obj_queries=hidden.transpose(1,2),image_ids=image_ids,
                                            encoder_hidden_states=memory,prompt=prompt,prompt_mask=pm))
                    with torch.autocast(a.device,dtype=torch.bfloat16):
                        actual=torch.ops.sam3_native.detection_heads(str(a.store),model,pyramid,image_ids,memory,prompt,pm,hidden,refs,pl,presence,joint,mode)
                        assert torch.is_autocast_enabled(a.device) and torch.get_autocast_dtype(a.device)==torch.bfloat16
                    errors={}
                    for name,value in expected.items():
                        if value is None:continue
                        tolerance={'fp32':5e-5,'fp16':.005,'bf16_reference':.03}[mode]
                        torch.testing.assert_close(actual[name],value,rtol=tolerance,atol=tolerance)
                        errors[name]=(actual[name]-value).abs().max().item()
                    item=dict(model=model,mode=mode,case=case,joint_scores=joint,mask_shape=list(actual['pred_masks'].shape),max_errors=errors)
                    results.append(item);print(json.dumps(item),flush=True)
    a.report.write_text(json.dumps(dict(torch=torch.__version__,device=a.device,cases=results),indent=2)+'\n')
if __name__=='__main__':main()
