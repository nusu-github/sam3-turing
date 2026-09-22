"""Real-pixel image batches through the native reusable image session."""
import argparse,gc,json,warnings
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image
from torchvision.transforms import v2
from sam3.model_builder import _create_vit_backbone
from sam3.model.necks import Sam3DualViTDetNeck,Sam3TriViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.sam.prompt_encoder import PromptEncoder
from sam3.sam.mask_decoder import MaskDecoder
from sam3.sam.transformer import TwoWayTransformer
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    first=v2.functional.to_image(Image.open('assets/images/truck.jpg').convert('RGB')).cuda()
    pixels=[first,first[:,:401,:777].flip(-1)]
    transform=v2.Compose([v2.ToDtype(torch.uint8,scale=True),v2.Resize((1008,1008)),v2.ToDtype(torch.float32,scale=True),v2.Normalize([.5]*3,[.5]*3)])
    results=[]
    for spec in a.checkpoint:
        model,path=spec.split('=',1);tri=model=='sam3.1';root='tracker.model.interactive_' if tri else 'tracker.'
        neck=(Sam3TriViTDetNeck if tri else Sam3DualViTDetNeck)(trunk=_create_vit_backbone(),position_encoding=PositionEmbeddingSine(256),
            d_model=256,scale_factors=[4.,2.,1.] if tri else [4.,2.,1.,.5],**({} if tri else {'add_sam2_neck':True})).eval()
        pe=PromptEncoder(256,(72,72),(1008,1008),16).eval()
        decoder=MaskDecoder(transformer_dim=256,transformer=TwoWayTransformer(2,256,8,2048),num_multimask_outputs=3,
            use_high_res_features=True,iou_prediction_use_sigmoid=not tri,pred_obj_scores=True,pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,dynamic_multimask_via_stability=True).eval()
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);state=state.get('model',state)
        for module,prefix in [(neck,'detector.backbone.vision_backbone.'),(pe,root+'sam_prompt_encoder.'),(decoder,root+'sam_mask_decoder.')]:
            module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);module.cuda()
        no_mem=state['tracker.model.interactivity_no_mem_embed' if tri else 'tracker.no_mem_embed'].cuda()
        for block in neck.trunk.blocks:block.mlp.forward=lambda x,m=block.mlp:m.fc2(m.act(m.fc1(x)))
        ref=SAM3InteractiveImagePredictor(SimpleNamespace(image_size=1008,device=torch.device('cuda'),sam_prompt_encoder=pe,sam_mask_decoder=decoder))
        for batch in [1,2]:
            batch_pixels=pixels[:batch];sizes=[tuple(x.shape[-2:]) for x in batch_pixels]
            inputs=torch.stack([transform(x) for x in batch_pixels])
            with warnings.catch_warnings():
                warnings.simplefilter('error',UserWarning)
                with torch.autocast('cuda',dtype=torch.float16):
                    output=neck(inputs);pyramid=[getattr(x,'tensors',x) for x in output[2]][:3]
                    cached=(pyramid[2].flatten(2).permute(2,0,1)+no_mem).permute(1,2,0).view(batch,256,72,72)
                    ref._features=dict(image_embed=cached,high_res_feats=[decoder.conv_s0(pyramid[0]),decoder.conv_s1(pyramid[1])]);ref._is_image_set=True;ref._orig_hw=sizes
                    expected=[ref._predict(None,None,multimask_output=True,img_idx=i) for i in range(batch)]
            actual=torch.ops.sam3_native.interactive_image_batch_pixels(str(a.store),model,batch_pixels,'fp16')
            torch.testing.assert_close(actual[3],cached,rtol=0,atol=0)
            for i,item in enumerate(expected):
                for j,value in enumerate(item):torch.testing.assert_close(actual[j][i],value,rtol=0,atol=0)
            entry=dict(model=model,mode='fp16',batch=batch,original_sizes=sizes,max_abs_error=0.,differing_mask_pixels=0)
            results.append(entry);print(json.dumps(entry),flush=True)
            del actual,expected,output,pyramid,cached;ref._features=None;torch.cuda.empty_cache()
        del ref,neck,pe,decoder,state,no_mem;gc.collect();torch.cuda.empty_cache()
    a.report.write_text(json.dumps(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),cases=results),indent=2)+'\n')
if __name__=='__main__':main()
