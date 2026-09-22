"""Image predictor host transforms, re-prompting and postprocessing parity."""
import argparse,json,warnings
from pathlib import Path
from types import SimpleNamespace
import torch
from sam3.sam.prompt_encoder import PromptEncoder
from sam3.sam.mask_decoder import MaskDecoder
from sam3.sam.transformer import TwoWayTransformer
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.utils.sam1_utils import SAM2Transforms

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(281)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    results=[]
    for model_path in a.checkpoint:
        model,path=model_path.split('=',1);state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);state=state.get('model',state)
        root='tracker.' if model=='sam3' else 'tracker.model.interactive_'
        pe=PromptEncoder(256,(72,72),(1008,1008),16).eval()
        decoder=MaskDecoder(transformer_dim=256,transformer=TwoWayTransformer(2,256,8,2048),num_multimask_outputs=3,
            use_high_res_features=True,iou_prediction_use_sigmoid=model=='sam3',pred_obj_scores=True,pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,dynamic_multimask_via_stability=True).eval()
        for module,suffix in [(pe,'sam_prompt_encoder.'),(decoder,'sam_mask_decoder.')]:
            prefix=root+suffix;module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);module.to(a.device)
        no_mem=state['tracker.no_mem_embed' if model=='sam3' else 'tracker.model.interactivity_no_mem_embed'].to(a.device)
        ref=SAM3InteractiveImagePredictor(SimpleNamespace(image_size=1008,device=torch.device(a.device),sam_prompt_encoder=pe,sam_mask_decoder=decoder))
        sizes=[(97,149),(121,79)]
        for mode in a.modes:
            dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
            pyramid=[torch.randn(2,256,s,s,device=a.device).to(dtype).contiguous(memory_format=torch.channels_last) for s in [288,144,72]]
            with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                cached=(pyramid[2].flatten(2).permute(2,0,1)+no_mem).permute(1,2,0).view(2,256,72,72)
                ref._features=dict(image_embed=cached,high_res_feats=[decoder.conv_s0(pyramid[0]),decoder.conv_s1(pyramid[1])])
            ref._is_image_set=True;ref._is_batch=True;ref._orig_hw=sizes
            previous=None
            cases=[('none',0,None,None,None,True,True,0.,256.,0.),
                   ('points',1,torch.tensor([[20.,30.],[40.,65.]]),torch.tensor([1,0]),None,True,True,0.,256.,0.),
                   ('boxes',0,None,None,torch.tensor([[10.,15.,120.,80.],[40.,30.,80.,70.]]),True,False,0.,0.,0.),
                   ('combined',1,torch.tensor([[[20.,30.]],[[50.,60.]]]),torch.tensor([[1],[0]]),torch.tensor([[1.,1.,40.,60.],[30.,40.,70.,110.]]),True,True,.25,9.,7.),
                   ('normalized',1,torch.tensor([[.2,.3],[.7,.8]]),torch.tensor([1,0]),torch.tensor([.1,.1,.9,.9]),False,False,-.25,0.,3.),
                   ('refine',1,torch.tensor([[40.,45.]]),torch.tensor([1]),None,True,False,0.,256.,0.),
                   ('mask_only',0,None,None,None,True,True,0.,256.,0.)]
            for name,index,points,labels,boxes,pixels,multi,threshold,holes,sprinkles in cases:
                mask=previous if name=='refine' else torch.randn(1,288,288) if name=='mask_only' else None
                ref.mask_threshold=threshold;ref._transforms=SAM2Transforms(1008,threshold,holes,sprinkles)
                with warnings.catch_warnings():
                    warnings.simplefilter('error',UserWarning)
                    with torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                        mi,pc,pl,bc=ref._prep_prompts(points,labels,boxes,mask,pixels,img_idx=index)
                        expected=ref._predict(pc,pl,bc,mi,multi,return_logits=name!='boxes',img_idx=index)
                projected=name=='refine'
                features=ref._features['high_res_feats']+[pyramid[2]] if projected else pyramid
                actual=torch.ops.sam3_native.interactive_image(str(a.store),model,features,[h for h,w in sizes],[w for h,w in sizes],index,
                    points,labels,boxes,mask,pixels,multi,name!='boxes',threshold,holes,sprinkles,mode,projected)
                errors={}
                for key,value,target in zip(['masks','iou','low_res_logits','cached_image'],actual,(*expected,cached)):
                    torch.testing.assert_close(value,target,rtol=0,atol=0)
                    errors[key]=(value.float()-target.float()).abs().max().item()
                if name=='points':previous=actual[2][:,:1].clone()
                item=dict(model=model,mode=mode,case=name,index=index,projected_high=projected,mask_shape=list(actual[0].shape),holes=holes,sprinkles=sprinkles,max_errors=errors)
                results.append(item);print(json.dumps(item),flush=True)
    # Direct postprocessing covers positive/negative components and both passes
    # reading original logits, independent of random model predictions.
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        masks=torch.full((2,3,13,17),-4.,device=a.device,dtype=dtype)
        masks[:,:,2:11,2:15]=4.;masks[:,:,5,5]=-4.;masks[:,:,0,0]=4.;masks[:,:,3,3]=0.
        masks[1,1].zero_();masks[0,2].fill_(5.)
        for threshold,holes,sprinkles in [(0.,0.,0.),(0.,1.,0.),(0.,0.,1.),(.25,9.,7.),(-.25,256.,256.)]:
            transform=SAM2Transforms(1008,threshold,holes,sprinkles)
            with warnings.catch_warnings():
                warnings.simplefilter('error',UserWarning)
                expected=transform.postprocess_masks(masks,(37,29))
            actual=torch.ops.sam3_native.interactive_postprocess(masks,37,29,threshold,holes,sprinkles)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            results.append(dict(case='postprocess_components',dtype=str(dtype),threshold=threshold,holes=holes,sprinkles=sprinkles,max_errors={'masks':0.}))
    a.report.write_text(json.dumps(dict(torch=torch.__version__,device=a.device,cases=results,
        scope='Original SAM3 image predictor host behavior; SAM3.1 interactive modules use the same image host here. Multiplex/video orchestration is not tested.'),indent=2)+'\n')
if __name__=='__main__':main()
