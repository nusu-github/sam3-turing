"""Real RGB pixels to standalone native interactive masks and mask refinement."""
import argparse,gc,json,os,subprocess,time,warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
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
    p=argparse.ArgumentParser();p.add_argument('executable',type=Path);p.add_argument('store',type=Path)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--image',type=Path,default=Path('assets/images/truck.jpg'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--modes',nargs='+',default=['bf16_reference','fp16','fp32']);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    a.output.mkdir(parents=True,exist_ok=True);pil=Image.open(a.image).convert('RGB');pil.save(a.output/'input.ppm')
    transform=v2.Compose([v2.ToDtype(torch.uint8,scale=True),v2.Resize((1008,1008)),v2.ToDtype(torch.float32,scale=True),v2.Normalize([.5]*3,[.5]*3)])
    image=transform(v2.functional.to_image(pil).cuda())[None];results=[]
    points=torch.tensor([[500.,600.],[100.,100.]]);labels=torch.tensor([1,0])
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
        ref=SAM3InteractiveImagePredictor(SimpleNamespace(image_size=1008,device=torch.device('cuda'),sam_prompt_encoder=pe,sam_mask_decoder=decoder))
        original=[block.mlp.forward for block in neck.trunk.blocks]
        for mode in a.modes:
            for block,fn in zip(neck.trunk.blocks,original):block.mlp.forward=fn if mode=='bf16_reference' else lambda x,m=block.mlp:m.fc2(m.act(m.fc1(x)))
            dtype=torch.float16 if mode=='fp16' else torch.bfloat16
            expected=[]
            with warnings.catch_warnings():
                warnings.simplefilter('error',UserWarning)
                with torch.autocast('cuda',enabled=mode!='fp32',dtype=dtype):
                    output=neck(image);pyramid=[getattr(x,'tensors',x) for x in output[2]][:3]
                    cached=(pyramid[2].flatten(2).permute(2,0,1)+no_mem).permute(1,2,0).view(1,256,72,72)
                    ref._features=dict(image_embed=cached,high_res_feats=[decoder.conv_s0(pyramid[0]),decoder.conv_s1(pyramid[1])]);ref._is_image_set=True;ref._orig_hw=[(pil.height,pil.width)]
                    mask=None
                    for stage in ['initial','refined']:
                        mi,pc,pl,bc=ref._prep_prompts(points,labels,None,mask,True)
                        value=ref._predict(pc,pl,bc,mi,stage=='initial',False)
                        expected.append(tuple(x.cpu() for x in value))
                        best=value[1][0].argmax();mask=value[2][0,best].unsqueeze(0)
            prefix=a.output/(model+'-'+mode)
            command=[str(a.executable.resolve()),str(a.store.resolve()),model,'cuda',mode,str((a.output/'input.ppm').resolve()),str(prefix.resolve()),'500','600','1','100','100','0']
            started=time.perf_counter();child=subprocess.run(command,env={**os.environ,'PATH':'/nonexistent'},text=True,capture_output=True,check=True)
            elapsed=time.perf_counter()-started;Path(str(prefix)+'.log').write_text(child.stdout+child.stderr)
            for stage,(emasks,eiou,elow) in zip(['initial','refined'],expected):
                base=Path(str(prefix)+'.'+stage);metadata=json.loads(Path(str(base)+'.json').read_text())
                assert metadata['count']==eiou.numel() and (metadata['height'],metadata['width'])==(pil.height,pil.width)
                assert metadata['mask_row_bytes']==(pil.height*pil.width+7)//8
                packed=np.frombuffer(Path(str(base)+'.masks.bin').read_bytes(),dtype=np.uint8).reshape(metadata['count'],-1)
                masks=np.unpackbits(packed,axis=1,bitorder='little')[:,:pil.height*pil.width].reshape(emasks.shape)
                np.testing.assert_array_equal(masks,emasks.numpy())
                torch.testing.assert_close(torch.tensor(metadata['iou']).reshape(eiou.shape),eiou.float(),rtol=0,atol=0)
                low=np.frombuffer(Path(str(base)+'.lowres.f32.bin').read_bytes(),dtype='<f4').reshape(elow.shape)
                np.testing.assert_array_equal(low,elow.float().numpy())
                torch.save(dict(masks=emasks,iou=eiou,low_res_logits=elow),Path(str(base)+'.reference.pt'))
                item=dict(model=model,mode=mode,stage=stage,count=metadata['count'],differing_mask_pixels=0,max_iou_error=0,max_low_res_error=0,child_path='/nonexistent',process_elapsed_seconds=elapsed)
                results.append(item);print(json.dumps(item),flush=True)
            del output,pyramid,cached,expected,value,mask;ref._features=None;torch.cuda.empty_cache()
        del neck,pe,decoder,state,ref,original,no_mem;gc.collect();torch.cuda.empty_cache()
    a.report.write_text(json.dumps(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),image=str(a.image),points=points.tolist(),labels=labels.tolist(),cases=results,
        scope='Real-image interactive image host, two-step refinement with feature reuse. SAM3.1 uses its interactive neural modules; video object gating/memory/multiplex is separate.'),indent=2)+'\n')
if __name__=='__main__':main()
