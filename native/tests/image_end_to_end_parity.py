"""Real image: native preprocess, vision, text, grounding, scores and all masks."""
import argparse,json,gc
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image
from torchvision.transforms import v2
from sam3.model_builder import (_create_vit_backbone,_create_vit_neck,_create_position_encoding,
    _create_multiplex_tri_backbone,_create_text_encoder,_create_sam3_transformer,_create_geometry_encoder,
    _create_segmentation_head,_create_dot_product_scoring,_create_sam3_model)
from sam3.model.vl_combiner import SAM3VLBackbone,SAM3VLBackboneTri
from sam3.model.geometry_encoders import Prompt

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--image',type=Path,default=Path('assets/images/truck.jpg'))
    p.add_argument('--modes',nargs='+',default=['bf16_reference','fp16','fp32']);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(189)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    pixels=v2.functional.to_image(Image.open(a.image).convert('RGB')).cuda()
    transform=v2.Compose([v2.ToDtype(torch.uint8,scale=True),v2.Resize((1008,1008)),v2.ToDtype(torch.float32,scale=True),v2.Normalize([.5]*3,[.5]*3)])
    image=transform(pixels)[None];native_image=torch.ops.sam3_native.preprocess_rgb(pixels)
    torch.testing.assert_close(image,native_image,rtol=0,atol=0)
    assert image.stride()==native_image.stride()
    results=[];bpe='sam3/assets/bpe_simple_vocab_16e6.txt.gz'
    for spec in a.checkpoint:
        model,path=spec.split('=',1);tri=model=='sam3.1'
        neck=_create_multiplex_tri_backbone() if tri else _create_vit_neck(_create_position_encoding(),_create_vit_backbone(),True)
        text=_create_text_encoder(bpe);backbone=(SAM3VLBackboneTri if tri else SAM3VLBackbone)(visual=neck,text=text,scalp=0 if tri else 1)
        ref=_create_sam3_model(backbone,_create_sam3_transformer(),_create_geometry_encoder(),_create_segmentation_head(),_create_dot_product_scoring(),None,True).eval()
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);state=state.get('model',state)
        ref.load_state_dict({k[len('detector.'):]:v for k,v in state.items() if k.startswith('detector.')},strict=True);ref.cuda()
        ref.supervise_joint_box_scores=tri
        original=[block.mlp.forward for block in neck.trunk.blocks]
        prompts=['truck','wheel','visual','赤い車']
        tokens=text.tokenizer(prompts,context_length=32).cuda()
        for mode in a.modes:
            dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
            for block,fn in zip(neck.trunk.blocks,original):
                block.mlp.forward=fn if mode=='bf16_reference' else lambda x,m=block.mlp:m.fc2(m.act(m.fc1(x)))
            with torch.autocast('cuda',enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                features=backbone.forward_image(image)
                features.update(backbone.forward_text(prompts,device='cuda'))
            if tri:
                # The multiplex neck wraps unpadded image tensors. Compare its
                # detector math through the common Sam3Image implementation.
                assert all(x.mask is None for x in features['backbone_fpn'])
                features['backbone_fpn']=[x.tensors for x in features['backbone_fpn']]
            native_vision=torch.ops.sam3_native.vision_encode(str(a.store),model,native_image,mode,['convs'])
            native_pyramid=[native_vision[f'convs.{i}'] for i in range(3)]
            native_pos=native_vision['position.2']
            tp,tf,_=torch.ops.sam3_native.text_encode(str(a.store),model,tokens,mode)
            for i in range(3):torch.testing.assert_close(native_pyramid[i],features['backbone_fpn'][i],rtol=0,atol=0)
            torch.testing.assert_close(tf,features['language_features'],rtol=0,atol=0)
            for case in ['text','geometry']+(['mixed_visual_previous'] if mode=='fp16' else []):
                batch=3 if case=='mixed_visual_previous' else 1
                image_ids=torch.zeros(batch,device='cuda',dtype=torch.long)
                text_ids=torch.tensor([1,0,3] if batch==3 else [2 if case=='geometry' else 0],device='cuda')
                np=0 if case=='text' else 2;nb=0 if case=='text' else 2
                points=torch.rand(np,batch,2,device='cuda');boxes=torch.rand(nb,batch,4,device='cuda')
                if nb:boxes[0]=torch.tensor([.5,.5,.8,.8],device='cuda')
                pl=torch.arange(np*batch,device='cuda').view(np,batch)%2;bl=torch.arange(nb*batch,device='cuda').view(nb,batch)%2
                pm=torch.zeros(batch,np,dtype=torch.bool,device='cuda');bm=torch.zeros(batch,nb,dtype=torch.bool,device='cuda')
                visual=visual_mask=previous=None
                if batch==3:
                    pm[1,1]=True;bm[2,1]=True
                    visual=torch.randn(2,batch,256,device='cuda')*.01;visual_mask=torch.tensor([[False,True],[False,False],[True,True]],device='cuda')
                    previous=torch.randn(72*72,batch,256,device='cuda')*.01
                geo=Prompt(point_embeddings=points,point_labels=pl,point_mask=pm,box_embeddings=boxes,box_labels=bl,box_mask=bm)
                find=SimpleNamespace(img_ids=image_ids,text_ids=text_ids)
                with torch.autocast('cuda',enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                    prompt,padding,_=ref._encode_prompt(features,find,geo,visual_prompt_embed=visual,visual_prompt_mask=visual_mask,prev_mask_pred=previous)
                    _,encoded,_=ref._run_encoder(features,find,prompt,padding)
                    expected={'encoder_hidden_states':encoded['encoder_hidden_states']}
                    expected,hs=ref._run_decoder(encoded['pos_embed'],encoded['encoder_hidden_states'],encoded['padding_mask'],expected,prompt,padding,encoded)
                    ref._run_segmentation_heads(expected,features,image_ids,encoded['vis_feat_sizes'],encoded['encoder_hidden_states'],prompt,padding,hs)
                actual=torch.ops.sam3_native.grounding(str(a.store),model,native_pyramid,native_pos,image_ids,text_ids,tf,tp,
                    points,pl,pm,boxes,bl,bm,visual,visual_mask,previous,True,tri,mode)
                errors={}
                for key,value in actual.items():
                    tolerance={'fp32':1e-4,'fp16':.01,'bf16_reference':.05}[mode]
                    torch.testing.assert_close(value,expected[key],rtol=tolerance,atol=tolerance)
                    errors[key]=(value-expected[key]).abs().max().item()
                item=dict(model=model,mode=mode,case=case,joint_scores=tri,mask_shape=list(actual['pred_masks'].shape),max_errors=errors)
                results.append(item);print(json.dumps(item),flush=True)
                del actual,expected,hs,encoded
            del features,native_vision,native_pyramid,tf,tp
        del ref,backbone,neck,text,state;gc.collect();torch.cuda.empty_cache()
    a.report.write_text(json.dumps(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),image=str(a.image),cases=results,
        scope='Real-image native tensor pipeline. Test tokenizer is Python; native string tokenization and interactive/video sessions remain outstanding.'),indent=2)+'\n')
if __name__=='__main__':main()
