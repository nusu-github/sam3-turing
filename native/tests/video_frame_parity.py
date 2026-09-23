"""One native trunk feeds detector and tracker on actual decoded video frames."""
import argparse,gc,json
from pathlib import Path
from types import SimpleNamespace
import torch
from sam3.model_builder import (_create_vit_backbone,_create_vit_neck,_create_position_encoding,_create_multiplex_tri_backbone,_create_text_encoder,_create_sam3_transformer,_create_geometry_encoder,_create_segmentation_head,_create_dot_product_scoring,_create_sam3_model)
from sam3.model.vl_combiner import SAM3VLBackbone,SAM3VLBackboneTri
from sam3.model.geometry_encoders import Prompt
from sam3.model.utils.sam2_utils import _load_img_as_tensor
from tracking_frame_parity import reference as sam3_tracker
from multiplex_frame_parity import reference as sam31_tracker

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--frames',nargs='+',type=Path,default=[Path(f'assets/videos/0001/{i}.jpg') for i in [0,1]]);p.add_argument('--modes',nargs='+',default=['fp16','bf16_reference','fp32']);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(1);torch.manual_seed(189);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;rows=[]
 for spec in a.checkpoint:
  model,path=spec.split('=',1);tri=model=='sam3.1';weights=torch.load(path,map_location='cpu',mmap=True,weights_only=True);weights=weights.get('model',weights)
  neck=_create_multiplex_tri_backbone() if tri else _create_vit_neck(_create_position_encoding(),_create_vit_backbone(),True)
  text=_create_text_encoder('sam3/assets/bpe_simple_vocab_16e6.txt.gz');backbone=(SAM3VLBackboneTri if tri else SAM3VLBackbone)(visual=neck,text=text,scalp=0 if tri else 1)
  ref=_create_sam3_model(backbone,_create_sam3_transformer(),_create_geometry_encoder(),_create_segmentation_head(),_create_dot_product_scoring(),None,True).eval()
  ref.load_state_dict({k[len('detector.'):]:v for k,v in weights.items() if k.startswith('detector.')},strict=True);ref.cuda();ref.supervise_joint_box_scores=True
  tracker=(sam31_tracker if tri else sam3_tracker)(weights,'cuda');original=[block.mlp.forward for block in neck.trunk.blocks]
  for mode in a.modes:
   dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
   for block,fn in zip(neck.trunk.blocks,original):block.mlp.forward=fn if mode=='bf16_reference' else lambda x,m=block.mlp:m.fc2(m.act(m.fc1(x)))
   with torch.autocast('cuda',enabled=mode!='fp32',dtype=dtype):language=backbone.forward_text(['person','ball'],device='cuda')
   for path in a.frames:
    print('START',model,mode,path,flush=True);image,_,_=_load_img_as_tensor(str(path),1008);image=image.float().cuda()[None].sub(.5).div(.5)
    batch=2;ids=torch.zeros(batch,device='cuda',dtype=torch.long);text_ids=torch.arange(batch,device='cuda');points=torch.tensor([[[.5,.6],[.4,.5]]],device='cuda');pl=torch.ones(1,batch,device='cuda',dtype=torch.long);pm=torch.tensor([[False],[True]],device='cuda');boxes=torch.empty(0,batch,4,device='cuda');bl=torch.empty(0,batch,device='cuda',dtype=torch.long);bm=torch.empty(batch,0,device='cuda',dtype=torch.bool)
    geo=Prompt(point_embeddings=points,point_labels=pl,point_mask=pm,box_embeddings=boxes,box_labels=bl,box_mask=bm);find=SimpleNamespace(img_ids=ids,text_ids=text_ids)
    with torch.autocast('cuda',enabled=mode!='fp32',dtype=dtype):
     features=backbone.forward_image(image);features.update(language)
     expected={};pyramid=features['backbone_fpn'];pyramid=[getattr(x,'tensors',x) for x in pyramid]
     for i,x in enumerate(pyramid):expected[f'detection/{i}']=x
     expected['position']=features['vision_pos_enc'][-1]
     for name,key,decoder in [('interactive','interactive' if tri else 'sam2_backbone_out',tracker.interactive_sam_mask_decoder if tri else tracker.sam_mask_decoder),('propagation','sam2_backbone_out',tracker.sam_mask_decoder)]:
      f=features[key];p=[getattr(x,'tensors',x) for x in f['backbone_fpn']];expected.update({name+'/image':p[2],name+'/position':f['vision_pos_enc'][2],name+'/high0':decoder.conv_s0(p[0]),name+'/high1':decoder.conv_s1(p[1])})
     features['backbone_fpn']=pyramid
     prompt,padding,_=ref._encode_prompt(features,find,geo);_,encoded,_=ref._run_encoder(features,find,prompt,padding)
     detections={'encoder_hidden_states':encoded['encoder_hidden_states']};detections,hs=ref._run_decoder(encoded['pos_embed'],encoded['encoder_hidden_states'],encoded['padding_mask'],detections,prompt,padding,encoded);ref._run_segmentation_heads(detections,features,ids,encoded['vis_feat_sizes'],encoded['encoder_hidden_states'],prompt,padding,hs)
    actual=torch.ops.sam3_native.video_frame_features(str(a.store),model,image,[ids,text_ids,language['language_features'],language['language_mask'],points,pl,pm,boxes,bl,bm],mode)
    for key,value in expected.items():torch.testing.assert_close(actual[key],value,rtol=0,atol=0)
    errors={}
    for key,value in actual.items():
     if key in expected:continue
     tolerance={'fp32':1e-4,'fp16':.01,'bf16_reference':.05}[mode];torch.testing.assert_close(value,detections[key],rtol=tolerance,atol=tolerance);errors[key]=(value-detections[key]).abs().max().item()
    rows.append(dict(model=model,mode=mode,frame=str(path),prompts=batch,exact_feature_tensors=len(expected),query_count=actual['pred_logits'].size(1),max_detection_errors=errors));print(rows[-1],flush=True)
    del actual,features,expected,detections,encoded,hs
   del language;gc.collect();torch.cuda.empty_cache()
  del ref,backbone,neck,text,tracker,weights;gc.collect();torch.cuda.empty_cache()
 a.report.write_text(json.dumps(dict(cases=rows,torch=torch.__version__,gpu=torch.cuda.get_device_name(),scope='Actual decoded video images through shared full 1008 vision trunk, detector neck and all projected tracker necks, arbitrary text+geometry prompt batches, all 200 detector queries with source video joint-presence scoring. Same-mode source MLP adaptations for FP16/FP32. Does not yet validate high-level prompt lifecycle or complete tracking output.'),indent=2)+'\n')
if __name__=='__main__':main()
