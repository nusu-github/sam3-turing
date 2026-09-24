"""Compare native video interactive heads against original tracker methods."""
import argparse
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
import torch
from sam3.model.sam3_tracker_base import Sam3TrackerBase
from sam3.model.video_tracking_multiplex import VideoTrackingMultiplex
from sam3.sam.prompt_encoder import PromptEncoder
from sam3.sam.mask_decoder import MaskDecoder, MLP
from sam3.sam.transformer import TwoWayTransformer

KEYS=['low_res_multimasks','high_res_multimasks','ious','low_res_masks','high_res_masks','obj_ptr','object_score_logits']

def reference(state,model,device):
    multiplex=model=='sam3.1';root='tracker.model.' if multiplex else 'tracker.'
    prompt=PromptEncoder(256,(72,72),(1008,1008),16).eval()
    decoder=MaskDecoder(transformer_dim=256,transformer=TwoWayTransformer(2,256,8,2048),num_multimask_outputs=3,
        use_high_res_features=True,iou_prediction_use_sigmoid=not multiplex,pred_obj_scores=True,pred_obj_scores_mlp=True,
        use_multimask_token_for_obj_ptr=True,dynamic_multimask_via_stability=True).eval()
    host=SimpleNamespace(training=False,hidden_dim=256,sam_prompt_embed_dim=256,sam_image_embedding_size=72,
        image_size=1008,backbone_stride=14,_maybe_clone=lambda x:x,pred_obj_scores=True,use_obj_ptrs_in_encoder=True,
        use_no_obj_ptr=True,use_linear_no_obj_ptr=True,decode_mask_with_shared_tokens=False,stability_score_attentuation=False,
        object_score_logit_threshold=0.)
    for module,name in [(prompt,'sam_prompt_encoder'),(decoder,'sam_mask_decoder'),
                        (MLP(256,256,256,3),'obj_ptr_proj'),(torch.nn.Conv2d(1,1,4,stride=4),'mask_downsample')]:
        name=('interactive_' if multiplex else '')+name;prefix=root+name+'.'
        module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
        setattr(host,name,module.eval().to(device))
    if multiplex:
        linear=torch.nn.Linear(256,256);prefix=root+'no_obj_ptr_linear.'
        linear.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
        host.no_obj_ptr_linear=linear.eval().to(device);cls=VideoTrackingMultiplex
    else:
        host.no_obj_ptr=state[root+'no_obj_ptr'].to(device);cls=Sam3TrackerBase
    host._forward_sam_heads=MethodType(cls._forward_sam_heads,host)
    host._use_mask_as_output=MethodType(cls._use_mask_as_output,host)
    return host

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('library',type=Path);parser.add_argument('store',type=Path)
    parser.add_argument('--checkpoint',action='append',required=True);parser.add_argument('--device',default='cuda')
    parser.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();torch.ops.load_library(str(args.library.resolve()));torch.set_num_threads(4);torch.manual_seed(392)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    results=[]
    for specification in args.checkpoint:
        model,path=specification.split('=',1);state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);state=state.get('model',state)
        host=reference(state,model,args.device);multiplex=model=='sam3.1'
        for mode in args.modes:
            dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
            batch=2;image_batch=1 if multiplex else batch
            image=torch.randn(image_batch,256,72,72,device=args.device,dtype=dtype).contiguous(memory_format=torch.channels_last)
            high=[torch.randn(image_batch,c,s,s,device=args.device,dtype=dtype).contiguous(memory_format=torch.channels_last) for c,s in [(32,288),(64,144)]]
            # Both models exercise all selected mask tokens, empty point streams,
            # arbitrary-size and low-res masks, direct supplied masks, and absence.
            cases=[('points_single',9,None,False,False,0.),('points_multi',1,None,True,False,0.),
                   ('empty_points',0,None,True,False,0.),('mask_prompt',None,(1152,1152),False,False,0.),
                   ('combined_prompt',9,(288,288),True,False,0.),('non_square_prompt',None,(213,319),True,False,0.),
                   ('direct_mask',None,(1152,1152),False,True,0.),('direct_mask_image_size',None,(1008,1008),False,True,0.)]
            if multiplex:cases += [('forced_present',1,None,True,False,-1e6),('forced_absent',1,None,False,False,1e6)]
            else:cases += [('no_prompts',None,None,False,False,0.)]
            for name,count,mask_size,multi,direct,threshold in cases:
                points=torch.rand(batch,count,2,device=args.device)*1008 if count is not None else None
                labels=torch.randint(-1,4,(batch,count),device=args.device,dtype=torch.int32) if count is not None else None
                masks=None
                if mask_size:
                    masks=torch.randn(batch,1,*mask_size,device=args.device)
                    if direct:
                        masks=masks>0
                        masks[0].zero_()
                host.object_score_logit_threshold=threshold
                prompt=dict(point_coords=points,point_labels=labels) if points is not None else None
                with torch.autocast(args.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                    if multiplex:
                        if direct:expected=host._use_mask_as_output(image,high,masks,None,objects_in_mask=[0,1])
                        else:expected=host._forward_sam_heads(image,point_inputs=prompt,mask_inputs=masks,interactive_high_res_features=high,
                            multimask_output=multi,multiplex_state=None,objects_to_interact=[0,1])
                        expected=[expected[key] for key in KEYS]
                    elif direct:expected=host._use_mask_as_output(image,high,masks)
                    else:expected=host._forward_sam_heads(image,prompt,masks,high,multi)
                actual=torch.ops.sam3_native.video_interactive_heads(str(args.store),model,image,high,points,labels,masks,multi,direct,mode,threshold)
                errors={}
                for key,value,target in zip(KEYS,actual,expected):
                    torch.testing.assert_close(value,target,rtol=0,atol=0)
                    errors[key]=(value.float()-target.float()).abs().max().item()
                item=dict(model=model,mode=mode,case=name,mask_shape=list(actual[0].shape),max_errors=errors,
                    object_present=(actual[-1]>threshold).flatten().tolist())
                results.append(item);print(json.dumps(item),flush=True)
    args.report.write_text(json.dumps(dict(torch=torch.__version__,device=args.device,cases=results,
        scope='Original SAM3 per-object video heads and SAM3.1 interactive video heads. Does not validate temporal memory or multiplex propagation.'),indent=2)+'\n')
if __name__=='__main__':main()
