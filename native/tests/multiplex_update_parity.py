"""Compare dynamic SAM3.1 mask insertion/reconditioning with original methods."""
import argparse,copy,json
from pathlib import Path
from types import MethodType
import torch
from sam3.model.video_tracking_multiplex import VideoTrackingDynamicMultiplex
from sam3.model.multiplex_utils import MultiplexState
from multiplex_frame_parity import reference
from memory_attention_parity import backend_context

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('library',type=Path);parser.add_argument('store',type=Path);parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--device',default='cuda');parser.add_argument('--modes',nargs='+',default=['fp16','fp32','bf16_reference']);parser.add_argument('--cases',nargs='+');parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    torch.ops.load_library(str(args.library.resolve()));torch.set_num_threads(4);torch.manual_seed(9173)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    weights=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True);host=reference(weights.get('model',weights),args.device)
    for name in ['add_new_masks_to_existing_state','recondition_masks_in_existing_state']:setattr(host,name,MethodType(getattr(VideoTrackingDynamicMultiplex,name),host))
    cases=[dict(name='append',append=True),dict(name='append_grow',append=True,old=15,new=3,grow=True),
           dict(name='append_prefer_new',append=True,grow=True,prefer=True),dict(name='append_removed_slot',append=True,removed=True),
           dict(name='append_capacity',append=True,capacity=4,new=1),dict(name='append_resize',append=True,resolution=1152),
           dict(name='append_low_resize',append=True,low=252),dict(name='append_deferred',append=True,encode=False,no_high=True),
           dict(name='append_ids_input_masks',append=True,ids=True,input_masks=True),dict(name='append_score_overlap',append=True,score=True,overlap=True,points=True),
           dict(name='recondition',append=False),dict(name='recondition_low_resize',append=False,low=252),
           dict(name='recondition_deferred',append=False,encode=False,no_high=True),dict(name='recondition_ids_input_masks',append=False,ids=True,input_masks=True),
           dict(name='recondition_score_overlap',append=False,score=True,overlap=True)]
    rows=[]
    for mode in args.modes:
        dtype={'fp16':torch.float16,'fp32':torch.float32,'bf16_reference':torch.bfloat16}[mode]
        for case in cases:
            if args.cases and case['name'] not in args.cases:continue
            print('START',args.device,mode,case['name'],flush=True)
            count=case.get('old',3);new=case.get('new',2);append=case['append'];resolution=case.get('resolution',1008);low=case.get('low',288);capacity=case.get('capacity',16)
            assignments=[list(range(count))+([-1116] if case.get('removed') else [])];assignments[0]+=[-1]*(16-len(assignments[0]));assignments[0]=assignments[0][::-1]
            old_ids=[100+index*17 for index in range(count)] if case.get('ids') else None
            state=MultiplexState(copy.deepcopy(assignments),device=torch.device(args.device),dtype=torch.float32,allowed_bucket_capacity=capacity,object_ids=copy.deepcopy(old_ids))
            def rand(shape,kind=dtype):return torch.randn(shape,device=args.device,dtype=kind)
            features=[rand((1,256,72,72)).contiguous(memory_format=torch.channels_last),rand((1,32,288,288)).contiguous(memory_format=torch.channels_last),
                      rand((1,64,144,144)).contiguous(memory_format=torch.channels_last),rand((1,256,72,72)).contiguous(memory_format=torch.channels_last)]
            masks=torch.rand(new,1,1008,1008,device=args.device)>.5;masks[0].zero_()
            indices=list(range(count,count+new)) if append else [count-1,0]
            object_ids=([2000+index for index in range(new)] if append else [old_ids[index] for index in indices]) if old_ids else None
            with torch.autocast(args.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):pointer=state.mux(rand((count,256)))
            previous=dict(pred_masks=rand((count,1,low,low),torch.float32),object_score_logits=rand((count,1),torch.float32),obj_ptr=pointer,
                maskmem_features=rand((1,256,72,72)),maskmem_pos_enc=rand((1,256,72,72)),
                image_features=features[3].flatten(2).permute(2,0,1),image_pos_enc=rand((5184,1,256)),
                candidates=rand((count,3,low,low)),eff_iou_score=rand((count,count),torch.float32))
            if not case.get('no_high'):previous['pred_masks_high_res']=rand((count,1,resolution,resolution),torch.float32)
            if case.get('score'):previous['iou_score']=rand((count,),torch.float32)
            if case.get('input_masks'):previous['input_masks']=torch.rand(count,1,1008,1008,device=args.device)>.5
            expected={key:value.clone() for key,value in previous.items()};expected['maskmem_pos_enc']=[expected['maskmem_pos_enc']];expected['conditioning_objects']={1}
            host.use_memory_selection=case.get('score',False);host.non_overlap_masks_for_mem_enc=case.get('overlap',False);host.save_image_features=True
            encode=case.get('encode',True)
            with backend_context(args.device=='cuda' and mode=='fp32',False):
                with torch.autocast(args.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                    interactive=(features[0].flatten(2).permute(2,0,1)+host.interactivity_no_mem_embed).permute(1,2,0).view(1,256,72,72)
                    kwargs=dict(interactive_pix_feat=interactive,interactive_high_res_features=features[1:3],propagation_vision_feats=[features[3].flatten(2).permute(2,0,1)],propagation_feat_sizes=[(72,72)],
                        new_masks=masks,obj_idxs_in_mask=indices,obj_ids_in_mask=object_ids,prev_output=expected,multiplex_state=state,add_mask_to_memory=encode)
                    if append:host.add_new_masks_to_existing_state(**kwargs,are_masks_from_pts=case.get('points',False),allow_new_buckets=case.get('grow',False),prefer_new_buckets=case.get('prefer',False))
                    else:host.recondition_masks_in_existing_state(**kwargs)
                flags=[append,encode,case.get('points',False),case.get('grow',False),case.get('prefer',False),case.get('score',False),case.get('overlap',False),True]
                actual=torch.ops.sam3_native.multiplex_update(str(args.store),features,masks,indices,object_ids,assignments,old_ids,capacity,previous,[1],flags,mode)
            expected['maskmem_pos_enc']=expected['maskmem_pos_enc'][-1];expected['conditioning_objects']=torch.tensor(sorted(expected['conditioning_objects']))
            expected['assignments']=torch.tensor(state.assignments);expected['affected']=torch.tensor(indices)
            if old_ids is not None:expected['object_ids']=torch.tensor(state.object_ids)
            assert actual.keys()==expected.keys(),(actual.keys(),expected.keys())
            for key,value in actual.items():
                try:torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
                except AssertionError as error:raise AssertionError(f'{mode} {case["name"]} {key}: {error}') from error
            rows.append(dict(mode=mode,case=case['name'],objects=state.total_valid_entries,buckets=state.num_buckets,outputs=len(actual),exact=True));print(json.dumps(rows[-1]),flush=True)
    args.report.write_text(json.dumps(dict(torch=torch.__version__,device=args.device,cases=rows,scope='Original dynamic mask add/recondition methods with full-grid synthetic shared image features. Current frame only; old history remapping and full demo session remain separate.'),indent=2)+'\n')
if __name__=='__main__':main()
