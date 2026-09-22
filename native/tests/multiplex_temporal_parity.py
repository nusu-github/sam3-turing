"""SAM3.1 temporal selection, dual-stream assembly and memory conditioning."""
import argparse
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from sam3.model.multiplex_utils import MultiplexController, MultiplexState
from sam3.model.video_tracking_multiplex import VideoTrackingMultiplex
from sam3.model_builder import _create_multiplex_transformer
from memory_attention_parity import backend_context
from temporal_memory_parity import source_transfer_device


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('library', type=Path)
    parser.add_argument('store', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--modes', nargs='+', default=['fp32', 'fp16', 'bf16_reference'])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.ops.load_library(str(args.library.resolve()))
    torch.set_num_threads(4)
    torch.manual_seed(1171)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    weights = weights.get('model', weights)
    encoder = _create_multiplex_transformer().encoder.eval()
    prefix = 'tracker.model.transformer.encoder.'
    encoder.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}, strict=True)
    encoder.to(args.device)
    if args.device == 'cpu':
        for module in encoder.modules():
            if hasattr(module, 'compute_cis'):
                module.freqs_cis = module.compute_cis(end_x=72, end_y=72, device='cpu')
    projection = torch.nn.Linear(256, 256).eval()
    prefix = 'tracker.model.obj_ptr_tpos_proj.'
    projection.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}, strict=True)
    projection.to(args.device)
    capture = {}

    def record(**kwargs):
        capture.update(memory=kwargs['memory'].clone(), position=kwargs['memory_pos'].clone(),
                       image=kwargs['memory_image'].clone(), image_position=kwargs['memory_image_pos'].clone(),
                       pointer_tokens=kwargs['num_obj_ptr_tokens'], fuse=True)
        return encoder(**kwargs)

    host = SimpleNamespace(
        hidden_dim=256, mem_dim=256, training=False, save_image_features=True,
        sincos_tpos_enc=True, proj_tpos_enc_in_obj_ptrs=True,
        transformer=SimpleNamespace(encoder=record), obj_ptr_tpos_proj=projection,
        maskmem_tpos_enc=weights['tracker.model.maskmem_tpos_enc'].to(args.device), mf_threshold=.01)
    for name in ['_prepare_memory_conditioned_features', 'frame_filter']:
        setattr(host, name, MethodType(getattr(VideoTrackingMultiplex, name), host))

    def time_encoding(self, positions, *args_, **kwargs):
        capture['pointer_tokens'] = len(positions) * 16
        return VideoTrackingMultiplex._get_tpos_enc(self, positions, *args_, **kwargs)

    host._get_tpos_enc = MethodType(time_encoding, host)
    results = []
    for mode in args.modes:
        dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16_reference': torch.bfloat16}[mode]
        cases = [dict(name='forward'), dict(name='reverse_stride', reverse=True, stride=3, keep=True),
                 dict(name='past_signed', past=True, signed=True, limit=2, keep=True),
                 dict(name='score_filter', filtered=True, stride=2),
                 dict(name='reverse_score_filter', reverse=True, filtered=True, stride=2, pointer_limit=8),
                 dict(name='start', current=0, filtered=True), dict(name='reverse_end', current=23, reverse=True, filtered=True),
                 dict(name='v1_time', v2=False), dict(name='dummy_pointer_time', pointer_time=False),
                 dict(name='no_pointers', use_pointers=False), dict(name='missing_pointer_entries', missing='pointers'),
                 dict(name='missing_some_spatial', missing='some'), dict(name='cleared_memory', missing='all'),
                 dict(name='pointer_only_fallback', missing='spatial'), dict(name='zero_batch_memory', missing='zero'),
                 dict(name='legacy_5d', legacy=True), dict(name='unlimited_conditioning', limit=-1, pointer_limit=4),
                 dict(name='disabled_memory', slots=0, initial=True),
                 dict(name='full_grid', side=72, batch=1, current=2, slots=2, pointer_limit=3)]
        if args.device == 'cuda' and mode == 'fp16':
            cases.append(dict(name='math_full_grid', side=72, batch=1, current=2, slots=2, pointer_limit=3, math=True))
        for case in cases:
            side=case.get('side',8); batch=case.get('batch',2); current=case.get('current',12)
            reverse=case.get('reverse',False); slots=case.get('slots',7); limit=case.get('limit',4)
            pointer_limit=case.get('pointer_limit',16); stride=case.get('stride',1); keep=case.get('keep',False)
            filtered=case.get('filtered',False); past=case.get('past',False); signed=case.get('signed',False)
            v2=case.get('v2',True); pointer_time=case.get('pointer_time',True); use_pointers=case.get('use_pointers',True)
            initial=case.get('initial',False); missing=case.get('missing'); legacy=case.get('legacy',False)
            if legacy:
                buckets=MultiplexState([[0]+[-1]*15,[1]+[-1]*15],torch.device(args.device),torch.float32,16)
            else:
                buckets=MultiplexController(16).eval().get_state(1 if batch==1 else 17,torch.device(args.device),torch.float32,random=True)
            def rand(shape,device=args.device):return torch.randn(shape,device=device,dtype=dtype)
            source=rand((1,256,side,side)).flatten(2).permute(2,0,1)
            source_pos=rand((1,256,side,side)).flatten(2).permute(2,0,1)
            cond=[17,0,8,23,14,3,12] if side!=72 else [0]
            tracked=[i for i in range(24) if i not in cond and i%5!=0] if side!=72 else [1]
            ids=cond+tracked; conditioning=[True]*len(cond)+[False]*len(tracked)
            arrays=[[] for _ in range(6)]; output={'cond_frame_outputs':{},'non_cond_frame_outputs':{}}
            for index,is_cond in zip(ids,conditioning):
                storage='cpu' if index%2 else args.device
                shape=(batch,16,256,side,side) if legacy else (batch,256,side,side)
                feature=rand(shape,storage); position=rand(shape,storage)
                pointer=rand((batch,16,256)); image=rand((side*side,1,256),storage); image_position=rand((side*side,1,256),storage)
                score=None if index%4==0 else torch.tensor(float('nan') if index==7 else .01 if index%3==0 else .7 if index%2 else -.1,device=storage,dtype=dtype)
                if missing in ['all','spatial'] or (missing=='some' and index%3==0):feature=None
                if missing in ['all','zero','pointers']:pointer=None
                if missing=='zero':feature=rand((0,256,side,side),storage)
                entry=dict(maskmem_features=feature,maskmem_pos_enc=[position],image_features=image,image_pos_enc=image_position)
                if pointer is not None:entry['obj_ptr']=pointer
                if score is not None:entry['eff_iou_score']=score
                output['cond_frame_outputs' if is_cond else 'non_cond_frame_outputs'][index]=entry
                for array,tensor in zip(arrays,[feature,position,pointer,score,image,image_position]):array.append(torch.empty(0) if tensor is None else tensor)
            host.num_maskmem=slots;host.max_cond_frames_in_attn=limit;host.max_obj_ptrs_in_encoder=pointer_limit
            host.memory_temporal_stride_for_eval=stride;host.keep_first_cond_frame=keep;host.use_memory_selection=filtered
            host.only_obj_ptrs_in_the_past_for_eval=past;host.use_signed_tpos_enc_to_obj_ptrs=signed
            host.use_maskmem_tpos_v2=v2;host.add_tpos_enc_to_obj_ptrs=pointer_time;host.use_obj_ptrs_in_encoder=use_pointers
            math_only=case.get('math',False); adapted=(args.device=='cuda' and mode=='fp32') or math_only
            capture.clear();capture.update(pointer_tokens=0,fuse=False)
            with backend_context(adapted,math_only),source_transfer_device(args.device):
                with torch.autocast(args.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
                    expected=host._prepare_memory_conditioned_features(frame_idx=current,is_init_cond_frame=initial,
                        current_vision_feats=[source],current_vision_masks=[None],current_vision_pos_embeds=[source_pos],feat_sizes=[(side,side)],
                        output_dict=output,num_frames=24,track_in_reverse=reverse,use_prev_mem_frame=True,multiplex_state=buckets)
                settings=[side,side,current,24,slots,limit,pointer_limit,stride]
                flags=[initial,reverse,True,keep,filtered,past,signed,v2,pointer_time,use_pointers]
                backend_before=[torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.cudnn_sdp_enabled(),torch.backends.cuda.math_sdp_enabled()]
                actual=torch.ops.sam3_native.multiplex_temporal(str(args.store),source,source_pos,buckets.assignments,ids,conditioning,*arrays,settings,flags,.01,mode)
                assert backend_before==[torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.cudnn_sdp_enabled(),torch.backends.cuda.math_sdp_enabled()]
                if math_only:assert backend_before==[False,False,False,True]
            expected_dict=dict(features=expected,counts=torch.tensor([capture['pointer_tokens'],int(capture['fuse'])]))
            if capture['fuse']:
                for key in ['memory','position','image','image_position']:expected_dict[key]=capture[key]
            for entries in output.values():
                for index,entry in entries.items():
                    if entry['maskmem_features'] is not None:expected_dict['stored_features_'+str(index)]=entry['maskmem_features']
                    if entry['maskmem_pos_enc'][-1] is not None:expected_dict['stored_position_'+str(index)]=entry['maskmem_pos_enc'][-1]
            assert actual.keys()==expected_dict.keys()
            errors={}
            for key,value in actual.items():
                torch.testing.assert_close(value,expected_dict[key],rtol=0,atol=0)
                if not key.startswith('stored_'):errors[key]=(value.float()-expected_dict[key].float()).abs().max().item()
            row=dict(mode=mode,case=case['name'],grid=side,buckets=batch,pointer_tokens=capture['pointer_tokens'],fuse=capture['fuse'],
                     source_flash_only_context_removed=adapted,math_only_verified=math_only,max_errors=errors,exact=True)
            results.append(row);print(json.dumps(row),flush=True)
    args.report.write_text(json.dumps(dict(torch=torch.__version__,device=args.device,cases=results,
        scope='Original SAM3.1 inference temporal selection, spatial/image/pointer assembly, legacy 5D normalization and conditioned features. CUDA FP32/math reference removes only Flash-only context; CPU redirects hard-coded CUDA transfers and recomputes rotary caches on CPU. No complete video session or dynamic history rebucketing tested.'),indent=2)+'\n')


if __name__=='__main__':main()
