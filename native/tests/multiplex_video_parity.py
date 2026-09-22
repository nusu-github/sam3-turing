"""Real decoded frames through the standalone SAM3.1 C++ visual/session path.

PPM fixes the decoded pixels for both implementations. Original demo staging
and annotation bookkeeping repairs are inherited from multiplex_session_parity.
This tests same-mode numerical parity, not ground-truth segmentation quality.
"""
import argparse,gc,json,os,subprocess,time
from pathlib import Path
from types import MethodType
import numpy as np
from PIL import Image
import torch
from sam3.model_builder import _create_vit_backbone
from sam3.model.necks import Sam3TriViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.utils.sam2_utils import _load_img_as_tensor
from multiplex_session_parity import session_reference,layout_staging,restore_annotation_indices
from memory_attention_parity import backend_context


def reference_features(host,arrays):
    def features(self,state,index,batch):
        values=arrays[index];out={}
        for name,offset in [('interactive',0),('sam2_backbone_out',4)]:
            image,position,h0,h1=values[offset:offset+4]
            out[name]=dict(vision_feats=[x.flatten(2).permute(2,0,1) for x in [h0,h1,image]],vision_masks=[None]*3,
                vision_pos_embeds=[None,None,position.flatten(2).permute(2,0,1)],feat_sizes=[(288,288),(144,144),(72,72)])
        return None,out
    host._get_image_feature=MethodType(features,host)


def run_reference(host,mode,height,width,count,case,masks):
    expected={};state=host.init_state(video_height=height,video_width=width,num_frames=count,offload_state_to_cpu=True)
    def save(operation,number,result):
        frame,ids,low,video=result
        expected[f'{operation}-{number}']=dict(frame=frame,ids=list(ids),low=None if low is None else low.cpu().clone(),video=video.cpu().clone())
    if case=='points':
        point=torch.tensor([[.6,.68],[.25,.3]])
        save(0,0,host.add_new_points(state,0,11,point,torch.tensor([1,0]),True,rel_coordinates=True))
    else:save(0,0,host.add_new_masks(state,0,[11,22],masks))
    host.propagate_in_video_preflight(state,True)
    for number,value in enumerate(host.propagate_in_video(state,0,count-1,False,tqdm_disable=True)):save(2,number,value)
    if case=='points':
        save(3,0,host.add_new_points(state,1,11,torch.tensor([[.63,.7]]),torch.tensor([1]),True,rel_coordinates=True))
        restore_annotation_indices(state)
        host.propagate_in_video_preflight(state,True)
        for number,value in enumerate(host.propagate_in_video(state,count-1,count-1,True,tqdm_disable=True)):save(5,number,value)
    torch.cuda.synchronize()
    return expected


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('executable',type=Path);p.add_argument('store',type=Path);p.add_argument('checkpoint',type=Path)
    p.add_argument('--frames',nargs='+',type=Path,default=[Path(f'assets/videos/0001/{i}.jpg') for i in [0,1,2]])
    p.add_argument('--modes',nargs='+',default=['fp16','bf16_reference','fp32']);p.add_argument('--cases',nargs='+',default=['points','masks'],choices=['points','masks'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(3481);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    assert len(a.frames)>=2
    a.output.mkdir(parents=True,exist_ok=True);images=torch.zeros(len(a.frames),3,1008,1008);names=[];height=width=None
    for index,path in enumerate(a.frames):
        image,h,w=_load_img_as_tensor(str(path),1008);images[index]=image
        if height is not None:assert (height,width)==(h,w)
        height,width=h,w;name=f'{index}.ppm';Image.open(path).convert('RGB').save(a.output/name);names.append(name)
    (a.output/'frames.txt').write_text('\n'.join(names)+'\n');images=images.cuda().sub_(.5).div_(.5)
    masks=torch.zeros(2,height,width);masks[0,int(.5*height):int(.86*height),int(.46*width):int(.73*width)]=1
    masks[1,int(.13*height):int(.94*height),int(.79*width):int(.99*width)]=1
    for index,mask in enumerate(masks):Image.fromarray((mask.numpy()*255).astype(np.uint8)).convert('RGB').save(a.output/f'mask-{index}.ppm')
    scripts={'points':['points 0 11 1 1 0 .6 .68 1 .25 .3 0','preflight 1',f'propagate 0 {len(a.frames)-1} 0 1 0 0 0',
                        'points 1 11 1 1 0 .63 .7 1','preflight 1',f'propagate {len(a.frames)-1} {len(a.frames)-1} 1 1 0 0 0','reset'],
             'masks':['masks 0 11 mask-0.ppm 22 mask-1.ppm','preflight 1',f'propagate 0 {len(a.frames)-1} 0 1 0 0 0','reset']}
    for name,lines in scripts.items():(a.output/f'{name}.txt').write_text('\n'.join(lines)+'\n')
    weights=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True);weights=weights.get('model',weights)
    neck=Sam3TriViTDetNeck(trunk=_create_vit_backbone(),position_encoding=PositionEmbeddingSine(256),d_model=256).eval()
    prefix='detector.backbone.vision_backbone.';neck.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
    host=session_reference(weights,'cuda');host.num_maskmem=7;host.max_obj_ptrs_in_encoder=16;host.max_cond_frames_in_attn=4
    host.non_overlap_masks_for_output=True;host.use_memory_selection=False;host.non_overlap_masks_for_mem_enc=False
    original=[block.mlp.forward for block in neck.trunk.blocks];encoder=host.transformer.encoder;original_encoder=encoder.forward
    results=[];timings=[]
    for mode in a.modes:
        print('START',mode,flush=True);neck.cuda();arrays=[]
        for block,fn in zip(neck.trunk.blocks,original):block.mlp.forward=fn if mode=='bf16_reference' else lambda x,m=block.mlp:m.fc2(m.act(m.fc1(x)))
        def attention(**kwargs):
            if mode=='fp32':kwargs['memory']=kwargs['memory'].float()
            return original_encoder(**kwargs)
        encoder.forward=attention
        dtype=torch.float16 if mode=='fp16' else torch.bfloat16
        with torch.autocast('cuda',enabled=mode!='fp32',dtype=dtype):
            for frame in images:
                out=neck(frame[None],need_sam3_out=False);values=[]
                for offset,decoder in [(2,host.interactive_sam_mask_decoder),(4,host.sam_mask_decoder)]:
                    pyramid=[x.tensors for x in out[offset]]
                    values.extend([pyramid[2],out[offset+1][2],decoder.conv_s0(pyramid[0]),decoder.conv_s1(pyramid[1])])
                arrays.append(values)
            reference_features(host,arrays)
            with backend_context(mode=='fp32',False),layout_staging(mode):
                expected={case:run_reference(host,mode,height,width,len(a.frames),case,masks) for case in a.cases}
        # Release source visual weights and features before the standalone child.
        neck.cpu();host._get_image_feature=None;del arrays,out,pyramid,values;gc.collect();torch.cuda.empty_cache()
        for case,outputs in expected.items():
            target=a.output/f'{mode}-{case}';started=time.perf_counter()
            command=[str(a.executable.resolve()),str(a.store.resolve()),'cuda',mode,str((a.output/'frames.txt').resolve()),str((a.output/f'{case}.txt').resolve()),str(target.resolve())]
            child=subprocess.run(command,env={**os.environ,'PATH':'/nonexistent'},capture_output=True,text=True)
            (a.output/f'{mode}-{case}.log').write_text(child.stdout+child.stderr);assert child.returncode==0,child.stdout+child.stderr
            elapsed=time.perf_counter()-started;timings.append(dict(mode=mode,case=case,process_seconds=elapsed))
            assert {p.stem for p in target.glob('*.json')}==set(outputs)
            for name,value in outputs.items():
                metadata=json.loads((target/(name+'.json')).read_text());assert metadata['ids']==value['ids'] and metadata['frame']==value['frame']
                masks_out=value['video'];shape=tuple(masks_out.shape)
                actual=np.fromfile(target/(name+'.video.f32.bin'),dtype='<f4').reshape(shape)
                try:np.testing.assert_array_equal(actual,masks_out.numpy())
                except AssertionError:
                    torch.save(value,a.output/f'{mode}-{case}-{name}-failure.pt');raise
                packed=np.fromfile(target/(name+'.masks.bin'),dtype=np.uint8).reshape(shape[0],-1)
                bits=np.unpackbits(packed,axis=1,bitorder='little')[:,:height*width].reshape(shape)
                np.testing.assert_array_equal(bits,masks_out.numpy()>0)
                if value['low'] is not None:
                    low=value['low'];np.testing.assert_array_equal(np.fromfile(target/(name+'.low.f32.bin'),dtype='<f4').reshape(low.shape),low.float().numpy())
                row=dict(mode=mode,case=case,output=name,frame=metadata['frame'],objects=len(metadata['ids']),positive_pixels=(masks_out>0).flatten(1).sum(1).tolist(),exact=True)
                results.append(row);print(json.dumps(row),flush=True)
            torch.save(outputs,a.output/f'{mode}-{case}-reference.pt')
            print(f'{mode} {case} process seconds={elapsed:.3f}',flush=True)
        del expected;gc.collect()
    a.report.write_text(json.dumps(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),frames=[str(x) for x in a.frames],height=height,width=width,cases=results,timings=timings,
        source_adaptations='FP16/FP32 ordinary MLP instead of forced BF16 fused MLP. FP32 memory restored to float and Flash-only restriction removed. D2H completion, offloaded mux staging, and post-refinement annotation indices repaired as documented in multiplex_session_parity.',
        scope='Exact same-mode real decoded RGB through full native shared ViT, both projected necks, session edits, memory and forward/reverse propagation. PATH=/nonexistent for native executable. PPM bypasses compressed codecs; no Turing/Windows runtime or ground-truth accuracy claim.'),indent=2)+'\n')
if __name__=='__main__':main()
