"""Actual video frames through native backbone, session edits and propagation.

PPM fixtures carry the exact decoded JPEG pixels to the standalone process;
codec equivalence is intentionally not claimed. Source session methods and
visual backbone are evaluated separately, then their end-to-end outputs compared.
"""
import argparse,gc,json,os,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from sam3.model_builder import _create_vit_backbone
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.utils.sam2_utils import _load_img_as_tensor
from tracking_frame_parity import reference
from tracking_session_parity import run_reference

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('executable',type=Path);p.add_argument('store',type=Path);p.add_argument('checkpoint',type=Path)
    p.add_argument('--frames',nargs='+',type=Path,default=[Path(f'assets/videos/0001/{i}.jpg') for i in [0,1,2]])
    p.add_argument('--modes',nargs='+',default=['fp16','bf16_reference','fp32']);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1692);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    a.output.mkdir(parents=True,exist_ok=True);images=torch.zeros(len(a.frames),3,1008,1008);names=[];height=width=None
    for index,path in enumerate(a.frames):
        image,h,w=_load_img_as_tensor(str(path),1008);images[index]=image
        if height is not None:assert (height,width)==(h,w)
        height,width=h,w;name=f'{index}.ppm';Image.open(path).convert('RGB').save(a.output/name);names.append(name)
    (a.output/'frames.txt').write_text('\n'.join(names)+'\n');images=images.cuda().sub_(.5).div_(.5)
    # Prompts are fixture choices, not fixed in the native executable/API.
    points=torch.tensor([[.6,.68,1.],[.25,.3,0.]])
    box=torch.tensor([.79,.13,.99,.94]);refine=torch.tensor([[.63,.7,1.]])
    blank=torch.empty(0)
    operations=[[0,0,11,1,1,0,0,0],[1,0,22,1,1,0,0,0],[3,0,0,1,0,0,0,0],
                [4,0,len(a.frames)-1,0,1,0,0,0],[0,1,11,1,1,1,0,0],
                [4,len(a.frames)-1,len(a.frames)-1,1,1,1,0,0],[6,0,22,0,0,0,0,0],[7,0,0,0,0,0,0,0]]
    payloads=[points,box,blank,blank,refine,blank,blank,blank]
    commands=['points 0 11 1 1 0 .6 .68 1 .25 .3 0','box 0 22 1 .79 .13 .99 .94','preflight 1',
              f'propagate 0 {len(a.frames)-1} 0 1 0 0 0','points 1 11 1 1 1 .63 .7 1',
              f'propagate {len(a.frames)-1} {len(a.frames)-1} 1 1 1 0 0','remove 22','reset']
    (a.output/'commands.txt').write_text('\n'.join(commands)+'\n')
    weights=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True);weights=weights.get('model',weights)
    neck=Sam3DualViTDetNeck(trunk=_create_vit_backbone(),position_encoding=PositionEmbeddingSine(256),d_model=256,scale_factors=[4.,2.,1.,.5],add_sam2_neck=True).eval()
    prefix='detector.backbone.vision_backbone.';neck.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
    host=reference(weights,'cuda');original=[block.mlp.forward for block in neck.trunk.blocks];results=[]
    for mode in a.modes:
        print(f'START {mode}',flush=True);neck.cuda();arrays=[[],[],[],[]]
        for block,fn in zip(neck.trunk.blocks,original):block.mlp.forward=fn if mode=='bf16_reference' else lambda x,m=block.mlp:m.fc2(m.act(m.fc1(x)))
        dtype=torch.float16 if mode=='fp16' else torch.bfloat16
        with torch.autocast('cuda',enabled=mode!='fp32',dtype=dtype):
            for frame in images:
                out=neck(frame[None]);pyramid=[getattr(x,'tensors',x) for x in out[2]][:3]
                values=[pyramid[2],out[3][2],host.sam_mask_decoder.conv_s0(pyramid[0]),host.sam_mask_decoder.conv_s1(pyramid[1])]
                for array,value in zip(arrays,values):array.append(value)
            expected=run_reference(host,arrays,operations,payloads,[len(a.frames),height,width,0,0,7,16],
                                   [True,False,True,False,True,False,False,False],mode,'cuda')
        # Free the reference visual model's GPU allocation before the native
        # process loads its weights; both paths still run full-resolution ViT.
        neck.cpu();del arrays,out,pyramid,values,value;gc.collect();torch.cuda.empty_cache()
        target=a.output/mode;started=time.perf_counter()
        command=[str(a.executable.resolve()),str(a.store.resolve()),'cuda',mode,str((a.output/'frames.txt').resolve()),str((a.output/'commands.txt').resolve()),str(target.resolve())]
        child=subprocess.run(command,env={**os.environ,'PATH':'/nonexistent'},capture_output=True,text=True)
        (a.output/(mode+'.log')).write_text(child.stdout+child.stderr);assert child.returncode==0,child.stdout+child.stderr
        elapsed=time.perf_counter()-started
        for i in range(len(operations)):
            for j in range(expected[f'{i}/outputs'].item()):
                key=f'{i}/out{j}/';name=f'{i}-{j}';metadata=json.loads((target/(name+'.json')).read_text())
                assert metadata['ids']==expected[key+'ids'].tolist() and metadata['frame']==expected[key+'frame'].item()
                masks=expected[key+'masks'];shape=tuple(masks.shape)
                actual=np.fromfile(target/(name+'.video.f32.bin'),dtype='<f4').reshape(shape)
                np.testing.assert_array_equal(actual,masks.numpy())
                packed=np.fromfile(target/(name+'.masks.bin'),dtype=np.uint8).reshape(shape[0],-1)
                bits=np.unpackbits(packed,axis=1,bitorder='little')[:,:height*width].reshape(shape)
                np.testing.assert_array_equal(bits,masks.numpy()>0)
                if key+'low' in expected:
                    low=expected[key+'low'];np.testing.assert_array_equal(np.fromfile(target/(name+'.low.f32.bin'),dtype='<f4').reshape(low.shape),low.numpy())
                    np.testing.assert_array_equal(np.asarray(metadata['object_logits'],dtype=np.float32),expected[key+'logits'].flatten().float().numpy())
                row=dict(mode=mode,operation=i,output=j,frame=metadata['frame'],objects=len(metadata['ids']),positive_pixels=(masks>0).flatten(1).sum(1).tolist(),exact=True)
                results.append(row);print(json.dumps(row),flush=True)
        torch.save({k:v for k,v in expected.items() if '/out' in k},a.output/(mode+'-reference.pt'))
        print(f'{mode} native process seconds={elapsed:.3f}',flush=True);del expected;gc.collect()
    a.report.write_text(json.dumps(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),frames=[str(x) for x in a.frames],height=height,width=width,cases=results,
        source_adaptations='FP16/FP32 ordinary MLP replaces hard-coded BF16 fused MLP; FP32 restores compressed temporal memory to float; D2H copies finish before CPU consumption.',
        scope='Real decoded frames through native preprocessing, full visual backbone, interactive SAM3 session and temporal tracking. Byte-identical decoded pixels are passed via PPM; compressed video decoding, high-level text tracking and SAM3.1 sessions are separate.'),indent=2)+'\n')
if __name__=='__main__':main()
