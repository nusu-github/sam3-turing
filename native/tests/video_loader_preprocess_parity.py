"""C ABI preprocessing against the actual upstream video loader functions.

Decode is isolated by supplying identical RGB; this does not assert equivalence
of all codecs. TorchCodec transform is called without constructing its decoder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
import torch
from sam3.model import io_utils


def main():
    p=argparse.ArgumentParser();p.add_argument('probe',type=Path);p.add_argument('directory',type=Path);p.add_argument('--cuda',action='store_true');p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    root=a.directory.resolve();root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);rows=[];rng=np.random.default_rng(137)
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env['PATH']='/nonexistent'
    class Capture:
        def __init__(self,rgb):self.rgb=rgb;self.done=False
        def isOpened(self):return True
        def get(self,key):return {cv2.CAP_PROP_FRAME_HEIGHT:self.rgb.shape[0],cv2.CAP_PROP_FRAME_WIDTH:self.rgb.shape[1],cv2.CAP_PROP_FRAME_COUNT:1}[key]
        def read(self):
            if self.done:return False,None
            self.done=True;return True,self.rgb[:,:,::-1].copy()
        def release(self):pass
    sizes=[(1,1),(1,9),(17,1),(5,7),(37,53),(1008,1008),(1080,1920),(2001,9)]
    for n,(h,w) in enumerate(sizes):
        pixels=rng.integers(0,256,(h,w,3),dtype=np.uint8);raw=root/f'{n}.rgb';raw.write_bytes(pixels.tobytes());path=root/f'{n}.png';Image.fromarray(pixels).save(path)
        for policy in ([0,1,2,3,4] if a.cuda else [0,1,2,4]):
            device='cuda' if policy==3 else 'cpu';input=torch.from_numpy(pixels).permute(2,0,1)
            if policy==0:reference=io_utils.load_image_as_single_frame_video(str(path),1008,True)[0]
            elif policy==1:reference=io_utils.load_resource_as_video_frames([Image.fromarray(pixels)],1008,True)[0]
            elif policy in (2,3):
                cfg=SimpleNamespace(image_size=1008,img_mean=torch.full((3,1,1),.5,dtype=torch.float16,device=device),img_std=torch.full((3,1,1),.5,dtype=torch.float16,device=device),offload_video_to_cpu=policy==2,out_device=torch.device(device))
                reference=io_utils.AsyncVideoFileLoaderWithTorchCodec._transform_frame(cfg,input.contiguous().to(device))[None]
            else:
                with patch.object(cv2,'VideoCapture',return_value=Capture(pixels)):
                    reference=io_utils.load_video_frames_from_video_file_using_cv2('identical-decoded-rgb',1008,offload_video_to_cpu=True)[0]
            expected=reference.float().cpu().numpy();output=root/'output.f32'
            r=subprocess.run([str(a.probe.resolve()),str(raw),str(h),str(w),str(policy),device,str(output)],env=env,capture_output=True,text=True)
            assert r.returncode==0,r.stderr
            actual=np.fromfile(output,np.float32).reshape(1,3,1008,1008);error=np.abs(actual-expected)
            row=dict(height=h,width=w,policy=policy,device=device,exact=bool(np.array_equal(actual,expected)),different=int(np.count_nonzero(error)),max_abs=float(error.max()),min=float(actual.min()),max=float(actual.max()),native_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),reference_sha256=hashlib.sha256(expected.tobytes()).hexdigest());rows.append(row);print(json.dumps(row),flush=True)
            if not row['exact']:
                np.savez_compressed(root/f'mismatch-{n}-{policy}.npz',actual=actual,reference=expected)
    runtime=[]
    maps=Path('/proc/self/maps')
    if maps.exists():
        for path in sorted({line.split()[-1] for line in maps.read_text().splitlines() if any(name in line for name in ('libtorch_cpu.so','libc10.so','libtorch_python.so'))}):
            h=hashlib.sha256()
            with open(path,'rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            runtime.append(dict(path=path,sha256=h.hexdigest()))
    a.report.write_text(json.dumps(dict(reference_runtime=runtime,reference_torch_build=torch.__config__.show(),cases=rows,opencv=cv2.__version__,numpy=np.__version__,torch=torch.__version__,all_exact=all(x['exact'] for x in rows),scope='Actual source loader functions on identical decoded RGB; native C subprocess has neither Python nor ffmpeg executable on PATH. TorchCodec transform only, not its codec. Cv2Source preserves missing /255; existing image-folder API default unchanged.'),indent=2)+'\n')
    assert all(x['exact'] for x in rows)


if __name__=='__main__':main()
