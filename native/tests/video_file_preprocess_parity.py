"""Actual OpenCV file loader versus native FFmpeg decode plus Cv2Source policy."""
import argparse,json,os,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
from sam3.model.io_utils import load_video_frames_from_video_file_using_cv2

p=argparse.ArgumentParser();p.add_argument('decoder',type=Path);p.add_argument('preprocessor',type=Path);p.add_argument('directory',type=Path);p.add_argument('--files',nargs='+',type=Path,required=True);p.add_argument('--color-policy',type=int,choices=[0,1],default=1);p.add_argument('--report',type=Path,required=True);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env['PATH']='/nonexistent';env['SAM3_PROBE_COLOR_POLICY']=str(a.color_policy);rows=[]
for path in a.files:
 out=a.directory/path.name;out.mkdir(exist_ok=True)
 r=subprocess.run([str(a.decoder.resolve()),str(path.resolve()),'video',str(out)],env=env,capture_output=True,text=True);(out/'decoder.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 reference,h,w=load_video_frames_from_video_file_using_cv2(str(path),1008,offload_video_to_cpu=True);count=int(np.fromfile(out/'info-frames.bin',np.int64)[0]);assert count==len(reference)
 for i in range(count):
  meta=json.loads((out/f'frame-{i}.json').read_text());shape=meta['rgb']['shape'];rgb=np.fromfile(out/f'frame-{i}-rgb.bin',np.uint8).reshape(shape).transpose(1,2,0);raw=out/'input.rgb';raw.write_bytes(rgb.tobytes());result=out/'image.f32'
  r=subprocess.run([str(a.preprocessor.resolve()),str(raw.resolve()),str(shape[1]),str(shape[2]),'4','cpu',str(result.resolve())],env=env,capture_output=True,text=True);assert r.returncode==0,r.stderr
  actual=np.fromfile(result,np.float32).reshape(3,1008,1008);expected=reference[i].numpy();error=abs(actual-expected);row=dict(file=path.name,frame=i,native_dimensions=shape[1:],source_dimensions=[h,w],exact=bool(np.array_equal(actual,expected)),different=int(np.count_nonzero(error)),max_abs=float(error.max()),native_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),reference_sha256=hashlib.sha256(expected.tobytes()).hexdigest());rows.append(row);print(json.dumps(row),flush=True)
  if not row['exact']:np.savez_compressed(out/f'mismatch-{i}.npz',native=actual,reference=expected)
a.report.write_text(json.dumps(dict(color_policy=a.color_policy,cases=rows,all_exact=all(x['exact'] for x in rows),scope='Actual complete file preprocessing versus unmodified upstream OpenCV loader; specific codecs only. No model or all-codec claim.'),indent=2)+'\n');assert all(x['exact'] for x in rows)
