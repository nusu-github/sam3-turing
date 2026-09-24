"""High-level video folder preprocessing differs from low-level tracking."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from sam3.model.io_utils import _load_img_as_tensor

p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(1);rng=np.random.default_rng(38);rows=[]
for h,w in [(1,1),(1,9),(17,1),(5,7),(37,53),(1008,1008),(1080,1920),(2001,9)]:
 pixels=rng.integers(0,256,(h,w,3),dtype=np.uint8);ref=TF.to_tensor(TF.resize(Image.fromarray(pixels),(1008,1008))).half().sub_(.5).div_(.5).float()[None]
 x=torch.tensor(pixels).permute(2,0,1);actual=torch.ops.sam3_native.preprocess_video_rgb(x)
 torch.testing.assert_close(actual,ref,rtol=0,atol=0);rows.append(dict(height=h,width=w,exact=True))
for i in range(3):
 path=Path(f'assets/videos/0001/{i}.jpg');x,h,w=_load_img_as_tensor(str(path),1008);ref=x.half().sub_(.5).div_(.5).float()[None]
 actual=torch.ops.sam3_native.preprocess_video_rgb(torch.tensor(np.array(Image.open(path))).permute(2,0,1));torch.testing.assert_close(actual,ref,rtol=0,atol=0);rows.append(dict(path=str(path),height=h,width=w,exact=True))
a.report.write_text(json.dumps(dict(cases=rows,scope='Exact original high-level io_utils image-folder preprocessing: Pillow bilinear byte resize, F32 division, F16 storage/normalization. Native output losslessly widens normalized half values to F32. Separate from low-level Pillow bicubic/F32 tracking path and codec-specific resize paths.'),indent=2)+'\n');print(len(rows),'exact preprocessing cases')
