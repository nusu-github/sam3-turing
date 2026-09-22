"""Byte-exact Pillow RGB resizing and original JPEG-sequence normalization."""
import argparse,json
from pathlib import Path
import numpy as np
import PIL
from PIL import Image
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('--report',type=Path,required=True);p.add_argument('--cuda',action='store_true');a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);rng=np.random.default_rng(1691);cases=[]
    for height,width in [(1,1),(1,37),(29,1),(3,5),(31,47),(137,193),(720,1280),(1200,1800),(1008,1008)]:
        pixels=rng.integers(0,256,(height,width,3),dtype=np.uint8)
        image=Image.fromarray(pixels);tensor=torch.from_numpy(pixels).permute(2,0,1)
        for h,w in [(1,1),(7,9),(32,31),(height,width),(1008,1008)]:
            expected=torch.from_numpy(np.array(image.resize((w,h)))).permute(2,0,1)
            actual=torch.ops.sam3_native.resize_tracking_rgb(tensor,h,w)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            cases.append(dict(input=[height,width],output=[h,w],exact=True))
        expected=torch.from_numpy(np.array(image.resize((1008,1008)))/255.).permute(2,0,1).float().contiguous().sub_(.5).div_(.5)[None]
        for source in ([tensor,tensor.cuda()] if a.cuda else [tensor]):
            actual=torch.ops.sam3_native.preprocess_tracking_rgb(source)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0);assert actual.stride()==expected.stride()
        np.testing.assert_array_equal(tensor.permute(1,2,0).numpy(),pixels)
    for filename in ['assets/videos/0001/0.jpg','assets/videos/0001/1.jpg','assets/images/truck.jpg']:
        image=Image.open(filename).convert('RGB');tensor=torch.from_numpy(np.array(image)).permute(2,0,1)
        expected=torch.from_numpy(np.array(image.resize((1008,1008)))/255.).permute(2,0,1).float().contiguous().sub_(.5).div_(.5)[None]
        actual=torch.ops.sam3_native.preprocess_tracking_rgb(tensor);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        cases.append(dict(image=filename,preprocess_exact=True))
    a.report.write_text(json.dumps(dict(torch=torch.__version__,pillow=PIL.__version__,cuda_input=a.cuda,cases=cases,
        scope='Decoded RGB to original synchronous JPEG-frame preprocessing; does not compare JPEG or video bitstream decoding.'),indent=2)+'\n')
    print(f'{len(cases)} exact resize/real-frame cases; 9 normalization/stride cases',flush=True)
if __name__=='__main__':main()
