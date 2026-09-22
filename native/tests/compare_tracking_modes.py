"""Measure precision-mode agreement from standalone tracking probe artifacts."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--reference',default='bf16_reference');p.add_argument('--modes',nargs='+',default=['fp16','fp32']);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    rows=[];files=sorted((a.directory/a.reference).glob('*.json'));assert files,'No reference outputs'
    for path in files:
        meta=json.loads(path.read_text());count=len(meta['ids']);pixels=meta['height']*meta['width'];name=path.stem
        def masks(mode):
            other=json.loads((a.directory/mode/path.name).read_text())
            for key in ['ids','height','width','frame','mask_row_bytes']:assert meta[key]==other[key]
            packed=np.fromfile(a.directory/mode/(name+'.masks.bin'),dtype=np.uint8).reshape(count,-1)
            return np.unpackbits(packed,axis=1,bitorder='little')[:,:pixels].astype(bool)
        reference=masks(a.reference);base=np.fromfile(a.directory/a.reference/(name+'.video.f32.bin'),dtype='<f4')
        for mode in a.modes:
            actual=masks(mode);intersection=(actual&reference).sum(1);union=(actual|reference).sum(1)
            logit=np.fromfile(a.directory/mode/(name+'.video.f32.bin'),dtype='<f4');assert logit.shape==base.shape
            rows.append(dict(mode=mode,output=name,frame=meta['frame'],ids=meta['ids'],
                mask_iou=np.divide(intersection,union,out=np.ones(count,dtype=float),where=union!=0).tolist(),
                different_mask_pixels=(actual!=reference).sum(1).tolist(),max_logit_error=float(np.abs(logit-base).max())))
    a.report.write_text(json.dumps(dict(reference=a.reference,cases=rows,
        scope='Precision-mode agreement on this frame/prompt fixture; not ground-truth segmentation accuracy or a dataset benchmark.'),indent=2)+'\n')
    for mode in a.modes:print(mode,'minimum mask IoU',min(v for row in rows if row['mode']==mode for v in row['mask_iou']))
if __name__=='__main__':main()
