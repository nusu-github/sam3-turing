"""Create explicit FC2 affine candidates from calibration-only native observations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('observations',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--shift',choices=['none','mean','midrange'],default='none')
    p.add_argument('--identity',action='store_true')
    p.add_argument('--mean-bias',action='store_true',help='Save calibration-only input means for weight error compensation')
    args=p.parse_args()
    if not 0<=args.alpha<=1: p.error('alpha must be between zero and one')
    run=json.loads((args.observations/'run.json').read_text())
    assert run['args']['kind']=='observe' and run['args']['role']=='calibration' and run['args']['mode']=='fp16'
    cases=sorted(d.parent for d in args.observations.glob('*/complete.json'))
    assert cases, 'no complete calibration images'
    transformed=[]
    means=[]
    summaries=[]
    hashes={}
    for layer in range(32):
        values=[]
        for case in cases:
            path=case/'observations'/f'fc2-{layer}.f32.bin'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            values.append(np.fromfile(path,dtype='<f4').reshape(5,4736))
        stats=np.stack(values)
        lo=stats[:,0].min(0)
        hi=stats[:,1].max(0)
        mean=stats[:,2].mean(0)
        means.append(mean)
        wmax=stats[0,4]
        assert np.all(stats[:,4]==wmax) and np.isfinite(stats).all()
        center={'none':np.zeros_like(mean),'mean':mean,'midrange':(lo+hi)/2}[args.shift]
        magnitude=np.maximum(np.abs(lo-center),np.abs(hi-center)).clip(1e-5)
        # SmoothQuant-like balancing, normalized globally to avoid arbitrary scale.
        r=magnitude**args.alpha / wmax.clip(1e-5)**(1-args.alpha)
        r=r/np.exp(np.log(r).mean())
        r=np.clip(r,1/16,16).astype(np.float32)
        shift=(center/r).astype(np.float32)
        if args.identity: r[:]=1;shift[:]=0
        transformed.append(np.stack([r,shift]))
        summaries.append(dict(layer=layer,min_scale=float(r.min()),max_scale=float(r.max()),
                              max_abs_shift=float(np.abs(shift).max()),
                              max_activation=float(np.maximum(np.abs(lo),np.abs(hi)).max())))
    args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'fc2-affine.f32.bin'
    if target.exists(): raise RuntimeError('refusing to overwrite calibration')
    np.stack(transformed).astype('<f4').tofile(target)
    report=dict(alpha=args.alpha,shift=args.shift,identity=args.identity,images=len(cases),
                observations_run=run,observation_sha256=hashes,
                file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),layers=summaries,
                rounding='GELU FP16 -> affine FP32 division/subtraction -> FP16 -> symmetric row INT8. Weight transformed in FP32 then stored FP16; bias correction uses transformed FP16 weight.')
    if args.mean_bias:
        mean_target=args.output/'fc2-mean.f32.bin'
        if mean_target.exists(): raise RuntimeError('refusing to overwrite calibration means')
        np.stack(means).astype('<f4').tofile(mean_target)
        report['mean_sha256']=hashlib.sha256(mean_target.read_bytes()).hexdigest()
        report['mean_bias_formula']='bias + W_original @ mean - W_dequantized @ (mean / r - shift); FP32 setup, bias stored FP16. Excludes activation quantization error and upstream propagation.'
    (args.output/'calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['alpha','shift','identity','images','file_sha256']}))


if __name__=='__main__':
    main()
