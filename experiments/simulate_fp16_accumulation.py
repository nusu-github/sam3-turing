"""Estimate GEMM rounding error of FP16 accumulation, two-level promotion and INT8 on CPU.

Synthetic, data-free inputs shaped like SAM3 vision GEMMs. The tensor-core model follows
Fasi et al. (PeerJ CS 2021): products are exact, each block of products is summed with the
accumulator at high precision and rounded once to the accumulator format (RN for FP16).
Errors are relative RMS errors against the unrounded float64 product.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

METHODS=['fp32 acc (current s1688)','fp16 acc, block 4','fp16 acc, block 8',
    'two-level 4/32','two-level 4/64','two-level 4/128','two-level 4/256',
    'INT8 row x channel','INT8 + block Hadamard','official BF16 autocast']
CASES={
    'qkv':'QKV/FC1-like: LN output with 8 outlier channels, K=1024',
    'fc2':'FC2-like: GELU output, K=4736',
    'pv':'PV-like: exp(logits-max) x V, K=5184',
}


def f16(x): return x.astype(np.float16).astype(np.float64)
def f32(x): return x.astype(np.float32).astype(np.float64)


def bf16(x):
    bits=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits=((bits+0x7FFF+((bits>>16)&1))&0xFFFF0000).astype(np.uint32)
    return bits.view(np.float32).astype(np.float64)


def blocks(a,w,block,rounding):
    acc=np.zeros((a.shape[0],w.shape[0]))
    for k in range(0,a.shape[1],block):
        acc=rounding(acc+a[:,k:k+block]@w[:,k:k+block].T)
    return acc


def two_level(a,w,block,interval):
    acc=np.zeros((a.shape[0],w.shape[0]))
    for k in range(0,a.shape[1],interval):
        acc=f32(acc+blocks(a[:,k:k+interval],w[:,k:k+interval],block,f16))
    return f16(acc)


def int8(a,w):
    sa=np.abs(a).max(1,keepdims=True)/127
    sw=np.abs(w).max(1,keepdims=True)/127
    return f16((np.clip(np.rint(a/sa),-127,127)@np.clip(np.rint(w/sw),-127,127).T)*sa*sw.T)


def hadamard(n):
    h=np.ones((1,1))
    while h.shape[0]<n: h=np.block([[h,h],[h,-h]])
    return h/math.sqrt(n)


def rotated_int8(a,w,group,rng):
    signs=rng.choice([-1.0,1.0],a.shape[1])
    h=hadamard(group)
    rotate=lambda x:((x*signs).reshape(x.shape[0],-1,group)@h).reshape(x.shape)
    return int8(f16(rotate(a)),f16(rotate(w)))


def inputs(case,rng,rows,cols):
    if case=='qkv':
        k=1024
        x=rng.standard_normal((rows,k))
        x[:,rng.choice(k,8,replace=False)]*=25
        return x,rng.standard_normal((cols,k))/math.sqrt(k),128
    if case=='fc2':
        k=4736
        x=rng.standard_normal((rows,k))*1.5
        erf=np.frompyfunc(math.erf,1,1)
        return .5*x*(1+erf(x/math.sqrt(2)).astype(np.float64)),rng.standard_normal((cols,k))/math.sqrt(k),128
    k=5184
    logits=rng.standard_normal((rows,k))*3
    return np.exp(logits-logits.max(1,keepdims=True)),rng.standard_normal((cols,k)),64


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,default=4)
    p.add_argument('--rows',type=int,default=64)
    p.add_argument('--cols',type=int,default=64)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    report={}
    for case,title in CASES.items():
        errors={m:[] for m in METHODS}
        for seed in range(args.seeds):
            rng=np.random.default_rng(seed)
            raw_a,raw_w,group=inputs(case,rng,args.rows,args.cols)
            true=raw_a@raw_w.T
            rms=math.sqrt((true**2).mean())
            a,w=f16(raw_a),f16(raw_w)
            outputs=[f16(blocks(a,w,4,f32)),blocks(a,w,4,f16),blocks(a,w,8,f16)]
            outputs+=[two_level(a,w,4,n) for n in (32,64,128,256)]
            outputs+=[int8(a,w),rotated_int8(a,w,group,rng),bf16(blocks(bf16(raw_a),bf16(raw_w),4,f32))]
            for method,out in zip(METHODS,outputs):
                errors[method].append(math.sqrt(((out-true)**2).mean())/rms)
        base=np.mean(errors[METHODS[0]])
        report[case]={m:{'relative_rms_error':float(np.mean(v)),'vs_current':float(np.mean(v)/base)} for m,v in errors.items()}
        print(f'\n## {title}\n')
        print('| Method | Relative RMS error | x current |\n|---|---:|---:|')
        for m,v in report[case].items():
            print(f"| {m} | {v['relative_rms_error']:.2e} | {v['vs_current']:.1f} |")
    if args.output:
        args.output.write_text(json.dumps({'args':{k:str(v) for k,v in vars(args).items()},'cases':report},indent=2)+'\n')


if __name__=='__main__':
    main()
