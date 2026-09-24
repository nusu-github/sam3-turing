"""Reproducible native calibration and quality runs with explicit split separation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from summarize_kitchen_extended import compare

DATA = Path('experiments/results/native_rtx2060/quant-research-data.json')
ROOT = Path('.cache/quant-research-20260924')
EXE = Path('build/native-windows-cu130/sam3_image_latency.exe')


def environment(mode, scope, calibration):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('SAM3_EXPERIMENT_') or key in ['SAM3_PROFILE_NVTX','SAM3_PROFILE_CAPTURE','SAM3_BENCH_CUDNN','SAM3_FUSED_NORM_CAST']:
            env.pop(key)
    env['TORCH_BLAS_PREFER_CUBLASLT'] = '0'
    # The same exact pixel conversion optimization applies to every candidate/reference.
    env['SAM3_EXPERIMENT_PIXEL'] = 'fused_nchw'
    if mode != 'fp16':
        env.update(SAM3_EXPERIMENT_ATTENTION='kitchen_all',SAM3_EXPERIMENT_KITCHEN_LAYOUT='sequence')
    if mode in ['selective','all','calibrated']:
        env.update(SAM3_EXPERIMENT_MLP='int8_boundary',SAM3_EXPERIMENT_PROJECTION='qkv',
                   SAM3_EXPERIMENT_QKV_ROPE='fused',SAM3_EXPERIMENT_FC2_NORM='fused',
                   SAM3_EXPERIMENT_MLP_SCOPE='global' if mode=='selective' else scope)
    if mode == 'calibrated':
        if calibration is None:
            raise ValueError('calibrated mode needs --calibration')
        env['SAM3_EXPERIMENT_FC2_CALIBRATION'] = str(calibration)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['observe','quality','timing'])
    parser.add_argument('--role', choices=['calibration','development','holdout'], default='development')
    parser.add_argument('--mode', choices=['fp16','attention','selective','all','calibrated'],default='fp16')
    parser.add_argument('--name', required=True)
    parser.add_argument('--scope', default='all',help='MLP scope for all/calibrated: all, firstN, lastN, global, local, mask:0xHEX')
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--limit-images',type=int)
    parser.add_argument('--primary-only',action='store_true')
    parser.add_argument('--image-id',type=int,help='Inspect one image from the selected split without changing its role')
    parser.add_argument('--attention',choices=['exact','kitchen','kitchen_all','kitchen_rot_all'],help='Explicit attention isolation override')
    parser.add_argument('--projection',choices=['exact','qkv'],help='Explicit QKV isolation override')
    parser.add_argument('--projection-scope',help='Quantized projection blocks: all, firstN, lastN, global, local, mask:0xHEX')
    parser.add_argument('--mlp-part',choices=['both','fc1','fc2'],help='Quantize only the selected MLP linear(s)')
    parser.add_argument('--mean-bias',action='store_true',help='Compensate FC2 weight quantization mean error using calibration data')
    args = parser.parse_args()
    if args.role=='holdout' and args.kind!='quality':
        parser.error('holdout is only for final quality evaluation')
    if args.kind=='observe' and (args.role!='calibration' or args.mode!='fp16'):
        parser.error('observations require calibration split and fp16')
    if args.mode=='fp16' and (args.attention is not None or args.projection is not None or args.mlp_part is not None):
        parser.error('FP16 reference and observations must not use quantization overrides')
    if args.mlp_part is not None and args.mode not in ['selective','all','calibrated']:
        parser.error('MLP part requires an MLP INT8 mode')
    if args.mlp_part=='fc1' and args.mode=='calibrated':
        parser.error('FC2 calibration cannot apply when only FC1 is quantized')
    if args.mean_bias and args.mode!='calibrated':
        parser.error('FC2 mean bias needs calibrated mode and fc2-mean.f32.bin')
    if args.scope!='all' and args.mode not in ['all','calibrated']:
        parser.error('explicit MLP scope requires all or calibrated mode')
    if args.projection_scope is not None and (args.mode=='fp16' or args.projection=='exact' or
            (args.mode=='attention' and args.projection!='qkv')):
        parser.error('projection scope requires a QKV INT8 mode')
    data=json.loads(DATA.read_text())
    images=[r for r in data['images'] if r['role']==args.role]
    if args.image_id is not None:
        images=[r for r in images if r['image_id']==args.image_id]
        if not images: parser.error('image ID not found in selected split')
    if args.limit_images: images=images[:args.limit_images]
    env=environment(args.mode,args.scope,args.calibration)
    if args.attention is not None: env['SAM3_EXPERIMENT_ATTENTION']=args.attention
    if args.mlp_part is not None: env['SAM3_EXPERIMENT_MLP_PART']=args.mlp_part
    if args.mean_bias: env['SAM3_EXPERIMENT_FC2_MEAN_BIAS']='enabled'
    if args.projection_scope is not None: env['SAM3_EXPERIMENT_PROJECTION_SCOPE']=args.projection_scope
    if args.projection is not None:
        env['SAM3_EXPERIMENT_PROJECTION']=args.projection
        env['SAM3_EXPERIMENT_QKV_ROPE']='fused' if args.projection=='qkv' else 'exact'
    root=ROOT/'runs'/args.name
    root.mkdir(parents=True,exist_ok=True)
    signature=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                   manifest_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                   binary_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(),
                   runtime_sha256=hashlib.sha256((EXE.parent/'sam3_native.dll').read_bytes()).hexdigest(),
                   environment={k:v for k,v in env.items() if k.startswith(('SAM3_','TORCH_BLAS'))})
    for optional in ['image_id','attention','projection','mlp_part','projection_scope']:
        if getattr(args,optional) is None: signature['args'].pop(optional)
    if not args.mean_bias: signature['args'].pop('mean_bias')
    if args.calibration:
        signature['calibration_sha256']=hashlib.sha256((args.calibration/'fc2-affine.f32.bin').read_bytes()).hexdigest()
        if args.mean_bias:
            signature['mean_sha256']=hashlib.sha256((args.calibration/'fc2-mean.f32.bin').read_bytes()).hexdigest()
    sigpath=root/'run.json'
    if sigpath.exists():
        if json.loads(sigpath.read_text())!=signature: raise RuntimeError('run signature changed; choose a new name')
    else: sigpath.write_text(json.dumps(signature,indent=2)+'\n')
    for row in images:
        assert hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()==row['sha256']
        cases=row['cases'][:1] if args.kind=='observe' or args.primary_only else row['cases']
        for case in cases:
            out=root/case['id']
            if (out/'complete.json').exists(): continue
            if out.exists(): raise RuntimeError(f'incomplete run requires inspection: {out}')
            assert Path(case['prompt_file']).read_text(encoding='utf-8')==case['prompt']
            ref=ROOT/'runs'/f'{args.role}-fp16'/case['id']
            if args.kind=='quality' and args.mode!='fp16' and not (ref/'complete.json').exists():
                raise RuntimeError(f'FP16 reference required before quality comparison: {ref}')
            local_env=env.copy()
            if args.kind=='observe':
                local_env['SAM3_EXPERIMENT_OBSERVE_FC2']=str(out/'observations')
            warmups,repeats=(5,15) if args.kind=='timing' else (0,1)
            cmd=[sys.executable,'experiments/monitor_native_bench.py',str(out),str(EXE),
                 '.cache/native-weights-sam3',row['image'],'sam3/assets/bpe_simple_vocab_16e6.txt.gz',
                 case['prompt_file'],str(warmups),str(repeats),str(out)]
            proc=subprocess.run(cmd,env=local_env,capture_output=True,text=True)
            (out/'driver.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
            if proc.returncode: raise RuntimeError(f'failed {out}; inspect driver.log')
            if args.kind=='observe':
                assert len(list((out/'observations').glob('fc2-*.f32.bin')))==32
            info=json.loads((out/'metrics.json').read_text())
            result=dict(case=case['id'],image_id=row['image_id'],mode=args.mode,count=info['count'])
            if args.kind=='quality' and args.mode!='fp16':
                result['comparison']=compare(ref,out)
            (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
    records=[json.loads(p.read_text()) for p in sorted(root.glob('*/complete.json'))]
    report=dict(signature=signature,results=records)
    destination=Path('experiments/results/native_rtx2060')/(args.name+'.json')
    destination.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
