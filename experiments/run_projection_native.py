"""Compare INT8 attention projections on top of fused-boundary INT8 MLP."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path('.cache/native-perf/projection-ba018a0')
root.mkdir(parents=True,exist_ok=True)
cases={'truck':('truck.ppm','prompt.txt'),'bag':('groceries.ppm','bag.txt'),
       'child':('test_image.ppm','child.txt'),'wheel':('truck.ppm','wheel.txt'),
       'empty':('truck.ppm','empty.txt')}
kind=sys.argv[1]
if kind=='timing':
    jobs=[(m,'truck',5,15) for m in ['exact','qkv','proj','both','both','proj','qkv','exact']]
elif kind=='quality':
    jobs=[(m,c,0,1) for c in cases for m in ['exact','qkv','proj','both']]
else: raise ValueError(kind)
counts={}
for mode,case,warmups,repeats in jobs:
    key=f'{kind}-{mode}-{case}'
    counts[key]=counts.get(key,0)+1
    out=root/f'{key}-{counts[key]}'
    if out.exists():raise RuntimeError(f'refusing overwrite: {out}')
    env=os.environ.copy()
    env.pop('SAM3_EXPERIMENT_INT4_FC2',None)
    env.pop("SAM3_EXPERIMENT_INT8_LT",None)
    env.pop("SAM3_EXPERIMENT_BOUNDARY",None)
    env.pop("SAM3_EXPERIMENT_FC2_NORM",None)
    env.pop("SAM3_EXPERIMENT_QKV_ROPE",None)
    for k in ['SAM3_PROFILE_NVTX','SAM3_PROFILE_CAPTURE','SAM3_BENCH_CUDNN','SAM3_FUSED_NORM_CAST']:
        env.pop(k,None)
    env.update(SAM3_EXPERIMENT_MLP='int8_boundary',SAM3_EXPERIMENT_PROJECTION=mode,SAM3_EXPERIMENT_RESTORE='flat',SAM3_EXPERIMENT_ATTENTION='exact',TORCH_BLAS_PREFER_CUBLASLT='0')
    image,prompt=cases[case]
    cmd=[sys.executable,'experiments/monitor_native_bench.py',str(out),
         'build/native-windows-cu130/sam3_image_latency.exe','.cache/native-weights-sam3',
         '.cache/native-perf/'+image,'sam3/assets/bpe_simple_vocab_16e6.txt.gz',
         '.cache/native-perf/'+prompt,str(warmups),str(repeats),str(out),'--profile']
    p=subprocess.run(cmd,env=env,capture_output=True,text=True)
    (out/'driver.log').write_text(p.stdout+p.stderr)
    (out/'experiment.json').write_text(json.dumps({'mode':mode,'case':case,'base':'ba018a0'}))
    if p.returncode:
        print(p.stdout+p.stderr,flush=True);raise SystemExit(p.returncode)
    metrics=json.loads((out/'metrics.json').read_text())
    print(out.name,metrics['median_wall_seconds']*1000,'ms',flush=True)
