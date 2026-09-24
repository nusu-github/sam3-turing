"""Capture current FP16 / INT8 MLP inference in alternating order with Nsight."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(sys.argv[1] if len(sys.argv)>1 else '.cache/native-perf/hotspots-current')
root.mkdir(parents=True,exist_ok=True)
nsys='C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe'
counts={}
for mode in (sys.argv[2:] or ['exact','int8','int8','exact']):
    counts[mode]=counts.get(mode,0)+1
    name=f'{mode}-{counts[mode]}'
    out=root/name
    if out.exists(): raise RuntimeError(f'already exists: {out}')
    env=os.environ.copy()
    env.update(SAM3_PROFILE_NVTX='1',SAM3_PROFILE_CAPTURE='1',SAM3_EXPERIMENT_MLP=mode,TORCH_BLAS_PREFER_CUBLASLT='0')
    env.pop('SAM3_BENCH_CUDNN',None)
    cmd=[sys.executable,'experiments/monitor_native_bench.py',str(out),nsys,'profile',
         '--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi',
         '--capture-range-end=stop','--export=sqlite','-o',str(root/name),
         'build/native-windows-cu130/sam3_image_latency.exe','.cache/native-weights-sam3',
         '.cache/native-perf/truck.ppm','sam3/assets/bpe_simple_vocab_16e6.txt.gz',
         '.cache/native-perf/prompt.txt','5','5',str(out),'--profile']
    p=subprocess.run(cmd,env=env,capture_output=True,text=True)
    (out/'driver.log').write_text(p.stdout+p.stderr)
    if p.returncode:
        print(p.stdout+p.stderr,flush=True)
        raise SystemExit(p.returncode)
    m=json.loads((out/'metrics.json').read_text())
    print(name,m['median_wall_seconds']*1000,'ms',flush=True)
