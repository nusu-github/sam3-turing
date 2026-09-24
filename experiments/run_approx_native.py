"""Fresh-process native approximation experiments, with per-run telemetry."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[2] if len(sys.argv) > 2 else '.cache/native-perf/approximation-v3')
root.mkdir(parents=True, exist_ok=True)
exe = Path('build/native-windows-cu130/sam3_image_latency.exe')
cases = {
    'truck': ('truck.ppm', 'prompt.txt'),
    'bag': ('groceries.ppm', 'bag.txt'),
    'child': ('test_image.ppm', 'child.txt'),
    'wheel': ('truck.ppm', 'wheel.txt'),
    'empty': ('truck.ppm', 'empty.txt'),
}
mode = sys.argv[1]
if mode == 'timing':
    jobs = [(name, 'truck', 5, 15) for name in ('exact','tanh','fused','int8','int8','fused','tanh','exact')]
elif mode == 'quality':
    jobs = [(name, case, 0, 1) for case in cases for name in ('exact','tanh','fused','int8')]
else:
    raise ValueError(mode)
counts = {}
for name, case, warmups, repeats in jobs:
    key = f'{mode}-{name}-{case}'
    counts[key] = counts.get(key, 0) + 1
    output = root / f'{key}-{counts[key]}'
    if output.exists():
        raise RuntimeError(f'refusing to overwrite {output}')
    env = os.environ.copy()
    env.pop('SAM3_EXPERIMENT_INT4_FC2',None)
    env.pop("SAM3_EXPERIMENT_INT8_LT",None)
    env.pop("SAM3_EXPERIMENT_BOUNDARY",None)
    env.pop("SAM3_EXPERIMENT_FC2_NORM",None)
    env.pop("SAM3_EXPERIMENT_QKV_ROPE",None)
    for key in ('SAM3_BENCH_CUDNN','SAM3_PROFILE_NVTX','SAM3_PROFILE_CAPTURE','SAM3_FUSED_NORM_CAST','SAM3_EXPERIMENT_PROJECTION','SAM3_EXPERIMENT_RESTORE','SAM3_EXPERIMENT_ATTENTION'):
        env.pop(key, None)
    env['SAM3_EXPERIMENT_MLP'] = name
    env['TORCH_BLAS_PREFER_CUBLASLT'] = '0'
    image, prompt = cases[case]
    command = [sys.executable, 'experiments/monitor_native_bench.py', str(output), str(exe),
               '.cache/native-weights-sam3', '.cache/native-perf/' + image,
               'sam3/assets/bpe_simple_vocab_16e6.txt.gz', '.cache/native-perf/' + prompt,
               str(warmups), str(repeats), str(output), '--profile']
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    (output / 'driver.log').write_text(proc.stdout + proc.stderr)
    (output / 'experiment.json').write_text(json.dumps({'mode': name, 'case': case}))
    if proc.returncode:
        print(proc.stdout + proc.stderr, flush=True)
        raise SystemExit(proc.returncode)
    metrics = json.loads((output / 'metrics.json').read_text())
    print(output.name, metrics['median_wall_seconds'] * 1000, 'ms', flush=True)
