"""Measure attention-only approximation against an otherwise FP16 model."""
import json
from pathlib import Path
import subprocess
import sys
from run_kitchen_extended import environment

root=Path('.cache/native-perf/kitchen-precision-ba018a0')
root.mkdir(parents=True,exist_ok=True)
counts={}
for mode in ['fp16','attention_only','attention_only','fp16']:
    counts[mode]=counts.get(mode,0)+1
    out=root/f'{mode}-{counts[mode]}'
    if out.exists():raise RuntimeError(f'refusing overwrite: {out}')
    env=environment(mode)
    env['SAM3_EXPERIMENT_KITCHEN_LAYOUT']='sequence'
    cmd=[sys.executable,'experiments/monitor_native_bench.py',str(out),
         'build/native-windows-cu130/sam3_image_latency.exe','.cache/native-weights-sam3',
         '.cache/native-perf/truck.ppm','sam3/assets/bpe_simple_vocab_16e6.txt.gz',
         '.cache/native-perf/prompt.txt','5','15',str(out),'--profile']
    result=subprocess.run(cmd,env=env,capture_output=True,text=True)
    (out/'driver.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(f'failed: {out}')
    print(mode,counts[mode],json.loads((out/'metrics.json').read_text())['median_wall_seconds']*1000,flush=True)
