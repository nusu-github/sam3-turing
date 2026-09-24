"""Diagnose the three previously failing cases; not an independent validation set."""
import json
from pathlib import Path
import subprocess
import sys
from run_kitchen_extended import ROOT as REFERENCE, environment
from summarize_kitchen_extended import compare

root=Path('.cache/native-perf/precision-isolation-ba018a0')
root.mkdir(parents=True,exist_ok=True)
report={}
for case in ['dance0-person','dance50-shirt','truck-window']:
    for mode in ['mlp_only','qkv_only']:
        out=root/f'{case}-{mode}'
        env=environment('fp16')
        if mode=='mlp_only':
            env.update(SAM3_EXPERIMENT_MLP='int8_boundary',SAM3_EXPERIMENT_FC2_NORM='fused')
        else:
            env.update(SAM3_EXPERIMENT_PROJECTION='qkv',SAM3_EXPERIMENT_QKV_ROPE='fused')
        if out.exists():raise RuntimeError(f'refusing overwrite: {out}')
        cmd=[sys.executable,'experiments/monitor_native_bench.py',str(out),
             'build/native-windows-cu130/sam3_image_latency.exe','.cache/native-weights-sam3',
             str(REFERENCE/(case+'.ppm')),'sam3/assets/bpe_simple_vocab_16e6.txt.gz',
             str(REFERENCE/(case+'.txt')),'0','1',str(out)]
        result=subprocess.run(cmd,env=env,capture_output=True,text=True)
        (out/'driver.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(str(out))
        value=compare(REFERENCE/f'{case}-fp16',out)
        report[f'{case}/{mode}']=value
        Path('experiments/results/native_rtx2060/precision-isolation-ba018a0.json').write_text(json.dumps(report,indent=2)+'\n')
        print(case,mode,json.dumps(value),flush=True)
