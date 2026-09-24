"""Selective MLP quantization; screening on known failures, then held-out checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_kitchen_extended import ROOT as REFERENCE, CASES, environment
from summarize_kitchen_extended import compare

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('scopes',nargs='+',choices=['first8','first16','first24','last8','last16','last24','global','local'])
parser.add_argument('--all-cases',action='store_true')
args=parser.parse_args()
root=Path('.cache/native-perf/mlp-scope-ba018a0')
root.mkdir(parents=True,exist_ok=True)
cases=[c[0] for c in CASES] if args.all_cases else ['dance0-person','dance50-shirt','truck-window']
for case in cases:
    for scope in args.scopes:
        out=root/f'{case}-{scope}'
        if (out/'comparison.json').exists():continue
        if out.exists():raise RuntimeError(f'incomplete run: {out}')
        env=environment('kitchen_all')
        env.update(SAM3_EXPERIMENT_MLP_SCOPE=scope,SAM3_EXPERIMENT_KITCHEN_LAYOUT='sequence')
        cmd=[sys.executable,'experiments/monitor_native_bench.py',str(out),
             'build/native-windows-cu130/sam3_image_latency.exe','.cache/native-weights-sam3',
             str(REFERENCE/(case+'.ppm')),'sam3/assets/bpe_simple_vocab_16e6.txt.gz',
             str(REFERENCE/(case+'.txt')),'0','1',str(out)]
        result=subprocess.run(cmd,env=env,capture_output=True,text=True)
        (out/'driver.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(str(out))
        value=compare(REFERENCE/f'{case}-fp16',out)
        (out/'comparison.json').write_text(json.dumps(value,indent=2)+'\n')
        print(case,scope,json.dumps(value),flush=True)
report={p.parent.name:json.loads(p.read_text()) for p in sorted(root.glob('*/comparison.json'))}
Path('experiments/results/native_rtx2060/mlp-scope-ba018a0.json').write_text(json.dumps(report,indent=2)+'\n')
