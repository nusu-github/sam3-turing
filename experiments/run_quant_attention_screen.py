"""Screen fixed attention masks on development images, with serial GPU runs.

All other precision choices and the FC2 calibration are fixed in the command.
The underlying runner guards run reuse by binary/data/calibration/env hashes.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from summarize_quant_research import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--scopes', nargs='+', required=True)
    p.add_argument('--images', nargs='+', type=int, default=[1584, 9590])
    p.add_argument('--mlp-scope', default='mask:0xfffffffe')
    p.add_argument('--projection-scope', default='mask:0xffffffbf')
    p.add_argument('--calibration', default='.cache/quant-research-20260924/calibrations/a050-none-mean-bias')
    args = p.parse_args()
    root = Path('.cache/quant-research-20260924/runs')
    reports = Path('experiments/results/native_rtx2060')
    results = []
    for scope in args.scopes:
        runs, cases = [], []
        label = scope.replace('mask:0x', 'm')
        if not label.isalnum():
            p.error('scope must use the native all/global/local/firstN/lastN/mask:0xHEX syntax')
        for image_id in args.images:
            name = f'{args.name}-{label}-{image_id}'
            cmd = [sys.executable, 'experiments/run_quant_research.py', 'quality',
                   '--mode', 'calibrated', '--scope', args.mlp_scope,
                   '--projection-scope', args.projection_scope, '--attention-scope', scope,
                   '--calibration', args.calibration, '--mean-bias', '--image-id', str(image_id), '--name', name]
            subprocess.run(cmd, check=True)
            runs.append(summarize(root / name))
            cases.extend(json.loads((reports / (name + '.json')).read_text())['results'])
        row = dict(attention_scope=scope, runs=runs, results=cases,
                   passed=sum(c['comparison']['gate_pass'] for c in cases), completed=len(cases))
        results.append(row)
        (reports / (args.name + '-summary.json')).write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({k: row[k] for k in ['attention_scope', 'passed', 'completed']}), flush=True)


if __name__ == '__main__':
    main()
