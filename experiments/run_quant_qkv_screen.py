"""Isolate QKV channel scaling, shifting and weight mean bias on development cases.

The Round 5 candidate stays fixed: MLP/attention block 0 and QKV block 6 are
FP16, remaining selected paths are INT8, FC2 uses alpha .5 plus mean bias,
and attention uses anchor centering. An explicit attention override isolates
its interaction with QKV calibration. No holdout input is accepted here.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_quant_research import ROOT


def main():
    variants = {'identity_bias': ('qkv-identity', True),
                'scale': ('qkv-a050-none', False),
                'scale_bias': ('qkv-a050-none', True),
                'shift_bias': ('qkv-a050-mean', True)}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--variants', nargs='+', choices=list(variants), default=list(variants))
    p.add_argument('--images', nargs='+', type=int, default=[7574, 1425, 1584, 4795, 5992, 1353, 9590])
    p.add_argument('--attention', choices=['exact', 'kitchen', 'kitchen_all'], help='Explicit interaction test; default keeps Round 5 attention scope')
    args = p.parse_args()
    destination = Path('experiments/results/native_rtx2060')
    report = dict(scope=__doc__, variants=args.variants, images=args.images, attention=args.attention, results=[])
    for variant in args.variants:
        folder, bias = variants[variant]
        results = []
        for image_id in args.images:
            name = f'{args.name}-{variant}-{image_id}'
            cmd = [sys.executable, 'experiments/run_quant_research.py', 'quality', '--mode', 'calibrated',
                   '--scope', 'mask:0xfffffffe', '--calibration', str(ROOT / 'calibrations/a050-none-mean-bias'),
                   '--mean-bias', '--projection-scope', 'mask:0xffffffbf',
                   '--qkv-calibration', str(ROOT / 'calibrations' / folder), '--image-id', str(image_id), '--name', name]
            if args.attention is not None:
                cmd += ['--attention', args.attention]
            if args.attention != 'exact':
                cmd += ['--attention-scope', 'mask:0xfffffffe']
            if bias:
                cmd.append('--qkv-mean-bias')
            subprocess.run(cmd, check=True)
            results.extend(json.loads((destination / (name + '.json')).read_text())['results'])
        comparisons = [r['comparison'] for r in results]
        summary = dict(variant=variant, passed=sum(c['gate_pass'] for c in comparisons), total=len(comparisons),
                       nonempty=sum(c['reference_count'] > 0 for c in comparisons),
                       nonempty_passed=sum(c['reference_count'] > 0 and c['gate_pass'] for c in comparisons),
                       failed=[r for r in results if not r['comparison']['gate_pass']], results=results)
        report['results'].append(summary)
        (destination / (args.name + '-summary.json')).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: summary[k] for k in ['variant', 'passed', 'total', 'nonempty', 'nonempty_passed']}), flush=True)


if __name__ == '__main__':
    main()
