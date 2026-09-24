"""Compare fresh processes for empirical QKV bias loading and both restore paths."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_quant_research import ROOT
from native_quality import compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    args = p.parse_args()
    common = ['--mode', 'calibrated', '--scope', 'mask:0xfffffffe',
              '--calibration', str(ROOT/'calibrations/a050-none-mean-bias'), '--mean-bias',
              '--attention-scope', 'mask:0xfffffffe', '--image-id', '1425', '--qkv-mean-bias']
    selected = ['--projection-scope', 'mask:0xffffffbf']
    excluded = ['--projection-scope', 'mask:0x00000000']
    initial = ['--qkv-calibration', str(ROOT/'calibrations/qkv-a050-mean')]
    identity = ['--qkv-calibration', str(ROOT/'calibrations/qkv-output-identity'), '--qkv-output-bias']
    corrected = ['--qkv-calibration', str(ROOT/'calibrations/qkv-output-mean'), '--qkv-output-bias']
    configs = {'base': selected + initial, 'identity': selected + identity,
               'excluded-base': excluded + initial, 'excluded-output': excluded + corrected,
               'output-fused': selected + corrected,
               'output-unfused': selected + corrected + ['--qkv-rope', 'exact']}
    for name, extra in configs.items():
        subprocess.run([sys.executable, 'experiments/run_quant_research.py', 'quality',
                        '--name', args.name+'-'+name, *common, *extra], check=True)
    pairs = [('r7-qkv-shift-development', args.name+'-base'),
             (args.name+'-base', args.name+'-identity'),
             (args.name+'-excluded-base', args.name+'-excluded-output'),
             (args.name+'-output-fused', args.name+'-output-unfused')]
    results = []
    for left, right in pairs:
        for prompt in range(2):
            case = f'coco-val-000000001425-p{prompt}'
            results.append(dict(reference=left, candidate=right, case=case,
                                comparison=compare(ROOT/'runs'/left/case, ROOT/'runs'/right/case)))
    dest = Path('experiments/results/native_rtx2060')/(args.name+'-summary.json')
    dest.write_text(json.dumps(dict(scope=__doc__, results=results), indent=2)+'\n')
    assert all(row['comparison']['byte_equal'] for row in results), 'output bias integration control failed'
    print('8/8 integration pairs byte equal', flush=True)


if __name__ == '__main__':
    main()
