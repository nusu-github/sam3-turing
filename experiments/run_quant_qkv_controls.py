"""Model integration controls for QKV identity, excluded layers and both restore paths."""
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
              '--calibration', str(ROOT / 'calibrations/a050-none-mean-bias'), '--mean-bias',
              '--attention-scope', 'mask:0xfffffffe', '--image-id', '1425']
    shifted = ['--qkv-calibration', str(ROOT / 'calibrations/qkv-a050-mean'), '--qkv-mean-bias']
    selected = ['--projection-scope', 'mask:0xffffffbf']
    excluded = ['--projection-scope', 'mask:0x00000000']
    configs = {'base': selected,
               'identity': selected + ['--qkv-calibration', str(ROOT / 'calibrations/qkv-identity')],
               'excluded-base': excluded,
               'excluded-shift': excluded + shifted,
               'shift-fused': selected + shifted,
               'shift-unfused': selected + shifted + ['--qkv-rope', 'exact']}
    for mode, extra in configs.items():
        subprocess.run([sys.executable, 'experiments/run_quant_research.py', 'quality',
                        '--name', args.name + '-' + mode, *common, *extra], check=True)
    pairs = [('r5-a0-development', args.name + '-base'),
             (args.name + '-base', args.name + '-identity'),
             (args.name + '-excluded-base', args.name + '-excluded-shift'),
             (args.name + '-shift-fused', args.name + '-shift-unfused')]
    comparisons = []
    for left, right in pairs:
        for prompt in range(2):
            case = f'coco-val-000000001425-p{prompt}'
            comparisons.append(dict(reference=left, candidate=right, case=case,
                                    comparison=compare(ROOT / 'runs' / left / case, ROOT / 'runs' / right / case)))
    path = Path('experiments/results/native_rtx2060') / (args.name + '-summary.json')
    path.write_text(json.dumps(dict(scope=__doc__, results=comparisons), indent=2) + '\n')
    print(json.dumps(comparisons, indent=2))
    assert all(r['comparison']['byte_equal'] for r in comparisons), 'integration control changed output bytes'


if __name__ == '__main__':
    main()
