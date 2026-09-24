"""Synthetic attention operator cost, including centering and quantization.

Uses the existing benchmark's head-major output, not the model's sequence
layout. CUDA-event medians here are not end-to-end wall latency or acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from run_quant_research import ROOT, environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    args = p.parse_args()
    exe = Path('build/native-windows-cu130/sam3_kitchen_bench.exe')
    destination = Path('experiments/results/native_rtx2060')
    report = dict(scope=__doc__, binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  runtime_sha256=hashlib.sha256((exe.parent / 'sam3_native.dll').read_bytes()).hexdigest(),
                  order=['anchor', 'mean_half', 'mean_half', 'anchor'], runs=[])
    for trial, mode in enumerate(report['order']):
        name = f'{args.name}-p{trial}-{mode}'
        output = ROOT / 'center-micro' / name
        result_file = destination / (name + '.json')
        if output.exists() or result_file.exists():
            raise RuntimeError('refusing to overwrite benchmark; inspect the previous run')
        env = environment('fp16', 'all', None)
        env['SAM3_EXPERIMENT_KITCHEN_CENTER'] = mode
        proc = subprocess.run([sys.executable, 'experiments/monitor_native_bench.py', str(output),
                               str(exe), str(result_file)], env=env, capture_output=True, text=True)
        (output / 'driver.log').write_text(proc.stdout + proc.stderr, encoding='utf-8')
        if proc.returncode:
            raise RuntimeError(f'benchmark failed: {output}')
        result = json.loads(result_file.read_text())
        telemetry = json.loads((output / 'telemetry.json').read_text())
        row = dict(mode=mode, trial=trial, file=str(result_file), samples=telemetry['samples'],
                   process_seconds=telemetry['wall_seconds'], timings=[])
        for case in result['cases']:
            if not case['kitchen_ms']:
                continue
            assert len(case['kitchen_ms']) == len(case['sdpa_ms']) == 4
            row['timings'].append(dict(batch=case['batch'], length=case['length'],
                                       sdpa_ms=case['sdpa_ms'], kitchen_ms=case['kitchen_ms'],
                                       kitchen_rot_ms=case['kitchen_rot_ms'],
                                       median_of_four_pass_medians_ms=statistics.median(case['kitchen_ms'])))
        report['runs'].append(row)
        (destination / (args.name + '-protocol.json')).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: row[k] for k in ['mode', 'trial', 'process_seconds', 'timings']}), flush=True)


if __name__ == '__main__':
    main()
