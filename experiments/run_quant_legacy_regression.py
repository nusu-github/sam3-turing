"""Compare a recorded candidate and fresh FP16 on the original 17 regression cases.

These familiar images/prompts are development regressions, never a holdout.
No candidate settings are selected from these results by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from native_quality import compare, regression_inputs
from run_quant_research import EXE, ROOT, environment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-run', required=True, help='Completed development run name')
    p.add_argument('--name', required=True)
    args = p.parse_args()
    source = ROOT / 'runs' / args.candidate_run
    candidate = json.loads((source / 'run.json').read_text())
    if candidate['args']['role'] != 'development' or candidate['args']['kind'] != 'quality':
        p.error('candidate must come from a development quality run')
    source_report = json.loads((Path('experiments/results/native_rtx2060') / (args.candidate_run + '.json')).read_text())
    assert source_report['signature'] == candidate and source_report['results'], 'candidate report mismatch'
    for row in source_report['results']:
        assert json.loads((source / row['case'] / 'complete.json').read_text()) == row, 'candidate case incomplete'
    assert candidate['binary_sha256'] == sha(EXE), 'candidate executable changed'
    assert candidate['runtime_sha256'] == sha(EXE.parent / 'sam3_native.dll'), 'candidate DLL changed'
    if 'qkv_calibration_sha256' in candidate:
        qkv_cal = Path(candidate['args']['qkv_calibration'])
        assert candidate['qkv_calibration_sha256'] == sha(qkv_cal / 'qkv-affine.f32.bin')
        if 'qkv_mean_sha256' in candidate:
            assert candidate['qkv_mean_sha256'] == sha(qkv_cal / 'qkv-mean.f32.bin')
        if 'qkv_output_bias_sha256' in candidate:
            assert candidate['qkv_output_bias_sha256'] == sha(qkv_cal / 'qkv-output-bias.f32.bin')
    if 'calibration_sha256' in candidate:
        cal = Path(candidate['args']['calibration'])
        assert candidate['calibration_sha256'] == sha(cal / 'fc2-affine.f32.bin')
        if 'mean_sha256' in candidate:
            assert candidate['mean_sha256'] == sha(cal / 'fc2-mean.f32.bin')
    inputs = regression_inputs()
    signature = dict(candidate_run=args.candidate_run, candidate=candidate,
                     weight_manifest_sha256=sha(Path('.cache/native-weights-sam3/manifest.json')),
                     scope='17 known regression cases; fresh FP16 and candidate; cold quality only',
                     inputs=[dict(case=n, image=str(i), image_sha256=sha(i), prompt=str(t),
                                  prompt_sha256=sha(t)) for n, i, t in inputs])
    root = ROOT / 'legacy' / args.name
    root.mkdir(parents=True, exist_ok=True)
    sig = root / 'run.json'
    if sig.exists():
        assert json.loads(sig.read_text()) == signature, 'run signature changed; use a new name'
    else:
        sig.write_text(json.dumps(signature, indent=2) + '\n')
    results = []
    for name, image, prompt in inputs:
        for mode in ['fp16', 'candidate']:
            out = root / name / mode
            if (out / 'complete.json').exists():
                continue
            if out.exists():
                raise RuntimeError(f'incomplete run requires inspection: {out}')
            env = environment('fp16', 'all', None)
            if mode == 'candidate':
                env.update(candidate['environment'])
            cmd = [sys.executable, 'experiments/monitor_native_bench.py', str(out), str(EXE),
                   '.cache/native-weights-sam3', str(image), 'sam3/assets/bpe_simple_vocab_16e6.txt.gz',
                   str(prompt), '0', '1', str(out)]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            (out / 'driver.log').write_text(proc.stdout + proc.stderr, encoding='utf-8')
            if proc.returncode:
                raise RuntimeError(f'failed {out}; inspect driver.log')
            (out / 'complete.json').write_text(json.dumps(dict(case=name, mode=mode)) + '\n')
        row = dict(case=name, comparison=compare(root / name / 'fp16', root / name / 'candidate'))
        results.append(row)
        report = dict(signature=signature, results=results, completed=len(results),
                      passed=sum(c['comparison']['gate_pass'] for c in results))
        (Path('experiments/results/native_rtx2060') / (args.name + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
