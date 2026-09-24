"""Pair a recorded development candidate with the fixed selective INT8 baseline.

Four images and two fresh processes per configuration/image by default.
Reverse both image and configuration order on the second pass. This measures
speed; independent quality acceptance is still a separate requirement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run_quant_research import DATA, EXE, ROOT, environment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-run', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--images', nargs='+', type=int, default=[7574, 1425, 1584, 9590])
    p.add_argument('--preheat-repeats', type=int, default=180)
    args = p.parse_args()
    if len(set(args.images)) < 2 or len(set(args.images)) != len(args.images):
        p.error('at least two distinct development images are required')
    if args.preheat_repeats < 0:
        p.error('preheat repeats must be nonnegative')
    source = json.loads((ROOT / 'runs' / args.candidate_run / 'run.json').read_text())
    if source['args']['kind'] != 'quality' or source['args']['role'] != 'development' or source['args']['mode'] != 'calibrated':
        p.error('source must be a recorded calibrated development quality candidate')
    assert source['binary_sha256'] == sha(EXE) and source['runtime_sha256'] == sha(EXE.parent / 'sam3_native.dll'), 'candidate binary changed'
    assert source['manifest_sha256'] == sha(DATA), 'candidate data changed'
    cal = Path(source['args']['calibration'])
    assert source['calibration_sha256'] == sha(cal / 'fc2-affine.f32.bin')
    if 'mean_sha256' in source:
        assert source['mean_sha256'] == sha(cal / 'fc2-mean.f32.bin')
    report_path = Path('experiments/results/native_rtx2060')
    completed = json.loads((report_path / (args.candidate_run + '.json')).read_text())
    assert completed['signature'] == source and completed['results'], 'source quality report is incomplete'
    # Keep every precision argument; only replace run controls below.
    controls = {'kind', 'role', 'name', 'limit_images', 'primary_only', 'image_id'}
    candidate_args = []
    for key, value in source['args'].items():
        if key in controls or value is None or value is False:
            continue
        candidate_args.append('--' + key.replace('_', '-'))
        if value is not True:
            candidate_args.append(str(value))
    configs = {'selective': ['--mode', 'selective'], 'candidate': candidate_args}
    data = {r['image_id']: r for r in json.loads(DATA.read_text())['images'] if r['role'] == 'development'}
    if any(i not in data for i in args.images):
        p.error('timing images must belong to the development split')
    warm = ROOT / 'timing-warmups' / args.name
    if warm.exists():
        raise RuntimeError('series already started; inspect it before choosing a new name')
    warm.mkdir(parents=True)
    row = data[args.images[0]]
    assert sha(Path(row['image'])) == row['sha256']
    cmd = [sys.executable, 'experiments/monitor_native_bench.py', str(warm), str(EXE),
           '.cache/native-weights-sam3', row['image'], 'sam3/assets/bpe_simple_vocab_16e6.txt.gz',
           row['cases'][0]['prompt_file'], str(args.preheat_repeats), '5', str(warm)]
    print('Preheating fixed selective baseline', flush=True)
    proc = subprocess.run(cmd, env=environment('selective', 'all', None), capture_output=True, text=True)
    (warm / 'driver.log').write_text(proc.stdout + proc.stderr, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'preheating failed; inspect {warm}')
    schedule, runs = [], {key: [] for key in configs}
    for trial in range(2):
        for image_id in (args.images if trial == 0 else list(reversed(args.images))):
            for mode in (list(configs) if trial == 0 else list(reversed(configs))):
                name = f'{args.name}-{mode}-{image_id}-p{trial}'
                subprocess.run([sys.executable, 'experiments/run_quant_research.py', 'timing',
                                '--name', name, '--image-id', str(image_id), '--primary-only', *configs[mode]], check=True)
                actual = json.loads((ROOT / 'runs' / name / 'run.json').read_text())
                if mode == 'candidate':
                    assert actual['environment'] == source['environment'], 'candidate environment changed'
                runs[mode].append(str(ROOT / 'runs' / name))
                schedule.append(dict(mode=mode, image_id=image_id, trial=trial, run=name))
    subprocess.run([sys.executable, 'experiments/summarize_quant_timing.py', *runs['selective'], *runs['candidate'],
                    '--output', str(report_path / (args.name + '-summary.json'))], check=True)
    report = dict(candidate_run=args.candidate_run, source_signature=source, images=args.images,
                  preheat_repeats=args.preheat_repeats, warmup_telemetry=json.loads((warm / 'telemetry.json').read_text()),
                  schedule=schedule, runs=runs,
                  protocol='Two fresh processes/config/image; 5 warmups + 15 timed samples. Reverse image and config order on pass 2. Fixed selective preheat. Speed qualification is separate from independent quality acceptance.')
    (report_path / (args.name + '-protocol.json')).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
