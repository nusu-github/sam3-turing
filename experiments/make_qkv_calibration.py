"""Build QKV norm folding from all 32 frozen calibration images, never development."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_quant_research import DATA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('observations', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--shift', choices=['none', 'mean', 'midrange'], default='none')
    p.add_argument('--identity', action='store_true')
    args = p.parse_args()
    if not 0 <= args.alpha <= 1:
        p.error('alpha must be between zero and one')
    source = json.loads((args.observations / 'run.json').read_text())
    assert source['args']['kind'] == 'observe' and source['args']['role'] == 'calibration'
    assert source['args']['mode'] == 'fp16' and source['args']['observe_target'] == 'qkv'
    assert source['manifest_sha256'] == hashlib.sha256(DATA.read_bytes()).hexdigest()
    cases = sorted(d.parent for d in args.observations.glob('*/complete.json'))
    observed_ids = [json.loads((c / 'complete.json').read_text())['image_id'] for c in cases]
    required = {r['image_id'] for r in json.loads(DATA.read_text())['images'] if r['role'] == 'calibration'}
    assert len(cases) == len(required) == 32 and set(observed_ids) == required
    assert not args.output.exists(), 'refusing to overwrite QKV calibration directory'
    transformed, means, layers, hashes = [], [], [], {}
    for layer in range(32):
        values = []
        for case in cases:
            path = case / 'observations' / f'qkv-{layer}.f32.bin'
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            values.append(np.fromfile(path, dtype='<f4').reshape(5, 1024))
        stats = np.stack(values)
        assert np.isfinite(stats).all()
        lo, hi, mean, wmax = stats[:, 0].min(0), stats[:, 1].max(0), stats[:, 2].mean(0), stats[0, 4]
        assert np.all(stats[:, 4] == wmax) and np.all(lo <= hi)
        center = {'none': np.zeros_like(mean), 'mean': mean, 'midrange': (lo + hi) / 2}[args.shift]
        magnitude = np.maximum(np.abs(lo - center), np.abs(hi - center)).clip(1e-5)
        r = magnitude**args.alpha / wmax.clip(1e-5)**(1 - args.alpha)
        r = np.clip(r / np.exp(np.log(r).mean()), 1 / 16, 16).astype(np.float32)
        shift = (center / r).astype(np.float32)
        if args.identity:
            r[:] = 1
            shift[:] = 0
        transformed.append(np.stack([r, shift]))
        means.append(mean)
        layers.append(dict(layer=layer, min_scale=float(r.min()), max_scale=float(r.max()),
                           max_abs_shift=float(np.abs(shift).max()),
                           max_activation=float(np.maximum(np.abs(lo), np.abs(hi)).max())))
    args.output.mkdir(parents=True)
    affine_path, mean_path = args.output / 'qkv-affine.f32.bin', args.output / 'qkv-mean.f32.bin'
    np.stack(transformed).astype('<f4').tofile(affine_path)
    np.stack(means).astype('<f4').tofile(mean_path)
    report = dict(alpha=args.alpha, shift=args.shift, identity=args.identity, images=len(cases),
                  source=source, observations_sha256=hashes, layers=layers,
                  affine_sha256=hashlib.sha256(affine_path.read_bytes()).hexdigest(),
                  mean_sha256=hashlib.sha256(mean_path.read_bytes()).hexdigest(),
                  transform='gamma/r; beta/r-shift in FP32. W_new=Half(W_original*r); bias=Half(b+W_new@shift). QKV FP16 layers keep original norm and weights.',
                  mean_bias='Optional: Half(b + W_original@mu - W_dequantized@(mu/r-shift)). Models weight error only, not input quantization, norm folding rounding or upstream drift.',
                  rounding='Fold before the norm output Half cast, unlike transforming an already rounded Half activation. Original FP16 QKV weights remain stored.')
    (args.output / 'calibration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['alpha', 'shift', 'identity', 'images', 'affine_sha256', 'mean_sha256']}))


if __name__ == '__main__':
    main()
