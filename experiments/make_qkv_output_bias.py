"""Calibrate output-channel biases from all 32 frozen FP16 shadow observations.

Initial QKV affine and weight mean bias are fixed before observing. This models
local normalization rounding and W8A8 errors, not upstream quantization drift.
The reported MSE estimates ignore output re-rounding after applying the bias.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_quant_research import DATA


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('observations', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--identity', action='store_true', help='Export unchanged initial biases for integration control')
    args = p.parse_args()
    source = json.loads((args.observations / 'run.json').read_text())
    assert source['args']['kind'] == 'observe' and source['args']['role'] == 'calibration'
    assert source['args']['mode'] == 'fp16' and source['args']['observe_target'] == 'qkv_error'
    assert source['manifest_sha256'] == sha(DATA)
    initial = Path(source['args']['qkv_shadow_calibration'])
    assert source['qkv_shadow_affine_sha256'] == sha(initial / 'qkv-affine.f32.bin')
    assert source['qkv_shadow_mean_sha256'] == sha(initial / 'qkv-mean.f32.bin')
    cases = sorted(d.parent for d in args.observations.glob('*/complete.json'))
    ids = [json.loads((c / 'complete.json').read_text())['image_id'] for c in cases]
    required = {r['image_id'] for r in json.loads(DATA.read_text())['images'] if r['role'] == 'calibration'}
    assert len(cases) == len(required) == 32 and set(ids) == required
    assert not args.output.exists(), 'refusing to overwrite output calibration'
    biases, layers, hashes = [], [], {}
    for layer in range(32):
        stats = []
        for case in cases:
            path = case / 'observations' / f'qkv_error-{layer}.f32.bin'
            hashes[str(path)] = sha(path)
            stats.append(np.fromfile(path, dtype='<f4').reshape(5, 3072))
        values = np.stack(stats).astype(np.float64)
        assert np.isfinite(values).all() and np.all(values[:, 1] >= 0)
        assert np.all(values[:, 4] == values[0, 4]), 'shadow initial bias changed across images'
        error_mean = values[:, 0].mean(0)
        error_square = values[:, 1].mean(0)
        base = values[0, 4]
        bias = (base + (0 if args.identity else error_mean)).astype(np.float16).astype(np.float32)
        assert np.isfinite(bias).all()
        delta = bias.astype(np.float64) - base
        mse_before = float(error_square.mean())
        mse_after = float((error_square - 2 * delta * error_mean + delta**2).mean())
        # Leave-one-image-out calculation uses calibration images only. It is
        # a diagnostic estimate, not an independent evaluation or actual rerun.
        loo_mean = np.zeros_like(values[:, 0]) if args.identity else (values[:, 0].sum(0) - values[:, 0]) / 31
        loo_delta = (base + loo_mean).astype(np.float16).astype(np.float64) - base
        loo_mse = float((values[:, 1] - 2 * loo_delta * values[:, 0] + loo_delta**2).mean())
        biases.append(bias)
        layers.append(dict(layer=layer, max_abs_correction=float(np.abs(delta).max()),
                           rms_correction=float(np.sqrt((delta**2).mean())),
                           mse_before=mse_before, mse_estimate_after=mse_after,
                           leave_one_calibration_image_out_mse_estimate=loo_mse,
                           max_abs_mean_error_before=float(np.abs(error_mean).max()),
                           max_abs_mean_error_estimate_after=float(np.abs(error_mean-delta).max())))
    args.output.mkdir(parents=True)
    for name in ['qkv-affine.f32.bin', 'qkv-mean.f32.bin']:
        (args.output / name).write_bytes((initial / name).read_bytes())
    output = args.output / 'qkv-output-bias.f32.bin'
    np.stack(biases).astype('<f4').tofile(output)
    report = dict(identity=args.identity, images=len(cases), tokens_per_image=5184,
                  source=source, observations_sha256=hashes, layers=layers,
                  affine_sha256=sha(args.output / 'qkv-affine.f32.bin'),
                  mean_sha256=sha(args.output / 'qkv-mean.f32.bin'), output_bias_sha256=sha(output),
                  formula='Half(base_weight_mean_bias + mean_image(mean_token(FP16_QKV - INT8_QKV)))',
                  scope=__doc__, holdout_used=False, development_used=False)
    (args.output / 'calibration.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['identity', 'images', 'affine_sha256', 'mean_sha256', 'output_bias_sha256']}))


if __name__ == '__main__':
    main()
