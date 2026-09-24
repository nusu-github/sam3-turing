"""Verify an observer preserved FP16 outputs and produced finite per-layer data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_quant_research import ROOT
from native_quality import FILES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('name', help='Completed observation run name')
    p.add_argument('--reference', default='calibration-fp16')
    args = p.parse_args()
    root = ROOT/'runs'/args.name
    source = json.loads((root/'run.json').read_text())
    assert source['args']['kind'] == 'observe' and source['args']['mode'] == 'fp16'
    assert source['args']['role'] == 'calibration'
    target = source['args'].get('observe_target', 'fc2')
    width = {'fc2': 4736, 'qkv': 1024, 'qkv_error': 3072}[target]
    report = json.loads((Path('experiments/results/native_rtx2060')/(args.name+'.json')).read_text())
    assert report['signature'] == source and report['results']
    results = []
    for result in report['results']:
        case = root/result['case']
        assert json.loads((case/'complete.json').read_text()) == result
        reference = ROOT/'runs'/args.reference/case.name
        assert (reference/'complete.json').exists()
        equal = all((case/f).read_bytes() == (reference/f).read_bytes() for f in FILES)
        assert equal, f'observer changed output: {case}'
        hashes = {}
        for layer in range(32):
            path = case/'observations'/f'{target}-{layer}.f32.bin'
            value = np.fromfile(path, dtype='<f4').reshape(5, width)
            assert np.isfinite(value).all(), f'nonfinite observations: {path}'
            if target == 'qkv_error':
                assert (value[1] >= 0).all()
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        results.append(dict(case=case.name, fp16_byte_equal=equal, observations_sha256=hashes))
    dest = Path('experiments/results/native_rtx2060')/(args.name+'-observer-check.json')
    dest.write_text(json.dumps(dict(source=source, reference=args.reference, results=results), indent=2)+'\n')
    print(json.dumps(dict(cases=len(results), all_fp16_byte_equal=True, layers_per_case=32)))


if __name__ == '__main__':
    main()
