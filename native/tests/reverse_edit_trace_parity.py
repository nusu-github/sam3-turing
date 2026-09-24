"""Compare captured native/source reverse inputs and audit point-pointer provenance.

This consumes real full-video regression traces; it never generates reference
predictions or changes the production predictor. Run after the opt-in traces in
video_pipeline_parity.py and sam3_video_pipeline_probe --edit-sequence-trace.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def read_native(directory):
    values = {}
    for line in (directory / 'tensors.tsv').read_text().splitlines():
        key, dtype, shape, stride = line.split('\t')
        shape = tuple(int(x) for x in shape.strip(',').split(','))
        values[key] = np.fromfile(directory / (key + '.bin'), np.float32).reshape(shape)
    return values


def compare(native, reference):
    rows = []
    for key, a in native.items():
        if key not in reference:
            rows.append(dict(key=key, exact=False, missing_reference=True))
            continue
        b = reference[key].float().numpy()
        row = dict(key=key, native_shape=list(a.shape), source_shape=list(b.shape))
        # Do not normalize away a source history-layout defect in this audit.
        row['exact'] = a.shape == b.shape and bool(np.array_equal(a, b))
        if a.shape == b.shape:
            row.update(max_abs=float(np.max(np.abs(a-b))),
                       different_elements=int(np.count_nonzero(a != b)))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    native = read_native(args.directory / 'native/reverse-input')
    before = torch.load(args.directory / 'reference/reverse-input.pt', weights_only=True)
    after = torch.load(args.directory / 'pointer-reference/reverse-input.pt', weights_only=True)
    calls = torch.load(args.directory / 'pointer-reference/point19-neural.pt', weights_only=True)
    assert len(calls) == 2
    first, latest = (c['pointer'].float().numpy() for c in calls)
    provenance = dict(
        first_point_count=calls[0]['labels'].shape[-1],
        latest_point_count=calls[1]['labels'].shape[-1],
        old_reference_uses_first_click=bool(np.array_equal(before['cond19.pointer'].float().numpy(), first)),
        native_uses_latest_original_click=bool(np.array_equal(native['cond19.pointer'], latest)),
        repaired_reference_uses_latest_original_click=bool(np.array_equal(after['cond19.pointer'].float().numpy(), latest)),
        first_and_latest_differ=not bool(np.array_equal(first, latest)),
    )
    initial = compare(native, before)
    repaired = compare(native, after)
    report = dict(scope='Actual full-video repeated-edit traces. Pointer provenance is from original neural outputs; the opt-in reference repair consumes no native tensors.',
                  before=initial, after=repaired, pointer_provenance=provenance,
                  all_repaired_inputs_and_conditioned_features_exact=all(r['exact'] for r in repaired))
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))
    print('Before:', sum(r['exact'] for r in initial), '/', len(initial))
    print('After:', sum(r['exact'] for r in repaired), '/', len(repaired))
    assert all(v for k, v in provenance.items() if not k.endswith('_count'))
    assert report['all_repaired_inputs_and_conditioned_features_exact']


if __name__ == '__main__':
    main()
