"""Pure C client output versus retained owning C++ predictor outputs.

The C++ fixtures have their own recorded original-model comparisons. This checks
the C boundary without regenerating or adapting the original neural reference.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(root, tag):
    meta = json.loads((root / f'{tag}.json').read_text())
    types = {1: np.uint8, 2: np.bool_, 3: np.int32, 4: np.int64,
             5: np.float16, 6: np.uint16, 7: np.float32, 8: np.float64}
    return {key: np.fromfile(root / f'{tag}-{key.replace("/", "_")}.bin', types[value['dtype']]).reshape(value['shape'])
            for key, value in meta.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--kind', choices=['image', 'video'], required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    tags = (['semantic', 'empty_stateless', 'point', 'cleared', 'cleared_twice', 'mask_new', 'mask_point', 'mask_cleared', 'mask_replaced', 'reset']
            if a.kind == 'image' else ['0.person', *[f'{i}.person_track' for i in range(4)], '1.box', '1.box_track', '2.box_track', '2.text_box', '2.fetch', '0.reset_person', '0.point_only', '0.mask_only'])
    rows = []
    for tag in tags:
        actual = load(a.native, tag)
        info = json.loads((a.reference / f'{tag}.json').read_text())
        ids = np.fromfile(a.reference / f'{tag}.ids.i64.bin', np.int64)
        n, h, w = len(ids), info['height'], info['width']
        masks = np.fromfile(a.reference / f'{tag}.masks.bin', np.uint8).reshape(n, h, (w + 7) // 8)
        expected = dict(ids=ids, probabilities=np.fromfile(a.reference / f'{tag}.scores.f32.bin', np.float32),
                        boxes_xywh=np.fromfile(a.reference / f'{tag}.boxes.f32.bin', np.float32).reshape(n, 4),
                        masks=np.unpackbits(masks, axis=-1, bitorder='little')[..., :w].astype(bool))
        equal = {key: bool(np.array_equal(value, actual[key])) for key, value in expected.items()}
        equal['centers_shape'] = actual['centers'].shape == (n, 2)
        if a.kind == 'video' and 'emitted_at' in info:
            equal['emission'] = int((a.native / f'{tag}.emission.txt').read_text()) == info['emitted_at']
        rows.append(dict(tag=tag, exact=equal))
    original_tag = 'semantic' if a.kind == 'image' else '0.person_track'
    original, retained = load(a.native, original_tag), load(a.native, 'retained_after_destroy')
    retained_equal = original.keys() == retained.keys() and all(np.array_equal(value, retained[key]) for key, value in original.items())
    report = dict(kind=a.kind, cases=rows, retained_after_owner_destruction_exact=retained_equal,
                  all_exact=retained_equal and all(all(row['exact'].values()) for row in rows),
                  scope='Pure C11 client versus retained C++ owner results. Both use full neural execution and the same shared modular store. C++ original-reference provenance is separate. Native client also asserts callback failures/reentry/cancel, shared-owner state isolation, source errors, and context lifetime.')
    if a.kind == 'image':
        restored = load(a.native, 'empty_stateless')
        id_ = int(restored['ids'][0])
        original_mask = np.fromfile(a.reference / 'restored-input.bool.bin', np.bool_).reshape(restored[f'cached_masks/{id_}'].shape)
        report['retained_original_mask_exact'] = bool(np.array_equal(original_mask, restored[f'cached_masks/{id_}']))
        report['all_exact'] &= report['retained_original_mask_exact']
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert report['all_exact'], 'C predictor boundary changed outputs'


if __name__ == '__main__':
    main()
