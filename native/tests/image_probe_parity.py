"""Validate a standalone native process against saved upstream image results.

Python prepares and checks fixtures; the child runs with no Python on PATH.
The child reads arbitrary UTF-8 text and runs its own normalization and BPE.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('executable', type=Path)
    p.add_argument('store', type=Path)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--vocabulary', type=Path, default=Path('sam3/assets/bpe_simple_vocab_16e6.txt.gz'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    image = Image.open(a.reference / 'input.png').convert('RGB')
    image_path = a.output / 'input.ppm'
    image.save(image_path)
    metadata = json.loads((a.reference / 'metadata.json').read_text())
    results = []
    for case in metadata['cases']:
        if 'text' not in case['inputs']:
            continue
        prefix = a.output / case['name']
        prompt_path = prefix.with_suffix('.prompt.txt')
        prompt_path.write_text(case['inputs']['text'], encoding='utf8')
        command = [str(a.executable.resolve()), str(a.store.resolve()), 'sam3', 'cuda',
                   'bf16_reference', str(image_path.resolve()), str(prefix.resolve()), '.5',
                   '--text-file', str(a.vocabulary.resolve()), str(prompt_path.resolve())]
        started = time.perf_counter()
        child = subprocess.run(command, env={**os.environ, 'PATH': '/nonexistent'},
                               text=True, capture_output=True, check=True)
        elapsed = time.perf_counter() - started
        prefix.with_suffix('.log').write_text(child.stdout + child.stderr)
        actual = json.loads(prefix.with_suffix('.json').read_text())
        saved = torch.load(a.reference / (case['name'] + '.pt'), weights_only=True, map_location='cpu')
        expected = saved['result']
        count = len(expected['scores'])
        assert actual['count'] == count
        assert (actual['width'], actual['height']) == image.size
        boxes = torch.tensor([x['box'] for x in actual['detections']], dtype=torch.float32).reshape(-1, 4)
        scores = torch.tensor([x['score'] for x in actual['detections']], dtype=torch.float32)
        torch.testing.assert_close(boxes, expected['boxes'], rtol=0, atol=0)
        torch.testing.assert_close(scores, expected['scores'].float(), rtol=0, atol=0)
        raw = saved['raw']
        keep = (raw['pred_logits'].sigmoid() * raw['presence_logit_dec'].sigmoid().unsqueeze(1))[0, :, 0] > .5
        assert [x['query'] for x in actual['detections']] == keep.nonzero().flatten().tolist()
        row_bytes = (image.width * image.height + 7) // 8
        assert actual['mask_row_bytes'] == row_bytes
        packed = np.frombuffer(prefix.with_suffix('.masks.bin').read_bytes(), dtype=np.uint8).reshape(count, row_bytes)
        masks = np.unpackbits(packed, axis=1, bitorder='little')[:, :image.width * image.height]
        expected_masks = expected['masks'].numpy().reshape(count, image.width * image.height)
        np.testing.assert_array_equal(masks, expected_masks)
        result = dict(case=case['name'], text=case['inputs']['text'], detections=count,
                      max_box_error=0, max_score_error=0, differing_mask_pixels=0,
                      elapsed_seconds=elapsed, child_path='/nonexistent')
        results.append(result)
        print(json.dumps(result), flush=True)
    a.report.write_text(json.dumps(dict(cases=results, gpu=torch.cuda.get_device_name(),
        scope='Python prepares/checks fixtures only. Native child normalizes/tokenizes UTF-8 text and runs image grounding; standalone LibTorch packaging and other model functions remain pending.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
