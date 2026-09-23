"""Causal replay: change only the recorded frame-19 pointer in temporal attention."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('library', type=Path)
    p.add_argument('store', type=Path)
    p.add_argument('directory', type=Path)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    torch.ops.load_library(str(a.library.resolve()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    values = {}
    root = a.directory / 'native/reverse-input'
    for line in (root / 'tensors.tsv').read_text().splitlines():
        key, dtype, shape, strides = line.split('\t')
        shape = tuple(int(x) for x in shape.strip(',').split(','))
        strides = tuple(int(x) for x in strides.strip(',').split(','))
        raw = torch.from_numpy(np.fromfile(root / (key+'.bin'), np.float32).reshape(shape))
        tensor = torch.empty_strided(shape, strides, device='cuda', dtype={'Float':torch.float32, 'BFloat16':torch.bfloat16}[dtype])
        values[key] = tensor.copy_(raw)
    source = torch.load(a.directory / 'reference/reverse-input.pt', weights_only=True)
    names = list(dict.fromkeys(k.split('.')[0] for k in values if re.match(r'^(cond|tracked)\d+\.', k)))
    ids = [int(re.search(r'\d+', key).group()) for key in names]
    conditioning = [key.startswith('cond') for key in names]
    arrays = [[values.get(key+'.'+field, torch.empty(0)) for key in names]
              for field in ['memory', 'position', 'pointer', 'score', 'image', 'image_position']]
    results = []
    for replace in [False, True]:
        pointers = list(arrays[2])
        if replace:
            pointers[names.index('cond19')] = source['cond19.pointer'].cuda()
        out = torch.ops.sam3_native.multiplex_temporal(
            str(a.store), values['source'], values['source_position'], [[0]+[-1]*15], ids, conditioning,
            arrays[0], arrays[1], pointers, arrays[3], arrays[4], arrays[5],
            [72,72,17,34,7,4,16,1],
            [False,True,True,False,False,False,False,True,True,True], .01, 'bf16_reference')
        expected = source['conditioned'].cuda() if replace else values['conditioned']
        exact = torch.equal(out['features'], expected)
        results.append(dict(pointer='old_source_click' if replace else 'latest_native_click',
                            expected='old_source_conditioned' if replace else 'native_conditioned',
                            exact=exact, max_abs=float((out['features']-expected).abs().max())))
    a.report.write_text(json.dumps(dict(scope='Replay recorded real reverse temporal inputs through the native neural conditioner. The second case changes only cond19.pointer to the old source pointer; all other inputs/weights/settings are unchanged.',cases=results),indent=2)+'\n')
    print(json.dumps(results,indent=2))
    assert all(r['exact'] for r in results)


if __name__ == '__main__':
    main()
