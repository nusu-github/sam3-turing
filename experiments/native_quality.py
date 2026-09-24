"""FP16-agreement gates and fixed regression cases for native image experiments.

A candidate passes when outputs are finite, detection counts match, and after
Hungarian matching on mask IoU the minimum IoU is >= .98, the maximum score
difference <= .02 and the maximum box-coordinate difference <= 1 px. These gates
measure agreement with the FP16 reference, not ground-truth accuracy.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

FILES = ['masks.u8.bin', 'scores.f32.bin', 'boxes.f32.bin', 'queries.i64.bin']

# The five original cases: truck / paper bag / child / wheel / empty elephant.
ORIGINAL_ROOT = Path('.cache/native-perf')
ORIGINAL_CASES = {'truck': ('truck.ppm', 'prompt.txt'), 'bag': ('groceries.ppm', 'bag.txt'),
                  'child': ('test_image.ppm', 'child.txt'), 'wheel': ('truck.ppm', 'wheel.txt'),
                  'empty': ('truck.ppm', 'empty.txt')}

# Twelve cases fixed before the INT8 attention study. Video frames are correlated;
# dance99-shoe, groceries-bread and court-person are empty even in FP16.
EXTENDED_ROOT = Path('.cache/native-perf/kitchen-extended-ba018a0')
EXTENDED_CASES = [
    ('dance0-person', 'assets/videos/0001/0.jpg', 'person'),
    ('dance50-person', 'assets/videos/0001/50.jpg', 'person'),
    ('dance99-person', 'assets/videos/0001/99.jpg', 'person'),
    ('dance0-shoe', 'assets/videos/0001/0.jpg', 'shoe'),
    ('dance99-shoe', 'assets/videos/0001/99.jpg', 'shoe'),
    ('dance50-shirt', 'assets/videos/0001/50.jpg', 'shirt'),
    ('groceries-car', 'assets/images/groceries.jpg', 'car'),
    ('groceries-bread', 'assets/images/groceries.jpg', 'bread'),
    ('court-person', 'assets/images/test_image.jpg', 'person'),
    ('court-blue-vest', 'assets/images/test_image.jpg', 'blue vest'),
    ('truck-window', 'assets/images/truck.jpg', 'window'),
    ('truck-tire', 'assets/images/truck.jpg', 'tire'),
]


def regression_inputs():
    """The 17 known regression cases as (name, image.ppm, prompt.txt).

    Writes missing extended-case PPM/prompt files from their source JPEGs and
    refuses to reuse a directory whose recorded case list differs.
    """
    EXTENDED_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = EXTENDED_ROOT / 'cases.json'
    content = json.dumps(EXTENDED_CASES, indent=2) + '\n'
    if manifest.exists():
        assert manifest.read_text() == content, 'case list changed'
    else:
        manifest.write_text(content)
    for case, source, prompt in EXTENDED_CASES:
        image, text = EXTENDED_ROOT / (case + '.ppm'), EXTENDED_ROOT / (case + '.txt')
        if not image.exists():
            with Image.open(source) as im:
                im.convert('RGB').save(image)
            text.write_text(prompt, encoding='utf-8')
    inputs = [(name, ORIGINAL_ROOT / image, ORIGINAL_ROOT / prompt)
              for name, (image, prompt) in ORIGINAL_CASES.items()]
    inputs += [(name, EXTENDED_ROOT / (name + '.ppm'), EXTENDED_ROOT / (name + '.txt'))
               for name, _, _ in EXTENDED_CASES]
    return inputs


def load(path):
    m = json.loads((path / 'metrics.json').read_text())
    return (m, np.fromfile(path / FILES[0], np.uint8).reshape(m['count'], m['height']*m['width']).astype(bool),
            np.fromfile(path / FILES[1], np.float32), np.fromfile(path / FILES[2], np.float32).reshape(-1,4))


def compare(base, candidate):
    ma, a, sa, ba = load(base)
    mb, b, sb, bb = load(candidate)
    assert (ma['width'],ma['height']) == (mb['width'],mb['height'])
    iou = np.zeros((len(a),len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            union = np.count_nonzero(a[i] | b[j])
            iou[i,j] = np.count_nonzero(a[i] & b[j])/union if union else 1
    ii,jj = linear_sum_assignment(-iou)
    minimum = float(iou[ii,jj].min()) if len(ii) else None
    score = float(np.abs(sa[ii]-sb[jj]).max()) if len(ii) else None
    box = float(np.abs(ba[ii]-bb[jj]).max()) if len(ii) else None
    finite = bool(ma['finite'] and mb['finite'] and all(np.isfinite(x).all() for x in [sa,sb,ba,bb]))
    passed = finite and len(a)==len(b) and (not len(ii) or (minimum>=.98 and score<=.02 and box<=1))
    return dict(reference_count=len(a), candidate_count=len(b), min_mask_iou=minimum,
                max_score_error=score, max_box_error_px=box, finite=finite, gate_pass=passed,
                byte_equal=all((base/f).read_bytes()==(candidate/f).read_bytes() for f in FILES))
