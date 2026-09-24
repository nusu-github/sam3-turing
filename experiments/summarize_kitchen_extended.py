"""Fixed quality gates for the extended attention experiment, with attribution."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from run_kitchen_extended import ROOT, CASES

FILES = ['masks.u8.bin', 'scores.f32.bin', 'boxes.f32.bin', 'queries.i64.bin']

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

def main():
    report = {'gates':dict(min_mask_iou=.98,max_score_error=.02,max_box_error_px=1,same_count=True,finite=True),
              'scope':'12 extra cases; video frames are correlated; no ground-truth accuracy claim',
              'quality':{},'summary':{}}
    modes = sorted({p.name[len(case)+1:] for case,_,_ in CASES for p in ROOT.glob(case+'-*') if (p/'complete.json').exists()})
    for mode in modes:
        if mode == 'fp16':
            continue
        for reference in ['fp16','int8']:
            if reference == mode:
                continue
            rows = []
            for case,_,_ in CASES:
                base,candidate = ROOT/f'{case}-{reference}', ROOT/f'{case}-{mode}'
                if not all((p/'complete.json').exists() for p in [base,candidate]):
                    continue
                value = compare(base,candidate)
                report['quality'][f'{case}/{mode}/{reference}'] = value
                rows.append((case,value))
            if rows:
                report['summary'][f'{mode}/{reference}'] = {
                    'completed':len(rows), 'passed':sum(v['gate_pass'] for _,v in rows),
                    'nonempty_reference_cases':sum(v['reference_count']>0 for _,v in rows),
                    'nonempty_passed':sum(v['reference_count']>0 and v['gate_pass'] for _,v in rows),
                    'failed_cases':[c for c,v in rows if not v['gate_pass']],
                    'min_mask_iou':min((v['min_mask_iou'] for _,v in rows if v['min_mask_iou'] is not None),default=None),
                    'max_score_error':max((v['max_score_error'] for _,v in rows if v['max_score_error'] is not None),default=None),
                    'max_box_error_px':max((v['max_box_error_px'] for _,v in rows if v['max_box_error_px'] is not None),default=None)}
    Path('experiments/results/native_rtx2060/kitchen-extended-ba018a0.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))

if __name__ == '__main__':
    main()
