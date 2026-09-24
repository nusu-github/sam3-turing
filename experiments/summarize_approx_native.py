"""Summarize timing and Hungarian IoU-matched output differences."""
from pathlib import Path
import json
import statistics
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

root = Path(sys.argv[1] if len(sys.argv) > 1 else '.cache/native-perf/approximation-v3')
report = {'root': str(root), 'timing': {}, 'quality': {}}
for name in ('exact', 'tanh', 'fused', 'int8'):
    runs = []
    for p in sorted(root.glob(f'timing-{name}-truck-*')):
        m = json.loads((p/'metrics.json').read_text())
        t = json.loads((p/'telemetry.json').read_text())
        s = json.loads((p/'stages.json').read_text())
        runs.append({'path': str(p), 'median_ms': m['median_wall_seconds']*1000,
                     'wall_ms': [v*1000 for v in m['wall_seconds']],
                     'peak_allocated_bytes': m['peak_allocated_bytes'],
                     'gpu_stage_mean_ms': np.mean(s['gpu_ms'],axis=0).tolist(),
                     'sm_mhz_tail_median': statistics.median(v['sm_mhz'] for v in t['samples'][-100:]),
                     'temperature_max': max(v['temperature_c'] for v in t['samples'])})
    if runs:
        all_ms = [v for run in runs for v in run['wall_ms']]
        report['timing'][name] = {'runs': runs, 'pooled_median_ms': statistics.median(all_ms),
                                  'pooled_p95_ms': float(np.percentile(all_ms,95))}

def read(p):
    m=json.loads((p/'metrics.json').read_text())
    shape=(m['count'],m['height'],m['width'])
    return (m, np.fromfile(p/'masks.u8.bin',dtype=np.uint8).reshape(shape).astype(bool),
            np.fromfile(p/'scores.f32.bin',dtype=np.float32),
            np.fromfile(p/'boxes.f32.bin',dtype=np.float32).reshape(-1,4))

for case in ('truck','bag','child','wheel','empty'):
    base=root/f'quality-exact-{case}-1'
    if not (base/'metrics.json').exists(): continue
    meta,a,score_a,box_a=read(base)
    for name in ('tanh','fused','int8'):
        p=root/f'quality-{name}-{case}-1'
        if not (p/'metrics.json').exists(): continue
        other,b,score_b,box_b=read(p)
        iou=np.zeros((len(a),len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                union=np.count_nonzero(a[i]|b[j])
                iou[i,j]=np.count_nonzero(a[i]&b[j])/union if union else 1.0
        ii,jj=linear_sum_assignment(-iou)
        row={'reference_count':len(a),'candidate_count':len(b),'matched_ious':iou[ii,jj].tolist(),
             'matched_changed_pixels':[int(np.count_nonzero(a[i]!=b[j])) for i,j in zip(ii,jj)],
             'unmatched_reference':len(a)-len(ii),'unmatched_candidate':len(b)-len(jj),
             'byte_equal':all((base/f).read_bytes()==(p/f).read_bytes() for f in
                 ('masks.u8.bin','scores.f32.bin','boxes.f32.bin','queries.i64.bin')),
             'max_matched_score_error':float(np.max(np.abs(score_a[ii]-score_b[jj]))) if len(ii) else None,
             'max_matched_box_error_px':float(np.max(np.abs(box_a[ii]-box_b[jj]))) if len(ii) else None}
        report['quality'][f'{case}/{name}']=row
out=Path('experiments/results/native_rtx2060/approximation-results.json')
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
