"""Summarize the fused INT8 boundary comparison without overwriting prior reports."""
import json
from pathlib import Path
import statistics as st
import numpy as np
from scipy.optimize import linear_sum_assignment

root=Path('.cache/native-perf/boundary-ba018a0')
files=['masks.u8.bin','scores.f32.bin','boxes.f32.bin','queries.i64.bin']
report={'base':'ba018a0','root':str(root),'timing':{},'quality':{}}
for mode in ['exact','int8','int8_boundary']:
    runs=[]
    for p in sorted(root.glob(f'timing-{mode}-truck-*')):
        m=json.loads((p/'metrics.json').read_text())
        s=json.loads((p/'stages.json').read_text())
        t=json.loads((p/'telemetry.json').read_text())
        runs.append({'path':str(p),'median_ms':m['median_wall_seconds']*1000,
                     'wall_ms':[v*1000 for v in m['wall_seconds']],
                     'stage_mean_ms':np.mean(s['gpu_ms'],axis=0).tolist(),
                     'peak_allocated_bytes':m['peak_allocated_bytes'],
                     'sm_mhz_tail_median':st.median(v['sm_mhz'] for v in t['samples'][-100:]),
                     'temperature_max':max(v['temperature_c'] for v in t['samples'])})
    if runs:
        vals=[v for r in runs for v in r['wall_ms']]
        report['timing'][mode]={'runs':runs,'pooled_median_ms':st.median(vals),
                               'p95_ms':float(np.percentile(vals,95)),
                               'stage_mean_ms':np.mean([r['stage_mean_ms'] for r in runs],axis=0).tolist()}
for case in ['truck','bag','child','wheel','empty']:
    base=root/f'quality-exact-{case}-1'
    if not (base/'metrics.json').exists():continue
    m=json.loads((base/'metrics.json').read_text())
    a=np.fromfile(base/'masks.u8.bin',np.uint8).reshape(m['count'],m['height']*m['width']).astype(bool)
    sa=np.fromfile(base/'scores.f32.bin',np.float32)
    ba=np.fromfile(base/'boxes.f32.bin',np.float32).reshape(-1,4)
    for mode in ['int8','int8_boundary']:
        p=root/f'quality-{mode}-{case}-1'
        if not (p/'metrics.json').exists():continue
        n=json.loads((p/'metrics.json').read_text())
        b=np.fromfile(p/'masks.u8.bin',np.uint8).reshape(n['count'],m['height']*m['width']).astype(bool)
        sb=np.fromfile(p/'scores.f32.bin',np.float32)
        bb=np.fromfile(p/'boxes.f32.bin',np.float32).reshape(-1,4)
        iou=np.zeros((len(a),len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                union=np.count_nonzero(a[i]|b[j])
                iou[i,j]=np.count_nonzero(a[i]&b[j])/union if union else 1
        ii,jj=linear_sum_assignment(-iou)
        ref=root/f'quality-int8-{case}-1'
        report['quality'][f'{case}/{mode}']={
            'reference_count':len(a),'candidate_count':len(b),'matched_ious':iou[ii,jj].tolist(),
            'max_score_error':float(np.abs(sa[ii]-sb[jj]).max()) if len(ii) else None,
            'max_box_error_px':float(np.abs(ba[ii]-bb[jj]).max()) if len(ii) else None,
            'byte_equal_to_original_int8':all((ref/f).read_bytes()==(p/f).read_bytes() for f in files)}
dest=Path('experiments/results/native_rtx2060/boundary-ba018a0.json')
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
