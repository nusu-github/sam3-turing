"""Attention donor timing and quality versus previous QKV INT8 and FP16."""
import json
from pathlib import Path
import statistics as st
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

root=Path('.cache/native-perf/attention-ba018a0')
previous=Path('.cache/native-perf/boundary-ba018a0')
files=['masks.u8.bin','scores.f32.bin','boxes.f32.bin','queries.i64.bin']
report={'base':'ba018a0','mlp':'int8_boundary','timing':{},'quality':{}}
report['gate_failures']=[]
report['quality_gates']={'same_count':True,'min_mask_iou':.98,'max_score_error':.02,'max_box_error_px':1.0}
for mode in ['exact','global32','global64']:
    for run in [1,2]:
        assert (root/f'timing-{mode}-truck-{run}'/'metrics.json').exists(), 'incomplete timing matrix'
    for case in ['truck','bag','child','wheel','empty']:
        assert (root/f'quality-{mode}-{case}-1'/'metrics.json').exists(), 'incomplete quality matrix'
for mode in ['exact','global32','global64']:
    runs=[]
    for p in sorted(root.glob(f'timing-{mode}-truck-*')):
        m=json.loads((p/'metrics.json').read_text())
        s=json.loads((p/'stages.json').read_text())
        t=json.loads((p/'telemetry.json').read_text())
        if mode=='exact':
            assert all((p/f).read_bytes()==(Path('.cache/native-perf/projection-ba018a0/timing-qkv-truck-1')/f).read_bytes() for f in files)
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

def load(p):
    m=json.loads((p/'metrics.json').read_text())
    return (np.fromfile(p/'masks.u8.bin',np.uint8).reshape(m['count'],m['height']*m['width']).astype(bool),
            np.fromfile(p/'scores.f32.bin',np.float32),np.fromfile(p/'boxes.f32.bin',np.float32).reshape(-1,4))
for case in ['truck','bag','child','wheel','empty']:
    for mode in ['exact','global32','global64']:
        p=root/f'quality-{mode}-{case}-1'
        if not (p/'metrics.json').exists():continue
        b,sb,bb=load(p)
        for reference,base in [('fp16',previous/f'quality-exact-{case}-1'),('previous',Path('.cache/native-perf/projection-ba018a0')/f'quality-qkv-{case}-1')]:
            a,sa,ba=load(base)
            iou=np.zeros((len(a),len(b)))
            for i in range(len(a)):
                for j in range(len(b)):
                    union=np.count_nonzero(a[i]|b[j])
                    iou[i,j]=np.count_nonzero(a[i]&b[j])/union if union else 1
            ii,jj=linear_sum_assignment(-iou)
            if reference=='previous' and mode!='exact':
                if len(a)!=len(b):report['gate_failures'].append(f'detection count changed: {case}/{mode}')
                if len(ii):
                    if float(iou[ii,jj].min())<.98:report['gate_failures'].append(f'mask IoU gate: {case}/{mode}')
                    if float(np.abs(sa[ii]-sb[jj]).max())>.02:report['gate_failures'].append(f'score gate: {case}/{mode}')
                    if float(np.abs(ba[ii]-bb[jj]).max())>1.0:report['gate_failures'].append(f'box gate: {case}/{mode}')
            report['quality'][f'{case}/{mode}/{reference}']={
                'reference_count':len(a),'candidate_count':len(b),'matched_ious':iou[ii,jj].tolist(),
                'unmatched_reference':len(a)-len(ii),'unmatched_candidate':len(b)-len(jj),
                'max_score_error':float(np.abs(sa[ii]-sb[jj]).max()) if len(ii) else None,
                'max_box_error_px':float(np.abs(ba[ii]-bb[jj]).max()) if len(ii) else None,
                'byte_equal':all((base/f).read_bytes()==(p/f).read_bytes() for f in files)}
report['reservation_byte_parity']={case:all(
    (root/f'quality-global32-{case}-1'/f).read_bytes()==(root/f'quality-global64-{case}-1'/f).read_bytes()
    for f in files) for case in ['truck','bag','child','wheel','empty']}
dest=Path('experiments/results/native_rtx2060/attention-ba018a0.json')
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
sys.exit(bool(report['gate_failures']))
