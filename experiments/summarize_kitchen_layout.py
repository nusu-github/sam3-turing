"""Evaluate ComfyKitchen attention against fixed INT8 and FP16 references; never relax gates."""
import json
from pathlib import Path
import statistics as st
import numpy as np
from scipy.optimize import linear_sum_assignment
root=Path('.cache/native-perf/kitchen-layout-ba018a0')
report={'quality_gates':{'same_count':True,'min_mask_iou':.98,'max_score_error':.02,'max_box_error_px':1.0},'quality':{},'timing':{},'gate_failures':[]}
files=['masks.u8.bin','scores.f32.bin','boxes.f32.bin','queries.i64.bin']
def load(p):
    m=json.loads((p/'metrics.json').read_text())
    return m,np.fromfile(p/'masks.u8.bin',np.uint8).reshape(m['count'],m['height']*m['width']).astype(bool),np.fromfile(p/'scores.f32.bin',np.float32),np.fromfile(p/'boxes.f32.bin',np.float32).reshape(-1,4)
for mode in ['head','sequence']:
    for case in ['truck','bag','child','wheel','empty']:
        p=root/f'quality-{mode}-{case}-1'
        if not (p/'metrics.json').exists():continue
        mb,b,sb,bb=load(p)
        for label,base in [('head',root/f'quality-head-{case}-1'),('fp16',Path('.cache/native-perf/boundary-ba018a0')/f'quality-exact-{case}-1')]:
            ma,a,sa,ba=load(base);iou=np.zeros((len(a),len(b)))
            for i in range(len(a)):
                for j in range(len(b)):
                    u=np.count_nonzero(a[i]|b[j]);iou[i,j]=np.count_nonzero(a[i]&b[j])/u if u else 1
            ii,jj=linear_sum_assignment(-iou)
            min_iou=float(iou[ii,jj].min()) if len(ii) else None
            score=float(np.abs(sa[ii]-sb[jj]).max()) if len(ii) else None
            box=float(np.abs(ba[ii]-bb[jj]).max()) if len(ii) else None
            passed=mb['finite'] and len(a)==len(b) and (not len(ii) or (min_iou>=.98 and score<=.02 and box<=1))
            key=f'{case}/{mode}/{label}'
            report['quality'][key]={'reference_count':len(a),'candidate_count':len(b),'min_mask_iou':min_iou,'max_score_error':score,'max_box_error_px':box,'gate_pass':passed,'byte_equal':all((base/f).read_bytes()==(p/f).read_bytes() for f in files)}
            if not passed:report['gate_failures'].append(key)
        if mode=='head':
            old=Path('.cache/native-perf/kitchen-ba018a0')/f'quality-kitchen_all-{case}-1'
            assert all((old/f).read_bytes()==(p/f).read_bytes() for f in files),'INT8 baseline regression'
    runs=[]
    for p in sorted(root.glob(f'timing-{mode}-truck-*')):
        if not (p/'metrics.json').exists():continue
        m=json.loads((p/'metrics.json').read_text());s=json.loads((p/'stages.json').read_text())
        runs.append({'path':str(p),'median_ms':m['median_wall_seconds']*1000,'wall_ms':[v*1000 for v in m['wall_seconds']],'stage_mean_ms':np.mean(s['gpu_ms'],axis=0).tolist(),'peak_allocated_bytes':m['peak_allocated_bytes']})
    if runs:
        values=[v for r in runs for v in r['wall_ms']]
        report['timing'][mode]={'runs':runs,'median_ms':st.median(values),'p95_ms':float(np.percentile(values,95)),'stage_mean_ms':np.mean([r['stage_mean_ms'] for r in runs],axis=0).tolist()}
Path('experiments/results/native_rtx2060/kitchen-layout-ba018a0.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
