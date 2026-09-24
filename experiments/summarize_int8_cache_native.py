"""Require bit-identical model outputs; report same-build paired process timing."""
import json
from pathlib import Path
import statistics as st
import sys
import numpy as np

root=Path('.cache/native-perf/int8-cache-ba018a0')
modes=sys.argv[1:] or ['exact','fc2','all']
files=['masks.u8.bin','scores.f32.bin','boxes.f32.bin','queries.i64.bin']
report={'base':'ba018a0','mlp':'int8_boundary','projection':'qkv','attention':'exact','qkv_rope':'fused','fc2_norm':'fused',
        'gate':'bit-identical masks/scores/boxes/queries','timing':{},'quality':{},'gate_failures':[]}
for mode in modes:
    runs=[]
    for i in [1,2]:
        p=root/f'timing-{mode}-truck-{i}'
        m=json.loads((p/'metrics.json').read_text())
        s=json.loads((p/'stages.json').read_text())
        t=json.loads((p/'telemetry.json').read_text())
        runs.append({'path':str(p),'median_ms':m['median_wall_seconds']*1000,
                     'wall_ms':[v*1000 for v in m['wall_seconds']],
                     'stage_mean_ms':np.mean(s['gpu_ms'],axis=0).tolist(),
                     'peak_allocated_bytes':m['peak_allocated_bytes'],
                     'sm_mhz_tail_median':st.median(v['sm_mhz'] for v in t['samples'][-100:]),
                     'temperature_max':max(v['temperature_c'] for v in t['samples'])})
        reference=Path('.cache/native-perf/mlp-sweep-ba018a0/timing-fc2-truck-1')
        if not all((p/f).read_bytes()==(reference/f).read_bytes() for f in files):
            report['gate_failures'].append(p.name)
    vals=[v for r in runs for v in r['wall_ms']]
    report['timing'][mode]={'runs':runs,'pooled_median_ms':st.median(vals),
        'p95_ms':float(np.percentile(vals,95)),
        'stage_mean_ms':np.mean([r['stage_mean_ms'] for r in runs],axis=0).tolist()}
for case in ['truck','bag','child','wheel','empty']:
    ref=Path(f'.cache/native-perf/mlp-sweep-ba018a0/quality-fc2-{case}-1')
    for mode in modes:
        p=root/f'quality-{mode}-{case}-1'
        m=json.loads((p/'metrics.json').read_text())
        parity={f:(p/f).read_bytes()==(ref/f).read_bytes() for f in files}
        report['quality'][f'{case}/{mode}']={'count':m['count'],'byte_equal':parity}
        if not all(parity.values()):report['gate_failures'].append(p.name)
Path('experiments/results/native_rtx2060/int8-cache-ba018a0.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
sys.exit(bool(report['gate_failures']))
