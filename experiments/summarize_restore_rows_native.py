"""Compare restored output indexing while requiring binary output parity."""
import json
from pathlib import Path
import statistics as st
import numpy as np
root=Path('.cache/native-perf/restore-rows-ba018a0')
previous=Path('.cache/native-perf/projection-ba018a0')
files=['masks.u8.bin','scores.f32.bin','boxes.f32.bin','queries.i64.bin']
report={'base':'ba018a0','mlp':'int8_boundary','timing':{},'quality':{}}
for mode in ['base','rows','qkv','qkv_rows']:
    runs=[]
    for p in sorted(root.glob(f'timing-{mode}-truck-*')):
        m=json.loads((p/'metrics.json').read_text())
        s=json.loads((p/'stages.json').read_text())
        t=json.loads((p/'telemetry.json').read_text())
        ref=previous/('timing-qkv-truck-1' if mode.startswith('qkv') else 'timing-exact-truck-1')
        identical=all((p/f).read_bytes()==(ref/f).read_bytes() for f in files)
        assert identical, f'Output mismatch: {p}'
        runs.append({'path':str(p),'median_ms':m['median_wall_seconds']*1000,
                     'wall_ms':[v*1000 for v in m['wall_seconds']],
                     'stage_mean_ms':np.mean(s['gpu_ms'],axis=0).tolist(),
                     'peak_allocated_bytes':m['peak_allocated_bytes'],
                     'sm_mhz_tail_median':st.median(v['sm_mhz'] for v in t['samples'][-100:]),
                     'temperature_max':max(v['temperature_c'] for v in t['samples']),
                     'byte_equal_to_flat_restore':identical})
    if runs:
        vals=[v for r in runs for v in r['wall_ms']]
        report['timing'][mode]={'runs':runs,'pooled_median_ms':st.median(vals),
            'p95_ms':float(np.percentile(vals,95)),
            'stage_mean_ms':np.mean([r['stage_mean_ms'] for r in runs],axis=0).tolist()}
for case in ['truck','bag','child','wheel','empty']:
    for mode in ['rows','qkv_rows']:
        p=root/f'quality-{mode}-{case}-1'
        if not (p/'metrics.json').exists():continue
        ref=previous/f"quality-{'qkv' if mode=='qkv_rows' else 'exact'}-{case}-1"
        identical=all((p/f).read_bytes()==(ref/f).read_bytes() for f in files)
        assert identical, f'Output mismatch: {p}'
        report['quality'][f'{case}/{mode}']={'byte_equal_to_flat_restore':identical,
            'count':json.loads((p/'metrics.json').read_text())['count']}
Path('experiments/results/native_rtx2060/restore-rows-ba018a0.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
