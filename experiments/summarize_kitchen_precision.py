"""Summarize the paired FP16 / attention-only timing run."""
import json
import statistics
from pathlib import Path
import numpy as np

root=Path('.cache/native-perf/kitchen-precision-ba018a0')
report={}
for mode in ['fp16','attention_only']:
    runs=[]
    for path in sorted(root.glob(mode+'-*')):
        metrics=json.loads((path/'metrics.json').read_text())
        stages=json.loads((path/'stages.json').read_text())
        runs.append(dict(path=str(path),median_ms=metrics['median_wall_seconds']*1000,
                         wall_ms=[v*1000 for v in metrics['wall_seconds']],
                         stage_mean_ms=np.mean(stages['gpu_ms'],axis=0).tolist()))
    assert len(runs)==2, 'paired runs incomplete'
    values=[v for run in runs for v in run['wall_ms']]
    report[mode]=dict(runs=runs,median_ms=statistics.median(values),p95_ms=float(np.percentile(values,95)),
                      stage_mean_ms=np.mean([r['stage_mean_ms'] for r in runs],axis=0).tolist())
Path('experiments/results/native_rtx2060/kitchen-precision-ba018a0.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({mode:{k:v for k,v in data.items() if k!='runs'} for mode,data in report.items()},indent=2))
