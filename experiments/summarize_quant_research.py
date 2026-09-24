"""Summarize completed research runs; cold quality timings are never speed claims."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root):
    signature=json.loads((root/'run.json').read_text())
    cases=sorted(root.glob('*/complete.json'))
    comparisons=[]
    metrics=[]
    peaks=[]
    duration=0
    for case in cases:
        row=json.loads(case.read_text())
        if 'comparison' in row:
            comparisons.append((row['case'],row['comparison']))
        metrics.append(json.loads((case.parent/'metrics.json').read_text()))
        telemetry=json.loads((case.parent/'telemetry.json').read_text())
        peaks.append(telemetry['sampled_peak_bytes'])
        duration+=telemetry['wall_seconds']
    summary=dict(run=root.name,kind=signature['args']['kind'],mode=signature['args']['mode'],
                 completed=len(cases),process_wall_seconds=duration,
                 max_peak_allocated_bytes=max((m['cold_peak_allocated_bytes'] for m in metrics),default=None),
                 max_sampled_whole_gpu_bytes=max(peaks,default=None))
    if comparisons:
        summary.update(compared=len(comparisons),passed=sum(v['gate_pass'] for _,v in comparisons),
                       nonempty=sum(v['reference_count']>0 for _,v in comparisons),
                       nonempty_passed=sum(v['reference_count']>0 and v['gate_pass'] for _,v in comparisons),
                       count_changes=sum(v['reference_count']!=v['candidate_count'] for _,v in comparisons),
                       failed_cases=[c for c,v in comparisons if not v['gate_pass']],
                       min_mask_iou=min((v['min_mask_iou'] for _,v in comparisons if v['min_mask_iou'] is not None),default=None),
                       max_score_error=max((v['max_score_error'] for _,v in comparisons if v['max_score_error'] is not None),default=None),
                       max_box_error_px=max((v['max_box_error_px'] for _,v in comparisons if v['max_box_error_px'] is not None),default=None))
    if signature['args']['kind']=='timing':
        summary['timing_by_case']={c.parent.name:dict(median_ms=float(np.median(m['wall_seconds'])*1000),
                                                       p95_ms=float(np.percentile(m['wall_seconds'],95)*1000),
                                                       samples=len(m['wall_seconds'])) for c,m in zip(cases,metrics)}
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=[summarize(root) for root in args.runs]
    rendered=json.dumps(result,indent=2)+'\n'
    if args.output: args.output.write_text(rendered)
    print(rendered)


if __name__=='__main__':
    main()
