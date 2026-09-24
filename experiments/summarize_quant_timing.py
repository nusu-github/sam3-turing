"""Compare fresh-process timing pairs; report quality separately from speed."""
import argparse
import json
from pathlib import Path

import numpy as np

from native_quality import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[]
    versions=set()
    for run in args.runs:
        sig=json.loads((run/'run.json').read_text())
        assert sig['args']['kind']=='timing'
        versions.add(tuple(sig[k] for k in ['binary_sha256','runtime_sha256','manifest_sha256']))
        for case in run.glob('*/complete.json'):
            m=json.loads((case.parent/'metrics.json').read_text())
            t=json.loads((case.parent/'telemetry.json').read_text())
            assert m['warmups']==5 and m['repeats']==15 and t['exit_code']==0
            mode=sig['args']['mode']
            rows.append(dict(run=run.name,case=case.parent.name,mode=mode,signature=sig,
                             ms=[v*1000 for v in m['wall_seconds']],
                             median_ms=m['median_wall_seconds']*1000,
                             peak_allocated_bytes=m['peak_allocated_bytes'],
                             sampled_whole_gpu_peak_bytes=t['sampled_peak_bytes'],
                             temperature_start_c=t['samples'][0].get('temperature_c'),
                             temperature_end_c=t['samples'][-1].get('temperature_c'),
                             process_peak_sm_mhz=max((v.get('sm_mhz',0) for v in t['samples']),default=None),
                             quality=compare(run.parent/'development-fp16'/case.parent.name,case.parent)))
    assert len(versions)==1, 'incomparable runtime/data versions'
    modes=sorted({r['mode'] for r in rows})
    assert modes==['calibrated','selective'], 'expected one calibrated candidate and selective reference'
    summaries={}
    for case in sorted({r['case'] for r in rows})+['pooled']:
        summary={}
        for mode in modes:
            selected=[r for r in rows if r['mode']==mode and (case=='pooled' or r['case']==case)]
            if case!='pooled': assert len(selected)==2, 'two fresh processes per mode/case required'
            fingerprints={json.dumps(r['signature']['environment'],sort_keys=True) for r in selected}
            assert len(fingerprints)==1, 'candidate configuration changed across trials'
            samples=[v for r in selected for v in r['ms']]
            summary[mode]=dict(processes=len(selected),samples=len(samples),
                               median_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),
                               process_medians_ms=[r['median_ms'] for r in selected],
                               max_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in selected),
                               max_sampled_whole_gpu_bytes=max(r['sampled_whole_gpu_peak_bytes'] for r in selected))
        summary['median_reduction_percent']=(1-summary['calibrated']['median_ms']/summary['selective']['median_ms'])*100
        summary['p95_change_percent']=(summary['calibrated']['p95_ms']/summary['selective']['p95_ms']-1)*100
        summaries[case]=summary
    result=dict(scope='Diagnostic speed comparison; independent quality gate is required before adoption.',
                protocol='Two fresh processes/mode/image, 5 warmups and 15 samples; forward/reverse order.',
                results=summaries,processes=rows)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
