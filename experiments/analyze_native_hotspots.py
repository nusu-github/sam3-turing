"""Attribute actual GPU kernel durations via CUDA launch correlation to NVTX.

Inclusive ranges overlap and must not be added; leaf buckets do not overlap.
No CPU NVTX duration is interpreted as GPU time. GPU gaps/copies are excluded.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

root=Path(sys.argv[1] if len(sys.argv)>1 else '.cache/native-perf/hotspots-current')
reports={}
for path in sorted(root.glob('*.sqlite')):
    db=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True)
    strings=dict(db.execute('select id,value from StringIds'))
    ranges=defaultdict(list)
    for start,end,tid,text,textid in db.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null'):
        label=text if text is not None else strings.get(textid,'')
        if label.startswith(('vision.','detector.')):
            ranges[tid].append((start,end,label))
    assert ranges, 'No application NVTX ranges captured'
    launches={cid:(start,tid) for start,tid,cid in db.execute('select start,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
    # Sort by start; a sweep per CPU thread maintains the active nesting stack.
    for tid in ranges: ranges[tid].sort(key=lambda r:(r[0],-r[1]))
    tagged=[]
    for start,end,cid,name,regs,smem in db.execute('select start,end,correlationId,demangledName,registersPerThread,dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL'):
        if cid in launches:
            cpu,tid=launches[cid]
            tagged.append((cpu,tid,start,end,strings[name],regs,smem))
        else: tagged.append((-1,-1,start,end,strings[name],regs,smem))
    tagged.sort()
    indices=defaultdict(int); stacks=defaultdict(list)
    inclusive=defaultdict(float);leaf=defaultdict(float);kernels=defaultdict(float)
    categories=defaultdict(float); counts=defaultdict(int)
    category_kernels=defaultdict(lambda:defaultdict(float))
    total=0
    for cpu,tid,start,end,kernel,regs,smem in tagged:
        rs=ranges[tid];idx=indices[tid];stack=stacks[tid]
        while idx<len(rs) and rs[idx][0]<=cpu:
            r=rs[idx]
            stack[:]=[a for a in stack if a[1]>r[0]]
            stack.append(r);idx+=1
        indices[tid]=idx
        stack[:]=[r for r in stack if r[1]>=cpu]
        names=[r[2] for r in stack]
        duration=(end-start)/1e6
        total+=duration;kernels[kernel]+=duration
        for name in set(names): inclusive[name]+=duration
        last=names[-1] if names else 'unattributed'
        leaf[last]+=duration; counts[last]+=1
        section=next((n for n in names if n in ('detector.encoder','detector.decoder','detector.heads')),'')
        category=last
        if section:
            if 'detector.qkv' in names: category=section+'.qkv'
            elif 'detector.sdpa' in names: category=section+'.sdpa'
            elif 'detector.explicit_attention' in names: category=section+'.explicit_attention'
            elif 'detector.decoder.rpb' in names: category='detector.decoder.rpb'
            elif 'detector.decoder.ffn_fp32' in names: category='detector.decoder.ffn_fp32'
            elif last.startswith('detector.linear.'):
                category=section+('.output_projection' if 'out_proj' in last else '.linear_other')
            else: category=section+'.other'
        if last.startswith('vision.block.'):category='vision.norm_residual_other'
        if last.startswith('vision.neck.'):category='vision.neck'
        if last=='vision.total':category='vision.embedding_position_other'
        categories[category]+=duration
        category_kernels[category][kernel]+=duration
    metrics=json.loads((root/path.stem/'metrics.json').read_text())
    repeats=metrics['repeats']
    mode=path.stem.split('-')[0]
    reference=Path(sys.argv[3] if len(sys.argv)>3 else '.cache/native-perf/approximation-v3')
    if not (reference/'metrics.json').exists(): reference=reference/f'timing-{mode}-truck-1'
    files=['masks.u8.bin','scores.f32.bin','boxes.f32.bin','queries.i64.bin']
    hashes={name:hashlib.sha256((root/path.stem/name).read_bytes()).hexdigest() for name in files}
    identical=all((root/path.stem/name).read_bytes()==(reference/name).read_bytes() for name in files)
    assert identical, f'Instrumentation changed {path.stem} outputs versus prior run'
    def ordered(d): return dict(sorted(((k,v/repeats) for k,v in d.items()),key=lambda x:-x[1]))
    reports[path.stem]={'repeats':repeats,'wall_median_ms':metrics['median_wall_seconds']*1000,
        'gpu_kernel_ms_per_inference':total/repeats,'inclusive_ms':ordered(inclusive),
        'leaf_ms':ordered(leaf),'categories_ms':ordered(categories),'kernel_ms':ordered(kernels),
        'kernel_count':len(tagged),'leaf_launches_per_inference':{k:v/repeats for k,v in counts.items()},
        'category_kernels_ms':{k:ordered(v) for k,v in category_kernels.items()},
        'output_sha256':hashes,'identical_to_prior_same_mode':identical}
dest=Path(sys.argv[2] if len(sys.argv)>2 else 'experiments/results/native_rtx2060/hotspots-current.json')
dest.write_text(json.dumps(reports,indent=2)+'\n')
for name,r in reports.items():
    print(name,'wall',r['wall_median_ms'],'kernel sum',r['gpu_kernel_ms_per_inference'])
    print(json.dumps(r['categories_ms'],indent=2))
