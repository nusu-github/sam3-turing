import os,subprocess,sys,json,statistics
from pathlib import Path
for mode in ['exact','threads128','threads512','readonly','readonly','threads512','threads128','exact']:
 env=os.environ.copy();env['SAM3_EXPERIMENT_BOUNDARY']=mode
 root=Path('.cache/native-perf/boundary-sweep');root.mkdir(exist_ok=True)
 idx=1
 while (root/f'{mode}-{idx}').exists():idx+=1
 out=root/f'{mode}-{idx}';result=Path(f'experiments/results/native_rtx2060/boundary-{mode}-{idx}.json')
 p=subprocess.run([sys.executable,'experiments/monitor_native_bench.py',str(out),'build/native-windows-cu130/sam3_boundary_bench.exe',str(result)],env=env,capture_output=True,text=True)
 if p.returncode:print(p.stdout+p.stderr,flush=True);raise SystemExit(p.returncode)
 r=next(x for x in json.loads(result.read_text())['cases'] if x['n']==4736)
 print(mode,idx,'separate',statistics.median(r['separate_ms']),'fused',statistics.median(r['fused_ms']),flush=True)
