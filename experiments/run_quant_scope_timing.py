"""Diagnostic paired timing for the Round 4 scope candidates (not quality acceptance)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run_quant_research import DATA, EXE, ROOT, environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True,help='Unique series name')
    parser.add_argument('--preheat-repeats',type=int,default=180)
    args=parser.parse_args()
    if args.preheat_repeats<0: parser.error('preheat repeats must be nonnegative')
    warm=ROOT/'timing-warmups'/args.name
    if warm.exists(): raise RuntimeError('choose a new series name; warmup already exists')
    warm.mkdir(parents=True)
    row=next(r for r in json.loads(DATA.read_text())['images'] if r['role']=='development' and r['image_id']==7574)
    assert hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()==row['sha256']
    env=environment('selective','all',None)
    command=[sys.executable,'experiments/monitor_native_bench.py',str(warm),str(EXE),
             '.cache/native-weights-sam3',row['image'],'sam3/assets/bpe_simple_vocab_16e6.txt.gz',
             row['cases'][0]['prompt_file'],str(args.preheat_repeats),'5',str(warm)]
    subprocess.run(command,env=env,check=True)
    common=['--mode','calibrated','--scope','mask:0xfffffffe','--mean-bias',
            '--calibration',str(ROOT/'calibrations/a050-none-mean-bias'),
            '--projection-scope','mask:0xffffffbf']
    configs={'selective':['--mode','selective'],
             'all-attention':common,
             'global-attention':common+['--attention','kitchen']}
    runs={name:[] for name in configs}
    schedule=[]
    for trial in range(2):
        for image in ([7574,1425] if trial==0 else [1425,7574]):
            order=list(configs) if trial==0 else list(reversed(configs))
            for candidate in order:
                name=f'{args.name}-{candidate}-{image}-p{trial}'
                command=[sys.executable,'experiments/run_quant_research.py','timing',
                         '--name',name,'--image-id',str(image),'--primary-only',*configs[candidate]]
                subprocess.run(command,check=True)
                runs[candidate].append(str(ROOT/'runs'/name))
                schedule.append(dict(candidate=candidate,image_id=image,trial=trial,run=name))
    destination=Path('experiments/results/native_rtx2060')
    for candidate in ['all-attention','global-attention']:
        subprocess.run([sys.executable,'experiments/summarize_quant_timing.py',
                        *runs['selective'],*runs[candidate],'--output',
                        str(destination/f'{args.name}-{candidate}-summary.json')],check=True)
    report=dict(preheat_repeats=args.preheat_repeats,warmup_directory=str(warm),
                warmup_telemetry=json.loads((warm/'telemetry.json').read_text()),
                protocol='Three configurations; two images, two fresh processes/configuration/image. Reverse configuration and image order on pass 2. 5 warmups + 15 timed samples per process. Diagnostic only; candidates have not passed quality gates.',
                schedule=schedule,runs=runs)
    (destination/f'{args.name}-protocol.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
