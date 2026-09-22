"""Page standalone real-video history and compare every byte to resident outputs."""
import argparse,hashlib,json,os,re,subprocess,tempfile,time
from pathlib import Path
import psutil

def run(command,log,cache=None):
    peak=0;started=time.perf_counter()
    with log.open('w') as stream:
        child=subprocess.Popen(command,env={**os.environ,'PATH':'/nonexistent'},stdout=stream,stderr=subprocess.STDOUT)
        process=psutil.Process(child.pid)
        while child.poll() is None:
            try:peak=max(peak,process.memory_info().rss)
            except psutil.NoSuchProcess:pass
            time.sleep(.02)
        assert child.returncode==0,log.read_text()
    if cache is not None:assert not list(cache.iterdir()),'temporary history leaked'
    text=log.read_text();resident=[int(x) for x in re.findall(r'resident_history_bytes=(\d+)',text)];archived=[int(x) for x in re.findall(r'archived_history_bytes=(\d+)',text)]
    return dict(seconds=time.perf_counter()-started,sampled_peak_host_rss=peak,max_retained_history_logical_bytes=max(resident),max_retained_history_disk_bytes=max(archived))

def compare(expected,actual):
    names={p.name for p in expected.iterdir()};assert names=={p.name for p in actual.iterdir()}
    for name in names:assert hashlib.sha256((expected/name).read_bytes()).digest()==hashlib.sha256((actual/name).read_bytes()).digest(),name
    return len(names)

def main():
    p=argparse.ArgumentParser();p.add_argument('executable',type=Path);p.add_argument('store',type=Path);p.add_argument('fixtures',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--long-frames',type=int,default=32);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);base=[str(a.executable.resolve()),str(a.store.resolve()),'cuda'];rows=[]
    for mode in ['fp16','bf16_reference','fp32']:
        for case in ['points','masks']:
            target=a.output/f'{mode}-{case}';log=a.output/f'{mode}-{case}.log'
            with tempfile.TemporaryDirectory(prefix='sam3-video-history-') as directory:
                metrics=run(base+[mode,str((a.fixtures/'frames.txt').resolve()),str((a.fixtures/f'{case}.txt').resolve()),str(target.resolve()),directory],log,Path(directory))
            row=dict(mode=mode,case=case,exact_files=compare(a.fixtures/f'{mode}-{case}',target),**metrics);rows.append(row);print(json.dumps(row),flush=True)
    # Repeat three real decoded images to stress retention length, not to claim
    # a 32-frame unique-content accuracy benchmark. No model/prompt restriction.
    manifest=a.output/'long-frames.txt';names=(a.fixtures/'frames.txt').read_text().splitlines();manifest.write_text('\n'.join(str((a.fixtures/names[i%len(names)]).resolve()) for i in range(a.long_frames))+'\n')
    commands=a.output/'long-commands.txt';commands.write_text(f'points 0 11 1 1 0 .6 .68 1 .25 .3 0\npreflight 1\npropagate 0 {a.long_frames-1} 0 1 0 0 0\nreset\n')
    for policy in ['resident','paged']:
        target=a.output/f'long-{policy}'
        with tempfile.TemporaryDirectory(prefix='sam3-long-history-') as directory:
            command=base+['fp16',str(manifest.resolve()),str(commands.resolve()),str(target.resolve())]
            if policy=='paged':command.append(directory)
            metrics=run(command,a.output/f'long-{policy}.log',Path(directory));row=dict(mode='fp16',case=f'long-{policy}',frames=a.long_frames,**metrics)
            if policy=='paged':row['exact_files']=compare(a.output/'long-resident',target)
            rows.append(row);print(json.dumps(row),flush=True)
    a.report.write_text(json.dumps(dict(cases=rows,scope='Same decoded PPM pixels, native child PATH=/nonexistent. Short outputs compare prior same-mode original-parity fixtures; retention stress repeats three images. Exact byte equality includes masks, logits and metadata. Host RSS sampled via psutil every 20ms; single-run end-to-end timings include loading, disk I/O and allocator effects, not an inference benchmark. History bytes exclude model, active working set, annotation inputs and small selection metadata.'),indent=2)+'\n')
if __name__=='__main__':main()
