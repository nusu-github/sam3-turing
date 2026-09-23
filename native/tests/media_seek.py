"""Exact native seek/reverse reads versus a sequential FFmpeg pixel oracle.

The C executable has no Python/FFmpeg executable on PATH. Every read is checked,
including repeats; counters separate initial indexing from subsequent decoding.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import struct
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('probe',type=Path);p.add_argument('directory',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    root=a.directory.resolve();root.mkdir(parents=True,exist_ok=True)
    def ff(args):subprocess.run(['ffmpeg','-v','error','-y',*args],check=True)
    count,h,w=360,96,128
    y,x=np.mgrid[:h,:w];frames=np.stack([np.stack([(x*7+y*3+i*11)%256,(x*3+y*17+i*13)%256,(x*23+y*5+i*7)%256],-1) for i in range(count)]).astype(np.uint8)
    raw=root/'input.rgb';raw.write_bytes(frames.tobytes());common=['-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','30','-i',str(raw)]
    fixtures=[('mpeg2.ts',['-c:v','mpeg2video','-g','12','-bf','2']),('closed.mp4',['-c:v','libx264','-pix_fmt','yuv420p','-x264-params','keyint=24:min-keyint=24:scenecut=0:open-gop=0','-bf','3']),('open.mp4',['-c:v','libx264','-pix_fmt','yuv420p','-x264-params','keyint=24:min-keyint=24:scenecut=0:open-gop=1','-bf','3']),('vfr.mkv',['-vf','setpts=PTS+floor(N/17)*3','-fps_mode','passthrough','-c:v','ffv1','-g','12']),('duplicate.mkv',['-vf','setpts=floor(N/2)*2','-fps_mode','passthrough','-c:v','ffv1','-g','12']),('missing.h264',['-c:v','libx264','-pix_fmt','yuv420p','-g','24','-f','h264'])]
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.update(PATH='/nonexistent',SAM3_PROBE_CACHE_BYTES=str(w*h*3*16),SAM3_PROBE_SEQUENCE='1')
    rows=[]
    for name,args in fixtures+[("bad-index.mp4",None)]:
        path=root/name
        if args is not None:ff(common+args+[str(path)])
        else:
            # Deliberately inaccurate sync-sample table; encoded samples and
            # sequential pixels remain intact, exercising rejected seeks.
            data=bytearray((root/'closed.mp4').read_bytes());pos=data.index(b'stss');n=struct.unpack_from('>I',data,pos+8)[0]
            for k in range(1,n):
                off=pos+12+k*4;sample=struct.unpack_from('>I',data,off)[0];struct.pack_into('>I',data,off,sample+3)
            path.write_bytes(data)
        oracle=root/(name+'.rgb');ff(['-i',str(path),'-sws_flags','bilinear','-fps_mode','passthrough','-pix_fmt','rgb24','-f','rawvideo',str(oracle)])
        expected=np.memmap(oracle,dtype=np.uint8,mode='r',shape=(count,h,w,3))
        meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-of','json',str(path)]))['frames'];assert len(meta)==count
        rng=np.random.default_rng(42)
        # Start by seeking near EOF, then reverse across every GOP/cache boundary.
        orders={'reverse':list(range(count-1,-1,-1)),'random':rng.integers(0,count,80).tolist(),'forward':list(range(count))}
        if name in ('duplicate.mkv','missing.h264','bad-index.mp4','mpeg2.ts'):orders={'reverse':list(range(count-1,-1,-1))}
        if name=='closed.mp4':orders.update(zero_cache=[359,100,100,0,250,25],tiny_cache=[359,100,0,250,25],threaded=[359,100,0,250,25])
        for label,order in orders.items():
            env['SAM3_PROBE_CACHE_BYTES']=str(0 if label=='zero_cache' else 1 if label=='tiny_cache' else w*h*3*16)
            env['SAM3_PROBE_THREADS']='4' if label=='threaded' else '1'
            out=root/(name+'-'+label);out.mkdir(exist_ok=True)
            start=time.monotonic();r=subprocess.run([str(a.probe.resolve()),str(path),'video',str(out),*map(str,order)],env=env,capture_output=True,text=True);elapsed=time.monotonic()-start
            (out/'client.log').write_text(r.stdout+r.stderr);assert r.returncode==0,(name,label,r.stderr)
            for j,i in enumerate(order):
                tag=f'read-{j}-frame-{i}'
                actual=np.fromfile(out/(tag+'-rgb.bin'),np.uint8).reshape(3,h,w).transpose(1,2,0)
                assert np.array_equal(actual,expected[i]),(name,label,j,i,int(np.max(np.abs(actual.astype(int)-expected[i].astype(int)))))
                seconds=np.fromfile(out/(tag+'-seconds.bin'),np.float64)[0];target=meta[i].get('best_effort_timestamp_time')
                assert (np.isnan(seconds) if target is None else abs(seconds-float(target))<1e-6),(name,i,seconds,target)
            schema=json.loads((out/'stats.json').read_text());stats={k:np.fromfile(out/('stats-'+k+'.bin'),np.bool_ if k=='indexed_seek_enabled' else np.int64)[0].item() for k in schema}
            assert stats['cache_bytes']<=stats['cache_limit_bytes']
            assert stats['index_decoded_frames']==count
            if name in ('duplicate.mkv','missing.h264'):assert not stats['indexed_seek_enabled'] and stats['seek_attempts']==0
            if name=='mpeg2.ts':assert stats['seek_fallbacks']>0 and not stats['indexed_seek_enabled']
            if name=='closed.mp4' and label=='reverse':assert stats['read_decoded_frames']<count*3 and stats['seek_attempts']>0 and stats['seek_fallbacks']==0
            row=dict(case=name,order=label,reads=len(order),all_pixels_exact=True,all_timestamps_match=True,elapsed_seconds=elapsed,stats=stats,old_replay_decoded_frames=(count*(count+1)//2 if label=='reverse' else None));rows.append(row);print(json.dumps(row),flush=True)
    a.report.write_text(json.dumps(dict(cases=rows,scope='Every requested frame, not only the last read of each index. Elapsed includes initial scan, output files and process startup; not model inference.'),indent=2)+'\n')


if __name__=='__main__':main()
