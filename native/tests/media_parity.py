"""Native C media decode against Pillow and the matching FFmpeg CLI.

Python and ffmpeg are fixture/oracle tools only; the C client runs with neither
on PATH. Fixtures exercise lossy/lossless images, alpha, palette, numeric folders,
B-frame draining, random/reverse indexing, VFR timestamps, Unicode and PNG output.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('probe',type=Path);p.add_argument('directory',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    root=a.directory.resolve();root.mkdir(parents=True,exist_ok=True);inputs=root/'inputs';inputs.mkdir(exist_ok=True);results=[]
    y,x=np.mgrid[:48,:64];rgb=np.stack([(x*13+y*7)%256,(x*3+y*19)%256,(x*23+y*5)%256],-1).astype(np.uint8)
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env['PATH']='/nonexistent'
    def run(path,mode='image',indices=None):
        out=root/'outputs'/path.name;out.mkdir(parents=True,exist_ok=True)
        cmd=[str(a.probe.resolve()),str(path),mode,str(out)]+[str(i) for i in (indices or [])]
        r=subprocess.run(cmd,env=env,capture_output=True,text=True);(out/'client.log').write_text(r.stdout+r.stderr);assert r.returncode==0,(path,r.stderr)
        return out
    def read(out,name='frame-0',field='rgb',dtype=np.uint8):
        meta=json.loads((out/(name+'.json')).read_text());return np.fromfile(out/(name+'-'+field+'.bin'),dtype).reshape(meta[field]['shape'])
    # Each is compared to actual Pillow conversion, not only to the encoder input.
    for name,im,kwargs in [
        ('rgb.png',Image.fromarray(rgb),{}),('rgba.png',Image.fromarray(np.dstack([rgb,(x*7+y*3).astype(np.uint8)]),'RGBA'),{}),
        ('palette.png',Image.fromarray(rgb).quantize(colors=64),{}),('gray.png',Image.fromarray(rgb[:,:,0]),{}),
        ('rgb.bmp',Image.fromarray(rgb),{}),('rgb.tiff',Image.fromarray(rgb),{}),('rgb.webp',Image.fromarray(rgb),{'lossless':True}),
        ('彩色.jpeg',Image.fromarray(rgb),{'quality':91,'subsampling':2}),('progressive.jpg',Image.fromarray(rgb),{'quality':88,'progressive':True}),
        ('gray.jpg',Image.fromarray(rgb[:,:,0]),{'quality':93}),('cmyk.jpg',Image.fromarray(rgb).convert('CMYK'),{'quality':90})]:
        path=inputs/name;im.save(path,**kwargs);out=run(path);actual=read(out).transpose(1,2,0);expected=np.array(Image.open(path).convert('RGB'))
        exact=np.array_equal(actual,expected);row=dict(case=name,oracle='Pillow RGB',exact=bool(exact),max_abs=int(np.max(np.abs(actual.astype(int)-expected.astype(int)))));results.append(row)
        assert np.array_equal(np.array(Image.open(out/'roundtrip.png')),actual)
        assert np.array_equal(read(out,'retained'),read(out))
    folder=inputs/'順序';folder.mkdir(exist_ok=True)
    for name,offset in [('10.png',10),('2.png',2),('1.png',1)]:Image.fromarray((rgb+offset).astype(np.uint8)).save(folder/name)
    out=run(folder,'video',[2,0,1,2]);assert np.fromfile(out/'info-frames.bin',np.int64)[0]==3
    assert all(np.array_equal(read(out,'frame-'+str(i)).transpose(1,2,0),np.array(Image.open(folder/name))) for i,name in enumerate(['1.png','2.png','10.png']))
    results.append(dict(case='numeric_unicode_folder_reverse_repeat',exact=True))
    frames=np.stack([np.roll(rgb,i*3,axis=1) for i in range(9)])
    raw=inputs/'frames.rgb';raw.write_bytes(frames.tobytes())
    def ffmpeg(args):subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*args],check=True)
    common=['-f','rawvideo','-pixel_format','rgb24','-video_size','64x48','-framerate','7','-i',str(raw)]
    for name,codec in [('lossless.mkv',['-c:v','ffv1','-pix_fmt','bgr0']),('bframes.mp4',['-c:v','libx264','-bf','3','-g','7','-pix_fmt','yuv420p']),('bt709.mp4',['-c:v','libx264','-bf','2','-pix_fmt','yuv420p','-colorspace','bt709','-color_range','tv'])]:
        path=inputs/name;ffmpeg(common+codec+[str(path)])
    rotated=inputs/'rotated.mp4';ffmpeg(['-i',str(inputs/'bframes.mp4'),'-c','copy','-metadata:s:v:0','rotate=90',str(rotated)])
    for name,orientation in [('hflip.mp4',['-display_hflip']),('vflip.mp4',['-display_vflip']),('fliprot.mp4',['-display_rotation','90','-display_hflip']),('rotate180.mp4',['-display_rotation','180']),('rotate270.mp4',['-display_rotation','270']),('fliprot270.mp4',['-display_rotation','270','-display_hflip'])]:
        ffmpeg(orientation+['-i',str(inputs/'bframes.mp4'),'-c','copy',str(inputs/name)])
    listing=inputs/'vfr.txt';listing.write_text(''.join("file '順序/"+name+"'\nduration "+str(duration)+"\n" for name,duration in [('1.png',.04),('2.png',.16),('10.png',.08)])+"file '順序/10.png'\n")
    ffmpeg(['-f','concat','-safe','0','-i',str(listing),'-fps_mode','vfr','-c:v','ffv1',str(inputs/'variable.mkv')])
    for name in ['lossless.mkv','bframes.mp4','bt709.mp4','rotated.mp4','hflip.mp4','vflip.mp4','fliprot.mp4','rotate180.mp4','rotate270.mp4','fliprot270.mp4','variable.mkv']:
        path=inputs/name;meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_streams','-of','json',str(path)]));count=len(meta['frames']);order=list(range(count))+list(range(count-1,-1,-1))+[0,count-1,0];out=run(path,'video',order)
        assert np.fromfile(out/'info-frames.bin',np.int64)[0]==count
        # Match the C converter's interpolation explicitly; no spatial resize.
        expected_path=out/'ffmpeg.rgb';ffmpeg(['-i',str(path),'-sws_flags','bilinear','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24',str(expected_path)])
        shape=read(out).shape;expected=np.fromfile(expected_path,np.uint8).reshape(count,shape[1],shape[2],3)
        errors=[];times=[]
        for i in range(count):
            actual=read(out,'frame-'+str(i)).transpose(1,2,0);errors.append(int(np.max(np.abs(actual.astype(int)-expected[i].astype(int)))))
            actual_time=float(read(out,'frame-'+str(i),'seconds',np.float64));target=float(meta['frames'][i]['best_effort_timestamp_time']);times.append(abs(actual_time-target))
        row=dict(case=name,oracle='FFmpeg bilinear RGB24 and ffprobe decoded frame timestamps',frames=count,exact=all(e==0 for e in errors),max_abs=max(errors),max_timestamp_error=max(times));results.append(row)
        assert max(times)<1e-6
    a.report.write_text(json.dumps(dict(cases=results,scope='Decoded U8 RGB, exact frame count/ordering, random/reverse/repeated reads, timestamps, owning C results, PNG writing. No model or all-format/color/HDR claim.',all_exact=all(r['exact'] for r in results)),indent=2)+'\n');print(json.dumps(results,indent=2));assert all(r['exact'] for r in results)


if __name__=='__main__':main()
