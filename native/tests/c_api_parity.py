"""Pure C11 standalone client: model outputs, owned results and callback boundaries."""
import argparse,json,os,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch

DTYPES={1:torch.uint8,2:torch.bool,3:torch.int32,4:torch.int64,5:torch.float16,6:torch.bfloat16,7:torch.float32,8:torch.float64}
def read(root,name):
    values={};meta=json.loads((root/f'{name}.json').read_text())
    for key,spec in meta.items():
        data=(root/f'{name}-{key.replace("/","_")}.bin').read_bytes();assert len(data)==spec['bytes']
        values[key]=torch.frombuffer(bytearray(data),dtype=DTYPES[spec['dtype']]).clone().reshape(spec['shape']) if data else torch.empty(spec['shape'],dtype=DTYPES[spec['dtype']])
    return values

def fixtures(root,kind):
    output=root/'inputs'/kind;output.mkdir(parents=True,exist_ok=True)
    paths=[Path('assets/images/truck.jpg')] if kind in ['image','ground'] else [Path(f'assets/videos/0001/{i}.jpg') for i in range(3)]
    for i,path in enumerate(paths):
        image=Image.open(path).convert('RGB')
        if kind=='ground':image=image.resize((97,73),Image.Resampling.BILINEAR)
        (output/f'{i}.rgb').write_bytes(np.asarray(image).tobytes());(output/'size.txt').write_text(f'{image.height} {image.width}\n')
    return output

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('executable',type=Path);p.add_argument('store',type=Path);p.add_argument('reference',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['sam3','sam3.1']);p.add_argument('--modes',nargs='+',default=['fp16','bf16_reference','fp32']);p.add_argument('--kinds',nargs='+',default=['image','video','ground']);a=p.parse_args()
    rows=[];inputs={kind:fixtures(a.output,kind) for kind in a.kinds}
    for model in a.models:
        for mode in a.modes:
            for kind in a.kinds:
                name=f'{model}-{mode}-{kind}';target=a.output/name;target.mkdir(parents=True,exist_ok=True);print('START',name,flush=True)
                command=[str(a.executable.resolve()),str(a.store.resolve()),str(3 if model=='sam3' else 31),'cuda',str({'fp32':0,'fp16':1,'bf16_reference':2}[mode]),kind,str(inputs[kind].resolve()),str(target.resolve())]
                cache=a.output/f'{name}-cache'
                if kind=='ground':command.append(str(Path('sam3/assets/bpe_simple_vocab_16e6.txt.gz').resolve()))
                if kind=='video' and model=='sam3.1':cache.mkdir(exist_ok=True);command.append(str(cache.resolve()))
                child=subprocess.run(command,env={**os.environ,'PATH':'/nonexistent'},capture_output=True,text=True);(a.output/f'{name}.log').write_text(child.stdout+child.stderr);assert child.returncode==0,child.stdout+child.stderr
                if cache.exists():assert not list(cache.iterdir())
                checks=0
                if kind=='image':
                    for stage in ['initial','refined']:
                        expected=torch.load(a.reference/'interactive-image-native-v1'/f'{model}-{mode}.{stage}.reference.pt',weights_only=True,map_location='cpu');actual=read(target,stage)
                        for key,value in expected.items():torch.testing.assert_close(actual[key],value,rtol=0,atol=0);checks+=1
                    batch=read(target,'batch');assert batch['masks'].shape[:2]==(2,3)
                elif kind=='video':
                    base=a.reference/('tracking-video-v1' if model=='sam3' else 'multiplex-video-v1')/(mode if model=='sam3' else f'{mode}-points')
                    paths=sorted(base.glob('*.json'));assert paths,f'no video reference fixtures in {base}'
                    for path in paths:
                        meta=json.loads(path.read_text());actual=read(target,path.stem);shape=(len(meta['ids']),1,meta['height'],meta['width'])
                        assert actual['frame'].item()==meta['frame'] and actual['ids'].tolist()==meta['ids'];checks+=2
                        expected=np.fromfile(path.with_suffix('.video.f32.bin'),dtype='<f4').reshape(shape);np.testing.assert_array_equal(actual['masks'].float().numpy(),expected);checks+=1
                        if meta['low_masks']:
                            expected=np.fromfile(path.with_suffix('.low.f32.bin'),dtype='<f4').reshape(actual['low_masks'].shape);np.testing.assert_array_equal(actual['low_masks'].float().numpy(),expected);checks+=1
                        np.testing.assert_array_equal(actual['logits'].flatten().float().numpy(),np.asarray(meta['object_logits'],np.float32));checks+=1
                else:
                    ground=read(target,'ground');assert ground['0/scores'].shape==(200,) and ground['raw/logits'].shape==(1,200,1)
                    geom=read(target,'geometry');assert geom['batch_count'].item()==2 and geom['0/scores'].numel()==geom['1/scores'].numel()==0
                    # Detailed grounding/text comparisons run separately against
                    # the already verified C++ modules via their test dispatcher.
                row=dict(model=model,mode=mode,kind=kind,exact_comparisons=checks,child_path='/nonexistent');rows.append(row);print(json.dumps(row),flush=True)
    a.report.write_text(json.dumps(dict(cases=rows,scope='Pure C11 client linked to the shared library. Image two-stage outputs and video sessions compare prior original-parity fixture outputs; Batch outputs have a separate component-composition comparison; ABI lifetime/callback checks run inside C. Grounding count/raw shape smoke here; separate report covers detailed geometry/text/grounding parity.'),indent=2)+'\n')
if __name__=='__main__':main()
