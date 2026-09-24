"""All six decoder layers, 200 queries, box anchors and presence outputs."""
import argparse,json
from pathlib import Path
import torch
from sam3.model_builder import _create_transformer_decoder
from sam3.model.position_encoding import PositionEmbeddingSine

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--modes',nargs='+',default=['fp32','fp16','bf16_reference']);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(177)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results=[]
    for spec in a.checkpoint:
        model,path=spec.split('=',1);ref=_create_transformer_decoder().eval()
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);state=state.get('model',state)
        prefix='detector.transformer.decoder.'
        ref.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True);ref.to(a.device)
        # Preserve the constructor's standard grid. CPU tests move this cache
        # explicitly because it is not a registered buffer in the source model.
        ref.compilable_cord_cache=tuple(t.to(a.device) for t in ref.compilable_cord_cache)
        for mode in a.modes:
            dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16_reference':torch.bfloat16}[mode]
            for case,b,h,w,length in [('full',1,72,72,33),('batch_padding',3,15,21,17),('geometry_only',2,13,19,1)]:
                image=torch.randn(b,256,h,w,device=a.device).to(dtype);pos=PositionEmbeddingSine(256,normalize=True)(image).flatten(2).permute(2,0,1)
                memory=image.flatten(2).permute(2,0,1);prompt=torch.randn(length,b,256,device=a.device)
                mask=torch.zeros(b,length,dtype=torch.bool,device=a.device);padding=None
                ratios=torch.ones(b,1,2,device=a.device);shapes=torch.tensor([[h,w]],device=a.device);starts=torch.zeros(1,device=a.device,dtype=torch.long)
                if case=='batch_padding':
                    pm=torch.zeros(b,h,w,dtype=torch.bool,device=a.device)
                    for i in range(b):
                        mask[i,length-i-2:]=True;pm[i,h-i-1:,:]=True;pm[i,:,w-i-2:]=True
                    padding=pm.flatten(1).t()
                    ratios=torch.stack([(~pm[:,0,:]).sum(1).float()/w,(~pm[:,:,0]).sum(1).float()/h],-1).unsqueeze(1)
                with torch.autocast(a.device,dtype=dtype if mode!='fp32' else torch.bfloat16,enabled=mode!='fp32'):
                    expected=ref(tgt=ref.query_embed.weight.unsqueeze(1).repeat(1,b,1),memory=memory,memory_key_padding_mask=padding,
                        pos=pos,reference_boxes=None,level_start_index=starts,spatial_shapes=shapes,valid_ratios=ratios,
                        memory_text=prompt,text_attention_mask=mask,apply_dac=False)
                with torch.autocast(a.device,dtype=torch.bfloat16):
                    actual=torch.ops.sam3_native.detector_decode(str(a.store),model,memory,pos,prompt,mask,padding,shapes,ratios,mode)
                    assert torch.is_autocast_enabled(a.device) and torch.get_autocast_dtype(a.device)==torch.bfloat16
                errors={}
                for name,x,y in zip(['hidden','references','presence_logits','presence_features'],actual,expected):
                    tolerance={'fp32':5e-5,'fp16':.003,'bf16_reference':.02}[mode]
                    torch.testing.assert_close(x,y,rtol=tolerance,atol=tolerance)
                    errors[name]=(x-y).abs().max().item()
                item=dict(model=model,mode=mode,case=case,hidden_shape=list(actual[0].shape),max_errors=errors)
                results.append(item);print(json.dumps(item),flush=True)
    a.report.write_text(json.dumps(dict(torch=torch.__version__,device=a.device,cases=results),indent=2)+'\n')
if __name__=='__main__':main()
