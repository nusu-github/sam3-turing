"""Original object-addition host methods and actual-weight local session collections."""
import argparse,json,hashlib,tempfile
from pathlib import Path
from types import SimpleNamespace,MethodType
import torch
from sam3.model.sam3_video_base import Sam3VideoBase
from sam3.model.sam3_multiplex_base import Sam3MultiplexBase
from tracking_frame_parity import reference as sam3_reference
from tracking_session_parity import run_reference as configure_sam3
from multiplex_session_parity import session_reference,layout_staging
from memory_attention_parity import backend_context
from temporal_memory_parity import source_transfer_device

def policy_checks(device):
    checks=0
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        for n in [1,3,17]:
            raw=torch.randn(n,7,9,device=device,dtype=dtype).transpose(1,2)
            expected=torch.nn.functional.interpolate(raw[:,None],(1152,1152),mode='bilinear',align_corners=False)[:,0]>0
            torch.testing.assert_close(torch.ops.sam3_native.prepare_video_object_masks(raw),expected,rtol=0,atol=0);checks+=1
    # Drive the original high-level addition code with a recording neural tracker.
    # This checks selection, stable ties, grouping and exact mask preprocessing.
    class Recorder:
        input_mask_size=1152
        def init_state(self,**kwargs):return dict(obj_ids=[],multiplex_state=SimpleNamespace(available_slots=16),backbone_out=None)
        def add_new_masks(self,**kw):self.target=kw['inference_state'];self.masks=kw['masks'];self.target['obj_ids']+=kw['obj_ids']
        def add_new_mask(self,**kw):self.target=kw['inference_state'];self.masks.append(kw['mask']);self.target['obj_ids'].append(kw['obj_id'])
        def propagate_in_video_preflight(self,state,run_mem_encoder):assert run_mem_encoder;self.final=state
    generator=torch.Generator().manual_seed(71)
    for policy in range(3):
        for i in range(80):
            slots=torch.randint(0,65,(i%7,),generator=generator).tolist();n=1+i%33
            states=[dict(obj_ids=[],multiplex_state=SimpleNamespace(available_slots=s),backbone_out=None) for s in slots]
            original=list(states);tracker=Recorder();tracker.is_multiplex_dynamic=policy==0;tracker.per_obj_inference=policy==1
            raw=torch.randn(n,2,3,device=device);owner=SimpleNamespace(tracker=tracker,is_multiplex=True)
            states=Sam3MultiplexBase._tracker_add_new_objects(owner,0,3,list(range(n)),raw,states,37,53,{})
            expected=next((j for j,s in enumerate(original) if tracker.target is s),-1)
            actual=torch.ops.sam3_native.video_object_destination(slots,n,policy);assert actual==expected,(slots,n,policy,actual,expected)
            torch.testing.assert_close(tracker.masks,torch.ops.sam3_native.prepare_video_object_masks(raw),rtol=0,atol=0);checks+=2
    return checks

def configure_host(weights,arrays,mux,device,mode):
    if not mux:
        host=sam3_reference(weights,device)
        configure_sam3(host,[[a[j] for a in arrays] for j in range(4)],[],[],[4,37,53,0,0,3,4],[False,False,False,False,True,False,False,False],mode,device)
        return host
    host=session_reference(weights,device);host.non_overlap_masks_for_output=False;host.use_memory_selection=False;host.non_overlap_masks_for_mem_enc=False
    encoder=host.transformer.encoder;original=encoder.forward
    def attention(**kwargs):
        if mode=='fp32':kwargs['memory']=kwargs['memory'].float()
        return original(**kwargs)
    encoder.forward=attention
    def features(self,state,index,batch):
        out={}
        for name,offset in [('interactive',0),('sam2_backbone_out',4)]:
            image,position,h0,h1=arrays[index%len(arrays)][offset:offset+4]
            out[name]=dict(vision_feats=[x.flatten(2).permute(2,0,1) for x in [h0,h1,image]],vision_masks=[None]*3,vision_pos_embeds=[None,None,position.flatten(2).permute(2,0,1)],feat_sizes=[(288,288),(144,144),(72,72)])
        return None,out
    host._get_image_feature=MethodType(features,host);return host

def source_run(host,ops,payloads,mux,device):
    owner=SimpleNamespace(tracker=host,is_multiplex=mux);base=Sam3MultiplexBase if mux else Sam3VideoBase
    if mux:host.is_multiplex_dynamic=False;host.per_obj_inference=False # explicitly test the new-state grouping policy
    states=[];out={}
    def save(k,v):
        if v is not None:out[k]=v.detach().cpu().clone()
    for step,(op,payload) in enumerate(zip(ops,payloads)):
        root=f'{step}/'
        if op[0]==0:
            base._tracker_add_new_objects(owner,op[1],4,op[2:],payload,states,37,53,{})
            save(root+'destination',torch.tensor(len(states)-1))
        elif op[0]==1:
            # Bind the SAM3 helper called by the high-level batch-removal method.
            owner._tracker_remove_object=MethodType(Sam3VideoBase._tracker_remove_object,owner)
            base._tracker_remove_objects(owner,states,op[1:])
        elif op[0]==2:
            for i,state in enumerate(states):
                kwargs=dict(tqdm_disable=True,run_mem_encoder=True)
                for j,value in enumerate(host.propagate_in_video(state,op[1],op[2],bool(op[3]),**kwargs)):
                    frame,ids,low,masks=value[:4];key=root+f'state{i}/out{j}/';save(key+'frame',torch.tensor(frame));save(key+'ids',torch.tensor(ids));save(key+'low',low);save(key+'masks',masks)
        elif op[0]==3:states=[]
        save(root+'states',torch.tensor(len(states)))
        for i,state in enumerate(states):
            key=root+f'state{i}/';save(key+'ids',torch.tensor(state['obj_ids']))
            for src,dst in [('cond_frame_outputs','cond'),('non_cond_frame_outputs','tracked')]:
                for index,frame in state['output_dict'][src].items():
                    for name,field in [('low','pred_masks'),('memory','maskmem_features'),('pointer','obj_ptr'),('logits','object_score_logits')]:save(key+f'{dst}/{index}/{name}',frame.get(field))
                    if frame.get('maskmem_pos_enc') is not None:save(key+f'{dst}/{index}/position',frame['maskmem_pos_enc'][-1])
    return out

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('library',type=Path);p.add_argument('store',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('--model',choices=['sam3','sam3.1'],required=True);p.add_argument('--device',default='cuda');p.add_argument('--modes',nargs='+',default=['fp16','fp32','bf16_reference']);p.add_argument('--report',type=Path,required=True);p.add_argument('--policy-only',action='store_true');a=p.parse_args()
    torch.ops.load_library(str(a.library.resolve()));torch.set_num_threads(4);torch.manual_seed(619);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    policies=policy_checks(a.device)
    if a.policy_only:a.report.write_text(json.dumps(dict(policy_checks=policies,device=a.device)));return
    weights=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=True);weights=weights.get('model',weights);mux=a.model=='sam3.1';rows=[]
    for mode in a.modes:
        dtype={'fp16':torch.float16,'fp32':torch.float32,'bf16_reference':torch.bfloat16}[mode]
        arrays=[[torch.randn(shape,device=a.device,dtype=dtype) for neck in range(2 if mux else 1) for shape in [(1,256,72,72),(1,256,72,72),(1,32,288,288),(1,64,144,144)]] for _ in range(3)]
        host=configure_host(weights,arrays,mux,a.device,mode)
        # Fresh groups share a frame core/cache but have independent temporal state.
        ops=[[0,0,101,202],[2,0,1,0],[0,1,303],[2,2,0,0],[1,303,999],[2,2,2,1],[1,101,202],[3]]
        empty=torch.empty(0);raw=torch.randn(2,37,53,device=a.device);payloads=[raw,empty,raw[:1].neg(),empty,empty,empty,empty,empty]
        with backend_context(a.device=='cuda' and mode=='fp32',False),source_transfer_device(a.device),layout_staging(mode),torch.autocast(a.device,enabled=mode!='fp32',dtype=dtype if mode!='fp32' else torch.bfloat16):
            expected=source_run(host,ops,payloads,mux,a.device)
        if a.device=='cuda':torch.cuda.synchronize()
        actual=torch.ops.sam3_native.video_objects(str(a.store),mux,arrays,ops,payloads,[4,37,53],False,mode,'',2)
        for k,v in expected.items():
            try:torch.testing.assert_close(actual[k],v,rtol=0,atol=0)
            except AssertionError as e:raise AssertionError(f'{a.model} {mode} {k}: {e}') from e
        row=dict(mode=mode,source_exact_tensors=len(expected))
        # Exercise dynamic best-fit reuse with historical bucket reconstruction,
        # then multi-ID removal and both directions. This is native dense-history
        # policy, not a claim of equivalence to the source's packed-history bugs.
        if mux:
            dynamic_ops=[[0,0,101,202],[2,0,1,0],[0,1,303,404],[2,2,0,0],[1,202,404,999,404],[2,2,2,1],[1,101,303]]
            dynamic_payloads=[raw,empty,raw.flip(0),empty,empty,empty,empty]
            resident=torch.ops.sam3_native.video_objects(str(a.store),True,arrays,dynamic_ops,dynamic_payloads,[4,37,53],False,mode)
            assert resident['2/destination'].item()==0 and resident['2/states'].item()==1
            assert resident['4/state0/ids'].tolist()==[101,303] and resident['6/states'].item()==0
            comparisons=0
            sequential_ops=dynamic_ops[:4]+[[1,202],[1,404,999,404]]
            sequential_payloads=dynamic_payloads[:4]+[empty,empty]
            sequential=torch.ops.sam3_native.video_objects(str(a.store),True,arrays,sequential_ops,sequential_payloads,[4,37,53],False,mode)
            for k,v in resident.items():
                if k.startswith('4/'):
                    torch.testing.assert_close(sequential['5/'+k[2:]],v,rtol=0,atol=0);comparisons+=1
            for paged in [False,True]:
                with tempfile.TemporaryDirectory(prefix='sam3-objects-') as directory:
                    actual=torch.ops.sam3_native.video_objects(str(a.store),True,arrays,dynamic_ops,dynamic_payloads,[4,37,53],True,mode,directory if paged else '')
                    assert actual.keys()==resident.keys()
                    for k,v in resident.items():torch.testing.assert_close(actual[k],v,rtol=0,atol=0);comparisons+=1
                    assert not list(Path(directory).iterdir())
            row['storage_exact_tensors']=comparisons
        rows.append(row);print(a.model,row,flush=True)
        del host
    a.report.write_text(json.dumps(dict(model=a.model,device=a.device,policy_checks=policies,cases=rows,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('sam3/model/sam3_video_base.py'),Path('sam3/model/sam3_multiplex_base.py')]},scope='Actual-weight original high-level add/remove calls, new-state grouping and forward/reverse propagation using synthetic full-grid projected features. Source comparisons remove whole states; native best-fit insertion, multi-ID removal (including equivalence with sequential removal) and dense-history rebuilding checked separately across resident/offloaded/paged storage. No coherent full-video accuracy or multi-GPU communication claim.'),indent=2)+'\n')
if __name__=='__main__':main()
