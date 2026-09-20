"""Inference research patches. No edits to upstream; batch-one image scope."""
import sys,contextlib,inspect,ast,textwrap
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sam3_gpu_lab'))
# gpu_variants imports a different module named variants. Load it before this
# file through importlib under claims_variants (see run_full.py).
from gpu_variants import choose_variant,group_stream,gpu_stream_head

class LazyHigh:
    def __init__(self,x,conv):
        self.x=x;self.conv=conv;self.shape=(x.shape[0],256,x.shape[2]*4,x.shape[3]*4)
        self.device=x.device
        self.dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
    def __getitem__(self,key):
        assert key[0] is Ellipsis and isinstance(key[1],slice) and key[2]==slice(None)
        lo,hi=key[1].start,key[1].stop
        a=max(0,(lo-1)//4);b=min(self.x.shape[-2],(hi+1+3)//4)
        y=self.conv(self.x[...,a:b,:])
        return y[...,lo-4*a:hi-4*a,:]
    def new_empty(self,shape):return torch.empty(shape,device=self.device,dtype=self.dtype)
    def new_zeros(self,shape):return torch.zeros(shape,device=self.device,dtype=self.dtype)

@contextlib.contextmanager
def lazy_neck(model):
    neck=model.backbone.vision_backbone
    assert model.num_feature_levels==1 and neck.sam2_convs is None and len(neck.convs)==4 and model.backbone.scalp==1
    def forward(tensor_list):
        x=neck.trunk(tensor_list)[-1]
        mid=neck.convs[1](x);low=neck.convs[2](x)
        return [LazyHigh(x,neck.convs[0]),mid,low,None],[None,None,neck.position_encoding(low).to(low.dtype),None],None,None
    with patch.object(neck,'forward',forward):yield

@contextlib.contextmanager
def chunk_mlps(model,tokens):
    with contextlib.ExitStack() as st:
        for block in model.backbone.vision_backbone.trunk.blocks:
            mlp=block.mlp;original=mlp.forward
            def forward(x,original=original):
                shape=x.shape;rows=x.reshape(-1,shape[-1]);output=None
                for lo in range(0,rows.shape[0],tokens):
                    y=original(rows[lo:lo+tokens])
                    if output is None:output=y.new_empty((rows.shape[0],y.shape[-1]))
                    output[lo:lo+tokens].copy_(y)
                return output.reshape(*shape[:-1],output.shape[-1])
            st.enter_context(patch.object(mlp,'forward',forward))
        yield

@contextlib.contextmanager
def lean_neck(model):
    neck=model.backbone.vision_backbone
    assert model.num_feature_levels==1 and neck.sam2_convs is None and model.backbone.scalp==1
    def forward(tensor_list):
        x=neck.trunk(tensor_list)[-1]
        feats=[conv(x) for conv in neck.convs[:3]]
        return feats+[None],[None,None,neck.position_encoding(feats[-1]).to(feats[-1].dtype),None],None,None
    with patch.object(neck,'forward',forward):yield

@contextlib.contextmanager
def early_keep(model,processor):
    original=model._run_segmentation_heads
    def selected(out,backbone_out,img_ids,vis_feat_sizes,encoder_hidden_states,prompt,prompt_mask,hs):
        assert hs.shape[1]==1 and not model.training
        probs=(out['pred_logits'].sigmoid()*out['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
        keep=probs[0]>processor.confidence_threshold
        original(out,backbone_out,img_ids,vis_feat_sizes,encoder_hidden_states,prompt,prompt_mask,hs[:,:,keep])
        out['_selected_masks']=True
    def grounding(state):
        from sam3.model import box_ops
        from sam3.model.data_misc import interpolate
        outputs=model.forward_grounding(backbone_out=state['backbone_out'],find_input=processor.find_stage,geometric_prompt=state['geometric_prompt'],find_target=None)
        probs=(outputs['pred_logits'].sigmoid()*outputs['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
        keep=probs>processor.confidence_threshold
        bbox=outputs['pred_boxes'][keep];masks=outputs['pred_masks'][0]
        boxes=box_ops.box_cxcywh_to_xyxy(bbox)
        h,w=state['original_height'],state['original_width']
        boxes=boxes*torch.tensor([w,h,w,h],device=processor.device)[None,:]
        masks=interpolate(masks.unsqueeze(1),(h,w),mode='bilinear',align_corners=False).sigmoid()
        state.update(masks_logits=masks,masks=masks>.5,boxes=boxes,scores=probs[keep]);return state
    with patch.object(model,'_run_segmentation_heads',selected),patch.object(processor,'_forward_grounding',grounding):yield

@contextlib.contextmanager
def apply_variant(model,processor,mode,tile=16,groups=2,tokens=512):
    if model.training or torch.is_grad_enabled():raise ValueError('Inference-only image variants')
    head=model.segmentation_head
    with contextlib.ExitStack() as st:
        if mode!='stock':st.enter_context(choose_variant(head,'no_clone'))
        if 'early' in mode:st.enter_context(early_keep(model,processor))
        if 'chunk' in mode:st.enter_context(chunk_mlps(model,tokens))
        if 'lazy' in mode:st.enter_context(lazy_neck(model))
        if 'lean' in mode:st.enter_context(lean_neck(model))
        if 'prune' in mode:
            cache=model.backbone.vision_backbone.position_encoding.cache
            for key in list(cache):
                if key!=(72,72):del cache[key]
        if 'group' in mode:
            st.enter_context(patch.object(head,'forward',lambda **kw:group_stream(head,kw,tile_rows=tile,groups_per_pass=groups)))
        elif 'stream' in mode or 'lazy' in mode:
            st.enter_context(patch.object(head,'forward',lambda **kw:gpu_stream_head(head,kw,tile_rows=tile,store_conv=False,stats_dtype=torch.float32)))
        yield
