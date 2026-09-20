"""Explicit FP16/FP32 execution; no BF16 benchmark paths."""
import contextlib
from unittest.mock import patch
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

@contextlib.contextmanager
def execution(precision='fp16',backend='auto',cache=False):
    import sam3.model.vitdet as vitdet
    original=F.scaled_dot_product_attention
    def activation(act,linear,x):
        y=F.linear(x,linear.weight,linear.bias)
        if act in (F.gelu,torch.nn.GELU):return F.gelu(y)
        if act in (F.relu,torch.nn.ReLU):return F.relu(y)
        raise ValueError('Unexpected activation')
    def attention(*args,**kwargs):
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):return original(*args,**kwargs)
    with contextlib.ExitStack() as st:
        st.enter_context(patch.object(vitdet,'addmm_act',activation))
        if backend=='efficient':st.enter_context(patch.object(F,'scaled_dot_product_attention',attention))
        st.enter_context(torch.autocast('cuda',dtype=torch.float16,enabled=precision=='fp16',cache_enabled=cache))
        yield

def cast_linear_weights(model,preserve_precision=False):
    """Selective policy preserves FP32 decoder FFNs and embedding lookups."""
    protected=set()
    if preserve_precision:
        for m in model.modules():
            if hasattr(m,'forward_ffn') and m.__class__.__module__=='sam3.model.decoder':
                protected.update([id(m.linear1),id(m.linear2)])
    modules=(torch.nn.Linear,torch.nn.Conv2d,torch.nn.ConvTranspose2d)
    if not preserve_precision:modules=modules+(torch.nn.Embedding,)
    for m in model.modules():
        if isinstance(m,modules) and id(m) not in protected:m.to(dtype=torch.float16)

@contextlib.contextmanager
def mlp_arena(model):
    """One FP16 hidden buffer reused across sequential ViT blocks, freed at exit."""
    trunk=model.backbone.vision_backbone.trunk
    arena={}
    original_trunk=trunk.forward
    def trunk_forward(*args,**kwargs):
        try:return original_trunk(*args,**kwargs)
        finally:arena.clear()
    with contextlib.ExitStack() as st:
        st.enter_context(patch.object(trunk,'forward',trunk_forward))
        for block in trunk.blocks:
            m=block.mlp
            if not isinstance(m.norm,torch.nn.Identity) or not isinstance(m.act,torch.nn.GELU):raise ValueError('Unsupported ViT MLP')
            def forward(x,m=m):
                shape=x.shape;flat=x.reshape(-1,shape[-1])
                with torch.autocast('cuda',enabled=False):
                    flat=flat.half();w=m.fc1.weight.half();b=m.fc1.bias.half()
                    needed=(flat.shape[0],w.shape[0])
                    if 'hidden' not in arena:arena['hidden']=torch.empty(needed,device=x.device,dtype=torch.float16)
                    y=arena['hidden']
                    if y.shape!=needed:raise ValueError('Arena shape changed')
                    torch.addmm(b,flat,w.T,out=y)
                    torch.ops.aten.gelu.out(y,approximate='none',out=y)
                    out=torch.addmm(m.fc2.bias.half(),y,m.fc2.weight.half().T)
                    return out.reshape(*shape[:-1],out.shape[-1])
            st.enter_context(patch.object(m,'forward',forward))
        yield

def differences(ref,actual):
    result={}
    for k,v in actual.items():
        if k not in ref:continue
        r=ref[k]
        if r.shape!=v.shape:
            result[k]=dict(shape_changed=True,reference_shape=list(r.shape),actual_shape=list(v.shape));continue
        d=(r.float()-v.float()).abs()
        result[k]=dict(equal=bool(torch.equal(r,v)),raw_bytes_equal=bool(r.dtype==v.dtype and torch.equal(r.contiguous().view(torch.uint8),v.contiguous().view(torch.uint8))),
            changed=int((r!=v).sum()),elements=v.numel(),max_abs=float(d.max()) if d.numel() else 0,
            rms=float(d.square().mean().sqrt()) if d.numel() else 0)
    return result
