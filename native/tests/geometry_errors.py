"""Validate geometry input rejection and restoration of caller autocast."""
import argparse
import torch

p=argparse.ArgumentParser()
p.add_argument('library')
p.add_argument('store')
a=p.parse_args()
torch.ops.load_library(a.library)
for device in ['cpu','cuda']:
    image=torch.zeros(1,256,3,5,device=device)
    points=torch.zeros(2,1,2,device=device)
    labels=torch.ones(2,1,device=device,dtype=torch.long)
    mask=torch.zeros(1,2,device=device,dtype=torch.bool)
    boxes=torch.empty(0,1,4,device=device)
    bl=torch.empty(0,1,device=device,dtype=torch.long)
    bm=torch.empty(1,0,device=device,dtype=torch.bool)
    cases=[(points,labels,mask,'bad'),(points,labels,torch.tensor([[True,False]],device=device),'fp32'),
           (points,labels*2,mask,'fp32'),(points+float('nan'),labels,mask,'fp32')]
    for pt,pl,pm,mode in cases:
        with torch.autocast(device,dtype=torch.bfloat16):
            try:
                torch.ops.sam3_native.geometry_encode(a.store,'sam3',image,image,pt,pl,pm,boxes,bl,bm,mode)
            except RuntimeError:
                pass
            else:
                raise AssertionError('invalid geometry input accepted')
            assert torch.is_autocast_enabled(device) and torch.get_autocast_dtype(device)==torch.bfloat16
    print(f'{device}: 4 invalid inputs rejected; autocast restored')
