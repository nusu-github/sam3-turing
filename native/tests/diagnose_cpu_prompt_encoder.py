"""Original-only reproduction of the development CPU runtime crash.

Run repeatedly in fresh processes with PYTHONFAULTHANDLER=1. This intentionally
does not load libsam3_native or any checkpoint. It is a diagnostic, not a passing
regression test; see docs/native/CPU_RUNTIME_ISSUE.md.
"""
import torch
from sam3.sam.prompt_encoder import PromptEncoder

torch.set_num_threads(4)
with torch.inference_mode():
    model = PromptEncoder(256, (72, 72), (1008, 1008), 16).eval()
    for iteration in range(8):
        print(iteration, model.get_dense_pe().shape, flush=True)
