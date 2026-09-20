import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
import torch
from sam3.turing_int8 import DynamicInt8Linear
from fused_int8_gemm import quantize, matmul

torch.set_num_threads(4)
torch.manual_seed(3)
with torch.inference_mode():
    for m, k, n in [(128, 64, 128), (5184, 1024, 4736), (5184, 4736, 1024)]:
        linear = DynamicInt8Linear(
            torch.nn.Linear(k, n, device="cuda", dtype=torch.float16)
        )
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        q, s = quantize(x)
        mm = torch._int_mm(q, linear.weight_int8.T)
        ref = (mm.float() * s[:, None]) * linear.weight_scale[None, :] + linear.bias
        for gelu in (False, True):
            expected = torch.nn.functional.gelu(ref.half()) if gelu else ref.half()
            y = matmul(q, s, linear, [64, 128, 64, 4], gelu)
            diff = (y - expected).abs()
            print(
                json.dumps(
                    dict(
                        shape=[m, k, n],
                        gelu=gelu,
                        finite=bool(y.isfinite().all()),
                        max_abs=diff.max().item(),
                        mean_abs=diff.float().mean().item(),
                        actual_first=y[0, :8].tolist(),
                        expected_first=expected[0, :8].tolist(),
                    )
                ),
                flush=True,
            )
