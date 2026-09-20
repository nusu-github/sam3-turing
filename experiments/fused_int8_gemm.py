"""INT8 GEMM with FP16/GELU epilogues instead of materialized INT32 outputs."""

from unittest.mock import patch

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sam3.turing_int8 import _quantize


@triton.jit
def _int8_epilogue(
    x,
    w,
    row_scales,
    weight_scales,
    bias,
    y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    GELU: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = pid // (GROUP * num_n)
    first_m = group * GROUP
    group_m = tl.minimum(num_m - first_m, GROUP)
    within = pid % (GROUP * num_n)
    mi = (first_m + within % group_m) * BM + tl.arange(0, BM)
    ni = (within // group_m) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + ki
        a = tl.load(
            x + mi[:, None] * K + ks[None, :], (mi[:, None] < M) & (ks[None, :] < K), 0
        )
        b = tl.load(
            w + ni[None, :] * K + ks[:, None], (ni[None, :] < N) & (ks[:, None] < K), 0
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    out = acc.to(tl.float32) * tl.load(row_scales + mi, mi < M, 0)[:, None]
    out = (
        out * tl.load(weight_scales + ni, ni < N, 0)[None, :]
        + tl.load(bias + ni, ni < N, 0)[None, :]
    )
    if GELU:
        out = out.to(tl.float16).to(tl.float32)
        out = 0.5 * out * (1 + libdevice.erf(out * 0.7071067811865476))
    tl.store(
        y + mi[:, None] * N + ni[None, :], out, (mi[:, None] < M) & (ni[None, :] < N)
    )


def quantize(x):
    rows, cols = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(rows, device=x.device, dtype=torch.float32)
    _quantize[(rows,)](x, q, scale, cols, triton.next_power_of_2(cols))
    return q, scale


def matmul(q, scale, linear, tile, gelu=False):
    rows, cols = q.shape
    out = torch.empty((rows, linear.out_features), device=q.device, dtype=torch.float16)
    bm, bn, bk, warps = tile
    _int8_epilogue[(triton.cdiv(rows, bm) * triton.cdiv(linear.out_features, bn),)](
        q,
        linear.weight_int8,
        scale,
        linear.weight_scale,
        linear.bias,
        out,
        rows,
        linear.out_features,
        cols,
        bm,
        bn,
        bk,
        8,
        gelu,
        num_warps=warps,
        num_stages=3,
    )
    return out


def apply_fused_int8_gemm(model, stack, tile, fc2=True):
    for block in model.backbone.vision_backbone.trunk.blocks:
        mlp = block.mlp

        def forward(x, mlp=mlp):
            shape = x.shape
            flat = x.reshape(-1, mlp.fc1.in_features).contiguous()
            if not flat.shape[0]:
                return x.new_empty(
                    (*shape[:-1], mlp.fc2.out_features), dtype=torch.float16
                )
            q, scale = quantize(flat)
            hidden = matmul(q, scale, mlp.fc1, tile, gelu=True)
            if fc2:
                q, scale = quantize(hidden)
                out = matmul(q, scale, mlp.fc2, tile)
            else:
                out = mlp.fc2(hidden)
            return out.reshape(*shape[:-1], mlp.fc2.out_features)

        stack.enter_context(patch.object(mlp, "forward", forward))
