"""Keep MLP weights in INT8, using FP16 activations and tensor-core products."""

import torch
import triton
import triton.language as tl

from sam3.turing_int8 import DynamicInt8Linear


@triton.jit
def _matmul(
    x,
    w,
    scales,
    bias,
    y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    if GROUP_M:
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BM)
        num_n = tl.cdiv(N, BN)
        group = pid // (GROUP_M * num_n)
        first_m = group * GROUP_M
        group_m = tl.minimum(num_m - first_m, GROUP_M)
        within = pid % (GROUP_M * num_n)
        pid_m = first_m + within % group_m
        pid_n = within // group_m
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    mi = pid_m * BM + tl.arange(0, BM)
    ni = pid_n * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + ki
        a = tl.load(
            x + mi[:, None] * K + ks[None, :], (mi[:, None] < M) & (ks[None, :] < K), 0
        ).to(tl.float16)
        b = tl.load(
            w + ni[None, :] * K + ks[:, None], (ni[None, :] < N) & (ks[:, None] < K), 0
        ).to(tl.float16)
        acc = tl.dot(a, b, acc)
    out = acc * tl.load(scales + ni, ni < N, 0)[None, :]
    out += tl.load(bias + ni, ni < N, 0)[None, :]
    tl.store(
        y + mi[:, None] * N + ni[None, :], out, (mi[:, None] < M) & (ni[None, :] < N)
    )


class WeightOnlyInt8Linear(DynamicInt8Linear):
    def __init__(self, linear, tile, group=0):
        super().__init__(linear)
        self.tile = tuple(tile)
        self.group = group

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features).contiguous()
        rows = x.shape[0]
        y = torch.empty((rows, self.out_features), device=x.device, dtype=torch.float16)
        if rows:
            bm, bn, bk, warps = self.tile
            num_m = triton.cdiv(rows, bm)
            num_n = triton.cdiv(self.out_features, bn)
            grid = (num_m * num_n,) if self.group else (num_m, num_n)
            _matmul[grid](
                x,
                self.weight_int8,
                self.weight_scale,
                self.bias,
                y,
                rows,
                self.out_features,
                self.in_features,
                bm,
                bn,
                bk,
                self.group,
                num_warps=warps,
                num_stages=3,
            )
        return y.reshape(*shape[:-1], self.out_features)


def apply_weight_only_int8(model, tile, group=0):
    for block in model.backbone.vision_backbone.trunk.blocks:
        block.mlp.fc1 = WeightOnlyInt8Linear(block.mlp.fc1, tile, group)
        block.mlp.fc2 = WeightOnlyInt8Linear(block.mlp.fc2, tile, group)
