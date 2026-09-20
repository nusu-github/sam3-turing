"""Choose INT8 weight scales from weight error alone, before inference."""

from contextlib import contextmanager
import time
from unittest.mock import patch

import torch

from sam3.turing_int8 import DynamicInt8Linear


@contextmanager
def weight_scale_search(refine=0):
    original = DynamicInt8Linear.__init__
    stats = {
        "layers": 0,
        "rows": 0,
        "original_squared_error": 0.0,
        "selected_squared_error": 0.0,
        "scale_ratio_sum": 0.0,
    }

    def initialize(self, linear):
        original(self, linear)
        with torch.no_grad():
            w = linear.weight.detach().float()
            initial = self.weight_scale.clone()
            best_scale = initial.clone()
            best_error = (
                (w - self.weight_int8.float() * initial[:, None]).square().sum(1)
            )
            original_error = best_error.sum().item()
            # Per-row clipping candidates; no images, activation fitting or training.
            for ratio in (0.975, 0.95, 0.925, 0.9, 0.875, 0.85, 0.825, 0.8):
                scale = initial * ratio
                q = (w / scale[:, None]).round().clamp(-127, 127)
                for _ in range(refine):
                    scale = ((w * q).sum(1) / q.square().sum(1).clamp_min(1)).clamp_min(
                        1e-10
                    )
                    q = (w / scale[:, None]).round().clamp(-127, 127)
                error = (w - q * scale[:, None]).square().sum(1)
                better = error < best_error
                best_scale = torch.where(better, scale, best_scale)
                best_error = torch.minimum(best_error, error)
            self.weight_scale = best_scale
            self.weight_int8 = (
                (w / best_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
            )
            stats["layers"] += 1
            stats["rows"] += w.shape[0]
            stats["original_squared_error"] += original_error
            stats["selected_squared_error"] += best_error.sum().item()
            stats["scale_ratio_sum"] += (best_scale / initial).sum().item()

    start = time.perf_counter()
    with patch.object(DynamicInt8Linear, "__init__", initialize):
        yield stats
    stats["setup_seconds"] = time.perf_counter() - start
    stats["relative_squared_error"] = stats["selected_squared_error"] / max(
        stats["original_squared_error"], 1e-30
    )
    stats["mean_scale_ratio"] = stats.pop("scale_ratio_sum") / max(stats["rows"], 1)
