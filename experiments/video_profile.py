"""Small scoped timers for finding the next SAM 3.1 bottleneck."""

import time
from collections import defaultdict
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch


@contextmanager
def component_profile(predictor):
    detector = predictor.model.detector
    tracker = predictor.model.tracker.model
    records = defaultdict(list)
    report = {}
    targets = [
        (detector.backbone.vision_backbone.trunk, "forward", "vision_trunk"),
        (detector.backbone.vision_backbone, "forward", "vision_with_necks"),
        (detector.backbone, "forward_text", "text"),
        (detector.transformer.encoder, "forward", "detector_encoder"),
        (detector.transformer.decoder, "forward", "detector_decoder"),
        (detector.segmentation_head, "forward", "detector_masks"),
        (tracker, "_prepare_memory_conditioned_features", "tracker_conditioning"),
        (tracker.transformer.encoder, "forward", "tracker_memory_attention"),
        (tracker.sam_mask_decoder, "forward", "tracker_masks"),
        (tracker.maskmem_backbone, "forward", "tracker_memory_encoder"),
    ]
    with ExitStack() as stack:
        for owner, method, label in targets:
            original = getattr(owner, method)

            def timed(*args, original=original, label=label, **kwargs):
                first = torch.cuda.Event(enable_timing=True)
                last = torch.cuda.Event(enable_timing=True)
                first.record()
                start = time.perf_counter()
                output = original(*args, **kwargs)
                wall = time.perf_counter() - start
                last.record()
                records[label].append((first, last, wall))
                return output

            stack.enter_context(patch.object(owner, method, timed))
        yield report
        torch.cuda.synchronize()
        for label, values in records.items():
            report[label] = {
                "calls": len(values),
                "cuda_span_ms": sum(a.elapsed_time(b) for a, b, _ in values),
                "cpu_wall_ms": sum(wall for _, _, wall in values) * 1000,
            }
        report["note"] = (
            "Intervals can overlap/nest and must not be summed. CUDA spans include idle host waits. A separate pass after the measured runs; timer overhead is not a benchmark speedup."
        )
