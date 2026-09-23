"""Score all native candidates with the official COCO evaluator.

Python is an offline evaluation dependency only. Inference runs in the C++ tool.
No candidate threshold is applied here; COCO's usual maxDets=100 is a metric
convention and does not change the 200 candidates emitted by inference.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def read_case(path):
    data = json.loads(path.read_text())
    detections = data["detections"]
    h, w = data["height"], data["width"]
    if h <= 0 or w <= 0 or len(detections) != 200:
        raise ValueError(f"invalid dimensions/candidate count: {path}")
    if sorted(d["query"] for d in detections) != list(range(200)):
        raise ValueError(f"missing/duplicate queries: {path}")
    size = (h * w + 7) // 8
    packed = np.fromfile(path.with_suffix(".masks.bin"), dtype=np.uint8)
    if data["mask_bytes"] != size or packed.size != 200 * size:
        raise ValueError(f"incorrect mask payload size: {path}")
    packed = packed.reshape(200, size)
    masks = np.unpackbits(packed, axis=1, bitorder="little")[:, : h * w]
    for detection in detections:
        values = np.asarray([detection["score"], *detection["box"]])
        if not np.isfinite(values).all() or not 0 <= values[0] <= 1:
            raise ValueError(f"invalid score/box: {path}")
    return data, masks.reshape(200, h, w)


def distribution(values):
    return dict(
        min=float(np.min(values)),
        median=float(np.median(values)),
        p90=float(np.percentile(values, 90)),
        max=float(np.max(values)),
    )


def evaluate(annotations, manifest, directory):
    expected = [
        tuple(map(int, line.split("\t")[:2]))
        for line in manifest.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate image/category pair in manifest")
    gt = COCO(str(annotations))
    boxes, segments = [], []
    content_hash = hashlib.sha256()
    for image, category in expected:
        path = directory / f"{image}-{category}.json"
        data, masks = read_case(path)
        if (data["image_id"], data["category_id"]) != (image, category):
            raise ValueError(f"incorrect identity: {path}")
        im = gt.imgs[image]
        if masks.shape[1:] != (im["height"], im["width"]):
            raise ValueError(f"annotation/image dimension mismatch: {path}")
        for source in (path, path.with_suffix(".masks.bin")):
            content_hash.update(source.name.encode())
            content_hash.update(hashlib.sha256(source.read_bytes()).digest())
        for detection, mask in zip(data["detections"], masks):
            x1, y1, x2, y2 = detection["box"]
            if x2 < x1 or y2 < y1:
                raise ValueError(f"inverted box: {path}")
            common = dict(
                image_id=image, category_id=category, score=detection["score"]
            )
            boxes.append(dict(common, bbox=[x1, y1, x2 - x1, y2 - y1]))
            # Omit bbox from segmentation input: loadRes must derive mask area.
            segments.append(
                dict(common, segmentation=mask_utils.encode(np.asfortranarray(mask)))
            )
    names = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    quality = {}
    for kind, predictions in (("bbox", boxes), ("segm", segments)):
        evaluator = COCOeval(gt, gt.loadRes(predictions), kind)
        evaluator.params.imgIds = sorted({image for image, _ in expected})
        evaluator.params.catIds = sorted({cat for _, cat in expected})
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        quality[kind] = {
            key: float(value) if value >= 0 else None
            for key, value in zip(names, evaluator.stats)
        }
    rows = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text().splitlines()
    ]
    if [(r["image_id"], r["category_id"]) for r in rows] != expected:
        raise ValueError("incomplete/misordered metrics")
    if not all(r["finite"] and r["queries"] == 200 for r in rows):
        raise ValueError("nonfinite or incomplete inference")
    summary = json.loads((directory / "summary.json").read_text())
    if summary["cases"] != len(expected) or summary["threshold"] != -1:
        raise ValueError("incorrect native run summary")
    return dict(
        summary=summary,
        quality=quality,
        candidates=len(boxes),
        output_sha256=content_hash.hexdigest(),
        all_finite=True,
        performance={
            key: distribution([r[key] for r in rows if r[key] > 0])
            for key in (
                "image_seconds",
                "detector_postprocess_seconds",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
            if any(r[key] > 0 for r in rows)
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {
        path.name: evaluate(
            args.dataset / "instances-slice.json", args.dataset / "manifest.tsv", path
        )
        for path in args.run
    }
    if len(runs) != len(args.run):
        raise ValueError("run directory names must be unique")
    deltas = {}
    for model in sorted({r["summary"]["model"] for r in runs.values()}):
        modes = {
            r["summary"]["mode"]: r
            for r in runs.values()
            if r["summary"]["model"] == model
        }
        if "fp16" in modes and "bf16_reference" in modes:
            deltas[model] = {
                kind: {
                    key: modes["fp16"]["quality"][kind][key] - value
                    for key, value in modes["bf16_reference"]["quality"][kind].items()
                    if value is not None
                    and modes["fp16"]["quality"][kind][key] is not None
                }
                for kind in ("bbox", "segm")
            }
    report = dict(
        evaluator_environment=dict(
            python=platform.python_version(),
            numpy=importlib.metadata.version("numpy"),
            pycocotools=importlib.metadata.version("pycocotools"),
            cocoeval_sha256=hashlib.sha256(
                Path(inspect.getfile(COCOeval)).read_bytes()
            ).hexdigest(),
        ),
        selection=json.loads((args.dataset / "selection.json").read_text()),
        runs=runs,
        fp16_minus_bf16=deltas,
        metric_max_detections=[1, 10, 100],
        scope="Small predetermined COCO slice, not full COCO or representative quality. "
        "All 200 native candidates retained; standard metric maxDets=100. "
        "Native FP16 versus native BF16, not an unmodified source parity claim. "
        "Local Blackwell hardware; Windows/Turing execution remains user-owned. "
        "Allocator peaks include live model/cache and validation intermediates; "
        "exclude driver/context and allocations outside the PyTorch allocator. "
        "Timing excludes validation, packing and file output; text is cached. "
        "Two detector warmups; image times include cold first image and decode.",
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
