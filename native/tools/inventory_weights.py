"""Inventory exact tensor sharing before choosing native module shards.

Development-time utility. Does not convert precision or change any parameters.
No model data is emitted; the inventory contains keys, shapes, dtypes and hashes.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch


def group(key):
    for prefix in [
        "detector.backbone.vision_backbone",
        "detector.backbone.language_backbone",
        "detector.transformer",
        "detector.geometry_encoder",
        "detector.segmentation_head",
        "detector.dot_prod_scoring",
        "tracker",
    ]:
        if key == prefix or key.startswith(prefix + "."):
            return prefix
    return ".".join(key.split(".")[:2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="label=/path/to/checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blobs = {}
    models = {}
    module_bytes = defaultdict(int)
    logical_bytes = 0
    for spec in args.checkpoint:
        label, filename = spec.split("=", 1)
        if label in models:
            raise ValueError(f"duplicate label: {label}")
        weights = torch.load(filename, map_location="cpu", weights_only=True, mmap=True)
        weights = weights.get("model", weights)
        tensors = {}
        for key, value in sorted(weights.items()):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"non-tensor checkpoint entry: {label}:{key}")
            value = value.detach().contiguous()
            raw = value.reshape(-1).view(torch.uint8).numpy()
            sha = hashlib.sha256(memoryview(raw)).hexdigest()
            dtype = str(value.dtype).removeprefix("torch.")
            shape = list(value.shape)
            # Keep metadata in identity: bytes alone cannot determine semantics.
            identity = f"{dtype}:{','.join(map(str, shape))}:{sha}"
            entry = blobs.setdefault(identity, {"sha256": sha, "dtype": dtype, "shape": shape, "bytes": raw.nbytes, "references": []})
            entry["references"].append([label, key])
            tensors[key] = identity
            module_bytes[f"{label}/{group(key)}"] += raw.nbytes
            logical_bytes += raw.nbytes
        models[label] = tensors
    unique_bytes = sum(entry["bytes"] for entry in blobs.values())
    cross_model_bytes = sum(entry["bytes"] for entry in blobs.values() if len({ref[0] for ref in entry["references"]}) > 1)
    result = {
        "format": "sam3-native-weight-inventory-v1",
        "summary": {
            "tensor_references": sum(map(len, models.values())),
            "unique_tensors": len(blobs),
            "logical_tensor_bytes": logical_bytes,
            "unique_tensor_bytes": unique_bytes,
            "bytes_saved_by_exact_deduplication": logical_bytes - unique_bytes,
            "bytes_shared_across_models": cross_model_bytes,
            "module_logical_bytes": dict(sorted(module_bytes.items())),
        },
        "models": models,
        "blobs": blobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
