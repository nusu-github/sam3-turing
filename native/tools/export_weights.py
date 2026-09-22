"""Export lossless module shards for the Python-independent native weight reader."""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import torch
from inventory_weights import group

DTYPES = ["bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64", "complex64", "complex128"]


def write_string(file, value):
    encoded = value.encode("utf-8")
    if not 0 < len(encoded) <= 65536 or b"\0" in encoded:
        raise ValueError("invalid index string")
    file.write(struct.pack("<I", len(encoded)))
    file.write(encoded)


def export(inventory, checkpoints, output):
    if sys.byteorder != "little":
        raise RuntimeError("little-endian export host required")
    if inventory["format"] != "sam3-native-weight-inventory-v1":
        raise ValueError("unsupported inventory")
    if set(checkpoints) != set(inventory["models"]):
        raise ValueError("checkpoint labels must match inventory")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + "-", dir=output.parent))
    handles = {}
    try:
        states = {}
        for label, path in checkpoints.items():
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            states[label] = state.get("model", state)
            if set(states[label]) != set(inventory["models"][label]):
                raise ValueError(f"checkpoint keys changed: {label}")
        records = {}
        for identity, blob in inventory["blobs"].items():
            references = blob["references"]
            locations = {label + "--" + group(key) for label, key in references}
            shard = (next(iter(locations)) if len(locations) == 1 else "shared") + ".s3w"
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", shard):
                raise ValueError("invalid shard name")
            label, key = references[0]
            value = states[label][key].detach().contiguous()
            dtype = str(value.dtype).removeprefix("torch.")
            raw = memoryview(value.reshape(-1).view(torch.uint8).numpy())
            digest = hashlib.sha256(raw).hexdigest()
            if [*value.shape] != blob["shape"] or dtype != blob["dtype"] or digest != blob["sha256"] or raw.nbytes != blob["bytes"]:
                raise ValueError(f"stale inventory: {label}/{key}")
            # Verify every alias, not merely the chosen representative.
            for alias_label, alias_key in references[1:]:
                alias = states[alias_label][alias_key].detach().contiguous()
                alias_raw = memoryview(alias.reshape(-1).view(torch.uint8).numpy())
                if alias.dtype != value.dtype or alias.shape != value.shape or hashlib.sha256(alias_raw).hexdigest() != digest:
                    raise ValueError(f"stale alias: {alias_label}/{alias_key}")
            if shard not in handles:
                handles[shard] = (staging / shard).open("wb")
            file = handles[shard]
            file.write(b"\0" * (-file.tell() % 64))
            offset = file.tell()
            file.write(raw)
            records[identity] = {"shard": shard, "offset": offset, "bytes": raw.nbytes, "crc32": zlib.crc32(raw), "dtype": dtype, "shape": blob["shape"], "sha256": digest}
        for file in handles.values():
            file.close()
        with (staging / "weights.s3i").open("wb") as file:
            file.write(b"SAM3WGT1")
            file.write(struct.pack("<Q", sum(map(len, inventory["models"].values()))))
            for label, tensors in sorted(inventory["models"].items()):
                for key, identity in sorted(tensors.items()):
                    record = records[identity]
                    write_string(file, label + "/" + key)
                    write_string(file, record["shard"])
                    file.write(struct.pack("<QQIII", record["offset"], record["bytes"], record["crc32"], DTYPES.index(record["dtype"]), len(record["shape"])))
                    for dim in record["shape"]:
                        file.write(struct.pack("<Q", dim))
        files = {}
        for path in sorted(staging.iterdir()):
            sha = hashlib.sha256()
            with path.open("rb") as file:
                for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                    sha.update(block)
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha.hexdigest()}
        manifest = {"format": "sam3-native-weights-v1", "models": inventory["models"], "tensors": records, "files": files, "summary": inventory["summary"]}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(staging, output)
        return {"output": str(output), "files": len(files) + 1, "bytes": sum(x["bytes"] for x in files.values()) + (output / "manifest.json").stat().st_size, **inventory["summary"]}
    except BaseException:
        for file in handles.values():
            file.close()
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="label=/path/to/checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = {}
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        if label in checkpoints:
            raise ValueError(f"duplicate checkpoint label: {label}")
        checkpoints[label] = path
    print(json.dumps(export(json.loads(args.inventory.read_text()), checkpoints, args.output), indent=2))


if __name__ == "__main__":
    main()
