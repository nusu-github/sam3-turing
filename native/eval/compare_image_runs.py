"""Require byte-identical native image outputs for every manifest pair.

This compares all emitted candidates/masks, plus full floating probability maps
when the native audit was run with --save-probabilities. It does not score AP or
allow tolerances, threshold filtering or missing cases.
"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.digest()


def compare(manifest, before, after):
    pairs = [
        tuple(map(int, line.split("\t")[:2]))
        for line in manifest.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("empty manifest or duplicate case")
    fingerprints = []
    combined = hashlib.sha256()
    probabilities = 0
    for image, category in pairs:
        prefix = f"{image}-{category}"
        data = json.loads((before / (prefix + ".json")).read_text())
        if (data["image_id"], data["category_id"]) != (image, category):
            raise ValueError(f"incorrect case identity: {prefix}")
        if sorted(d["query"] for d in data["detections"]) != list(range(200)):
            raise ValueError(f"incomplete candidate set: {prefix}")
        h, w = data["height"], data["width"]
        if h <= 0 or w <= 0 or data["mask_bytes"] != (h * w + 7) // 8:
            raise ValueError(f"invalid mask dimensions: {prefix}")
        if (before / (prefix + ".masks.bin")).stat().st_size != 200 * data[
            "mask_bytes"
        ]:
            raise ValueError(f"incomplete mask tensor: {prefix}")
        suffixes = [".json", ".masks.bin"]
        if "probability_bytes" in data:
            probabilities += 1
            suffixes.append(".probabilities.bin")
            itemsize = {"Half": 2, "BFloat16": 2, "Float": 4, "Double": 8}[
                data["probability_dtype"]
            ]
            if data["probability_bytes"] != 200 * h * w * itemsize:
                raise ValueError(f"incorrect probability dimensions: {prefix}")
            if (before / (prefix + suffixes[-1])).stat().st_size != data[
                "probability_bytes"
            ]:
                raise ValueError(f"incomplete probability tensor: {prefix}")
        for suffix in suffixes:
            name = prefix + suffix
            left, right = before / name, after / name
            expected, actual = digest(left), digest(right)
            if left.stat().st_size != right.stat().st_size or expected != actual:
                raise ValueError(f"output differs: {name}")
            combined.update(name.encode())
            combined.update(expected)
            fingerprints.append(
                dict(file=name, bytes=left.stat().st_size, sha256=expected.hex())
            )
    return dict(
        cases=len(pairs),
        candidates=200 * len(pairs),
        full_probability_cases=probabilities,
        all_bytes_equal=True,
        content_sha256=combined.hexdigest(),
        files=fingerprints,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.manifest, args.before, args.after)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}))


if __name__ == "__main__":
    main()
