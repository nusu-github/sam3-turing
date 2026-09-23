"""Deterministic evaluation slice; no selection based on model output.

Uses the official COCO2017 annotations archive and its public S3 image objects.
The small slice is an audit, not a representative COCO benchmark.
"""

import argparse
import hashlib
import json
import random
import subprocess
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def fetch(url, path):
    # Interrupted downloads must not become apparently complete cached inputs.
    temporary = path.with_name(path.name + ".part")
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "2", url, "-o", str(temporary)],
        check=True,
        capture_output=True,
    )
    temporary.replace(path)


p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
p.add_argument("--count", type=int, default=8)
p.add_argument("--seed", type=int, default=20260923)
a = p.parse_args()
root = a.directory.resolve()
root.mkdir(parents=True, exist_ok=True)
archive = root / "annotations_trainval2017.zip"
if not archive.exists():
    fetch(
        "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
        archive,
    )
with zipfile.ZipFile(archive) as z:
    data = json.loads(z.read("annotations/instances_val2017.json"))
images = sorted(data["images"], key=lambda x: x["id"])
if not 0 < a.count <= len(images):
    p.error(f"--count must be between 1 and {len(images)}")
selected = sorted(random.Random(a.seed).sample(images, a.count), key=lambda x: x["id"])
ids = {x["id"] for x in selected}
annotations = [x for x in data["annotations"] if x["image_id"] in ids]
cats = sorted({x["category_id"] for x in annotations})
categories = [x for x in data["categories"] if x["id"] in cats]
subset = dict(data)
subset.update(images=selected, annotations=annotations, categories=categories)
(root / "instances-slice.json").write_text(json.dumps(subset) + "\n")
(root / "images").mkdir(exist_ok=True)
(root / "prompts").mkdir(exist_ok=True)


def download(item):
    path = root / "images" / item["file_name"]
    url = "https://s3.amazonaws.com/images.cocodataset.org/val2017/" + item["file_name"]
    if not path.exists():
        fetch(url, path)
    return dict(
        image_id=item["id"],
        file=item["file_name"],
        url=url,
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        license=item["license"],
        flickr_url=item.get("flickr_url"),
    )


with ThreadPoolExecutor(max_workers=4) as pool:
    downloads = list(pool.map(download, selected))
for c in categories:
    (root / "prompts" / f"{c['id']}.txt").write_text(c["name"])
with (root / "manifest.tsv").open("w") as f:
    for im in selected:
        for c in categories:
            f.write(
                f"{im['id']}\t{c['id']}\timages/{im['file_name']}\tprompts/{c['id']}.txt\n"
            )
report = dict(
    seed=a.seed,
    selection="Python random.Random(seed).sample of image-ID-sorted COCO val2017, sorted after selection; fixed before inference",
    images=len(selected),
    annotations=len(annotations),
    categories=categories,
    prompts=len(selected) * len(categories),
    downloads=downloads,
    annotations_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    scope="Evaluate each category present in the slice on EVERY selected image, including negatives. All original annotations (including crowd) and per-image license/Flickr metadata retained. Not representative/full COCO AP; inference retains all200 queries.",
)
(root / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
