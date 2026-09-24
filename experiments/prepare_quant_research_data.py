"""Freeze disjoint COCO calibration/development/holdout samples before experiments.

Only small image previews are downloaded; model weights and parquet shards are not.
The holdout manifest is prepared here but no model is evaluated on it.
"""
import hashlib
import io
import json
from pathlib import Path
import random
import urllib.parse
import urllib.request

from PIL import Image

ROOT = Path('.cache/quant-research-20260924/data')
MANIFEST = Path('experiments/results/native_rtx2060/quant-research-data.json')
DATASET = 'detection-datasets/coco'


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def restore_manifest(manifest):
    """Restore missing local files without changing the frozen image selection."""
    source_rows={}
    for row in manifest['images']:
        path=Path(row['image'])
        if not path.exists():
            if row['split'] not in source_rows:
                metadata=fetch_json('https://huggingface.co/api/datasets/'+manifest['dataset'])
                assert metadata['sha']==manifest['revision'], 'dataset revision changed; do not silently replace images'
                query=urllib.parse.urlencode(dict(dataset=manifest['dataset'],config='default',split=row['split']))
                data=fetch_json('https://datasets-server.huggingface.co/first-rows?'+query)
                source_rows[row['split']]={item['row']['image_id']:item['row'] for item in data['rows']}
            source=source_rows[row['split']][row['image_id']]
            with urllib.request.urlopen(source['image']['src'],timeout=45) as response:
                image=Image.open(io.BytesIO(response.read())).convert('RGB')
            assert image.size==(row['width'],row['height'])
            path.parent.mkdir(parents=True,exist_ok=True)
            image.save(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'], f'image hash mismatch: {path}'
        for case in row['cases']:
            prompt_path=Path(case['prompt_file'])
            if not prompt_path.exists(): prompt_path.write_text(case['prompt'],encoding='utf-8')
            assert prompt_path.read_text(encoding='utf-8')==case['prompt']


def main():
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        restore_manifest(manifest)
        print('Existing manifest and image hashes verified; no selection changed.')
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    metadata = fetch_json('https://huggingface.co/api/datasets/' + DATASET)
    records = []
    seen_ids = set()
    seen_hashes = set()
    for split in ['train', 'val']:
        query = urllib.parse.urlencode(dict(dataset=DATASET, config='default', split=split))
        data = fetch_json('https://datasets-server.huggingface.co/first-rows?' + query)
        names = next(f for f in data['features'] if f['name'] == 'objects')['type']['feature']['category']['names']
        rows = data['rows']
        assert len(rows) >= 48 and not any(r['truncated_cells'] for r in rows)
        random.Random(20260924 + (split == 'val')).shuffle(rows)
        chosen = rows[:32] if split == 'train' else rows[:48]
        for pos, item in enumerate(chosen):
            row = item['row']
            role = 'calibration' if split == 'train' else ('development' if pos < 16 else 'holdout')
            key = f"coco-{split}-{row['image_id']:012d}"
            assert row['image_id'] not in seen_ids
            seen_ids.add(row['image_id'])
            path = ROOT / (key + '.ppm')
            if not path.exists():
                with urllib.request.urlopen(row['image']['src'], timeout=45) as response:
                    image = Image.open(io.BytesIO(response.read())).convert('RGB')
                assert image.size == (row['width'], row['height'])
                image.save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest not in seen_hashes
            seen_hashes.add(digest)
            objects = row['objects']
            categories = objects['category']
            largest = max(range(len(categories)), key=lambda i: objects['area'][i])
            prompts = [(names[categories[largest]], 'largest_annotated_category')]
            small = sorted(range(len(categories)), key=lambda i: objects['area'][i])
            alternatives = [i for i in small if categories[i] != categories[largest]]
            if alternatives:
                prompts.append((names[categories[alternatives[0]]], 'smallest_other_annotated_category'))
            if pos % 4 == 0:
                absent = next(i for i in range(80) if i not in categories)
                prompts.append((names[absent], 'category_not_annotated'))
            cases = []
            for j, (prompt, kind) in enumerate(prompts):
                prompt_path = ROOT / f'{key}-p{j}.txt'
                prompt_path.write_text(prompt, encoding='utf-8')
                cases.append(dict(id=f'{key}-p{j}', prompt=prompt, kind=kind, prompt_file=str(prompt_path)))
            records.append(dict(id=key, image_id=row['image_id'], row_idx=item['row_idx'],
                                split=split, role=role, image=str(path), sha256=digest,
                                width=row['width'], height=row['height'], objects=objects, cases=cases))
            print(role, key, [p[0] for p in prompts], flush=True)
    manifest = dict(dataset=DATASET, revision=metadata['sha'], seed=20260924,
                    method='Seeded shuffle of first 100 viewer rows in each split; train32, val16 dev + val32 holdout.',
                    limits='Small convenience subset, not full COCO accuracy. Missing category annotation is not proof of absence. No segmentation ground-truth masks included.',
                    images=records)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print({role: sum(r['role'] == role for r in records) for role in ['calibration','development','holdout']})


if __name__ == '__main__':
    main()
