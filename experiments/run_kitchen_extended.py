"""Fixed extra prompts/frames; compare FP16, prior INT8 and INT8 attention."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image

ROOT = Path('.cache/native-perf/kitchen-extended-ba018a0')
CASES = [
    ('dance0-person', 'assets/videos/0001/0.jpg', 'person'),
    ('dance50-person', 'assets/videos/0001/50.jpg', 'person'),
    ('dance99-person', 'assets/videos/0001/99.jpg', 'person'),
    ('dance0-shoe', 'assets/videos/0001/0.jpg', 'shoe'),
    ('dance99-shoe', 'assets/videos/0001/99.jpg', 'shoe'),
    ('dance50-shirt', 'assets/videos/0001/50.jpg', 'shirt'),
    ('groceries-car', 'assets/images/groceries.jpg', 'car'),
    ('groceries-bread', 'assets/images/groceries.jpg', 'bread'),
    ('court-person', 'assets/images/test_image.jpg', 'person'),
    ('court-blue-vest', 'assets/images/test_image.jpg', 'blue vest'),
    ('truck-window', 'assets/images/truck.jpg', 'window'),
    ('truck-tire', 'assets/images/truck.jpg', 'tire'),
]

def environment(mode):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('SAM3_EXPERIMENT_') or key in (
            'SAM3_PROFILE_NVTX', 'SAM3_PROFILE_CAPTURE', 'SAM3_BENCH_CUDNN', 'SAM3_FUSED_NORM_CAST'
        ):
            env.pop(key)
    env['TORCH_BLAS_PREFER_CUBLASLT'] = '0'
    if mode == 'attention_only':
        env['SAM3_EXPERIMENT_ATTENTION'] = 'kitchen_all'
    elif mode != 'fp16':
        env.update(SAM3_EXPERIMENT_MLP='int8_boundary', SAM3_EXPERIMENT_PROJECTION='qkv',
                   SAM3_EXPERIMENT_QKV_ROPE='fused', SAM3_EXPERIMENT_FC2_NORM='fused',
                   SAM3_EXPERIMENT_ATTENTION='exact' if mode == 'int8' else mode)
    return env

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modes', nargs='*')
    parser.add_argument('--cases', nargs='+', choices=[c[0] for c in CASES])
    args = parser.parse_args()
    args.modes = args.modes or ['fp16','int8','kitchen_all']
    if any(m not in ['fp16','int8','kitchen','kitchen_rot','kitchen_all','kitchen_rot_all','attention_only'] for m in args.modes):
        parser.error('unknown experiment mode')
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'cases.json'
    content = json.dumps(CASES, indent=2) + '\n'
    if manifest.exists():
        assert manifest.read_text() == content, 'case list changed'
    else:
        manifest.write_text(content)
    modes = args.modes
    for case, source, prompt in CASES:
        if args.cases and case not in args.cases:
            continue
        image, text = ROOT / (case + '.ppm'), ROOT / (case + '.txt')
        if not image.exists():
            with Image.open(source) as im:
                im.convert('RGB').save(image)
            text.write_text(prompt, encoding='utf-8')
        for mode in modes:
            out = ROOT / f'{case}-{mode}'
            if (out / 'complete.json').exists():
                continue
            if out.exists():
                raise RuntimeError(f'incomplete run, inspect before retrying: {out}')
            cmd = [sys.executable, 'experiments/monitor_native_bench.py', str(out),
                   'build/native-windows-cu130/sam3_image_latency.exe', '.cache/native-weights-sam3',
                   str(image), 'sam3/assets/bpe_simple_vocab_16e6.txt.gz', str(text), '0', '1', str(out)]
            result = subprocess.run(cmd, env=environment(mode), capture_output=True, text=True)
            (out / 'driver.log').write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f'failed: {out}; see driver.log')
            (out / 'complete.json').write_text(json.dumps({'case':case, 'mode':mode, 'source':source, 'prompt':prompt}))
            metrics = json.loads((out / 'metrics.json').read_text())
            print(case, mode, 'count', metrics['count'], flush=True)

if __name__ == '__main__':
    main()
