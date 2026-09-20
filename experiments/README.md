# Image patch experiments

Use the container's existing Python/PyTorch installation. The sweep does not
create an environment, replace PyTorch, provision machines, or control RunPod.

```bash
python experiments/image_sweep.py --sweep experiments/round1.json
python experiments/image_sweep.py --sweep experiments/round2.json
python experiments/image_sweep.py --sweep experiments/round3.json --cases 5 --reps 7
python experiments/image_sweep.py --sweep experiments/round4.json --cases 5 --reps 9
python experiments/image_sweep.py --sweep experiments/round5.json --cases 5 --reps 9
python experiments/packed_masks.py
python experiments/summarize.py
```

`--checkpoint /path/to/sam3.pt` uses a local checkpoint; otherwise the script
fetches the revision used by the supplied experiments from Hugging Face.
`--output` selects a result directory. Each candidate runs in its own process.
The default 180-second candidate timeout includes model construction/compilation.

The five round files preserve the candidates, including losers. JSON results
contain whole-image latency, allocated/reserved VRAM, sampled whole-device NVML
usage, detection counts, box-matched mask IoU, pixel differences, score error and
box error. These are practical A/B checks against the stock BF16 and FP16 paths,
not a ground-truth accuracy benchmark. Text caching timings include a cache hit;
`combined_uncached` measures encoding the prompt on every image.

There were 55 image attempts over 50 named candidates (including repeats and
three initial failures later corrected). The three 4K output configurations
are recorded separately in `results/packed_masks.json`.

`archive_reference/` contains the seven Python helpers used from the supplied
`SAM3_FP16_reproduction.zip`. That archive also bundled helpers from the earlier
GPU/20USD experiments. We kept only inference/measurement helpers; the RunPod
controllers, setup scripts and credentials are not part of this checkout.
The runtime patch in `sam3/turing.py` is self-contained and does not import these
experiment helpers.

The first round had two CPU initialization crashes in the container's NVIDIA
PyTorch build (before any candidate patch was applied). Later rounds initialize
with one CPU thread, then restore four threads for inference. The initial
in-place output prototype had a Python decorator namespace error; the third
round fixes it. Earlier successful baseline measurements are preserved with
the `_before_round3` suffix.
