# RTX 2060 Max-Q 6 GB: local Windows validation

Measured on 2026-09-22, using the public patch at commit `327e7e5`.
GPU: NVIDIA GeForce RTX 2060 with Max-Q Design, compute capability 7.5,
6 GB VRAM. CPU: AMD Ryzen 9 4900HS. Windows 11, driver 610.88.
Python 3.12.13, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128,
triton-windows 3.6.0.post26. A project-local `.venv` was created with uv.

## Reproduce

Run these PowerShell commands from the repository root:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r experiments/results/local_rtx2060/requirements-windows.txt
.venv/Scripts/hf.exe auth login
.venv/Scripts/python.exe experiments/local_turing_bench.py --output experiments/results/local_rtx2060_rerun
```

Existing Hugging Face authentication is reused. Access to `facebook/sam3` is
required unless `--checkpoint C:/path/to/sam3.pt` is supplied. The default
checkpoint is pinned to revision `3c879f39826c281e95690f02c7821c4de09afae7`.
Use `--variants patched_eager patched_cpu_text` to restrict the configurations.
Reruns should use a new output directory so old comparison tensors are not reused.

## Method

- Every configuration runs in a fresh subprocess with a bounded timeout.
- Resolution 1008, confidence threshold 0.5, batch size 1. No image downscaling
  beyond the standard processor input transform.
- Model initialization uses one CPU thread; inference uses four.
- Timing includes `set_image` + `set_text_prompt`, preprocessing and GPU transfer,
  but excludes reading the image file and building the model.
- `truck.jpg` / `truck`, first-call timing separately, two warmups, median of five
  measured calls. Patched configurations reuse the text cache in timed calls.
- PyTorch peaks cover first inference, warmups and timed calls. Model construction
  peaks are separate: 3.330 GiB allocated, before any runtime patch is applied.
- NVML samples whole-device memory at a requested 5 ms interval, including other
  applications. Sampling can miss instantaneous peaks.
- Quality checks: truck / paper bag / child / wheel / empty elephant, over the
  same three images used by earlier experiments. Box-matched mask comparisons
  are numerical agreement checks, not ground-truth accuracy measurements.

All configurations use an outer FP16 autocast. **Stock** leaves upstream code
unchanged, including `sam3/perflib/fused.py` explicitly converting the vision MLP
to BF16. **FP16 reference** uses the existing reproduction harness's FP16 adapter
(`F.linear` + GELU instead of that BF16 fused operation), with autocast cache on.
It does not apply the Turing memory optimizations. Consequently, the stock versus
patch comparison includes precision changes, memory optimizations and text caching.

## Results

| Configuration | Status | Median ms/image | PyTorch allocated GiB | Whole-device NVML GiB | Changed mask pixels vs stock |
|---|---|---:|---:|---:|---:|
| Stock, upstream BF16 MLP retained | OK | 3752.08 | 4.992 | 5.867 | — |
| FP16 reference, no Turing patch | OOM | — | — | — | — |
| Turing FP16, eager | OK | 1584.63 | 1.997 | 2.678 | 102 |
| Turing FP16, compiled | OK | 1446.95 | 1.802 | 2.542 | 104 |
| Turing FP16, eager + CPU text, trim padding | OK | 1589.21 | 1.336 | 1.974 | 104 |
| Turing INT8, attention projections + fused MLP + CPU text | OK | 1364.67 | 0.926 | 1.935 | 941 |

All successful configurations have finite outputs and detection counts
**1 / 4 / 6 / 4 / 0**, matching stock. Pixel counts use 18,038,400 compared mask
pixels. FP16 eager changes 0.000565%; INT8 changes 0.005217%.

FP16 eager reduces allocated inference memory by about 60% and latency by about
58% versus the mixed-precision stock run. INT8 + CPU text reduces allocated
inference memory by about 81% and latency by about 64%.

The FP16 reference failed in the pixel decoder while allocating another 82 MiB;
PyTorch reported 5.12 GiB allocated and zero free device memory. Its JSON records
the exception and failure stage. Stock completed on this run: it is incorrect to
claim that every unpatched path necessarily OOMs on 6 GB. Windows WDDM can use
shared system memory, so successful allocation alone does not establish physical
VRAM residency; shared-memory residency was not separately measured here.

First inference took 2.03 s for eager FP16, 2.05 s with CPU text, and 4.94 s for
INT8. Compiled FP16 took 138.46 s on its first inference, then 1.447 s per image
in the timed repetitions (about 9% faster than eager FP16). These values exclude
model construction and depend on cache state. All six configurations completed
within the assigned time limits: five succeeded, and the FP16 reference produced
a recorded CUDA OOM. No configuration timed out.

For short sessions, `compile=False` avoids the large first-call compilation cost.
CPU text offload substantially reduces inference memory with nearly unchanged
cached-prompt timing. INT8 + CPU text was the fastest tested configuration here,
at the cost of larger numerical differences. Compiled INT8 and novel-prompt
latency were not measured in this local run.

JSON files contain raw timings, memory readings and per-case comparisons.
Local `references/*.pt` files preserve CPU outputs for comparisons; model/tensor
files and logs are ignored by Git. `requirements-windows.txt` records installed
versions, including the editable checkout.
