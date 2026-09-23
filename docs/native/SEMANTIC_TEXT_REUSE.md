# Reuse completed text encoding during semantic replacement

An owning predictor previously reloaded the text model and re-encoded its token
batch on every `add_prompt`, including replacements with exactly the same text.
It now retains the existing completed text encoding across that internal reset
when the optional UTF-8 text is identical. Geometry, visual prompts, observations,
image features, tracking state and displayed output still reset/update normally.

The model, device and inference mode are fixed for the owner, so an identical
optional string produces the same complete token batch, including the model's
visual/geometric auxiliary strings. All four text-input tensors must be defined
before reuse. This avoids retaining a partially completed encoding after an
exception. Geometry and visual inputs are attached from the replacement prompt.

The retained encoding also records Torch inference settings. Thread counts,
TF32/FP32 policies, reduction/accumulation policy, deterministic/backend choices
and attention enablement/priority must still match. Changing them invalidates
reuse. As with the existing C API, process-wide settings must not be changed
concurrently with inference. Untracked application floating-point environment
changes require an explicit predictor reset.

An absent string, an empty string and the source's special `visual` string remain
distinct keys. Any changed string is re-encoded. Public `reset()` still clears the
encoding. No extra cache entry, persistent text-weight copy, fixed prompt, input
limit, public ABI change, dependency or model distribution variant is introduced.
Arbitrary new prompts use the existing full computation path.

## Native regression probe

The installed `sam3_semantic_text_probe` creates a temporary hard-linked view of
the weight store. After the first prompt, it hides only the temporary text-shard
names. A same-text replacement must succeed in the new SDK without rereading
weights. In the previous SDK it must fail; restoring the temporary links and
retrying supplies the exact baseline output. Source shard bytes/names are never
modified, and all temporary links are removed on exit.

The probe covers six optional-text values (ordinary, empty, `visual`, absent,
UTF-8 and a longer phrase). For each it changes box labels and visual tokens,
flips source RGB, and checks a fresh provider read and vision encoding. Changed
text and explicit reset must attempt to reload the temporarily hidden weights;
restoration must recover. Invalid geometry must preserve the preceding state.
All output tensor bytes, types, layouts, metadata and masks are compared.
The real-model probe also requires misses after CPU-thread, TF32 and attention
policy changes, then restores settings and checks recovery. A separate CPU test
checks ten policy changes/restorations, including attention priority order.

```sh
sam3_semantic_text_probe WEIGHTS sam3.1 cuda fp16 FRAME.ppm BPE.gz OUTPUT 1 5
```

Use a local filesystem supporting hard links, with output and weights on the
same filesystem. This requirement belongs to the fault-injection diagnostic;
inference does not require hard links. The probe's text values are test fixtures,
not a runtime prompt restriction. The production change uses existing C++/ATen
code and no new platform-specific path; SDK.md retains the Windows build basis.

## Measurement scope

The final validation records both models in FP16, BF16 reference and FP32.
FP16 uses three alternating baseline/current process pairs per model; other
precisions use one pair each for correctness. Each process warms up, then times
five complete same-text replacements with alternating box labels/frame indices.
These calls include preprocessing, image encoding, detection, tracking and text
handling. They exclude construction, output dumping and C-wrapper result packing.
No competing builds or GPU work run during the measured matrix.

| Model | Before (median) | After (median) | Reduction |
| --- | ---: | ---: | ---: |
| SAM3 FP16 | 804.563 ms | 133.461 ms | 83.41% |
| SAM3.1 FP16 | 877.822 ms | 227.182 ms | 74.12% |

The ten final process pairs retain all 2,440 compared output/metadata files.
Every current process passes six same-text replacements with hidden shard links
and 15 required misses (changed text, public reset and settings changes).
CPU24/CUDA42 CTests pass. Initial comparisons before the complete-encoding and
inference-policy guards are superseded by this final matrix.
The final 13-owner/six-C-API regressions retain all 1,940 files. Recovered SDK
C lifecycle and the installed semantic probe retain 153/243 files respectively;
all 35 initialized workspace libraries resolve within the recovered SDK and
no Python/Triton runtime is loaded. All 466 CPU / 473 CUDA preceding public
strong symbols remain available, without additions or removals.

These are warm-filesystem Blackwell measurements for repeated text. They do not
describe arbitrary changed-text requests, complete video throughput, cold storage
or Turing performance. Both variants perform the same number of fresh image
reads/encodes; this change does not reuse potentially stale caller-provided images.

`semantic-text-validation.json` records final samples, lifecycle comparisons,
ABI symbols and recovered SDK checks. The private SDK recipe is
`semantic-text-sdk-overlay/overlays.json`, after compute-storage; evidence is
under `native-foundation/semantic-text-linux`. Matching payloads and all weights/
dependencies are reused. Original-source FP16 residuals remain documented in
VIDEO_COLLECTIVE_REFERENCE.md. Windows/Turing execution remains user-owned;
no GitHub Actions are used and no broader completion claim is made.
