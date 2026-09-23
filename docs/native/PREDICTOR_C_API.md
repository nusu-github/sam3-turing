# Owning predictor through C ABI 1

`sam3_predictor` exposes the owning image/video host through the existing C-only
header. It accepts runtime counted UTF-8 text, positive/negative normalized xywh
boxes, caller-encoded visual tokens, points and masks. It manages object IDs,
source policies, memory sessions, edit routing, output scheduling and reset.
The earlier `sam3_video` remains the low-level instance tracker. Existing ABI 1
structures are unchanged; the predictor adds new handles, structures and symbols.

```c
#include <sam3/c_api.h>

/* context was created with weights, vocabulary, model, device and precision;
 * read_frame fills an RGB view whose buffer survives until its next call. */
sam3_predictor_options options;
sam3_predictor_options_init(&options, SAM3_MODEL_31);
options.tracking.frames = frame_count;
options.tracking.height = height;
options.tracking.width = width;
options.tracking.provider = read_frame;
options.tracking.provider_user = decoder;
sam3_predictor *predictor = NULL;
sam3_status status = sam3_predictor_create(context, &options, &predictor);
/* Check every status before using a handle/result. */
sam3_utf8_view text = {user_text, user_text_bytes};
sam3_semantic_prompt prompt;
sam3_semantic_prompt_init(&prompt);
prompt.text = &text;
sam3_result *preview = NULL;
status = sam3_predictor_add_prompt(predictor, prompt_frame, &prompt, &preview);
/* Consume/release preview. emit borrows each output and can retain it. */
sam3_result_release(preview);
status = sam3_predictor_propagate(predictor, prompt_frame, -1, 0, 0, emit, user);
sam3_predictor_release(predictor);
```

Use `image_only=1` with exactly one frame for explicit image mode. A one-frame
video keeps it false. `sam3_predictor_options_init` selects the model's owning
source defaults, which differ from low-level tracker defaults. Options expose
detector/NMS thresholds, association, hotstart, reconditioning, confirmation,
mask cleanup, temporal selection, offload and SAM3.1 history paging. Buffer batch
size and optional track padding are scheduling/allocation choices, not object
caps. Image and video use the same modular store and cores.

`add_prompt` replaces semantic state. `add_points` and `add_mask` edit an ID or
create a new instance. `fetch`, `remove_object`, `reset` and `info` operate on the
same owner. Forward propagation includes the start frame; reverse propagation
starts at `start-1`, matching the source high-level host. This differs from the
low-level tracker API. `force_tracker` is the existing SAM3-specific route.
Image empty-point restoration follows the behavior and source-defect notes in
[IMAGE_PREDICTOR.md](IMAGE_PREDICTOR.md).

## Results and callbacks

Outputs contain `frame`, `ids`, `probabilities`, normalized `boxes_xywh`, boolean
`masks[N,H,W]` and optional normalized `centers[N,2]`. `cached_ids` and individual
`cached_masks/ID` fields preserve pre-overlap binary inputs, including empty masks.
Available frame statistics are `stats/NAME`. `sam3_result_get` returns host views
owned by the result; retention survives reset and owner/context destruction.
Results from the high-level host do not invent low-resolution neural logits;
the existing low-level API exposes those separately.

All input tensor buffers are copied before use. Frame providers can reuse padded
RGB buffers. Provider failure returns `SAM3_CALLBACK_ERROR`. Calls on an owner
are exclusive; provider/output callback reentry returns `SAM3_BUSY`.
`sam3_predictor_cancel` is allowed concurrently or inside a callback. It stops
further buffered emissions as well as later frame processing, without interrupting
a running kernel. Output callbacks return 0 to continue, 1 to stop, or another
value to report failure. Propagation exceptions discard pending emissions and
record the SAM3.1 cancellation action before translating the error. Completed
frame updates are not rolled back. Reset or a new semantic prompt can start fresh.

## Core sharing

C API children of one context share the same immutable vision, detector and
tracker modules. A new C++ `VideoPredictorModules` constructor provides the same
option for direct C++ users. Context trimming/release cannot invalidate a live
owner. Features, object state and caches remain per owner. Temporary text encoder
loads are unchanged; this is not a claim of sharing all transient allocations.

The CUDA allocator audit `native/tests/predictor_c_sharing.py` calls only public
C functions. Before frame inference, a SAM3 owner uses 2,045,219,328 active CUDA
bytes and a SAM3.1 owner uses 2,096,884,224. A second owner on the same context
adds **zero active CUDA bytes** in both cases. Releasing the first owner/context
keeps the surviving owner's allocation; releasing the final owner returns to
the zero-byte baseline. These numbers exclude frame features/history and text
encoding, and refer to this development build's Blackwell allocation accounting.

## Validation and limits

`sam3_predictor_c_probe` is compiled as C11 without Torch/C++ headers and runs with
`PATH=/nonexistent`. It exercises provider errors, mismatched/invalid options,
callback reentry, all three output stop/error forms, state isolation between two
owners, parent-before-child destruction, retained results, image mask restoration,
point/mask-only initialization, forward and reverse propagation.

The comparator `native/tests/predictor_c_parity.py` checks retained C++ owner
outputs: all 10 SAM3.1 image checkpoints and all 13 video checkpoints for each
model match exactly in IDs, probabilities, boxes and masks. Video emission timing
also matches; retained result fields remain unchanged after owner destruction.
The image's retained original mask matches too. These are C-boundary comparisons;
the earlier original-reference configuration/adapters remain applicable. The
120-pixel reverse-edit difference was later traced to a stale original pointer;
see [REVERSE_EDIT_POINTER.md](REVERSE_EDIT_POINTER.md) for the explicit repair. Reverse execution in this C client
is an invariant check, not a newly established original-output parity result.

The existing low-level C tracker also retains 86 exact comparisons (46 SAM3,
40 SAM3.1) after the common options conversion refactor. CTest passes 28
CUDA-enabled and 16 custom-CUDA-disabled tests. See
[predictor-c-validation.json](predictor-c-validation.json) for the recorded scope.

No query/point/object cap or prompt/weight variant is introduced. This is still a
development library: broader media coverage, multi-GPU transport, CPU
stability and broader quality/performance remain. The native dependency graph
contains no `libpython` or `libtorch_python`; the standalone development SDK uses official LibTorch and bundled runtime
libraries; see [SDK.md](SDK.md). Windows/Turing execution remains the user's verification task.
No GitHub Actions are used.

Local media sources can now be retained by `sam3_predictor_create_from_media`;
see [MEDIA_IO.md](MEDIA_IO.md) for decode semantics and validation limits.
