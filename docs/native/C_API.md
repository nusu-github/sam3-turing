# Native C API (ABI 1)

`native/include/sam3/c_api.h` exposes the implemented image and low-level video
inference modules through an ordinary C ABI. A client needs that header and the
generated `sam3_native_export.h`, plus the native shared library at link/runtime.
It does not need Torch headers, C++ source, Python or Triton. The shared library
still needs its LibTorch/CUDA/ICU/zlib dependencies; a relocatable distribution
is separate work. High-level text-guided video association is not implemented
by this API yet. The goal of full SAM3/SAM3.1 feature parity remains open.

## Ownership and errors

Check `sam3_abi_version() == SAM3_ABI_VERSION` before using typed structures.
ABI 1 layouts are fixed; incompatible structure changes require a new ABI.
Use each `_init` function to obtain defaults, then set input pointers/options.
`struct_size` is checked when structures are consumed. Keep the header and
library ABI in agreement; do not pass undersized storage to an init function.

Every status-returning function contains C++ exceptions. `SAM3_INVALID_ARGUMENT`
covers wrapper validation; model-side Torch validation can report
`SAM3_RUNTIME_ERROR`. Host and Torch CUDA out-of-memory exceptions map to
`SAM3_OUT_OF_MEMORY`. `sam3_last_error()` is thread-local and must be copied
before another API call on that thread. Null releases are safe. Foreign invalid
pointers or destroyed handles are caller errors, not recoverable exceptions.

Context, image, video and result handles are opaque. Image/video children retain
their context, so releasing the caller's context handle does not invalidate
live children. Context modules load lazily and are shared across sessions of the
same type. The visual backbone is shared across image/video sessions. Image
sessions copy small module maps while sharing the actual tensor weights with
a cached prototype. `sam3_context_trim` releases cached modules that have no
active users. Image features remain usable after the visual weights are released;
a later image load can reload those weights from the same modular store.

Results own their tensors. `sam3_result_get` lazily creates a contiguous host
view and returns its dtype, shape, byte length and data pointer. The view stays
valid while the result has an owned reference. Retaining a result avoids copying
its tensors. Output callbacks borrow their result; call `sam3_result_retain`
inside the callback to keep it. Releasing an image/video does not invalidate
retained results. Named fields are immutable; concurrent result readers are
synchronized.

Input tensor views are contiguous host buffers and are copied synchronously.
Optional NULL views mean absent; zero-length tensors are distinct. Raw F16 and
BF16 are 16-bit encodings, not C floating types. RGB views are interleaved RGB8
and support row padding. Frame providers may reuse a buffer between calls but
must keep it valid until the next provider call or the enclosing API returns.

## Concurrency and callbacks

Serialize calls on each image/video handle. An atomic guard returns `SAM3_BUSY`
for simultaneous calls or callback reentry; it does not recursively lock a
mutex. `sam3_video_cancel` is the exception and can be called concurrently or
from an output callback. It stops propagation after a consistent completed
frame. Do not destroy a handle during a call.

Different sessions share immutable module weights. Their state remains separate.
Context cache mutation and tokenizer use are synchronized. Process-wide thread
count/TF32 settings are exposed through `sam3_runtime_configure`; configure them
before inference or other concurrent Torch use. The reference comparisons use
four CPU threads with TF32 disabled.

A frame-provider nonzero return gives `SAM3_CALLBACK_ERROR`. Output callbacks
return zero to continue, one to stop normally, and minus one for failure. A
callback error can occur after a completed frame or object removal; it is not
a request to undo those already-completed operations.

## Image inference

Create an image handle with grounding, interactive, or both feature flags.
Both heads reuse one visual trunk evaluation. `set_rgb` and `set_rgb_batch` are
separate operations because the source single-image/batch stacking layouts must
be preserved. Batch images can have different original dimensions. Prompts and
object/query counts are caller supplied; no new cap is introduced.

Interactive prediction supports arbitrary point batches/labels, boxes, previous
288x288 mask logits, multimask selection, thresholding, hole/sprinkle correction
and either pixel or normalized coordinates. It returns `masks`, `iou` and
`low_res_logits`. `sam3_image_embedding` exposes the cached embedding.

Text can be provided as UTF-8 or raw token IDs. Counted UTF-8 entry points retain
embedded NULs through the source cleaning rules. Encoding returns `tokens`,
`padding`, `features` and `embeddings`, with the original context length.
Grounding accepts text features, repeated/reordered image and text indices,
positive/negative point/box geometry with padding, visual features and previous
mask features; text can be disabled. See the header for exact layouts.

Grounding results contain raw full-query tensors and postprocessed per-prompt
results. For example, `0/boxes`, `0/scores`, `0/masks`,
`0/mask_probabilities` and `0/query_indices` describe the first prompt batch
item. `batch_count` reports the batch size. The raw fields retain all 200 queries.
Resize chunk size bounds intermediates and does not reduce detection count.
The SAM3.1 joint-score convention is selected automatically for that model.

## Interactive video

A video session receives an RGB callback and frame dimensions/count. Both models
support points/boxes, brush masks, preflight, forward/reverse propagation,
corrections, clear/removal, reset, cancellation and arbitrary int64 object IDs.
SAM3.1 additionally exposes simultaneous brush input and optional lossless
history paging. The existing SAM3 source session disallows introducing new IDs
after propagation starts; SAM3.1 has dynamic insertion. These underlying native
session behaviors are unchanged by the C interface. In particular, SAM3.1
retains the documented native stable-slot/dense-memory policy and original-demo
adaptations; this interface does not establish equivalence to every upstream
dynamic extraction/merge path.

Temporal, multimask, non-overlap, conditioning and memory-selection options map
to the corresponding C++ session settings. Point count is uncapped. SAM3.1
history paging uses the caller's directory; SAM3 rejects a nonempty history
directory until that backend is integrated there. Outputs expose `frame`, `ids`,
`masks`, optional `low_masks` and object `logits`. Original-resolution video
masks are logits. C++ codec adapters or application-owned decoders can feed the
RGB callback; this API does not claim JPEG/MP4 decode parity.

## Example and validation

`native/tools/c_api_probe.c` is a C11 model client, including raw RGB row padding,
result retention, context lifetime, batching, callback cancellation/reentry,
provider errors, text and geometry. Its fixed fixture prompts are tests; the
library API accepts caller-supplied inputs. `c_api_smoke.c` checks ABI defaults,
invalid arguments and exception translation without weights. It also compiles
and links using `cc` directly, without any Torch include path.

`c_api_parity.py` runs the C executable with PATH=/nonexistent and compares image
refinement/video outputs against previously validated reference fixtures.
`c_api_grounding_parity.py` compares text/geometry/visual-feature composition
against the previously original-validated native components. These are distinct
validation scopes, not a claim that every high-level upstream feature is done.
The known intermittent development CPU runtime failure remains documented in
`CPU_RUNTIME_ISSUE.md`; passing CPU tests do not establish CPU stability.

No GitHub Actions are used. Windows/Turing runtime testing remains with the user.
The header uses standard C types and the implementation uses C++17/LibTorch and
existing precompiled CUDA, without a Linux-only API dependency. Current local
binaries are Linux development builds, not validated Windows/Turing releases.

### Minimal C client build

After building the shared library, this standalone smoke client can be compiled
without Torch headers or a C++ compiler invocation:

```sh
cc -std=c11 -Wall -Wextra -Werror \
  -Inative/include -Ibuild/native native/tests/c_api_smoke.c \
  -Lbuild/native -lsam3_native \
  -Wl,-rpath,"$PWD/build/native" -o build/c_api_c11
PATH=/nonexistent build/c_api_c11
```

The example uses Linux compiler/linker flags. Windows hosts use the same C
header, generated export header and the MSVC import library/DLL from their
native build. No Windows execution result is claimed. The pure C model client
additionally uses the standard C math library (`-lm` on Linux).

Grounding tensor layouts are explicit: text features `[L,Ntext,256]` and boolean
padding `[Ntext,L]`, visual features `[Nvisual,B,256]` and boolean padding
`[B,Nvisual]`, previous-mask features `[5184,B,256]` at the current 72x72 detector
grid. These are model features, not RGB pixels or binary mask prompts. Interactive
image/video brush entry points accept masks as described in the header.

`c_api_batch_parity.py` compares both image-batch positions against the matching
C++ component outputs. Floating-point execution can give slightly different
results at different batch positions, even for identical input values; that
comparison does not assume equality between positions or weaken the reference
comparison tolerance. BF16-reference mode is for reference validation on
capable hardware and is not a claim of Turing BF16 support.
