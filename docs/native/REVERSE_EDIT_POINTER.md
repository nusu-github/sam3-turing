# Reverse point-edit pointer provenance

The previously unresolved 120-pixel SAM3.1 reverse-edit discrepancy is now
localized to **a stale object pointer in the explicitly adapted original
reference**, not a native neural decoder discrepancy. The native production
predictor did not change in this investigation. This does not establish parity
with unmodified original high-level state handling or dataset-wide quality.

## Evidence

The existing full 34-frame / 200-query regression still reproduces the 120-pixel
difference at reverse frame 17 after repeated point edits on frame 19. It uses
BF16/no TF32 on Blackwell and the previously documented reference adapters in
[VIDEO_EDIT_SEQUENCE.md](VIDEO_EDIT_SEQUENCE.md). Native outputs match the prior
native run byte-for-byte at the affected frame.

An opt-in capture compares 51 tensors immediately before temporal conditioning:

- 48 match exactly. History memory, positions, image streams and other
  object pointers agree.
- `cond19.pointer` differs in 256 elements, maximum absolute difference 0.53515625.
  The assembled memory differs only in those 256 elements; conditioned image
  features consequently differ.
- Capturing the original neural point outputs proves that the reference pointer
  comes from the first, two-point edit. The native pointer exactly equals the
  original neural output of the subsequent three-point edit.
- Replaying identical recorded inputs through the native temporal conditioner
  reproduces its original conditioned features exactly. Changing **only** the
  frame-19 pointer to the old reference pointer reproduces the old reference
  conditioned features exactly. Both maximum errors are zero.

`VideoTrackingMultiplexDemo._consolidate_temp_output_across_obj` chooses a
conditioning entry before a non-conditioning entry even when consolidating newer
non-conditioning point output. Temporary masks carry the newer click while
`obj_ptr` is copied from the older conditioning entry. The earlier
`--sam31-refresh-refined-memory` adapter preserves newly encoded memory, but that
consolidated output already contains the stale pointer.

## Explicit reference repair and limits

The new opt-in `--sam31-refresh-refined-pointer` adapter retains `obj_ptr` from
the original newer non-conditioning entry when it collides with an older
conditioning entry on a point-annotated frame. It uses no native tensor and
changes no neural computation. It composes with the previous opt-in memory,
extraction, repeated-refinement and singleton-history adapters. The actual audit
records one pointer replacement at frame 19.

With this additional repair, all 51 compared tensors—including assembled memory
and conditioned features—match exactly. All 15 edited/propagated/fetched outputs
match in IDs, probabilities, boxes and binary masks. The 34 pre-edit raw and 34
final outputs also remain exact. The unmodified-pointer reference and its 120
pixel difference remain preserved separately; tolerances were not relaxed.

This closes the cause investigation for this fixture. It is not evidence that
all interactive sequences match the unmodified original, nor that all source
state-management defects have been identified. The full-function deployment and
broader quality/performance goal remains open. Windows/Turing physical execution
remains with the user; no Actions were used.

## Reproduction and retained data

`sam3_video_pipeline_probe --edit-sequence-trace` adds `reverse-input/tensors.tsv`,
float32 value files and `plan.tsv` to the existing full sequence outputs. It
runs an extra diagnostic temporal conditioner without changing predictor state.
`video_pipeline_parity.py --trace-edit-state` saves original history/assembled
inputs and the original frame-19 neural point outputs. These are development
probes; production execution has no Python dependency.

Run the original sequence with its previously documented flags first, storing
its trace in `reference`, then again with `--sam31-refresh-refined-pointer` in
`pointer-reference`. Keep native output in `native`. The comparators are:

```sh
python native/tests/reverse_edit_trace_parity.py /path/to/reverse-edit-audit \
  --report provenance.json
python native/tests/reverse_edit_pointer_replay.py /path/to/libsam3_native.so \
  /path/to/native-weights-v1 /path/to/reverse-edit-audit --report replay.json
```

The provenance comparator checks actual original first/latest pointer values
and requires all 51 repaired comparisons to be exact. The replay checks the two
causal cases using recorded layouts, dtypes and temporal settings. Private
`reverse-edit-audit` retains the real tensors; the matching development snapshot
contains commands, logs, probe and public reports. Historical reports and the
unrepaired reference are retained. The first new reference harness run failed
because nested diagnostic decorators did not preserve their signature; `wraps`
fixed that test-hook issue before the successful exact run.
