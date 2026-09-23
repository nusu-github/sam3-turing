# Intermittent CPU prompt-encoder crash in the development environment

Observed on 2026-09-22 with the NVIDIA development PyTorch build
`2.10.0a0+b558c986e8.nv25.11`, Python 3.12, NumPy 1.26.4, Linux x86-64.
CPU image-host parity intermittently terminates with SIGSEGV while the original
Python `PromptEncoder.get_dense_pe()` executes `torch.sin` / `torch.cos` and
concatenates their results (`sam3/sam/prompt_encoder.py:231`). The UCX signal
handler reports a null-address access. This does not identify the faulty native
library or establish a root cause.

The following **original-only** diagnostic does not load `libsam3_native`, model
weights, or the added interactive image session:

```sh
PYTHONFAULTHANDLER=1 .venv/bin/python native/tests/diagnose_cpu_prompt_encoder.py
```

Five fresh-process runs of this diagnostic returned `[0, 0, -11, 0, 0]`.
The failing process's Python stack ends in the same positional-encoding call.
This rules out the new native session as a prerequisite for reproducing the
failure. It does not prove that native CPU execution is unaffected. A GDB run
of the complete CPU comparison exited normally; no useful failing native
backtrace has yet been captured. No concurrent library relinking occurred
during these failures.

The completed CPU comparison reports 29 exactly matching cases. Earlier failed
runs and the original-only reproduction logs are retained privately alongside
the successful report in `native-foundation/interactive-image-linux-cuda13`.
CUDA image-host, standalone real-image and batch comparisons completed without
this failure. These observations establish output parity for the completed runs,
not CPU stability. No speculative runtime workaround has been added. Root-cause
investigation and validation against a redistributable LibTorch build remain
open; Windows/Turing validation is separately owned by the user.

During lossless history-paging validation, a CPU FP32 native-only session
comparison also terminated with SIGSEGV (exit 139). The test had printed
`START fp32 score False`; it had not yet printed a completed comparison. UCX
again reported a null-address access, without a useful native stack. This
runner loads `libsam3_native` but does not invoke the original Python model.
Its first comparison executes the resident policy before the paged policy;
the failure log alone does not localize which operation failed or establish
whether paging had begun. No shared-library relinking occurred during the run
(only the independent standalone update executable was compiled afterward).
This broadens the observed CPU instability; it does not prove the new archive
code caused the failure or that it is the same root cause as the earlier
original-only crash. The failure log is retained with the history-paging
artifacts. CPU stability remains unverified even when individual comparisons
and standalone probes pass.

A GDB rerun of the complete native-only history comparison subsequently exited
normally: two 16-operation CPU FP32 workflows produced 926 exact resident-vs-paged
tensor comparisons. It did not capture a failing native backtrace or resolve
the intermittent crash. Standalone CPU session and direct mask-update probes
also completed, including archive-read rollback and cleanup checks.

During shared-video-frame comparison on 2026-09-23, original-model construction
failed again before the first frame. A GDB rerun captured a null program counter
inside `mkl_vml_serv_threader_s_1i_1o._omp_fn`, reached through `vmsErfInv`,
ATen's AVX2 `erfinv_kernel`, TensorIterator and the OpenMP parallel launcher.
The MKL libraries were `/usr/local/lib/libmkl_gnu_thread.so.1` and
`libmkl_intel_lp64.so.1`. This occurred during source weight initialization with
four ATen CPU threads, before native frame evaluation. No library relinking
occurred. The failing log and GDB backtrace are retained in the private
`video-frame-linux-cuda13` snapshot.

Setting `torch.set_num_threads(1)` in this development comparison allowed the
initial SAM3/SAM3.1 FP16 real-frame comparison to complete. This is a local test
workaround, not a demonstrated portable runtime fix. The new stack localizes
this particular failure; it does not establish that the earlier positional
encoding and native history crashes share its cause. CPU stability and testing
against a redistributable LibTorch build remain open.

During the singleton-stride regression, the CPU FP32 memory-storage comparison
again terminated with a null-address UCX/SIGSEGV report under four ATen threads.
No relinking of its loaded CPU library occurred. The log does not localize the
failure. A fresh run with one ATen thread completed all ten operations and 452
exact comparisons, including immediate source memory encoding, bucket rebuild,
resident/offloaded/paged equivalence and reverse propagation. The test now accepts
an explicit `--threads` setting (default remains four). This repeat is retained
privately with `video-memory-stride-linux-cuda13`; the successful one-thread run
does not establish general CPU stability or prove a common cause with earlier
MKL crashes.

## Official standalone LibTorch follow-up

Official LibTorch 2.10.0 CPU has now passed all 16 native CTests and actual text
encoding through an installed C11 client, including a Python-free chroot. Its
dynamic dependency tree does not use the development image's MPI/UCX/external
MKL libraries. This changes the runtime under test; it does not diagnose the
earlier intermittent crash or establish long-run CPU stability. See SDK.md.
