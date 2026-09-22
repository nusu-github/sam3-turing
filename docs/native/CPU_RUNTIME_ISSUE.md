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
