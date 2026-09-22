<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 57. `0` is not a null pipeline index - an unused kernel slot must hold a sentinel, and only `backend-cpu` makes that a silent corruption

Every model in this tree fills an index struct (`model::block::KernelIds`,
`model::vit::VitKernelIds`, `vision::ConvKernelIds`, ...) from its own
`PIPELINES` list, and every such struct has slots the model never dispatches -
backward ids on a forward-only path, GQA ids on a model that rotates through
`rope2d`, RMSNorm ids on a model whose only `block` call is `swiglu_fwd`. The
tempting filler is `0`.

**`0` is a real, registered kernel in every pipeline list in this workspace.**
It was `conv1d` in one file and `matmul` in its sibling - both dispatched on
every single pass. A builder that reaches such a slot does not fail; it runs a
live kernel against a different kernel's bindings and uniform.

What makes it a *silent* defect rather than a loud one is the backend split:

* On a GPU backend the bind-group/binding-count check turns the misroute into a
  panic, so it surfaces the first time anyone takes that branch.
* On `backend-cpu` there is **no buffer-count and no uniform-size check** at
  dispatch. The kernel reads whatever is at those pointers. It is an
  out-of-bounds read that returns plausible numbers, and it is invisible to
  every unit test on that backend - which, for at least one crate here, is
  *every* unit test.

`model::vit::UNREGISTERED = usize::MAX` already existed and already documented
exactly this; `model::block` did not have it, and 13 sites across the workspace
had filled slots with `0` or - the same defect wearing a better disguise - with
a live index for a *different* kernel, commented as a harmless placeholder.
A sentinel that is out of range of `PIPELINES` converts all of it into an
index-out-of-bounds panic at the first dispatch.

**Gate it on DISPATCH, not statically.** "Every unused slot equals the
sentinel" is easy to write and cannot tell you whether getting it wrong would
have mattered. Arm the device's own per-kernel counters
(`Gpu::reset_ops_counters` / `Gpu::ops_counters`), run the real pass, and assert
that no unused slot's index names a kernel that pass actually dispatched. That
version names the kernel a wrong slot would have run - and it goes red on the
one-character mutation, which is the only evidence that a gate is a gate.

The general shape, beyond kernel indices: **a sentinel whose value is a legal
member of the domain is not a sentinel.** Index 0, empty string, epoch zero,
`Duration::ZERO` - each of them means something real somewhere. Pick a value the
consuming layer will reject, and check that the consuming layer really does
reject it.
