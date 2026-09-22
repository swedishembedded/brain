<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 120. `threads` means invocations to a catalogue kernel and blocks to a native one

`Backend::step` counts INVOCATIONS and each backend divides them by the
kernel's declared work-group size to lay out a grid. A kernel registered
through `register_native` has no such declaration to read - CUDA C++ has no
`@workgroup_size` attribute and SPIR-V's is not what the caller was thinking
in - so `step_native` takes the WORK-GROUP COUNT directly instead, the
convention `backend-vulkan` already established for its own native path.

The two conventions look identical at the call site and differ by a factor of
the block size. Dividing a block count by the block size a second time
launches a fraction of the blocks needed, and every kernel in this engine
bounds itself, so the result is not a crash or a fault: it is an output whose
tail is simply never written. Any backend implementing both paths has to
branch on which kind of kernel a recorded step names, and the branch belongs
at the single point the grid is computed rather than at each caller.
