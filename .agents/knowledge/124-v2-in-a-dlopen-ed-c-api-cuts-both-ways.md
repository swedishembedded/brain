<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 124. `_v2` in a dlopen'ed C API cuts both ways

The rule that a header's `#define`d `_v2` suffix must be spelled out for
`dlsym` has a mirror image that is easier to get wrong, because the fix for
the first one makes it feel solved. CUDA 12 grew a second version of the
kernel-node parameter struct and pointed the unsuffixed *function* names at
entry points taking it. The driver keeps exporting the unsuffixed symbols at
the OLD ABI, for binaries compiled before that change - which is exactly what
a `dlsym` of the unsuffixed name binds.

So resolving the unsuffixed name is right, and it obliges the caller to pass
the OLD struct. Symbol name and struct layout are one decision, and a
comment saying so belongs on the struct: a later reader who knows only the
first half of the rule will "fix" the name and silently corrupt every field
past the point the two layouts diverge.
