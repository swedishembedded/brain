<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 124. `_v2` in a dlopen'ed C API cuts both ways

**Lesson:** Symbol name and struct layout are one decision. Resolving an unsuffixed name obliges the
caller to pass the struct of that name's ABI, and a comment saying so belongs on the struct.

## What happened
The rule that a header's `#define`d `_v2` suffix must be spelled out for `dlsym` has a mirror image
that is easier to get wrong, because the fix for the first one makes it feel solved.

CUDA 12 grew a second version of the kernel-node parameter struct and pointed the unsuffixed
*function* names at entry points taking it.

## Why
The driver keeps exporting the unsuffixed symbols at the OLD ABI, for binaries compiled before that
change. That is exactly what a `dlsym` of the unsuffixed name binds.

## What to do
Resolve the unsuffixed name, and pass the OLD struct.

Put a comment saying so on the struct. A later reader who knows only the first half of the rule will
"fix" the name and silently corrupt every field past the point the two layouts diverge.
