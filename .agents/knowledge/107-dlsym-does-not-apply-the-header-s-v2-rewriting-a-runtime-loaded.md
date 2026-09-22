<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 107. `dlsym` does not apply the header's `_v2` rewriting - a runtime-loaded driver binds the OLD ABI unless the suffixed name is asked for

`cuda.h` declares `cuDeviceTotalMem(size_t *bytes, CUdevice)` and, 4000 lines
earlier, `#define cuDeviceTotalMem cuDeviceTotalMem_v2`. A C compiler therefore
emits a reference to `cuDeviceTotalMem_v2`, which is what the 64-bit
out-parameter belongs to. `libcuda.so.1` still exports the *unsuffixed* v1
symbol for old binaries, and its `bytes` out-parameter is a 32-bit `unsigned
int`.

Nothing about that is visible from Rust: `libloading` resolves whatever name
you spell, the signature you declare is never checked against the library, and
the call succeeds. The result is a write of 4 bytes where the caller allocated
and reads 8 - so on a little-endian host a card's VRAM comes back correct
whenever it fits in 32 bits and wildly wrong (upper half uninitialised) the
moment it does not, with no error anywhere. Every `_v2`/`_v3` entry point in
the CUDA Driver API has this shape; the `#define` block at the top of `cuda.h`
is the authoritative list of which ones.

**Generalises to any `dlopen`ed C API**: a header's `#define` layer is part of
the ABI contract and is invisible to a symbol loader. Read the `#define`s, not
just the prototypes, and bind the name the preprocessor would have produced.

Related, same file: resolve each symbol AT its function-pointer type
(`lib.get::<unsafe extern "C" fn(..) -> ..>`) rather than fetching a
`*mut c_void` and `transmute`ing it. Both are equally unchecked against the
library, but the typed form writes each signature exactly once - at the struct
field it fills - so a mismatch between the declaration and the call site is a
compile error instead of silent stack corruption.
