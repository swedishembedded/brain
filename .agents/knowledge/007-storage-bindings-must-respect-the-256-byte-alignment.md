<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 7. Storage bindings must respect the 256-byte alignment

**Lesson:** Storage bindings must respect the 256-byte alignment, or a parity gate can be
silently disabled by a validation error nobody re-runs.

## What happened
`min_storage_buffer_offset_alignment` is 256 bytes = **64 floats**. Slicing a buffer at a row
offset requires `r0 * d` to be a multiple of 64. Violating it is a wgpu **validation error**,
not a wrong number. That sounds safe until it silently disables a parity gate nobody re-runs.

## What to do
Where offsets are genuinely ragged (windowed attention), pass them in the kernel's own `Params`
instead of binding a slice.
