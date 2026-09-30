<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 180. A declared range is enforced, or it is not a limit

Nearly every numeric action param declared a `min`/`max`, and the served
manifest published them, but `ActionSpec::validate` only checked a value's
type. The ranges were slider hints. A D-Bus caller could ask `lora_train`
for a context of 10⁹ tokens, and the action would try to allocate it.

`validate` now refuses a caller's value outside the declared range and names
the param and the bound. Two consequences followed.

- **A declared max has to be a real bound.** qwen3's `generate` capped
  `max_new` at 32768, and `lora_train` capped `block` at 32768. Neither
  number came from the model or the hardware. Once enforced, both would have
  refused long-context requests the engine can serve. Both maxima are gone:
  - a generation's context is bounded by its footprint check
    (`footprint::place_and_build`);
  - a training build now passes through the same check;
  - the served chat surface answers `context_length_exceeded`.

  A max stays only where it is real: T5's text length, an image size a
  pipeline supports, or `top_k`'s 0..=1000.
- **The API surface checks its own fields first.** apiserve maps
  `max_tokens`/`max_completion_tokens` to `max_new` and `temperature` to
  `temp`. It checks `temperature` (0..=2), `top_p` (0..=1) and the
  completion budget (a positive integer) before building the invocation. A
  client then gets a 400 that names its own field, instead of the generic
  failure a refused invocation maps to.

Removing `max_new`'s max exposed two defects in `generate` that the cap had
hidden. The context size was computed as `(prompt + max_new) as u32`, which
truncated silently, and the output buffer was preallocated with
`Vec::with_capacity(max_new)`, which aborts the process on a large value.
The context size is now checked arithmetic, and the preallocation is bounded
by the built context.

`catalog`'s `every_default_lies_within_its_declared_range` keeps a default
from falling outside its own range, where a UI echoing it back would be
refused.
