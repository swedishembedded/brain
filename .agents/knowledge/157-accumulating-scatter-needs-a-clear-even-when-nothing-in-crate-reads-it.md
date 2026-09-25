<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 157. Half a gradient buffer was self-correcting, and that hid the other half

`Decide::accumulate_kept` is the head-only reverse pass: a frozen encoder's
output is already known, so the head runs alone over kept rows. It writes its
hidden-state gradient into the encoder's seed buffer, and
`Features::from_parts` exists precisely so an outside caller can splice in rows
the encoder never produced and then **train whatever produced them on that
gradient**.

The buffer is written by two different mechanisms, and only one of them is
self-correcting:

| rows | kernel | semantics |
|---|---|---|
| state | `row_scatter` | **ASSIGNS** - a stale row is overwritten |
| option `[CLS]` | `emb_bwd` | **ACCUMULATES** - a stale row is added to |

`Head::clear_seed` is what zeroes it, and its own doc says so: "It must be
zeroed and this is the only thing that does it." Both call sites were guarded
by `!self.frozen_encoder` - and `accumulate_kept` **requires** a frozen
encoder, so neither could ever cover it. It called `clear_seed` nowhere.

So the state half was always right and the slot half was the running sum of
every step ever taken. Repeating one identical step, nothing between them
touching a weight, measured the signature exactly:

```text
slot row 11: -0.008866322 -> -0.017732644
```

Twice, then three times, then ten.

## Why it survived

Nothing inside `crates/decide` reads that buffer on this path - a frozen
encoder's reverse pass is not run. The only consumer is an external projector,
and the one that existed (`samples/learning/visualclick`) read **only the state
region**, which is the assigned half. The defect sat behind a correct-looking
gate, in the half nobody had yet asked for, in a buffer the owning crate
deliberately ignores.

It is not visible as a crash, a NaN or a shape error. A projector trained on it
follows a gradient whose magnitude grows without bound while its direction stays
roughly right, so the run still *trains* - it just trains on a number that means
something different at every step.

## The general shape

**A buffer written by an accumulating kernel needs its clear on every path that
writes it, including the paths whose own crate never reads the result.** The
guard to check is not "does anything here consume this" but "can anything
consume this" - and an API that exists to hand a buffer to an outside caller
answers that yes, by construction.

Where two regions of one buffer are filled by different kernels, the
assigning one will mask the accumulating one for exactly as long as no caller
needs both. Test the halves separately, and test them by **repeating an
identical step**: any residue from the previous one shows up as a gradient that
moved when nothing did. A finite-difference check against the loss catches the
direction being wrong but not the magnitude drifting, because it re-derives
from a single step.

Fixed by clearing in `accumulate_kept` when it is passed a seed buffer - once
per call, unlike `accumulate_batch`'s once-per-pass, because this path carries a
single example and has no sibling whose gradient a clear could erase. Regression
test: `crates/decide/tests/frozen_encoder.rs`'s
`the_kept_path_leaves_this_steps_gradient_not_a_running_sum`, plus the
`no stale gradient` check in `samples/learning/visualclick --validate`.
