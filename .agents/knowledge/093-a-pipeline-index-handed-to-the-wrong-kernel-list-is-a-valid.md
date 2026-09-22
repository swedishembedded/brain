<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 93. A pipeline INDEX handed to the wrong kernel list is a valid dispatch of the wrong kernel - and the wgpu error names the innocent call site, not the wrong list

`pulid::caps::Bundle::load` built the EVA-CLIP image tower with
`EvaVision::new_on(gpu.new_like(clip::model::TEXT_PIPELINES), ..)`. `EvaVision`
is the **vision** tower: its `V_*` constants are positions in
`VISION_PIPELINES`. Given the text list instead, every index still resolved -
to a different, perfectly valid kernel:

| `EvaVision` wants | index | got, in `TEXT_PIPELINES` | bindings |
|---|---|---|---|
| `conv2d` | 0 | `embed` | 4 vs 4 - **accepted** |
| `nchw_nlc` | 1 | `region_copy` | 3 vs 3 - **accepted** |
| `region_copy` | 2 | `pos_add` | 3 vs 3 - **accepted** |
| `bias_add` | 3 | `layernorm` | 3 vs 5 - rejected |

The first three arities agree, so wgpu accepts them and the tower would have
computed silent garbage. The fourth is where the arities finally disagree, and
that is the only reason this failed loudly at all:

```
In Device::create_bind_group
  Number of bindings in bind group descriptor (3) does not match the number
  of bindings defined in the bind group layout (5)
```

**The trap.** The panic frame really is in `EvaVision::new_on` - it is not
wgpu deferring an error from an earlier dispatch, which is the first thing
everyone suspects. But `EvaVision::new_on` only uploads weights and pre-builds
its step tape, and nothing in it legitimately builds a 5-binding bind group, so
the frame looks impossible and the search goes looking for a `.wgsl`/Rust
arity mismatch in some *other* crate. There is no such mismatch. Every kernel
is self-consistent; only the *list* was wrong. Two corollaries:

* A minimal repro of the suspected call site **cannot** reproduce it, because
  in isolation you naturally pass the correct list. That negative result is
  not evidence the call site is innocent - it is evidence the caller is guilty.
* The load order is a red herring. Nothing that ran earlier (the DiT, BiSeNet,
  ArcFace) mattered; the argument at the failing call site was the whole bug.

**Fixes, in the same change.** `EvaVision` now resolves `VISION_PIPELINES` by
NAME through `gpu.kernel_index` and remaps `V_*` through it - the same seam
`imaging::ImagingKernelIds` and `vision::ConvKernelIds` already use. That is
required, not cosmetic: `clip::caps::Session` puts the text tower, the EVA
tower and `imaging`'s resize on ONE device, and two towers cannot both be
0-based on one list, so `caps`'s own `embed_image` had the identical latent
defect. A missing kernel now panics naming the kernel instead of corrupting a
bind group.

`backend-wgpu` now LABELS every bind group with its kernel's registered name,
so the same failure reads `create_bind_group, label = 'layernorm'` while the
code is plainly building a `bias_add`. One word turns a multi-hour hunt into
a glance. Any crate still indexing positionally gets this for free.

**Smell to grep for.** `new_like(<some other module>::PIPELINES)` where the
module does not match the type being constructed. Related: lesson #90, a
`Step` is only meaningful to the `Gpu` handle that created it - same family,
an index or a handle crossing a boundary it was never valid across.

The regression test (`crates/clip/tests/vision_kernel_ids.rs`) reproduces the
original 1.1 GiB-checkpoint crash in 0.35 s from a 2-block synthetic tower, by
A/B-ing the same tower on the bare vision list and on the shared union. A
whole class of "wrong device/wrong list" bug is reachable this cheaply.
