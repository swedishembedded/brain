<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 463. A fused kernel that promises the chain's bits must follow the chain's selection: a transparent upgrade of one link moved them

**Lesson:** a native kernel gated bit for bit against "the WGSL chain it replaces" reproduces one particular kernel
per link of that chain. When the handle starts dispatching a different kernel for a link (a cooperative twin
appended by `gpu_core::upgrade`, an autotuned candidate), the fused kernel keeps its old arithmetic and the two
paths part by a rounding. Each is still a correct RMSNorm; together they break every claim that two tapes of one
model agree to the last bit.

## What happened

`gqa_decode_prep.cu` reproduced the per-element `rmsnorm.wgsl`: one ascending sum of squares, then
`(weight * x) * inv`. Once every device with a workgroup reduction was given the cooperative `rmsnorm_rows` for
`rmsnorm` (`rms_variant` falls back to `Gpu::twin_of`), the decode chain's QK norm became 64 strided partials folded
in order and `(x * inv) * weight`. `gqa_decode_prep_native` then failed on CUDA: with no cached keys the attention
output is the new V row alone, so the K pool was the first output to show the norm (`K pool differs ... ctx=0`).
With `gpu_core::set_kernel_upgrade(false)` the chain went back to `rmsnorm` and the unchanged kernel matched it
bit for bit, which isolated the cause to the selection, not the kernel's rotation or append.

The exact Gated DeltaNet decode bodies (`gdn_decode`, `gdn_decode_pool`, `gdn_recur_pool`, `gdn_recur_verify_pool`) had
the same defect in their gated RMSNorm (`gdn_decode_native`: "layer output differs" while the state and the window
matched), hidden on a part below compute 8.0 because the kernels were not offered there.

## What to do

- A fused kernel names the exact kernel of each link it reproduces, and its dispatch asks the chain's own selection
  seam whether that kernel is the one the chain runs now (`model::block::rms_rows_selected`); where it is not, the
  fused kernel declines rather than computing a third answer.
- A change to a link's selection (a twin, a candidate, a capability gate) reruns the bit-identity gates of every
  fused kernel whose chain contains that link.
