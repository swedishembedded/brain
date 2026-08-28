// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A Q8_0 GGUF tensor straight into brain's own packed-int8 layout: a byte
//! repack, not a dequantize-then-requantize, now that `model::int8::GROUP`
//! (32) matches Q8_0's own block size exactly. `crates/flux2` proved this
//! bit-exact against the fp32 round trip first (`tests/gguf_direct_int8.rs`);
//! this module is that proof's implementation, lifted here so a second
//! GGUF-sourced model does not re-derive it - the underlying arithmetic fact
//! (`deq_q8_0` needs at most 18 significand bits against fp32's 24, so
//! decoding a Q8_0 block reproduces the round trip's f32 input exactly) does
//! not change per caller.

use checkpoint::gguf::MmapGguf;

/// Requantize the rectangle `rows [r0, r0+n_out) x cols [c0, c0+k)` of a
/// tensor stored as `[_, stride]` straight from Q8_0 blocks to
/// `(packed, scales)` - [`model::int8::quantize_weight`]'s own output shape.
///
/// `None` means "use the fp32 route": the tensor is not Q8_0, or the
/// rectangle's bounds are not Q8_0-block-aligned (`stride`/`c0`/`k` each a
/// multiple of the block size). Returning `None` is always safe - a caller's
/// fp32 fallback produces the same bytes by the longer route. Callers that
/// need to exclude a tensor for a reason of their own (flux2 excludes one a
/// LoRA touches, since the fold needs a float domain) check that BEFORE
/// calling this - it is not this function's concern.
pub fn try_i8_rect(gguf: &MmapGguf, name: &str, stride: usize, r0: usize, n_out: usize, c0: usize, k: usize) -> Option<(Vec<u32>, Vec<f32>)> {
    let be = checkpoint::gguf::Q8_0_BLOCK_ELEMS;
    if !stride.is_multiple_of(be) || !c0.is_multiple_of(be) || !k.is_multiple_of(be) {
        return None;
    }
    let (raw, ty) = gguf.raw_tensor_bytes(name)?;
    if ty != checkpoint::gguf::TYPE_Q8_0 {
        return None;
    }
    let kg = k / 4;
    let gs = k / model::int8::GROUP;
    let mut packed = vec![0u32; n_out * kg];
    let mut sw = vec![0f32; n_out * gs];
    // One output row per task: each reads only its own block range and
    // writes only its own words and scales, so this is bit-identical to the
    // serial form and to the fp32 round trip alike.
    backend_cpu::par::chunks2_mut(&mut packed, kg, &mut sw, gs, |i, prow, srow| {
        let mut row = Vec::with_capacity(k);
        let e0 = (r0 + i) * stride + c0;
        checkpoint::gguf::q8_0_expand(raw, e0, e0 + k, &mut row).expect("block-aligned above");
        model::int8::group_scales(&row, srow);
        model::int8::pack_row(&row, srow, prow);
    });
    Some((packed, sw))
}
