// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The codec registry: CPU reference decode (and, for the formats a load can
//! target, encode) of one *tile* of a quantized tensor.
//!
//! A tile is a flat element range aligned to [`TensorDescriptor::tile_quantum`]
//! together with exactly the payload bytes and scale values that cover it
//! ([`Tile`]). Decoding never needs bytes outside the tile, so a consumer's
//! scratch is bounded by the tile, not by the tensor.
//!
//! [`CodecRegistry::reference`] holds the built-in codecs. Adding a format means
//! registering a [`Codec`]; no loader or model crate changes. Existing decoders
//! are reused, not copied: GGUF block formats decode through
//! [`crate::gguf::block_expand`], FP8 bytes through [`crate::safetensors`], the
//! NF4 table is the one `kernels` renders into the device decode.

use std::ops::Range;
use std::sync::Arc;

use super::encoding::{Codebook, Element, FloatFormat, QuantEncoding, ScaleGranularity, TensorDescriptor};
use crate::safetensors::{e4m3fn_to_f32, e5m2_to_f32};

pub mod rows;

/// The OCP E2M1 magnitudes for the low three bits; bit 3 is the sign.
const E2M1_MAGNITUDE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// The NF4 codebook: code `i` decodes to `NF4_CODEBOOK[i]` times its block scale.
pub use kernels::template::NF4_CODEBOOK;

/// The E2M1 codebook indexed by the 4-bit code (sign in bit 3).
pub const E2M1_CODEBOOK: [f32; 16] = {
    let mut t = [0f32; 16];
    let mut i = 0;
    while i < 8 {
        t[i] = E2M1_MAGNITUDE[i];
        t[i + 8] = -E2M1_MAGNITUDE[i];
        i += 1;
    }
    t
};

/// One tile of a tensor as read from storage.
#[derive(Debug)]
pub struct Tile<'a> {
    /// Flat element range of the tensor this tile covers.
    pub elems: Range<u64>,
    /// `desc.payload_range(&elems)` worth of payload bytes.
    pub payload: &'a [u8],
    /// For each scale level `l`, `desc.scale_range(l, &elems)` worth of sidecar bytes.
    pub scales: Vec<&'a [u8]>,
}

/// What an encoder needs to know beyond the tile itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct EncodeCtx {
    /// Largest magnitude over the whole tensor, for formats with a per-tensor scale (NVFP4);
    /// the load executor measures it in a first pass when the destination needs it.
    pub tensor_absmax: f32,
}

/// A tile being produced by an encoder: owned buffers the codec fills.
#[derive(Debug, Default, Clone)]
pub struct TileBuf {
    pub payload: Vec<u8>,
    pub scales: Vec<Vec<u8>>,
}

/// Reference decode (and optionally encode) for a family of encodings.
pub trait Codec: Send + Sync {
    /// Short name for reports.
    fn name(&self) -> &'static str;
    fn can_decode(&self, enc: &QuantEncoding) -> bool;
    fn can_encode(&self, _enc: &QuantEncoding) -> bool {
        false
    }
    /// Decode `tile` to f32 into `out` (cleared first; reserve ahead to keep it
    /// allocation free). The tile must be aligned to the tensor's tile quantum.
    fn decode_tile(&self, desc: &TensorDescriptor, tile: &Tile<'_>, out: &mut Vec<f32>) -> Result<(), String>;
    /// Encode `values` (the f32 values of tile `elems`) into `out`, replacing its
    /// contents. The tile must be aligned to [`TensorDescriptor::tile_quantum`].
    fn encode_tile(&self, desc: &TensorDescriptor, _elems: &Range<u64>, _values: &[f32], _ctx: &EncodeCtx, _out: &mut TileBuf) -> Result<(), String> {
        Err(format!("{}: codec {} cannot encode {}", desc.name, self.name(), desc.encoding))
    }
}

/// An ordered set of codecs; the first that accepts an encoding serves it.
#[derive(Clone)]
pub struct CodecRegistry {
    codecs: Vec<Arc<dyn Codec>>,
}

impl CodecRegistry {
    /// The built-in reference codecs.
    pub fn reference() -> CodecRegistry {
        CodecRegistry { codecs: vec![Arc::new(ScalarCodec), Arc::new(GgmlCodec)] }
    }

    /// Add a codec, consulted before the built-ins so it can override them.
    pub fn register(&mut self, codec: Arc<dyn Codec>) {
        self.codecs.insert(0, codec);
    }

    pub fn decoder(&self, enc: &QuantEncoding) -> Result<&dyn Codec, String> {
        self.codecs.iter().find(|c| c.can_decode(enc)).map(|c| c.as_ref()).ok_or_else(|| format!("no codec can decode {enc}"))
    }

    pub fn encoder(&self, enc: &QuantEncoding) -> Result<&dyn Codec, String> {
        self.codecs.iter().find(|c| c.can_encode(enc)).map(|c| c.as_ref()).ok_or_else(|| format!("no codec can encode {enc}"))
    }
}

fn check_tile(desc: &TensorDescriptor, tile: &Tile<'_>) -> Result<(), String> {
    let q = desc.tile_quantum();
    let numel = desc.numel();
    let e = &tile.elems;
    if e.start >= e.end || e.end > numel || !e.start.is_multiple_of(q) || !(e.end - e.start).is_multiple_of(q) && e.end != numel {
        return Err(format!("{}: tile {e:?} is not aligned to quantum {q} within {numel} elements", desc.name));
    }
    let want = desc.payload_range(e);
    if tile.payload.len() as u64 != want.end - want.start {
        return Err(format!("{}: tile payload is {} bytes, expected {}", desc.name, tile.payload.len(), want.end - want.start));
    }
    if tile.scales.len() != desc.encoding.scales.len() {
        return Err(format!("{}: tile has {} scale slices, encoding has {} levels", desc.name, tile.scales.len(), desc.encoding.scales.len()));
    }
    for (l, s) in tile.scales.iter().enumerate() {
        let r = desc.scale_range(l, e);
        let bytes = (r.end - r.start) * desc.encoding.scales[l].format.bits() as u64 / 8;
        if s.len() as u64 != bytes {
            return Err(format!("{}: scale level {l} slice is {} bytes, expected {bytes}", desc.name, s.len()));
        }
    }
    Ok(())
}

/// Decode one stored code of a float format.
pub fn decode_float_code(f: FloatFormat, code: u32) -> f32 {
    match f {
        FloatFormat::F32 => f32::from_bits(code),
        FloatFormat::F16 => half::f16::from_bits(code as u16).to_f32(),
        FloatFormat::Bf16 => f32::from_bits(code << 16),
        FloatFormat::E4M3Fn => e4m3fn_to_f32(code as u8),
        FloatFormat::E5M2 => e5m2_to_f32(code as u8),
        FloatFormat::E2M1 => E2M1_CODEBOOK[(code & 0xF) as usize],
        FloatFormat::E8M0 if code == 0xFF => f32::NAN,
        FloatFormat::E8M0 => 2f32.powi(code as i32 - 127),
    }
}

/// The `i`-th value of a little-endian packed sidecar of format `f`.
fn scale_at(f: FloatFormat, bytes: &[u8], i: usize) -> f32 {
    match f.bits() {
        32 => decode_float_code(f, u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())),
        16 => decode_float_code(f, u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]) as u32),
        8 => decode_float_code(f, bytes[i] as u32),
        _ => unreachable!("sub-byte scale formats are not supported (validated by ScalarCodec::can_decode)"),
    }
}

/// Decode the unscaled element values of `n` codes.
fn decode_elements(el: Element, payload: &[u8], n: usize, out: &mut Vec<f32>) -> Result<(), String> {
    out.clear();
    match el {
        Element::Float(f) => match f.bits() {
            // The checkpoint formats of weights get a loop of their own so the format match is outside it and the
            // conversion vectorizes.
            32 if f == FloatFormat::F32 => fill_decoded::<4>(payload, n, out, f32::from_le_bytes),
            16 if f == FloatFormat::Bf16 => fill_decoded::<2>(payload, n, out, |b| f32::from_bits(u32::from(u16::from_le_bytes(b)) << 16)),
            32 => fill_decoded::<4>(payload, n, out, |b| decode_float_code(f, u32::from_le_bytes(b))),
            16 => fill_decoded::<2>(payload, n, out, |b| decode_float_code(f, u32::from(u16::from_le_bytes(b)))),
            8 => fill_decoded::<1>(payload, n, out, |b| decode_float_code(f, u32::from(b[0]))),
            4 => decode_nibbles(payload, n, &E2M1_CODEBOOK, out),
            b => return Err(format!("unsupported float width {b}")),
        },
        Element::Codebook(Codebook::Nf4) => decode_nibbles(payload, n, &NF4_CODEBOOK, out),
        Element::Int { bits, signed } => match (bits, signed) {
            (8, true) => out.extend(payload.iter().take(n).map(|&b| b as i8 as f32)),
            (8, false) => out.extend(payload.iter().take(n).map(|&b| b as f32)),
            (16, true) => out.extend(payload.as_chunks::<2>().0.iter().take(n).map(|b| i16::from_le_bytes(*b) as f32)),
            (16, false) => out.extend(payload.as_chunks::<2>().0.iter().take(n).map(|b| u16::from_le_bytes(*b) as f32)),
            (32, true) => out.extend(payload.as_chunks::<4>().0.iter().take(n).map(|b| i32::from_le_bytes(*b) as f32)),
            (32, false) => out.extend(payload.as_chunks::<4>().0.iter().take(n).map(|b| u32::from_le_bytes(*b) as f32)),
            (4, signed) => out.extend((0..n).map(|i| {
                let c = (payload[i / 2] >> (4 * (i % 2))) & 0xF;
                if signed { (((c << 4) as i8) >> 4) as f32 } else { c as f32 }
            })),
            (b, _) => return Err(format!("unsupported integer width {b}")),
        },
        Element::Ggml(_) => unreachable!("GGUF blocks are decoded by GgmlCodec"),
    }
    Ok(())
}

/// Elements decoded per parallel work item: large enough that dispatch is noise, small enough to spread one tile.
#[cfg(not(target_arch = "wasm32"))]
const DECODE_CHUNK: usize = 1 << 16;

/// Decode the first `n` fixed-width codes of `payload` into `out` (replacing its contents), `decode` mapping one
/// little-endian code to its value. Elements are independent, so the tile is cut into contiguous chunks run on the
/// CPU scheduler's pool; the values are the serial loop's.
fn fill_decoded<const W: usize>(payload: &[u8], n: usize, out: &mut Vec<f32>, decode: impl Fn([u8; W]) -> f32 + Sync) {
    let codes = &payload.as_chunks::<W>().0[..n.min(payload.len() / W)];
    out.clear();
    out.resize(codes.len(), 0.0);
    #[cfg(not(target_arch = "wasm32"))]
    backend_cpu::par::chunks_mut(out, DECODE_CHUNK, |c, dst| {
        for (d, &code) in dst.iter_mut().zip(&codes[c * DECODE_CHUNK..]) {
            *d = decode(code);
        }
    });
    #[cfg(target_arch = "wasm32")]
    for (d, &code) in out.iter_mut().zip(codes) {
        *d = decode(code);
    }
}

fn decode_nibbles(payload: &[u8], n: usize, table: &[f32; 16], out: &mut Vec<f32>) {
    out.extend((0..n).map(|i| table[((payload[i / 2] >> (4 * (i % 2))) & 0xF) as usize]));
}

/// Multiply the decoded tile by every scale level covering it.
fn apply_scales(desc: &TensorDescriptor, tile: &Tile<'_>, out: &mut [f32]) {
    let inner = *desc.shape.last().unwrap_or(&1) as usize;
    for (l, level) in desc.encoding.scales.iter().enumerate() {
        let bytes = tile.scales[l];
        let first = desc.scale_range(l, &tile.elems).start as usize;
        let at = |abs: usize| scale_at(level.format, bytes, abs - first);
        match level.granularity {
            ScaleGranularity::Tensor => {
                let s = at(0);
                out.iter_mut().for_each(|v| *v *= s);
            }
            ScaleGranularity::Block(n) => {
                let base = tile.elems.start as usize / n as usize;
                for (b, c) in out.chunks_mut(n as usize).enumerate() {
                    let s = at(base + b);
                    c.iter_mut().for_each(|v| *v *= s);
                }
            }
            ScaleGranularity::Row => {
                let base = tile.elems.start as usize / inner;
                for (r, c) in out.chunks_mut(inner).enumerate() {
                    let s = at(base + r);
                    c.iter_mut().for_each(|v| *v *= s);
                }
            }
            ScaleGranularity::Tile2d { rows, cols } => {
                let (rows, cols) = (rows as usize, cols as usize);
                let per_band = inner.div_ceil(cols);
                let r0 = tile.elems.start as usize / inner;
                for (r, row) in out.chunks_mut(inner).enumerate() {
                    let band = (r0 + r) / rows;
                    for (cb, c) in row.chunks_mut(cols).enumerate() {
                        let s = at(band * per_band + cb);
                        c.iter_mut().for_each(|v| *v *= s);
                    }
                }
            }
        }
    }
}

/// Integer, float and codebook elements with a scale hierarchy.
struct ScalarCodec;

impl Codec for ScalarCodec {
    fn name(&self) -> &'static str {
        "scalar"
    }

    fn can_decode(&self, enc: &QuantEncoding) -> bool {
        !matches!(enc.element, Element::Ggml(_)) && enc.scales.iter().all(|s| s.format.bits() >= 8)
    }

    fn can_encode(&self, enc: &QuantEncoding) -> bool {
        rows::encode_kind(enc).is_some()
    }

    fn decode_tile(&self, desc: &TensorDescriptor, tile: &Tile<'_>, out: &mut Vec<f32>) -> Result<(), String> {
        check_tile(desc, tile)?;
        decode_elements(desc.encoding.element, tile.payload, (tile.elems.end - tile.elems.start) as usize, out).map_err(|e| format!("{}: {e}", desc.name))?;
        apply_scales(desc, tile, out);
        Ok(())
    }

    fn encode_tile(&self, desc: &TensorDescriptor, elems: &Range<u64>, values: &[f32], ctx: &EncodeCtx, out: &mut TileBuf) -> Result<(), String> {
        rows::encode(desc, elems, values, ctx, out)
    }
}

/// GGUF block formats (plain F32/F16/BF16 GGUF tensors resolve to float encodings instead).
struct GgmlCodec;

impl Codec for GgmlCodec {
    fn name(&self) -> &'static str {
        "ggml"
    }

    fn can_decode(&self, enc: &QuantEncoding) -> bool {
        matches!(enc.element, Element::Ggml(t) if t.block_elems() > 1) && enc.scales.is_empty()
    }

    fn can_encode(&self, enc: &QuantEncoding) -> bool {
        // Probe with one zero block: the quantizers cover only some types.
        matches!(enc.element, Element::Ggml(t) if t.block_elems() > 1 && crate::quant::quantize(t.id(), &vec![0f32; t.block_elems()]).is_ok())
    }

    fn decode_tile(&self, desc: &TensorDescriptor, tile: &Tile<'_>, out: &mut Vec<f32>) -> Result<(), String> {
        check_tile(desc, tile)?;
        let Element::Ggml(ty) = desc.encoding.element else { unreachable!("can_decode") };
        let n = (tile.elems.end - tile.elems.start) as usize;
        crate::gguf::block_expand(ty, tile.payload, 0, n, out).map_err(|e| format!("{}: {e}", desc.name))
    }

    fn encode_tile(&self, desc: &TensorDescriptor, elems: &Range<u64>, values: &[f32], _ctx: &EncodeCtx, out: &mut TileBuf) -> Result<(), String> {
        let Element::Ggml(ty) = desc.encoding.element else { unreachable!("can_encode") };
        let be = ty.block_elems() as u64;
        if !elems.start.is_multiple_of(be) || !(elems.end - elems.start).is_multiple_of(be) {
            return Err(format!("{}: tile {elems:?} is not aligned to the {be}-element {} block", desc.name, ty.name()));
        }
        out.scales.clear();
        out.payload = crate::quant::quantize(ty.id(), values).map_err(|e| format!("{}: {e}", desc.name))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgmlType;
    use crate::packed::testfix::{bf16_bytes, bytes, e4m3_bytes, f32_bytes};

    /// Decode the whole tensor as a single tile.
    fn decode_all(desc: &TensorDescriptor, payload: &[u8], scales: &[&[u8]]) -> Vec<f32> {
        let tile = Tile { elems: 0..desc.numel(), payload, scales: scales.to_vec() };
        let mut out = Vec::new();
        CodecRegistry::reference().decoder(&desc.encoding).unwrap().decode_tile(desc, &tile, &mut out).unwrap();
        out
    }

    #[test]
    fn fp8_tiles_decode_to_code_times_the_covering_bf16_scale() {
        let desc = TensorDescriptor::new("w", vec![4, 6], QuantEncoding::fp8_e4m3_tiled(2));
        let payload = e4m3_bytes(1, 24);
        let grid = [0.5f32, 2.0, 4.0, 8.0, 0.25, 1.0];
        let got = decode_all(&desc, &payload, &[&bf16_bytes(&grid)]);
        for (i, &g) in got.iter().enumerate() {
            let (r, c) = (i / 6, i % 6);
            assert_eq!(g, e4m3fn_to_f32(payload[i]) * grid[(r / 2) * 3 + c / 2], "element {i}");
        }
    }

    /// A tile of the checkpoint float formats decodes to each code's own value however the tile is cut across the CPU
    /// pool: more elements than one work item, an odd length, and every code of the narrow formats.
    #[test]
    fn float_tiles_decode_to_each_codes_value_across_work_items() {
        let n = 3 * DECODE_CHUNK + 17;
        for (enc, width) in [(QuantEncoding::bf16(), 2usize), (QuantEncoding::f32(), 4), (QuantEncoding::f16(), 2)] {
            let desc = TensorDescriptor::new("w", vec![1, n as u64], enc.clone());
            let payload = bytes(9, n * width);
            let got = decode_all(&desc, &payload, &[]);
            let Element::Float(f) = enc.element else { unreachable!() };
            let want: Vec<f32> = payload.chunks_exact(width).map(|c| decode_float_code(f, c.iter().rev().fold(0u32, |a, &b| a << 8 | u32::from(b)))).collect();
            assert_eq!(got.len(), n);
            assert!(got.iter().zip(&want).all(|(g, w)| g.to_bits() == w.to_bits()), "{enc}");
        }
    }

    #[test]
    fn nvfp4_decodes_e2m1_times_block16_e4m3_times_global_scale() {
        let desc = TensorDescriptor::new("w", vec![2, 32], QuantEncoding::nvfp4());
        let payload = bytes(2, 32);
        let block_scales = e4m3_bytes(3, 4);
        let got = decode_all(&desc, &payload, &[&f32_bytes(&[0.5]), &block_scales]);
        let e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (i, &g) in got.iter().enumerate() {
            let code = (payload[i / 2] >> (4 * (i % 2))) & 0xF;
            let mag = e2m1[(code & 7) as usize] * if code & 8 != 0 { -1.0 } else { 1.0 };
            assert_eq!(g, mag * e4m3fn_to_f32(block_scales[i / 16]) * 0.5, "element {i}");
        }
    }

    #[test]
    fn mxfp4_scales_are_powers_of_two() {
        let desc = TensorDescriptor::new("w", vec![1, 64], QuantEncoding::mxfp4());
        let payload = vec![0x22u8; 32]; // every code is 1.0
        let got = decode_all(&desc, &payload, &[&[127, 130]]);
        assert!(got[..32].iter().all(|&v| v == 1.0));
        assert!(got[32..].iter().all(|&v| v == 8.0));
    }

    #[test]
    fn ggml_tiles_equal_the_whole_tensor_decode() {
        let (rows, cols) = (4usize, 256usize);
        let data: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0).collect();
        for ty in [GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K] {
            let raw = crate::quant::quantize(ty.id(), &data).unwrap();
            let want = crate::gguf::dequantize(ty.id(), &raw, data.len()).unwrap();
            let desc = TensorDescriptor::new("w", vec![rows as u64, cols as u64], QuantEncoding::ggml(ty));
            let mut got = Vec::new();
            let mut tile_out = Vec::new();
            for r in 0..rows as u64 {
                let elems = r * cols as u64..(r + 1) * cols as u64;
                let pr = desc.payload_range(&elems);
                let tile = Tile { elems, payload: &raw[pr.start as usize..pr.end as usize], scales: Vec::new() };
                CodecRegistry::reference().decoder(&desc.encoding).unwrap().decode_tile(&desc, &tile, &mut tile_out).unwrap();
                got.extend_from_slice(&tile_out);
            }
            assert_eq!(got, want, "{}", ty.name());
        }
    }

    #[test]
    fn encode_then_decode_stays_close_for_the_formats_a_load_can_target() {
        let (rows, cols) = (3usize, 64usize);
        let data: Vec<f32> = (0..rows * cols).map(|i| ((i * 29 % 97) as f32 - 48.0) / 25.0).collect();
        let reg = CodecRegistry::reference();
        for (enc, tol) in [(QuantEncoding::f32(), 0.0), (QuantEncoding::bf16(), 0.01), (QuantEncoding::int8_blocked(32), 0.01), (QuantEncoding::nf4_blocked(32), 0.3), (QuantEncoding::ggml(GgmlType::Q8_0), 0.02)] {
            let desc = TensorDescriptor::new("w", vec![rows as u64, cols as u64], enc.clone());
            let mut buf = TileBuf::default();
            reg.encoder(&enc).unwrap().encode_tile(&desc, &(0..desc.numel()), &data, &EncodeCtx::default(), &mut buf).unwrap();
            let got = decode_all(&desc, &buf.payload, &buf.scales.iter().map(Vec::as_slice).collect::<Vec<_>>());
            // NF4's widest codebook gap is 0.30 of the block maximum (~1.9 here).
            let worst = got.iter().zip(&data).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
            assert!(worst <= tol, "{enc}: worst abs error {worst}");
        }
    }

    /// Narrowing a tile to bf16 or f16 gives the bits `half` gives for every value class: ties, subnormals, the
    /// rounding carry into the exponent, infinities and NaN, across more elements than one work item.
    #[test]
    fn narrowing_tiles_gives_the_bits_of_the_reference_conversion() {
        let mut vals: Vec<f32> = vec![0.0, -0.0, 1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::MIN_POSITIVE, 1e-40, 3.4e38, f32::from_bits(0x3F80_8000), f32::from_bits(0x3F81_8000), f32::from_bits(0x7F7F_FFFF)];
        vals.extend(bytes(5, 4 * (2 * 65536 + 5)).as_chunks::<4>().0.iter().map(|c| f32::from_bits(u32::from_le_bytes(*c))).filter(|v| !v.is_nan()));
        let reg = CodecRegistry::reference();
        for (enc, reference) in [(QuantEncoding::bf16(), (|v: f32| half::bf16::from_f32(v).to_le_bytes().to_vec()) as fn(f32) -> Vec<u8>), (QuantEncoding::f16(), |v| half::f16::from_f32(v).to_le_bytes().to_vec())] {
            let desc = TensorDescriptor::new("w", vec![1, vals.len() as u64], enc.clone());
            let mut buf = TileBuf::default();
            reg.encoder(&enc).unwrap().encode_tile(&desc, &(0..desc.numel()), &vals, &EncodeCtx::default(), &mut buf).unwrap();
            let want: Vec<u8> = vals.iter().flat_map(|&v| reference(v)).collect();
            assert_eq!(buf.payload, want, "{enc}");
        }
    }

    #[test]
    fn a_misaligned_or_missized_tile_is_an_error_not_a_guess() {
        let desc = TensorDescriptor::new("w", vec![2, 32], QuantEncoding::nvfp4());
        let tile = Tile { elems: 8..40, payload: &[0; 16], scales: vec![&[0; 4], &[0; 8]] };
        let mut out = Vec::new();
        assert!(CodecRegistry::reference().decoder(&desc.encoding).unwrap().decode_tile(&desc, &tile, &mut out).is_err());
    }

    /// Decode of an encode, with per-block bounds: the error of each value is at most `frac` of its
    /// block's largest magnitude (the widest gap of the element grid, relative to its maximum).
    fn round_trip(enc: QuantEncoding, shape: &[u64], block: usize, frac: f32) {
        let desc = TensorDescriptor::new("w", shape.to_vec(), enc.clone());
        desc.validate().unwrap();
        let data: Vec<f32> = (0..desc.numel()).map(|i| ((i * 2654435761 % 1000) as f32 / 500.0 - 1.0) * (1.0 + (i / 97) as f32)).collect();
        let amax = data.iter().fold(0f32, |m, v| m.max(v.abs()));
        let mut buf = TileBuf::default();
        let ctx = EncodeCtx { tensor_absmax: amax };
        let reg = CodecRegistry::reference();
        reg.encoder(&enc).unwrap().encode_tile(&desc, &(0..desc.numel()), &data, &ctx, &mut buf).unwrap();
        let scales: Vec<&[u8]> = buf.scales.iter().map(Vec::as_slice).collect();
        let got = decode_all(&desc, &buf.payload, &scales);
        for (b, (g, w)) in got.chunks(block).zip(data.chunks(block)).enumerate() {
            let bmax = w.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = g.iter().zip(w).map(|(a, c)| (a - c).abs()).fold(0f32, f32::max);
            assert!(worst <= frac * bmax + 1e-6, "{enc}: block {b} error {worst} exceeds {frac} of its maximum {bmax}");
        }
    }

    #[test]
    fn fp8_nvfp4_and_mxfp4_encoders_round_trip_within_their_grid_gaps() {
        round_trip(QuantEncoding::fp8_e4m3_tiled(16), &[32, 48], 48, 0.07);
        round_trip(QuantEncoding::fp8(FloatFormat::E4M3Fn), &[4, 64], 64, 0.07);
        round_trip(QuantEncoding::fp8(FloatFormat::E5M2), &[4, 64], 64, 0.13);
        // E2M1's widest gap is 2 of 6; the E4M3 block scale adds up to ~6% on top.
        round_trip(QuantEncoding::nvfp4(), &[8, 64], 16, 0.36);
        // The power-of-two scale lets the block maximum sit anywhere in [4, 8): gap 2 of 4 at worst.
        round_trip(QuantEncoding::mxfp4(), &[8, 64], 32, 0.5);
    }

    #[test]
    fn values_on_the_grid_encode_exactly() {
        // E2M1 magnitudes at scale 1 (MXFP4 block max 6 -> exponent 0): every grid value survives.
        let desc = TensorDescriptor::new("w", vec![1, 32], QuantEncoding::mxfp4());
        let grid = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let data: Vec<f32> = (0..32).map(|i| grid[i % 8] * if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let mut buf = TileBuf::default();
        CodecRegistry::reference().encoder(&desc.encoding).unwrap().encode_tile(&desc, &(0..32), &data, &EncodeCtx::default(), &mut buf).unwrap();
        let got = decode_all(&desc, &buf.payload, &[&buf.scales[0]]);
        assert_eq!(got.iter().map(|v| v.abs()).collect::<Vec<_>>(), data.iter().map(|v| v.abs()).collect::<Vec<_>>());
        // An E4M3 value and its negative are encoded symmetrically.
        let d2 = TensorDescriptor::new("w", vec![1, 4], QuantEncoding::fp8(FloatFormat::E4M3Fn));
        CodecRegistry::reference().encoder(&d2.encoding).unwrap().encode_tile(&d2, &(0..4), &[448.0, -448.0, 1000.0, 0.0], &EncodeCtx::default(), &mut buf).unwrap();
        assert_eq!(decode_all(&d2, &buf.payload, &[]), [448.0, -448.0, 448.0, 0.0], "values past the format maximum saturate");
    }

    /// Spec: a large tile of tiled-scale FP8 is encoded by groups of row bands on several threads, and its payload and
    /// scales are exactly those of encoding the bands one at a time.
    #[test]
    fn a_large_fp8_tile_encodes_in_parallel_with_the_bytes_of_band_by_band_encoding() {
        let (rows, cols) = (1000u64, 512u64);
        let desc = TensorDescriptor::new("w", vec![rows, cols], QuantEncoding::fp8_e4m3_tiled(128));
        let values: Vec<f32> = (0..rows * cols).map(|i| (((i * 2_654_435_761) % 10_007) as f32 / 10_007.0 - 0.5) * (1.0 + (i / (cols * 128)) as f32)).collect();
        let reg = CodecRegistry::reference();
        let codec = reg.encoder(&desc.encoding).unwrap();
        let mut whole = TileBuf::default();
        codec.encode_tile(&desc, &(0..rows * cols), &values, &EncodeCtx::default(), &mut whole).unwrap();
        let (mut payload, mut scales) = (Vec::new(), Vec::new());
        let band = 128 * cols;
        for start in (0..rows * cols).step_by(band as usize) {
            let end = (start + band).min(rows * cols);
            let mut part = TileBuf::default();
            codec.encode_tile(&desc, &(start..end), &values[start as usize..end as usize], &EncodeCtx::default(), &mut part).unwrap();
            payload.extend_from_slice(&part.payload);
            scales.extend_from_slice(&part.scales[0]);
        }
        assert!(whole.payload == payload && whole.scales[0] == scales);
    }
}
