// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded loading of quantized
// checkpoints straight into device memory for its clients. If your team needs
// expertise in uploading packed weights tile by tile without ever expanding a
// tensor to full precision, you can procure our services by sending an email
// to info@swedishembedded.com.

//! Spec of `DeviceWeightSink`: a tiled load in a quantized destination encoding becomes the resident
//! `Weight` tier of that encoding, byte for byte, whatever the tile boundaries are, and an encoding with
//! no tier is refused with a message (never widened). The oracle is the host quantizer of each tier and
//! the exact device tile decode of the resulting weight.

use checkpoint::packed::{CodecRegistry, Element, FloatFormat, QuantEncoding, ScaleGranularity, ScaleLevel, TensorDescriptor, TensorSink, Tile};
use data::rng::Lcg;
use gpu_core::Gpu;
use model::fp8::{dequantize_weight_fp8, quantize_weight_fp8, Fp8Format};
use model::half::pack_bf16;
use model::int8::{dequantize_weight, quantize_weight};
use model::lut4::{dequantize_weight_lut4, quantize_weight_lut4, NF4_LUT};
use model::ops::{training_kernel_list, DeviceWeightSink, Ops, Weight};

const N: usize = 40;
const K: usize = 256;

fn words_to_bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Drive `sink` with `payload`/`scale` cut into tiles of `tile_elems` elements (a multiple of the quantum).
fn load(sink: &mut DeviceWeightSink, desc: &TensorDescriptor, payload: &[u8], scale: Option<&[u8]>, tile_elems: u64) -> Result<(), String> {
    sink.begin(desc)?;
    let total = desc.numel();
    let mut start = 0;
    while start < total {
        let end = (start + tile_elems).min(total);
        let elems = start..end;
        let pr = desc.payload_range(&elems);
        let scales: Vec<&[u8]> = match scale {
            Some(s) => {
                let w = desc.encoding.scales[0].format.bits() as u64 / 8;
                let r = desc.scale_range(0, &elems);
                vec![&s[(r.start * w) as usize..(r.end * w) as usize]]
            }
            None => Vec::new(),
        };
        sink.write_tile(desc, &Tile { elems, payload: &payload[pr.start as usize..pr.end as usize], scales })?;
        start = end;
    }
    sink.finish(desc)
}

fn decoded(ops: &Ops, w: &Weight) -> Vec<f32> {
    let g = ops.gpu();
    let out = g.storage((w.n() * w.k()) as u64);
    let mut steps = Vec::new();
    ops.decode_tile(&mut steps, w, 0, w.n(), &out);
    g.submit(&[], &steps);
    g.read(&out, (w.n() * w.k()) as usize)
}

#[test]
fn a_tiled_load_becomes_the_resident_tier_of_its_encoding_whatever_the_tiling() {
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    let raw = Lcg::random().vec_scaled(N * K, 0.5);
    let shape = vec![N as u64, K as u64];
    let (i8p, i8s) = quantize_weight(&raw, N, K);
    let (n4p, n4s) = quantize_weight_lut4(&raw, N, K, &NF4_LUT);
    let (f8p, f8s) = quantize_weight_fp8(&raw, N, K, Fp8Format::E4M3, 128);
    // The checkpoint convention stores the FP8 scales as bf16: widening them back is exact.
    let f8s_bf16: Vec<f32> = f8s.iter().map(|&s| f32::from_bits(s.to_bits() & 0xffff_0000)).collect();
    let bf16_words = pack_bf16(&raw);
    let bf16_host: Vec<f32> = raw.iter().map(|&v| f32::from_bits((model::half::f32_to_bf16(v) as u32) << 16)).collect();
    let scale_bf16_bytes: Vec<u8> = f8s_bf16.iter().flat_map(|s| ((s.to_bits() >> 16) as u16).to_le_bytes()).collect();

    type Case<'a> = (&'a str, QuantEncoding, Vec<u8>, Option<Vec<u8>>, Vec<f32>);
    let cases: Vec<Case> = vec![
        ("int8", QuantEncoding::int8_blocked(32), words_to_bytes(&i8p), Some(f32_bytes(&i8s)), dequantize_weight(&i8p, &i8s, N, K)),
        ("nf4", QuantEncoding::nf4_blocked(32), words_to_bytes(&n4p), Some(f32_bytes(&n4s)), dequantize_weight_lut4(&n4p, &n4s, N, K, &NF4_LUT)),
        (
            "fp8 e4m3, bf16 tile scales",
            QuantEncoding::fp8_e4m3_tiled(128),
            words_to_bytes(&f8p),
            Some(scale_bf16_bytes),
            dequantize_weight_fp8(&f8p, &f8s_bf16, N, K, Fp8Format::E4M3, 128),
        ),
        ("bf16", QuantEncoding::bf16(), words_to_bytes(&bf16_words), None, bf16_host),
    ];
    for (label, enc, payload, scale, want) in cases {
        let desc = TensorDescriptor::new("w", shape.clone(), enc);
        desc.validate().unwrap();
        let quantum = desc.tile_quantum();
        // One tile, the smallest legal tiles, and an uneven cut: all give the same resident weight.
        for tile in [desc.numel(), quantum, quantum * 5 + quantum * 3] {
            let mut sink = DeviceWeightSink::new(&ops);
            load(&mut sink, &desc, &payload, scale.as_deref(), tile).unwrap_or_else(|e| panic!("{label} tile {tile}: {e}"));
            let w = sink.take_weight().expect("a finished load yields a weight");
            assert_eq!((w.n(), w.k()), (N as u32, K as u32), "{label}");
            let got = decoded(&ops, &w);
            assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{label} (tile {tile}): the resident weight differs from the host decode");
        }
    }
}

#[test]
fn an_encoding_without_a_tier_is_refused_not_widened() {
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    let desc = TensorDescriptor::new("moe.expert.weight", vec![N as u64, K as u64], QuantEncoding::ggml(checkpoint::gguf::GgmlType::Q4K));
    let mut sink = DeviceWeightSink::new(&ops);
    let err = sink.begin(&desc).expect_err("a GGUF block format has no resident tier yet");
    assert!(err.contains("moe.expert.weight") && err.contains("no resident Weight tier"), "{err}");
    let rank3 = TensorDescriptor::new("t", vec![2, 3, 32], QuantEncoding::bf16());
    assert!(sink.begin(&rank3).is_err());
}

/// The microscaling family (NVFP4, MXFP4, MXFP8 E4M3/E5M2) loads as `Weight::Micro` with its scale BYTES kept as
/// bytes, and decodes - on the portable kernel and, where the device has the tensor-core path, to bf16 - to exactly
/// what the checkpoint crate's reference codec decodes the same bytes to, whatever the tiling.
#[test]
fn microscaling_tiers_preserve_their_bytes_and_decode_like_the_reference_codec() {
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    let mut rng = Lcg::random();
    let level = |granularity, format, suffix: &str| ScaleLevel { granularity, format, suffix: suffix.into() };
    let nvfp4 = QuantEncoding::nvfp4();
    let mxfp4 = QuantEncoding::mxfp4();
    let mxfp8 = |f| QuantEncoding { element: Element::Float(f), scales: vec![level(ScaleGranularity::Block(32), FloatFormat::E8M0, "_scale")] };
    let cases = [("nvfp4", nvfp4), ("mxfp4", mxfp4), ("mxfp8 e4m3", mxfp8(FloatFormat::E4M3Fn)), ("mxfp8 e5m2", mxfp8(FloatFormat::E5M2))];
    for (label, enc) in cases {
        let desc = TensorDescriptor::new("w", vec![N as u64, K as u64], enc.clone());
        desc.validate().unwrap();
        // Valid codes only: no NaN/inf elements, scales around 1.
        let element_is_float8 = matches!(enc.element, Element::Float(FloatFormat::E4M3Fn | FloatFormat::E5M2));
        let payload: Vec<u8> = (0..desc.payload_bytes())
            .map(|_| loop {
                let b = rng.next_u32() as u8;
                let bad = match enc.element {
                    Element::Float(FloatFormat::E4M3Fn) => b & 0x7f == 0x7f,
                    Element::Float(FloatFormat::E5M2) => (b >> 2) & 31 == 31,
                    _ => false,
                };
                if !(element_is_float8 && bad) {
                    break b;
                }
            })
            .collect();
        let scales: Vec<Vec<u8>> = (0..enc.scales.len())
            .map(|l| {
                let lvl = &enc.scales[l];
                if lvl.format == FloatFormat::F32 {
                    return (1.5f32).to_le_bytes().to_vec(); // NVFP4's per-tensor scale
                }
                (0..desc.scale_bytes(l)).map(|_| if lvl.format == FloatFormat::E8M0 { 120 + (rng.next_u32() % 16) as u8 } else { 0x30 + (rng.next_u32() % 0x20) as u8 }).collect()
            })
            .collect();

        // The oracle: the reference codec over the whole tensor.
        let codecs = CodecRegistry::reference();
        let whole = Tile { elems: 0..desc.numel(), payload: &payload, scales: scales.iter().map(Vec::as_slice).collect() };
        let mut want = Vec::new();
        codecs.decoder(&enc).unwrap().decode_tile(&desc, &whole, &mut want).unwrap();

        let quantum = desc.tile_quantum();
        for tile_elems in [desc.numel(), quantum, quantum * 3] {
            let mut sink = DeviceWeightSink::new(&ops);
            sink.begin(&desc).unwrap();
            let mut start = 0;
            while start < desc.numel() {
                let end = (start + tile_elems).min(desc.numel());
                let elems = start..end;
                let pr = desc.payload_range(&elems);
                let tile_scales: Vec<&[u8]> = (0..enc.scales.len())
                    .map(|l| {
                        let w = enc.scales[l].format.bits() as u64 / 8;
                        let r = desc.scale_range(l, &elems);
                        &scales[l][(r.start * w) as usize..(r.end * w) as usize]
                    })
                    .collect();
                sink.write_tile(&desc, &Tile { elems: elems.clone(), payload: &payload[pr.start as usize..pr.end as usize], scales: tile_scales }).unwrap();
                start = end;
            }
            sink.finish(&desc).unwrap();
            let w = sink.take_weight().unwrap();
            assert!(matches!(w, Weight::Micro { .. }), "{label}: preserved as the microscaling tier");
            let got = decoded(&ops, &w);
            assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{label} (tile {tile_elems}): the portable decode differs from the reference codec");
            let g = ops.gpu();
            let out16 = g.storage((N * K / 2) as u64);
            let mut steps = Vec::new();
            if ops.decode_tile_bf16(&mut steps, &w, 0, N as u32, &out16) {
                g.submit(&[], &steps);
                let words = g.read(&out16, N * K / 2);
                for (i, v) in want.iter().enumerate() {
                    let half = (words[i / 2].to_bits() >> ((i % 2) * 16)) & 0xffff;
                    assert_eq!(half, model::half::f32_to_bf16(*v) as u32, "{label}: bf16 element {i}");
                }
            }
        }
    }
}
