// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fast inference and fine-tuning over quantized checkpoints for its clients. If your
// team needs expertise in storage-tier weight kernels then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Shared test kit: one device `Weight` per storage tier built from random valid bytes, with the f32 values its
//! stored bytes decode to (an independent host decode, the oracle of every tier test).

#![allow(dead_code)]

use checkpoint::packed::{CodecRegistry, Element, FloatFormat, QuantEncoding, ScaleGranularity, ScaleLevel, TensorDescriptor, TensorSink, Tile};
use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::fp8::{dequantize_weight_fp8, quantize_weight_fp8, Fp8Format};
use model::half::f32_to_bf16;
use model::int8::quantize_weight;
use model::lut4::{dequantize_weight_lut4, quantize_weight_lut4, NF4_LUT};
use model::ops::{DeviceWeightSink, KqScale, Ops, Weight, FP8_BLOCK};

pub fn upload_u32(g: &Gpu, data: &[u32]) -> DeviceBuffer {
    let b = g.storage(data.len() as u64);
    g.write(&b, data);
    b
}

/// A tier under test: its device weight and the f32 values its stored bytes decode to.
pub struct Case {
    pub label: &'static str,
    pub weight: Weight,
    pub decoded: Vec<f32>,
    /// The device reads this tier as bf16 on the tensor cores (the weight is decoded to a bf16 tile and the
    /// activation rounded to bf16), so its products are held to the looser bf16 tolerance; the f32 tier
    /// runs TF32 products and is held to the tighter one.
    pub bf16_path: bool,
}

pub fn bf16_round(v: f32) -> f32 {
    f32::from_bits((f32_to_bf16(v) as u32) << 16)
}

pub fn build_case(ops: &Ops, label: &'static str, n: usize, k: usize, rng: &mut Lcg) -> Case {
    let g = ops.gpu();
    let raw = rng.vec_scaled(n * k, 0.5);
    match label {
        "f32" => Case { label, weight: Weight::upload(ops, &raw, n, k, Dtype::F32), decoded: raw, bf16_path: false },
        "bf16" => {
            let w = Weight::upload(ops, &raw, n, k, Dtype::BF16);
            assert!(matches!(w, Weight::BF16 { .. }), "bf16 must not be promoted away");
            let decoded = raw.iter().map(|&v| bf16_round(v)).collect();
            Case { label, weight: w, decoded, bf16_path: true }
        }
        "f16" => {
            let w = Weight::upload(ops, &raw, n, k, Dtype::F16);
            assert!(matches!(w, Weight::F16 { .. }), "f16 must not be promoted away");
            let decoded = raw.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
            Case { label, weight: w, decoded, bf16_path: true }
        }
        "q4" => {
            let (packed, scales) = model::int4::quantize_weight_q4(&raw, n, k);
            let decoded = model::int4::dequantize_weight_q4(&packed, &scales, n, k);
            let w = Weight::upload(ops, &raw, n, k, Dtype::Q4);
            assert!(matches!(w, Weight::Q4 { .. }), "q4 must not be promoted away");
            Case { label, weight: w, decoded, bf16_path: true }
        }
        "nf4" => {
            let (packed, scales) = quantize_weight_lut4(&raw, n, k, &NF4_LUT);
            let decoded = dequantize_weight_lut4(&packed, &scales, n, k, &NF4_LUT);
            let w = Weight::upload(ops, &raw, n, k, Dtype::NF4);
            assert!(matches!(w, Weight::NF4 { .. }), "nf4 must not be promoted away");
            Case { label, weight: w, decoded, bf16_path: true }
        }
        "i8" => {
            let (packed, scales) = quantize_weight(&raw, n, k);
            let decoded = model::int8::dequantize_weight(&packed, &scales, n, k);
            let w = Weight::I8 { w: upload_u32(g, &packed), s: g.storage_init("s", &scales), n: n as u32, k: k as u32 };
            Case { label, weight: w, decoded, bf16_path: true }
        }
        "fp8_e4m3" | "fp8_e5m2" => {
            let (dt, fmt) = if label == "fp8_e4m3" { (Dtype::F8E4M3, Fp8Format::E4M3) } else { (Dtype::F8E5M2, Fp8Format::E5M2) };
            let w = Weight::upload(ops, &raw, n, k, dt);
            assert!(matches!(w, Weight::FP8 { .. }), "fp8 must be preserved as FP8 storage");
            let (packed, scales) = quantize_weight_fp8(&raw, n, k, fmt, FP8_BLOCK as usize);
            let decoded = dequantize_weight_fp8(&packed, &scales, n, k, fmt, FP8_BLOCK as usize);
            Case { label, weight: w, decoded, bf16_path: true }
        }
        "q4k" | "q5k" => {
            let bits = if label == "q4k" { 4u32 } else { 8 };
            let (groups, spb) = (k / 32, k / 256);
            let codes: Vec<u32> = (0..n * k).map(|_| rng.next_u32() % (1 << if bits == 4 { 4 } else { 5 })).collect();
            let per_word = (32 / bits) as usize;
            let wq: Vec<u32> = codes.chunks(per_word).map(|c| c.iter().enumerate().fold(0u32, |w, (b, &v)| w | (v << (bits * b as u32)))).collect();
            let sc: Vec<u32> = (0..n * groups).map(|_| 1 + rng.next_u32() % 62).collect();
            let mn: Vec<u32> = (0..n * groups).map(|_| rng.next_u32() % 63).collect();
            let f16 = |v: f32| half::f16::from_f32(v);
            let d: Vec<half::f16> = (0..n * spb).map(|_| f16(0.001 + rng.unit() * 0.01)).collect();
            let dmin: Vec<half::f16> = (0..n * spb).map(|_| f16(0.002 + rng.unit() * 0.02)).collect();
            let wsm_words = groups.div_ceil(2);
            let mut wsm = vec![0u32; n * wsm_words];
            for r in 0..n {
                for gi in 0..groups {
                    let shift = if gi % 2 == 0 { 0 } else { 16 };
                    wsm[r * wsm_words + gi / 2] |= (sc[r * groups + gi] << shift) | (mn[r * groups + gi] << (shift + 8));
                }
            }
            let wd: Vec<u32> = d.iter().zip(&dmin).map(|(a, b)| a.to_bits() as u32 | ((b.to_bits() as u32) << 16)).collect();
            let decoded = (0..n * k)
                .map(|i| {
                    let (r, c) = (i / k, i % k);
                    let (gi, sb) = (c / 32, c / 256);
                    let ds = d[r * spb + sb].to_f32() * sc[r * groups + gi] as f32;
                    let dm = dmin[r * spb + sb].to_f32() * mn[r * groups + gi] as f32;
                    ds * codes[i] as f32 - dm
                })
                .collect();
            let sz = KqScale::Packed { wsm: upload_u32(g, &wsm), wd: upload_u32(g, &wd) };
            let weight = Weight::KQuant { w: upload_u32(g, &wq), sz, n: n as u32, k: k as u32, group: 32, bits, affine: true };
            Case { label, weight, decoded, bf16_path: true }
        }
        "nvfp4" | "mxfp4" | "mxfp8_e4m3" | "mxfp8_e5m2" => {
            // The microscaling family, loaded the way a checkpoint is: random valid bytes through the tiled
            // device sink; the oracle is the checkpoint crate's reference codec over the same bytes.
            let level = |granularity, format| ScaleLevel { granularity, format, suffix: "_scale".into() };
            let enc = match label {
                "nvfp4" => QuantEncoding::nvfp4(),
                "mxfp4" => QuantEncoding::mxfp4(),
                _ => QuantEncoding {
                    element: Element::Float(if label == "mxfp8_e4m3" { FloatFormat::E4M3Fn } else { FloatFormat::E5M2 }),
                    scales: vec![level(ScaleGranularity::Block(32), FloatFormat::E8M0)],
                },
            };
            let desc = TensorDescriptor::new("w", vec![n as u64, k as u64], enc.clone());
            desc.validate().unwrap();
            let fp8_elements = label.starts_with("mxfp8");
            // FP8 element bytes stay below 0x40 (no NaN/inf, magnitudes near 1); scales sit around 2^-4..2^-7.
            let payload: Vec<u8> = (0..desc.payload_bytes()).map(|_| if fp8_elements { (rng.next_u32() % 0x40) as u8 | (rng.next_u32() as u8 & 0x80) } else { rng.next_u32() as u8 }).collect();
            let scales: Vec<Vec<u8>> = enc
                .scales
                .iter()
                .enumerate()
                .map(|(l, lvl)| match lvl.format {
                    FloatFormat::F32 => 1.0f32.to_le_bytes().to_vec(),
                    FloatFormat::E8M0 => (0..desc.scale_bytes(l)).map(|_| 120 + (rng.next_u32() % 4) as u8).collect(),
                    _ => (0..desc.scale_bytes(l)).map(|_| 0x18 + (rng.next_u32() % 8) as u8).collect(),
                })
                .collect();
            let whole = Tile { elems: 0..desc.numel(), payload: &payload, scales: scales.iter().map(Vec::as_slice).collect() };
            let mut decoded = Vec::new();
            CodecRegistry::reference().decoder(&enc).unwrap().decode_tile(&desc, &whole, &mut decoded).unwrap();
            let mut sink = DeviceWeightSink::new(ops);
            sink.begin(&desc).unwrap();
            sink.write_tile(&desc, &whole).unwrap();
            sink.finish(&desc).unwrap();
            let weight = sink.take_weight().unwrap();
            assert!(matches!(weight, Weight::Micro { .. }), "{label} must be preserved as the microscaling tier");
            Case { label, weight, decoded, bf16_path: true }
        }
        "q6k" => {
            let codes: Vec<i32> = (0..n * k).map(|_| (rng.next_u32() % 64) as i32 - 32).collect();
            let wq: Vec<u32> = codes.chunks(4).map(|c| c.iter().enumerate().fold(0u32, |w, (b, &v)| w | (((v as u8) as u32) << (8 * b)))).collect();
            let scales: Vec<f32> = (0..n * k / 16).map(|_| 0.001 + rng.unit() * 0.02).collect();
            let decoded = (0..n * k).map(|i| codes[i] as f32 * scales[i / 16]).collect();
            let weight = Weight::KQuant {
                w: upload_u32(g, &wq),
                sz: KqScale::PlainF32(g.storage_init("s", &scales)),
                n: n as u32,
                k: k as u32,
                group: 16,
                bits: 8,
                affine: false,
            };
            Case { label, weight, decoded, bf16_path: true }
        }
        other => panic!("unknown case {other}"),
    }
}
