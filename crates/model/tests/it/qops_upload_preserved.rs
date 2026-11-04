// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded loading of quantized
// checkpoints straight into device memory for its clients. If your team needs
// expertise in keeping 4-bit and 8-bit floating-point weights packed from disk
// to the GPU, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `upload_preserved`: a checkpoint tensor stored as NVFP4, MXFP4 or MXFP8 (raw bytes plus scale sidecars in a
//! safetensors file) reaches the device in the microscaling tier of its stored encoding, and decodes to exactly
//! what the checkpoint crate's reference codec decodes the same bytes to. The encoding is declared by
//! descriptor, as a model's loader does for a layout the dtype alone does not name.

use checkpoint::packed::{BoundedWorkspace, CodecRegistry, Element, FloatFormat, MemoryBudget, PackedReader, PackedSource, QuantEncoding, ScaleGranularity, ScaleLevel, TensorDescriptor, Tile};
use data::rng::Lcg;
use gpu_core::Gpu;
use model::ops::{training_kernel_list, upload_preserved, Ops, Weight};

const N: usize = 24;
const K: usize = 256;

fn write_safetensors(path: &std::path::Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut off = 0;
    for (name, dtype, shape, bytes) in tensors {
        header.insert((*name).into(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}));
        off += bytes.len();
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for (.., bytes) in tensors {
        out.extend(bytes);
    }
    std::fs::write(path, out).unwrap();
}

#[test]
fn microscaling_checkpoint_tensors_upload_in_their_stored_tier() {
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    let staging = BoundedWorkspace::new(MemoryBudget::new(8 << 20, 0));
    let mut rng = Lcg::random();
    let mxfp8 = QuantEncoding {
        element: Element::Float(FloatFormat::E4M3Fn),
        scales: vec![ScaleLevel { granularity: ScaleGranularity::Block(32), format: FloatFormat::E8M0, suffix: "_scale".into() }],
    };
    let dir = std::env::temp_dir().join(format!("qops_upload_preserved_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (label, enc) in [("nvfp4", QuantEncoding::nvfp4()), ("mxfp4", QuantEncoding::mxfp4()), ("mxfp8", mxfp8)] {
        let desc = TensorDescriptor::new("w", vec![N as u64, K as u64], enc.clone());
        // Valid bytes: FP8 elements below 0x40 (no NaN), E4M3 scales around 2^-4, E8M0 scales around 2^-6.
        let fp8 = matches!(enc.element, Element::Float(FloatFormat::E4M3Fn));
        let payload: Vec<u8> = (0..desc.payload_bytes()).map(|_| if fp8 { (rng.next_u32() % 0x40) as u8 } else { rng.next_u32() as u8 }).collect();
        let scales: Vec<Vec<u8>> = enc
            .scales
            .iter()
            .enumerate()
            .map(|(l, lvl)| match lvl.format {
                FloatFormat::F32 => 1.25f32.to_le_bytes().to_vec(),
                FloatFormat::E8M0 => (0..desc.scale_bytes(l)).map(|_| 120 + (rng.next_u32() % 4) as u8).collect(),
                _ => (0..desc.scale_bytes(l)).map(|_| 0x18 + (rng.next_u32() % 8) as u8).collect(),
            })
            .collect();
        let mut tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = vec![("w", "U8", vec![payload.len()], payload.clone())];
        let names: Vec<String> = enc.scales.iter().map(|l| format!("w{}", l.suffix)).collect();
        for (i, lvl) in enc.scales.iter().enumerate() {
            let (dtype, shape) = if lvl.format == FloatFormat::F32 { ("F32", vec![1]) } else { ("U8", vec![scales[i].len()]) };
            tensors.push((names[i].as_str(), dtype, shape, scales[i].clone()));
        }
        let file = dir.join(format!("{label}.safetensors"));
        write_safetensors(&file, &tensors);
        let reader = PackedReader::open(&file).unwrap().with_descriptor(desc.clone());
        let src = PackedSource::identity(Box::new(reader), CodecRegistry::reference()).unwrap();

        let weight = upload_preserved(&ops, &src, "w", N, K, &staging).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(matches!(weight, Weight::Micro { .. }), "{label}: stored microscaling bytes stay microscaling on the device");

        let whole = Tile { elems: 0..desc.numel(), payload: &payload, scales: scales.iter().map(Vec::as_slice).collect() };
        let mut want = Vec::new();
        CodecRegistry::reference().decoder(&enc).unwrap().decode_tile(&desc, &whole, &mut want).unwrap();
        let g = ops.gpu();
        let out = g.storage((N * K) as u64);
        let mut steps = Vec::new();
        ops.decode_tile(&mut steps, &weight, 0, N as u32, &out);
        g.submit(&[], &steps);
        let got = g.read(&out, N * K);
        assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{label}: the uploaded weight differs from the reference decode");
        assert!(upload_preserved(&ops, &src, "w", N + 1, K, &staging).is_err(), "{label}: a wrong shape is refused");
    }
    std::fs::remove_dir_all(&dir).ok();
}
