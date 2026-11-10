// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device decoder against the host decoder: teacher-forced logits over a
//! batch of memories of different lengths, and greedy generation against the
//! recorded reference tokens.

use data::rng::Lcg;
use mirage::decoder::generate::{absent_references, GenerateParams, Stop};
use mirage::decoder::{Decoder, Positions};
use mirage::device::decoder::DeviceDecoder;
use mirage::device::reader::MemoryBatch;
use mirage::device::{Device, WeightForm};
use mirage::vision::weights::Weights;

use crate::common::{released_dir, tiny_checkpoint};
use crate::vision::{assert_stage, compare};

/// The device with the Hadamard rotations resident in each form; both must reproduce the host.
fn devices(test: &str) -> Vec<Device> {
    crate::common::cap_threads();
    let mut out = Vec::new();
    for form in [WeightForm::Structured, WeightForm::Folded] {
        match Device::new_with(gpu_core::Gpu::new(&Device::kernels()), form) {
            Ok(d) => {
                eprintln!("{test}: device kind {}, {form:?} rotations", d.gpu().kind());
                out.push(d);
            }
            Err(e) => {
                brain_testutil::skip_unavailable(&format!("{test}: no device: {e}"));
                return Vec::new();
            }
        }
    }
    out
}

/// Memories of different lengths padded to a common stride.
fn memory_batch(dev: &Device, memories: &[&[f32]], d: usize) -> MemoryBatch {
    let stride = memories.iter().map(|m| m.len() / d).max().unwrap();
    let mut rows = vec![0f32; memories.len() * stride * d];
    let mut mask = vec![0f32; memories.len() * stride];
    for (i, m) in memories.iter().enumerate() {
        rows[i * stride * d..i * stride * d + m.len()].copy_from_slice(m);
        mask[i * stride..i * stride + m.len() / d].fill(1.0);
    }
    MemoryBatch { rows: dev.buf(&rows), mask: dev.buf(&mask), stride, batch: memories.len() }
}

/// Teacher-force `tokens[i]` (equal lengths) through the device and compare each step's logits with the host.
fn check_teacher_forcing(dev: &Device, dec: &Decoder, d: usize, memories: &[&[f32]], tokens: &[Vec<u32>], label: &str) {
    let dd = DeviceDecoder::upload(dev, dec);
    let mem = memory_batch(dev, memories, d);
    let cross = dd.cross(dev, &mem);
    let steps = tokens[0].len();
    let mut st = dd.new_state(dev, tokens.len(), steps + 1);
    st.set_tokens(dev, tokens);
    let vocab = dec.vocabulary();
    let mut got: Vec<Vec<f32>> = vec![Vec::new(); tokens.len()];
    for _ in 0..steps {
        let logits = dd.step(dev, &mut st, &cross, true, None).unwrap();
        let all = dev.read_at(&logits, 0, tokens.len() * vocab);
        for (i, g) in got.iter_mut().enumerate() {
            g.extend_from_slice(&all[i * vocab..(i + 1) * vocab]);
        }
    }
    for (i, seq) in tokens.iter().enumerate() {
        let cross_host = dec.cross_memory(memories[i], memories[i].len() / d);
        let want = dec.forward(&mut dec.new_state(), &cross_host, seq, Positions::All);
        assert_stage(&format!("{label} seq{i} logits"), &got[i], &want);
        let agree = got[i].chunks_exact(vocab).zip(want.chunks_exact(vocab)).filter(|(a, b)| argmax(a) == argmax(b)).count();
        eprintln!("{label} seq{i}: argmax agrees on {agree} of {steps}");
        assert_eq!(agree, steps, "{label} seq{i}: greedy choices differ");
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0
}

#[test]
fn the_tiny_decoder_matches_the_host_in_a_batch_of_unequal_memories() {
    let _g = gpu_core::testgpu::exclusive();
    let ck = tiny_checkpoint();
    let cfg = ck.config.clone();
    let dec = Decoder::new(&cfg, &Weights::new(&ck));
    let mut rng = Lcg::new(5); // seed: filler compared between two paths
    let m0: Vec<f32> = (0..7 * cfg.dim).map(|_| rng.signed()).collect();
    let m1: Vec<f32> = (0..4 * cfg.dim).map(|_| rng.signed()).collect();
    let tokens: Vec<Vec<u32>> = (0..2).map(|_| (0..12).map(|_| (rng.next_u64() % cfg.embedding_rows() as u64) as u32).collect()).collect();
    for dev in devices("tiny decoder") {
        check_teacher_forcing(&dev, &dec, cfg.dim, &[&m0, &m1], &tokens, "tiny");
    }
}

#[test]
fn the_released_decoder_matches_the_host_and_generates_the_reference_tokens() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("released decoder: the released checkpoint is absent");
        return;
    };
    let (Some(cats), Some(text)) = (crate::decoder::golden("device decoder", "cats"), crate::decoder::golden("device decoder", "text")) else { return };
    let model = mirage::model::Mirage::open(&dir).unwrap();
    let dec = model.decoder();
    for dev in devices("released decoder") {
        let d = 512;
        let gold = [&cats, &text];
        let steps = 40;
        let tokens: Vec<Vec<u32>> = gold.iter().map(|g| g["tokens"][..steps].iter().map(|&t| t as u32).collect()).collect();
        let mems: Vec<&[f32]> = gold.iter().map(|g| &g["memory"][..]).collect();
        check_teacher_forcing(&dev, dec, d, &mems, &tokens, "released");

        // Greedy generation, both images in one batch, against the recorded tokens.
        let dd = DeviceDecoder::upload(&dev, dec);
        let mem = memory_batch(&dev, &mems, d);
        let cfg = mirage::config::MirageConfig::from_json(crate::common::RELEASED_ARCHITECTURE_JSON).unwrap();
        let all: Vec<Vec<u32>> = gold.iter().map(|g| g["tokens"].iter().map(|&t| t as u32).collect()).collect();
        let banned: Vec<Vec<u32>> = gold.iter().map(|g| absent_references(&cfg, g["peaks_obj"].len() / 7, g["peaks_txt"].len() / 7)).collect();
        let params: Vec<GenerateParams> =
            (0..2).map(|i| GenerateParams { prompt: &all[i][..2], banned: &banned[i], max_tokens: 600, max_rank: 5, object_slots: 64, guard: true }).collect();
        let started = std::time::Instant::now();
        let out = dd.generate_batch(&dev, &mem, model.tokenizer(), &params);
        eprintln!("device generation of 2 sequences: {:.1}s", started.elapsed().as_secs_f64());
        for (i, g) in out.iter().enumerate() {
            assert_eq!(g.stop, Stop::Eos, "seq{i}");
            let diff = g.tokens.iter().zip(&all[i][2..]).position(|(a, b)| a != b);
            eprintln!("seq{i}: {} tokens, first difference {:?}", g.tokens.len(), diff);
            assert_eq!(&g.tokens[..], &all[i][2..], "seq{i}: greedy tokens differ from the reference");
        }
    }
    let _ = compare;
}
