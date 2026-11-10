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

/// The decoder step floor: one recorded step replayed at batch 1, 8, 32 and 64 on the real cats memory, timed as
/// the minimum over replays (`poll_wait` brackets every timed region) and priced per kernel kind with
/// `Gpu::profile_tape`. A probe, not a gate: run it on an idle card and note the card load.
#[test]
#[ignore = "profiling entry point: prints the step time and the per-kernel table"]
fn decoder_step_floor_probe() {
    step_floor(mirage::model::Precision::Fp32);
}

/// [`decoder_step_floor_probe`] on the packed decoder.
#[test]
#[ignore = "profiling entry point: prints the step time and the per-kernel table"]
fn decoder_step_floor_probe_cq() {
    step_floor(mirage::model::Precision::Cq);
}

/// [`decoder_step_floor_probe`] on the packed decoder with int8 activations.
#[test]
#[ignore = "profiling entry point: prints the step time and the per-kernel table"]
fn decoder_step_floor_probe_cq_a8() {
    step_floor(mirage::model::Precision::CqA8);
}

fn step_floor(precision: mirage::model::Precision) {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else { return };
    let Some(cats) = crate::decoder::golden("step floor", "cats") else { return };
    let loaded = mirage::model::load_precision(&dir, precision).expect("loads");
    let model = mirage::model::Mirage::new(&loaded.checkpoint, &loaded.tokenizer, loaded.charset).unwrap();
    for mut dev in devices("step floor").into_iter().take(1) {
        dev.set_cq_activations(precision.activations());
        let dd = DeviceDecoder::upload(&dev, model.decoder());
        let memory: Vec<f32> = cats["memory"].iter().map(|&v| v as f32).collect();
        let tokens: Vec<u32> = cats["tokens"].iter().map(|&t| t as u32).collect();
        let (rounds, inner) = (40, 10);
        for batch in [1usize, 8, 32, 64] {
            let mems: Vec<&[f32]> = (0..batch).map(|_| &memory[..]).collect();
            let mem = memory_batch(&dev, &mems, 512);
            let cross = dd.cross(&dev, &mem);
            let mut st = dd.new_state(&dev, batch, 700);
            st.set_tokens(&dev, &vec![tokens.clone(); batch]);
            st.first = 2;
            let ban = dev.buf(&vec![0f32; batch * dd.vocabulary()]);
            let picked = dev.alloc(batch);
            let tape = dd.record_step(&dev, &st, &cross, true, Some((&ban, &picked)));
            let g = dev.gpu();
            let mut best = f64::MAX;
            for round in 0..rounds {
                st.seek(&dev, 150);
                let t = std::time::Instant::now();
                for _ in 0..inner {
                    g.replay_tape(&tape);
                }
                g.poll_wait();
                if round >= 5 {
                    best = best.min(t.elapsed().as_secs_f64() / inner as f64);
                }
            }
            eprintln!("STEP batch {batch}: {:.3} ms (min over replays, position 150)", best * 1e3);
            st.seek(&dev, 150);
            if let Some(rows) = g.profile_tape(&tape, 20) {
                let total: f64 = rows.iter().map(|r| r.1).sum();
                eprintln!("PROFILE batch {batch}: {total:.3} ms in {} dispatches", rows.iter().map(|r| r.2).sum::<u64>());
                for (k, ms, n) in rows.iter().take(14) {
                    eprintln!("  {k:<28} {ms:8.4} ms {:5.1}%  x{n}", ms / total * 100.0);
                }
            }
        }
    }
}

/// Product times at the decoder's shapes: `rows` activation rows against dense f32 weights of the step's shapes,
/// eight products per recorded tape (the library declined, as inside a replayed step), minimum over replays.
#[test]
#[ignore = "profiling entry point: prints microseconds per product"]
fn decoder_gemm_shapes_probe() {
    const REPS: usize = 4;
    let rounds = 30;
    let _g = gpu_core::testgpu::exclusive();
    for dev in devices("gemm shapes").into_iter().take(1) {
        let g = dev.gpu();
        let shapes = [("phi", 24usize, 2048usize), ("qkvg", 1120, 512), ("out", 512, 512), ("cq", 384, 512), ("engram", 512, 512), ("logits", 8299, 512)];
        for rows in [16usize, 32, 33, 64, 128] {
            let mut line = format!("GEMM rows {rows:3}:");
            for (name, n, k) in shapes {
                let w = model::ops::Weight::dense_f32(dev.buf(&vec![0.01f32; n * k]), n as u32, k as u32);
                let x = dev.buf(&vec![0.5f32; rows * k]);
                let mut s = Vec::new();
                g.set_declines_library(true);
                g.begin_tape();
                for _ in 0..REPS {
                    let _ = dev.linear_new(&mut s, &w, &x, rows as u32);
                }
                g.submit(&[], &s);
                let tape = g.end_tape();
                g.set_declines_library(false);
                let mut best = f64::MAX;
                for round in 0..rounds {
                    let t = std::time::Instant::now();
                    g.replay_tape(&tape);
                    g.poll_wait();
                    if round >= 5 {
                        best = best.min(t.elapsed().as_secs_f64() / REPS as f64);
                    }
                }
                line += &format!(" {name} {:6.1}us", best * 1e6);
            }
            eprintln!("{line}");
        }
    }
}

/// One recorded step replayed once at batch 1 / batch 64, for a kernel profiler (`ncu --launch-count`): the
/// timed probes above replay hundreds of steps.
#[test]
#[ignore = "profiling entry point: run under ncu"]
fn decoder_step_for_profiler_b1() {
    step_once(1);
}

/// See [`decoder_step_for_profiler_b1`].
#[test]
#[ignore = "profiling entry point: run under ncu"]
fn decoder_step_for_profiler_b64() {
    step_once(64);
}

fn step_once(batch: usize) {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else { return };
    let Some(cats) = crate::decoder::golden("step profile", "cats") else { return };
    let model = mirage::model::Mirage::open(&dir).unwrap();
    let dev = devices("step profile").into_iter().next().expect("a device");
    let dd = DeviceDecoder::upload(&dev, model.decoder());
    let memory: Vec<f32> = cats["memory"].iter().map(|&v| v as f32).collect();
    let tokens: Vec<u32> = cats["tokens"].iter().map(|&t| t as u32).collect();
    let mems: Vec<&[f32]> = (0..batch).map(|_| &memory[..]).collect();
    let cross = dd.cross(&dev, &memory_batch(&dev, &mems, 512));
    let mut st = dd.new_state(&dev, batch, 700);
    st.set_tokens(&dev, &vec![tokens; batch]);
    st.first = 2;
    let (ban, picked) = (dev.buf(&vec![0f32; batch * dd.vocabulary()]), dev.alloc(batch));
    let tape = dd.record_step(&dev, &st, &cross, true, Some((&ban, &picked)));
    st.seek(&dev, 150);
    dev.gpu().replay_tape(&tape);
    dev.gpu().poll_wait();
}
