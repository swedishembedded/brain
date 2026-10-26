// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded full fine-tuning of language models in bf16 and FP8 for its
// clients. If your team needs expertise in training every weight of a real model on a fixed accelerator budget
// then you can procure our services by sending an email to info@swedishembedded.com.

//! Bounded real-model check of the resident low-precision full fine-tune: Qwen3-0.6B with its linears resident in
//! bf16 or FP8 on the device, an f32 master and the optimizer state in host memory, trained on a synthetic
//! next-token rule; loss before and after on train and held-out sequences, step time, peak device memory, the
//! streamed export in the resident format and the loss of the model reloaded from it.
//!
//! ```text
//! card --card 1 --gb 16 --wait -- cargo test --release -p brain-qwen3 --test real_resident_full_ft -- --ignored --nocapture
//! ```

use std::time::Instant;

use checkpoint::packed::QuantEncoding;
use model::resident_ft::{ClipPass, LayerWindow, ResidentFtConfig, ResidentValue};
use optim::rule::Rule;
use paramstore::precision::{GradDtype, MasterPolicy, StateFormat};
use qwen3::Qwen;

const T: u32 = 64;
const STEPS: u32 = 24;

fn sequence(seed: u32, vocab: u32) -> (Vec<u32>, Vec<u32>) {
    let mut cur = (seed * 131 + 17) % 997;
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for _ in 0..T {
        let nxt = (7 * cur + 3) % 997;
        x.push(cur % vocab);
        y.push(nxt % vocab);
        cur = nxt;
    }
    (x, y)
}

fn mean_loss(m: &Qwen, seqs: &[(Vec<u32>, Vec<u32>)]) -> f32 {
    let mut s = 0.0;
    for (x, y) in seqs {
        m.set_batch(x, y);
        s += m.forward();
    }
    s / seqs.len() as f32
}

/// One run: how the linears are held and updated, and the format the result is exported in.
struct Case {
    value: ResidentValue,
    state: StateFormat,
    grad: GradDtype,
    export: QuantEncoding,
    rule: Rule,
    /// Layers of the linears kept on the device at once (`None`: every layer resident).
    window: Option<u32>,
}

impl Case {
    fn new(value: ResidentValue, state: StateFormat, grad: GradDtype, export: QuantEncoding, rule: Rule) -> Case {
        Case { value, state, grad, export, rule, window: None }
    }
}

fn run(name: &str, path: &str, case: Case) {
    let Case { value, state, grad, export, rule, window } = case;
    let (cfg, init) = qwen3::open_checkpoint(path).expect("open checkpoint");
    let vocab = cfg.vocab;
    let opts = model::FitOpts { batch_size: 1, block_size: T, ..Default::default() };
    let rcfg = ResidentFtConfig {
        value,
        master: MasterPolicy::F32,
        state,
        rule,
        grad,
        stochastic_state: true,
        seed: None,
        tile_elems: 1 << 23,
        state_dir: None,
        defer_publication: false,
        window: window.map(|layers| LayerWindow { layers }),
        clip_pass: ClipPass::TwoPass,
    };
    let m = qwen3::finetune::build_resident(cfg, &opts, &*init, &rcfg).expect("build");
    println!("{name}: device bytes of the resident linears: {} MiB", m.resident_device_weight_bytes().unwrap_or(0) >> 20);
    drop(init);
    let train: Vec<_> = (0..8).map(|i| sequence(i, vocab)).collect();
    let held: Vec<_> = (100..108).map(|i| sequence(i, vocab)).collect();
    let (t0_train, t0_held) = (mean_loss(&m, &train), mean_loss(&m, &held));
    let mut vram = brain_testutil::own_gpu_memory_mib().unwrap_or(0);
    let t = Instant::now();
    let mut optimizer = std::time::Duration::ZERO;
    for step in 1..=STEPS {
        let (x, y) = &train[(step as usize - 1) % train.len()];
        m.zero_grads();
        m.set_batch(x, y);
        m.forward();
        m.backward();
        m.poll_wait();
        let o = Instant::now();
        m.adamw_step(step, 1e-5, 0.0, Default::default(), Some(1.0), 1.0);
        optimizer += o.elapsed();
        vram = vram.max(brain_testutil::own_gpu_memory_mib().unwrap_or(0));
    }
    m.poll_wait();
    let ms = t.elapsed().as_secs_f64() * 1e3 / STEPS as f64;
    let opt_ms = optimizer.as_secs_f64() * 1e3 / STEPS as f64;
    let (t1_train, t1_held) = (mean_loss(&m, &train), mean_loss(&m, &held));
    println!("{name}: train loss {t0_train:.3} -> {t1_train:.3}, held-out {t0_held:.3} -> {t1_held:.3}, {ms:.0} ms/step ({opt_ms:.0} ms in the host optimizer step, {:.0} ms forward+backward), device peak {vram} MiB", ms - opt_ms);
    if let Some(h) = m.resident_host_times() {
        let per = |d: std::time::Duration| d.as_secs_f64() * 1e3 / h.steps.max(1) as f64;
        println!(
            "{name}: host update per step: gradient readback {:.0} ms, value encode {:.0} ms, device write {:.0} ms, whole update {:.0} ms ({} steps)",
            per(h.grad_read),
            per(h.value_encode),
            per(h.value_write),
            per(h.update),
            h.steps
        );
    }
    brain_testutil::mem(name);
    assert!(t1_train < t0_train, "{name}: training loss must fall");

    let dir = std::env::temp_dir().join(format!("brain-real-resident-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("export.safetensors");
    let started = Instant::now();
    let report = m.export_resident(&file, &export).expect("export");
    println!("{name}: exported {} MiB in {:.1} s, host staging {} KiB", report.bytes >> 20, started.elapsed().as_secs_f64(), report.linears.staging_bytes >> 10);
    drop(m);
    let (cfg, src) = qwen3::open_checkpoint(file.to_str().unwrap()).expect("reload");
    let reloaded = Qwen::new_shard(cfg.clone(), 1, T, &*src, false, qwen3::Shard::whole(cfg.n_layers as usize));
    let (r_train, r_held) = (mean_loss(&reloaded, &train), mean_loss(&reloaded, &held));
    println!("{name}: reloaded from the export: train loss {r_train:.3}, held-out {r_held:.3}");
    assert!((r_train - t1_train).abs() < 0.1 * t1_train.max(1.0), "{name}: the exported model computes the trained loss ({r_train} vs {t1_train})");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore]
fn qwen3_0_6b_resident_bf16_and_fp8_full_fine_tune() {
    let Some(path) = brain_testutil::model_dir("Qwen/Qwen3-0.6B") else { return };
    if !std::path::Path::new(&path).exists() {
        brain_testutil::skip_unavailable("Qwen/Qwen3-0.6B is not in the model store");
        return;
    }
    let adamw = Rule::AdamW { adam: model::Adam::default(), wd: 0.0 };
    run("resident bf16, f32 master, f32 moments", &path, Case::new(ResidentValue::Bf16, StateFormat::F32, GradDtype::F32, QuantEncoding::bf16(), adamw));
    run("resident bf16, f32 master, int8 moments, bf16 grads", &path, Case::new(ResidentValue::Bf16, StateFormat::Int8Block { block: 64 }, GradDtype::Bf16, QuantEncoding::bf16(), adamw));
    run("resident fp8, f32 master, f32 moments", &path, Case::new(ResidentValue::Fp8E4M3 { block: 128 }, StateFormat::F32, GradDtype::F32, QuantEncoding::fp8_e4m3_tiled(128), adamw));
    run("resident bf16, f32 master, adafactor", &path, Case::new(ResidentValue::Bf16, StateFormat::F32, GradDtype::F32, QuantEncoding::bf16(), Rule::adafactor(0.0)));
}

/// The layer-streamed form against the resident one on the real model: the same loss, with the linears' device
/// bytes bounded by the window (the loss of a run on a GPU is not bit-equal run to run, the embedding gradient is
/// accumulated with atomics; the comparison is in the printed losses).
#[test]
#[ignore]
fn qwen3_0_6b_layer_streamed_full_fine_tune_reaches_the_resident_loss() {
    let Some(path) = brain_testutil::model_dir("Qwen/Qwen3-0.6B") else { return };
    if !std::path::Path::new(&path).exists() {
        brain_testutil::skip_unavailable("Qwen/Qwen3-0.6B is not in the model store");
        return;
    }
    let adamw = Rule::AdamW { adam: model::Adam::default(), wd: 0.0 };
    for (label, window) in [("resident (every layer)", None), ("layer-streamed, window of 1 layer", Some(1)), ("layer-streamed, window of 2 layers", Some(2))] {
        let case = Case { window, ..Case::new(ResidentValue::Bf16, StateFormat::F32, GradDtype::F32, QuantEncoding::bf16(), adamw) };
        run(&format!("bf16, f32 master, f32 moments, {label}"), &path, case);
    }
    let fp8 = Case { window: Some(2), ..Case::new(ResidentValue::Fp8E4M3 { block: 128 }, StateFormat::F32, GradDtype::F32, QuantEncoding::fp8_e4m3_tiled(128), adamw) };
    run("fp8, f32 master, f32 moments, layer-streamed, window of 2 layers", &path, fp8);
}
