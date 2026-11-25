// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training of composite models (several decoders optimised as one) for its clients.
// If your team needs expertise in training multi-stage speech and language models on one device then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec of the composite-training seams of `qwen3::Qwen`: a training build on another Qwen's device handle
//! (`Qwen::new_training_on`) computes what a build on its own device does, and its buffers are addressable from the
//! parent's handle; a frozen parameter (`Qwen::freeze_params`) leaves the optimised set and is never stepped, weight
//! decay included; and the final-norm hidden states the head read are readable after a forward
//! (`Qwen::final_hidden`).

use gpu_core::provider::library::GemmPrecision;
use qwen3::model::FullFtBuild;
use qwen3::{Qwen, QwenConfig};

const TOKENS: [u32; 12] = [1, 5, 9, 2, 7, 3, 11, 4, 8, 6, 10, 2];
const TARGETS: [u32; 12] = [5, 9, 2, 7, 3, 11, 4, 8, 6, 10, 2, 1];

fn config() -> QwenConfig {
    QwenConfig { block_size: 6, tie_embeddings: false, ..QwenConfig::tiny() }
}

fn build(parent: Option<&gpu_core::Gpu>) -> Qwen {
    build_at(parent, 6)
}

fn build_at(parent: Option<&gpu_core::Gpu>, t: u32) -> Qwen {
    let cfg = QwenConfig { block_size: t, max_position_embeddings: t, ..config() };
    // A fixed seed: both builds must hold the same weights for their losses to be compared.
    let init = qwen3::init_weights(&cfg, 11);
    Qwen::new_training_on(parent, cfg, 2, t, &init, GemmPrecision::Fp32Reference, &FullFtBuild::ambient()).expect("a full fine-tune build")
}

#[test]
fn a_build_on_a_parent_handle_trains_like_a_build_on_its_own() {
    let own = build(None);
    let first = build(None);
    let second = build(Some(first.gpu()));
    for m in [&own, &second] {
        m.set_batch(&TOKENS, &TARGETS);
        m.zero_grads();
    }
    let (want, got) = (own.forward(), second.forward());
    own.backward();
    second.backward();
    assert_eq!(want.to_bits(), got.to_bits(), "the loss of the build on a parent handle: {got} vs {want}");
    assert_eq!(own.read_grad("lm_head.weight"), second.read_grad("lm_head.weight"));
    // One handle reaches both stores: the parent's device grads list holds the child's buffers too.
    let grads = model::Model::device_grads(&second).expect("store gradients");
    assert_eq!(grads.tensors.len(), model::Model::param_names(&second).len());
    let h = first.gpu().read(&grads.tensors[0].0, grads.tensors[0].1 as usize);
    assert_eq!(h, second.read_grad(&model::Model::param_names(&second)[0]), "the parent handle reads the child's gradient buffer");
}

#[test]
fn a_frozen_parameter_leaves_the_optimised_set_and_is_never_stepped() {
    let mut m = build(None);
    let before = m.read_weight("tok.weight");
    assert_eq!(m.freeze_params(|n| n == "tok.weight"), 1);
    assert!(!model::Model::param_names(&m).iter().any(|n| n == "tok.weight"));
    m.set_batch(&TOKENS, &TARGETS);
    m.zero_grads();
    m.forward();
    m.backward();
    m.adamw_step(1, 1e-2, 0.1, model::Adam::default(), Some(1.0), 1.0);
    assert_eq!(m.read_weight("tok.weight"), before, "a frozen table moved (weight decay included)");
    assert_ne!(m.read_weight("lm_head.weight"), qwen3::init_weights(&config(), 11)["lm_head.weight"], "the rest still trains");
}

#[test]
fn the_final_hidden_states_are_what_the_head_reads() {
    let m = build(None);
    m.set_batch(&TOKENS, &TARGETS);
    let loss = m.forward();
    let hidden = m.final_hidden();
    let (d, v) = (config().d_model as usize, config().vocab as usize);
    assert_eq!(hidden.len(), TOKENS.len() * d);
    let head = m.read_weight("lm_head.weight");
    // The mean CE of the head over the hidden states read back is the loss the forward returned.
    let mut sum = 0f64;
    for (i, &y) in TARGETS.iter().enumerate() {
        let logits: Vec<f64> = (0..v).map(|j| (0..d).map(|c| f64::from(hidden[i * d + c]) * f64::from(head[j * d + c])).sum()).collect();
        let mx = logits.iter().cloned().fold(f64::MIN, f64::max);
        sum += mx + logits.iter().map(|l| (l - mx).exp()).sum::<f64>().ln() - logits[y as usize];
    }
    let want = sum / TARGETS.len() as f64;
    assert!((f64::from(loss) - want).abs() <= 1e-4 * want.abs(), "loss {loss} vs the head over the read hidden states {want}");
}

/// A build fed input rows (`Qwen::enable_input_rows`) with the token embeddings of a batch trains exactly as the build
/// that gathers them, also when the batch is trimmed (a tail of ignored targets): the loss and every gradient but the
/// embedding table's (which the rows bypass, so it stays zero) match bit for bit.
#[test]
fn input_rows_train_like_the_tokens_they_embed() {
    const T: usize = 192;
    let tokens = build_at(None, T as u32);
    let mut rows = build_at(None, T as u32);
    rows.enable_input_rows();
    let (d, v) = (config().d_model as usize, config().vocab);
    let table = tokens.read_weight("tok.weight");
    let ids: Vec<u32> = (0..2 * T as u32).map(|i| (i * 7 + 3) % v).collect();
    let embedded: Vec<f32> = ids.iter().flat_map(|&t| table[t as usize * d..(t as usize + 1) * d].iter().copied()).collect();
    let full: Vec<u32> = (0..2 * T as u32).map(|i| (i * 5 + 1) % v).collect();
    // Supervised up to position 40 of each row: the run is trimmed to 64.
    let trimmed: Vec<u32> = full.iter().enumerate().map(|(i, &y)| if i % T < 40 { y } else { qwen3::IGNORE }).collect();
    for targets in [&full, &trimmed] {
        tokens.set_batch(&ids, targets);
        rows.set_batch(&vec![0; 2 * T], targets);
        rows.write_input_rows(&embedded);
        for m in [&tokens, &rows] {
            m.zero_grads();
        }
        let (want, got) = (tokens.forward(), rows.forward());
        tokens.backward();
        rows.backward();
        assert_eq!(tokens.run_len(), rows.run_len());
        assert_eq!(want.to_bits(), got.to_bits(), "loss {got} vs {want} (run length {})", rows.run_len());
        for name in model::Model::param_names(&rows) {
            if name == "tok.weight" {
                assert!(rows.read_grad(&name).iter().all(|&g| g == 0.0), "the bypassed table took a gradient");
            } else {
                assert_eq!(rows.read_grad(&name), tokens.read_grad(&name), "{name}");
            }
        }
    }
    assert!((rows.run_len() as usize) < T, "the second batch must be trimmed for the test to cover it");
}

/// Rows injected from another model's hidden states on the device (`Qwen::enable_row_injection`) give the loss and
/// gradients of the same rows written from the host: a second model reads the first one's final-norm hidden state at
/// planned positions with no host round trip. The table the injected rows replace must be frozen first.
#[test]
fn injected_rows_are_the_other_models_hidden_states() {
    let first = build(None);
    let mut injected = build(Some(first.gpu()));
    assert!(injected.enable_row_injection(first.final_hidden_buffer(), 3, None).is_err(), "a trainable table under injected rows is refused");
    injected.freeze_params(|n| n == "tok.weight");
    injected.enable_row_injection(first.final_hidden_buffer(), 3, None).expect("a frozen table");
    let mut host = build(None);
    host.freeze_params(|n| n == "tok.weight");
    host.enable_input_rows();

    first.set_batch(&TOKENS, &TARGETS);
    first.forward();
    let hidden = first.final_hidden();
    // Source rows of the first model's [2, 6] block, into destination rows 0, 6 and 9 of the second's.
    let plan = [(5u32, 0u32), (0, 6), (7, 9)];
    let d = config().d_model as usize;
    let table = injected.read_weight("tok.weight");
    let mut rows: Vec<f32> = TOKENS.iter().flat_map(|&t| table[t as usize * d..(t as usize + 1) * d].iter().copied()).collect();
    for &(src, dst) in &plan {
        rows[dst as usize * d..(dst as usize + 1) * d].copy_from_slice(&hidden[src as usize * d..(src as usize + 1) * d]);
    }
    injected.set_batch(&TOKENS, &TARGETS);
    injected.write_injection(&plan, 3, 0);
    host.set_batch(&[0; 12], &TARGETS);
    host.write_input_rows(&rows);
    for m in [&injected, &host] {
        m.zero_grads();
    }
    let (want, got) = (host.forward(), injected.forward());
    host.backward();
    injected.backward();
    assert_eq!(want.to_bits(), got.to_bits(), "loss {got} vs {want}");
    for name in model::Model::param_names(&injected) {
        assert_eq!(injected.read_grad(&name), host.read_grad(&name), "{name}");
    }
}

/// A build that trims its batch (`Qwen::enable_batch_trimming`) computes only the leading sequences that carry a
/// target, and its loss and gradients are those of the untrimmed build bit for bit.
#[test]
fn a_trimmed_batch_trains_like_the_whole_block() {
    let cfg = config();
    let init = qwen3::init_weights(&cfg, 11);
    let make = || Qwen::new_training_on(None, cfg.clone(), 16, 6, &init, GemmPrecision::Fp32Reference, &FullFtBuild::ambient()).expect("a build");
    let whole = make();
    let mut trimmed = make();
    trimmed.enable_batch_trimming();
    let tokens: Vec<u32> = (0..96u32).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    // Targets in the first five sequences only: those five are computed.
    let targets: Vec<u32> = (0..96u32).map(|i| if i < 30 { (i * 3 + 2) % cfg.vocab } else { qwen3::IGNORE }).collect();
    for m in [&whole, &trimmed] {
        m.set_batch(&tokens, &targets);
        m.zero_grads();
    }
    assert_eq!((whole.batch_run(), trimmed.batch_run()), (16, 5));
    let (want, got) = (whole.forward(), trimmed.forward());
    whole.backward();
    trimmed.backward();
    assert_eq!(want.to_bits(), got.to_bits(), "loss {got} vs {want}");
    for name in model::Model::param_names(&trimmed) {
        assert_eq!(trimmed.read_grad(&name), whole.read_grad(&name), "{name}");
    }
}
