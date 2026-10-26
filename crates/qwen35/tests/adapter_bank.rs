// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one large model from a single resident copy of its weights for
// its clients. If your team needs expertise in multi-tenant inference of
// hybrid-attention models then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The served adapter bank of `qwen35::serve::Engine` against the training
//! forward.
//!
//! * a sequence under a served adapter produces the logits the LoRA build
//!   with the same adapter computes (`logits_all`), at the prefill's last
//!   position and at every decode step;
//! * two different adapters and a base row share ONE batched decode step and
//!   each gets its own answer;
//! * the base row of that mixed batch is bit-identical to an engine that has
//!   no bank;
//! * a bank of one slot swaps adapters in and out as the pins allow.
//!
//! The weights are the tiny config's (3 Gated-DeltaNet layers and one GQA
//! layer, every targetable leaf), the adapters random and large enough to move
//! the logits, so applied and ignored cannot look alike.

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::Gpu;
use model::adapter::bank::{AdapterBankConfig, AdapterServing, AdapterTargets, VerifiedAdapter};
use model::lora::device_adapter::{AdapterSite, DeviceAdapter};
use model::paged::BlockTable;
use model::serve::{AdapterAvailability, AdapterId, PagedDecoder};
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};
use qwen35::serve::Engine;

// seed: the TOL below was measured on this weight draw
const SEED: u64 = 7;
const RANK: u32 = 5;
const ALPHA: f32 = 8.0;
const PREFIX: usize = 6;
const TOL: f32 = 2e-3;

fn tokens(cfg: &Qwen35Config) -> Vec<u32> {
    (0..cfg.block_size).map(|i| (i * 5 + 3) % cfg.vocab).collect()
}

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// The base weights, and one adapter's factors drawn from `seed`: the LoRA
/// build's init map with its `lora_a`/`lora_b` made non-zero.
fn weights(adapter_seed: u64) -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<f32>>) {
    let base_cfg = Qwen35Config::tiny();
    let lora = Qwen35Config { lora: Some(lora_cfg(RANK, ALPHA)), ..Qwen35Config::tiny() };
    let mut with_lora = qwen35::init::init_weights(&lora, SEED);
    let base = qwen35::init::init_weights(&base_cfg, SEED);
    with_lora.extend(base.clone());
    let mut rng = Rng::new(adapter_seed);
    for (name, w) in with_lora.iter_mut().filter(|(n, _)| n.ends_with(".lora_a") || n.ends_with(".lora_b")) {
        if name.ends_with(".lora_b") {
            w.iter_mut().for_each(|x| *x = 2.0 * (rng.next_f32() - 0.5));
        } else {
            w.iter_mut().for_each(|x| *x *= 20.0);
        }
    }
    (base, with_lora)
}

/// The adapter held in `with_lora`'s `lora_a`/`lora_b` tensors, as a served one.
fn served(cfg: &Qwen35Config, with_lora: &HashMap<String, Vec<f32>>, label: &str) -> VerifiedAdapter {
    use model::adapter::sites::LoraSites;
    let sites = cfg
        .lora_sites()
        .iter()
        .map(|s| AdapterSite { base: s.name.clone(), a: with_lora[&format!("{}.lora_a", s.name)].clone(), b: with_lora[&format!("{}.lora_b", s.name)].clone() })
        .collect();
    VerifiedAdapter { label: label.into(), digest: format!("sha256:{label}"), adapter: DeviceAdapter { rank: RANK, alpha: ALPHA, sites } }
}

/// Logits of the training forward with the adapter, row per position.
fn reference_logits(make_gpu: fn(&[(&str, &str)]) -> Gpu, with_lora: &HashMap<String, Vec<f32>>) -> Vec<f32> {
    let cfg = Qwen35Config { lora: Some(lora_cfg(RANK, ALPHA)), ..Qwen35Config::tiny() };
    let m = Qwen35::new_train_on(make_gpu(pipelines()), cfg.clone(), 1, cfg.block_size, with_lora);
    m.logits_all(&tokens(&cfg))
}

/// An engine on `base`, with a bank of `slots` when asked: grouped as the model
/// declares ([`AdapterServing::input_groups`]), the way a server builds it, or
/// with every linear on its own.
fn engine_with(make_gpu: fn(&[(&str, &str)]) -> Gpu, base: &HashMap<String, Vec<f32>>, slots: Option<u32>, grouped: bool) -> Engine {
    let cfg = Qwen35Config::tiny();
    let mut e = Engine::from_map_on(&make_gpu(pipelines()), cfg.clone(), base, cfg.block_size, 4);
    if let Some(slots) = slots {
        let targets = AdapterTargets::ALL.resolve(&cfg).unwrap();
        let mut bank = AdapterBankConfig::new(slots, 8, &targets);
        if grouped {
            bank.groups = cfg.input_groups().into_iter().map(|g| g.into_iter().map(str::to_string).collect()).collect();
        }
        e.enable_adapters(bank).unwrap();
    }
    e
}

fn engine(make_gpu: fn(&[(&str, &str)]) -> Gpu, base: &HashMap<String, Vec<f32>>, slots: Option<u32>) -> Engine {
    engine_with(make_gpu, base, slots, true)
}

fn admit(e: &mut Engine, id: AdapterId) {
    assert_eq!(e.adapter_availability(id), AdapterAvailability::Resident);
    e.pin_adapter(id);
}

fn bits(v: &[(u32, f32)]) -> Vec<(u32, u32)> {
    v.iter().map(|&(t, l)| (t, l.to_bits())).collect()
}

fn mixed_batch_matches_training_forward_and_base_rows_are_untouched(make_gpu: fn(&[(&str, &str)]) -> Gpu) {
    let cfg = Qwen35Config::tiny();
    let toks = tokens(&cfg);
    let v = cfg.vocab as usize;
    let (base, lora_a) = weights(99);
    let (_, lora_b) = weights(1234);
    let (want_a, want_b) = (reference_logits(make_gpu, &lora_a), reference_logits(make_gpu, &lora_b));
    let want_base = {
        let m = Qwen35::new_on(make_gpu(pipelines()), cfg.clone(), 1, cfg.block_size, &base);
        m.logits_all(&toks)
    };
    assert!(maxabs(&want_a, &want_base) > 20.0 * TOL && maxabs(&want_a, &want_b) > 20.0 * TOL, "the adapters must move the logits for the test to tell applied from ignored");

    let mut banked = engine(make_gpu, &base, Some(2));
    let mut plain = engine(make_gpu, &base, None);
    let id_a = banked.load_adapter(served(&cfg, &lora_a, "a")).unwrap();
    let id_b = banked.load_adapter(served(&cfg, &lora_b, "b")).unwrap();
    admit(&mut banked, id_a);
    admit(&mut banked, id_b);

    // One sequence per adapter and one base sequence, prefilled alone.
    let mut tables = [BlockTable::with_namespace(id_a.kv_namespace()), BlockTable::new(), BlockTable::with_namespace(id_b.kv_namespace())];
    let mut plain_base = BlockTable::new();
    let hidden: Vec<Vec<f32>> = tables.iter_mut().map(|t| banked.prefill(t, &toks[..PREFIX])).collect();
    let plain_hidden = plain.prefill(&mut plain_base, &toks[..PREFIX]);
    let last = (PREFIX - 1) * v..PREFIX * v;
    assert!(maxabs(&banked.logits(&hidden[0]), &want_a[last.clone()]) < TOL, "adapter a, prefill");
    assert!(maxabs(&banked.logits(&hidden[2]), &want_b[last.clone()]) < TOL, "adapter b, prefill");
    assert_eq!(hidden[1].iter().map(|x| x.to_bits()).collect::<Vec<_>>(), plain_hidden.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "the base sequence's prefill is the no-bank engine's");

    // The rest of the sequence, teacher forced, all three rows in ONE batch.
    for (i, &tok) in toks.iter().enumerate().skip(PREFIX) {
        let [ta, tbase, tb] = &mut tables;
        let got = banked.forward_batched_topk(&mut [ta, tbase, tb], &[tok; 3], 1);
        let plain_got = plain.forward_batched_topk(&mut [&mut plain_base], &[tok], 1);
        for (row, want) in [(0, &want_a), (2, &want_b), (1, &want_base)] {
            let truth = &want[i * v..(i + 1) * v];
            let (best, best_logit) = truth.iter().copied().enumerate().fold((0, f32::MIN), |m, (t, l)| if l > m.1 { (t, l) } else { m });
            assert_eq!(got[row][0].0, best as u32, "row {row} picks the training forward's token at position {i}");
            assert!((got[row][0].1 - best_logit).abs() < TOL, "row {row} logit at position {i}: {} vs {best_logit}", got[row][0].1);
        }
        assert_eq!(bits(&got[1]), bits(&plain_got[0]), "the base row of a mixed batch equals the no-bank engine's, bit for bit (position {i})");
    }
}

#[test]
fn mixed_batch_matches_training_forward_and_base_rows_are_untouched_cpu() {
    mixed_batch_matches_training_forward_and_base_rows_are_untouched(Gpu::new_cpu);
}

/// Run on the default backend (CUDA under the gpu wrapper).
#[test]
fn mixed_batch_matches_training_forward_and_base_rows_are_untouched_default_backend() {
    mixed_batch_matches_training_forward_and_base_rows_are_untouched(Gpu::new);
}

/// The grouped bank (one projection and one epilogue for the linears that read
/// one activation) computes what the bank with every linear on its own does:
/// the same adapter rows to the bit, because the zero padding of a stacked
/// bank adds nothing. And it records fewer dispatches doing it.
#[test]
fn a_grouped_bank_equals_the_ungrouped_one_bit_for_bit() {
    let cfg = Qwen35Config::tiny();
    let toks = tokens(&cfg);
    let (base, lora_a) = weights(99);
    let run = |grouped: bool| {
        let mut e = engine_with(Gpu::new_cpu, &base, Some(1), grouped);
        let id = e.load_adapter(served(&cfg, &lora_a, "a")).unwrap();
        admit(&mut e, id);
        let mut t = BlockTable::with_namespace(id.kv_namespace());
        let hidden = e.prefill(&mut t, &toks[..PREFIX]);
        let mut out = vec![bits_of_f32(&hidden)];
        for &tok in &toks[PREFIX..] {
            let top = e.forward_batched_topk(&mut [&mut t], &[tok], 4);
            out.push(top[0].iter().flat_map(|&(id, l)| [id, l.to_bits()]).collect());
        }
        out
    };
    assert_eq!(run(true), run(false));

    // The byte count a placement budgets is what the grouped bank allocates.
    let targets = AdapterTargets::ALL.resolve(&cfg).unwrap();
    for grouped in [true, false] {
        let e = engine_with(Gpu::new_cpu, &base, Some(1), grouped);
        let mut bank = AdapterBankConfig::new(1, 8, &targets);
        if grouped {
            bank.groups = cfg.input_groups().into_iter().map(|g| g.into_iter().map(str::to_string).collect()).collect();
        }
        assert_eq!(e.adapter_bank_bytes(), bank.device_bytes(&cfg, 256), "grouped: {grouped}");
    }
}

fn bits_of_f32(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn one_slot_swaps_adapters_as_pins_allow_and_unknown_or_unfit_adapters_are_refused() {
    let cfg = Qwen35Config::tiny();
    let (base, lora_a) = weights(99);
    let (_, lora_b) = weights(1234);
    let mut e = engine(Gpu::new_cpu, &base, Some(1));
    let (a, b) = (e.load_adapter(served(&cfg, &lora_a, "a")).unwrap(), e.load_adapter(served(&cfg, &lora_b, "b")).unwrap());
    assert_eq!(e.adapter_availability(AdapterId::new(77).unwrap()), AdapterAvailability::Unknown, "an unregistered adapter is rejected, not waited for");

    admit(&mut e, a);
    assert_eq!(e.adapter_availability(b), AdapterAvailability::Loading, "the only slot is pinned by a sequence");
    e.unpin_adapter(a);
    assert_eq!(e.adapter_availability(b), AdapterAvailability::Resident, "the freed slot is b's");
    assert!(e.adapters().iter().any(|i| i.id == a && !i.resident) && e.adapters().iter().any(|i| i.id == b && i.resident));

    // Rank above the bank's, and a site of another model.
    let mut too_big = served(&cfg, &lora_a, "big");
    too_big.adapter.rank = 9;
    assert!(e.load_adapter(too_big).is_err());
    let mut foreign = served(&cfg, &lora_a, "foreign");
    foreign.adapter.sites[0].base = "blocks.0.attn.wq.weight".into();
    assert!(e.load_adapter(foreign).is_err(), "a Qwen3 site is not a site of this model");
    let none = engine(Gpu::new_cpu, &base, None);
    assert_eq!(none.adapter_bank_bytes(), 0);
}

#[test]
fn a_banks_targets_group_per_model_and_attention_is_the_mixers() {
    let cfg = Qwen35Config::tiny();
    let attention = AdapterTargets::ATTENTION.resolve(&cfg).unwrap();
    assert_eq!(attention, ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"]);
    assert_eq!(AdapterTargets::ALL.resolve(&cfg).unwrap().len(), 12);
    assert_eq!(cfg.default_rank_cap(), 16);
    assert!(AdapterTargets::parse("wq").unwrap().resolve(&cfg).is_err(), "wq is a Qwen3 leaf");
}

/// What the trainer writes is what the bank serves: an adapter saved with
/// `qwen35::lora::save_adapter_with_lineage` is read back through the generic
/// reader (which refuses a card whose targets differ from its tensors), its
/// card's targets resolve against the model's sites, and the engine answers
/// under it as the training forward does.
#[test]
fn an_adapter_the_trainer_wrote_is_served() {
    let cfg = Qwen35Config::tiny();
    let toks = tokens(&cfg);
    let v = cfg.vocab as usize;
    let (base, with_lora) = weights(99);
    let lora_cfg_full = Qwen35Config { lora: Some(lora_cfg(RANK, ALPHA)), ..Qwen35Config::tiny() };
    let trained = Qwen35::new_train_on(Gpu::new_cpu(pipelines()), lora_cfg_full.clone(), 1, cfg.block_size, &with_lora);
    let dir = std::env::temp_dir().join(format!("brain-qwen35-adapter-bank-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("adapter.safetensors");
    qwen35::lora::save_adapter_with_lineage(path.to_str().unwrap(), &trained, "local/tiny:lab:chat:v1", "local/tiny", None, None).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let (adapter, card) = model::lora::device_adapter::parse_adapter(&bytes, "adapter.safetensors").unwrap();
    std::fs::remove_dir_all(&dir).ok();
    let descriptor = card.adapter.as_ref().unwrap();
    assert_eq!((card.family.as_str(), descriptor.kind.as_str()), ("qwen35", "lora"));
    let card_targets = AdapterTargets::parse(&descriptor.targets.clone().unwrap().join(",")).unwrap();
    assert_eq!(card_targets.resolve(&cfg).unwrap().len(), 12, "the card's targets are leaves of this model, every one of them");

    let mut e = engine(Gpu::new_cpu, &base, Some(1));
    let id = e.load_adapter(VerifiedAdapter { label: card.id.clone(), digest: "sha256:trainer".into(), adapter }).unwrap();
    admit(&mut e, id);
    let mut table = BlockTable::with_namespace(id.kv_namespace());
    let hidden = e.prefill(&mut table, &toks[..PREFIX]);
    let want = trained.logits_all(&toks);
    assert!(maxabs(&e.logits(&hidden), &want[(PREFIX - 1) * v..PREFIX * v]) < TOL, "the served trainer adapter answers as the training forward does");
}
