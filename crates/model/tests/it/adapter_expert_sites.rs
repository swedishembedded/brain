// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements per-request fine-tuned variants of mixture-of-experts models served from one resident
// copy of their quantized expert banks for its clients. If your team needs expertise in training, storing and serving
// stacked-expert adapters then you can procure our services by sending an email to info@swedishembedded.com.

//! Spec: a stacked-expert adapter (per expert `e`, `A_e [r, in]` and `B_e [out, r]` over an `[E, out, in]` bank) goes
//! from training to serving through the shared adapter file and bank:
//! - trained over a frozen MXFP8 bank, it is saved with its real rank-3 shapes and read back exactly, card included;
//! - served from an [`AdapterBank`] beside the unmodified bank (the routed expert products then
//!   [`AdapterBank::correct_experts`]), a batch mixing rows with and without the adapter gives the adapted rows the
//!   training forward's output and leaves the other rows the base products' bits;
//! - folded into an f32 copy of the bank, expert by expert, it is the per-expert low-rank delta;
//! - a malformed stacked file, a linear reader handed one, or a pair that does not fit the model's expert sites is
//!   refused by name.

use std::cell::RefCell;
use std::collections::HashMap;

use checkpoint::st::{Adapter, ModelCard};
use data::rng::Lcg;
use gpu_core::Gpu;
use model::adapter::bank::{AdapterBank, AdapterBankConfig, AdapterError, AdapterServing, AdapterTargets, VerifiedAdapter};
use model::adapter::sites::{validate, validate_with_experts, ExpertSite, LoraSites, SiteMismatch};
use model::adapter::LinearSite;
use model::dispatch::{LORA_BANK_PIPELINES, LORA_EXPERT_BANK_PIPELINES, LORA_SLOT_NONE};
use model::glu::GluIds;
use model::lora::device_adapter::{fold_adapter_into, fold_delta, read_adapter, read_adapter_with_experts, save_lora_with_lineage, DeviceAdapter};
use model::ops::{micro_bank_kernels, MicroScaled};
use model::serve::AdapterAvailability;
use model::{Batch, Model, ModelConfig};

use crate::moe_expert_lora::{DeviceLayer, Layer, D, E, FF, LORA_SCALE, RANK, ROWS, TOP_K};
use crate::tier_kit::upload_u32;

const GATE_UP: &str = "layers.0.mlp.experts.w13";
const DOWN: &str = "layers.0.mlp.experts.w2";

fn expert_sites() -> Vec<ExpertSite> {
    let site = |name: &str, leaf, out, inn| ExpertSite { name: name.into(), leaf, layer: Some(0), experts: E as usize, out, inn, top_k: TOP_K as usize };
    vec![site(GATE_UP, "w13", 2 * FF as usize, D as usize), site(DOWN, "w2", D as usize, FF as usize)]
}

/// The device of the layer plus the adapter bank's kernels.
fn device() -> Gpu {
    let mut pipes = crate::moe_expert_lora::pipelines();
    for (name, src) in LORA_BANK_PIPELINES.into_iter().chain(LORA_EXPERT_BANK_PIPELINES).chain(micro_bank_kernels()) {
        if !pipes.iter().any(|(n, _)| *n == name) {
            pipes.push((name, src));
        }
    }
    gpu_core::testgpu::dev(Box::leak(pipes.into_boxed_slice()))
}

/// A model whose only adapter sites are the layer's two expert banks.
struct ExpertModel;

impl LoraSites for ExpertModel {
    fn lora_sites(&self) -> Vec<LinearSite> {
        Vec::new()
    }
    fn expert_sites(&self) -> Vec<ExpertSite> {
        expert_sites()
    }
}

impl AdapterServing for ExpertModel {}

// ---- a minimal model holding the adapter factors, for the shared save path ----------------------------------------

#[derive(Clone)]
struct HolderCfg;

impl ModelConfig for HolderCfg {
    fn param_list(&self) -> Vec<(String, usize)> {
        Vec::new()
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    fn from_json(_: &serde_json::Value) -> Self {
        HolderCfg
    }
    fn vocab(&self) -> u32 {
        0
    }
    fn block_size(&self) -> u32 {
        0
    }
    fn finalize_for_dataset(self, _v: u32, _b: u32) -> Self {
        self
    }
}

/// The stacked factors a trainer holds, under their adapter tensor names.
struct Holder {
    params: RefCell<HashMap<String, Vec<f32>>>,
}

impl Model for Holder {
    type Config = HolderCfg;
    fn new(_cfg: HolderCfg, _b: u32, _t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        Holder { params: RefCell::new(init.clone()) }
    }
    fn init_weights(_cfg: &HolderCfg, _seed: u64) -> HashMap<String, Vec<f32>> {
        HashMap::new()
    }
    fn config(&self) -> &HolderCfg {
        &HolderCfg
    }
    fn set_batch(&self, _b: Batch) {}
    fn forward(&self) -> f32 {
        0.0
    }
    fn backward(&self) {}
    fn zero_grads(&self) {}
    fn adamw_step(&self, _t: u32, _lr: f32, _wd: f32, _adam: model::Adam, _c: Option<f32>, _s: f32) {}
    fn poll_wait(&self) {}
    fn param_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.params.borrow().keys().cloned().collect();
        names.sort();
        names
    }
    fn adapter_expert_sites(&self) -> Vec<ExpertSite> {
        expert_sites()
    }
    fn read_weight(&self, n: &str) -> Vec<f32> {
        self.params.borrow()[n].clone()
    }
    fn write_weight(&self, n: &str, data: &[f32]) {
        self.params.borrow_mut().insert(n.to_string(), data.to_vec());
    }
    fn write_grad(&self, _n: &str, _data: &[f32]) -> Result<(), String> {
        Ok(())
    }
    fn read_grad(&self, _n: &str) -> Vec<f32> {
        Vec::new()
    }
    fn logits_all(&self, _t: &[u32]) -> Option<Vec<f32>> {
        None
    }
    fn save(&self, _p: &str) {}
    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

/// Save the layer's adapters through the shared writer, as a trainer holding them would.
fn save(host: &Layer, path: &str) {
    let params = HashMap::from([
        (format!("{GATE_UP}.lora_a"), host.lora_gate_up.a.clone()),
        (format!("{GATE_UP}.lora_b"), host.lora_gate_up.b.clone()),
        (format!("{DOWN}.lora_a"), host.lora_down.a.clone()),
        (format!("{DOWN}.lora_b"), host.lora_down.b.clone()),
    ]);
    let holder = Holder::new(HolderCfg, 1, 1, &params);
    let lora = model::adapter::device::LoraCfg::new(RANK, LORA_SCALE * RANK as f32, vec!["w13".into(), "w2".into()]);
    save_lora_with_lineage(path, &holder, &lora, "test/experts", "test/base", "moe-test", None, None).expect("save the stacked adapter");
}

/// One SGD step on the adapters from the layer's gradient (router held constant), applied to the host factors and the
/// device copies the training forward reads.
fn train_step(g: &Gpu, dev: &DeviceLayer, host: &mut Layer) {
    let (_, acts) = dev.forward();
    let zeros = [host.lora_gate_up.a.len(), host.lora_gate_up.b.len(), host.lora_down.a.len(), host.lora_down.b.len()].map(|n| vec![0.0f32; n]);
    let (_, grads) = dev.backward(&host.wloss, &acts, false, &zeros);
    let lr = 0.05;
    for (i, f) in [&mut host.lora_gate_up.a, &mut host.lora_gate_up.b, &mut host.lora_down.a, &mut host.lora_down.b].into_iter().enumerate() {
        f.iter_mut().zip(&grads[i]).for_each(|(w, d)| *w -= lr * d);
        g.write_f32(&dev.factors[i], f);
    }
}

/// The serving-shaped forward of the layer: per row the routed experts' bank products in decode layout (slot
/// `row * top_k + j`), corrected by `bank` when given, gate-combined on the host. Returns the combined rows and the
/// per-slot down products.
fn serve_forward(g: &Gpu, dev: &DeviceLayer, routing: &[Vec<(usize, f32)>], bank: Option<&AdapterBank>) -> (Vec<f32>, Vec<f32>) {
    let (slots, ff, d) = (ROWS * TOP_K, FF, D);
    let ids: Vec<u32> = routing.iter().flat_map(|r| r.iter().map(|&(e, _)| e as u32)).collect();
    let idb = upload_u32(g, &ids);
    let (gu, h, out) = (g.storage((slots * 2 * ff) as u64), g.storage((slots * ff) as u64), g.storage((slots * d) as u64));
    let glu = GluIds { act: crate::moe_expert_lora::idx(g, "glu_act"), act_bwd: crate::moe_expert_lora::idx(g, "glu_act_bwd") };
    let mut steps = vec![dev.gate_up.gather(g, &dev.x, &idb, &gu, slots, TOP_K).unwrap()];
    if let Some(b) = bank {
        b.correct_experts(g, &mut steps, GATE_UP, &dev.x, TOP_K, &idb, &gu, slots);
    }
    steps.push(model::glu::glu_act_step(g, &glu, &dev.spec, slots, ff, &gu, None, &h));
    steps.push(dev.down.gather(g, &h, &idb, &out, slots, 1).unwrap());
    if let Some(b) = bank {
        b.correct_experts(g, &mut steps, DOWN, &h, 1, &idb, &out, slots);
    }
    g.submit(&[], &steps);
    let per_slot = g.read(&out, (slots * d) as usize);
    let mut y = vec![0.0f32; (ROWS * D) as usize];
    for (r, rt) in routing.iter().enumerate() {
        for (j, &(_, w)) in rt.iter().enumerate() {
            let s = r * TOP_K as usize + j;
            for o in 0..d as usize {
                y[r * d as usize + o] += w * per_slot[s * d as usize + o];
            }
        }
    }
    (y, per_slot)
}

/// Train a step, save, read back, register in a bank and serve a mixed batch: adapted rows equal the training forward,
/// the others keep the base products' bits.
#[test]
fn a_trained_stacked_adapter_round_trips_and_serves_beside_the_bank() {
    let g = device();
    let mut rng = Lcg::random();
    let (dev, mut host) = DeviceLayer::new(&g, MicroScaled::Mxfp8E4m3, &mut rng);
    train_step(&g, &dev, &mut host);
    let (train_out, _) = dev.forward();

    let dir = brain_testutil::Scratch::dir("adapter-expert-sites");
    let path = dir.path().join("experts.safetensors");
    let path = path.to_str().unwrap();
    save(&host, path);
    let adapter = read_adapter_with_experts(path).expect("read the stacked adapter back");
    assert_eq!((adapter.rank, adapter.alpha, adapter.sites.len(), adapter.experts.len()), (RANK, LORA_SCALE * RANK as f32, 0, 2));
    let st = checkpoint::st::load_safetensors(path).unwrap();
    assert_eq!(st.shapes[&format!("{GATE_UP}.lora_a")], [E as usize, RANK as usize, D as usize], "A is stored [experts, rank, in]");
    assert_eq!(st.shapes[&format!("{DOWN}.lora_b")], [E as usize, D as usize, RANK as usize], "B is stored [experts, out, rank]");
    assert_eq!(st.card().unwrap().adapter.unwrap().targets.unwrap(), ["w13", "w2"]);
    let by_name: HashMap<&str, _> = adapter.experts.iter().map(|p| (p.base.as_str(), p)).collect();
    assert_eq!((by_name[GATE_UP].a.as_slice(), by_name[GATE_UP].b.as_slice()), (host.lora_gate_up.a.as_slice(), host.lora_gate_up.b.as_slice()));
    assert_eq!((by_name[DOWN].a.as_slice(), by_name[DOWN].b.as_slice()), (host.lora_down.a.as_slice(), host.lora_down.b.as_slice()));
    assert_eq!(validate_with_experts(&adapter, &[], &expert_sites(), &[]), Ok(()));

    // The routing the training forward took, in decode layout.
    let gate = g.read(&dev.gate, (ROWS * E) as usize);
    let routing: Vec<Vec<(usize, f32)>> = (0..ROWS as usize).map(|r| (0..E as usize).filter(|&e| gate[r * E as usize + e] > 0.0).map(|e| (e, gate[r * E as usize + e])).collect()).collect();
    assert!(routing.iter().all(|r| r.len() == TOP_K as usize));

    // The operator names the expert leaves like any site; the bank's budget is what it allocates.
    let mut cfg = AdapterBankConfig::for_model(&ExpertModel, &AdapterTargets::parse("w13,w2").unwrap()).unwrap();
    assert_eq!(cfg.targets, ["w13", "w2"]);
    assert_eq!(AdapterTargets::ALL.resolve(&ExpertModel).unwrap(), ["w13", "w2"]);
    cfg.r_max = 4;
    let budget = cfg.device_bytes(&ExpertModel, ROWS);
    let mut bank = AdapterBank::new_with_experts(&g, &ExpertModel.lora_sites(), &ExpertModel.expert_sites(), cfg, ROWS, 0..usize::MAX).expect("a bank over the expert sites");
    assert_eq!(bank.bytes(), budget, "the bank allocates what its config budgets");
    let id = bank.register(VerifiedAdapter { label: "test/experts".into(), digest: "sha256:experts".into(), adapter }).expect("the adapter fits the bank");
    let mut asked = 0;
    while bank.availability(&g, id) != AdapterAvailability::Resident {
        asked += 1;
        assert!(asked < 64, "the adapter never became resident");
    }
    bank.pin(id);
    let slot = bank.slot_of(id.kv_namespace());
    // Even rows run the adapter, odd rows none.
    let row_slots: Vec<u32> = (0..ROWS).map(|r| if r % 2 == 0 { slot } else { LORA_SLOT_NONE }).collect();
    bank.write_slots(&g, &row_slots);

    let (base_y, base_slots) = serve_forward(&g, &dev, &routing, None);
    let (served_y, served_slots) = serve_forward(&g, &dev, &routing, Some(&bank));
    let d = D as usize;
    let scale = (train_out.iter().map(|v| v * v).sum::<f32>() / train_out.len() as f32).sqrt();
    for r in 0..ROWS as usize {
        let row = r * d..(r + 1) * d;
        if r % 2 == 0 {
            for (o, (s, t)) in served_y[row.clone()].iter().zip(&train_out[row.clone()]).enumerate() {
                assert!((s - t).abs() <= 1e-4 * scale + 1e-5 * t.abs(), "row {r} col {o}: served {s}, trained {t}");
            }
            assert!(served_y[row.clone()].iter().zip(&base_y[row]).any(|(a, b)| (a - b).abs() > 1e-3 * scale), "row {r}: the adapter changes the adapted row");
        } else {
            let (k, n) = (TOP_K as usize, d);
            let slots = r * k * n..(r + 1) * k * n;
            assert!(served_slots[slots.clone()].iter().zip(&base_slots[slots]).all(|(a, b)| a.to_bits() == b.to_bits()), "row {r} has no adapter: its products keep the base bits");
        }
    }
}

/// An f32 copy of the bank folds the stacked pair expert by expert.
#[test]
fn a_stacked_adapter_folds_into_an_f32_bank_expert_by_expert() {
    let g = crate::moe_expert_lora::device();
    let mut rng = Lcg::random();
    let (_dev, host) = DeviceLayer::new(&g, MicroScaled::Mxfp8E4m3, &mut rng);
    let dir = brain_testutil::Scratch::dir("adapter-expert-fold");
    let path = dir.path().join("experts.safetensors");
    let path = path.to_str().unwrap();
    save(&host, path);
    let mut base = HashMap::from([(GATE_UP.to_string(), host.gate_up.clone()), (DOWN.to_string(), host.down.clone())]);
    assert_eq!(fold_adapter_into(&mut base, path).unwrap(), (RANK, LORA_SCALE * RANK as f32));
    let r = RANK as usize;
    for (name, bank, lora) in [(GATE_UP, &host.gate_up, &host.lora_gate_up), (DOWN, &host.down, &host.lora_down)] {
        let per = lora.n * lora.k;
        for e in 0..E as usize {
            let mut want = bank[e * per..(e + 1) * per].to_vec();
            fold_delta(&mut want, &lora.a[e * r * lora.k..(e + 1) * r * lora.k], &lora.b[e * lora.n * r..(e + 1) * lora.n * r], r, LORA_SCALE);
            assert_eq!(&base[name][e * per..(e + 1) * per], want.as_slice(), "{name} expert {e}");
        }
    }
    let mut short = HashMap::from([(GATE_UP.to_string(), vec![0.0; 3]), (DOWN.to_string(), host.down.clone())]);
    let err = fold_adapter_into(&mut short, path).unwrap_err().to_string();
    assert!(err.contains(GATE_UP) && err.contains("experts"), "{err}");
    assert_eq!(short[DOWN], host.down, "a refused fold leaves the base untouched");
}

/// Tensor names with their header shapes.
type Shapes<'a> = Vec<(&'a str, Vec<u64>)>;

fn write(path: &str, rank: u32, tensors: &[(&str, Vec<u64>)]) {
    let list: Vec<(String, Vec<u64>, Vec<f32>)> = tensors.iter().map(|(n, s)| (n.to_string(), s.clone(), vec![0.5; s.iter().product::<u64>() as usize])).collect();
    let mut card = ModelCard::new("test/bad", "moe-test");
    card.adapter = Some(Adapter { kind: "lora".into(), rank: Some(rank), alpha: Some(rank as f32), ..Adapter::default() });
    checkpoint::st::save_safetensors(path, &list, &serde_json::json!({}), Some(&card)).unwrap();
}

/// Malformed stacked files, a linear reader handed a stacked file, and pairs that do not fit the expert sites or the
/// bank are refused by name.
#[test]
fn malformed_or_misdirected_stacked_adapters_are_refused_by_name() {
    let dir = brain_testutil::Scratch::dir("adapter-expert-refusals");
    let file = |n: &str| dir.path().join(n).to_str().unwrap().to_string();
    let (e, r, k, n) = (4u64, 2u64, 8u64, 6u64);
    let a = format!("{GATE_UP}.lora_a");
    let b = format!("{GATE_UP}.lora_b");
    let cases: [(&str, Shapes, &str); 4] = [
        ("mixed.safetensors", vec![(a.as_str(), vec![e, r, k]), (b.as_str(), vec![e * n, r])], "neither a linear pair"),
        ("experts.safetensors", vec![(a.as_str(), vec![e, r, k]), (b.as_str(), vec![e - 1, n, r])], "one expert count"),
        ("rank.safetensors", vec![(a.as_str(), vec![e, r + 1, k]), (b.as_str(), vec![e, n, r + 1])], "stacked expert pair needs"),
        ("rank4.safetensors", vec![(a.as_str(), vec![1, e, r, k]), (b.as_str(), vec![1, e, n, r])], "neither a linear pair"),
    ];
    for (name, tensors, why) in cases {
        let path = file(name);
        write(&path, r as u32, &tensors);
        let err = read_adapter_with_experts(&path).err().unwrap_or_else(|| panic!("{name} is refused")).to_string();
        assert!(err.contains(GATE_UP) && err.contains(why), "{name}: {err}");
    }

    // A well-formed stacked file: the linear readers refuse it instead of misreading its flat data as a matrix.
    let good = file("good.safetensors");
    write(&good, r as u32, &[(a.as_str(), vec![e, r, k]), (b.as_str(), vec![e, n, r])]);
    let err = read_adapter(&good).err().expect("a linear reader refuses a stacked pair").to_string();
    assert!(err.contains(GATE_UP) && err.contains("stacked expert pair") && err.contains("read_adapter_with_experts"), "{err}");
    let err = model::adapter::file::AdapterFile::read(&good).err().expect("the kind-general reader refuses a stacked pair").to_string();
    assert!(err.contains(GATE_UP) && err.contains("stacked expert factor"), "{err}");

    // The model's sites: a model without expert sites, and one whose bank has another expert count.
    let adapter: DeviceAdapter = read_adapter_with_experts(&good).unwrap();
    assert_eq!(validate(&adapter, &[]), Err(SiteMismatch::UnknownSite { name: GATE_UP.into() }));
    let site = ExpertSite { name: GATE_UP.into(), leaf: "w13", layer: Some(0), experts: 3, out: n as usize, inn: k as usize, top_k: 2 };
    let err = validate_with_experts(&adapter, &[], std::slice::from_ref(&site), &[]).unwrap_err();
    assert!(matches!(err, SiteMismatch::ExpertShape { expected_a: [3, 2, 8], expected_b: [3, 6, 2], got_experts: 4, .. }), "{err:?}");

    // A bank that does not target the site, and one over a bank of another expert count.
    let g = device();
    let fits = ExpertSite { experts: e as usize, ..site.clone() };
    let other = ExpertSite { name: DOWN.into(), leaf: "w2", out: k as usize, inn: n as usize, ..fits.clone() };
    let verified = VerifiedAdapter { label: "test/bad".into(), digest: "sha256:bad".into(), adapter: adapter.clone() };
    let mut untargeted = AdapterBank::new_with_experts(&g, &[], &[fits.clone(), other], AdapterBankConfig::new(1, 4, &["w2"]), 2, 0..usize::MAX).unwrap();
    assert!(matches!(untargeted.register(verified.clone()), Err(AdapterError::Incompatible(why)) if why.contains(GATE_UP) && why.contains("targets")));
    let mut wrong = AdapterBank::new_with_experts(&g, &[], &[site], AdapterBankConfig::new(1, 4, &["w13"]), 2, 0..usize::MAX).unwrap();
    assert!(matches!(wrong.register(verified), Err(AdapterError::Incompatible(why)) if why.contains(GATE_UP) && why.contains("3 experts")));
    let err = AdapterBank::new_with_experts(&g, &[], &[fits], AdapterBankConfig::new(1, 4, &["w13", "w99"]), 2, 0..usize::MAX).err().expect("an unknown target is refused");
    assert!(matches!(&err, AdapterError::BadConfig(why) if why.contains("w99") && why.contains("w13")), "{err:?}");
}
