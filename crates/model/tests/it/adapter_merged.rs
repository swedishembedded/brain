// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-rank adaptation of every learned tensor of a model for its clients. If your team
// needs expertise in parameter-efficient fine-tuning beyond linear layers then you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: a merged LoRA pair (`model::adapter::merged`) forms `W + s B A` exactly, and projects a gradient of the
//! effective matrix onto its factors as the chain rule says: `dB += s dW A^T`, `dA += s B^T dW`, both added to what the
//! gradients held, `A`'s skipped when it is frozen.

use data::rng::{random_seed, Lcg};
use model::adapter::merged::{MergedLora, KERNELS};

fn pipes() -> &'static [(&'static str, &'static str)] {
    static P: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let mut v: Vec<(&str, &str)> = model::ops::kernel_list().to_vec();
        model::ops::extend_kernel_list(&mut v, KERNELS);
        v
    })
}

fn close(got: &[f32], want: &[f64], what: &str) {
    let scale = want.iter().fold(1e-6f64, |m, v| m.max(v.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((f64::from(*g) - w).abs() <= 1e-5 * scale, "{what}[{i}]: {g} vs {w}");
    }
}

#[test]
fn a_merged_pair_forms_the_effective_matrix_and_projects_its_gradient_onto_the_factors() {
    let g = gpu_core::testgpu::dev(pipes());
    let ops = model::ops::Ops::new(g.share()).unwrap();
    let mut rng = Lcg::new(random_seed());
    let (out, inn, r, s) = (5usize, 7usize, 3usize, 0.75f32);
    let (w, a, b, dw) = (rng.vec_scaled(out * inn, 1.0), rng.vec_scaled(r * inn, 1.0), rng.vec_scaled(out * r, 1.0), rng.vec_scaled(out * inn, 1.0));
    let (ga0, gb0) = (rng.vec_scaled(r * inn, 1.0), rng.vec_scaled(out * r, 1.0));
    let m = MergedLora::new(&g, out as u32, inn as u32, r as u32, s).unwrap();
    let (wb, ab, bb) = (g.storage_init("w", &w), g.storage_init("a", &a), g.storage_init("b", &b));
    let (ga, gb) = (g.storage_init("ga", &ga0), g.storage_init("gb", &gb0));
    let mut steps = Vec::new();
    m.refresh(&ops, &mut steps, &wb, &ab, &bb);
    g.submit(&[], &steps);
    let eff: Vec<f64> = (0..out * inn).map(|i| f64::from(w[i]) + f64::from(s) * (0..r).map(|k| f64::from(b[(i / inn) * r + k]) * f64::from(a[k * inn + i % inn])).sum::<f64>()).collect();
    close(&g.read(m.eff(), out * inn), &eff, "W_eff");

    g.write_f32(m.d_eff(), &dw);
    let mut steps = Vec::new();
    m.project(&ops, &mut steps, &ab, &bb, Some(&ga), &gb);
    g.submit(&[], &steps);
    let want_gb: Vec<f64> = (0..out * r).map(|i| f64::from(gb0[i]) + f64::from(s) * (0..inn).map(|c| f64::from(dw[(i / r) * inn + c]) * f64::from(a[(i % r) * inn + c])).sum::<f64>()).collect();
    let want_ga: Vec<f64> = (0..r * inn).map(|i| f64::from(ga0[i]) + f64::from(s) * (0..out).map(|o| f64::from(b[o * r + i / inn]) * f64::from(dw[o * inn + i % inn])).sum::<f64>()).collect();
    close(&g.read(&gb, out * r), &want_gb, "dB");
    close(&g.read(&ga, r * inn), &want_ga, "dA");

    // A frozen A: only B's gradient moves.
    let gb2 = g.storage_init("gb2", &gb0);
    let mut steps = Vec::new();
    m.project(&ops, &mut steps, &ab, &bb, None, &gb2);
    g.submit(&[], &steps);
    close(&g.read(&gb2, out * r), &want_gb, "dB with A frozen");
    assert!(MergedLora::new(&g, 0, 3, 2, 1.0).is_err(), "an empty pair is refused");
}

// ---- serving: a merged site in the adapter bank --------------------------------------------------------------------

use model::adapter::bank::{AdapterBank, AdapterBankConfig, AdapterError, AdapterServing, AdapterTargets, MergedRows, VerifiedAdapter};
use model::adapter::sites::LoraSites;
use model::adapter::{LinearSite, TargetSpec};
use model::dispatch::{LORA_BANK_PIPELINES, LORA_EXPERT_BANK_PIPELINES, LORA_SLOT_NONE};
use model::lora::device_adapter::{AdapterSite, DeviceAdapter};
use model::serve::{AdapterAvailability, AdapterId};

/// A `[3, 5]` bank of queries an operator reads whole (the merged site: three parts of width 5), beside one linear.
const QUERIES: &str = "layers.0.queries";
const Q_OUT: usize = 3;
const Q_IN: usize = 5;

struct QueryModel;

impl LoraSites for QueryModel {
    fn lora_sites(&self) -> Vec<LinearSite> {
        vec![LinearSite { name: "layers.0.wq.weight".into(), leaf: "wq", layer: Some(0), spec: TargetSpec::whole(4, 5), save_name: None }]
    }
    fn merged_sites(&self) -> Vec<LinearSite> {
        vec![LinearSite { name: QUERIES.into(), leaf: "queries", layer: Some(0), spec: TargetSpec::whole(Q_OUT, Q_IN), save_name: None }]
    }
}

impl AdapterServing for QueryModel {}

fn bank_device() -> gpu_core::Gpu {
    let mut v: Vec<(&str, &str)> = model::ops::kernel_list().to_vec();
    model::ops::extend_kernel_list(&mut v, &LORA_BANK_PIPELINES);
    model::ops::extend_kernel_list(&mut v, &LORA_EXPERT_BANK_PIPELINES);
    gpu_core::testgpu::dev(Box::leak(v.into_boxed_slice()))
}

fn query_adapter(rng: &mut Lcg, rank: usize, label: &str) -> (VerifiedAdapter, Vec<f32>, Vec<f32>) {
    let (a, b) = (rng.vec_scaled(rank * Q_IN, 1.0), rng.vec_scaled(Q_OUT * rank, 1.0));
    let adapter = DeviceAdapter { rank: rank as u32, alpha: 2.0 * rank as f32, sites: vec![AdapterSite { base: QUERIES.into(), a: a.clone(), b: b.clone() }], saved: Vec::new(), experts: Vec::new(), leaves: Default::default() };
    (VerifiedAdapter { label: label.into(), digest: format!("sha256:{label}"), adapter }, a, b)
}

fn resident(bank: &mut AdapterBank, g: &gpu_core::Gpu, id: AdapterId) -> u32 {
    for _ in 0..100 {
        if bank.availability(g, id) == AdapterAvailability::Resident {
            bank.pin(id);
            return bank.slot_of(id.kv_namespace());
        }
    }
    panic!("adapter never became resident");
}

/// Spec: a bank that targets a merged site (a matrix an operator reads whole) serves it as a per-slot effective matrix
/// `W + (alpha / rank) B A`: the gather a forward records hands every row its own slot's matrix (the base's for a row of
/// the base and for an adapter that does not carry the site), one shared row when every row reads the same slot, and
/// nothing when every row is the base's. Its device cost is in the bank's budget; a pair of the wrong shape, an adapter
/// over a site whose base value the model never set, and an untargeted merged site are refused by name.
#[test]
fn a_bank_serves_a_merged_site_as_each_slots_effective_matrix() {
    let g = bank_device();
    let mut rng = Lcg::new(random_seed());
    let base = rng.vec_scaled(Q_OUT * Q_IN, 1.0);
    let mut cfg = AdapterBankConfig::for_model(&QueryModel, &AdapterTargets::ALL).unwrap();
    cfg.slots = 2;
    let rows = 4u32;
    let mut bank = AdapterBank::new(&g, &QueryModel.lora_sites(), cfg.clone(), rows).unwrap();
    assert_eq!(bank.bytes(), cfg.device_bytes(&QueryModel, rows), "the budget is what the bank allocates");

    let (first, a, b) = query_adapter(&mut rng, 2, "first");
    let unset = format!("{:?}", bank.register(first.clone()).unwrap_err());
    assert!(unset.contains(QUERIES) && unset.contains("base"), "{unset}");
    bank.set_base_saved(&g, QUERIES, &base).unwrap();

    let mut wrong = first.clone();
    wrong.adapter.sites[0].a.pop();
    wrong.digest = "sha256:wrong".into();
    let err = format!("{:?}", bank.register(wrong).unwrap_err());
    assert!(err.contains(QUERIES), "{err}");

    let id = bank.register(first).unwrap();
    let other = bank.register(VerifiedAdapter { label: "linear-only".into(), digest: "sha256:linear-only".into(), adapter: DeviceAdapter { rank: 2, alpha: 2.0, sites: vec![AdapterSite { base: "layers.0.wq.weight".into(), a: vec![0.1; 10], b: vec![0.1; 8] }], saved: Vec::new(), experts: Vec::new(), leaves: Default::default() } }).unwrap();
    let (s1, s2) = (resident(&mut bank, &g, id), resident(&mut bank, &g, other));
    let mut eff = base.clone();
    model::lora::device_adapter::fold_delta(&mut eff, &a, &b, 2, 2.0);

    // Every row of the base: nothing to gather.
    assert!(matches!(bank.merged_rows(QUERIES, &[LORA_SLOT_NONE; 3]), MergedRows::Base));
    // One slot for every row: one shared matrix, its parts in order.
    let MergedRows::Rows(shared) = bank.merged_rows(QUERIES, &[s1, s1]) else { panic!("one slot reads its matrix") };
    assert_eq!(shared.len(), 1, "one shared row");
    let mut s = Vec::new();
    let got = bank.gather_merged(&g, &mut s, QUERIES, &shared, &[0, 1, 2], Q_IN as u32);
    g.submit(&[], &s);
    close(&g.read(&got, Q_OUT * Q_IN), &eff.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(), "shared effective matrix");
    // Mixed rows: query-major `[parts, rows, width]`, each row its own slot's matrix.
    let slots = [s1, LORA_SLOT_NONE, s2, s1];
    let MergedRows::Rows(per_row) = bank.merged_rows(QUERIES, &slots) else { panic!("mixed rows read their matrices") };
    let parts = [2u32, 0];
    let mut s = Vec::new();
    let buf = bank.gather_merged(&g, &mut s, QUERIES, &per_row, &parts, Q_IN as u32);
    g.submit(&[], &s);
    let got = g.read(&buf, parts.len() * slots.len() * Q_IN);
    for (p, &part) in parts.iter().enumerate() {
        for (r, &slot) in slots.iter().enumerate() {
            let m = if slot == s1 { &eff } else { &base };
            let want = &m[part as usize * Q_IN..(part as usize + 1) * Q_IN];
            close(&got[(p * slots.len() + r) * Q_IN..(p * slots.len() + r + 1) * Q_IN], &want.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(), &format!("part {part} of row {r}"));
        }
    }

    // A bank whose targets leave the merged site out refuses an adapter over it.
    let mut narrow = AdapterBankConfig::for_model(&QueryModel, &AdapterTargets::parse("wq").unwrap()).unwrap();
    narrow.slots = 1;
    let mut narrow = AdapterBank::new(&g, &QueryModel.lora_sites(), narrow, rows).unwrap();
    let err = narrow.register(query_adapter(&mut rng, 2, "untargeted").0).unwrap_err();
    assert!(matches!(&err, AdapterError::Incompatible(why) if why.contains(QUERIES)), "{err:?}");
}
