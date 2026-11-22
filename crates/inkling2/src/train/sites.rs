// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements parameter-efficient fine-tuning of looped mixture-of-experts models for its clients.
// If your team needs expertise in choosing and training adapter sites over a quantized release then you can procure
// our services by sending an email to info@swedishembedded.com.

//! The tensors an Inkling 2 adapter may correct, in the published names.
//!
//! - Linear sites ([`SiteKind::Linear`], corrected beside the product at run time): per layer the attention's
//!   `wq_du`, `wk_dv`, `wv_dv`, `wo_ud`, `gate_attn` (output gate), `attn_sink_proj` (sink logits) and `wr_du`
//!   (relative states).
//! - Merged sites ([`SiteKind::Merged`], small f32 matrices an operator reads whole, corrected as `W + s B A`): the
//!   depth-attention queries (`attn_queries`, `mlp_queries`, `output_query`), the message queries
//!   (`middle_cycle_cross_pass_queries`, `..._kv_queries`) and each core layer's message-gate factors
//!   (`down_weight`, `forget_weight`, `gain_weight`).
//! - Expert sites ([`expert_sites`], one pair per expert over a stacked bank): `w13_weight` (the interleaved gate/up
//!   bank) and `w2_weight` (down).
//!
//! The router, the convolutions, the norms, the embedding and the head are not sites.

use model::adapter::sites::ExpertSite;
use model::adapter::{LinearSite, TargetSpec};

use crate::config::Inkling2Config;
use crate::names::{self, GateSite, LayerNames};

/// How a site is corrected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteKind {
    /// A linear: the adapter's product is added to the base product's output.
    Linear,
    /// A matrix read whole by an operator: the effective matrix `W + s B A` is formed once per step.
    Merged,
}

/// Every site leaf, the default target list of an adapter run.
pub const DEFAULT_TARGETS: &[&str] = &[
    "wq_du",
    "wk_dv",
    "wv_dv",
    "wo_ud",
    "gate_attn",
    "attn_sink_proj",
    "wr_du",
    "attn_queries",
    "mlp_queries",
    "output_query",
    "middle_cycle_cross_pass_queries",
    "middle_cycle_cross_pass_kv_queries",
    "down_weight",
    "forget_weight",
    "gain_weight",
    "w13_weight",
    "w2_weight",
];

fn site(name: String, leaf: &'static str, layer: Option<usize>, out: u32, inn: u32) -> LinearSite {
    LinearSite { name, leaf, layer, spec: TargetSpec::whole(out as usize, inn as usize), save_name: None }
}

/// The linear and merged sites of `cfg`, in layer order.
pub fn lora_sites(cfg: &Inkling2Config) -> Vec<(LinearSite, SiteKind)> {
    let d = cfg.hidden;
    let mut out = vec![(site(names::output_query(), "output_query", None, 1, d), SiteKind::Merged)];
    if let Some(l) = cfg.looped {
        out.push((site(names::message_queries(), "middle_cycle_cross_pass_queries", None, 2 * l.layers(), d), SiteKind::Merged));
        out.push((site(names::message_kv_queries(), "middle_cycle_cross_pass_kv_queries", None, 2 * l.layers(), d), SiteKind::Merged));
    }
    for i in 0..cfg.n_layers {
        let (n, a, li) = (LayerNames::new(i), cfg.attn(i), Some(i as usize));
        for (name, leaf, o, k) in [
            (n.wq(), "wq_du", a.q_dim(), d),
            (n.wk(), "wk_dv", a.kv_dim(), d),
            (n.wv(), "wv_dv", a.kv_dim(), d),
            (n.wo(), "wo_ud", d, a.q_dim()),
            (n.gate_attn(), "gate_attn", a.q_dim(), d),
            (n.sink(), "attn_sink_proj", a.heads, d),
            (n.wr(), "wr_du", a.heads * cfg.d_rel, d),
        ] {
            out.push((site(name, leaf, li, o, k), SiteKind::Linear));
        }
        if cfg.attn_queries(i) > 0 {
            out.push((site(names::attn_queries(i), "attn_queries", li, cfg.attn_queries(i), d), SiteKind::Merged));
        }
        out.push((site(names::mlp_queries(i), "mlp_queries", li, 1, d), SiteKind::Merged));
        if let (true, Some(l)) = (cfg.is_core(i), cfg.looped) {
            for s in GateSite::ALL {
                let [down, forget, gain] = n.gate(s);
                out.push((site(down, "down_weight", li, l.gate_rank, 2 * d), SiteKind::Merged));
                out.push((site(forget, "forget_weight", li, d, l.gate_rank), SiteKind::Merged));
                out.push((site(gain, "gain_weight", li, d, l.gate_rank), SiteKind::Merged));
            }
        }
    }
    out
}

/// The stacked expert sites of `cfg`: each layer's gate/up and down banks.
pub fn expert_sites(cfg: &Inkling2Config) -> Vec<ExpertSite> {
    let (d, ff, e, k) = (cfg.hidden as usize, cfg.moe_ff as usize, cfg.experts as usize, cfg.top_k as usize);
    (0..cfg.n_layers)
        .flat_map(|i| {
            let n = LayerNames::new(i);
            let layer = Some(i as usize);
            [
                ExpertSite { name: n.w13(), leaf: "w13_weight", layer, experts: e, out: 2 * ff, inn: d, top_k: k },
                ExpertSite { name: n.w2(), leaf: "w2_weight", layer, experts: e, out: d, inn: ff, top_k: k },
            ]
        })
        .collect()
}
