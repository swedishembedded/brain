// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA (low-rank adapters) for [`crate::lmgrad`]'s host Qwen2-style LM
//! reference.
//!
//! ## Why `model::lora::Pair`, not `qwen3::lora`/`qwen3::LoraCfg`
//!
//! The plan for this workstream was to wire `crate::llm::CosyVoiceLm` to
//! `qwen3::lora` directly, the way `qwen3`'s own device-resident `Model`
//! (`Qwen::lora_fwd`/`Qwen::lora_for`) applies it during a batched forward.
//! That is not reachable here: `crate::lmgrad` is a fresh host reference (see
//! its module doc for why), not `qwen3::Qwen`'s own training graph, so there
//! is no `qwen3::Qwen` instance in this crate's training path for
//! `qwen3::lora`'s device-adapter machinery (`.lora_a`/`.lora_b` tensors in a
//! live `ParamStore`) to attach to.
//!
//! What this module reuses instead is the OTHER LoRA family already shared
//! across this workspace's host training references: `model::lora::Pair`,
//! the same `W_eff = W + (α/r)·B·A` host adapter `wan::lora`/`flux2::lora`/
//! `s3dit::lora` build on. It is the correct analogue for a host `Fp`-generic
//! reference (materialise `W_eff`, run the gradchecked backward, project
//! `dL/dW_eff` onto `(dA, dB)`) the same way `qwen3::lora` is the correct
//! analogue for a device-resident `Model` - both trace back to the same
//! `B = 0`-at-init, frozen-base convention, just wired through the shape each
//! architecture's training graph actually has.
//!
//! Targets: `wq`/`wk`/`wv`/`wo` per layer - the same four projections
//! `qwen3::LoraCfg::attn` targets by default, so a rank/alpha choice means the
//! same thing here as it does for `qwen3`-hosted models in this workspace.
//! The MLP (`gate`/`up`/`down`) and the embedding/decoder tables are left to
//! full fine-tune (see `crate::lmgrad`'s own full set of trainable tensors) -
//! LoRA's premise is a low-rank correction to a big square-ish attention
//! projection, and CosyVoice's speech vocabulary is exactly the kind of
//! architecture-specific table a low-rank update does not suit well (a
//! handful of tokens would need to move by a large amount each, which a
//! shared rank-`r` factor does not represent efficiently).

use crate::lmgrad::{Fp, LayerW, LmDims, LmGrads, LmWeights};
pub use model::lora::LoraCfg;
use model::lora::{proj_step, randn, Pair};

struct LayerLora {
    wq: Pair,
    wk: Pair,
    wv: Pair,
    wo: Pair,
}

/// A LoRA adapter over every layer of [`crate::lmgrad`]'s LM.
pub struct LmLora {
    scale: f32,
    rank: usize,
    layers: Vec<LayerLora>,
    t: u64,
}

impl LmLora {
    /// Fresh adapter sized for `d`. `B = 0`, so [`Self::apply`] returns
    /// weights bit-identical to the base - `tests/lm_overfit.rs` asserts this
    /// rather than assumes it.
    pub fn new(d: &LmDims, lc: LoraCfg) -> LmLora {
        let (dm, hq, hkv, r) = (d.d_model, d.n_heads * d.head_dim, d.n_kv_heads * d.head_dim, lc.rank);
        let mut seed = lc.seed ^ 0x434F_5359_564F_4943; // "COSYVOIC"
        let mk = |out: usize, inn: usize, seed: &mut u64| Pair::new(out, inn, r, || (randn(seed) * 0.02) as f32);
        let layers = (0..d.n_layers)
            .map(|_| LayerLora { wq: mk(hq, dm, &mut seed), wk: mk(hkv, dm, &mut seed), wv: mk(hkv, dm, &mut seed), wo: mk(dm, hq, &mut seed) })
            .collect();
        LmLora { scale: lc.scale(), rank: r, layers, t: 0 }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn scale(&self) -> f32 {
        self.scale
    }
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Effective weights `W_eff = W + scale·B·A` on `wq/wk/wv/wo`; every other
    /// tensor (embeddings, norms, MLP, decoder head) passes through frozen -
    /// base is cloned, never mutated.
    pub fn apply(&self, base: &LmWeights<f32>) -> LmWeights<f32> {
        let mut w = base.clone();
        for (a, l) in self.layers.iter().zip(w.layers.iter_mut()) {
            a.wq.delta(self.scale, &mut l.wq);
            a.wk.delta(self.scale, &mut l.wk);
            a.wv.delta(self.scale, &mut l.wv);
            a.wo.delta(self.scale, &mut l.wo);
        }
        w
    }

    /// One optimisation step: project the trainer's `dL/dW_eff` (from a
    /// forward on [`Self::apply`]'s own output) onto `(dA, dB)` per targeted
    /// linear and Adam-step them. The base itself never moves.
    pub fn step(&mut self, base_grads: &LmGrads<f32>, lr: f32) {
        self.t += 1;
        for (a, gl) in self.layers.iter_mut().zip(base_grads.layers.iter()) {
            proj_step(&mut a.wq, &gl.wq, self.scale, lr, self.t);
            proj_step(&mut a.wk, &gl.wk, self.scale, lr, self.t);
            proj_step(&mut a.wv, &gl.wv, self.scale, lr, self.t);
            proj_step(&mut a.wo, &gl.wo, self.scale, lr, self.t);
        }
    }
}

/// Sanity helper for tests: every targeted tensor's `(A, B)` shapes match the
/// base layer's own projection shapes.
pub fn shapes_match<T: Fp>(d: &LmDims, lora: &LmLora, base: &LmWeights<T>) -> bool {
    let check = |pair: &Pair, w: &Vec<T>, out: usize, inn: usize| pair.out == out && pair.inn == inn && w.len() == out * inn;
    let (dm, hq, hkv) = (d.d_model, d.n_heads * d.head_dim, d.n_kv_heads * d.head_dim);
    lora.layers.iter().zip(base.layers.iter()).all(|(a, l): (&LayerLora, &LayerW<T>)| {
        check(&a.wq, &l.wq, hq, dm) && check(&a.wk, &l.wk, hkv, dm) && check(&a.wv, &l.wv, hkv, dm) && check(&a.wo, &l.wo, dm, hq)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmgrad::{grads, init_weights, Example};

    fn tiny_example(d: &LmDims) -> Example {
        Example { text_ids: vec![2, 4, 1], special_sos: 0, special_task: if d.special_vocab > 0 { 1 } else { d.speech_vocab - 2 }, speech_tokens: vec![1, 3, 5, 2] }
    }

    #[test]
    fn lora_is_an_exact_no_op_at_init() {
        let d = LmDims::tiny();
        let base = init_weights::<f32>(&d, 7);
        let lora = LmLora::new(&d, LoraCfg::new(4));
        assert!(shapes_match(&d, &lora, &base));
        let applied = lora.apply(&base);
        assert!(applied == base, "a fresh LoRA adapter (B=0) must not change a single weight");
    }

    #[test]
    fn lora_training_descends_with_the_base_frozen() {
        let d = LmDims::tiny();
        let base = init_weights::<f32>(&d, 11);
        let mut lora = LmLora::new(&d, LoraCfg::new(4));
        let ex = tiny_example(&d);

        let (l0, _) = grads(&d, &lora.apply(&base), &ex);
        let mut last = l0;
        for _ in 0..120 {
            let w_eff = lora.apply(&base);
            let (l, g) = grads(&d, &w_eff, &ex);
            lora.step(&g, 5e-3);
            last = l;
        }
        assert!(last < l0 * 0.9, "LoRA training must descend: {l0} -> {last}");

        let base_again = init_weights::<f32>(&d, 11);
        assert!(base == base_again, "the base weights must never move during LoRA training");
    }
}
