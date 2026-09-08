// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic PEFT/adapter substrate: one `AdapterKind` trait both LoRA
//! families in this tree (host-side [`crate::lora::Pair`] and the
//! device-side `.lora_a`/`.lora_b` `ParamStore` tensors) dispatch through,
//! so a new parameterization (DoRA, LoKr, LoHa, OFT, ...) is written once
//! instead of once per model crate.
//!
//! Swedish Embedded AB implements from-scratch training engines where a new
//! fine-tuning method is composed from one shared adapter seam instead of
//! forked into every model crate that wants it. If your team needs
//! expertise in parameter-efficient fine-tuning infrastructure, you can
//! procure our services by sending an email to info@swedishembedded.com.

pub mod select;

use std::collections::HashMap;

/// Where one adapter's delta lands inside a (possibly FUSED) base tensor.
/// The whole-tensor case is `row0=0, row_stride=inn, col0=0`, which is
/// exactly what [`crate::lora::Pair::delta`] already does; a row slice of a
/// fused tensor (e.g. flux2's separate `wq`/`wk`/`wv` inside one packed
/// `qkv` weight) sets `row0`/`row_stride`; a column slice (flux2's `wo`
/// sharing a row with the MLP's `w2`) sets `col0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetSpec {
    pub out: usize,
    pub inn: usize,
    pub row0: usize,
    pub row_stride: usize,
    pub col0: usize,
}

impl TargetSpec {
    pub const fn whole(out: usize, inn: usize) -> TargetSpec {
        TargetSpec { out, inn, row0: 0, row_stride: inn, col0: 0 }
    }

    pub const fn row_slice(out: usize, inn: usize, row0: usize, row_stride: usize) -> TargetSpec {
        TargetSpec { out, inn, row0, row_stride, col0: 0 }
    }

    pub const fn col_slice(out: usize, inn: usize, row_stride: usize, col0: usize) -> TargetSpec {
        TargetSpec { out, inn, row0: 0, row_stride, col0 }
    }

    /// Minimum destination buffer length this spec writes into - what a
    /// fold must validate BEFORE writing anything into any target, so a
    /// short tensor never gets a partial adapter applied.
    pub fn dest_len(&self) -> usize {
        (self.row0 + self.out) * self.row_stride
    }
}

/// Per-target hyperparameters. The ONE place the delta scale is computed -
/// every [`AdapterKind`] impl calls [`TargetHp::scale`] rather than
/// re-deriving `alpha/rank` (or `alpha/sqrt(rank)`) itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetHp {
    pub rank: usize,
    pub alpha: f32,
    /// rsLoRA: `scale = alpha/sqrt(r)` instead of `alpha/r` - keeps the
    /// update from being increasingly suppressed as rank grows.
    pub rank_stabilized: bool,
    /// LoRA dropout probability on the adapter's input (0.0 = off).
    pub dropout: f32,
    /// LoRA+: `B`'s effective lr is `lr_ratio * lr`. 1.0 = plain LoRA.
    pub lr_ratio: f32,
    /// LoRA-FA: `A` stays at its init; only `B` trains.
    pub freeze_a: bool,
}

impl TargetHp {
    pub fn new(rank: usize, alpha: f32) -> TargetHp {
        TargetHp { rank, alpha, rank_stabilized: false, dropout: 0.0, lr_ratio: 1.0, freeze_a: false }
    }

    pub fn scale(&self) -> f32 {
        if self.rank_stabilized {
            self.alpha / (self.rank as f32).sqrt()
        } else {
            self.alpha / self.rank as f32
        }
    }
}

impl From<crate::lora::LoraCfg> for TargetHp {
    fn from(cfg: crate::lora::LoraCfg) -> TargetHp {
        TargetHp::new(cfg.rank, cfg.alpha)
    }
}

/// One adaptable linear a model offers to a [`select::TargetSelector`].
/// `out`/`inn` live inside `spec`, as DATA - so a selector (and an
/// [`AdapterKind`] constructor) never has to re-derive a model's own
/// dimensions the way a hard-coded leaf table does.
#[derive(Clone, Debug)]
pub struct LinearSite {
    /// Full base tensor name a fold writes into (e.g.
    /// `"double_blocks.0.img.attn.wq.weight"`).
    pub name: String,
    /// The leaf name used for exact/suffix matching (e.g. `"wq"`).
    pub leaf: &'static str,
    /// The transformer layer/block index, if any - selectors filter on
    /// this field directly rather than parsing it back out of `name`.
    pub layer: Option<usize>,
    pub spec: TargetSpec,
}

/// A model that can enumerate the linears a [`select::TargetSelector`] may
/// target. This is the manifest-driven enumeration style
/// (`crates/supir/src/lora.rs`'s today) generalized to every model.
pub trait AdaptableLinears {
    fn linear_sites(&self) -> Vec<LinearSite>;
}

/// The on-disk key spelling for one adapter tensor pair. `Brain` is this
/// repo's own writer convention (`<name>.lora_a`/`.lora_b`); the other three
/// are the third-party spellings `read_external_adapter`'s
/// (now-superseded) `EXTERNAL_SUFFIXES` table recognized on read, plus
/// ltxv's own ComfyUI-style writer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyStyle {
    Brain,
    Peft { prefix: &'static str },
    DownUp,
    DotDownUp,
}

impl KeyStyle {
    /// Map a canonical suffix (as returned by [`AdapterKind::to_tensors`],
    /// e.g. `".lora_a"`) to this style's on-disk spelling. A suffix this
    /// style has no special spelling for passes through unchanged.
    pub fn suffix(&self, canonical: &str) -> String {
        match (self, canonical) {
            (KeyStyle::Peft { .. }, ".lora_a") => ".lora_A.weight".to_string(),
            (KeyStyle::Peft { .. }, ".lora_b") => ".lora_B.weight".to_string(),
            (KeyStyle::DownUp, ".lora_a") => ".lora_down.weight".to_string(),
            (KeyStyle::DownUp, ".lora_b") => ".lora_up.weight".to_string(),
            (KeyStyle::DotDownUp, ".lora_a") => ".lora.down.weight".to_string(),
            (KeyStyle::DotDownUp, ".lora_b") => ".lora.up.weight".to_string(),
            (_, s) => s.to_string(),
        }
    }

    pub fn prefix(&self) -> &'static str {
        match self {
            KeyStyle::Peft { prefix } => prefix,
            _ => "",
        }
    }

    /// `base_name` is the exact key a caller's base-weight map uses (so
    /// `fold_into`'s lookups can use [`LinearSite::name`] directly); every
    /// known adapter naming convention in this tree - brain's own,
    /// ComfyUI's, PEFT's - names the adapter tensor against the STEM, so a
    /// trailing `.weight` is stripped before the suffix is appended.
    pub fn format(&self, base_name: &str, canonical_suffix: &str) -> String {
        let stem = base_name.strip_suffix(".weight").unwrap_or(base_name);
        format!("{}{}{}", self.prefix(), stem, self.suffix(canonical_suffix))
    }
}

/// One parameter-efficient adapter parameterization for a single linear.
/// Implementors: [`crate::lora::LoraPair`] (LoRA, wrapping
/// [`crate::lora::Pair`] unchanged) today; DoRA/LoKr/LoHa/OFT later.
///
/// `Grads` is an associated type, NOT `dw: &[f32]`, because the device
/// trainers in this tree (flux2's `LoraAdapter::step_projected`, wan's
/// `project`/`step_projected` split) compute an adapter's gradient directly
/// from low-rank device intermediates and never materialize a dense
/// `dL/dW_eff` to project - a `step(&mut self, dw: &[f32])` signature would
/// exclude exactly those (the *default*, not exceptional) training paths.
pub trait AdapterKind: Sized + Send {
    type Grads: Sync;

    fn kind_name() -> &'static str;

    /// The extra trainable tensors this kind adds for ONE target, as
    /// `(canonical_suffix, shape)` relative to the base leaf name - e.g.
    /// LoRA's `[(".lora_a", [r,in]), (".lora_b", [out,r])]`. Both the
    /// device param list and the host serializer read this, so they cannot
    /// disagree about what a kind needs.
    fn param_suffixes(hp: &TargetHp, spec: &TargetSpec) -> Vec<(&'static str, Vec<usize>)>;

    /// Fresh adapter for one target. `init` is the CALLER's distribution,
    /// drawn in the same order [`crate::lora::Pair::new`] always has, so
    /// every existing seed keeps reproducing bit-identical adapters.
    fn new(spec: TargetSpec, hp: TargetHp, init: &mut dyn FnMut() -> f32) -> Self;

    fn spec(&self) -> TargetSpec;
    fn hp(&self) -> &TargetHp;

    /// `dst[(row0+o)*row_stride + col0 + i] += strength * scale * delta[o,i]`.
    /// `strength` is the inference dial (flux2's `fold_into_tensors_at`),
    /// multiplied ON TOP of `hp().scale()`, never replacing it.
    fn delta_into(&self, strength: f32, dst: &mut [f32]);

    /// Dense `dL/dW_eff` -> this kind's grads. Host dense-projection path
    /// only - device trainers call [`AdapterKind::step`] directly with
    /// grads they computed some other way.
    fn project(&self, dw: &[f32]) -> Self::Grads;

    /// One optimizer step at 1-based Adam counter `t`.
    fn step(&mut self, g: &Self::Grads, lr: f32, t: u64);

    /// The fused one-liner every per-linear walk calls - the exact body of
    /// [`crate::lora::proj_step`].
    fn proj_step(&mut self, dw: &[f32], lr: f32, t: u64) {
        let g = self.project(dw);
        self.step(&g, lr, t);
    }

    fn to_tensors(&self) -> Vec<(&'static str, Vec<usize>, Vec<f32>)>;

    /// `get(canonical_suffix)` returns the tensor this adapter should load
    /// for that suffix, if present in the source file.
    fn load_tensors(&mut self, get: &dyn Fn(&str) -> Option<(Vec<usize>, Vec<f32>)>) -> Result<(), String>;
}

/// The canonical-order engine every per-model `LoraAdapter`-shaped struct
/// hand-writes today: a `Vec<(LinearSite, K)>` in caller-declared order,
/// with the shared step/serialize/fold operations implemented once.
pub struct AdapterSet<K: AdapterKind> {
    entries: Vec<(LinearSite, K)>,
    key_style: KeyStyle,
    t: u64,
}

impl<K: AdapterKind> AdapterSet<K> {
    pub fn build(sites: Vec<LinearSite>, hp: TargetHp, key_style: KeyStyle, init: &mut dyn FnMut() -> f32) -> Self {
        let entries = sites.into_iter().map(|site| {
            let k = K::new(site.spec, hp, init);
            (site, k)
        }).collect();
        AdapterSet { entries, key_style, t: 0 }
    }

    pub fn from_tensors(
        sites: Vec<LinearSite>,
        hp: TargetHp,
        key_style: KeyStyle,
        src: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    ) -> Result<Self, String> {
        let mut zero = || 0.0f32;
        let mut entries = Vec::with_capacity(sites.len());
        for site in sites {
            let mut k = K::new(site.spec, hp, &mut zero);
            let name = site.name.clone();
            let get = |suffix: &str| -> Option<(Vec<usize>, Vec<f32>)> {
                src.get(&key_style.format(&name, suffix)).cloned()
            };
            k.load_tensors(&get)?;
            entries.push((site, k));
        }
        Ok(AdapterSet { entries, key_style, t: 0 })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &(LinearSite, K)> {
        self.entries.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut (LinearSite, K)> {
        self.entries.iter_mut()
    }

    /// Chunk length 1, matching flux2's `step_projected` - documented there
    /// as bit-identical to a serial walk, which this must not disturb.
    pub fn step_projected(&mut self, grads: &[K::Grads], lr: f32) {
        self.t += 1;
        let t = self.t;
        backend_cpu::par::chunks_mut(&mut self.entries, 1, |i, chunk| {
            chunk[0].1.step(&grads[i], lr, t);
        });
    }

    /// Validate every target's destination exists and is long enough
    /// BEFORE writing any delta - flux2's `fold_into_tensors` two-pass
    /// shape, made mechanical by [`TargetSpec::dest_len`].
    pub fn fold_into(&self, dst: &mut HashMap<String, (Vec<usize>, Vec<f32>)>, strength: f32) -> Result<(), String> {
        for (site, _) in &self.entries {
            let (_, data) = dst
                .get(&site.name)
                .ok_or_else(|| format!("adapter fold: missing base tensor {:?}", site.name))?;
            if data.len() < site.spec.dest_len() {
                return Err(format!(
                    "adapter fold: {:?} has {} elements, need at least {}",
                    site.name,
                    data.len(),
                    site.spec.dest_len()
                ));
            }
        }
        for (site, k) in &self.entries {
            let (_, data) = dst.get_mut(&site.name).expect("validated above");
            k.delta_into(strength, data);
        }
        Ok(())
    }

    pub fn to_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut out = Vec::new();
        for (site, k) in &self.entries {
            for (suffix, shape, data) in k.to_tensors() {
                out.push((self.key_style.format(&site.name, suffix), shape, data));
            }
        }
        out
    }
}
