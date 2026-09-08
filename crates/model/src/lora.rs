// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared host-side LoRA (low-rank adapter) machinery: the generic
//! `W_eff = W + (α/r)·B·A` pair - init, delta apply (plain and
//! strided-into-fused-tensor), the `dW → (dA, dB)` projection, and the Adam
//! moments - hoisted from `flux2::lora` / `s3dit::lora`, which carried it as
//! two near-verbatim copies (the next `chw_to_hwc`, per the hoist-and-migrate
//! policy). Each model keeps only what genuinely differs: its block walk
//! (which linears are targeted, fused-tensor offsets), serialization naming,
//! and its own init distribution (passed into [`Pair::new`] as a closure, so
//! existing seeds keep producing bit-identical adapters).
//!
//! [`device_adapter`] is the OTHER LoRA family in this codebase - device-side
//! param-list adapters (`.lora_a`/`.lora_b` tensors living in a `Model`'s own
//! `ParamStore`, not a host-side `Pair`) used by `qwen3`/`qwen35moe`/
//! `deepseek2`. It is a genuinely different representation from [`Pair`]
//! above (deliberately not merged into it - same reasoning as this module's
//! original split from `flux2`/`s3dit`), but its SAVE/FOLD logic was, until
//! self-improve roadmap P4, three near-verbatim copies (`qwen35moe::lora`'s
//! and `deepseek2::lora`'s own doc comments called theirs "a direct port" of
//! `qwen3::lora`'s). [`device_adapter`] ends that duplication; each of the
//! three crates' own `lora.rs` is now a thin wrapper supplying its
//! architecture's `LoraCfg`/family name.

use std::collections::HashMap;

use backend_cpu::par;

/// LoRA hyper-parameters. `alpha/rank` is the delta scale ([`LoraCfg::scale`]).
#[derive(Clone, Copy)]
pub struct LoraCfg {
    pub rank: usize,
    pub alpha: f32,
    pub seed: u64,
}

impl LoraCfg {
    pub fn new(rank: usize) -> LoraCfg {
        LoraCfg { rank, alpha: rank as f32, seed: 0 }
    }
    pub fn scale(&self) -> f32 {
        crate::adapter::TargetHp::from(*self).scale()
    }
}

/// A single linear's adapter: `A [r×in]`, `B [out×r]`, plus Adam moments.
/// Weights are public so a model's serializer can read/overwrite `a`/`b`
/// directly (`to_tensors`/`from_tensors`); the moments stay private - they
/// are reset on reload by design.
#[derive(Clone)]
pub struct Pair {
    pub out: usize,
    pub inn: usize,
    pub r: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    ma: Vec<f32>,
    va: Vec<f32>,
    mb: Vec<f32>,
    vb: Vec<f32>,
}

impl Pair {
    /// Standard LoRA init: `A` drawn from `init` (the caller's own small
    /// random distribution - kept caller-side so each model's existing seeds
    /// reproduce bit-identical adapters), `B = 0` (initial no-op).
    pub fn new(out: usize, inn: usize, r: usize, mut init: impl FnMut() -> f32) -> Pair {
        let a: Vec<f32> = (0..r * inn).map(|_| init()).collect();
        Pair {
            out,
            inn,
            r,
            a,
            b: vec![0.0; out * r],
            ma: vec![0.0; r * inn],
            va: vec![0.0; r * inn],
            mb: vec![0.0; out * r],
            vb: vec![0.0; out * r],
        }
    }

    /// A pair over ALREADY-TRAINED `A [r×in]` / `B [out×r]` weights - the
    /// read-only shape a fold needs, with the Adam moments left empty because
    /// nothing here will ever be stepped. Used by [`ExternalPair::as_pair`] so
    /// folding a third-party adapter reuses [`Pair::delta`] rather than
    /// growing a second `B·A`.
    pub fn from_ab(out: usize, inn: usize, r: usize, a: Vec<f32>, b: Vec<f32>) -> Pair {
        Pair { out, inn, r, a, b, ma: Vec::new(), va: Vec::new(), mb: Vec::new(), vb: Vec::new() }
    }

    /// LoRA-FA: drop `A`'s Adam moments, making "never train `A`" structural
    /// rather than a promise every caller of [`Self::adam_step`]/
    /// [`Self::adam_a`] has to remember to keep. `adam_a` indexes `ma`/`va`
    /// at every position `da` has (`model::lora::adam`'s `for i in
    /// 0..p.len()`), so calling it after this is an immediate
    /// index-out-of-bounds panic, not a silent no-op - "we remember not to"
    /// becomes "it cannot".
    pub fn freeze_a(&mut self) {
        self.ma = Vec::new();
        self.va = Vec::new();
    }

    /// Has [`Self::freeze_a`] been called? What a caller building a param
    /// list checks before deciding whether `A` needs Adam state at all.
    pub fn a_is_frozen(&self) -> bool {
        self.ma.is_empty() && self.va.is_empty()
    }

    /// `w += scale·B·A` in `W`'s `[out×in]` row-major layout.
    pub fn delta(&self, scale: f32, w: &mut [f32]) {
        self.delta_strided(scale, w, 0, self.inn, 0);
    }

    /// `out_buf[(row0+o)·row_stride + col0 + i] += scale·(B·A)[o,i]` - the
    /// fused-tensor fold: row slices use `row0`, a column split uses
    /// `col0`/`row_stride`.
    pub fn delta_strided(&self, scale: f32, out_buf: &mut [f32], row0: usize, row_stride: usize, col0: usize) {
        // One output row per task. The rows are disjoint and each keeps its own
        // `k` loop in ascending order, so this is bit-identical to the serial
        // walk - which matters: a LoRA run's whole point is that `apply` at
        // `B = 0` reproduces the base weights exactly.
        let span = &mut out_buf[row0 * row_stride..(row0 + self.out) * row_stride];
        par::rows_mut(span, row_stride, |o, row| {
            let brow = &self.b[o * self.r..(o + 1) * self.r];
            let wrow = &mut row[col0..col0 + self.inn];
            for (k, &bk) in brow.iter().enumerate() {
                let bok = bk * scale;
                if bok == 0.0 {
                    continue;
                }
                let arow = &self.a[k * self.inn..(k + 1) * self.inn];
                for i in 0..self.inn {
                    wrow[i] += bok * arow[i];
                }
            }
        });
    }

    /// Project the base-weight grad `dW [out×in]` to `(dA [r×in], dB [out×r])`:
    /// `dA = scale·Bᵀ·dW`, `dB = scale·dW·Aᵀ`.
    /// Both halves run over the same `dW` but are parallel on different axes,
    /// because that is what keeps each of them bit-identical to the serial
    /// walk: `dB`'s rows are independent per output row, while `dA`'s entries
    /// are a SUM over output rows, so its tasks split the `r` adapter rows and
    /// keep the `o` accumulation in ascending order. A `dA` split over `o`
    /// would reassociate that sum and move a training trajectory's last bits.
    pub fn project(&self, dw: &[f32], scale: f32) -> (Vec<f32>, Vec<f32>) {
        let mut da = vec![0.0f32; self.r * self.inn];
        let mut db = vec![0.0f32; self.out * self.r];
        par::rows_mut(&mut db, self.r, |o, dbrow| {
            let dwrow = &dw[o * self.inn..(o + 1) * self.inn];
            for (k, slot) in dbrow.iter_mut().enumerate() {
                let arow = &self.a[k * self.inn..(k + 1) * self.inn];
                let mut acc = 0.0f32;
                for i in 0..self.inn {
                    acc += dwrow[i] * arow[i];
                }
                *slot = acc * scale;
            }
        });
        par::rows_mut(&mut da, self.inn, |k, darow| {
            for o in 0..self.out {
                let bok = self.b[o * self.r + k] * scale;
                if bok == 0.0 {
                    continue;
                }
                let dwrow = &dw[o * self.inn..(o + 1) * self.inn];
                for i in 0..self.inn {
                    darow[i] += bok * dwrow[i];
                }
            }
        });
        (da, db)
    }

    /// One Adam step on `A` alone (β 0.9/0.999, eps 1e-8, no weight decay).
    /// Split out from [`Pair::adam_step`] so LoRA+ can give `A`/`B`
    /// different effective learning rates while a plain `lr_ratio == 1.0`
    /// caller reproduces today's single call bit-for-bit.
    pub fn adam_a(&mut self, da: &[f32], lr: f32, t: u64) {
        adam(&mut self.a, &mut self.ma, &mut self.va, da, lr, t);
    }

    /// One Adam step on `B` alone. See [`Pair::adam_a`].
    pub fn adam_b(&mut self, db: &[f32], lr: f32, t: u64) {
        adam(&mut self.b, &mut self.mb, &mut self.vb, db, lr, t);
    }

    /// One Adam step on `A,B` (β 0.9/0.999, eps 1e-8, no weight decay).
    pub fn adam_step(&mut self, da: &[f32], db: &[f32], lr: f32, t: u64) {
        self.adam_a(da, lr, t);
        self.adam_b(db, lr, t);
    }
}

/// One linear's adapter exactly as a THIRD-PARTY file stores it, with the
/// base tensor key it targets already resolved. See [`read_external_adapter`].
pub struct ExternalPair {
    /// The base tensor this adapts, in the model's own naming
    /// (`<stem>.weight`, the `diffusion_model.` prefix stripped).
    pub base_key: String,
    /// The adapter's own stem, for error messages.
    pub stem: String,
    pub out: usize,
    pub inn: usize,
    pub r: usize,
    /// `A [r×in]` - the "down" projection.
    pub a: Vec<f32>,
    /// `B [out×r]` - the "up" projection.
    pub b: Vec<f32>,
    /// `alpha/r`, or 1.0 when the file carries no `.alpha` tensor.
    pub alpha_mult: f32,
}

impl ExternalPair {
    /// A [`Pair`] over the same `A`/`B`, so the fold reuses the ONE `B·A`
    /// implementation ([`Pair::delta`]) rather than growing a second one.
    pub fn as_pair(&self) -> Pair {
        Pair::from_ab(self.out, self.inn, self.r, self.a.clone(), self.b.clone())
    }
}

/// The `(A-suffix, B-suffix)` spellings a third-party adapter may use, in the
/// aliases ComfyUI's own loader accepts. `A`/"down" first, `B`/"up" second.
const EXTERNAL_SUFFIXES: [(&str, &str); 3] = [
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down.weight", ".lora_up.weight"),
    (".lora.down.weight", ".lora.up.weight"),
];

/// Read a third-party (ai-toolkit / ComfyUI / diffusers) LoRA `.safetensors`
/// into per-linear [`ExternalPair`]s, resolving each to the base tensor key it
/// targets.
///
/// Key matching is ComfyUI's: strip a leading `diffusion_model.`, strip the
/// `.lora_{A,B}.weight` suffix, and the remaining stem plus `.weight` is the
/// base tensor. `.alpha`, when present, is read as the scalar it is.
///
/// **Every key must be understood.** An unrecognised name, a half pair, or a
/// mismatched `r` is an error naming the tensor - never a skip. A loader that
/// quietly drops keys returns base-model output that looks like a successful
/// adapted run, which is the single worst outcome for this feature.
pub fn read_external_adapter(path: &str) -> Result<Vec<ExternalPair>, String> {
    let tensors = checkpoint::safetensors::read(path)?;
    let mut a: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut b: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut alpha: HashMap<String, f32> = HashMap::new();
    for t in tensors {
        let name = t.name.as_str();
        // `__metadata__` never reaches here (the reader drops it); anything
        // else that is not an adapter key means we do not understand the file.
        let mut matched = false;
        for (sa, sb) in EXTERNAL_SUFFIXES {
            if let Some(stem) = name.strip_suffix(sa) {
                a.insert(stem.to_string(), (t.shape.clone(), t.data.clone()));
                matched = true;
                break;
            }
            if let Some(stem) = name.strip_suffix(sb) {
                b.insert(stem.to_string(), (t.shape.clone(), t.data.clone()));
                matched = true;
                break;
            }
        }
        if !matched {
            if let Some(stem) = name.strip_suffix(".alpha") {
                let v = t.data.first().copied().ok_or_else(|| {
                    format!("lora {path}: '{name}' is an empty alpha scalar")
                })?;
                alpha.insert(stem.to_string(), v);
                continue;
            }
            return Err(format!(
                "lora {path}: unrecognised tensor '{name}' (expected a \
                 .lora_A/.lora_B, .lora_down/.lora_up or .alpha key)"
            ));
        }
    }
    if a.is_empty() {
        return Err(format!("lora {path}: no LoRA pairs found in this file"));
    }
    let mut stems: Vec<String> = a.keys().cloned().collect();
    stems.sort();
    for stem in b.keys() {
        if !a.contains_key(stem) {
            return Err(format!("lora {path}: '{stem}' has an up/B half but no down/A half"));
        }
    }
    let mut out = Vec::with_capacity(stems.len());
    for stem in stems {
        let (ashape, adata) = a.remove(&stem).expect("stem came from a");
        let (bshape, bdata) = b
            .remove(&stem)
            .ok_or_else(|| format!("lora {path}: '{stem}' has a down/A half but no up/B half"))?;
        if ashape.len() != 2 || bshape.len() != 2 {
            return Err(format!(
                "lora {path}: '{stem}' is {ashape:?}/{bshape:?}, expected two 2-D matrices"
            ));
        }
        let (r, inn) = (ashape[0], ashape[1]);
        let (o, rb) = (bshape[0], bshape[1]);
        if r != rb {
            return Err(format!(
                "lora {path}: '{stem}' rank disagrees - A is {ashape:?} (r={r}), B is {bshape:?} (r={rb})"
            ));
        }
        if adata.len() != r * inn || bdata.len() != o * r {
            return Err(format!("lora {path}: '{stem}' tensor data does not match its shape"));
        }
        let base_key =
            format!("{}.weight", stem.strip_prefix("diffusion_model.").unwrap_or(&stem));
        out.push(ExternalPair {
            base_key,
            stem: stem.clone(),
            out: o,
            inn,
            r,
            a: adata,
            b: bdata,
            // Both references resolve an absent `.alpha` to a multiplier of
            // exactly 1.0: ai-toolkit writes alpha == rank (so alpha/r == 1)
            // and strips the key on PEFT-format saves; ComfyUI's adapter uses
            // `alpha = 1.0` outright when the tensor is missing.
            alpha_mult: alpha.get(&stem).map(|al| al / r as f32).unwrap_or(1.0),
        });
    }
    Ok(out)
}

/// In-place bias-corrected Adam (β 0.9/0.999, eps 1e-8, no weight decay).
pub fn adam(p: &mut [f32], m: &mut [f32], v: &mut [f32], g: &[f32], lr: f32, t: u64) {
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let bc1 = 1.0 - b1.powi(t as i32);
    let bc2 = 1.0 - b2.powi(t as i32);
    for i in 0..p.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        p[i] -= lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps);
    }
}

/// Project `dw` onto `p`'s adapter grads and Adam-step them - the one-liner
/// every per-linear walk calls.
pub fn proj_step(p: &mut Pair, dw: &[f32], scale: f32, lr: f32, t: u64) {
    let (da, db) = p.project(dw, scale);
    p.adam_step(&da, &db, lr, t);
}

/// A cheap deterministic standard-normal (xorshift + Box–Muller half) - the
/// init distribution `s3dit::lora` seeds `A` with.
pub fn randn(s: &mut u64) -> f64 {
    let mut nx = || {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 11) as f64 / (1u64 << 53) as f64).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    };
    let (u1, u2) = (nx(), nx());
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Device-side param-list LoRA adapters - see this module's own top doc
/// comment on why this is a separate family from [`Pair`] above, not a
/// unification of the two representations. `qwen3`/`qwen35moe`/`deepseek2`
/// each keep their own `LoraCfg` type (rank/alpha/targets, plus an
/// architecture-specific `targets_leaf` matcher `param_list()` consults -
/// genuinely different per architecture, not folded in here) and their own
/// public `save_adapter`/`fold_adapter_into` signatures; each is now a thin
/// wrapper over this module's generic versions, so the actual save/fold I/O
/// and the `fold_delta` math exist exactly once.
pub mod device_adapter {
    use std::collections::HashMap;

    use checkpoint::st::{Adapter, ModelCard};

    use crate::Model;

    /// Write only `model`'s `.lora_a`/`.lora_b` tensors - never the frozen
    /// base - to `path`, carrying a `ModelCard` with `variant_of: base_id`
    /// and an `Adapter` descriptor, so the adapter is discoverable and
    /// reloadable without the base's shape being re-derived by guesswork.
    /// `family` is the `ModelCard`'s architecture tag (e.g. `"qwen"`,
    /// `"qwen35"`, `"deepseekv2"`).
    pub fn save_adapter<M: Model>(
        path: &str,
        model: &M,
        rank: u32,
        alpha: f32,
        targets: &[String],
        card_id: &str,
        base_id: &str,
        family: &str,
        dataset_id: Option<&str>,
    ) -> std::io::Result<()> {
        let r = rank as usize;
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = model
            .param_names()
            .into_iter()
            .filter(|name| crate::adapter::device::is_adapter_param(name))
            .map(|name| {
                let data = model.read_weight(&name);
                // Real 2-D shape, not the flattened `[len]` this used to
                // write - recovered from `data.len()`/`rank` exactly like
                // `read_external_adapter` already does for third-party
                // files, since the device family has no other place a
                // shape could come from. `.lora_a` is `[r,in]`, `.lora_b`
                // is `[out,r]`.
                let shape = if name.ends_with(".lora_a") { vec![r as u64, (data.len() / r) as u64] } else { vec![(data.len() / r) as u64, r as u64] };
                (name.clone(), shape, data)
            })
            .collect();
        assert!(!tensors.is_empty(), "save_adapter: no .lora_a/.lora_b tensors in the param store");

        let mut card = ModelCard::new(card_id, family);
        card.variant_of = Some(base_id.to_string());
        card.adapter = Some(Adapter {
            kind: "lora".to_string(),
            rank: Some(rank),
            base: Some(base_id.to_string()),
            alpha: Some(alpha),
            targets: Some(targets.to_vec()),
            dataset_id: dataset_id.map(str::to_string),
            per_target: None,
        });

        let config = serde_json::json!({ "rank": rank, "alpha": alpha, "targets": targets });
        checkpoint::st::save_safetensors(path, &tensors, &config, Some(&card))
    }

    /// Fold an adapter saved by [`save_adapter`] into a base model's host
    /// tensor map (name -> row-major `[out, in]` data), in place. `base`
    /// must already contain every targeted linear's weight under its plain
    /// name; this only reads the `.lora_a`/`.lora_b` pair alongside it and
    /// adds the low-rank delta. Returns `(rank, alpha)` read back from the
    /// adapter's own `ModelCard` - the fold itself needs no `targets` (only
    /// the tensor names actually present in the file), so callers build
    /// their own crate-local `LoraCfg` from this plus whatever `targets`
    /// they already know.
    pub fn fold_adapter_into(base: &mut HashMap<String, Vec<f32>>, adapter_path: &str) -> std::io::Result<(u32, f32)> {
        let st = checkpoint::st::load_safetensors(adapter_path)?;
        let card = st
            .card()
            .unwrap_or_else(|| panic!("fold_adapter_into: {adapter_path} has no ModelCard"));
        let a = card.adapter.as_ref().unwrap_or_else(|| panic!("fold_adapter_into: {adapter_path}'s card has no adapter descriptor"));
        // `Adapter.kind` is a free-form string nothing branched on before
        // this - make it load-bearing: an unknown kind is a hard error
        // naming it, not a silent "treat everything as lora" (this fold's
        // math IS the lora fold; a DoRA/LoKr file would need a different
        // one, and getting that wrong silently would be the exact
        // "loader that quietly drops keys" failure `read_external_adapter`
        // already refuses to allow for third-party files).
        assert_eq!(a.kind, "lora", "fold_adapter_into: {adapter_path}'s adapter kind is {:?}, but this fold only implements \"lora\"", a.kind);
        let rank = a.rank.unwrap_or_else(|| panic!("fold_adapter_into: {adapter_path}'s adapter has no rank"));
        let alpha = a.alpha.unwrap_or(rank as f32);
        let scale = alpha / rank as f32;

        let mut names: Vec<&str> = st
            .tensors
            .keys()
            .filter_map(|n| n.strip_suffix(".lora_a"))
            .collect();
        names.sort();
        for base_name in names {
            let a_name = format!("{base_name}.lora_a");
            let b_name = format!("{base_name}.lora_b");
            let a_data = st.tensors.get(&a_name).unwrap_or_else(|| panic!("{adapter_path}: missing {a_name}"));
            let b_data = st.tensors.get(&b_name).unwrap_or_else(|| panic!("{adapter_path}: missing {b_name}"));
            let w = base
                .get_mut(base_name)
                .unwrap_or_else(|| panic!("fold_adapter_into: base has no weight named {base_name}"));
            fold_delta(w, a_data, b_data, rank as usize, scale);
        }

        Ok((rank, alpha))
    }

    /// `W[o,i] += scale * sum_k B[o,k] * A[k,i]`, `A` is `[r,in]`, `B` is
    /// `[out,r]`, both row-major - the convention every device-adapter
    /// model's unfolded LoRA forward computes.
    fn fold_delta(w: &mut [f32], a: &[f32], b: &[f32], r: usize, scale: f32) {
        let inn = a.len() / r;
        let out = b.len() / r;
        assert_eq!(w.len(), out * inn, "fold_delta: base weight shape does not match adapter rank/dims");
        for o in 0..out {
            let brow = &b[o * r..o * r + r];
            let wrow = &mut w[o * inn..o * inn + inn];
            for (k, &bok) in brow.iter().enumerate() {
                if bok == 0.0 {
                    continue;
                }
                let bok = bok * scale;
                let arow = &a[k * inn..k * inn + inn];
                for i in 0..inn {
                    wrow[i] += bok * arow[i];
                }
            }
        }
    }
}

/// [`Pair`]'s gradient pair, as consumed by [`crate::adapter::AdapterKind::step`].
pub struct LoraGrads {
    pub da: Vec<f32>,
    pub db: Vec<f32>,
}

/// [`Pair`] wearing the generic [`crate::adapter::AdapterKind`] seam. `Pair`
/// itself is untouched by this wrapper - every method below calls straight
/// through to it, so every existing seed reproduces bit-identical adapters
/// and the documented parallel-split axis contract on [`Pair::project`]
/// (dB splits on output rows, dA splits on the rank axis) is preserved by
/// construction rather than re-derived.
pub struct LoraPair {
    pair: Pair,
    spec: crate::adapter::TargetSpec,
    hp: crate::adapter::TargetHp,
}

impl LoraPair {
    /// The underlying [`Pair`], e.g. for a caller that still wants the
    /// non-generic `delta`/`delta_strided`/`project` directly.
    pub fn pair(&self) -> &Pair {
        &self.pair
    }

    /// The underlying [`Pair`], mutably - e.g. for a device trainer that
    /// uploads/downloads `A`/`B` directly rather than through
    /// [`crate::adapter::AdapterKind`]'s `delta_into`/`project`/`step`.
    pub fn pair_mut(&mut self) -> &mut Pair {
        &mut self.pair
    }
}

impl crate::adapter::AdapterKind for LoraPair {
    type Grads = LoraGrads;

    fn kind_name() -> &'static str {
        "lora"
    }

    fn param_suffixes(hp: &crate::adapter::TargetHp, spec: &crate::adapter::TargetSpec) -> Vec<(&'static str, Vec<usize>)> {
        vec![(".lora_a", vec![hp.rank, spec.inn]), (".lora_b", vec![spec.out, hp.rank])]
    }

    fn new(spec: crate::adapter::TargetSpec, hp: crate::adapter::TargetHp, init: &mut dyn FnMut() -> f32) -> LoraPair {
        let mut pair = Pair::new(spec.out, spec.inn, hp.rank, init);
        if hp.freeze_a {
            pair.freeze_a();
        }
        LoraPair { pair, spec, hp }
    }

    fn spec(&self) -> crate::adapter::TargetSpec {
        self.spec
    }

    fn hp(&self) -> &crate::adapter::TargetHp {
        &self.hp
    }

    fn delta_into(&self, strength: f32, dst: &mut [f32]) {
        self.pair.delta_strided(self.hp.scale() * strength, dst, self.spec.row0, self.spec.row_stride, self.spec.col0);
    }

    fn project(&self, dw: &[f32]) -> LoraGrads {
        let (da, db) = self.pair.project(dw, self.hp.scale());
        LoraGrads { da, db }
    }

    fn step(&mut self, g: &LoraGrads, lr: f32, t: u64) {
        if !self.hp.freeze_a {
            self.pair.adam_a(&g.da, lr, t);
        }
        self.pair.adam_b(&g.db, lr * self.hp.lr_ratio, t);
    }

    fn to_tensors(&self) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
        vec![
            (".lora_a", vec![self.pair.r, self.pair.inn], self.pair.a.clone()),
            (".lora_b", vec![self.pair.out, self.pair.r], self.pair.b.clone()),
        ]
    }

    fn load_tensors(&mut self, get: &dyn Fn(&str) -> Option<(Vec<usize>, Vec<f32>)>) -> Result<(), String> {
        // An EMPTY shape means the source genuinely does not carry one (e.g.
        // ltxv's checkpoint round-trip loses shape - `crate::checkpoint`'s
        // safetensors path is the source of truth there, not this adapter),
        // not "shape [0]" - fall back to a length check in that case rather
        // than reject every such reader. A non-empty, WRONG shape is still a
        // hard error: wan's own reason (an A/B swap is length-compatible on
        // square targets) is exactly why a real shape must be checked when
        // one is available.
        if let Some((shape, data)) = get(".lora_a") {
            let want = [self.pair.r, self.pair.inn];
            if !shape.is_empty() && shape != want {
                return Err(format!("LoraPair::load_tensors: .lora_a shape {shape:?} does not match {want:?}"));
            }
            if data.len() != want[0] * want[1] {
                return Err(format!("LoraPair::load_tensors: .lora_a has {} elements, expected {}", data.len(), want[0] * want[1]));
            }
            self.pair.a = data;
        }
        if let Some((shape, data)) = get(".lora_b") {
            let want = [self.pair.out, self.pair.r];
            if !shape.is_empty() && shape != want {
                return Err(format!("LoraPair::load_tensors: .lora_b shape {shape:?} does not match {want:?}"));
            }
            if data.len() != want[0] * want[1] {
                return Err(format!("LoraPair::load_tensors: .lora_b has {} elements, expected {}", data.len(), want[0] * want[1]));
            }
            self.pair.b = data;
        }
        Ok(())
    }
}
