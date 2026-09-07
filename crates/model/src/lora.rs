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
//! Above the per-linear math sits the **fold** layer - [`Placement`] /
//! [`fold_placements`] (add a loaded adapter's deltas into a name-keyed host
//! tensor map) and [`fold_adapter_files`] (do that for a LIST of adapter
//! files, in order, each at its own strength). Same split as [`Pair::new`]'s:
//! the shared code owns the validation contract, the arithmetic and the
//! ordering; the caller supplies what is genuinely architecture-specific -
//! which tensors are targeted, the fused-tensor offsets, and how to load its
//! own trained-adapter container.
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

/// A `[rows, cols]` row-major matrix read straight out of an adapter file -
/// one Kronecker factor, or one factor's reconstruction. See [`Lokr`].
#[derive(Clone)]
pub struct Factor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Factor {
    fn new(shape: &[usize], data: Vec<f32>, what: &str) -> Result<Factor, String> {
        if shape.len() != 2 || shape[0] * shape[1] != data.len() {
            return Err(format!("{what} is {shape:?}, expected a 2-D matrix matching its data"));
        }
        Ok(Factor { rows: shape[0], cols: shape[1], data })
    }

    /// `self · rhs`, the reconstruction of a factor a file stored low-rank
    /// (`lokr_w1 = lokr_w1_a @ lokr_w1_b`).
    fn matmul(&self, rhs: &Factor, what: &str) -> Result<Factor, String> {
        if self.cols != rhs.rows {
            return Err(format!(
                "{what}: [{}, {}] @ [{}, {}] do not compose",
                self.rows, self.cols, rhs.rows, rhs.cols
            ));
        }
        let mut out = vec![0.0f32; self.rows * rhs.cols];
        for i in 0..self.rows {
            for k in 0..self.cols {
                let s = self.data[i * self.cols + k];
                if s == 0.0 {
                    continue;
                }
                let (r, o) = (&rhs.data[k * rhs.cols..(k + 1) * rhs.cols], &mut out[i * rhs.cols..(i + 1) * rhs.cols]);
                for j in 0..rhs.cols {
                    o[j] += s * r[j];
                }
            }
        }
        Ok(Factor { rows: self.rows, cols: rhs.cols, data: out })
    }
}

/// A **LoKr** (low-rank Kronecker) adapter for one linear:
/// `ΔW = mult · (W1 ⊗ W2)`, `[w1.rows·w2.rows, w1.cols·w2.cols]`.
///
/// Genuinely different math from LoRA, not a naming variant of it: LoRA's
/// delta is a low-rank PRODUCT `B·A`, LoKr's is a Kronecker product of two
/// small factors, which is full rank in general.
pub struct Lokr {
    pub w1: Factor,
    pub w2: Factor,
}

/// Which family a third-party file uses for one target, with that family's
/// own factors. See [`read_external_adapter`].
pub enum ExternalDelta {
    /// `ΔW = mult · B·A`, `A [r×in]` ("down"), `B [out×r]` ("up").
    Lora { r: usize, a: Vec<f32>, b: Vec<f32> },
    /// `ΔW = mult · (W1 ⊗ W2)`.
    Lokr(Lokr),
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
    /// The file's own rank for this target: LoRA's `r`, LoKr's decomposition
    /// dimension, or 0 for a LoKr whose factors are both stored full (which
    /// has no rank - see [`ExternalPair::alpha_mult`]).
    pub r: usize,
    /// The multiplier the file's own `.alpha` resolves to, ON TOP of the
    /// caller's strength.
    ///
    /// * **LoRA**: `alpha/r`, or 1.0 when the file carries no `.alpha`.
    ///   ai-toolkit writes `alpha == rank` and strips the key on PEFT-format
    ///   saves; ComfyUI's adapter uses 1.0 outright when it is missing.
    /// * **LoKr**: `alpha/dim` ONLY when a factor is stored decomposed, `dim`
    ///   being that decomposition's inner dimension; **1.0 when both factors
    ///   are stored full, whatever the file's `.alpha` says**. Both reference
    ///   implementations do exactly this (ComfyUI leaves its `dim` unset and
    ///   falls back to `alpha = 1.0`; LyCORIS overwrites `alpha` with
    ///   `lora_dim` in full-factor mode so `alpha/dim == 1`), and real
    ///   ai-toolkit LoKr files rely on it - they store a sentinel `alpha`
    ///   around 1e10 that would destroy the weights if it were honoured.
    pub alpha_mult: f32,
    /// The family and its factors.
    pub delta: ExternalDelta,
}

impl ExternalPair {
    /// `"lora"` or `"lokr"` - which family this target's delta comes from.
    pub fn family(&self) -> &'static str {
        match self.delta {
            ExternalDelta::Lora { .. } => "lora",
            ExternalDelta::Lokr(_) => "lokr",
        }
    }

    /// Add `scale · alpha_mult · ΔW` into `w`, the base tensor's row-major
    /// `[out, in]` data.
    pub fn add_delta(&self, scale: f32, w: &mut [f32]) {
        let s = scale * self.alpha_mult;
        match &self.delta {
            // Reuses the ONE `B·A` implementation ([`Pair::delta`]) rather
            // than growing a second one.
            ExternalDelta::Lora { r, a, b } => {
                Pair::from_ab(self.out, self.inn, *r, a.clone(), b.clone()).delta(s, w)
            }
            ExternalDelta::Lokr(kr) => kron_delta(kr, s, w),
        }
    }
}

/// `w[i1·r2+i2, j1·c2+j2] += scale · W1[i1,j1] · W2[i2,j2]` - the Kronecker
/// product, accumulated straight into the base tensor rather than
/// materialized. On klein-9b's `qkv` the product is `[12288, 4096]`, 50M
/// floats; building it only to add it once would double the peak for nothing.
///
/// One task per output row. The rows are disjoint and each keeps its `j1`
/// walk in ascending order, so this is bit-identical to the serial version -
/// the same property [`Pair::delta_strided`] is written for, and for the same
/// reason: a fold whose result depended on the thread count would make an
/// adapted generation irreproducible.
fn kron_delta(kr: &Lokr, scale: f32, w: &mut [f32]) {
    let (r1, c1) = (kr.w1.rows, kr.w1.cols);
    let (r2, c2) = (kr.w2.rows, kr.w2.cols);
    let stride = c1 * c2;
    debug_assert_eq!(w.len(), r1 * r2 * stride);
    par::rows_mut(w, stride, |o, row| {
        let (i1, i2) = (o / r2, o % r2);
        let w2row = &kr.w2.data[i2 * c2..(i2 + 1) * c2];
        for j1 in 0..c1 {
            let s = kr.w1.data[i1 * c1 + j1] * scale;
            if s == 0.0 {
                continue;
            }
            let chunk = &mut row[j1 * c2..(j1 + 1) * c2];
            for j2 in 0..c2 {
                chunk[j2] += s * w2row[j2];
            }
        }
    });
}

/// The `(A-suffix, B-suffix)` spellings a third-party adapter may use, in the
/// aliases ComfyUI's own loader accepts. `A`/"down" first, `B`/"up" second.
const EXTERNAL_SUFFIXES: [(&str, &str); 3] = [
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down.weight", ".lora_up.weight"),
    (".lora.down.weight", ".lora.up.weight"),
];

/// The LyCORIS / kohya-ss LoKr factor spellings, per target stem. A factor is
/// either stored whole (`lokr_w1`) or as its own low-rank pair
/// (`lokr_w1_a` @ `lokr_w1_b`), and a single file may do it differently for
/// `w1` than for `w2`.
/// `.lokr_t2` is listed so it is REFUSED by name rather than falling through
/// to "unrecognised tensor": it is the convolutional CP-decomposition factor,
/// a shape none of this workspace's LoKr targets (all linears) can be, and
/// folding the rest of such a file would half-apply the adapter.
const LOKR_SUFFIXES: [&str; 7] =
    [".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2"];

/// Read a third-party (ai-toolkit / ComfyUI / diffusers / LyCORIS)
/// `.safetensors` adapter into per-linear [`ExternalPair`]s, resolving each to
/// the base tensor key it targets.
///
/// Key matching is ComfyUI's: strip a leading `diffusion_model.`, strip the
/// family suffix, and the remaining stem plus `.weight` is the base tensor.
/// `.alpha`, when present, is read as the scalar it is.
///
/// **Two families, decided per target stem**, because one file adapts every
/// linear the same way but a STACK may mix files:
/// * **LoRA** - `.lora_A/.lora_B` (and the `.lora_down/.lora_up`,
///   `.lora.down/.lora.up` aliases). `ΔW = (α/r)·B·A`.
/// * **LoKr** - `.lokr_w1`/`.lokr_w2`, either of which may instead be stored
///   as its own low-rank pair (`.lokr_w1_a` @ `.lokr_w1_b`).
///   `ΔW = mult·(W1 ⊗ W2)`. See [`ExternalPair::alpha_mult`] for the scale
///   convention, which is NOT LoRA's.
///
/// **Every key must be understood.** An unrecognised name, a half pair, a
/// mismatched `r`, or a stem carrying both families is an error naming the
/// tensor - never a skip. A loader that quietly drops keys returns base-model
/// output that looks like a successful adapted run, which is the single worst
/// outcome for this feature.
pub fn read_external_adapter(path: &str) -> Result<Vec<ExternalPair>, String> {
    let tensors = checkpoint::safetensors::read(path)?;
    let mut a: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut b: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    // stem -> suffix (without the leading dot) -> factor.
    let mut kr: HashMap<String, HashMap<&'static str, Factor>> = HashMap::new();
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
        if matched {
            continue;
        }
        // Longest first: `.lokr_w1_a` must not be read as `.lokr_w1` plus a
        // stem ending in `_a`.
        let mut best: Option<&'static str> = None;
        for s in LOKR_SUFFIXES {
            if name.ends_with(s) && best.is_none_or(|prev| s.len() > prev.len()) {
                best = Some(s);
            }
        }
        if let Some(s) = best {
            let stem = name.strip_suffix(s).expect("matched above").to_string();
            let f = Factor::new(&t.shape, t.data, &format!("lora {path}: '{name}'"))?;
            kr.entry(stem).or_default().insert(&s[1..], f);
            continue;
        }
        if let Some(stem) = name.strip_suffix(".alpha") {
            let v = t
                .data
                .first()
                .copied()
                .ok_or_else(|| format!("lora {path}: '{name}' is an empty alpha scalar"))?;
            alpha.insert(stem.to_string(), v);
            continue;
        }
        return Err(format!(
            "lora {path}: unrecognised tensor '{name}' (expected a \
             .lora_A/.lora_B, .lora_down/.lora_up, .lokr_w1/.lokr_w2 \
             (optionally _a/_b) or .alpha key)"
        ));
    }
    if a.is_empty() && kr.is_empty() {
        return Err(format!("lora {path}: no LoRA or LoKr pairs found in this file"));
    }
    for stem in b.keys() {
        if !a.contains_key(stem) {
            return Err(format!("lora {path}: '{stem}' has an up/B half but no down/A half"));
        }
    }
    for stem in kr.keys() {
        if a.contains_key(stem) || b.contains_key(stem) {
            return Err(format!(
                "lora {path}: '{stem}' carries BOTH LoRA and LoKr keys - the two are different \
                 deltas, and there is no defined way to read a target as both"
            ));
        }
    }
    let mut stems: Vec<String> = a.keys().chain(kr.keys()).cloned().collect();
    stems.sort();
    let mut out = Vec::with_capacity(stems.len());
    for stem in stems {
        let base_key = format!("{}.weight", stem.strip_prefix("diffusion_model.").unwrap_or(&stem));
        let al = alpha.get(&stem).copied();
        let pair = match kr.remove(&stem) {
            Some(factors) => read_lokr(path, &stem, base_key, factors, al)?,
            None => {
                let (ashape, adata) = a.remove(&stem).expect("stem came from a or kr");
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
                ExternalPair {
                    base_key,
                    stem: stem.clone(),
                    out: o,
                    inn,
                    r,
                    // Both references resolve an absent `.alpha` to a
                    // multiplier of exactly 1.0: ai-toolkit writes
                    // alpha == rank (so alpha/r == 1) and strips the key on
                    // PEFT-format saves; ComfyUI's adapter uses `alpha = 1.0`
                    // outright when the tensor is missing.
                    alpha_mult: al.map(|v| v / r as f32).unwrap_or(1.0),
                    delta: ExternalDelta::Lora { r, a: adata, b: bdata },
                }
            }
        };
        out.push(pair);
    }
    Ok(out)
}

/// One LoKr target: resolve `W1`/`W2` (reconstructing either from its own
/// `_a`/`_b` pair when the file stored it that way) and the scale.
///
/// The scale follows ComfyUI's `weight_adapter/lokr.py` exactly: `dim` is set
/// only by a DECOMPOSED factor's inner dimension (`w2`'s winning over `w1`'s
/// when both are decomposed, which is the order that file assigns them in),
/// and the multiplier is `alpha/dim` only when both an alpha and a `dim`
/// exist - otherwise 1.0. LyCORIS reaches the same number from the other end
/// by writing `alpha = lora_dim` in full-factor mode. See
/// [`ExternalPair::alpha_mult`].
fn read_lokr(
    path: &str,
    stem: &str,
    base_key: String,
    mut factors: HashMap<&'static str, Factor>,
    alpha: Option<f32>,
) -> Result<ExternalPair, String> {
    let mut dim = None;
    let mut take = |whole: &str, af: &str, bf: &str, dim: &mut Option<usize>| -> Result<Factor, String> {
        match factors.remove(whole) {
            Some(f) => Ok(f),
            None => {
                let (fa, fb) = (factors.remove(af), factors.remove(bf));
                match (fa, fb) {
                    (Some(fa), Some(fb)) => {
                        // ComfyUI reads the decomposition dimension off the
                        // `_b` half: `dim = w1_b.shape[0]`.
                        *dim = Some(fb.rows);
                        fa.matmul(&fb, &format!("lora {path}: '{stem}.{af}' @ '{stem}.{bf}'"))
                    }
                    _ => Err(format!(
                        "lora {path}: '{stem}' has neither a whole '{whole}' nor a complete \
                         '{af}'/'{bf}' pair"
                    )),
                }
            }
        }
    };
    let w1 = take("lokr_w1", "lokr_w1_a", "lokr_w1_b", &mut dim)?;
    let w2 = take("lokr_w2", "lokr_w2_a", "lokr_w2_b", &mut dim)?;
    if let Some((leftover, _)) = factors.into_iter().next() {
        return Err(format!("lora {path}: '{stem}.{leftover}' is a LoKr key this loader does not implement"));
    }
    Ok(ExternalPair {
        base_key,
        stem: stem.to_string(),
        out: w1.rows * w2.rows,
        inn: w1.cols * w2.cols,
        r: dim.unwrap_or(0),
        alpha_mult: match (alpha, dim) {
            (Some(al), Some(d)) => al / d as f32,
            _ => 1.0,
        },
        delta: ExternalDelta::Lokr(Lokr { w1, w2 }),
    })
}

/// A name-keyed host tensor map: `name -> (shape, row-major data)`. Every
/// architecture in this workspace already spells its own alias for exactly
/// this type (`flux2::import::Tensors`, `wan::model::Tensors`,
/// `vae::blocks::Tensors`, `s3dit::block::Tensors`, ...), so a shared fold can
/// take one map and serve all of them without any crate changing its type.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Where ONE [`Pair`]'s delta lands inside a base tensor.
///
/// Two cases, and only two, occur across this workspace's models:
/// * an UNFUSED linear - the pair covers the whole `[out, in]` tensor
///   ([`Placement::whole`]); and
/// * a FUSED matrix - a checkpoint stores several linears stacked into one
///   tensor (`flux2`'s `qkv`, `mlp.0`, `linear1`) or side by side
///   (`linear2`'s column split), and the pair owns one rectangle of it
///   ([`Placement::fused`]).
///
/// The rectangle is expressed exactly as [`Pair::delta_strided`] takes it, so
/// this type is a description of a fold, not a second implementation of one.
pub struct Placement<'a> {
    /// The base tensor this delta is added into.
    pub key: String,
    pub pair: &'a Pair,
    /// How many values the base tensor must hold IN TOTAL - the whole fused
    /// tensor's size, not this rectangle's. It is the one number that catches
    /// an adapter built for a different variant before anything is written.
    pub elems: usize,
    /// First row of the rectangle within the base tensor.
    pub row0: usize,
    /// The base tensor's row length (`pair.inn` for an unfused linear).
    pub row_stride: usize,
    /// First column of the rectangle within a row.
    pub col0: usize,
}

impl<'a> Placement<'a> {
    /// The whole of an unfused `[out, in]` tensor.
    pub fn whole(key: impl Into<String>, pair: &'a Pair) -> Placement<'a> {
        Placement { key: key.into(), pair, elems: pair.out * pair.inn, row0: 0, row_stride: pair.inn, col0: 0 }
    }

    /// One rectangle of a fused tensor holding `elems` values in rows of
    /// `row_stride`, starting at `(row0, col0)`.
    pub fn fused(key: impl Into<String>, pair: &'a Pair, elems: usize, row0: usize, row_stride: usize, col0: usize) -> Placement<'a> {
        Placement { key: key.into(), pair, elems, row0, row_stride, col0 }
    }
}

/// Add `scale·(B·A)` into `ts` for every placement, in list order.
///
/// **Every placement is validated against `ts` BEFORE anything is written**,
/// so a rejected adapter leaves the map exactly as it was rather than half
/// folded. That matters more here than the extra pass costs: the map is what a
/// model is then built from, and a half-folded map builds a model that is
/// neither the base nor the adapted one, from a call that returned an error
/// the caller may well have logged and continued past.
///
/// An absent or mis-sized target is an error naming the tensor - never a skip.
/// A loader that quietly drops a target returns base-model output from a run
/// the user believes is adapted, which is the worst outcome this layer has.
///
/// An empty list is a byte-for-byte no-op and touches nothing.
pub fn fold_placements(ts: &mut Tensors, scale: f32, ps: &[Placement<'_>]) -> Result<(), String> {
    for p in ps {
        match ts.get(&p.key) {
            None => return Err(format!("lora: base tensor {} missing", p.key)),
            Some((_, data)) if data.len() != p.elems => {
                return Err(format!("lora: {} is {} elems, adapter expects {}", p.key, data.len(), p.elems))
            }
            Some(_) => {}
        }
    }
    for p in ps {
        let w = &mut ts.get_mut(&p.key).expect("validated above").1;
        p.pair.delta_strided(scale, w, p.row0, p.row_stride, p.col0);
    }
    Ok(())
}

/// What one adapter's fold moved, for the caller to log. A run that claims to
/// be adapted should be able to say how much of the model it actually changed,
/// so a silent no-op cannot hide behind a clean exit.
#[derive(Clone, Debug, PartialEq)]
pub struct FoldReport {
    pub path: String,
    /// Which adapter family the file turned out to be: `"brain"` for the
    /// caller's own trained container, or a third-party file's own family -
    /// `"lora"`, `"lokr"`, or `"lora+lokr"` for one that uses both across
    /// different targets. Worth saying out loud: the families are different
    /// math, and "the adapter loaded" is not the same claim as "it loaded as
    /// what you think it is".
    pub family: &'static str,
    /// Adapted linears (a full-coverage FLUX.2 klein-9b adapter has 112).
    pub pairs: usize,
    /// The file's rank, or the largest one if it is not uniform. 0 for a LoKr
    /// whose factors are all stored full, which has no rank.
    pub rank: usize,
    /// The strength the delta was scaled by.
    pub strength: f32,
}

impl FoldReport {
    /// True for a third-party `.safetensors` file, false for the caller's own
    /// trained-adapter container.
    pub fn external(&self) -> bool {
        self.family != "brain"
    }
}

impl std::fmt::Display for FoldReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "folded {} adapter {} - {} linears, rank {}, strength {}", self.family, self.path, self.pairs, self.rank, self.strength)
    }
}

/// Fold a LIST of adapter files into one tensor map - the multi-adapter
/// primitive. `specs` is `(path, strength)` in the order the caller wants them
/// applied.
///
/// ## Order
///
/// Adapters fold **in list order onto the same map**: adapter *n+1* reads the
/// map adapter *n* already changed, so the fold is a composition, not a set.
/// With additive low-rank deltas the arithmetic is
/// `W' = W + Σᵢ sᵢ·(αᵢ/rᵢ)·Bᵢ·Aᵢ`, i.e. each strength multiplies only its own
/// adapter's delta. Two adapters over the same linear therefore SUM there -
/// they do not average, and the second does not replace the first - so
/// stacking a face adapter at 1.0 with a style adapter at 1.0 moves that
/// weight by the sum of both trained deltas, which is usually more than either
/// was validated at. Lowering each one's strength is the dial for that; the
/// order itself only reaches the result through float rounding, which is why
/// it is defined rather than left to a hash map's iteration.
///
/// ## Format
///
/// A `.safetensors` path is a THIRD-PARTY (ai-toolkit / ComfyUI / diffusers)
/// adapter over whole fused matrices and is handled here, by
/// [`fold_external_into`]. Anything else is the architecture's own trained
/// container and goes to `native`, which must load it and fold it at the given
/// strength, returning `(adapted linears, rank)` for the report. Only the
/// architecture knows its own block walk and fused offsets, so that half stays
/// a closure - exactly as [`Pair::new`] keeps the init distribution caller-side.
///
/// `arch` names the architecture in error messages ("...which this FLUX.2
/// variant does not have").
///
/// An empty `specs` is a byte-for-byte no-op: nothing is read, nothing is
/// loaded, and the map is not touched.
pub fn fold_adapter_files(
    ts: &mut Tensors,
    specs: &[(&str, f32)],
    arch: &str,
    mut native: impl FnMut(&str, &mut Tensors, f32) -> Result<(usize, usize), String>,
) -> Result<Vec<FoldReport>, String> {
    let mut out = Vec::with_capacity(specs.len());
    for &(path, strength) in specs {
        if path.ends_with(".safetensors") {
            let info = fold_external_into(path, ts, strength, arch)?;
            out.push(FoldReport { path: path.to_string(), family: info.family, pairs: info.pairs, rank: info.rank, strength });
        } else {
            let (pairs, rank) = native(path, ts, strength)?;
            out.push(FoldReport { path: path.to_string(), family: "brain", pairs, rank, strength });
        }
    }
    Ok(out)
}

/// What [`fold_external_into`] folded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExternalFold {
    /// Adapted linears.
    pub pairs: usize,
    /// The file's rank, or the largest one if it is not uniform. 0 for a LoKr
    /// stored with full factors, which has no rank.
    pub rank: usize,
    /// The `strength` the delta was scaled by.
    pub scale: f32,
    /// `"lora"`, `"lokr"`, or `"lora+lokr"`.
    pub family: &'static str,
}

/// Fold a THIRD-PARTY (ai-toolkit / ComfyUI / diffusers / LyCORIS)
/// `.safetensors` adapter into a base tensor map, so an unchanged generation
/// run produces adapter-conditioned output.
///
/// This is the other direction from an architecture's own trained container:
/// that one holds per-slice pairs over `q`/`k`/`v` separately, while a
/// third-party file adapts the FUSED matrices - one shared `A` for the whole
/// `qkv` - which is a strictly simpler fold, because every target is then a
/// whole tensor at offset 0.
///
/// ## Semantics, taken from the reference implementations
///
/// **LoRA**: `W += strength · (alpha/r) · B·A`, matching ComfyUI's weight
/// adapter (`comfy/weight_adapter/lora.py`: `weight += (strength * alpha) *
/// mm(mat1, mat2)` with `mat1` the up/`lora_B` and `mat2` the down/`lora_A`,
/// and `alpha = v[2]/rank` or `1.0` when no `.alpha` tensor is present) and
/// ai-toolkit's trainer (`toolkit/network_mixins.py`: `scale = alpha /
/// lora_dim`, alpha initialised to the rank and stripped from PEFT-format
/// saves). `B·A` needs no transpose: both store PyTorch `nn.Linear` weights
/// `[out, in]`, which is already brain's row-major manifest layout.
///
/// **LoKr**: `W += strength · mult · (W1 ⊗ W2)`, matching
/// `comfy/weight_adapter/lokr.py` (`weight += (strength * alpha) *
/// torch.kron(w1, w2)`), with `mult` per [`ExternalPair::alpha_mult`] - which
/// for a full-factor file is 1.0 and NOT the stored alpha.
///
/// `strength` is ComfyUI's `strength_model` - a user dial, default 1.0, NOT a
/// value read from the file.
///
/// `arch` names the architecture in the "this adapter targets a tensor you do
/// not have" message, which is the message a wrong-base-model adapter
/// produces and therefore the one that has to say which base was expected.
///
/// Every target is validated against the base map BEFORE anything is written,
/// so a rejected adapter leaves the weights untouched rather than half folded.
pub fn fold_external_into(path: &str, ts: &mut Tensors, strength: f32, arch: &str) -> Result<ExternalFold, String> {
    let pairs = read_external_adapter(path)?;
    // A key that matches nothing is a hard error naming the tensor: silently
    // skipping it would return base-model output from a run the user believes
    // is adapted. This layer can also say WHY a name is unknown - the adapter
    // is for another model - and name the adapter's own stem alongside it.
    for p in &pairs {
        match ts.get(&p.base_key) {
            None => {
                return Err(format!(
                    "lora {path}: adapter targets '{}' (from '{}'), which this {arch} variant \
                     does not have - wrong base model for this adapter?",
                    p.base_key, p.stem
                ))
            }
            Some((shape, data)) if shape.as_slice() != [p.out, p.inn] || data.len() != p.out * p.inn => {
                return Err(format!(
                    "lora {path}: '{}' is {shape:?} ({} values), but the {} adapter for it is [{}, {}]",
                    p.base_key,
                    data.len(),
                    p.family(),
                    p.out,
                    p.inn
                ))
            }
            Some(_) => {}
        }
    }
    let rank = pairs.iter().map(|p| p.r).max().unwrap_or(0);
    let family = match (pairs.iter().any(|p| p.family() == "lora"), pairs.iter().any(|p| p.family() == "lokr")) {
        (true, true) => "lora+lokr",
        (false, true) => "lokr",
        _ => "lora",
    };
    // Each target carries its own alpha multiplier and its own family's math,
    // so the deltas are added one at a time, in file order.
    for p in &pairs {
        let w = &mut ts.get_mut(&p.base_key).expect("validated above").1;
        p.add_delta(strength, w);
    }
    Ok(ExternalFold { pairs: pairs.len(), rank, scale: strength, family })
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
        // LoRA dropout needs the adapter's INPUT activation `x` (mask
        // `drop(x)` before `A·x`) - this host path's whole shape is
        // `project(dw: &[f32])`, projecting a dense `dL/dW_eff` that never
        // carries `x` at all, so there is no dW to project once masking
        // makes the adapter non-linear in a single step. A silently-ignored
        // dropout is exactly the "config field parsed but never read"
        // failure this fails loudly instead - see
        // `crate::adapter::AdapterKind`'s doc for which paths CAN apply it
        // (a device trainer with `x` at the adapter's input; none of this
        // workspace's device model crates wire a dropout config through
        // yet, tracked as open work).
        assert_eq!(
            hp.dropout, 0.0,
            "LoraPair: dropout requires an activation-aware path (this host dense-dW-projection path never sees the adapter's input x)"
        );
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
