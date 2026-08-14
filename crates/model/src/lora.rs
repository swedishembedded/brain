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
        self.alpha / self.rank as f32
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

    /// `w += scale·B·A` in `W`'s `[out×in]` row-major layout.
    pub fn delta(&self, scale: f32, w: &mut [f32]) {
        self.delta_strided(scale, w, 0, self.inn, 0);
    }

    /// `out_buf[(row0+o)·row_stride + col0 + i] += scale·(B·A)[o,i]` - the
    /// fused-tensor fold: row slices use `row0`, a column split uses
    /// `col0`/`row_stride`.
    pub fn delta_strided(&self, scale: f32, out_buf: &mut [f32], row0: usize, row_stride: usize, col0: usize) {
        for o in 0..self.out {
            let brow = &self.b[o * self.r..(o + 1) * self.r];
            let wrow = &mut out_buf[(row0 + o) * row_stride + col0..(row0 + o) * row_stride + col0 + self.inn];
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
        }
    }

    /// Project the base-weight grad `dW [out×in]` to `(dA [r×in], dB [out×r])`:
    /// `dA = scale·Bᵀ·dW`, `dB = scale·dW·Aᵀ`.
    pub fn project(&self, dw: &[f32], scale: f32) -> (Vec<f32>, Vec<f32>) {
        let mut da = vec![0.0f32; self.r * self.inn];
        let mut db = vec![0.0f32; self.out * self.r];
        for o in 0..self.out {
            let dwrow = &dw[o * self.inn..(o + 1) * self.inn];
            let brow = &self.b[o * self.r..(o + 1) * self.r];
            for k in 0..self.r {
                let arow = &self.a[k * self.inn..(k + 1) * self.inn];
                let mut acc = 0.0f32;
                for i in 0..self.inn {
                    acc += dwrow[i] * arow[i];
                }
                db[o * self.r + k] = acc * scale;
                let bok = brow[k] * scale;
                if bok != 0.0 {
                    let darow = &mut da[k * self.inn..(k + 1) * self.inn];
                    for i in 0..self.inn {
                        darow[i] += bok * dwrow[i];
                    }
                }
            }
        }
        (da, db)
    }

    /// One Adam step on `A,B` (β 0.9/0.999, eps 1e-8, no weight decay).
    pub fn adam_step(&mut self, da: &[f32], db: &[f32], lr: f32, t: u64) {
        adam(&mut self.a, &mut self.ma, &mut self.va, da, lr, t);
        adam(&mut self.b, &mut self.mb, &mut self.vb, db, lr, t);
    }
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
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = model
            .param_names()
            .into_iter()
            .filter(|name| name.ends_with(".lora_a") || name.ends_with(".lora_b"))
            .map(|name| {
                let data = model.read_weight(&name);
                (name.clone(), vec![data.len() as u64], data)
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
