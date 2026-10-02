// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Gated-DeltaNet value-head order llama.cpp stores, undone.
//!
//! Swedish Embedded AB implements checkpoint conversion and loading for LLM
//! serving for its clients, where a silently permuted tensor produces fluent
//! but wrong text. If your team needs expertise in importing third-party
//! quantised checkpoints without losing their meaning then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! `convert_hf_to_gguf.py` re-orders the **value heads** of every Gated
//! DeltaNet layer of a model whose value-head count is a multiple of its
//! key-head count (Qwen3.5 / Qwen3.6 / Qwen3.8, dense and MoE alike). The
//! reference checkpoint indexes value head `h` as `h = k * group + g` (key head
//! `k` outer and slow, repeat index `g` inner and fast - `repeat_interleave`);
//! the GGUF stores it at `g * num_k_heads + k` ("group-major"). Every leaf that
//! is indexed by value head therefore needs the same permutation on read, and
//! EIGHT leaves are:
//!
//! | leaf | axis permuted |
//! |---|---|
//! | `A_log` (llama.cpp's `ssm_a`, which also holds `-exp(A_log)`), `dt_bias` | the vector |
//! | `conv1d.weight` | the v-channel rows after the q|k prefix |
//! | `in_proj_qkv.weight` | the v-rows after the q|k prefix |
//! | `in_proj_z.weight` | head blocks of rows |
//! | `in_proj_a.weight`, `in_proj_b.weight` | one row per head |
//! | `out_proj.weight` | head blocks of COLUMNS |
//!
//! Skipping it is invisible to every structural check - names, shapes and
//! value ranges all match - and only changes which head's state a head's own
//! decay, bias and projection act on. Measured on the real Qwen3.6-35B-A3B
//! Q8_0 GGUF against the bf16 checkpoint, layer 0, before the fix: cosine 0.30
//! (`A_log`), 0.01 (`dt_bias`), 0.82 (`conv1d`), 0.54 (`in_proj_qkv`), 0.08
//! (`in_proj_z`), 0.25/0.14 (`in_proj_a/b`), 0.06 (`out_proj`); after it every
//! leaf is at or above 0.9996 (the unquantised ones exactly 1).
//!
//! The one definition lives here so the offline converter ([`ElemOp::
//! GdnVHeads`] in a [`crate::Mapped::Transformed`]) and the streaming
//! resident loader ([`GdnFixSource`]) cannot disagree about which leaves are
//! reordered or how.

use std::borrow::Cow;

use checkpoint::gguf::BlockLayout;
use checkpoint::TensorSource;

use crate::import::ElemOp;

/// The eight value-head-indexed leaves a Gated-DeltaNet layer owns, named by
/// brain's own parameter suffix (`blocks.{l}.linear_attn.{leaf}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnLeaf {
    /// `A_log`. Also undoes llama.cpp's `-exp(A_log)` ([`ElemOp::LnNeg`]).
    ALog,
    DtBias,
    Conv1d,
    InProjQkv,
    InProjA,
    InProjB,
    InProjZ,
    OutProj,
}

impl GdnLeaf {
    /// The leaf a brain parameter name denotes, if it is one of the eight.
    pub fn of(brain_name: &str) -> Option<GdnLeaf> {
        const LEAVES: [(&str, GdnLeaf); 8] = [
            ("linear_attn.A_log", GdnLeaf::ALog),
            ("linear_attn.dt_bias", GdnLeaf::DtBias),
            ("linear_attn.conv1d.weight", GdnLeaf::Conv1d),
            ("linear_attn.in_proj_qkv.weight", GdnLeaf::InProjQkv),
            ("linear_attn.in_proj_a.weight", GdnLeaf::InProjA),
            ("linear_attn.in_proj_b.weight", GdnLeaf::InProjB),
            ("linear_attn.in_proj_z.weight", GdnLeaf::InProjZ),
            ("linear_attn.out_proj.weight", GdnLeaf::OutProj),
        ];
        LEAVES.iter().find(|(suffix, _)| brain_name.ends_with(suffix)).map(|&(_, leaf)| leaf)
    }
}

/// The value-head geometry the permutation needs, read once from a model's
/// config.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnHeadOrder {
    pub num_k_heads: usize,
    /// Value heads per key head (`num_v_heads / num_k_heads`).
    pub group: usize,
    /// `linear_value_head_dim` - rows/columns per value head in the
    /// row/column-block leaves.
    pub head_dim: usize,
    /// `num_k_heads * linear_key_head_dim`: the q|k prefix width `conv1d` and
    /// `in_proj_qkv` carry before their v-portion begins (`2 * key_dim` rows).
    pub key_dim: usize,
    /// Row width of `in_proj_qkv`/`in_proj_z`/`in_proj_a`/`in_proj_b`.
    pub d_model: usize,
    /// Causal-conv kernel width: `conv1d.weight`'s row width.
    pub conv_kernel: usize,
}

impl GdnHeadOrder {
    /// `src_head` = the GROUP-MAJOR position holding SUB-MAJOR value head
    /// `h`'s data (`h = s*group+g`, stored at `g*num_k_heads+s`). The one
    /// formula every leaf-shaped method below applies at its own granularity.
    pub fn src_head(&self, h: usize) -> usize {
        let (s, g) = (h / self.group, h % self.group);
        g * self.num_k_heads + s
    }

    /// `[num_v_heads]` flat vector - `A_log`, `dt_bias`.
    pub fn degroup_heads(&self, v: &[f32]) -> Vec<f32> {
        let nvh = self.num_k_heads * self.group;
        assert_eq!(v.len(), nvh, "GdnHeadOrder::degroup_heads: length must be num_k_heads * group");
        (0..nvh).map(|h| v[self.src_head(h)]).collect()
    }

    /// Row-major `[n_rows, row_width]`, value heads occupying `head_dim`
    /// CONSECUTIVE rows each, starting at `row_offset`. `head_dim` is a
    /// parameter because `in_proj_a`/`in_proj_b` project to ONE scalar per
    /// head, not a `linear_value_head_dim`-wide block. Rows outside the
    /// v-portion (the q|k prefix, for the leaves that have one) pass through.
    pub fn degroup_rows(&self, v: &[f32], row_offset: usize, row_width: usize, head_dim: usize) -> Vec<f32> {
        let nvh = self.num_k_heads * self.group;
        let mut out = v.to_vec();
        for h in 0..nvh {
            let src_head = self.src_head(h);
            for r in 0..head_dim {
                let dst_row = row_offset + h * head_dim + r;
                let src_row = row_offset + src_head * head_dim + r;
                out[dst_row * row_width..(dst_row + 1) * row_width].copy_from_slice(&v[src_row * row_width..(src_row + 1) * row_width]);
            }
        }
        out
    }

    /// Row-major `[n_rows, total_cols]`, value heads occupying `head_dim`
    /// CONSECUTIVE columns each - `out_proj.weight`, whose input axis is the
    /// value dimension.
    pub fn degroup_cols(&self, v: &[f32], n_rows: usize, total_cols: usize) -> Vec<f32> {
        let nvh = self.num_k_heads * self.group;
        let mut out = v.to_vec();
        for row in 0..n_rows {
            for h in 0..nvh {
                let src_head = self.src_head(h);
                for c in 0..self.head_dim {
                    out[row * total_cols + h * self.head_dim + c] = v[row * total_cols + src_head * self.head_dim + c];
                }
            }
        }
        out
    }

    /// One leaf's fix, dispatched by its shape. The only place that decision
    /// is made, so the streaming and the offline route cannot drift.
    pub fn fix(&self, leaf: GdnLeaf, name: &str, d: &[f32]) -> Result<Vec<f32>, String> {
        let nvh = self.num_k_heads * self.group;
        let value_dim = nvh * self.head_dim;
        Ok(match leaf {
            GdnLeaf::ALog => ElemOp::LnNeg.applied(name, &self.degroup_heads(d))?,
            GdnLeaf::DtBias => self.degroup_heads(d),
            GdnLeaf::Conv1d => self.degroup_rows(d, 2 * self.key_dim, self.conv_kernel, self.head_dim),
            GdnLeaf::InProjQkv => self.degroup_rows(d, 2 * self.key_dim, self.d_model, self.head_dim),
            GdnLeaf::InProjA | GdnLeaf::InProjB => self.degroup_rows(d, 0, self.d_model, 1),
            GdnLeaf::InProjZ => self.degroup_rows(d, 0, self.d_model, self.head_dim),
            GdnLeaf::OutProj => {
                if value_dim == 0 || !d.len().is_multiple_of(value_dim) {
                    return Err(format!("{name}: {} elements do not tile a value dimension of {value_dim}", d.len()));
                }
                self.degroup_cols(d, d.len() / value_dim, value_dim)
            }
        })
    }
}

/// A [`TensorSource`] over brain-named tensors that applies the GDN
/// value-head fix to the eight leaves that need it, on read, and passes
/// everything else through - including the zero-copy paths.
///
/// A transformed leaf is NEVER lent as `raw_words`/`raw_blocks`: a zero-copy
/// borrow would hand the caller llama.cpp's untransformed bytes and bypass the
/// fix entirely, the silent wrong-weights failure this type exists to prevent.
/// A transformed leaf is also never streamed past the transform
/// (`with_tensor_chunks` serves it whole): the largest, `in_proj_qkv`, is a few
/// hundred MB of fp32 at the real shapes.
pub struct GdnFixSource<'a> {
    inner: Box<dyn TensorSource + 'a>,
    order: GdnHeadOrder,
}

impl<'a> GdnFixSource<'a> {
    pub fn new(inner: impl TensorSource + 'a, order: GdnHeadOrder) -> GdnFixSource<'a> {
        GdnFixSource { inner: Box::new(inner), order }
    }
}

impl TensorSource for GdnFixSource<'_> {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        let Some(leaf) = GdnLeaf::of(name) else {
            return self.inner.with_tensor(name, f);
        };
        let mut fixed = None;
        let found = self.inner.with_tensor(name, &mut |d| fixed = Some(self.order.fix(leaf, name, d)));
        match (found, fixed) {
            (true, Some(Ok(v))) => {
                f(&v);
                true
            }
            (true, Some(Err(e))) => panic!("GdnFixSource: {e}"),
            _ => false,
        }
    }

    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        if GdnLeaf::of(name).is_some() {
            return None;
        }
        self.inner.raw_words(name)
    }

    fn raw_blocks(&self, name: &str) -> Option<(BlockLayout, Cow<'_, [u8]>)> {
        if GdnLeaf::of(name).is_some() {
            return None;
        }
        self.inner.raw_blocks(name)
    }

    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        if GdnLeaf::of(name).is_some() {
            return self.with_tensor(name, &mut |d| f(0, d));
        }
        self.inner.with_tensor_chunks(name, max_elems, f)
    }

    fn numel(&self, name: &str) -> Option<usize> {
        self.inner.numel(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn order(num_k_heads: usize, group: usize, head_dim: usize) -> GdnHeadOrder {
        GdnHeadOrder { num_k_heads, group, head_dim, key_dim: 0, d_model: 0, conv_kernel: 0 }
    }

    /// `degroup_heads` in isolation: `nkh=2, group=3`. Group-major
    /// `v[g*nkh+k]` -> sub-major `out[k*group+g]`.
    #[test]
    fn degroup_heads_matches_the_hand_computed_transpose() {
        let v = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]; // group-major: g0=[10,20] g1=[30,40] g2=[50,60]
        assert_eq!(order(2, 3, 0).degroup_heads(&v), vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0], "k=0's 3 repeats first, then k=1's");
    }

    /// `degroup_rows` at `row_offset=0`: head order is `[0,2,1,3]` at
    /// `nkh=2, group=2`, so head h's 2-row block reads from `src_head(h)`'s.
    #[test]
    fn degroup_rows_at_zero_offset_matches_the_hand_computed_block_permutation() {
        let o = order(2, 2, 2);
        assert_eq!([o.src_head(0), o.src_head(1), o.src_head(2), o.src_head(3)], [0, 2, 1, 3]);
        let v: Vec<f32> = (0..8).map(|r| 10.0 * r as f32).collect();
        assert_eq!(o.degroup_rows(&v, 0, 1, o.head_dim), vec![0.0, 10.0, 40.0, 50.0, 20.0, 30.0, 60.0, 70.0]);
    }

    /// A non-zero `row_offset` (the `conv1d`/`in_proj_qkv` shape): the q|k
    /// prefix passes through untouched.
    #[test]
    fn degroup_rows_leaves_the_prefix_before_row_offset_untouched() {
        let o = order(2, 2, 2);
        let v: Vec<f32> = (0..11).map(|r| 10.0 * r as f32).collect();
        let got = o.degroup_rows(&v, 3, 1, o.head_dim);
        assert_eq!(&got[0..3], &v[0..3], "the prefix must be untouched");
        assert_eq!(&got[3..], &[30.0, 40.0, 70.0, 80.0, 50.0, 60.0, 90.0, 100.0][..]);
    }

    /// `head_dim=1` (`in_proj_a`/`in_proj_b`) must ignore the struct's own
    /// `head_dim` and use the caller's.
    #[test]
    fn degroup_rows_at_head_dim_one_matches_the_hand_computed_row_permutation() {
        let v: Vec<f32> = (0..4).map(|r| 10.0 * r as f32).collect();
        assert_eq!(order(2, 2, 128).degroup_rows(&v, 0, 1, 1), vec![0.0, 20.0, 10.0, 30.0]);
    }

    /// `degroup_cols` (`out_proj`): the same permutation per column block,
    /// repeated independently for every row.
    #[test]
    fn degroup_cols_matches_the_hand_computed_block_permutation_per_row() {
        let v: Vec<f32> = vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0];
        let got = order(2, 2, 2).degroup_cols(&v, 2, 8);
        assert_eq!(&got[0..8], &[0.0, 10.0, 40.0, 50.0, 20.0, 30.0, 60.0, 70.0][..]);
        assert_eq!(&got[8..16], &[100.0, 110.0, 140.0, 150.0, 120.0, 130.0, 160.0, 170.0][..]);
    }

    #[test]
    fn exactly_the_eight_value_head_leaves_are_recognised() {
        for n in [
            "blocks.3.linear_attn.A_log",
            "blocks.3.linear_attn.dt_bias",
            "blocks.3.linear_attn.conv1d.weight",
            "blocks.3.linear_attn.in_proj_qkv.weight",
            "blocks.3.linear_attn.in_proj_a.weight",
            "blocks.3.linear_attn.in_proj_b.weight",
            "blocks.3.linear_attn.in_proj_z.weight",
            "blocks.3.linear_attn.out_proj.weight",
        ] {
            assert!(GdnLeaf::of(n).is_some(), "{n}");
        }
        for n in ["blocks.3.linear_attn.norm.weight", "blocks.3.self_attn.o_proj.weight", "blocks.3.mlp.router.weight", "tok.weight"] {
            assert!(GdnLeaf::of(n).is_none(), "{n}");
        }
    }

    /// The streaming source applies `ln(-x)` AND the degroup to `A_log`, the
    /// degroup alone to `dt_bias`, and nothing to anything else - and lends
    /// no zero-copy words for a transformed leaf.
    #[test]
    fn the_source_fixes_a_log_and_dt_bias_and_nothing_else() {
        let (nkh, group) = (2usize, 3usize);
        let a_log_target = [-5.5f32, -3.2, -1.1, -2.0, -4.0, -0.5];
        let dt_bias_target = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        // llama.cpp's on-disk layout: q = g*nkh+k holds sub-major p = k*group+g;
        // `ssm_a` additionally holds -exp(A_log).
        let regroup = |target: &[f32; 6], transform: fn(f32) -> f32| {
            let mut stored = vec![0f32; 6];
            for (p, &t) in target.iter().enumerate() {
                stored[(p % group) * nkh + p / group] = transform(t);
            }
            stored
        };
        let untouched = vec![0.25f32, -0.5, 2.0];
        let inner: HashMap<String, Vec<f32>> = [
            ("blocks.0.linear_attn.A_log".to_string(), regroup(&a_log_target, |x| -x.exp())),
            ("blocks.0.linear_attn.dt_bias".to_string(), regroup(&dt_bias_target, |x| x)),
            ("blocks.0.linear_attn.norm.weight".to_string(), untouched.clone()),
        ]
        .into_iter()
        .collect();
        let src = GdnFixSource::new(inner, order(nkh, group, 0));

        let mut a_log = Vec::new();
        assert!(src.with_tensor("blocks.0.linear_attn.A_log", &mut |d| a_log = d.to_vec()));
        for (g, want) in a_log.iter().zip(a_log_target) {
            assert!((g - want).abs() < 1e-5, "A_log must be ln(-ssm_a) AND degrouped: got {g}, want {want}");
        }
        assert!(src.raw_words("blocks.0.linear_attn.A_log").is_none(), "a transformed leaf must never be lent zero-copy");
        let mut chunked = Vec::new();
        assert!(src.with_tensor_chunks("blocks.0.linear_attn.A_log", 1, &mut |_, d| chunked.extend_from_slice(d)));
        assert_eq!(chunked, a_log, "the chunked path must deliver the transformed values too");

        let mut dt_bias = Vec::new();
        assert!(src.with_tensor("blocks.0.linear_attn.dt_bias", &mut |d| dt_bias = d.to_vec()));
        assert_eq!(dt_bias, dt_bias_target, "dt_bias is degrouped but has no value transform");
        assert!(src.raw_words("blocks.0.linear_attn.dt_bias").is_none());

        let mut other = Vec::new();
        assert!(src.with_tensor("blocks.0.linear_attn.norm.weight", &mut |d| other = d.to_vec()));
        assert_eq!(other, untouched);
        assert!(src.raw_words("blocks.0.linear_attn.norm.weight").is_some(), "an untransformed leaf keeps its zero-copy path");
        assert_eq!(src.numel("blocks.0.linear_attn.A_log"), Some(6));
    }

    /// `raw_blocks` over a REAL quantised GGUF, so the bypass it prevents is
    /// reachable: a transformed leaf must not lend its Q8_0 blocks.
    #[test]
    fn raw_blocks_never_lends_a_transformed_leafs_untransformed_blocks() {
        use checkpoint::gguf::MmapGguf;
        use checkpoint::gguf_write::{write, TensorOut};
        use checkpoint::remap::{Fetch, RemapSource};

        let block = |v: f32| checkpoint::quant::quantize_par(checkpoint::gguf::TYPE_Q8_0, &[v; 32]).unwrap();
        let path = std::env::temp_dir().join(format!("brain-gdn-order-rawblocks-{}.gguf", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        let t = |name: &str, v: f32| TensorOut { name: name.to_string(), shape: vec![32], ty: checkpoint::gguf::TYPE_Q8_0, data: block(v) };
        write(
            &path,
            &[],
            &[t("blocks.0.linear_attn.A_log", 1.0), t("blocks.0.linear_attn.dt_bias", 2.0), t("blocks.0.linear_attn.norm.weight", 3.0)],
            32,
        )
        .unwrap();
        let mg = MmapGguf::open(&path).unwrap();
        let names = ["blocks.0.linear_attn.A_log", "blocks.0.linear_attn.dt_bias", "blocks.0.linear_attn.norm.weight"];
        let plan: HashMap<String, Fetch> = names.iter().map(|n| (n.to_string(), Fetch::Whole(n.to_string()))).collect();
        let src = GdnFixSource::new(RemapSource::new(&mg, plan), order(2, 2, 2));

        assert!(src.raw_blocks("blocks.0.linear_attn.A_log").is_none(), "a block lend would bypass LnNeg and the degroup");
        assert!(src.raw_blocks("blocks.0.linear_attn.dt_bias").is_none(), "a block lend would bypass the degroup");
        assert!(src.raw_blocks("blocks.0.linear_attn.norm.weight").is_some(), "an untransformed leaf keeps its block path");
        std::fs::remove_file(&path).ok();
    }
}
