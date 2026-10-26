// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model compression that keeps a pretrained
// model a standard checkpoint for its clients. If your team needs expertise
// in shrinking large language models for constrained hardware, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A Qwen3 decoder with its residual stream rotated into an orthonormal basis
//! and cut to the basis's first `k` directions: an ordinary Qwen3 checkpoint
//! of width `k`.
//!
//! Every map that reads the residual stream does so through an RMSNorm, and
//! an orthogonal rotation `Q` of the stream preserves its norm. Folding each
//! norm's gain `γ` into the linears it feeds (`W diag(γ)`) leaves a norm
//! without a gain, which commutes with `Q`; the rotated model is then
//! `W diag(γ) Qᵀ` for every linear reading the stream (q, k, v, gate, up and
//! the head), `Q W` for every linear writing it (o and down) and `E Qᵀ` for
//! the embedding table, and computes the same function exactly. Keeping only
//! the first `k` rows `Q_k` drops the stream's component outside their span.
//! The norm then sees `k` coordinates of a vector whose length it took over
//! `d`, so its gain becomes `sqrt(d / k)`, exact when the dropped component
//! is zero. With the principal directions of the stream as `Q`, this is
//! SliceGPT's deletion of rows and columns (Ashkboos et al., 2024) with one
//! basis for the whole stream, which keeps the result a standard checkpoint:
//! no extra map on the residual connections.
//!
//! [`Slicing`] produces the sliced tensors lazily, one per request, from any
//! [`TensorSource`], so a model is built or saved from it with host memory
//! for one tensor at a time.

use checkpoint::TensorSource;
use model::hostmath::linear_rows;

use crate::config::QwenConfig;

/// The sliced view of a Qwen3 checkpoint. See the module documentation.
pub struct Slicing<'a> {
    src: &'a dyn TensorSource,
    /// The original and the sliced config.
    cfg: QwenConfig,
    sliced: QwenConfig,
    /// `Q_k`, `[k, d]` row-major.
    basis: Vec<f32>,
}

impl<'a> Slicing<'a> {
    /// The width-`k` slice of the decoder `cfg`/`src` in the basis whose
    /// orthonormal rows are `basis` (`[rows, d]`, row-major, at least `k`
    /// rows; the first `k` are kept). Orthonormality is the caller's
    /// contract: the result is a rotation only for orthonormal rows.
    pub fn new(cfg: &QwenConfig, src: &'a dyn TensorSource, basis: &[f32], k: usize) -> Result<Slicing<'a>, String> {
        let d = cfg.d_model as usize;
        if k == 0 || k > d {
            return Err(format!("slice width {k} must be in 1..={d}"));
        }
        if !basis.len().is_multiple_of(d) || basis.len() / d < k {
            return Err(format!("a basis of {} values does not hold {k} rows of width {d}", basis.len()));
        }
        if cfg.lora.is_some() {
            return Err("slice a merged checkpoint: LoRA factors are not rotated".into());
        }
        let sliced = QwenConfig { d_model: k as u32, tie_embeddings: false, ..cfg.clone() };
        Ok(Slicing { src, cfg: cfg.clone(), sliced, basis: basis[..k * d].to_vec() })
    }

    /// The config of the sliced checkpoint: width `k`, untied head.
    pub fn config(&self) -> &QwenConfig {
        &self.sliced
    }

    /// Tensor `name` in blocks of whole rows of `row` values, at most
    /// [`ROWS_PER_BLOCK`] rows each: `f(first row, rows)`. False if absent.
    fn row_blocks(&self, name: &str, row: usize, f: &mut dyn FnMut(usize, &[f32])) -> bool {
        let mut pending: Vec<f32> = Vec::with_capacity(row * ROWS_PER_BLOCK);
        let mut first = 0;
        let found = self.src.with_tensor_chunks(name, row * ROWS_PER_BLOCK, &mut |_, chunk| {
            let mut chunk = chunk;
            while !chunk.is_empty() {
                let take = (row * ROWS_PER_BLOCK - pending.len()).min(chunk.len());
                pending.extend_from_slice(&chunk[..take]);
                chunk = &chunk[take..];
                if pending.len() == row * ROWS_PER_BLOCK {
                    f(first, &pending);
                    first += ROWS_PER_BLOCK;
                    pending.clear();
                }
            }
        });
        assert!(pending.len().is_multiple_of(row), "slice: tensor {name} is not a whole number of {row}-wide rows");
        if found && !pending.is_empty() {
            f(first, &pending);
        }
        found
    }

    /// A vector tensor, read whole: a norm gain of `d` values.
    fn gain(&self, name: &str) -> Option<Vec<f32>> {
        let mut out = Vec::new();
        self.row_blocks(name, self.cfg.d_model as usize, &mut |_, rows| out.extend_from_slice(rows)).then_some(out)
    }

    /// `W diag(γ) Q_kᵀ` for a linear `[rows, d]` reading the stream through
    /// a norm with gain `gamma` (`None` for the embedding table).
    fn reader(&self, weight: &str, gamma: Option<&[f32]>) -> Option<Vec<f32>> {
        let (d, k) = (self.cfg.d_model as usize, self.sliced.d_model as usize);
        let mut out = Vec::new();
        let found = self.row_blocks(weight, d, &mut |_, rows| {
            let mut rows = rows.to_vec();
            if let Some(gamma) = gamma {
                for row in rows.chunks_mut(d) {
                    row.iter_mut().zip(gamma).for_each(|(v, g)| *v *= g);
                }
            }
            out.extend(linear_rows(&rows, &self.basis, rows.len() / d, d, k));
        });
        found.then_some(out)
    }

    /// `Q_k W` for a linear `[d, cols]` writing the stream, accumulated over
    /// blocks of its rows: `Q_k[:, j0..j1] W[j0..j1, :]`.
    fn writer(&self, weight: &str, cols: usize) -> Option<Vec<f32>> {
        let (d, k) = (self.cfg.d_model as usize, self.sliced.d_model as usize);
        let mut out = vec![0.0f32; k * cols];
        let found = self.row_blocks(weight, cols, &mut |j0, rows| {
            let n = rows.len() / cols;
            let q: Vec<f32> = (0..k).flat_map(|i| self.basis[i * d + j0..i * d + j0 + n].iter().copied()).collect();
            let mut wt = vec![0.0f32; rows.len()];
            for (r, row) in rows.chunks(cols).enumerate() {
                for (c, v) in row.iter().enumerate() {
                    wt[c * n + r] = *v;
                }
            }
            out.iter_mut().zip(linear_rows(&q, &wt, k, n, cols)).for_each(|(o, v)| *o += v);
        });
        found.then_some(out)
    }
}

/// Rows per block a tensor is read in: bounds the host memory a slice takes
/// beyond the tensor it produces.
const ROWS_PER_BLOCK: usize = 512;

impl TensorSource for Slicing<'_> {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        let (d, k) = (self.cfg.d_model as usize, self.sliced.d_model as usize);
        let head = self.cfg.head_weight();
        let tensor = if name == "tok.weight" {
            self.reader(name, None)
        } else if name == "lm_head.weight" {
            self.gain("norm.weight").and_then(|g| self.reader(head, Some(&g)))
        } else if name == "norm.weight" || name.ends_with(".ln1.weight") || name.ends_with(".ln2.weight") {
            Some(vec![(d as f32 / k as f32).sqrt(); k])
        } else if let Some(rest) = name.strip_prefix("blocks.") {
            let layer = rest.split('.').next().unwrap_or_default();
            let reader = |norm: &str| self.gain(&format!("blocks.{layer}.{norm}.weight")).and_then(|g| self.reader(name, Some(&g)));
            match &rest[layer.len() + 1..] {
                "attn.wq.weight" | "attn.wk.weight" | "attn.wv.weight" => reader("ln1"),
                "mlp.gate.weight" | "mlp.up.weight" => reader("ln2"),
                "attn.wo.weight" => self.writer(name, self.cfg.q_dim() as usize),
                "mlp.down.weight" => self.writer(name, self.cfg.d_ff as usize),
                // Per-head q/k norms and q/k/v biases live off the stream.
                leaf => {
                    let width = if leaf.ends_with("norm.weight") { self.cfg.head_dim as usize } else { 1 };
                    let mut out = Vec::new();
                    self.row_blocks(name, width, &mut |_, rows| out.extend_from_slice(rows)).then_some(out)
                }
            }
        } else {
            None
        };
        match tensor {
            Some(t) => {
                f(&t);
                true
            }
            None => false,
        }
    }
}
