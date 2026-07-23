// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pipeline-parallel Qwen: the decoder layers are split into contiguous ranges,
//! one **stage** per GPU, so a model whose weights exceed a single card fits
//! across several. Each stage ([`crate::model::Qwen`] built with a [`Shard`])
//! allocates weights + activations only for its layers (plus the embedding on
//! stage 0 and the final-norm+lm_head+cross-entropy on the last stage).
//!
//! The only tensor crossing a stage boundary is the residual stream `res` (one
//! `[b·t·d_model]` slab): forward passes it host-staged from stage *i* to *i+1*;
//! backward passes the residual gradient `dres` back the other way. This traffic
//! is tiny (a few MB per cut per pass) and needs no NVLink.
//!
//! Tied embeddings: `tok.weight` is used by both the embedding (stage 0) and the
//! lm_head (last stage). It is **replicated** on those two stages (read-only in
//! the forward). In the backward each contributes half the tied gradient
//! (`emb_bwd` on stage 0, the lm_head `matmul_dw` on the last stage); the
//! optimiser sums them and applies one globally-clipped update to both replicas,
//! keeping them bit-identical — exactly the single-device tied-weight math.
//!
//! Correctness is validated in `tests/shard_parity.rs`: sharded forward loss and
//! per-parameter gradients are bit-exact against the single-device model, and a
//! sharded overfit run reduces the loss.

use std::collections::HashMap;

use crate::config::QwenConfig;
use crate::model::{Qwen, Shard};

/// A pipeline of decoder stages across one or more GPUs.
pub struct Pipeline {
    stages: Vec<Qwen>,
    cfg: QwenConfig,
    tied: bool,
}

/// Split `n_layers` into `gpus.len()` contiguous ranges, as even as possible,
/// assigning stage *s* to `gpus[s]`. Stage 0 embeds; the last stage carries the
/// head.
fn plan_shards(n_layers: usize, gpus: &[usize]) -> Vec<Shard> {
    let n = gpus.len();
    assert!(n >= 1, "pipeline needs at least one stage");
    let base = n_layers / n;
    let rem = n_layers % n;
    let mut out = Vec::with_capacity(n);
    let mut start = 0;
    for (s, &gpu) in gpus.iter().enumerate() {
        let cnt = base + if s < rem { 1 } else { 0 };
        let end = start + cnt;
        out.push(Shard { start, end, embed: s == 0, head: s == n - 1, gpu_index: gpu });
        start = end;
    }
    debug_assert_eq!(start, n_layers);
    out
}

impl Pipeline {
    /// Build a pipeline with one stage per entry of `gpus` (the physical GPU
    /// index for that stage — repeats are allowed, e.g. `&[0, 0]` puts two stages
    /// on one card, which still exercises the full cross-stage transfer path).
    /// `init` is the full model's weights, held once in host RAM; each stage
    /// uploads only its own slice to its GPU.
    pub fn new(
        cfg: QwenConfig,
        b: u32,
        t: u32,
        init: &HashMap<String, Vec<f32>>,
        train: bool,
        gpus: &[usize],
    ) -> Pipeline {
        let shards = plan_shards(cfg.n_layers as usize, gpus);
        let prev_gpu = std::env::var("BRAIN_GPU_INDEX").ok();
        let prev_off = std::env::var("BRAIN_OFFLOAD_ADAM").ok();
        // Sharded training uses the host offload optimiser (moments in RAM) — the
        // pipeline optimiser step drives exactly that path, and it pairs naturally
        // with sharding for large models. Force it on for the stages we build.
        if train {
            std::env::set_var("BRAIN_OFFLOAD_ADAM", "1");
        }
        let mut stages = Vec::with_capacity(shards.len());
        for sh in shards {
            // The backend reads BRAIN_GPU_INDEX at device creation; set it per
            // stage (construction is sequential, so no race).
            std::env::set_var("BRAIN_GPU_INDEX", sh.gpu_index.to_string());
            stages.push(Qwen::new_shard(cfg.clone(), b, t, init, train, sh));
        }
        match prev_gpu {
            Some(v) => std::env::set_var("BRAIN_GPU_INDEX", v),
            None => std::env::remove_var("BRAIN_GPU_INDEX"),
        }
        match prev_off {
            Some(v) => std::env::set_var("BRAIN_OFFLOAD_ADAM", v),
            None => std::env::remove_var("BRAIN_OFFLOAD_ADAM"),
        }
        let tied = cfg.head_weight() == "tok.weight";
        Pipeline { stages, cfg, tied }
    }

    /// Convenience: `n_stages` stages on GPUs `0..n_stages`.
    pub fn even(cfg: QwenConfig, b: u32, t: u32, init: &HashMap<String, Vec<f32>>, train: bool, n_stages: usize) -> Pipeline {
        let gpus: Vec<usize> = (0..n_stages).collect();
        Pipeline::new(cfg, b, t, init, train, &gpus)
    }

    pub fn n_stages(&self) -> usize {
        self.stages.len()
    }

    /// Forward the batch through every stage, returning the cross-entropy loss.
    /// The residual is carried host-staged from each stage to the next.
    pub fn forward(&self, x: &[u32], y: &[u32]) -> f32 {
        for st in &self.stages {
            st.set_batch(x, y);
        }
        let last = self.stages.len() - 1;
        let mut carry: Option<Vec<f32>> = None;
        for (i, st) in self.stages.iter().enumerate() {
            if let Some(res) = &carry {
                st.write_in_res(res);
            }
            if i == last {
                return st.forward();
            }
            st.run_forward();
            carry = Some(st.read_out_res());
        }
        unreachable!("pipeline has at least one stage")
    }

    /// Backward through every stage (reverse order), carrying the residual
    /// gradient host-staged from each stage back to the previous one. Requires a
    /// preceding [`Self::forward`] on the same batch.
    pub fn backward(&self) {
        let mut carry: Option<Vec<f32>> = None;
        for i in (0..self.stages.len()).rev() {
            let st = &self.stages[i];
            if let Some(d) = &carry {
                st.write_out_dres(d);
            }
            st.run_backward();
            if i > 0 {
                carry = Some(st.read_in_dres());
            }
        }
    }

    pub fn zero_grads(&self) {
        for st in &self.stages {
            st.zero_grads();
        }
    }

    pub fn poll_wait(&self) {
        for st in &self.stages {
            st.poll_wait();
        }
    }

    /// One AdamW step across all stages. The tied `tok.weight` gradient is summed
    /// across its two replicas and written back to both; the global grad-norm
    /// (tied counted once) yields one clip coefficient applied on every stage, so
    /// the replicas stay bit-identical. Mirrors the single-device optimiser math.
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        // 1. Sum the replicated tied gradient across the stages that hold it.
        if self.tied {
            let holders: Vec<usize> = (0..self.stages.len())
                .filter(|&i| self.stages[i].has_param("tok.weight"))
                .collect();
            if holders.len() > 1 {
                let mut sum = self.stages[holders[0]].read_grad("tok.weight");
                for &i in &holders[1..] {
                    for (a, b) in sum.iter_mut().zip(self.stages[i].read_grad("tok.weight")) {
                        *a += b;
                    }
                }
                for &i in &holders {
                    self.stages[i].write_grad("tok.weight", &sum);
                }
            }
        }

        // 2. Global grad-norm -> one clip coefficient (tied weight counted once).
        let gscale = if extra_scale != 0.0 { 1.0 / extra_scale } else { 1.0 };
        let scale = if let Some(max_norm) = clip {
            let mut counted_tok = false;
            let mut gsq = 0.0f64;
            for st in &self.stages {
                let has_tok = st.has_param("tok.weight");
                let exclude: &[&str] = if has_tok && counted_tok { &["tok.weight"] } else { &[] };
                gsq += st.grad_sq(exclude);
                if has_tok {
                    counted_tok = true;
                }
            }
            let norm = (gsq.sqrt() as f32) * gscale;
            gscale * (max_norm / norm.max(max_norm)).min(1.0)
        } else {
            gscale
        };

        // 3. Per-stage AdamW with the shared, globally-reduced scale.
        for st in &self.stages {
            st.opt_step_scaled(t, lr, wd, scale);
        }
        self.poll_wait();
    }

    /// The true gradient for `name`: summed across replicas (tied `tok.weight`)
    /// or read from its single owning stage. Matches the single-device gradient.
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        let holders: Vec<&Qwen> = self.stages.iter().filter(|s| s.has_param(name)).collect();
        assert!(!holders.is_empty(), "no stage holds param {name}");
        let mut sum = holders[0].read_grad(name);
        for st in &holders[1..] {
            for (a, b) in sum.iter_mut().zip(st.read_grad(name)) {
                *a += b;
            }
        }
        sum
    }

    /// Read weight `name` from its (first) owning stage.
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        let st = self.stages.iter().find(|s| s.has_param(name)).unwrap_or_else(|| panic!("no stage holds param {name}"));
        st.read_weight(name)
    }

    /// Gather every stage's weights into one checkpoint (tied `tok.weight` taken
    /// from its first holder — the replicas are identical).
    pub fn save(&self, path: &str) {
        let mut have: HashMap<String, Vec<f32>> = HashMap::new();
        for st in &self.stages {
            for (name, _) in &st.ps.params {
                have.entry(name.clone()).or_insert_with(|| st.read_weight(name));
            }
        }
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self
            .cfg
            .param_list()
            .into_iter()
            .map(|(n, numel)| {
                let w = have.remove(&n).unwrap_or_else(|| panic!("pipeline save: missing param {n}"));
                (n, vec![numel as u64], w)
            })
            .collect();
        checkpoint::save(path, self.cfg.to_json(), &tensors);
    }
}
