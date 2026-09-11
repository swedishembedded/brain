// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gate for **TimesFM-3**'s backward
//! (`timesfm3::train::Timesfm3Train`).
//!
//! ## What this check covers, and what is frozen
//!
//! **Nothing is frozen.** The harness walks `ParamStore`'s full parameter
//! list, which is exactly [`timesfm3::config::Timesfm3Config::param_list`] -
//! for `Timesfm3Config::tiny()` (3 layers) that is 71 tensors, under the
//! reference checkpoint's own names:
//!
//! | tensor | shape | what its gradient exercises |
//! |---|---|---|
//! | `pre_transformer_resblock.{hidden,output,residual}_layer.weight` | `[D, in]` / `[D, D]` | the pre-transformer ResidualBlock: a ReLU branch (`leaky_relu_bwd` at `slope=0`) summed with a bare linear skip, both reading the same input |
//! | `transformer_stack.{l}.pre_{seq,var}_attn_ln.weight` | `[D]` | `rms_inv_eps` + `rmsnorm_dw` on each sublayer's input norm |
//! | `transformer_stack.{l}.{seq,var}_attn.{query,key,value,out}_proj.weight` | `[D, D]` | four SEPARATE square GEMMs (this model fuses no QKV), so `matmul_dw` is gated per projection rather than through one packed region |
//! | `transformer_stack.{l}.{seq,var}_attn.key_ln.weight` | `[head_dim]` | per-head QK-norm on K, an ordinary trainable gain |
//! | `transformer_stack.{l}.{seq,var}_attn.query_ln.weight` | `[head_dim]` | **half of the PerDimScale fold** - see below |
//! | `transformer_stack.{l}.{seq,var}_attn.per_dim_scale.per_dim_scale` | `[head_dim]` | **the other half** |
//! | `transformer_stack.{l}.post_{seq,var}_attn_ln.weight` | `[D]` | the sandwich norm on each sublayer's output |
//! | `transformer_stack.{l}.pre_ff_ln.weight` / `ff0` / `ff1` / `post_ff_ln.weight` | `[D]` / `[H, D]` / `[D, H]` / `[D]` | the ReLU feedforward (hidden width `H`, no 4x expansion and no gate) |
//! | `output_head.weight` / `output_head.bias` | `[P·Q, D]` / `[P·Q]` | the quantile head's GEMM and its `bias_grad` row-sum |
//!
//! Two structural properties of this model are the reason this check exists
//! rather than being inferred from [`crate::t5`]:
//!
//! * **two attention kinds on the same residual stream** - a causal,
//!   RoPE'd sequence attention over patches, and a NON-causal variate
//!   attention over channels that is bracketed by a `swap_axes12_vec` pair.
//!   The variate backward has to undo that permutation on `d_ctx` going in
//!   and on `d_q`/`d_k`/`d_v` coming out; a transposed adjoint is invisible
//!   at `v == n` and visible at `v != n`, which is why the harness runs at
//!   `v = 2, n = 3`;
//! * **attention scale folded to 1.0** - the model pushes the `1/√head_dim`
//!   into the query gain, so the backward must use `attn_bwd_d{q,k}_bias`
//!   (which take `scale` as a Param) and never the `_bidir` pair (which
//!   hardcode `1/√head_dim`). Same trap [`crate::t5`] documents, reached by a
//!   different route.
//!
//! ## The PerDimScale fold needs a per-entry check, not just this one
//!
//! Inference folds `per_dim_scale` into an effective `query_ln.weight` once
//! at load. A trainer cannot: both are live, independently trained tensors
//! that the checkpoint must keep separate. So the forward recomputes
//! `query_ln.weight · log2(e) · softplus(per_dim_scale)` per step, and the
//! reverse gets ONE `d(effective gain)` from `rmsnorm_dw` which it splits
//! host-side into the two originals.
//!
//! That is a **shared/folded parameter**, exactly the shape AGENTS.md
//! requires an `elementwise_check` for: [`directional_check`] contracts a
//! tensor onto one ±1 direction and keeps the best-agreeing of `n_dirs`,
//! which is the wrong selection rule when a *share* of the gradient is
//! wrong rather than all of it (measured on T5's `rel_bias`: a 33 % error
//! passed every directional check). [`check_timesfm3_per_dim_scale_elementwise`]
//! is the per-ENTRY detector; [`check_timesfm3`] is not a substitute for it.
//!
//! ## The objective
//!
//! ```text
//! L = <c, core_forward(x)>
//! ```
//! with `c` a fixed random `[b·v·n, P·Q]` direction on the RAW output-head
//! logits. `L` is exactly linear in them, so `backward()` seeds `d_logits`
//! with `c` directly - the same `dL/dy = r` trick [`crate::t5`] uses, and for
//! the same reason: it turns the whole graph into one differentiable scalar
//! without inventing an objective.
//!
//! Using the real pinball loss here would gate a different thing.
//! `Timesfm3Train::backward` takes `d_logits` FROM the caller, so the
//! objective is outside the graph under test; `mean_pinball_grad` and its
//! `pinball_grad_w` kernel are gated against each other directly in
//! `crates/gradcheck/tests/glue.rs`, which is where an objective belongs.
//!
//! ## Epsilon
//!
//! `5e-4`, not the workspace default `5e-3`. A `±1` direction over `numel`
//! entries is an L2 step of `eps·√numel`, and the largest tensor at the tiny
//! config is `output_head.weight` at 480 entries, where `5e-3` is a 0.11 step
//! in weight space - outside the region where a 3-layer stack with two
//! unscaled softmaxes is locally linear.
//!
//! That is not asserted, it is MEASURED - [`check_timesfm3_eps_sweep`]
//! returns the whole table and `timesfm3_eps_plateau` gates it, per
//! AGENTS.md's rule that a failing gradcheck is probed and reported, never
//! widened. Max relative error over all 71 tensors (seed 7, tiny config):
//!
//! | eps | P40 | backend-cpu |
//! |---|---|---|
//! | 5e-3 | 4.32e-2 | 4.33e-2 |
//! | 2e-3 | 6.52e-2 | 6.52e-2 |
//! | 1e-3 | 3.16e-2 | 2.19e-2 |
//! | **5e-4** | **2.21e-2** | **2.74e-2** |
//! | 2e-4 | 8.12e-2 | 1.18e-1 |
//! | 1e-4 | 1.84e-1 | 6.44e-2 |
//! | 5e-5 | 2.49e-1 | 3.26e-1 |
//!
//! The U is the textbook one: truncation error dominates above 1e-3, fp32
//! cancellation below 2e-4, and 5e-4 is the floor on both backends.
//!
//! The per-ENTRY checks sit twenty times higher, at `1e-2`, because a
//! single-entry step has no `√numel` amplification - the loss difference is
//! `eps·|dL/dw_i|` directly. Measured over `head_dim = 6` entries (seed 7):
//!
//! | eps | 2e-2 | **1e-2** | 5e-3 | 2e-3 | 1e-3 | 5e-4 |
//! |---|---|---|---|---|---|---|
//! | P40 | 2.26e-3 | **5.84e-3** | 1.39e-2 | 2.78e-2 | 4.65e-2 | 1.94e-1 |
//! | backend-cpu | 2.21e-3 | **5.84e-3** | 9.70e-3 | 1.57e-2 | 4.46e-2 | 1.04e-1 |
//!
//! `timesfm3_per_dim_scale_eps_plateau` gates that table.

use std::cell::Cell;

use data::rng::Rng;

use timesfm3::config::Timesfm3Config;
use timesfm3::train::{Timesfm3Train, TRAIN_PIPELINES};

use crate::{directional_check, CheckModel, Report};

/// One trainable TimesFM-3 core, a fixed resblock input + patch mask, and the
/// fixed proxy direction on the raw logits that defines `L`.
struct Timesfm3Harness {
    m: Timesfm3Train,
    /// `[b*v*n, output_patch_len*num_quantiles]` - the proxy direction on the
    /// raw output-head logits.
    c: Vec<f32>,
    /// The reverse pass reads activation caches only a forward leaves valid.
    /// `directional_check` always calls `loss()` first, but a caller driving
    /// the harness by hand might not.
    fwd_done: Cell<bool>,
}

impl Timesfm3Harness {
    /// The device is the **pooled test device** (`gpu_core::testgpu::dev`),
    /// not a fresh `Gpu::new`: several entry points share one test binary,
    /// and a device per model object is the pattern AGENTS.md bans.
    ///
    /// The mask is all-visible. A masked patch contributes `MASK_NEG` to the
    /// scores and its own row is still produced, so masking changes WHICH
    /// activations matter but adds no parameter path; the mask construction
    /// itself is locked by `timesfm3::train`'s own bitwise forward-parity
    /// test, which runs a non-trivial mask against `core_forward`.
    fn new(cfg: Timesfm3Config, b: usize, v: usize, n: usize, seed: u64) -> Timesfm3Harness {
        let init = timesfm3::train::init_weights(&cfg, seed);
        let x = timesfm3::train::fixed_input(&cfg, b, v, n, seed);
        let mask = vec![false; b * v * n];
        let rows = b * v * n;
        let head_out = cfg.head_out_dim();
        let m = Timesfm3Train::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), cfg, &x, &mask, b, v, n, &init);
        let mut rng = Rng::new(seed ^ 0x713);
        Timesfm3Harness { c: (0..rows * head_out).map(|_| rng.next_f32() - 0.5).collect(), m, fwd_done: Cell::new(false) }
    }
}

impl CheckModel for Timesfm3Harness {
    fn param_names(&self) -> Vec<String> {
        self.m.ps.params.iter().map(|(n, _)| n.clone()).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.m.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.m.write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.m.read_grad(name)
    }
    fn loss(&self) -> f32 {
        self.m.forward();
        self.m.poll_wait();
        self.fwd_done.set(true);
        // Accumulate in f64. The sum is a host reduction over `b·v·n·P·Q`
        // terms and it is then DIFFERENCED by finite differences, so an f32
        // accumulator's round-off lands directly in the numerator of
        // `(L(w+eps) - L(w-eps))`. `elementwise_check` perturbs ONE entry, so
        // that difference is tiny relative to `L` and the accumulator's noise
        // is the binding error term.
        let dot: f64 = self.m.read_logits().iter().zip(&self.c).map(|(y, c)| *y as f64 * *c as f64).sum();
        dot as f32
    }
    fn zero_grads(&self) {
        self.m.zero_grads();
    }
    fn backward(&self) {
        if !self.fwd_done.get() {
            let _ = self.loss();
        }
        // dL/d(logits) = c - L is linear in the raw head output.
        self.m.backward(&self.c);
        self.m.poll_wait();
    }
}

/// **The gate.** The TimesFM-3 core backward at gradcheck scale: the tiny
/// config's 3 layers, `b=1`, `v=2`, `n=3`.
///
/// `v != n` on purpose. Variate attention is the sequence sublayer's graph
/// with the `[b,v,n,D] <-> [b,n,v,D]` swap wrapped around it; at `v == n`
/// both the forward permute and its adjoint are shape-compatible with their
/// own transpose, so an index swap in `swap12_adjoint` produces a
/// wrong-but-well-formed buffer and every tensor downstream still gradients
/// plausibly. `2 != 3` makes that a shape error instead.
pub fn check_timesfm3(seed: u64) -> Report {
    let h = Timesfm3Harness::new(Timesfm3Config::tiny(), 1, 2, 3, seed);
    directional_check(&h, 5e-4, 4, seed ^ 0x1234)
}

/// The same graph at a **single** layer: a failure here is local to one
/// layer's own three sublayers, so it separates "the layer backward is wrong"
/// from anything cross-layer (the `add2` residual convergences that carry
/// `d_h` from the head down through the stack to the resblock).
pub fn check_timesfm3_one_layer(seed: u64) -> Report {
    let cfg = Timesfm3Config { num_layers: 1, ..Timesfm3Config::tiny() };
    let h = Timesfm3Harness::new(cfg, 1, 2, 3, seed);
    directional_check(&h, 5e-4, 4, seed ^ 0x1234)
}

/// **The gate that actually covers the PerDimScale fold.** Per-ENTRY finite
/// differences on one `per_dim_scale.per_dim_scale`, the tensor whose
/// gradient is not produced by any kernel at all: the reverse gets a single
/// `d(effective query gain)` from `rmsnorm_dw` and splits it host-side into
/// `d(query_ln.weight) = D·log2(e)·softplus(s)` and
/// `d(per_dim_scale) = D·g·log2(e)·sigmoid(s)`.
///
/// [`check_timesfm3`] does cover this tensor, but only through a contraction
/// that best-of-4 actively selects to minimise - the selection rule AGENTS.md
/// records as blind to a *partial* gradient error, which is exactly the shape
/// a wrong `softplus`/`sigmoid` pairing (they differ by a factor that is
/// smooth and O(1) near zero) would produce. `head_dim` is 6 at the tiny
/// config, so this is 12 extra forwards.
///
/// `eps = 1e-2`, twenty times [`check_timesfm3`]'s: a single-entry step has
/// no `√numel` amplification, so the loss difference is `eps·|dL/dw_i|` and
/// fp32 cancellation bites far sooner. [`check_timesfm3_per_dim_scale_eps_sweep`]
/// returns the table and `timesfm3_per_dim_scale_eps_plateau` gates it.
pub fn check_timesfm3_per_dim_scale_elementwise(seed: u64) -> Report {
    let h = Timesfm3Harness::new(Timesfm3Config::tiny(), 1, 2, 3, seed);
    crate::elementwise_check(&h, "transformer_stack.layers.1.seq_attn.per_dim_scale.per_dim_scale", 1e-2)
}

/// The same per-ENTRY check on the fold's OTHER half. `query_ln.weight` and
/// `per_dim_scale` enter the effective gain as a product, so a split that
/// mixed the two up (or dropped one factor) can leave one of them correct;
/// checking only one entry of the pair would gate half the fold.
pub fn check_timesfm3_query_ln_elementwise(seed: u64) -> Report {
    let h = Timesfm3Harness::new(Timesfm3Config::tiny(), 1, 2, 3, seed);
    crate::elementwise_check(&h, "transformer_stack.layers.1.seq_attn.query_ln.weight", 1e-2)
}

/// The eps table behind [`check_timesfm3_per_dim_scale_elementwise`]'s `1e-2`.
pub fn check_timesfm3_per_dim_scale_eps_sweep(seed: u64) -> Vec<(f32, f32)> {
    let h = Timesfm3Harness::new(Timesfm3Config::tiny(), 1, 2, 3, seed);
    [2e-2f32, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4]
        .iter()
        .map(|&eps| (eps, crate::elementwise_check(&h, "transformer_stack.layers.1.seq_attn.per_dim_scale.per_dim_scale", eps).max_rel()))
        .collect()
}

/// The eps/error relationship on this graph, measured rather than assumed.
///
/// Returns `(eps, max_rel_err)` over the whole tiny-config sweep. AGENTS.md's
/// rule when a gradcheck fails is to PROBE this table and report it, never to
/// widen the bound.
pub fn check_timesfm3_eps_sweep(seed: u64) -> Vec<(f32, f32)> {
    let h = Timesfm3Harness::new(Timesfm3Config::tiny(), 1, 2, 3, seed);
    [5e-3f32, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
        .iter()
        .map(|&eps| (eps, directional_check(&h, eps, 4, seed ^ 0x1234).max_rel()))
        .collect()
}

