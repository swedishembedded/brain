// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The correctness gate for the SAM-1 tower's backward pass.**
//!
//! Finite differences against the crate's own forward, over every one of the
//! 39 tensors in [`SamViTConfig::tiny`]'s manifest -- the patch-embed conv, the
//! learned absolute position embedding, both ViT blocks (LayerNorms, fused qkv,
//! output projection, both relative-position tables, both MLP linears), the two
//! neck convs with their `LayerNorm2d`s, and the two stride-2 compressor convs.
//! No parameter is frozen and none is exempt.
//!
//! ## The fixture, and what each number is chosen to break
//!
//! ```text
//!   image        1 x 3 x 26 x 14, patch 2      -> grid 13 x 7 = 91 tokens
//!   d_model      16 = 2 heads x head_dim 8
//!   block 0      WINDOWED 4 x 3. Neither extent divides: 13 -> 16, 7 -> 9,
//!                so the zero-pad path runs, 12 windows of 12 rows, and 53 of
//!                the 144 window-major rows are pad.
//!   block 1      GLOBAL, one 13 x 7 span of 91 rows.
//!   attn_chunk   4  -> every span is multi-chunk (12 rows = 3 chunks;
//!                91 rows = 22 chunks + a SHORT final chunk of 3)
//!   rel tables   block 0: h = 7 == 2*4-1  IDENTITY, w = 11 > 2*3-1 DOWNsample
//!                block 1: h = 15 < 2*13-1 UPsample, w = 13 == 2*7-1 IDENTITY
//!   mlp          17;  neck 6;  compress 9 -> 11;  compressor grid 4 x 2
//! ```
//!
//! The SAM geometry deliberately mirrors the checkpoint-free golden dumper's
//! own tiny SAM sub-fixture, so the two are directly comparable once a parity
//! run exists. `d_model` deliberately does **not**: the golden uses 10, and 10
//! is not dispatchable through `model::block::chunked_bidir_fwd` at a 12-row
//! window -- that builder binds `ctx` sliced at `row0 * d_model` floats, a
//! storage-binding offset must be 64-float (256 B) aligned, and `12 * 10 = 120`
//! is not. A 12-row window forces `16 | d_model`, so 16 is the smallest legal
//! width. `SamViTConfig::check_bindable` states that as an assertion, and
//! `config::tests::a_width_of_ten_is_rejected_loudly` pins it.
//!
//! Residual coincidences, stated rather than glossed: `d_model == pad_h == 16`
//! and `compress_mid == pad_w == 9`. Both pair numbers from subsystems that
//! never index each other (a channel width against a grid extent); the
//! coincidences that would actually hide a bug -- `head_dim` against any grid or
//! window extent, `window_h` against `window_w`, `grid_h` against `grid_w` -- are
//! all broken.
//!
//! ## What the windowed block covers that the kernel-level gate does not
//!
//! `gradcheck::check_deepseekocr_relpos` already exercises the six
//! `attn_relpos_*` kernels on a synthetic two-"block" graph whose qkv buffer is
//! itself the leaf parameter. This gate adds everything between: the pad is
//! produced by a REAL `norm1` + `WindowPlan::padded` gather rather than assumed,
//! the qkv that reaches attention is a REAL projection (so a pad row's key and
//! value are the qkv **bias**, and `attn.qkv.bias`'s gradient is only right if
//! the pad rows are included in `bias_grad`), and the block's output goes
//! through `window_reverse`, a residual, an MLP, a conv neck and a compressor
//! before reaching the loss.
//!
//! ## Objective
//!
//! `L = <compressor_out, r>` for a fixed random `r`, so `backward()` seeds
//! `d_out` with `r` directly.
//!
//! ## Results, and the mutation that proves the gate can fail
//!
//! Green on both backends: 39/39 tensors, worst relative error **1.359e-3** on
//! an Intel Arc (Vulkan) and **1.245e-3** on the `backend-cpu` Cranelift JIT,
//! with **no** `not JIT-compiled` fallback for any kernel -- including all six
//! `attn_relpos_*`. The per-entry table check's worst *relative* number is
//! larger (1.19e-1 GPU / 1.54e-1 CPU) but every entry passes on the combined
//! `(atol, rtol)` rule: those are entries whose derivative is ~2e-3, where a
//! 3e-4 absolute difference is finite-difference noise, not disagreement.
//!
//! A gate nobody has seen fail is a hypothesis, so one was broken on purpose:
//! summing `attn.qkv.bias`'s gradient over `rows` instead of `attn_rows` --
//! i.e. dropping exactly the 53 zero-padded window positions, the single
//! subtlest thing this fixture exists to cover. **The directional check did NOT
//! catch it**: `vision.sam.blocks.0.attn.qkv.bias` came back at
//! `analytic +1.614e1` vs `numeric +1.722e1`, `rel 6.28e-2` -- a clean pass
//! inside the `(4e-3, 8e-2)` gate, with every other tensor unmoved. That is the
//! documented partial-gradient blindness of a ±1 contraction, and it is why
//! [`windowed_pad_rows_contribute_to_the_qkv_bias_gradient`] exists: per entry,
//! the same mutation turns **44 of 48 entries RED, worst rel 1.41**. The
//! mutation was then reverted and both gates re-run green.

use gradcheck::{directional_check, elementwise_check, CheckModel, Report};
use sam1::{SamEncoder, SamViTConfig};

/// fp32 finite differences on a device: the workspace-standard tolerance.
const ATOL: f32 = 4e-3;
const RTOL: f32 = 8e-2;

/// `eps = 5e-4`, MEASURED rather than argued. The a-priori case for going below
/// the workspace default `5e-3` is that a ±1 direction over `numel` entries is
/// an L2 step of `eps*sqrt(numel)`, so `pos_embed`'s 1456 entries at `5e-3`
/// move 0.19 in weight space. [`eps_plateau`] says that reasoning overstates it
/// here -- the plateau is wide and `5e-3` is fine:
///
/// ```text
///   eps      5e-3      1e-3      5e-4      1e-4
///   GPU    1.03e-3   1.32e-3   2.07e-3   1.89e-2
///   CPU    1.04e-3   1.58e-3   2.95e-3   7.96e-3
/// ```
///
/// `5e-4` is kept as the mid-plateau point (an order of magnitude clear of the
/// `1e-4` cancellation knee on both backends, and of the truncation end), not
/// because `5e-3` was shown to fail.
const EPS: f32 = 5e-4;

/// Orphan-rule wrapper: `gradcheck::CheckModel` has a blanket impl over
/// `model::Model`, so a foreign type cannot implement it directly. Same
/// workaround as `gradcheck::sam2::Sam2DecoderCheck`.
///
/// `SamEncoder` deliberately does not implement `model::Model`: that trait's
/// `Config: ModelConfig` bound requires `vocab()` and `block_size()`, which a
/// vision encoder has no honest answer for -- the same reason `crates/sam2` and
/// `crates/vqgan` wrap instead of implementing it.
struct Check(SamEncoder);

impl CheckModel for Check {
    fn param_names(&self) -> Vec<String> {
        self.0.param_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.0.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.0.write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.0.read_grad(name)
    }
    fn loss(&self) -> f32 {
        self.0.forward()
    }
    fn zero_grads(&self) {
        self.0.zero_grads();
    }
    fn backward(&self) {
        self.0.backward();
    }
}

fn fixture(seed: u64) -> Check {
    // The pooled test device, never a fresh `Gpu::new` -- the entry points below
    // share one device per test binary.
    let g = gpu_core::testgpu::dev(sam1::PIPELINES);
    Check(SamEncoder::with_dense_init(g, SamViTConfig::tiny(), seed))
}

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn gate(report: Report, what: &str) {
    report.print();
    let fails = report.failures(ATOL, RTOL);
    assert!(
        fails.is_empty(),
        "{what} gradient check failed for {:?}",
        fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
    );
    let dead = report.dead_gradients();
    assert!(
        dead.is_empty(),
        "{what}: exactly-zero analytic gradients for {:?}",
        dead.iter().map(|c| &c.param).collect::<Vec<_>>()
    );
    println!("{what}: {} tensors, worst rel {:.3e}", report.checks.len(), report.max_rel());
}

/// The gate.
#[test]
fn sam1_analytic_grads_match_finite_differences() {
    if skip() {
        return;
    }
    let h = fixture(7);
    gate(directional_check(&h, EPS, 4, 0x1234), "SAM-1 tower");
}

/// The shared relative-position tables, **per entry**.
///
/// Not redundant with the directional check above, and the reason is measured
/// rather than assumed (see `gradcheck::elementwise_check`'s rustdoc, and
/// `gradcheck::deepseekocr`'s mutation table): the windowed block's two tables
/// are folded across all twelve of its windows, every head and every query
/// chunk, so an error that drops a *share* of the gradient contracts to nearly
/// nothing on a single ±1 direction -- and `directional_check` then keeps the
/// best of four directions, which actively selects for that.
///
/// Restricted to block 0's tables: block 0 is the windowed one, so it is the
/// only place a table is shared by more than one span.
///
/// Some entries of `rel_pos_w` come back with an analytic AND a numeric
/// gradient of exactly zero, and that is correct rather than a dead gradient:
/// its 11 rows are resampled down to `2*3-1 = 5`, and the half-pixel rule
/// `src = (d + 0.5)*11/5 - 0.5` reads sources 0/1, 2/3, 5/5, 7/8, 9/10 -- row 4
/// is never read at all, and row 6 is only ever the second tap of an exactly
/// integral `src`, whose weight is 0. `Report::dead_gradients` therefore does
/// not flag them (it needs a nonzero NUMERIC derivative), and the downsample
/// path is still covered by the 9 rows that do move.
#[test]
fn windowed_block_rel_pos_tables_are_the_sum_over_windows() {
    if skip() {
        return;
    }
    let h = fixture(7);
    let mut checks = Vec::new();
    for name in ["vision.sam.blocks.0.attn.rel_pos_h", "vision.sam.blocks.0.attn.rel_pos_w"] {
        // `eps = 5e-3`: a single-entry step has no `sqrt(numel)` amplification,
        // so the loss difference is `eps*|dL/dw_i|` and fp32 cancellation bites
        // well before the directional check's `5e-4`.
        checks.extend(elementwise_check(&h, name, 5e-3).checks);
    }
    gate(Report { checks }, "SAM-1 rel-pos tables (per entry)");
}

/// **The windowed block's zero-pad rows really do reach `attn.qkv.bias`**, per
/// entry.
///
/// This test exists because the directional check above was MEASURED not to
/// catch its failure. Summing that bias gradient over `rows` instead of
/// `attn_rows` -- dropping exactly the 53 padded window positions, whose keys
/// and values are the qkv bias and which therefore do contribute -- leaves
/// `directional_check` reporting `rel 6.28e-2` on the affected tensor, a clean
/// PASS inside the `(4e-3, 8e-2)` gate, because the missing share is ~6 % of a
/// large number and best-of-4-directions then picks the kindest projection.
/// Perturbing one entry at a time removes both effects: the same mutation turns
/// **44 of 48 entries RED, worst rel 1.41**.
///
/// 48 entries, so 96 extra forwards.
#[test]
fn windowed_pad_rows_contribute_to_the_qkv_bias_gradient() {
    if skip() {
        return;
    }
    let h = fixture(7);
    gate(elementwise_check(&h, "vision.sam.blocks.0.attn.qkv.bias", 5e-3), "SAM-1 windowed qkv bias (per entry)");
}

/// The eps probe, run as a gate: it asserts [`EPS`] is not sitting on a knee and
/// prints the table if a future change moves it. The repo's rule when a
/// gradcheck fails is to probe this and report it, never to widen the bound.
#[test]
fn eps_plateau() {
    if skip() {
        return;
    }
    let h = fixture(7);
    let table: Vec<(f32, f32)> = [5e-3f32, 1e-3, 5e-4, 1e-4]
        .iter()
        .map(|&e| (e, directional_check(&h, e, 2, 0x1234).max_rel()))
        .collect();
    for (e, rel) in &table {
        println!("  eps={e:.1e}  max_rel={rel:.3e}");
    }
    let at = |e: f32| table.iter().find(|(x, _)| *x == e).expect("eps in table").1;
    assert!(at(EPS) <= RTOL, "eps {EPS:.1e} max_rel {:.3e} exceeds rtol", at(EPS));
    assert!(at(EPS) <= at(5e-3).max(at(1e-4)) * 4.0, "eps {EPS:.1e} is not on the plateau: {table:?}");
}

/// Shape + liveness smoke: the tower runs, produces the documented output
/// shape, and no parameter comes back with an identically-zero gradient.
///
/// The zero-gradient half is the structural guard `gradcheck::zero_grad_params`
/// exists for -- a backend-specific kernel that silently returns all-zero
/// gradients presents as a *passing* directional check at small dims.
#[test]
fn every_parameter_is_live() {
    if skip() {
        return;
    }
    let h = fixture(11);
    let cfg = SamViTConfig::tiny();
    let l = h.loss();
    assert!(l.is_finite() && l != 0.0, "loss {l} is not a usable objective");
    // `[1, compress_out, grid/4] = [1, 11, 4, 2]`, every entry finite and the
    // map not collapsed to a constant.
    let (ch, cw) = cfg.compress_grid();
    let n = (cfg.compress_out * ch * cw) as usize;
    assert_eq!(n, 11 * 4 * 2);
    let out = h.0.gpu.read(h.0.output(), n);
    assert!(out.iter().all(|v| v.is_finite()), "non-finite compressor output");
    assert!(out.iter().any(|v| (v - out[0]).abs() > 1e-6), "compressor output is constant: {out:?}");
    let dead = gradcheck::zero_grad_params(&h, |_| true);
    assert!(dead.is_empty(), "parameters with an identically-zero gradient: {dead:?}");
}
