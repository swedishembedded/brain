// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Phase 8: the AdaLN-precompute checkpoint transform - what makes MiniMax-H3
//! fit two 24GB P40s.
//!
//! `adaln_proj` is one `Linear(time_embed_dim, 6*hidden*MODALITY_NUM)` PER
//! BLOCK (260M params/block x 50 blocks = 13.0B of the 33B total, confirmed
//! from the real checkpoint's own safetensors header). Its INPUT (`temb`,
//! `crate::model`'s per-forward `silu(time_embedder(time_proj(timestep)))`)
//! is a function of nothing but the scalar timestep - and at inference the
//! only timesteps that ever occur are the finite set the DiT's own dual
//! schedule (`crate::schedule::DualSchedule`) produces for a chosen step
//! count `N`. So `adaln_proj`'s entire output space at inference is exactly
//! `N * MODALITY_NUM` rows, computable once, ahead of time - after which the
//! 260M-param/block weight matrix itself is dead weight: every forward only
//! ever needed a `gather` (`crate::block::gather_rows`, already how the
//! per-row lookup happens today - see `crate::block`'s own doc) against a
//! table, not a `Linear`.
//!
//! This module IS that transform: a plain function from one checkpoint's
//! `Tensors` map (+ config + a step-count-fixed schedule) to a smaller one,
//! following `checkpoint::quantize`'s own design shape (a policy-free
//! generic transform elsewhere would not fit - this one is genuinely
//! H3-specific, hence living here rather than in `crates/checkpoint`) -
//! never a runtime hack folded into `crate::model::H3Transformer::forward`.
//! Deliberately GPU-free: [`convert`] never opens a [`crate::block::Ctx`],
//! reading/writing only the flat `Tensors` map every other import/export
//! tool in this workspace already speaks.
//!
//! Swedish Embedded AB implements checkpoint-shrinking transforms like this
//! one for its clients running large diffusion transformers on constrained
//! hardware. If your team needs expertise in fitting oversized checkpoints
//! onto a fixed GPU budget, you can procure our services by sending an email
//! to info@swedishembedded.com.

use std::collections::HashMap;

use model::hostmath::{linear_rows, silu_slice, timestep_embedding};
use vae::blocks::Tensors;

use crate::config::{H3TransformerConfig, MODALITY_NUM, TAG_AUDIO, TAG_TEXT, TAG_VIDEO};
use crate::schedule::DualSchedule;

/// `Timesteps`/`TimestepEmbedding`'s own `max_period` - see `crate::model`'s
/// identical constant of the same name. Duplicated rather than imported:
/// this module deliberately works from the raw checkpoint [`Tensors`] map,
/// never a loaded `H3Transformer`/GPU instance (this is a checkpoint
/// transform tool, not an inference path).
const TIME_MAX_PERIOD: f64 = 10000.0;

/// The key suffix this module writes each block's precomputed table under -
/// full name `transformer_blocks.{i}.{ADALN_TABLE_KEY}`.
pub const ADALN_TABLE_KEY: &str = "adaln_precomputed";

/// The substring that names one block's (now-droppable) `adaln_proj`
/// weight/bias pair, shared by [`convert`]'s drop pass and its per-block
/// read.
const ADALN_PROJ_INFIX: &str = ".adaln_proj.linear.";

/// [`convert`]'s report: exactly what it decided and measured, so a caller
/// (or a test) can assert the transform actually shrinks the checkpoint
/// rather than trusting it silently - the same discipline
/// `checkpoint::quantize::Report` already applies to its own conversion.
#[derive(Clone, Debug)]
pub struct Report {
    pub num_blocks: usize,
    pub num_steps: usize,
    pub hidden: u32,
    /// `adaln_proj.linear.{weight,bias}` bytes (as f32) summed across every
    /// block, BEFORE the transform.
    pub adaln_bytes_before: u64,
    /// The precomputed tables' bytes (as f32) summed across every block,
    /// AFTER the transform.
    pub adaln_bytes_after: u64,
}

impl Report {
    /// Bytes the transform removed (positive) or added (negative, only
    /// possible at a `num_steps` large enough that `MODALITY_NUM*num_steps*
    /// 6*hidden` exceeds `6*hidden*MODALITY_NUM*time_embed_dim` - i.e. more
    /// distinct steps than `time_embed_dim`, far outside any real denoise
    /// step count).
    pub fn saved_bytes(&self) -> i64 {
        self.adaln_bytes_before as i64 - self.adaln_bytes_after as i64
    }
}

/// Row order for the `[MODALITY_NUM*num_steps, ...]` grid this module builds
/// and every precomputed table addresses: MODALITY-major, STEP-minor -
/// `row = modality*num_steps + step` - the SAME convention
/// `crate::model::adaln_indices` already established for the un-precomputed
/// path (see `crate::block`'s own doc for why: it keeps each modality's
/// `6*hidden` output-feature slice of the fused `adaln_proj.linear` weight a
/// CONTIGUOUS row range, so precomputing needs no interleaving/scatter
/// kernel - a plain per-modality batched matmul over that modality's own
/// `num_steps` grid rows).
pub fn table_row(modality: u32, step: usize, num_steps: usize) -> usize {
    modality as usize * num_steps + step
}

/// The raw (pre-`time_proj`) timestep scalar every `(step, modality)` grid
/// row uses, read from the DiT's own dual schedule (`crate::schedule::
/// DualSchedule`, already `set_timesteps`-called by the caller for `N`
/// steps - `schedule.num_steps()` fixes the grid size [`convert`]'s table
/// covers). Returned in [`table_row`]'s own row order, length
/// `MODALITY_NUM * schedule.num_steps()`.
///
/// TEXT rows have no independent third schedule of their own: `t2va`/`fl2va`
/// pack text conditioning alongside the video target within one forward
/// (`transformer_minimax_h3.py`'s packed-sequence row layout - see the
/// roadmap's "packed row order" entry), so this reuses the VIDEO schedule's
/// timestep at each step for text's grid row. This is this module's OWN
/// documented choice, not yet independently confirmed from the reference's
/// still-unread packing code (`MiniMaxH3PrepareLayoutStep`, open per the
/// roadmap) - but the precompute mechanism's own correctness (an exact
/// cache of `adaln_proj`'s forward at whichever grid points it is built
/// over) does not depend on which scalar is chosen here, only on this
/// module and the inference-time row lookup agreeing on it. Should Phase 9
/// settle on a different text timestep, only this one function's body needs
/// to change - [`precompute_block_table`] and [`convert`] are agnostic to
/// where the grid's values came from.
pub fn timestep_grid(schedule: &DualSchedule) -> Vec<f32> {
    let n = schedule.num_steps();
    let mut grid = vec![0f32; MODALITY_NUM as usize * n];
    for step in 0..n {
        let (video_ts, audio_ts) = schedule.current_timesteps(step);
        grid[table_row(TAG_VIDEO, step, n)] = video_ts;
        grid[table_row(TAG_TEXT, step, n)] = video_ts;
        grid[table_row(TAG_AUDIO, step, n)] = audio_ts;
    }
    grid
}

/// `Linear(inn,out).forward` on a single row: `y = w@x + b`, `w` is `[out,
/// inn]` row-major (PyTorch `nn.Linear` convention, matching `crate::model`'s
/// identically-shaped private helper of the same job).
fn linear_row(x: &[f32], w: &[f32], b: &[f32], inn: usize, out: usize) -> Vec<f32> {
    let mut y = linear_rows(x, w, 1, inn, out);
    for (yi, bi) in y.iter_mut().zip(b) {
        *yi += bi;
    }
    y
}

/// `silu(time_embedder(time_proj(t)))` for ONE raw timestep scalar - the
/// exact same formula as `crate::model::build_temb`'s per-timestep body
/// (duplicated, not imported: see this module's own doc for why it never
/// depends on a loaded `H3Transformer`).
fn temb_silu_row(cfg: &H3TransformerConfig, t: f32, w0: &[f32], b0: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    let sinusoid = timestep_embedding(t, cfg.freq_dim as usize, true, 0.0, TIME_MAX_PERIOD);
    let h0 = silu_slice(&linear_row(&sinusoid, w0, b0, cfg.freq_dim as usize, cfg.time_embed_hidden_dim as usize));
    let out = linear_row(&h0, w2, b2, cfg.time_embed_hidden_dim as usize, cfg.time_embed_dim as usize);
    silu_slice(&out)
}

/// Precompute ONE block's `[MODALITY_NUM*num_steps, 6*hidden]` AdaLN table -
/// `adaln_proj.linear`'s forward, evaluated at every grid row and sliced to
/// that row's own modality's `6*hidden` output-feature range (a plain
/// per-modality batched `linear_rows` call over that modality's `num_steps`
/// rows of `temb_silu_grid`, per this module's own doc on why that slice is
/// contiguous). `temb_silu_grid` is [`timestep_grid`]'s output already run
/// through [`temb_silu_row`], same `table_row` order, one `Vec<f32>` (width
/// `time_embed_dim`) per grid row.
fn precompute_block_table(cfg: &H3TransformerConfig, adaln_w: &[f32], adaln_b: &[f32], temb_silu_grid: &[Vec<f32>], num_steps: usize) -> Vec<f32> {
    let hidden = cfg.hidden_size as usize;
    let time_embed_dim = cfg.time_embed_dim as usize;
    let sextet = 6 * hidden;
    let mut table = vec![0f32; MODALITY_NUM as usize * num_steps * sextet];

    for modality in 0..MODALITY_NUM as usize {
        let w_slice = &adaln_w[modality * sextet * time_embed_dim..(modality + 1) * sextet * time_embed_dim];
        let b_slice = &adaln_b[modality * sextet..(modality + 1) * sextet];

        // Flatten this modality's `num_steps` temb rows into one contiguous
        // buffer so the whole modality precomputes in one batched matmul,
        // rather than `num_steps` separate single-row calls.
        let mut x = Vec::with_capacity(num_steps * time_embed_dim);
        for step in 0..num_steps {
            x.extend_from_slice(&temb_silu_grid[table_row(modality as u32, step, num_steps)]);
        }
        let y = linear_rows(&x, w_slice, num_steps, time_embed_dim, sextet);

        for step in 0..num_steps {
            let row = table_row(modality as u32, step, num_steps);
            let dst = &mut table[row * sextet..(row + 1) * sextet];
            let src = &y[step * sextet..(step + 1) * sextet];
            for c in 0..sextet {
                dst[c] = src[c] + b_slice[c];
            }
        }
    }
    table
}

/// The Phase 8 checkpoint transform: replace every block's `adaln_proj`
/// weight matrix (260M params/block at the real config, 13.0B of the 33B
/// total) with a small precomputed `[MODALITY_NUM*num_steps, 6*hidden]`
/// lookup table, built by running `adaln_proj`'s own forward over the finite
/// `(step, modality)` grid `schedule` (already `set_timesteps`-called for
/// `num_steps = schedule.num_steps()` denoise steps) actually produces -
/// see [`timestep_grid`] for exactly which scalars that is.
///
/// Every OTHER tensor in `tensors` passes through byte-for-byte unchanged:
/// this is a pure drop-and-replace of `transformer_blocks.{i}.adaln_proj.
/// linear.{weight,bias}` per block, nothing else in the checkpoint is
/// touched (`time_embedder`/`norm_out` stay - they are shared across every
/// block and already small, and this phase's own scope is `adaln_proj`
/// specifically, per the roadmap's "13.0B of the 33B total" line).
///
/// Errors (never silently skips a block) if `time_embedder`'s tensors or any
/// block's `adaln_proj` tensors are missing or the wrong element count for
/// `cfg` - the same two-way-coverage discipline `checkpoint::quantize`
/// applies to its own conversion.
pub fn convert(tensors: &Tensors, cfg: &H3TransformerConfig, schedule: &DualSchedule) -> Result<(Tensors, Report), String> {
    let num_steps = schedule.num_steps();
    if num_steps == 0 {
        return Err("precompute_adaln::convert: schedule has zero steps - call DualSchedule::set_timesteps first".to_string());
    }

    let hidden = cfg.hidden_size as usize;
    let time_embed_dim = cfg.time_embed_dim as usize;
    let sextet = 6 * hidden;
    let expected_w_len = MODALITY_NUM as usize * sextet * time_embed_dim;
    let expected_b_len = MODALITY_NUM as usize * sextet;

    let get = |name: &str| -> Result<&(Vec<usize>, Vec<f32>), String> { tensors.get(name).ok_or_else(|| format!("precompute_adaln: missing tensor {name:?}")) };

    let (_, w0) = get("time_embedder.linear_1.weight")?;
    let (_, b0) = get("time_embedder.linear_1.bias")?;
    let (_, w2) = get("time_embedder.linear_2.weight")?;
    let (_, b2) = get("time_embedder.linear_2.bias")?;

    let grid = timestep_grid(schedule);
    let temb_silu_grid: Vec<Vec<f32>> = grid.iter().map(|&t| temb_silu_row(cfg, t, w0, b0, w2, b2)).collect();

    // Every tensor passes through unchanged except the per-block
    // `adaln_proj` pair, dropped here and replaced below.
    let mut out: Tensors = HashMap::with_capacity(tensors.len());
    for (name, (shape, data)) in tensors {
        if name.contains(ADALN_PROJ_INFIX) {
            continue;
        }
        out.insert(name.clone(), (shape.clone(), data.clone()));
    }

    let mut adaln_bytes_before = 0u64;
    let mut adaln_bytes_after = 0u64;
    for i in 0..cfg.num_layers {
        let wname = format!("transformer_blocks.{i}.adaln_proj.linear.weight");
        let bname = format!("transformer_blocks.{i}.adaln_proj.linear.bias");
        let (wshape, w) = get(&wname)?;
        let (bshape, b) = get(&bname)?;
        if w.len() != expected_w_len {
            return Err(format!("precompute_adaln: {wname} has {} elements, expected {expected_w_len} (shape {wshape:?})", w.len()));
        }
        if b.len() != expected_b_len {
            return Err(format!("precompute_adaln: {bname} has {} elements, expected {expected_b_len} (shape {bshape:?})", b.len()));
        }
        adaln_bytes_before += (w.len() + b.len()) as u64 * 4;

        let table = precompute_block_table(cfg, w, b, &temb_silu_grid, num_steps);
        adaln_bytes_after += table.len() as u64 * 4;
        out.insert(format!("transformer_blocks.{i}.{ADALN_TABLE_KEY}"), (vec![MODALITY_NUM as usize * num_steps, sextet], table));
    }

    Ok((out, Report { num_blocks: cfg.num_layers as usize, num_steps, hidden: cfg.hidden_size, adaln_bytes_before, adaln_bytes_after }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(rng: &mut data::rng::Lcg, n: usize) -> Vec<f32> {
        rng.vec_scaled(n, 0.2)
    }

    /// A minimal `Tensors` map: `time_embedder.*` plus `num_layers` blocks'
    /// `adaln_proj.linear.*`, sized to `cfg`, plus one unrelated passthrough
    /// tensor to prove [`convert`] leaves everything else untouched.
    fn rand_tensors(cfg: &H3TransformerConfig, seed: u64) -> Tensors {
        let mut rng = data::rng::Lcg::new(seed);
        let (hidden, te, thd, freq) = (cfg.hidden_size as usize, cfg.time_embed_dim as usize, cfg.time_embed_hidden_dim as usize, cfg.freq_dim as usize);
        let sextet_all = 6 * hidden * MODALITY_NUM as usize;

        let mut t: Tensors = HashMap::new();
        t.insert("time_embedder.linear_1.weight".to_string(), (vec![thd, freq], rand_vec(&mut rng, thd * freq)));
        t.insert("time_embedder.linear_1.bias".to_string(), (vec![thd], rand_vec(&mut rng, thd)));
        t.insert("time_embedder.linear_2.weight".to_string(), (vec![te, thd], rand_vec(&mut rng, te * thd)));
        t.insert("time_embedder.linear_2.bias".to_string(), (vec![te], rand_vec(&mut rng, te)));
        for i in 0..cfg.num_layers {
            t.insert(format!("transformer_blocks.{i}.adaln_proj.linear.weight"), (vec![sextet_all, te], rand_vec(&mut rng, sextet_all * te)));
            t.insert(format!("transformer_blocks.{i}.adaln_proj.linear.bias"), (vec![sextet_all], rand_vec(&mut rng, sextet_all)));
        }
        t.insert("proj_in.weight".to_string(), (vec![hidden, cfg.video_patch_dim() as usize], rand_vec(&mut rng, hidden * cfg.video_patch_dim() as usize)));
        t
    }

    fn tiny_schedule(num_steps: usize) -> DualSchedule {
        // +1: `DualSchedule::set_timesteps` takes `num_inference_steps` (the
        // sigma grid's own point count, terminal 0.0 included) and drives
        // one fewer model evaluation - see `crate::schedule`'s own doc.
        let mut sched = DualSchedule::new();
        sched.set_timesteps(num_steps + 1);
        sched
    }

    /// A fully independent reference computation of `adaln_proj`'s forward
    /// at ONE grid row - plain nested loops, no `linear_rows`/batching, no
    /// reuse of anything [`precompute_block_table`] itself calls - the
    /// cross-check this module's own numeric-identity test needs.
    fn reference_adaln_row(cfg: &H3TransformerConfig, adaln_w: &[f32], adaln_b: &[f32], modality: u32, temb_silu: &[f32]) -> Vec<f32> {
        let hidden = cfg.hidden_size as usize;
        let te = cfg.time_embed_dim as usize;
        let sextet = 6 * hidden;
        let feat0 = modality as usize * sextet;
        let mut y = vec![0f32; sextet];
        for (o, yo) in y.iter_mut().enumerate() {
            let w_row = &adaln_w[(feat0 + o) * te..(feat0 + o + 1) * te];
            let mut acc = 0f64;
            for c in 0..te {
                acc += w_row[c] as f64 * temb_silu[c] as f64;
            }
            *yo = acc as f32 + adaln_b[feat0 + o];
        }
        y
    }

    #[test]
    fn table_row_is_modality_major_step_minor() {
        assert_eq!(table_row(TAG_VIDEO, 0, 5), 0);
        assert_eq!(table_row(TAG_VIDEO, 4, 5), 4);
        assert_eq!(table_row(TAG_TEXT, 0, 5), 5);
        assert_eq!(table_row(TAG_AUDIO, 2, 5), 12);
    }

    #[test]
    fn timestep_grid_reuses_video_timestep_for_text_rows() {
        let sched = tiny_schedule(4);
        let grid = timestep_grid(&sched);
        let n = sched.num_steps();
        for step in 0..n {
            let (v, a) = sched.current_timesteps(step);
            assert_eq!(grid[table_row(TAG_VIDEO, step, n)], v);
            assert_eq!(grid[table_row(TAG_TEXT, step, n)], v, "text reuses the video schedule's timestep, see this module's own doc");
            assert_eq!(grid[table_row(TAG_AUDIO, step, n)], a);
        }
    }

    /// The transform actually drops `adaln_proj` and adds the table, leaves
    /// everything else byte-identical, and errors two-way (missing weight,
    /// wrong shape) rather than silently skipping a block.
    #[test]
    fn convert_replaces_adaln_proj_and_leaves_everything_else_untouched() {
        let cfg = H3TransformerConfig::tiny();
        let tensors = rand_tensors(&cfg, 11);
        let sched = tiny_schedule(3);

        let (out, report) = convert(&tensors, &cfg, &sched).expect("convert");

        assert_eq!(report.num_blocks, cfg.num_layers as usize);
        assert_eq!(report.num_steps, 3);
        for i in 0..cfg.num_layers {
            assert!(!out.contains_key(&format!("transformer_blocks.{i}.adaln_proj.linear.weight")), "adaln_proj.weight must be dropped");
            assert!(!out.contains_key(&format!("transformer_blocks.{i}.adaln_proj.linear.bias")), "adaln_proj.bias must be dropped");
            let (shape, data) = out.get(&format!("transformer_blocks.{i}.{ADALN_TABLE_KEY}")).expect("precomputed table must be present");
            assert_eq!(shape, &vec![MODALITY_NUM as usize * 3, 6 * cfg.hidden_size as usize]);
            assert_eq!(data.len(), shape.iter().product::<usize>());
        }
        // Untouched passthrough tensor, byte-for-byte.
        assert_eq!(out.get("proj_in.weight"), tensors.get("proj_in.weight"));
        assert_eq!(out.get("time_embedder.linear_1.weight"), tensors.get("time_embedder.linear_1.weight"));

        // Byte count actually drops (exact values checked separately below).
        assert!(report.saved_bytes() > 0, "adaln precompute must shrink the checkpoint at a realistic step count, saved {}", report.saved_bytes());
    }

    #[test]
    fn convert_errors_on_a_missing_block_weight() {
        let cfg = H3TransformerConfig::tiny();
        let mut tensors = rand_tensors(&cfg, 5);
        tensors.remove("transformer_blocks.0.adaln_proj.linear.weight");
        let sched = tiny_schedule(2);
        let err = convert(&tensors, &cfg, &sched).unwrap_err();
        assert!(err.contains("transformer_blocks.0.adaln_proj.linear.weight"), "error must name the missing tensor: {err}");
    }

    #[test]
    fn convert_errors_on_a_wrong_shaped_block_weight() {
        let cfg = H3TransformerConfig::tiny();
        let mut tensors = rand_tensors(&cfg, 6);
        tensors.insert("transformer_blocks.0.adaln_proj.linear.weight".to_string(), (vec![4, 4], vec![0f32; 16]));
        let sched = tiny_schedule(2);
        let err = convert(&tensors, &cfg, &sched).unwrap_err();
        assert!(err.contains("transformer_blocks.0.adaln_proj.linear.weight"), "error must name the mismatched tensor: {err}");
    }

    /// The core numeric claim: the precomputed table's lookup at every
    /// `(step, modality)` grid point is IDENTICAL to running `adaln_proj`'s
    /// real forward at that same point, checked against a fully independent
    /// (`reference_adaln_row`, plain nested loops) computation - never the
    /// same code path as [`precompute_block_table`] itself.
    #[test]
    fn precomputed_table_matches_running_adaln_proj_directly() {
        let cfg = H3TransformerConfig::tiny();
        let tensors = rand_tensors(&cfg, 42);
        let sched = tiny_schedule(5);
        let num_steps = sched.num_steps();

        let (out, _report) = convert(&tensors, &cfg, &sched).expect("convert");

        let (_, w0) = tensors.get("time_embedder.linear_1.weight").unwrap();
        let (_, b0) = tensors.get("time_embedder.linear_1.bias").unwrap();
        let (_, w2) = tensors.get("time_embedder.linear_2.weight").unwrap();
        let (_, b2) = tensors.get("time_embedder.linear_2.bias").unwrap();
        let grid = timestep_grid(&sched);
        let sextet = 6 * cfg.hidden_size as usize;

        let mut worst_max_abs = 0f32;
        for i in 0..cfg.num_layers {
            let (_, adaln_w) = tensors.get(&format!("transformer_blocks.{i}.adaln_proj.linear.weight")).unwrap();
            let (_, adaln_b) = tensors.get(&format!("transformer_blocks.{i}.adaln_proj.linear.bias")).unwrap();
            let (_, table) = out.get(&format!("transformer_blocks.{i}.{ADALN_TABLE_KEY}")).unwrap();

            for modality in 0..MODALITY_NUM {
                for step in 0..num_steps {
                    let row = table_row(modality, step, num_steps);
                    let temb_silu = temb_silu_row(&cfg, grid[row], w0, b0, w2, b2);
                    let want = reference_adaln_row(&cfg, adaln_w, adaln_b, modality, &temb_silu);
                    let got = &table[row * sextet..(row + 1) * sextet];
                    for (g, w) in got.iter().zip(&want) {
                        worst_max_abs = worst_max_abs.max((g - w).abs());
                    }
                    assert_eq!(got.len(), want.len());
                }
            }
        }
        assert!(worst_max_abs < 1e-4, "precomputed table vs directly-run adaln_proj: worst max_abs {worst_max_abs}, expected float32-noise-level agreement");
    }

    /// Byte counts at BOTH the tiny-config scale (measured directly) and the
    /// real config's scale (computed by the same formula this module's
    /// `convert` uses, at a representative `num_steps=50` - real weights are
    /// never loaded here, this is arithmetic on `H3TransformerConfig::real()`'s
    /// own numbers) - confirms the roadmap's "~19.3B backbone" estimate is in
    /// the right neighborhood (13.0B removed from a 33B total leaves ~20B;
    /// this transform alone removes very close to that 13.0B, independent of
    /// everything else in the checkpoint that a later phase's own
    /// quantization pass would further shrink).
    #[test]
    fn byte_counts_drop_at_tiny_scale_and_the_real_config_extrapolates_near_13b_removed() {
        let cfg = H3TransformerConfig::tiny();
        let tensors = rand_tensors(&cfg, 3);
        let sched = tiny_schedule(3);
        let (_, report) = convert(&tensors, &cfg, &sched).expect("convert");

        // Tiny scale, hand-computed: adaln_proj is [6*20*3, 24] weight +
        // [6*20*3] bias = 360*24 + 360 = 8640 + 360 = 9000 elements/block,
        // times 2 blocks x 4 bytes = 72000 bytes before.
        assert_eq!(report.adaln_bytes_before, 72_000, "tiny adaln_proj bytes before");
        // Table: [3*3, 6*20] = [9, 120] = 1080 elements/block, times 2
        // blocks x 4 bytes = 8640 bytes after.
        assert_eq!(report.adaln_bytes_after, 8_640, "tiny precomputed table bytes after");
        assert_eq!(report.saved_bytes(), 72_000 - 8_640);

        // Real config, extrapolated by the identical formula (no real
        // weights loaded - `H3TransformerConfig::real()`'s own numbers are
        // the checkpoint header's, already asserted against the real
        // checkpoint in `config.rs`'s own test).
        let real = H3TransformerConfig::real();
        let sextet = 6 * real.hidden_size as u64; // 32256
        let te = real.time_embed_dim as u64; // 2688
        let modality = MODALITY_NUM as u64; // 3
        let blocks = real.num_layers as u64; // 50
        let adaln_before_params_per_block = modality * sextet * te + modality * sextet; // weight + bias
        let adaln_before_params_total = adaln_before_params_per_block * blocks;
        assert_eq!(adaln_before_params_total, 13_010_457_600, "matches the roadmap's '260M params/block x 50 = 13.0B' line exactly (260,209,152/block)");
        // The real checkpoint stores adaln_proj in bf16 (2 bytes/param), not
        // f32 - this module's own `Report` counts f32-equivalent bytes (what
        // it actually READS the tensor as, per `checkpoint::TensorSource`'s
        // f32-only contract - the same honest-denominator choice
        // `checkpoint::quantize::Report::f32_bytes` makes), so the real
        // on-disk bf16 total this transform removes is half of this bound.
        let adaln_before_bytes_total = adaln_before_params_total * 4;

        let num_steps = 50u64; // a representative denoise step count
        let adaln_after_params_per_block = modality * num_steps * sextet;
        let adaln_after_bytes_total = adaln_after_params_per_block * 4 * blocks;
        assert!(
            adaln_after_bytes_total < adaln_before_bytes_total / 20,
            "precomputed table at N={num_steps} steps must be well under 5% of adaln_proj's own size, got {adaln_after_bytes_total} vs {adaln_before_bytes_total}"
        );

        let removed_params = adaln_before_params_total - adaln_after_params_per_block * blocks;
        // 13.0B removed of a 33B total leaves ~20B - close to the roadmap's
        // own "~19.3B backbone" estimate (the small remaining gap is other,
        // out-of-scope-for-this-phase savings: bf16-vs-f32 on-disk storage,
        // and norm_out/time_embedder's own much smaller tensors).
        assert!(removed_params > 12_700_000_000, "expected precompute to remove north of 12.7B params at N=50, removed {removed_params}");
    }
}
