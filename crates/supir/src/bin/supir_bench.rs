// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SUPIR profiler: where a forward's time actually goes, per kernel kind.
//!
//! Profile per kernel-kind BEFORE touching anything, and publish the table -
//! measure first, because a confident guess about the bottleneck is wrong
//! more often than not, the same discipline `sdxlunet::unet_bench` already
//! follows for the frozen backbone alone. This bench covers the two
//! things SUPIR adds on top of that backbone: the 1.24B-param `GLVControl`
//! trunk, and the 12 `ZeroSFT`/`ZeroCrossAttn` adaptors fused into the
//! backbone's up path.
//!
//! ## Why this is TWO phases, not one combined submit
//!
//! `supir::model::Supir::new` records trunk + adaptors + frozen backbone into
//! ONE graph (the "one-graph" design: recording both chains into a single
//! `Rec`/`Builder`/`Gpu`/submit avoids a host round-trip for the trunk's
//! control residuals). That IS the real dispatch sequence, and on a card with enough
//! headroom this bench's `full` subcommand replays it verbatim. But on THIS
//! machine (one Intel iGPU, no discrete card) the combined weight set - SDXL
//! backbone 10.27 GB + `GLVControl` trunk 4.97 GB + adaptors 0.22 GB, ~15.5 GB
//! fp32 - is already measured, in `crates/supir/tests/parity.rs`, to hit
//! `wgpu error: Out of Memory` recording the graph, taps off, even under int8
//! HOST storage (the device buffers stay fp32-sized regardless - see that
//! module's own doc). That OOM is driven by total resident weight bytes, not
//! by latent resolution, so no `h`/`w` choice avoids it.
//!
//! So this bench splits the SAME real dispatch sequence into two independent
//! phases, each well inside what this box's backbone-alone bench
//! (`unet_bench`, 10.27 GB, confirmed to fit) already proves fits:
//!
//! * `trunk` - `GLVControl` alone, full real SDXL-shaped `UNetConfig`
//!   (4.97 GB), over its own `Rec`/`Gpu`.
//! * `fused` - the frozen backbone + the 12 adaptors, driving
//!   `sdxlunet::model::Rec`/`Unet::record_into` directly over
//!   `supir::adaptors::Adaptors` as the `SkipFuse` (10.27 GB + 0.22 GB;
//!   `Unet::new_fused`'s own convenience wrapper cannot be used here - see
//!   `run_fused`'s doc), reading the trunk's 10 control tensors from
//!   SHAPE-CORRECT SCRATCH buffers (sized from `UNetConfig::skip_shapes` and
//!   `AdaptorConfig::mid`, the model's own real per-join channel/resolution
//!   schedule) rather than a real trunk forward - the trunk's own cost is
//!   exactly what the `trunk` phase already
//!   measures, so re-running it here would double-count it, not add fidelity.
//!
//! Each phase is a full, real, unmodified production code path
//! (`supir::trunk::record`, `Unet::record_into` + `supir::adaptors::Adaptors`) -
//! nothing about the KERNELS dispatched or their shapes is approximated, only
//! which two phases share one process. The default (no subcommand) runs both
//! phases in one process, each in its own scope so the first phase's ~5 GB of
//! weight buffers are dropped before the second's are uploaded, and prints a
//! merged share-of-combined-total table alongside each phase's own. The
//! merged total pays two submit/drain round trips instead of the real graph's
//! one, so it is a slight OVER-estimate of the true fused wall time - noted
//! once here rather than at every call site.
//!
//! Usage:
//!   supir_bench [h w] [reps]          # both phases, merged table (default 32 32 3)
//!   supir_bench trunk [h w] [reps]    # GLVControl alone
//!   supir_bench fused [h w] [reps]    # backbone + adaptors alone
//!   supir_bench full [h w] [reps]     # the TRUE one-graph Supir::new - needs
//!                                     # BRAIN_SUPIR_ALLOW_FULL_MEMORY=1 and a
//!                                     # box with enough device memory; this
//!                                     # box does not have it (see above)
//!
//! ## A pre-existing measurement artifact, observed not introduced here
//!
//! On this box's adapter (`Intel(R) Arc(tm) Graphics (MTL)`, Vulkan), the
//! per-row `ms`/`GFLOP/s`/`%roof` columns [`gpu_core::profile::profile`]'s
//! device-timed path prints for this crate's kernel set (and
//! `sdxlunet::unet_bench`'s - reproduced there too, unmodified) are off by
//! many orders of magnitude (values like `1e16`-`1e17` "ms" for a pass whose
//! real wall time is under 3 seconds). The WHOLE-PASS number
//! (`PassProfile::total_secs`, from `best_of`'s wall-clock `poll_wait`-bracketed
//! timing) and the per-row RATIOS (each row's share of the corrupted domain's
//! own sum, which cancels a shared corruption factor) both stayed sane and are
//! what this bench's own analysis and `[print_merged]`'s percentage column
//! rely on - never the absolute per-row `ms` `gpu_core::profile::PassProfile::
//! print` prints. This is a `gpu-core`/`backend-wgpu` timestamp-query
//! conversion defect, cross-cutting and pre-existing, not a SUPIR issue and
//! out of this crate's scope to fix - recorded here, and in this port's own
//! ledger, so a future reader does not re-diagnose it from scratch.

use std::collections::HashMap;
use std::time::Instant;

use gpu_core::Gpu;
use sdxlunet::config::UNetConfig;
use sdxlunet::model::{Rec, Unet};
use vae::blocks::skipfuse::{Map, SkipFuse};

use supir::adaptors::Adaptors;
use supir::config::SupirConfig;
use supir::model::KERNELS;

/// `[h w]` in latent units, `reps` for `best_of`. The established small-latent
/// precedent this whole port gates parity at on this machine (32x32,
/// non-square variants also used elsewhere) - not a bespoke shrink invented
/// for this bench - and it is the size `unet_bench`
/// already proved the 10.27 GB backbone alone fits at.
const DEFAULT_HW: u32 = 32;
const DEFAULT_REPS: usize = 3;
const DEFAULT_T_ENC: u32 = 77;

fn parse_hw_reps(a: &[String], start: usize) -> (u32, u32, usize) {
    let h: u32 = a.get(start).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_HW);
    let w: u32 = a.get(start + 1).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_HW);
    let reps: usize = a.get(start + 2).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_REPS);
    (h, w, reps)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("trunk") => {
            let (h, w, reps) = parse_hw_reps(&a, 2);
            run_trunk(h, w, reps);
        }
        Some("fused") => {
            let (h, w, reps) = parse_hw_reps(&a, 2);
            run_fused(h, w, reps);
        }
        Some("full") => {
            let (h, w, reps) = parse_hw_reps(&a, 2);
            run_full(h, w, reps);
        }
        _ => {
            let (h, w, reps) = parse_hw_reps(&a, 1);
            eprintln!("supir_bench: trunk + fused, latent {h}x{w}, {reps} reps\n");
            let trunk = run_trunk(h, w, reps);
            let fused = run_fused(h, w, reps);
            print_merged(&trunk, &fused);
        }
    }
}

/// One phase's result: what [`print_merged`] needs, kept separate from
/// [`gpu_core::profile::PassProfile`] itself so nothing here depends on that
/// struct's field layout beyond what is actually public (it is - this is
/// just naming the subset).
struct Phase {
    label: &'static str,
    profile: gpu_core::profile::PassProfile,
}

fn roofs(gpu: &Gpu) -> Option<gpu_core::roof::Roofs> {
    match gpu_core::roof::ensure(gpu) {
        Some(r) => {
            println!(
                "measured roofline: {:.0} GFLOP/s, {:.1} GB/s DRAM, {:.1} GB/s cache, ridge {:.1} FLOP/byte",
                r.gflops, r.gbs, r.cache_gbs, r.ridge()
            );
            Some(r)
        }
        None => {
            println!("roofline unmeasured - utilisation columns print '-' rather than a guess");
            None
        }
    }
}

fn print_defects(p: &gpu_core::profile::PassProfile, rf: Option<gpu_core::roof::Roofs>) {
    p.print(rf);
    if let Some(r) = rf {
        for (row, bound, pct) in p.defects(r, 5.0) {
            println!(
                "  DEFECT  {:<24} {:>5.1}% of its {} roof (floor {:.0}%) - {:.1}% of this pass",
                row.name, pct, bound.as_str(), bound.defect_pct(),
                100.0 * row.secs / p.summed_secs,
            );
        }
    }
}

/// `GLVControl` alone: the trunk's own down path + mid block, full real SDXL
/// shape (`UNetConfig::sdxl_base()`), 1.243B params / 4.97 GB fp32 (matches
/// the real checkpoint's own measured `model.control_model.*` param count) -
/// the same weights `supir::trunk::record` and `Supir::build` dispatch, just
/// over its own `Rec`/`Gpu` rather than one shared with the backbone.
fn run_trunk(h: u32, w: u32, reps: usize) -> Phase {
    let cfg = UNetConfig::sdxl_base();
    let manifest = supir::config::trunk_manifest(&cfg);
    eprintln!("trunk: {} tensors, building synthetic weights ...", manifest.len());
    let tensors = supir::init::init_weights_for(&manifest, 11);
    let params: usize = tensors.values().map(|(_, d)| d.len()).sum();
    eprintln!("trunk: {params} params = {:.2} GB fp32, latent {h}x{w}", params as f64 * 4.0 / 1e9);

    let gpu = Gpu::new(&KERNELS);
    let c0 = cfg.block_out_channels[0];
    let sample_in = gpu.storage((cfg.in_channels * h * w) as u64);
    let hint_in = gpu.storage((4 * h * w) as u64);
    let enc_in = gpu.storage((DEFAULT_T_ENC * cfg.cross_attention_dim) as u64);
    let temb_in = gpu.storage(c0 as u64);
    let aug_in = gpu.storage(cfg.projection_class_embeddings_input_dim as u64);

    let t0 = Instant::now();
    let mut r = Rec::new(&gpu, &cfg, &tensors, DEFAULT_T_ENC, false);
    r.set_prefix("control_model.");
    let hs = supir::trunk::record(&mut r, &cfg, "control_model.", h, w, &enc_in, &hint_in, &sample_in, &temb_in, &aug_in);
    let _ = r.take_temb_act();
    let (steps, _taps) = r.into_blocks().finish();
    eprintln!("trunk: built in {:.1}s, {} dispatches, {} control outputs\n", t0.elapsed().as_secs_f32(), steps.len(), hs.len());

    let rf = roofs(&gpu);
    let p = gpu_core::profile::profile(&gpu, "TRUNK", &steps, reps);
    print_defects(&p, rf);
    println!("trunk forward: {:.2} ms\n", p.total_secs * 1e3);
    Phase { label: "trunk", profile: p }
}

/// The frozen backbone + the 12 adaptors, over [`supir::adaptors::Adaptors`] -
/// the SAME `SkipFuse` implementor `supir::model::Supir::build` installs. The
/// 10 control tensors `Adaptors` reads are SHAPE-CORRECT SCRATCH (uninitialised
/// device buffers sized from the backbone's own `skip_shapes`/`AdaptorConfig::
/// mid`), not a real trunk output - see the module doc for why that does not
/// lose fidelity here.
///
/// Drives `Rec`/`Unet::record_into` directly rather than `Unet::new_fused`:
/// `Adaptors` needs `Builder::mix` (the ZeroSFT/ZeroCrossAttn lerp), which
/// needs `Builder::set_mix_ids` resolved to THIS Gpu's `edm_mix`/`scale_row`
/// slots first - `new_fused` has no way to take that (it does not know a
/// SUPIR-specific kernel extends its caller's set), so `Supir::build` itself
/// does not call it either. This mirrors `Supir::build`'s real sequence for
/// the backbone half exactly, just without the trunk half sharing the tape.
fn run_fused(h: u32, w: u32, reps: usize) -> Phase {
    let cfg = SupirConfig::sdxl();
    let mut tensors = sdxlunet::init::init_weights(&cfg.backbone, 17);
    eprintln!("fused: backbone {} tensors, adding {} adaptor tensors ...", tensors.len(), cfg.adaptors.tensor_manifest().len());
    tensors.extend(supir::init::init_weights_for(&cfg.adaptors.tensor_manifest(), 19));
    let params: usize = tensors.values().map(|(_, d)| d.len()).sum();
    eprintln!("fused: {params} params = {:.2} GB fp32, latent {h}x{w}", params as f64 * 4.0 / 1e9);

    let gpu = Gpu::new(&KERNELS);
    let hs = synthetic_control_tensors(&gpu, &cfg, h, w);
    let adaptors = Adaptors::new(cfg.adaptors.clone(), hs, 1.0);
    for (name, _) in adaptors.kernels() {
        assert!(gpu.kernel_index(name).is_some(), "supir_bench: fused phase needs the `{name}` kernel");
    }

    let c0 = cfg.backbone.block_out_channels[0];
    let sample_in = gpu.storage((cfg.backbone.in_channels * h * w) as u64);
    let enc_in = gpu.storage((DEFAULT_T_ENC * cfg.backbone.cross_attention_dim) as u64);
    let temb_in = gpu.storage(c0 as u64);
    let aug_in = gpu.storage(cfg.backbone.projection_class_embeddings_input_dim as u64);

    let t0 = Instant::now();
    let mut r = Rec::new(&gpu, &cfg.backbone, &tensors, DEFAULT_T_ENC, false);
    let fwd = gpu.kernel_index("edm_mix").expect("edm_mix registered");
    let bwd = gpu.kernel_index("scale_row").expect("scale_row registered");
    r.blocks().set_mix_ids(vae::blocks::MixIds { fwd, bwd });
    r.set_fuse(&adaptors);
    let inputs = sdxlunet::model::Inputs { sample_in, enc_in, temb_in, aug_in };
    let _recorded = Unet::record_into(&mut r, &cfg.backbone, h, w, &inputs, false);
    let (steps, _taps) = r.into_blocks().finish();
    eprintln!("fused: built in {:.1}s, {} dispatches\n", t0.elapsed().as_secs_f32(), steps.len());

    let rf = roofs(&gpu);
    let p = gpu_core::profile::profile(&gpu, "FUSED (backbone + adaptors)", &steps, reps);
    print_defects(&p, rf);
    println!("fused forward: {:.2} ms\n", p.total_secs * 1e3);
    Phase { label: "fused", profile: p }
}

/// The trunk's 10 control outputs (`control[0..9]`), as shape-correct scratch:
/// sized from [`UNetConfig::skip_shapes`] (the trunk's own down-path push
/// order - `control[0]` is the FINEST resolution, same order
/// `crate::trunk::record` appends `hs` in) and
/// [`supir::adaptors::AdaptorConfig`]'s own `mid` spec (the post-mid site,
/// appended last) - the model's real per-join channel/resolution schedule,
/// not a guess. `JoinSpec::control_idx = n_joins - 1 - k` already does the
/// pop-order reversal on the READ side (`Adaptors::fuse_skip`), so `hs`
/// itself must stay in push order - reversing it here would double-reverse.
fn synthetic_control_tensors(gpu: &Gpu, cfg: &SupirConfig, h: u32, w: u32) -> Vec<Map> {
    let shapes = cfg.backbone.skip_shapes(h, w);
    let mut hs: Vec<Map> = shapes
        .into_iter()
        .map(|(c, ch, cw)| Map { buf: gpu.storage((c * ch * cw) as u64), c, h: ch, w: cw })
        .collect();
    let scale = 1u32 << (cfg.backbone.levels() - 1);
    let (mh, mw) = (h / scale, w / scale);
    let mc = cfg.adaptors.mid.c;
    hs.push(Map { buf: gpu.storage((mc * mh * mw) as u64), c: mc, h: mh, w: mw });
    assert_eq!(hs.len(), cfg.adaptors.joins.len() + 1, "supir_bench: control tensor count vs the adaptor schedule");
    hs
}

/// The TRUE one-graph `Supir::new` dispatch, verbatim - what `supir::model`
/// actually records for a real restoration call. Gated behind
/// `BRAIN_SUPIR_ALLOW_FULL_MEMORY=1`, same convention as
/// `crates/supir/tests/parity.rs`'s own full-forward tests: on THIS box it is
/// measured to hit `wgpu error: Out of Memory` recording the combined ~15.5 GB
/// weight set regardless of `h`/`w` (see the module doc), so it is not run by
/// default - a `full` invocation without the env var costs nothing rather
/// than crashing the bench process.
fn run_full(h: u32, w: u32, reps: usize) {
    if std::env::var_os("BRAIN_SUPIR_ALLOW_FULL_MEMORY").is_none_or(|v| v.is_empty() || v == "0") {
        println!(
            "BRAIN_SUPIR_ALLOW_FULL_MEMORY unset - the combined trunk+adaptors+backbone graph's \
             device-resident weight buffers (~15.5 GB fp32) are measured to hit \
             `wgpu error: Out of Memory` on this box's Intel iGPU (2047 MiB per-buffer cap, one \
             Intel iGPU sharing 30 GB system RAM, no discrete card) regardless of latent size - \
             see `crates/supir/tests/parity.rs`'s own full-forward tests and this bench's module \
             doc. Use `supir_bench trunk`/`supir_bench fused` (the default, no subcommand) for a \
             profile that fits this hardware, or set BRAIN_SUPIR_ALLOW_FULL_MEMORY=1 on a machine \
             with enough device memory to attempt the real combined graph."
        );
        return;
    }
    let cfg = SupirConfig::sdxl();
    eprintln!("full: building synthetic weights for the WHOLE {} -tensor manifest ...", cfg.tensor_manifest().len() + sdxlunet::config::UNetConfig::sdxl_base().tensor_manifest().len());
    let mut tensors = sdxlunet::init::init_weights(&cfg.backbone, 11);
    tensors.extend(supir::init::init_weights(&cfg, 13));
    let params: usize = tensors.values().map(|(_, d)| d.len()).sum();
    eprintln!("full: {params} params = {:.2} GB fp32, latent {h}x{w}", params as f64 * 4.0 / 1e9);

    let gpu = Gpu::new(&KERNELS);
    let t0 = Instant::now();
    let m = supir::model::Supir::new(gpu.share(), cfg, &tensors, h, w, DEFAULT_T_ENC, false, 1.0);
    eprintln!("full: built in {:.1}s, {} dispatches\n", t0.elapsed().as_secs_f32(), m.steps().len());

    let rf = roofs(&gpu);
    let p = gpu_core::profile::profile(&gpu, "FULL (one graph)", m.steps(), reps);
    print_defects(&p, rf);
    println!("full forward: {:.2} ms\n", p.total_secs * 1e3);
}

/// The §F.1 table over the COMBINED trunk + fused dispatch: rows merged by
/// kernel name (the two phases share most of their kernel set - `matmul_reg3`,
/// `gn_apply`, the cross-attention family), ranked by summed per-kernel time
/// across both. This is the number "attack by share of time" reads for SUPIR
/// as a whole - a single phase's own table only ranks that phase's kernels
/// against that phase's total, which can hide a kernel that is small in the
/// trunk but the top row once the (larger) fused phase is folded in.
///
/// The percentage column is each row's share of the SUM OF ROWS, not of the
/// real (reliable, `best_of`-measured) combined wall time printed in the
/// header - on this box's adapter `KernelRow::secs` (from
/// `Gpu::kernel_times`'s device timestamps) has been observed to read many
/// orders of magnitude off the wall clock for this kernel set (reproduces on
/// `sdxlunet::unet_bench` too, so it predates this bench and is not a SUPIR
/// defect - see the module doc's closing note). Dividing by the wall time
/// would silently launder that corruption into a nonsense percentage; dividing
/// by the same corrupted domain the numerator lives in keeps the RATIO
/// meaningful even though neither number is a real millisecond count.
fn print_merged(trunk: &Phase, fused: &Phase) {
    let mut by_name: HashMap<String, (f64, usize, u64, u64, u64, bool)> = HashMap::new();
    for phase in [trunk, fused] {
        for r in &phase.profile.rows {
            let e = by_name.entry(r.name.clone()).or_insert((0.0, 0, 0, 0, 0, true));
            e.0 += r.secs;
            e.1 += r.calls;
            e.2 += r.flops;
            e.3 += r.int_ops;
            e.4 += r.bytes;
            e.5 &= r.covered;
        }
    }
    let combined_total = trunk.profile.total_secs + fused.profile.total_secs;
    let mut rows: Vec<_> = by_name.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
    let row_domain_total: f64 = rows.iter().map(|(_, (secs, ..))| secs).sum();

    println!("\n=== MERGED: {} + {}, {:.2} ms combined wall time ({:.2} ms trunk + {:.2} ms fused) ===", trunk.label, fused.label, combined_total * 1e3, trunk.profile.total_secs * 1e3, fused.profile.total_secs * 1e3);
    println!("(sum of two separate submits - pays one extra submit/drain round trip the real \
               one-graph `Supir::new` does not, so this over-estimates the true fused wall time.)");
    println!("{:<26} {:>6} {:>14}", "kernel", "n", "% of row total");
    println!("{}", "-".repeat(50));
    for (name, (secs, calls, _flops, _int_ops, _bytes, _covered)) in rows.iter().take(15) {
        println!("{:<26} {:>6} {:>13.1}%", name, calls, 100.0 * secs / row_domain_total);
    }
    println!("\ntrunk share of combined WALL time: {:.1}%  fused share: {:.1}%", 100.0 * trunk.profile.total_secs / combined_total, 100.0 * fused.profile.total_secs / combined_total);
}
