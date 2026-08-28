// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-checkpoint forward parity for `crates/supir`, against the
//! `--s-churn 0.0` golden dumped into `testdata/supir_forward_parity/`.
//!
//! `tests/schedule_parity.rs` already explains why the ORIGINAL, committed
//! `testdata/supir/` golden cannot be used for a tensor-level forward
//! replay: it was dumped with `s_churn = 5.0`, so every one of its taps
//! reflects a forward pass that consumed a random churn-noise draw the
//! dumper never saved. This suite reads a SEPARATE golden, dumped with
//! `tools/goldens/supir_dump_reference.py --s-churn 0.0` (see that script's
//! `--s-churn` flag), where `gamma` is identically zero at every step -
//! `sigma_hat == sigma`, so the forward at each tapped step is a
//! deterministic function of the fixed seed alone. `Golden::assert_gamma_
//! is_zero` re-checks this against the golden's own `steps.gamma` tap rather
//! than trusting the directory name.
//!
//! ## Scope: the FIRST denoiser call only, both CFG branches
//!
//! Every trunk/adaptor/UNet tap in the golden was captured during ONE real
//! forward - the sampler's very first `denoise()` call, batched as
//! `cat([uncond, cond])` along dim 0 (verified against
//! `sgm/modules/diffusionmodules/guiders.py`'s `torch.cat((uc[k], c[k]), 0)`
//! - uncond is batch index 0, cond is batch index 1). `brain-supir` records
//! its graph at batch 1 (the same design `sdxlunet`/`controlnet` already
//! use: CFG is two separate `run()` calls, not one batched call), so this
//! suite reproduces the golden's batched forward as two batch-1 runs and
//! compares each against its own half of every `[2, ...]`-shaped golden
//! tensor. `x` and the hint (`_z`) are IDENTICAL between the two halves
//! (`in.noised_z` before any churn, and `cond.c.control == cond.uc.control`
//! - both asserted below); only the text conditioning (`crossattn`/
//! `pooled`) differs between the two runs.
//!
//! ## A real finding, carried from the dumper script into this test
//!
//! SUPIR's own `SUPIR_model.py::prepare_condition` hardcodes
//! `original_size_as_tuple`/`target_size_as_tuple` to `[1024, 1024]` and
//! `crop_coords_top_left` to `[0, 0]` - **regardless of the actual LQ image
//! size** (this golden's is 256x256). That is a genuine departure from
//! `sdxlunet::sampler::sample`'s own convention (which uses the GENERATED
//! size), worth carrying into `pipeline.rs` rather than re-deriving from
//! the image size a second time and getting it wrong.
//!
//! ## Two tests, split on memory footprint, not on what they gate
//!
//! [`supir_trunk_matches_the_real_checkpoint`] gates rung 1a (all 10
//! `GLVControl` trunk hidden states) using ONLY the SUPIR delta checkpoint
//! (~5.3 GB fp32) - achievable on this box today, runs by default.
//! [`supir_full_forward_matches_the_real_checkpoint_at_s_churn_zero`] gates
//! rungs 1b (every adaptor's output AND input side) and 2 (single-forward
//! parity against `unet_final_raw_out` - stage parity then a full-forward
//! check, the standard climb this workspace's parity suites all follow),
//! but needs the FULL trunk+adaptors+frozen-backbone graph (~3.9 B parameters)
//! resident at once - measured to exceed this box's 30 GB RAM and get
//! SIGKILL'd by the OOM killer, so it is gated behind an explicit
//! `BRAIN_SUPIR_ALLOW_FULL_MEMORY=1` opt-in and skips itself with a
//! detailed explanation otherwise. See that test's own doc for the measured
//! numbers and why a lower-memory path does not exist yet.
//!
//! Both need real checkpoints, env-gated exactly like `tests/import_real.rs`'s
//! `weights_path` (same `resources/supir/` default layout, overridable):
//! `BRAIN_SUPIR_WEIGHTS` (the SUPIR-v0Q delta) and, for the full test only,
//! `BRAIN_SUPIR_SDXL_LDM` (the single-file `sd_xl_base_1.0*.safetensors` -
//! SUPIR's checkpoint carries no frozen-backbone weights of its own, so the
//! frozen UNet is loaded from the SAME CompVis/LDM-format file the upstream
//! Python reference loads, via the new `sdxlunet::import::load_ldm`). Both
//! skip themselves (never fail) when a checkpoint or the golden is absent.
//!
//! A THIRD and FOURTH test, [`supir_full_forward_int8_fits_this_machine`] and
//! [`supir_full_forward_int8_no_taps_fits_this_machine`], are the int8
//! siblings of the full-memory test above: same combined graph, built via
//! `supir::int8::quantize_tensors` + `Supir::new_quantized`
//! (`crates/supir/src/int8.rs`) instead of a whole-model fp32 map.
//!
//! **What int8 actually closes here, measured**: `supir::int8`/
//! `sdxlunet::int8`'s own module docs are explicit that this tier is a
//! HOST-memory format only - `vae::blocks::Builder::set_packed` dequantizes
//! one packed tensor at a time at upload, but the device buffer it produces
//! is bit-for-bit what the plain fp32 path uploads. On a discrete-GPU box
//! that is still a real win (host peak drops from 15.6 GB to ~5.6 GB, so the
//! import no longer competes with the device upload for the SAME pool of
//! RAM before the graph finishes recording). On THIS box - one Intel iGPU,
//! no discrete card, `wgpu` reports a 2047 MiB per-buffer/per-binding cap -
//! the combined trunk+adaptors+backbone graph's device-resident buffers are
//! still fp32-sized, and recording it hits `wgpu error: Out of Memory`
//! before the forward ever runs (measured, see the two tests below for the
//! exact numbers). Closing that for real needs genuine on-device int8
//! storage with a dequantizing GEMM (the shape `crates/flux1`/`crates/s3dit`
//! already have, via `model::int8` + a DP4A kernel) threaded through
//! `vae::blocks::Builder` - the shared block recorder ~10 architectures
//! (the VAE family, VQGAN, RRDBNet, CodeFormer, DIAMOND, `sdxlunet`,
//! `supir`) depend on, per `sdxlunet::int8`'s own module doc explaining why
//! that was deliberately NOT done as part of this host-memory tier. That is
//! real, scoped follow-up work, not a same-session fix, so both tests below
//! are gated behind the SAME `BRAIN_SUPIR_ALLOW_FULL_MEMORY=1` opt-in the
//! fp32 sibling above uses, and skip themselves (never fail, never claim a
//! false pass) when it is unset - matching this workspace's "residency is
//! not a parity blocker" posture for VRAM-bound models (see `sdxlunet`'s own
//! `native_resolution_fits_one_card`, an `#[ignore]`d measurement-only test
//! for the same reason).

use std::collections::HashMap;
use std::path::PathBuf;

use brain_testutil::golden::Source;
use brain_testutil::parity::Table;
use checkpoint::safetensors::StTensor;
use diffusion::restore::{sigma_hat, DiscreteDenoiserWithControl};
use sdxlunet::model::Rec;
use supir::config::SupirConfig;
use supir::model::{Supir, KERNELS};

/// Per-tap direction gate. The reference is fp32 torch on the CPU; brain's
/// reduction order differs (and this graph is far deeper than
/// `controlnet`'s - a 1.24B trunk AND a 2.6B frozen backbone AND 12
/// adaptors, all in one forward), so exact equality is not the expectation.
const GATE: f64 = 0.999;

/// Per-tap MAGNITUDE gate - not redundant with [`GATE`], see
/// `crates/controlnet/tests/parity.rs`'s identical constant doc: cosine
/// alone cannot see a whole-tensor scale mistake (`control_scale` applied
/// twice, or applied to the wrong operand of a `ZeroSFT` lerp).
const REL_GATE: f64 = 1e-2;

fn testdata_dir() -> PathBuf {
    brain_testutil::testdata_path("supir_forward_parity")
}

/// `BRAIN_SUPIR_WEIGHTS`, else `resources/supir/supir_v0q/SUPIR/SUPIR-v0Q_fp32.safetensors` -
/// exactly `tests/import_real.rs`'s own `weights_path` helper, copied rather
/// than shared (each real-checkpoint test in this crate owns its skip
/// message, and the two vary in which env var and which file they name).
fn supir_weights_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_SUPIR_WEIGHTS") {
        let pb = PathBuf::from(p);
        return pb.is_file().then_some(pb);
    }
    let p = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/supir/supir_v0q/SUPIR/SUPIR-v0Q_fp32.safetensors"));
    p.is_file().then_some(p)
}

/// `BRAIN_SUPIR_SDXL_LDM`, else `resources/supir/sdxl_base/sd_xl_base_1.0_0.9vae.safetensors` -
/// the single-file CompVis/LDM checkpoint [`sdxlunet::import::load_ldm`]
/// reads. Deliberately a DIFFERENT env var from the `BRAIN_SDXL` other
/// crates' tests use: that convention points at a diffusers `unet/`
/// directory, an incompatible layout this test cannot read.
fn sdxl_ldm_weights_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_SUPIR_SDXL_LDM") {
        let pb = PathBuf::from(p);
        return pb.is_file().then_some(pb);
    }
    let p = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/supir/sdxl_base/sd_xl_base_1.0_0.9vae.safetensors"));
    p.is_file().then_some(p)
}

struct Golden(HashMap<String, StTensor>);

impl Golden {
    fn need(&self, name: &str) -> &[f32] {
        self.0.get(name).unwrap_or_else(|| panic!("golden missing {name}")).data.as_slice()
    }

    fn shape(&self, name: &str) -> &[usize] {
        &self.0.get(name).unwrap_or_else(|| panic!("golden missing {name}")).shape
    }

    /// Batch index `b` out of a `[2, ...]` CFG-stacked golden tensor - see
    /// the module doc's "the FIRST denoiser call only" section for why
    /// every trunk/adaptor/UNet tap has this shape.
    fn batch(&self, name: &str, b: usize) -> &[f32] {
        let t = self.0.get(name).unwrap_or_else(|| panic!("golden missing {name}"));
        assert_eq!(t.shape[0], 2, "{name}: expected a batch-2 (CFG-stacked) tensor, got {:?}", t.shape);
        let plane = t.data.len() / 2;
        &t.data[b * plane..(b + 1) * plane]
    }
}

/// Open the golden and prove it was dumped from the SAME architecture
/// [`SupirConfig::sdxl`] describes - shared by both tests below, since
/// neither may compare a single tensor before this passes. `None` means
/// already-skipped (absent fixture or a tier mismatch - see
/// `brain_testutil::golden::Source`'s own doc).
fn open_golden(cfg: &SupirConfig) -> Option<Golden> {
    let dir = testdata_dir();
    let manifest = dir.join("manifest.json");
    let src = Source::open_manifest(&manifest, "tools/goldens/supir_dump_reference.py --s-churn 0.0 --out testdata/supir_forward_parity")?;

    let bb = &cfg.backbone;
    let ok = src.require(&[
        ("model_channels", bb.block_out_channels[0] as i64),
        ("context_dim", bb.cross_attention_dim as i64),
        ("adm_in_channels", bb.projection_class_embeddings_input_dim as i64),
        ("num_res_blocks", bb.layers_per_block as i64),
        ("transformer_depth_0", bb.transformer_layers_per_block[0] as i64),
        ("transformer_depth_1", bb.transformer_layers_per_block[1] as i64),
        ("transformer_depth_2", bb.transformer_layers_per_block[2] as i64),
        ("n_project_modules", (cfg.adaptors.joins.len() + 1 + cfg.adaptors.cross.len()) as i64),
        ("n_trunk_outputs", (cfg.adaptors.joins.len() + 1) as i64),
    ]);
    if !ok {
        return None;
    }

    let stages = dir.join("stages.safetensors");
    if !stages.exists() {
        brain_testutil::skip(&format!("{} absent", stages.display()));
        return None;
    }
    let gold = Golden(brain_testutil::parity::load(&stages));

    // The whole reason this golden's forward is replayable at all: churn
    // OFF at every step, not just the first one either test actually uses.
    for (i, g) in gold.need("steps.gamma").iter().enumerate() {
        assert_eq!(*g, 0.0, "step {i}: gamma is nonzero - was this golden really dumped with --s-churn 0.0?");
    }
    assert_eq!(gold.need("cond.c.control"), gold.need("stage1._z"), "the trunk's hint must be _z itself");
    assert_eq!(gold.need("cond.c.control"), gold.need("cond.uc.control"), "uc/c must share the same LQ control latent");
    Some(gold)
}

/// The discrete UNet timestep index for the golden's step-0 forward
/// (`sigma_to_idx(sigma_hat)`, `s_churn = 0` so `sigma_hat == sigma`) -
/// shared by both tests, since both replay the same first denoiser call.
fn step0_timestep_idx(gold: &Golden) -> u32 {
    let sigma0 = gold.need("steps.sigma")[0];
    let gamma0 = gold.need("steps.gamma")[0];
    let sh = sigma_hat(sigma0, gamma0);
    DiscreteDenoiserWithControl::new().index(sh) as u32
}

/// SUPIR hardcodes original/target size to 1024x1024 with no crop
/// REGARDLESS of the actual LQ image size (`SUPIR_model.py::prepare_
/// condition`, verified against upstream source, not assumed) - a genuine
/// departure from `sdxlunet::sampler::sample`'s own convention (which uses
/// the GENERATED size), worth carrying into `pipeline.rs` rather than
/// re-deriving it from the image size a second time and getting it wrong.
const TIME_IDS: [f32; 6] = [1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0];

/// **Rung 1a** (stage parity, the `GLVControl` trunk only): all 10 trunk
/// hidden states against the real checkpoint, for both CFG branches.
///
/// Deliberately does NOT touch the frozen SDXL backbone at all -
/// `crate::trunk::record` only ever reads `control_model.*` weights (the
/// SUPIR delta, ~5.3 GB fp32), so this rung is achievable in isolation with
/// a MUCH smaller memory footprint than the combined trunk+adaptors+
/// backbone forward [`supir_full_forward_matches_the_real_checkpoint_at_s_
/// churn_zero`] needs (measured: that combined forward climbs past 29 GB
/// resident on this box's 30 GB and gets SIGKILL'd by the OOM killer before
/// finishing - see that test's own doc). This test is the rung that is
/// actually achievable here today, so it runs by default with no opt-in.
#[test]
fn supir_trunk_matches_the_real_checkpoint() {
    let cfg = SupirConfig::sdxl();
    let Some(gold) = open_golden(&cfg) else { return };

    let Some(supir_path) = supir_weights_path() else {
        brain_testutil::skip("set BRAIN_SUPIR_WEIGHTS to a SUPIR-v0Q_fp32.safetensors path");
        return;
    };

    let s = gold.shape("in.noised_z");
    let (lh, lw) = (s[2] as u32, s[3] as u32);
    let t_enc = gold.shape("cond.uc.crossattn")[1] as u32;
    let idx = step0_timestep_idx(&gold);
    println!("step0 discrete timestep idx {idx}; latent {lh}x{lw}, t_enc {t_enc}");

    println!("importing SUPIR delta from {} ...", supir_path.display());
    let mut raw_supir: HashMap<String, (Vec<usize>, Vec<f32>)> = checkpoint::safetensors::read(supir_path.to_str().expect("utf-8 path"))
        .expect("read SUPIR checkpoint")
        .into_iter()
        .map(|t| (t.name, (t.shape, t.data)))
        .collect();
    raw_supir.remove("model.control_model.mask_LQ");
    let tensors = supir::import::remap(raw_supir, &cfg).expect("supir::import::remap (mask_LQ removed)");
    let params: usize = tensors.values().map(|(_, d)| d.len()).sum();
    println!("{} tensors, {params} parameters = {:.2} GB fp32", tensors.len(), params as f64 * 4.0 / 1e9);

    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let sample = gold.need("in.noised_z").to_vec();
    let hint = gold.need("stage1._z").to_vec();

    let mut r = Table::new(GATE, REL_GATE);
    for (b, branch, tag) in [(0usize, "uc", "neg"), (1usize, "c", "pos")] {
        let pooled = gold.need(&format!("cond.clip_bigg.{tag}.pooled")).to_vec();
        let enc = gold.need(&format!("cond.{branch}.crossattn")).to_vec();

        let mut rec = Rec::new(&gpu, &cfg.backbone, &tensors, t_enc, true);
        rec.set_prefix("control_model.");

        let sample_in = gpu.storage((cfg.backbone.in_channels * lh * lw) as u64);
        let hint_in = gpu.storage((4 * lh * lw) as u64);
        let enc_in = gpu.storage((t_enc * cfg.backbone.cross_attention_dim) as u64);
        let temb_in = gpu.storage(cfg.backbone.block_out_channels[0] as u64);
        let aug_in = gpu.storage(cfg.backbone.projection_class_embeddings_input_dim as u64);
        gpu.write_f32(&sample_in, &sample);
        gpu.write_f32(&hint_in, &hint);
        gpu.write_f32(&enc_in, &enc);
        let temb = model::hostmath::timestep_embedding(idx as f32, cfg.backbone.block_out_channels[0] as usize, cfg.backbone.flip_sin_to_cos, cfg.backbone.freq_shift as f64, 10_000.0);
        let aug = sdxlunet::hostemb::added_cond(&pooled, &TIME_IDS, cfg.backbone.addition_time_embed_dim, cfg.backbone.flip_sin_to_cos, cfg.backbone.freq_shift);
        gpu.write_f32(&temb_in, &temb);
        gpu.write_f32(&aug_in, &aug);

        let hs = supir::trunk::record(&mut rec, &cfg.trunk, "control_model.", lh, lw, &enc_in, &hint_in, &sample_in, &temb_in, &aug_in);
        assert_eq!(hs.len(), 10, "GLVControl must return exactly 10 hidden states");
        let bufs: Vec<(gpu_core::DeviceBuffer, usize)> = hs.iter().map(|m| (m.buf.clone(), (m.c * m.h * m.w) as usize)).collect();
        let (steps, _taps) = rec.into_blocks().finish();
        gpu.submit(&[], &steps);

        for (k, (buf, n)) in bufs.iter().enumerate() {
            let got = gpu.read(buf, *n);
            r.check(&format!("{branch}.trunk.hs{k}"), &got, gold.batch(&format!("trunk.hs{k}"), b));
        }
    }

    r.print();
    let worst = r.worst_cosine();
    let worst_rel = r.worst_rel_l2();
    println!(
        "\n{} comparisons\n  worst cosine  {} 1-cos {:.3e}\n  worst rel_l2  {} {:.3e}\n",
        r.rows.len(),
        worst.0,
        1.0 - worst.1,
        worst_rel.0,
        worst_rel.1
    );
    r.assert_clean();
}

/// **Rungs 1b + 2** (adaptor input/output parity, plus single-forward
/// parity against `unet_final_raw_out`): the FULL trunk + adaptors + frozen
/// backbone graph, ~3.9 B parameters / 15.6 GB fp32.
///
/// Gated behind `BRAIN_SUPIR_ALLOW_FULL_MEMORY=1` in ADDITION to the usual
/// checkpoint/golden gates - measured on this box (30 GB RAM, no discrete
/// GPU; this crate's own roadmap ledger already anticipated exactly this
/// class of constraint, naming int8 a prerequisite for running SUPIR at all
/// on hardware like this rather than an optimisation): host-resident import
/// tensors (15.6 GB) plus the device-side weight upload `Supir::new`
/// performs while they are STILL resident climbs steadily past 29 GB and
/// the kernel OOM-kills the process before the graph finishes recording -
/// not a correctness failure, a resource ceiling, and letting it run
/// unguarded by default would crash any CI runner or developer machine with
/// the same or tighter limit. `supir::int8`/`sdxlunet::int8` now exist and
/// close this for real (see [`supir_full_forward_int8_fits_this_machine`]
/// below) - this fp32 test is kept as the tight-floor ceiling case for a
/// machine that genuinely has the RAM, not because int8 is unavailable. Set
/// the env var on a machine with enough headroom (measured need: north of
/// 30 GB) to actually run it.
#[test]
fn supir_full_forward_matches_the_real_checkpoint_at_s_churn_zero() {
    if std::env::var_os("BRAIN_SUPIR_ALLOW_FULL_MEMORY").is_none_or(|v| v.is_empty() || v == "0") {
        brain_testutil::skip(
            "BRAIN_SUPIR_ALLOW_FULL_MEMORY unset - this test's combined trunk+adaptors+backbone \
             graph needs north of 30 GB resident (measured: host import tensors 15.6 GB plus \
             Supir::new's device-side upload while they are still live climbed past 29 GB and \
             was SIGKILL'd by the OOM killer on this box). Set BRAIN_SUPIR_ALLOW_FULL_MEMORY=1 \
             only on a machine with enough RAM headroom to run it.",
        );
        return;
    }

    let cfg = SupirConfig::sdxl();
    let Some(gold) = open_golden(&cfg) else { return };
    let bb = cfg.backbone.clone();

    let Some(supir_path) = supir_weights_path() else {
        brain_testutil::skip("set BRAIN_SUPIR_WEIGHTS to a SUPIR-v0Q_fp32.safetensors path");
        return;
    };
    let Some(sdxl_path) = sdxl_ldm_weights_path() else {
        brain_testutil::skip("set BRAIN_SUPIR_SDXL_LDM to a single-file sd_xl_base_1.0*.safetensors path");
        return;
    };

    let s = gold.shape("in.noised_z");
    let (lh, lw) = (s[2] as u32, s[3] as u32);
    let t_enc = gold.shape("cond.uc.crossattn")[1] as u32;

    println!("importing SUPIR delta from {} ...", supir_path.display());
    // `supir::import::load` rejects `model.control_model.mask_LQ` outright
    // (by design - see that module's doc), so the real checkpoint needs the
    // same manual removal `tests/import_real.rs` already does before
    // handing it to `remap`.
    let mut raw_supir: HashMap<String, (Vec<usize>, Vec<f32>)> = checkpoint::safetensors::read(supir_path.to_str().expect("utf-8 path"))
        .expect("read SUPIR checkpoint")
        .into_iter()
        .map(|t| (t.name, (t.shape, t.data)))
        .collect();
    raw_supir.remove("model.control_model.mask_LQ");
    let mut tensors = supir::import::remap(raw_supir, &cfg).expect("supir::import::remap (mask_LQ removed)");
    println!("importing frozen SDXL backbone from {} ...", sdxl_path.display());
    tensors.extend(sdxlunet::import::load_ldm(sdxl_path.to_str().expect("utf-8 path"), &bb).expect("sdxlunet::import::load_ldm"));
    let params: usize = tensors.values().map(|(_, d)| d.len()).sum();
    println!("{} tensors, {params} parameters = {:.2} GB fp32; building the graph ...", tensors.len(), params as f64 * 4.0 / 1e9);

    let control_scale = gold.need("steps.control_scale_used")[0];
    let m = Supir::new(gpu_core::testgpu::dev(&KERNELS), cfg.clone(), &tensors, lh, lw, t_enc, true, control_scale);
    drop(tensors);
    println!("{} steps recorded (latent {lh}x{lw}, t_enc {t_enc}, control_scale {control_scale:.6}); running ...", m.steps().len());

    let idx = step0_timestep_idx(&gold);
    println!("step0 discrete timestep idx {idx}");

    let mut r = Table::new(GATE, REL_GATE);

    // ---- host conditioning first, both branches: a failure here explains
    // every device tap below it rather than the reverse (the same ordering
    // `crates/controlnet/tests/parity.rs` uses for its own host checks).
    for (tag, vec_name) in [("neg", "cond.uc.vector"), ("pos", "cond.c.vector")] {
        let pooled = gold.need(&format!("cond.clip_bigg.{tag}.pooled"));
        let add = sdxlunet::hostemb::added_cond(pooled, &TIME_IDS, bb.addition_time_embed_dim, bb.flip_sin_to_cos, bb.freq_shift);
        r.check(&format!("host.vector.{tag}"), &add, gold.need(vec_name));
    }

    let sample = gold.need("in.noised_z").to_vec();
    let hint = gold.need("stage1._z").to_vec();

    for (b, branch, tag) in [(0usize, "uc", "neg"), (1usize, "c", "pos")] {
        let pooled = gold.need(&format!("cond.clip_bigg.{tag}.pooled")).to_vec();
        let enc = gold.need(&format!("cond.{branch}.crossattn")).to_vec();

        let out = m.run(&sample, &hint, idx as f32, &enc, &pooled, &TIME_IDS);
        r.check(&format!("{branch}.unet_final_raw_out"), &out, gold.batch("unet_final_raw_out", b));

        // ---- rung 1a: all 10 GLVControl trunk outputs -------------------
        for k in 0..10 {
            let name = format!("trunk.hs{k}");
            let got = m.read_tap(&name).unwrap_or_else(|| panic!("no tap {name} (trunk::record's tap loop regressed?)"));
            r.check(&format!("{branch}.{name}"), &got, gold.batch(&name, b));
        }

        // ---- rung 1b: all 12 adaptor outputs AND every input side --------
        // (a permutation of two same-width control tensors would only show
        // up on an input-side comparison - see the crate's own roadmap doc).
        for pm in 0..12 {
            let name = format!("proj{pm}");
            if let Some(got) = m.read_tap(&name) {
                r.check(&format!("{branch}.{name}"), &got, gold.batch(&name, b));
            }
            for input_idx in 0..3 {
                let iname = format!("proj{pm}.in{input_idx}");
                if let Some(got) = m.read_tap(&iname) {
                    r.check(&format!("{branch}.{iname}"), &got, gold.batch(&iname, b));
                }
            }
        }
    }

    r.print();
    let worst = r.worst_cosine();
    let worst_rel = r.worst_rel_l2();
    println!(
        "\n{} comparisons\n  worst cosine  {} 1-cos {:.3e}\n  worst rel_l2  {} {:.3e}\n",
        r.rows.len(),
        worst.0,
        1.0 - worst.1,
        worst_rel.0,
        worst_rel.1
    );

    // A golden tap this run never compares is a silent hole in the ladder -
    // checked for BOTH branches (`uc.`/`c.` row-name prefixes), so a name
    // dropped from only one branch's loop still fails loudly.
    let untapped: Vec<&str> = gold
        .0
        .keys()
        .map(|s| s.as_str())
        .filter(|n| n.starts_with("trunk.hs") || n.starts_with("proj") || *n == "unet_final_raw_out")
        .filter(|n| !r.has(&format!("uc.{n}")) || !r.has(&format!("c.{n}")))
        .collect();
    assert!(untapped.is_empty(), "{} goldens have no matching tap in both branches: {untapped:?}", untapped.len());

    r.assert_clean();
}

/// Does the combined int8 graph fit this machine with taps OFF - the
/// production/serving shape (`pipeline.rs`, not yet written, would never
/// need per-stage taps)? Real weights, no golden dependency (finite-output
/// smoke check only, like `supir::model::tests::tiny_forward_is_finite` but
/// at real scale), so it isolates one variable at a time from
/// [`supir_full_forward_int8_fits_this_machine`]: `Builder::tap` pins every
/// tapped buffer and disables the activation pool
/// (`vae::blocks::Builder::free`'s doc), so a taps=true build pays for BOTH
/// int8's own remaining device-side cost (weights still upload as fp32 -
/// see `supir::int8`'s module doc: this tier saves HOST bytes, not device
/// ones) AND every intermediate activation across the whole trunk+adaptors+
/// backbone graph held live at once. This test removes the second cost to
/// see how much of the ceiling it was responsible for.
///
/// Gated behind `BRAIN_SUPIR_ALLOW_FULL_MEMORY=1` - see the module doc's
/// "what int8 actually closes here, measured" section: on this box (one
/// Intel iGPU, no discrete card, `wgpu` reports a 2047 MiB per-buffer cap)
/// even with taps off and int8 host-side storage, the device-resident
/// buffers are still fp32-sized. Run alone (nothing else concurrent, per
/// this crate's `--test-threads=1` convention for this file) with the env
/// var set: it still hits the same `wgpu error: Out of Memory` recording
/// the graph, i.e. removing the tap-pinned activation cost is NOT, by
/// itself, enough to close the gap - the base fp32-sized weight upload
/// alone already exceeds the adapter's per-buffer/per-binding ceiling. That
/// is the honest, measured answer to the question this test isolates for;
/// closing it needs genuine on-device int8 storage (see the module doc),
/// not a taps/activation-pool change.
#[test]
fn supir_full_forward_int8_no_taps_fits_this_machine() {
    if std::env::var_os("BRAIN_SUPIR_ALLOW_FULL_MEMORY").is_none_or(|v| v.is_empty() || v == "0") {
        brain_testutil::skip(
            "BRAIN_SUPIR_ALLOW_FULL_MEMORY unset - this test's combined trunk+adaptors+backbone \
             graph's device-resident buffers are still fp32-sized even under int8 HOST storage \
             (measured: hits `wgpu error: Out of Memory` on this box's Intel iGPU, 2047 MiB \
             per-buffer cap, even with taps off). See the module doc's \"what int8 actually \
             closes here, measured\" section. Set BRAIN_SUPIR_ALLOW_FULL_MEMORY=1 only on a \
             machine with enough VRAM/device-memory headroom to run it, alone (nothing else \
             concurrent).",
        );
        return;
    }

    let cfg = SupirConfig::sdxl();
    let Some(supir_path) = supir_weights_path() else {
        brain_testutil::skip("set BRAIN_SUPIR_WEIGHTS to a SUPIR-v0Q_fp32.safetensors path");
        return;
    };
    let Some(sdxl_path) = sdxl_ldm_weights_path() else {
        brain_testutil::skip("set BRAIN_SUPIR_SDXL_LDM to a single-file sd_xl_base_1.0*.safetensors path");
        return;
    };

    println!("importing SUPIR delta from {} ...", supir_path.display());
    let mut raw_supir: HashMap<String, (Vec<usize>, Vec<f32>)> = checkpoint::safetensors::read(supir_path.to_str().expect("utf-8 path"))
        .expect("read SUPIR checkpoint")
        .into_iter()
        .map(|t| (t.name, (t.shape, t.data)))
        .collect();
    raw_supir.remove("model.control_model.mask_LQ");
    let mut tensors = supir::import::remap(raw_supir, &cfg).expect("supir::import::remap (mask_LQ removed)");
    println!("importing frozen SDXL backbone from {} ...", sdxl_path.display());
    tensors.extend(sdxlunet::import::load_ldm(sdxl_path.to_str().expect("utf-8 path"), &cfg.backbone).expect("sdxlunet::import::load_ldm"));
    let params: usize = tensors.values().map(|(_, d)| d.len()).sum();
    println!("{} tensors, {params} parameters = {:.2} GB fp32; quantizing ...", tensors.len(), params as f64 * 4.0 / 1e9);

    let q = supir::int8::quantize_tensors(&tensors);
    drop(tensors);

    // A real 256x256 LQ image: latent 32x32, a plausible token count.
    let (lh, lw, t_enc) = (32u32, 32u32, 77u32);
    let m = Supir::new_quantized(gpu_core::testgpu::dev(&KERNELS), cfg.clone(), &q.full, &q.packed, lh, lw, t_enc, false, 1.0);
    drop(q);
    println!("{} steps recorded (taps off); running ...", m.steps().len());

    let c = &m.config().backbone;
    let sample: Vec<f32> = (0..(c.in_channels * lh * lw) as usize).map(|i| ((i as f32) * 0.013).sin()).collect();
    let hint: Vec<f32> = (0..(4 * lh * lw) as usize).map(|i| ((i as f32) * 0.021).cos()).collect();
    let enc: Vec<f32> = (0..(t_enc * c.cross_attention_dim) as usize).map(|i| ((i as f32) * 0.029).cos()).collect();
    let pooled: Vec<f32> = (0..c.pooled_dim() as usize).map(|i| ((i as f32) * 0.07).sin()).collect();
    let time_ids = [1024.0f32, 1024.0, 0.0, 0.0, 1024.0, 1024.0];

    let out = m.run(&sample, &hint, 601.0, &enc, &pooled, &time_ids);
    assert_eq!(out.len(), (c.out_channels * lh * lw) as usize);
    assert!(out.iter().all(|v| v.is_finite()), "int8 taps-off forward produced a non-finite output");
    println!("int8 taps-off full forward: {} outputs, all finite", out.len());
}

/// Per-tap gates for the int8 sibling below - looser than [`GATE`]/[`REL_GATE`]
/// by design: int8 is a lossy tier (`supir::int8`'s own module doc), and a
/// combined trunk+adaptors+backbone forward compounds that noise over many
/// more quantized layers than any single-model int8 test in this workspace
/// measures. These floors only need to catch a BROKEN port - the numbers
/// this test prints are the deliverable, matching `crates/flux1/tests/
/// dit_parity.rs`'s documented convention for its own int8 gate.
const GATE_INT8: f64 = 0.97;
const REL_GATE_INT8: f64 = 0.2;

/// The int8 sibling of [`supir_full_forward_matches_the_real_checkpoint_at_
/// s_churn_zero`]: the SAME combined trunk + adaptors + frozen-backbone
/// graph, built via `supir::int8::quantize_tensors` +
/// [`supir::model::Supir::new_quantized`] instead of a whole-model fp32
/// [`sdxlunet::import::Tensors`] map.
///
/// This was this port's roadmap ledger's original phase-5 aspiration: not
/// "int8 exists", but "the previously-OOMing full-forward test now runs to
/// completion on this box" with NO opt-in needed. That is not true on this
/// box - `supir::int8`/`sdxlunet::int8` are a HOST-memory tier only (see the
/// module doc's "what int8 actually closes here, measured" section).
/// Measured on this box, taps ON: import is 2608 tensors / 3 899 803 596
/// params = 15.60 GB fp32; `quantize_tensors` packs 889 of 2608 tensors to
/// int8 (3.34 GB) and leaves 1719 fp32 (2.29 GB) = 5.62 GB host-resident
/// (a genuine ~2.8x host reduction from 15.60 GB); recording the graph then
/// hits `wgpu error: Out of Memory` on this box's adapter (`Intel(R) Arc(tm)
/// Graphics (MTL)`, `IntegratedGpu`, Vulkan; `max_buffer_size` /
/// `max_storage_buffer_binding_size` both 2047 MiB) before a single forward
/// runs - the device-resident weight buffers are still fp32-sized (see
/// `Supir::new_quantized`'s own doc: "device buffers ... bit-identical to
/// `Supir::new`'s ... only the HOST-resident bytes differ"). Gated behind
/// `BRAIN_SUPIR_ALLOW_FULL_MEMORY=1`, same as the fp32 sibling above and
/// [`supir_full_forward_int8_no_taps_fits_this_machine`], and skips itself
/// (never fails) when it is unset, or when a checkpoint or the golden is
/// absent. The tracked follow-up is genuine on-device int8 storage - the
/// `flux1`/`s3dit` DP4A shape, threaded through `vae::blocks::Builder`.
#[test]
fn supir_full_forward_int8_fits_this_machine() {
    if std::env::var_os("BRAIN_SUPIR_ALLOW_FULL_MEMORY").is_none_or(|v| v.is_empty() || v == "0") {
        brain_testutil::skip(
            "BRAIN_SUPIR_ALLOW_FULL_MEMORY unset - this test's combined trunk+adaptors+backbone \
             graph's device-resident buffers are still fp32-sized even under int8 HOST storage. \
             Measured on this box: 2608 tensors / 15.60 GB fp32, quantized to 889 packed (3.34 GB) \
             + 1719 fp32 (2.29 GB) = 5.62 GB host-resident, then `wgpu error: Out of Memory` on \
             this box's Intel iGPU (2047 MiB per-buffer/per-binding cap) while recording the \
             device-side graph. See the module doc's \"what int8 actually closes here, measured\" \
             section. Set BRAIN_SUPIR_ALLOW_FULL_MEMORY=1 only on a machine with enough \
             VRAM/device-memory headroom to run it, alone (nothing else concurrent).",
        );
        return;
    }

    let cfg = SupirConfig::sdxl();
    let Some(gold) = open_golden(&cfg) else { return };
    let bb = cfg.backbone.clone();

    let Some(supir_path) = supir_weights_path() else {
        brain_testutil::skip("set BRAIN_SUPIR_WEIGHTS to a SUPIR-v0Q_fp32.safetensors path");
        return;
    };
    let Some(sdxl_path) = sdxl_ldm_weights_path() else {
        brain_testutil::skip("set BRAIN_SUPIR_SDXL_LDM to a single-file sd_xl_base_1.0*.safetensors path");
        return;
    };

    let s = gold.shape("in.noised_z");
    let (lh, lw) = (s[2] as u32, s[3] as u32);
    let t_enc = gold.shape("cond.uc.crossattn")[1] as u32;

    println!("importing SUPIR delta from {} ...", supir_path.display());
    let mut raw_supir: HashMap<String, (Vec<usize>, Vec<f32>)> = checkpoint::safetensors::read(supir_path.to_str().expect("utf-8 path"))
        .expect("read SUPIR checkpoint")
        .into_iter()
        .map(|t| (t.name, (t.shape, t.data)))
        .collect();
    raw_supir.remove("model.control_model.mask_LQ");
    let mut tensors = supir::import::remap(raw_supir, &cfg).expect("supir::import::remap (mask_LQ removed)");
    println!("importing frozen SDXL backbone from {} ...", sdxl_path.display());
    tensors.extend(sdxlunet::import::load_ldm(sdxl_path.to_str().expect("utf-8 path"), &bb).expect("sdxlunet::import::load_ldm"));
    let params: usize = tensors.values().map(|(_, d)| d.len()).sum();
    println!("{} tensors, {params} parameters = {:.2} GB fp32; quantizing ...", tensors.len(), params as f64 * 4.0 / 1e9);

    let q = supir::int8::quantize_tensors(&tensors);
    let packed_bytes: f64 = q.packed.values().map(|p| (p.packed.len() * 4 + p.scale.len() * 4) as f64).sum();
    let full_bytes: f64 = q.full.values().map(|(_, d)| d.len() as f64 * 4.0).sum();
    println!(
        "{} of {} tensors packed to int8 ({:.2} GB) + {} tensors left fp32 ({:.2} GB) = {:.2} GB resident, vs {:.2} GB for the plain fp32 map",
        q.packed.len(),
        tensors.len(),
        packed_bytes / 1e9,
        q.full.len(),
        full_bytes / 1e9,
        (packed_bytes + full_bytes) / 1e9,
        params as f64 * 4.0 / 1e9,
    );
    drop(tensors); // the fp32 source is no longer needed once it is quantized

    let control_scale = gold.need("steps.control_scale_used")[0];
    let m = Supir::new_quantized(gpu_core::testgpu::dev(&KERNELS), cfg.clone(), &q.full, &q.packed, lh, lw, t_enc, true, control_scale);
    drop(q);
    println!("{} steps recorded (latent {lh}x{lw}, t_enc {t_enc}, control_scale {control_scale:.6}); running ...", m.steps().len());

    let idx = step0_timestep_idx(&gold);
    println!("step0 discrete timestep idx {idx}");

    let mut r = Table::new(GATE_INT8, REL_GATE_INT8);

    for (tag, vec_name) in [("neg", "cond.uc.vector"), ("pos", "cond.c.vector")] {
        let pooled = gold.need(&format!("cond.clip_bigg.{tag}.pooled"));
        let add = sdxlunet::hostemb::added_cond(pooled, &TIME_IDS, bb.addition_time_embed_dim, bb.flip_sin_to_cos, bb.freq_shift);
        r.check(&format!("host.vector.{tag}"), &add, gold.need(vec_name));
    }

    let sample = gold.need("in.noised_z").to_vec();
    let hint = gold.need("stage1._z").to_vec();

    for (b, branch, tag) in [(0usize, "uc", "neg"), (1usize, "c", "pos")] {
        let pooled = gold.need(&format!("cond.clip_bigg.{tag}.pooled")).to_vec();
        let enc = gold.need(&format!("cond.{branch}.crossattn")).to_vec();

        let out = m.run(&sample, &hint, idx as f32, &enc, &pooled, &TIME_IDS);
        r.check(&format!("{branch}.unet_final_raw_out"), &out, gold.batch("unet_final_raw_out", b));

        for k in 0..10 {
            let name = format!("trunk.hs{k}");
            let got = m.read_tap(&name).unwrap_or_else(|| panic!("no tap {name} (trunk::record's tap loop regressed?)"));
            r.check(&format!("{branch}.{name}"), &got, gold.batch(&name, b));
        }

        for pm in 0..12 {
            let name = format!("proj{pm}");
            if let Some(got) = m.read_tap(&name) {
                r.check(&format!("{branch}.{name}"), &got, gold.batch(&name, b));
            }
            for input_idx in 0..3 {
                let iname = format!("proj{pm}.in{input_idx}");
                if let Some(got) = m.read_tap(&iname) {
                    r.check(&format!("{branch}.{iname}"), &got, gold.batch(&iname, b));
                }
            }
        }
    }

    r.print();
    let worst = r.worst_cosine();
    let worst_rel = r.worst_rel_l2();
    println!(
        "\nint8: {} comparisons\n  worst cosine  {} 1-cos {:.3e}\n  worst rel_l2  {} {:.3e}\n",
        r.rows.len(),
        worst.0,
        1.0 - worst.1,
        worst_rel.0,
        worst_rel.1
    );

    r.assert_clean();
}
