// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `diffusion::restore`'s scalar math against the real reference numbers
//! dumped by `tools/goldens/supir_dump_reference.py` - the porting
//! playbook's stage-parity rung (exact for pure math, no weights and no
//! device needed) for a seam that was previously gated only against
//! hand-computed values.
//!
//! Also checks [`SupirConfig::sdxl`]'s architecture identity against the
//! golden's own `source.identity` block, so a config drift is caught here
//! rather than certifying a comparison that silently ran against the wrong
//! shape.
//!
//! Deliberately does NOT attempt real-checkpoint forward parity (trunk
//! hidden states, adaptor taps, the frozen UNet's raw output): every one of
//! those taps was captured DURING a denoiser call that applies EDM's
//! `input * c_in` preconditioning AND the sampler's stochastic churn
//! (`s_churn = 5`, so `gamma > 0` at every one of the 4 dumped steps - see
//! `steps.gamma`), and the churn noise draw itself was not saved by the
//! dumper. Replaying the sampler step's SCALAR math (below) reproduces the
//! `sigma_hat`/`control_scale`/`cfg_scale` the reference used exactly; the
//! *tensor* the network actually saw also depends on that unsaved random
//! draw and is not reproducible from this golden alone. Real-checkpoint
//! forward parity needs either a re-dump with the churn noise captured, or
//! a `s_churn = 0` variant golden where `gamma` is identically zero and the
//! churned sample equals the un-churned one - noted here rather than
//! attempted with a guess.

use std::path::PathBuf;

use brain_testutil::golden::Source;
use diffusion::restore::{churn_gamma, control_scale_ramp, linear_cfg_scale, sigma_hat, SIGMA_MAX};
use supir::config::SupirConfig;

fn testdata_dir() -> PathBuf {
    brain_testutil::testdata_path("supir")
}

struct Golden(Vec<checkpoint::safetensors::StTensor>);

impl Golden {
    fn need(&self, name: &str) -> &[f32] {
        self.0
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.data.as_slice())
            .unwrap_or_else(|| panic!("golden missing {name}"))
    }
}

#[test]
fn restore_scalar_math_matches_the_real_sampler_trajectory() {
    let dir = testdata_dir();
    let manifest = dir.join("manifest.json");
    let Some(src) = Source::open_manifest(&manifest, "tools/goldens/supir_dump_reference.py") else {
        brain_testutil::skip(&format!("{} absent (run tools/goldens/supir_dump_reference.py)", manifest.display()));
        return;
    };

    let cfg = SupirConfig::sdxl();
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
        return;
    }

    let stages = dir.join("stages.safetensors");
    let Some(path) = stages.to_str() else {
        brain_testutil::skip("non-utf8 testdata path");
        return;
    };
    if !stages.exists() {
        brain_testutil::skip(&format!("{} absent", stages.display()));
        return;
    }
    let gold = Golden(checkpoint::safetensors::read(path).expect("read golden"));

    // Run parameters, read from the dump rather than hand-copied: a
    // re-dump at different sampler settings must not silently desync this
    // test's expectations from the fixture it actually reads.
    let sigmas = gold.need("steps.sigma");
    let next_sigmas = gold.need("steps.next_sigma");
    let want_gamma = gold.need("steps.gamma");
    let want_control_scale = gold.need("steps.control_scale_used");
    let want_cfg_scale = gold.need("steps.cfg_scale_used");
    let n_steps = sigmas.len();
    assert_eq!(n_steps, 4, "this test's constants (s_cfg, s_churn, ...) are pinned to the 4-step golden");

    // Sampler defaults the dumper deliberately turned ON/OFF - see its own
    // module docstring for why each departs from the CLI defaults.
    let s_cfg = 7.5f32; // scale AT sigma -> 0 (LinearCFG's `scale_min` argument)
    let spt_linear_cfg = 4.0f32; // scale AT sigma_max (LinearCFG's `scale` argument)
    let s_churn = 5.0f32;
    let control_scale = 1.0f32; // s_stage2, at sigma -> 0
    let control_scale_start = 0.0f32; // at sigma_max

    let gamma = churn_gamma(s_churn, n_steps);
    assert!((gamma - std::f64::consts::SQRT_2 as f32 + 1.0).abs() < 1e-6, "gamma sanity: {gamma}");

    for i in 0..n_steps {
        let sigma = sigmas[i];
        let sh = sigma_hat(sigma, gamma);

        let diff_gamma = (gamma - want_gamma[i]).abs();
        assert!(diff_gamma < 1e-6, "step {i}: gamma {gamma} vs golden {}", want_gamma[i]);

        let cs = control_scale_ramp(control_scale, control_scale_start, sigma, SIGMA_MAX);
        let diff_cs = (cs - want_control_scale[i]).abs();
        assert!(diff_cs < 1e-4, "step {i}: control_scale {cs} vs golden {} (sigma_hat {sh})", want_control_scale[i]);

        let cfg = linear_cfg_scale(spt_linear_cfg, s_cfg, sh, SIGMA_MAX);
        let diff_cfg = (cfg - want_cfg_scale[i]).abs();
        assert!(diff_cfg < 1e-3, "step {i}: cfg_scale {cfg} vs golden {} (sigma_hat {sh})", want_cfg_scale[i]);

        println!(
            "step {i}: sigma {sigma:.6} sigma_hat {sh:.6} next_sigma {:.6} gamma {gamma:.6} \
             control_scale {cs:.6} (want {:.6}, |d|={diff_cs:.3e}) cfg_scale {cfg:.6} (want {:.6}, |d|={diff_cfg:.3e})",
            next_sigmas[i], want_control_scale[i], want_cfg_scale[i]
        );
    }

    println!("\nrestore_scalar_math_matches_the_real_sampler_trajectory: {n_steps}/{n_steps} steps matched");
}
