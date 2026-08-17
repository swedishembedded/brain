// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity for [`diffusion::scheduler::ltx2_sigmas`] against LTX-2.5's real
//! `LTX2Scheduler.execute`, driven exactly as
//! `tools/goldens/ltxv_schedule_dump_reference.py` drove it.
//!
//! Fixture: `$BRAIN_TESTDATA/golden/ltxv/schedule/{schedule.safetensors,
//! manifest.json}` (default `<repo>/testdata`). The test **skips itself**
//! when the fixture is absent.
//!
//! Regenerate with:
//! ```text
//! python3 tools/goldens/ltxv_schedule_dump_reference.py
//! ```
//!
//! Bit-exactness is not the bar here (unlike Wan's schedule parity): the
//! golden's own dumper self-validates its torch-f32 run against an
//! independent numpy-f64 closed form to `< 1e-6` relative, and this test's
//! Rust reimplementation is that SAME closed form (see [`ltx2_sigmas`]'s own
//! doc) - so `1e-5` absolute (looser than the dumper's own relative bound, to
//! absorb the f32 cast at the golden's write time) is the right tolerance,
//! not a coin flip.

use diffusion::scheduler::{ltx2_sigmas, LTX2_DISTILLED_SIGMAS, LTX2_STAGE2_DISTILLED_SIGMAS, LTX2_TDP_DISTILLED_SIGMAS};

struct Golden {
    tensors: Vec<checkpoint::safetensors::StTensor>,
    manifest: serde_json::Value,
}

impl Golden {
    fn open() -> Option<Golden> {
        let dir = brain_testutil::testdata_path("golden/ltxv/schedule");
        let st = dir.join("schedule.safetensors");
        let mj = dir.join("manifest.json");
        if !st.exists() || !mj.exists() {
            brain_testutil::skip(&format!("{} absent - run `python3 tools/goldens/ltxv_schedule_dump_reference.py`", dir.display()));
            return None;
        }
        let tensors = checkpoint::safetensors::read(st.to_str().expect("utf-8 path")).expect("read golden");
        let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&mj).expect("read manifest")).expect("parse manifest");
        Some(Golden { tensors, manifest })
    }

    fn get(&self, name: &str) -> &[f32] {
        &self.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden missing tensor {name}")).data
    }
}

fn max_abs(got: &[f64], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch: {} vs {}", got.len(), want.len());
    got.iter().zip(want).map(|(&a, &b)| (a - b as f64).abs()).fold(0.0, f64::max)
}

const TOL: f64 = 1e-5;

#[test]
fn ltx2_sigmas_matches_the_real_scheduler_across_every_dumped_case() {
    let Some(g) = Golden::open() else { return };
    let cases = g.manifest["params"]["cases"].as_array().expect("params.cases");

    let mut rows: Vec<(String, f64)> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for case in cases {
        let tokens = case["tokens"].as_u64().unwrap() as usize;
        let steps = case["steps"].as_u64().unwrap() as usize;
        let base_shift = case["base_shift"].as_f64().unwrap();
        let max_shift = case["max_shift"].as_f64().unwrap();
        let stretch = case["stretch"].as_bool().unwrap();
        let terminal = case["terminal"].as_f64().unwrap();
        let key = case["key"].as_str().unwrap();

        let got = ltx2_sigmas(tokens, steps, base_shift, max_shift, stretch, terminal);
        let want = g.get(key);
        assert_eq!(got.len(), want.len(), "{key}: length mismatch");
        assert_eq!(got.len(), steps + 1, "{key}: must be steps+1 entries");

        let d = max_abs(&got, want);
        rows.push((key.to_string(), d));
        if d > TOL {
            failures.push(format!("{key}: max_abs {d:.3e} > {TOL:.0e}"));
        }
        // The schedule always starts at 1 and ends at exactly 0, in the
        // golden too - a cheap structural cross-check independent of the
        // per-element tolerance above.
        assert!((want[0] - 1.0).abs() < 1e-6, "{key}: golden sigma[0] = {}", want[0]);
        assert_eq!(*want.last().unwrap(), 0.0, "{key}: golden last sigma must be 0");
    }

    println!("\n{:<55} {:>12}", "case", "max_abs");
    for (k, d) in &rows {
        println!("{k:<55} {d:>12.3e}");
    }
    let worst = rows.iter().map(|r| r.1).fold(0.0, f64::max);
    println!("\n{} cases, worst max_abs {:.3e}\n", rows.len(), worst);
    assert!(failures.is_empty(), "{} failed:\n  {}", failures.len(), failures.join("\n  "));
}

/// The hardcoded distilled-schedule constants, transcribed by hand into
/// [`LTX2_DISTILLED_SIGMAS`]/[`LTX2_STAGE2_DISTILLED_SIGMAS`]/
/// [`LTX2_TDP_DISTILLED_SIGMAS`] rather than computed - so this is a literal
/// bit-exact comparison against the golden's own `distilled_*.sigmas`
/// tensors (dumped from `ltx_pipelines.utils.constants` directly, not
/// re-derived), not a tolerance check.
#[test]
fn distilled_sigma_constants_match_the_real_source_values() {
    let Some(g) = Golden::open() else { return };
    assert_eq!(LTX2_DISTILLED_SIGMAS.to_vec(), g.get("distilled_8step.sigmas").to_vec());
    assert_eq!(LTX2_STAGE2_DISTILLED_SIGMAS.to_vec(), g.get("distilled_stage2.sigmas").to_vec());
    assert_eq!(LTX2_TDP_DISTILLED_SIGMAS.to_vec(), g.get("distilled_tdp.sigmas").to_vec());
}
