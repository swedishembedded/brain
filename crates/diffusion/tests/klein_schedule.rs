// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 Klein schedule parity vs the diffusers reference
//! (`testdata/flux2/klein-4b/schedule.safetensors`, from
//! `tools/flux2_dump_reference.py`): `empirical_mu` + exponential shift must
//! reproduce the scheduler's sigma vectors exactly (float32).

use diffusion::scheduler::{empirical_mu, klein_sigmas};

use brain_testutil::testdata;

#[test]
fn klein_sigmas_match_reference() {
    let fixture = testdata("flux2/klein-4b/schedule.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        eprintln!("SKIP: fixture {fixture} absent");
        return;
    }
    let fx = checkpoint::safetensors::read(&fixture).unwrap();
    let get = |name: &str| {
        fx.iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("golden {name}"))
    };
    for (h, w, steps) in [(512usize, 512usize, 4usize), (1024, 1024, 4), (1024, 1024, 50), (768, 1360, 4)] {
        let seq = (h / 16) * (w / 16);
        let want_mu = get(&format!("mu_{h}x{w}_s{steps}")).data[0];
        let got_mu = empirical_mu(seq, steps);
        assert!(
            (got_mu - want_mu).abs() < 1e-5,
            "mu {h}x{w} s{steps}: got {got_mu} want {want_mu}"
        );
        let want = &get(&format!("sigmas_{h}x{w}_s{steps}")).data;
        let got = klein_sigmas(steps, seq);
        assert_eq!(got.len(), want.len(), "{h}x{w} s{steps} len");
        for (i, (g, wv)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - wv).abs() < 2e-6,
                "{h}x{w} s{steps} sigma[{i}]: got {g} want {wv}"
            );
        }
    }
}
