// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 Klein schedule parity vs the diffusers reference
//! (`testdata/flux2/klein-4b/schedule.safetensors`, from
//! `tools/goldens/flux2_dump_reference.py`): `empirical_mu` + exponential shift must
//! reproduce the scheduler's sigma vectors exactly (float32).

use diffusion::scheduler::{default_z_image_sigmas, empirical_mu, klein_sigmas, time_shift_exponential};

use brain_testutil::testdata;

/// `klein_sigmas` builds its ramp inline instead of calling
/// `default_z_image_sigmas`, which reads like duplication and invites a
/// cleanup. It is not interchangeable: the two spellings associate their float
/// operations differently, and f32 multiply/divide is not associative, so they
/// disagree by one ULP for 1979 of the first 2001 step counts.
///
/// The parity test below only exercises 4 and 50 steps, and both happen to
/// fall in the 22 that agree - so it would not catch the swap, and neither
/// would a spot check at other round numbers. This does, without needing a
/// fixture, so the failure arrives at the refactor rather than at whatever
/// consumes the goldens later.
#[test]
fn klein_ramp_is_not_the_shared_linspace_spelling() {
    let seq = (1024 / 16) * (1024 / 16);
    for steps in [6usize, 7, 10, 25] {
        let shared = time_shift_exponential(empirical_mu(seq, steps), &default_z_image_sigmas(steps));
        let got = klein_sigmas(steps, seq);
        assert_eq!(got.len(), shared.len() + 1, "klein appends the terminal zero");
        assert!(
            got[..steps].iter().zip(&shared).any(|(a, b)| a.to_bits() != b.to_bits()),
            "steps={steps}: klein_sigmas now agrees bit-for-bit with the shared linspace \
             spelling. Either the ramp was unified (which moves the FLUX.2 goldens by an \
             ULP - do that deliberately, with those goldens as the gate) or linspace's \
             arithmetic changed."
        );
    }
}

#[test]
fn klein_sigmas_match_reference() {
    let fixture = testdata("flux2/klein-4b/schedule.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        brain_testutil::skip(&format!("fixture {fixture} absent"));
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
