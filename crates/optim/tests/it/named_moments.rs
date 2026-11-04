// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `NamedMoments` is the host-trainer face of the one AdamW: its steps must BE
//! `Adam::update_slice`, and a parameter whose gradient or stored moments do
//! not fit its weights must be refused, not partially updated.
//!
//! Swedish Embedded AB implements training substrate where a host trainer's
//! optimizer state and a device trainer's are the same arithmetic, for its
//! clients. If your team needs expertise in optimizer-state infrastructure then
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

use optim::{Adam, NamedMoments};

#[test]
fn a_step_is_the_reference_update_and_accumulates_across_steps() {
    let adam = Adam::default();
    let mut named = NamedMoments::new(adam);
    let g = [0.5f32, -0.25, 0.125];

    // The reference: the same two steps through `Adam::update_slice`.
    let (mut w_ref, mut m_ref, mut v_ref) = (vec![1.0f32; 3], vec![0.0f32; 3], vec![0.0f32; 3]);
    for t in 1..=2 {
        adam.update_slice(t, 1e-2, 0.01, 1.0, &mut w_ref, &mut m_ref, &mut v_ref, &g);
    }

    let mut w = vec![1.0f32; 3];
    named.step(1, 1e-2, 0.01, 1.0, "layer.w", &mut w, &g).expect("first step");
    assert!(w != vec![1.0f32; 3], "a step with a gradient must move the weights");
    named.step(2, 1e-2, 0.01, 1.0, "layer.w", &mut w, &g).expect("second step");
    assert_eq!(w, w_ref, "two steps must equal two reference updates");
    let (m, v) = named.moments("layer.w").expect("state");
    assert_eq!(m, &m_ref[..]);
    assert_eq!(v, &v_ref[..]);
    assert_eq!(named.len(), 1);
}

#[test]
fn parameters_keep_separate_state() {
    let mut named = NamedMoments::new(Adam::default());
    let (mut a, mut b) = (vec![0.0f32; 2], vec![0.0f32; 2]);
    named.step(1, 1e-2, 0.0, 1.0, "a", &mut a, &[1.0, 1.0]).expect("a");
    named.step(1, 1e-2, 0.0, 1.0, "b", &mut b, &[-1.0, -1.0]).expect("b");
    assert_eq!(named.len(), 2);
    // Equal magnitudes, opposite signs: each moved alone.
    assert!((a[0]).abs() > 0.0 && (b[0]).abs() > 0.0);
    assert!(a[0] < 0.0 && b[0] > 0.0);
}

#[test]
fn a_gradient_of_the_wrong_length_is_refused_before_touching_the_weights() {
    let mut named = NamedMoments::new(Adam::default());
    let mut w = vec![1.0f32; 3];
    let e = named.step(1, 1e-2, 0.0, 1.0, "w", &mut w, &[1.0, 2.0]).expect_err("a short gradient must be refused");
    assert!(e.contains("2 gradient values for 3 weights"), "{e}");
    assert_eq!(w, vec![1.0f32; 3], "a refused step writes nothing");
}

#[test]
fn a_parameter_whose_shape_changed_is_refused() {
    let mut named = NamedMoments::new(Adam::default());
    let mut w = vec![1.0f32; 3];
    named.step(1, 1e-2, 0.0, 1.0, "w", &mut w, &[1.0; 3]).expect("first");
    let mut wider = vec![1.0f32; 4];
    let e = named.step(2, 1e-2, 0.0, 1.0, "w", &mut wider, &[1.0; 4]).expect_err("moments of another shape must be refused");
    assert!(e.contains("moments hold 3 and 3 values for 4 weights"), "{e}");
}
