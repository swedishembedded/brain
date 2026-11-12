// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for position encodings that agree between the device kernel and
// the model they were trained as, for its clients. If your team needs expertise in rotary embeddings and
// long-context models then you can procure our services by sending an email to info@swedishembedded.com.

//! `rope.wgsl` rotates channel pair `(2j, 2j+1)` of a head at position `t` by `t * theta^(-2j/head_dim)`, and
//! `theta` is a parameter of the dispatch: a model trained with a base other than 10000 must get its own angles,
//! not the default ones.

use wgsl_cpu::Jit;

const SEQ: usize = 3;
const HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const D_MODEL: usize = HEADS * HEAD_DIM;
const ROW_STRIDE: usize = 3 * D_MODEL;

/// A fixed, well-spread operand in `[-1, 1)`; the test checks a formula, not a distribution.
fn operands(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7919 + salt * 104_729) % 997) as f32 / 498.5 - 1.0).collect()
}

/// Rotate the region of the fused qkv buffer starting at `base_off` with the kernel.
fn rope_on_device(qkv: &[f32], base_off: usize, theta: f32) -> Vec<f32> {
    let jit = Jit::new(&[("rope", kernels::ROPE)]).expect("compile rope");
    let mut buf = qkv.to_vec();
    let uni = [SEQ as u32, HEADS as u32, HEAD_DIM as u32, ROW_STRIDE as u32, base_off as u32, theta.to_bits()];
    let bufs: Vec<*mut u8> = vec![buf.as_mut_ptr() as *mut u8];
    let n = (SEQ * HEADS * HEAD_DIM / 2) as u64;
    // SAFETY: one invocation per rotation pair, single-threaded; `uni` and `bufs` outlive the call and match the
    // kernel's bindings (one uniform block, one storage buffer).
    unsafe { jit.run(0, 0, n, 1, 1, uni.as_ptr(), bufs.as_ptr()) };
    buf
}

/// The formula, in f64: pair `(2j, 2j+1)` of head `h` at position `t` turns by `t * theta^(-2j/head_dim)`.
fn rope_on_host(qkv: &[f32], base_off: usize, theta: f64) -> Vec<f32> {
    let mut out = qkv.to_vec();
    for t in 0..SEQ {
        for h in 0..HEADS {
            for j in 0..HEAD_DIM / 2 {
                let i = t * ROW_STRIDE + base_off + h * HEAD_DIM + 2 * j;
                let angle = t as f64 * theta.powf(-((2 * j) as f64) / HEAD_DIM as f64);
                let (s, c) = angle.sin_cos();
                let (e, o) = (qkv[i] as f64, qkv[i + 1] as f64);
                out[i] = (e * c - o * s) as f32;
                out[i + 1] = (e * s + o * c) as f32;
            }
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn the_rotation_follows_the_theta_it_is_given() {
    let qkv = operands(SEQ * ROW_STRIDE, 1);
    for base_off in [0, D_MODEL] {
        for theta in [10_000.0f32, 500_000.0, 100.0] {
            let got = rope_on_device(&qkv, base_off, theta);
            let want = rope_on_host(&qkv, base_off, theta as f64);
            let err = max_abs_diff(&got, &want);
            assert!(err < 1e-4, "theta {theta}, region at {base_off}: max |device - formula| = {err}");
        }
    }
}

#[test]
fn a_different_theta_gives_different_angles() {
    let qkv = operands(SEQ * ROW_STRIDE, 2);
    let a = rope_on_device(&qkv, 0, 10_000.0);
    let b = rope_on_device(&qkv, 0, 500_000.0);
    assert!(max_abs_diff(&a, &b) > 1e-2, "theta did not change the rotation");
}
