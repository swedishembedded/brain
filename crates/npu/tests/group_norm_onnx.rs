// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `TopoBase::group_norm`, checked NUMERICALLY against a host reference.
//!
//! Every other ONNX gate in this tree checks structure — node counts, op sets,
//! shapes — because running the graph needs a runtime. This one can do better:
//! the decomposition is arithmetic, so a host implementation of the same
//! formula is a legitimate oracle, and comparing against it catches the class
//! of bug a structural test cannot see at all.
//!
//! And that class is the whole risk here. GroupNorm is emitted as
//! Reshape/ReduceMean/Sub/Mul/Add/Sqrt/Div because `GroupNormalization` is
//! opset 18 and this builder targets 13. The statistics must be taken over
//! `(C/G, H, W)` **jointly** — reshape to `[N, G, (C/G)*H*W]` and reduce the
//! last axis. Reduce the wrong axis, or reshape to `[N*G, ...]`, and you get an
//! INSTANCE norm: same shapes, same op set, no error, a picture with subtly
//! wrong contrast.
//!
//! The graph itself is exercised by `tools/goldens/onnx_eval.py` when
//! onnxruntime is installed (it is a dev tool, not a build dependency); without
//! it the test still checks the host formula against the emitted node sequence
//! and says which half ran.

use npu::topo::TopoBase;

/// The reference: GroupNorm exactly as `vae::blocks` computes it, on the host.
fn host_group_norm(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    groups: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let cg = c / groups;
    let per_group = cg * h * w;
    let mut y = vec![0.0f32; x.len()];
    for ni in 0..n {
        for g in 0..groups {
            let base = ni * c * h * w + g * per_group;
            let slice = &x[base..base + per_group];
            let mean: f32 = slice.iter().sum::<f32>() / per_group as f32;
            let var: f32 = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / per_group as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for (i, v) in slice.iter().enumerate() {
                // Channel this element belongs to, for the per-channel affine.
                let ch = g * cg + i / (h * w);
                y[base + i] = (v - mean) * inv * gamma[ch] + beta[ch];
            }
        }
    }
    y
}

fn build(n: usize, c: usize, h: usize, w: usize, groups: usize, gamma: &[f32], beta: &[f32], eps: f32) -> Vec<u8> {
    let mut g = onnx::GraphBuilder::new("gn");
    g.input_f32("x", &[n as i64, c as i64, h as i64, w as i64]);
    g.output_f32("y", &[n as i64, c as i64, h as i64, w as i64]);
    let mut b = TopoBase::new(&mut g);
    let out = b.group_norm("x", n, c, h, w, groups, "gamma", gamma.to_vec(), "beta", beta.to_vec(), eps);
    // Rename the last node's output to the declared graph output.
    b.node("Identity", &[&out], "y");
    g.finish()
}

/// Dims chosen so nothing is degenerate: C/G = 3 (not 1, which would make group
/// and instance norm identical and hide the whole bug this test exists for),
/// H != W, and N > 1 so a batch-collapsing reduction shows.
const N: usize = 2;
const C: usize = 12;
const H: usize = 5;
const W: usize = 7;
const G: usize = 4;

fn fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // The unified deterministic LCG (audit F39/F40).
    let mut l = data::rng::Lcg::new(12345);
    let mut next = || l.signed();
    let x: Vec<f32> = (0..N * C * H * W).map(|_| next()).collect();
    let gamma: Vec<f32> = (0..C).map(|_| 1.0 + 0.3 * next()).collect();
    let beta: Vec<f32> = (0..C).map(|_| 0.2 * next()).collect();
    (x, gamma, beta)
}

/// The graph must be a well-formed opset-13 model using only decomposed ops —
/// no `GroupNormalization`, which would not load at this opset.
#[test]
fn the_decomposition_uses_only_opset_13_ops() {
    let (_, gamma, beta) = fixture();
    let bytes = build(N, C, H, W, G, &gamma, &beta, 1e-6);
    let m = onnx::decode_model(&bytes).expect("valid ONNX");
    assert_eq!(m.opset_import[0].version, 13, "this builder targets opset 13");
    let g = m.graph.expect("graph");
    let ops: std::collections::BTreeSet<&str> = g.node.iter().map(|n| n.op_type.as_str()).collect();
    eprintln!("group_norm ops: {ops:?}");
    assert!(!ops.contains("GroupNormalization"), "GroupNormalization is opset 18 — it would not load");
    for op in &ops {
        assert!(
            ["Reshape", "ReduceMean", "Sub", "Mul", "Add", "Sqrt", "Div", "Identity"].contains(op),
            "unexpected op `{op}` in the decomposition"
        );
    }
}

/// NUMERICAL: run the emitted graph and compare against the host formula.
///
/// Skips loudly when onnxruntime is absent — and says so, rather than passing
/// silently and looking like it checked the numbers (`.agents/rules/lessons.md` #1).
#[test]
fn the_emitted_graph_matches_the_host_formula() {
    let (x, gamma, beta) = fixture();
    let eps = 1e-6f32;
    let bytes = build(N, C, H, W, G, &gamma, &beta, eps);
    let want = host_group_norm(&x, N, C, H, W, G, &gamma, &beta, eps);

    let dir = std::env::temp_dir().join(format!("brain-gn-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let model = dir.join("gn.onnx");
    std::fs::write(&model, &bytes).expect("write model");
    let xin = dir.join("x.bin");
    let raw: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&xin, &raw).expect("write input");

    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/goldens/onnx_eval.py");
    let out = std::process::Command::new("python3")
        .arg(script)
        .arg(&model)
        .arg(&xin)
        .args(["--shape", &format!("{N},{C},{H},{W}")])
        .output();

    let Ok(out) = out else {
        eprintln!("SKIP: could not run python3 — the numerical half did not run");
        return;
    };
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("No module named") {
            eprintln!("SKIP: onnxruntime not installed — the numerical half did not run\n{err}");
            return;
        }
        panic!("onnx_eval.py failed:\n{err}");
    }

    let got: Vec<f32> = out
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got.len(), want.len(), "runtime returned {} floats, want {}", got.len(), want.len());
    let max = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("group_norm vs host: max|delta| = {max:.3e} over {} elements", want.len());
    assert!(max < 1e-5, "emitted GroupNorm disagrees with the host formula by {max:.3e}");
    std::fs::remove_dir_all(&dir).ok();
}

/// The bug the joint reduction exists to prevent: with C/G = 3, a GROUP norm and
/// an INSTANCE norm (per-channel statistics) give different answers, so the host
/// oracle above is actually discriminating. If this ever fails, the fixture has
/// gone degenerate and the numerical test above stopped proving anything.
#[test]
fn the_fixture_distinguishes_group_from_instance_norm() {
    let (x, gamma, beta) = fixture();
    let group = host_group_norm(&x, N, C, H, W, G, &gamma, &beta, 1e-6);
    // Instance norm == group norm with one group per channel.
    let instance = host_group_norm(&x, N, C, H, W, C, &gamma, &beta, 1e-6);
    let max = group.iter().zip(&instance).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("group vs instance on this fixture: max|delta| = {max:.3e}");
    assert!(max > 0.1, "fixture cannot tell group from instance norm — it proves nothing");
}
