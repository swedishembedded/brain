// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of the short causal convolution family (`model::short_conv`): for static and dynamic taps, with and without the
//! fused residual and scale, document segments and a carried history,
//!
//! - the forward equals an f64 oracle of the formula;
//! - the backward (dx, d-base, d-dyn) equals central differences of that oracle;
//! - the single-token step over a ring state, token by token, equals the whole-block forward;
//! - a chunked prefill with the state a chunk leaves equals the unchunked forward;
//! - packed documents never see each other, and neither do blocks; a packed row's outputs and gradients are those of
//!   each document run alone (its rows' `dx` and `d_dyn`, and `d_base` the sum of the documents');
//! - the static family equals `causal_conv1d_step` (the existing static step) when no option is on.
//!
//! Swedish Embedded AB implements solutions for training and serving models with short causal convolutions for its
//! clients. If your team needs expertise in sequence-mixing convolutions then you can procure our services by sending an
//! email to info@swedishembedded.com.

use data::rng::{random_seed, Lcg};
use gpu_core::{DeviceBuffer, Gpu};
use model::short_conv::{conv_state_tail_step, short_conv_bwd, short_conv_fwd, short_conv_step, BaseLayout, DynTaps, ShortConvIds, ShortConvShape};

fn pipes() -> &'static [(&'static str, &'static str)] {
    static P: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let mut v = model::short_conv::pipelines().to_vec();
        v.push(("causal_conv1d_step", kernels::CAUSAL_CONV1D_STEP));
        v
    })
}

fn ids(g: &Gpu) -> ShortConvIds {
    ShortConvIds::resolve(g)
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&t| f64::from(t)).collect()
}

/// Everything one case feeds the convolution.
struct Case {
    shape: ShortConvShape,
    x: Vec<f32>,
    dyn_taps: Vec<f32>,
    base: Vec<f32>,
    seg: Vec<u32>,
    hist: Vec<f32>,
}

const N: u32 = 11;
const C: u32 = 6;
const TAPS: u32 = 4;
const GROUP: u32 = 3;
const STRIDE: u32 = 2 * TAPS * (C / GROUP) + 3; // two taps windows and padding, as a projection shared by two convolutions
const OFF: u32 = TAPS * (C / GROUP);

fn case(rng: &mut Lcg, dynamic: bool, layout: BaseLayout, res: Option<f32>, segments: bool, history: bool, block: u32) -> Case {
    let mut shape = if dynamic { ShortConvShape::dynamic_taps(N, C, TAPS, DynTaps { group_size: GROUP, stride: STRIDE, off: OFF }) } else { ShortConvShape::static_taps(N, C, TAPS, layout) };
    shape.base_layout = layout;
    shape.base_off = 5; // a base tensor with something in front of it
    shape.res_scale = res;
    shape.segments = segments;
    shape.history = history;
    shape.block = block;
    let seg = (0..N).map(|t| u32::from(t >= 4) + u32::from(t >= 8)).collect(); // documents of 4, 4 and 3 rows
    Case {
        x: rng.vec_scaled((N * C) as usize, 1.0),
        dyn_taps: rng.vec_scaled((N * STRIDE) as usize, 0.5),
        base: rng.vec_scaled((5 + TAPS * C) as usize, 0.5),
        seg,
        hist: rng.vec_scaled((C * (TAPS - 1)) as usize, 1.0),
        shape,
    }
}

fn tap_index(shape: &ShortConvShape, j: usize, c: usize) -> usize {
    shape.base_off as usize + if shape.base_layout == BaseLayout::ChannelMajor { c * TAPS as usize + (TAPS as usize - 1 - j) } else { j * C as usize + c }
}

/// The f64 oracle of the whole formula.
fn oracle(cs: &Case, x: &[f64], dyn_taps: &[f64], base: &[f64]) -> Vec<f64> {
    let (n, c, taps) = (N as usize, C as usize, TAPS as usize);
    let sh = &cs.shape;
    let mut y = vec![0.0; n * c];
    for t in 0..n {
        for ci in 0..c {
            let mut acc = 0.0;
            for j in 0..taps {
                let in_block = t % sh.block as usize;
                let xv = if j <= in_block {
                    let s = t - j;
                    if sh.segments && cs.seg[s] != cs.seg[t] { 0.0 } else { x[s * c + ci] }
                } else if sh.history && (!sh.segments || cs.seg[0] == cs.seg[t]) {
                    f64::from(cs.hist[ci * (taps - 1) + (taps - 1 - j + t)])
                } else {
                    0.0
                };
                let mut w = base[tap_index(sh, j, ci)];
                if let Some(d) = sh.dynamic {
                    w += dyn_taps[t * d.stride as usize + d.off as usize + j * (c / d.group_size as usize) + ci / d.group_size as usize];
                }
                acc += w * xv;
            }
            if let Some(s) = sh.res_scale {
                acc = (acc + x[t * c + ci]) * f64::from(s);
            }
            y[t * c + ci] = acc;
        }
    }
    y
}

fn seg_buffer(g: &Gpu, seg: &[u32]) -> DeviceBuffer {
    let b = g.storage(seg.len() as u64);
    g.write(&b, seg);
    b
}

struct Run {
    y: Vec<f32>,
}

fn forward(g: &Gpu, ids: &ShortConvIds, cs: &Case) -> Run {
    let (x, d, b, h) = (g.storage_init("x", &cs.x), g.storage_init("dyn", &cs.dyn_taps), g.storage_init("base", &cs.base), g.storage_init("hist", &cs.hist));
    let seg = seg_buffer(g, &cs.seg);
    let y = g.storage(u64::from(N * C));
    let step = short_conv_fwd(g, ids, &cs.shape, &x, cs.shape.dynamic.map(|_| &d), &b, cs.shape.segments.then_some(&seg), cs.shape.history.then_some(&h), &y);
    g.submit(&[], &[step]);
    Run { y: g.read(&y, (N * C) as usize) }
}

fn fd(loss: &dyn Fn(&[f64]) -> f64, at: &[f64], i: usize) -> f64 {
    let h = 1e-5;
    let mut p = at.to_vec();
    p[i] += h;
    let up = loss(&p);
    p[i] -= 2.0 * h;
    (up - loss(&p)) / (2.0 * h)
}

fn near(got: f32, want: f64, what: &str, i: usize) {
    assert!((f64::from(got) - want).abs() <= 2e-4 * (1.0 + want.abs()), "{what}[{i}]: kernel {got} vs oracle {want}");
}

/// The option matrix the forward is checked over.
fn matrix(rng: &mut Lcg) -> Vec<Case> {
    let mut out = Vec::new();
    for dynamic in [false, true] {
        for layout in [BaseLayout::TapMajor, BaseLayout::ChannelMajor] {
            if dynamic && layout == BaseLayout::ChannelMajor {
                continue; // the dynamic base is a tap-major tensor
            }
            for res in [None, Some(1.0 / 5f32.sqrt())] {
                for segments in [false, true] {
                    for history in [false, true] {
                        out.push(case(rng, dynamic, layout, res, segments, history, N));
                    }
                }
            }
        }
    }
    out.push(case(rng, true, BaseLayout::TapMajor, None, false, false, 1)); // blocks of one row: no history at all
    out
}

#[test]
fn forward_matches_the_oracle_over_every_option() {
    let g = gpu_core::testgpu::dev(pipes());
    let ids = ids(&g);
    let mut rng = Lcg::new(random_seed());
    for cs in matrix(&mut rng) {
        let want = oracle(&cs, &f64s(&cs.x), &f64s(&cs.dyn_taps), &f64s(&cs.base));
        let got = forward(&g, &ids, &cs);
        for i in 0..want.len() {
            near(got.y[i], want[i], &format!("y {:?}", cs.shape), i);
        }
    }
}

#[test]
fn backward_matches_finite_differences_of_the_oracle() {
    let g = gpu_core::testgpu::dev(pipes());
    let ids = ids(&g);
    let mut rng = Lcg::new(random_seed());
    for cs in matrix(&mut rng).into_iter().filter(|c| !c.shape.history) {
        let dy = rng.vec_scaled((N * C) as usize, 1.0);
        let dy64 = f64s(&dy);
        let (x, d, b) = (g.storage_init("x", &cs.x), g.storage_init("dyn", &cs.dyn_taps), g.storage_init("base", &cs.base));
        let seg = seg_buffer(&g, &cs.seg);
        let dyb = g.storage_init("dy", &dy);
        let (dx, dbase, ddyn) = (g.storage(u64::from(N * C)), g.storage_init("dbase", &vec![0.0; cs.base.len()]), g.storage_init("ddyn", &vec![0.0; (N * STRIDE) as usize]));
        let steps = short_conv_bwd(&g, &ids, &cs.shape, &x, &dyb, cs.shape.dynamic.map(|_| &d), &b, cs.shape.segments.then_some(&seg), &dx, &dbase, cs.shape.dynamic.map(|_| &ddyn));
        g.submit(&[], &steps);
        let (dx, dbase, ddyn) = (g.read(&dx, (N * C) as usize), g.read(&dbase, cs.base.len()), g.read(&ddyn, (N * STRIDE) as usize));
        let (x64, d64, b64) = (f64s(&cs.x), f64s(&cs.dyn_taps), f64s(&cs.base));
        let loss = |y: Vec<f64>| y.iter().zip(&dy64).map(|(a, b)| a * b).sum::<f64>();
        for (i, &got) in dx.iter().enumerate() {
            near(got, fd(&|p| loss(oracle(&cs, p, &d64, &b64)), &x64, i), &format!("dx {:?}", cs.shape), i);
        }
        let used: Vec<usize> = (0..b64.len()).filter(|&i| i >= cs.shape.base_off as usize).collect();
        for &i in &used {
            near(dbase[i], fd(&|p| loss(oracle(&cs, &x64, &d64, p)), &b64, i), &format!("dbase {:?}", cs.shape), i);
        }
        assert!(dbase[..cs.shape.base_off as usize].iter().all(|&v| v == 0.0), "{:?}: the gradient wrote in front of base_off", cs.shape);
        if let Some(dt) = cs.shape.dynamic {
            let window = |t: usize, r: usize| t * dt.stride as usize + dt.off as usize + r;
            for t in 0..N as usize {
                for r in 0..(TAPS * (C / GROUP)) as usize {
                    near(ddyn[window(t, r)], fd(&|p| loss(oracle(&cs, &x64, p, &b64)), &d64, window(t, r)), &format!("ddyn {:?}", cs.shape), window(t, r));
                }
            }
            // The other window of the shared projection row is not this convolution's and is left alone.
            assert!((0..N as usize).all(|t| ddyn[t * dt.stride as usize] == 0.0), "{:?}: the gradient wrote outside its window", cs.shape);
        }
    }
}

#[test]
fn stepping_token_by_token_equals_the_whole_block() {
    let g = gpu_core::testgpu::dev(pipes());
    let ids = ids(&g);
    let mut rng = Lcg::new(random_seed());
    for dynamic in [false, true] {
        for res in [None, Some(1.0 / 3.0f32)] {
            let layout = if dynamic { BaseLayout::TapMajor } else { BaseLayout::ChannelMajor };
            let cs = case(&mut rng, dynamic, layout, res, false, false, N);
            let whole = forward(&g, &ids, &cs).y;
            // One sequence, one token at a time.
            let mut shape = cs.shape;
            shape.n = 1;
            shape.block = 1;
            let hist = g.storage_init("hist", &vec![0.0; (C * (TAPS - 1)) as usize]);
            let base = g.storage_init("base", &cs.base);
            for t in 0..N as usize {
                let x = g.storage_init("x", &cs.x[t * C as usize..(t + 1) * C as usize]);
                let d = g.storage_init("dyn", &cs.dyn_taps[t * STRIDE as usize..(t + 1) * STRIDE as usize]);
                let y = g.storage(u64::from(C));
                g.submit(&[], &[short_conv_step(&g, &ids, &shape, &x, dynamic.then_some(&d), &base, &hist, &y)]);
                let got = g.read(&y, C as usize);
                for ci in 0..C as usize {
                    near(got[ci], f64::from(whole[t * C as usize + ci]), &format!("step dynamic={dynamic} res={res:?} t={t}"), ci);
                }
            }
        }
    }
}

#[test]
fn a_chunked_prefill_over_the_carried_state_equals_the_unchunked_forward() {
    let g = gpu_core::testgpu::dev(pipes());
    let ids = ids(&g);
    let mut rng = Lcg::new(random_seed());
    let cs = case(&mut rng, true, BaseLayout::TapMajor, Some(1.0 / 3.0), false, false, N);
    let whole = forward(&g, &ids, &cs).y;
    // Chunks of 5, 1 (shorter than the state) and 5 rows.
    let (base, mut hist) = (g.storage_init("base", &cs.base), g.storage_init("hist", &vec![0.0; (C * (TAPS - 1)) as usize]));
    let mut at = 0usize;
    for rows in [5usize, 1, 5] {
        let mut shape = cs.shape;
        shape.n = rows as u32;
        shape.block = rows as u32;
        shape.history = true;
        let x = g.storage_init("x", &cs.x[at * C as usize..(at + rows) * C as usize]);
        // The per-token taps of these rows, still addressed from the start of the chunk's own projection rows.
        let d = g.storage_init("dyn", &cs.dyn_taps[at * STRIDE as usize..(at + rows) * STRIDE as usize]);
        let y = g.storage(rows as u64 * u64::from(C));
        let next = g.storage(u64::from(C * (TAPS - 1)));
        g.submit(&[], &[short_conv_fwd(&g, &ids, &shape, &x, Some(&d), &base, None, Some(&hist), &y), conv_state_tail_step(&g, &ids, rows as u32, C, TAPS, &x, &hist, &next)]);
        let got = g.read(&y, rows * C as usize);
        for i in 0..rows * C as usize {
            near(got[i], f64::from(whole[at * C as usize + i]), &format!("chunk at {at}"), i);
        }
        hist = next;
        at += rows;
    }
}

#[test]
fn packed_documents_and_blocks_never_see_each_other() {
    let g = gpu_core::testgpu::dev(pipes());
    let ids = ids(&g);
    let mut rng = Lcg::new(random_seed());
    for (segments, block) in [(true, N), (false, 1)] {
        let base_case = case(&mut rng, true, BaseLayout::TapMajor, Some(1.0 / 3.0), segments, false, block);
        let before = forward(&g, &ids, &base_case).y;
        // Rows 0..4 are document 0 (or, with block 1, every row is alone). Perturbing row 1 may change rows of its own
        // document but not any row of a later one.
        let mut perturbed = Case { shape: base_case.shape, x: base_case.x.clone(), dyn_taps: base_case.dyn_taps.clone(), base: base_case.base.clone(), seg: base_case.seg.clone(), hist: base_case.hist.clone() };
        for ci in 0..C as usize {
            perturbed.x[C as usize + ci] += 10.0;
        }
        let after = forward(&g, &ids, &perturbed).y;
        let unaffected_from = if segments { 4 } else { 2 };
        for t in unaffected_from..N as usize {
            for ci in 0..C as usize {
                assert_eq!(before[t * C as usize + ci].to_bits(), after[t * C as usize + ci].to_bits(), "segments={segments} block={block}: row {t} depends on row 1");
            }
        }
        // And it does matter inside the document (the perturbation is not a no-op).
        if segments {
            assert!((0..C as usize).any(|ci| before[2 * C as usize + ci] != after[2 * C as usize + ci]), "row 2 should see row 1 inside its document");
        }
    }
}

#[test]
fn the_static_family_equals_the_existing_static_step_when_no_option_is_on() {
    let g = gpu_core::testgpu::dev(pipes());
    let ids = ids(&g);
    let old = g.kernel_index("causal_conv1d_step").expect("registered");
    let mut rng = Lcg::new(random_seed());
    let cs = case(&mut rng, false, BaseLayout::ChannelMajor, None, false, false, N);
    let mut shape = cs.shape;
    shape.n = 1;
    shape.block = 1;
    shape.base_off = 0;
    let w: Vec<f32> = cs.base[5..].to_vec(); // [c, taps], last tap current: causal_conv1d_step's own layout
    let (wb, new_hist, old_hist) = (g.storage_init("w", &w), g.storage_init("h1", &vec![0.0; (C * (TAPS - 1)) as usize]), g.storage_init("h2", &vec![0.0; (C * (TAPS - 1)) as usize]));
    for t in 0..N as usize {
        let x = g.storage_init("x", &cs.x[t * C as usize..(t + 1) * C as usize]);
        let (y_new, y_old) = (g.storage(u64::from(C)), g.storage(u64::from(C)));
        g.submit(&[], &[short_conv_step(&g, &ids, &shape, &x, None, &wb, &new_hist, &y_new), g.step(old, &[&x, &wb, &old_hist, &y_old], &[1, C, TAPS], C)]);
        let (a, b) = (g.read(&y_new, C as usize), g.read(&y_old, C as usize));
        // The taps are summed newest first here and oldest first there: equal to rounding, not to the bit.
        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() <= 1e-6), "token {t}: {a:?} vs causal_conv1d_step {b:?}");
    }
    let (a, b) = (g.read(&new_hist, (C * (TAPS - 1)) as usize), g.read(&old_hist, (C * (TAPS - 1)) as usize));
    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()), "the ring state differs from causal_conv1d_step's");
}

/// Forward and backward of `x` (`n` rows) with `shape`; `d_base` accumulates from zero.
fn run_with_grads(g: &Gpu, ids: &ShortConvIds, shape: &ShortConvShape, x: &[f32], dyn_taps: &[f32], base: &[f32], seg: Option<&[u32]>, dy: &[f32]) -> [Vec<f32>; 4] {
    let n = shape.n as usize;
    let (xb, db, bb, dyb) = (g.storage_init("x", x), g.storage_init("dyn", dyn_taps), g.storage_init("base", base), g.storage_init("dy", dy));
    let segb = seg.map(|s| seg_buffer(g, s));
    let y = g.storage((n * C as usize) as u64);
    let (dx, dbase, ddyn) = (g.storage((n * C as usize) as u64), g.storage_init("dbase", &vec![0.0; base.len()]), g.storage_init("ddyn", &vec![0.0; n * STRIDE as usize]));
    let mut steps = vec![short_conv_fwd(g, ids, shape, &xb, Some(&db), &bb, segb.as_ref(), None, &y)];
    steps.extend(short_conv_bwd(g, ids, shape, &xb, &dyb, Some(&db), &bb, segb.as_ref(), &dx, &dbase, Some(&ddyn)));
    g.submit(&[], &steps);
    [g.read(&y, n * C as usize), g.read(&dx, n * C as usize), g.read(&dbase, base.len()), g.read(&ddyn, n * STRIDE as usize)]
}

/// A packed row of documents (dynamic taps, residual and scale, segments on) computes each document exactly as the
/// document alone: its rows of `y`, `dx` and `d_dyn` bit for bit (the same taps in the same order), and `d_base`
/// the sum of the documents' own.
#[test]
fn a_packed_row_convolves_each_document_as_it_convolves_alone() {
    let g = gpu_core::testgpu::dev(pipes());
    let ids = ids(&g);
    let mut rng = Lcg::new(random_seed());
    let packed = case(&mut rng, true, BaseLayout::TapMajor, Some(1.0 / 5f32.sqrt()), true, false, N);
    let dy = rng.vec_scaled((N * C) as usize, 1.0);
    let [y, dx, dbase, ddyn] = run_with_grads(&g, &ids, &packed.shape, &packed.x, &packed.dyn_taps, &packed.base, Some(&packed.seg), &dy);
    let (c, stride) = (C as usize, STRIDE as usize);
    let mut dbase_sum = vec![0f64; dbase.len()];
    for (first, len) in [(0usize, 4usize), (4, 4), (8, 3)] {
        let alone = ShortConvShape { n: len as u32, block: len as u32, segments: false, ..packed.shape };
        let rows = |v: &[f32], width: usize| v[first * width..][..len * width].to_vec();
        let [ya, dxa, dbasea, ddyna] = run_with_grads(&g, &ids, &alone, &rows(&packed.x, c), &rows(&packed.dyn_taps, stride), &packed.base, None, &rows(&dy, c));
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&ya), bits(&rows(&y, c)), "document at {first}: y");
        assert_eq!(bits(&dxa), bits(&rows(&dx, c)), "document at {first}: dx");
        assert_eq!(bits(&ddyna), bits(&rows(&ddyn, stride)), "document at {first}: d_dyn");
        for (s, v) in dbase_sum.iter_mut().zip(&dbasea) {
            *s += f64::from(*v);
        }
    }
    for (i, (&got, &want)) in dbase.iter().zip(&dbase_sum).enumerate() {
        near(got, want, "d_base is the sum of the documents'", i);
    }
}
