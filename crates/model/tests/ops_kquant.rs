// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::ops::{Ops, Weight::KQuant}` façade parity (M12).
//!
//! For each affine K-quant dtype (`Q4K` = GGUF Q4_K, `Q8K` = GGUF Q5_K), the
//! façade's `Ops::act_kq` + `Ops::matmul` must reproduce EXACTLY what a
//! hand-driven dispatch of M9's `quant_group_sum` prepass plus M11's
//! `matmul_kq_dyn`/`matmul_kq_gemv` kernels produces on the SAME input - the
//! same "not close, bit for bit" contract `ops_facade_parity.rs` already
//! holds `Weight::I8`/`Weight::Q4` to, extended to the new dtypes this
//! milestone wires into the shared dispatch seam. Both paths run on the SAME
//! `Gpu` (so the SAME compiled kernel pipeline) and start from the SAME f32
//! activation and the SAME packed weight bytes - only the CALL SITE differs
//! (`Ops::matmul`'s selector-driven dispatch vs. a direct `Gpu::step_sliced`
//! call), so any divergence is a real façade wiring bug, not a numeric
//! difference between two independently-derived oracles.
//!
//! `Weight::KQuant` is built directly from hand-packed device buffers here
//! (there is no `Weight::upload` path for it - `Weight::upload` explicitly
//! refuses `Q4K`/`Q8K`, see that function's own doc comment: K-quant's whole
//! point is reaching the device without ever materializing fp32, which a
//! generic `raw: &[f32]` upload path cannot express). The packing helpers
//! below mirror `gguf::kquant`'s canonical device layout (`crates/gguf/tests/
//! kquant.rs`) and `crates/kernels/tests/matmul_kq.rs`'s own scenario
//! builder, restated locally per this test tree's convention of
//! self-contained fixtures.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use kernels::template::interned;
use model::dispatch::I8Scratch;
use model::int8::{quant_rows_steps, QuantRows, GROUP};
use model::ops::{kernel_list, KqScale, Ops, Weight};

/// M14: groups per `wd` super-block entry - `matmul_kq_dyn.wgsl`'s own fixed
/// `GPS=8` (256-element GGUF Q4_K/Q5_K super-block / 32-element group).
const GPS: usize = 8;

/// Pack per-group `(sc, m)` sub-scale byte pairs into `wsm: [n, ceil(ng/2)]`
/// - `gguf::kquant`'s own bit layout, restated here since this crate's tests
/// do not depend on `gguf`.
fn pack_wsm(sc: &[u8], mn: &[u8], n: usize, ng: usize) -> Vec<u32> {
    let words = ng.div_ceil(2);
    let mut out = vec![0u32; n * words];
    for r in 0..n {
        for g in 0..ng {
            let (w, shift) = (g / 2, if g % 2 == 0 { 0u32 } else { 16u32 });
            out[r * words + w] |= (sc[r * ng + g] as u32) << shift;
            out[r * words + w] |= (mn[r * ng + g] as u32) << (shift + 8);
        }
    }
    out
}

/// Pack per-super-block `(d, dmin)` f16 pairs into `wd: [n, ceil(ng/GPS)]`.
fn pack_wd(d: &[f32], dmin: &[f32], n: usize, spb_per_row: usize) -> Vec<u32> {
    let mut out = vec![0u32; n * spb_per_row];
    for r in 0..n {
        for s in 0..spb_per_row {
            let db = half::f16::from_f32(d[r * spb_per_row + s]).to_bits() as u32;
            let dmb = half::f16::from_f32(dmin[r * spb_per_row + s]).to_bits() as u32;
            out[r * spb_per_row + s] = db | (dmb << 16);
        }
    }
    out
}

/// One `(sc, m, d_super, dmin_super)` decomposition - the effective `(ds,
/// dm)` product `wsm`/`wd` decode to is `ds = d*sc`, `dm = dmin*m`, but this
/// file's own gate compares the façade against a HAND-DISPATCHED call to the
/// SAME underlying kernels on the SAME packed buffers (not a host f64
/// oracle), so only the packed pieces themselves are needed here - `sc`/`m`
/// positive-but-small (mimicking the real 6-bit `0..63` sub-scale range),
/// `d_super`/`dmin_super` a per-super-block f16-representable magnitude.
struct KqScaleFixture {
    sc: Vec<u8>,
    mn: Vec<u8>,
    d_super: Vec<f32>,
    dmin_super: Vec<f32>,
}

fn build_kq_scale(rng: &mut Lcg, n: usize, ng: usize, dmin_nonzero: bool) -> KqScaleFixture {
    let spb_per_row = ng.div_ceil(GPS);
    let sc: Vec<u8> = (0..n * ng).map(|_| 1 + (rng.next_u32() % 62) as u8).collect();
    let mn: Vec<u8> = if dmin_nonzero { (0..n * ng).map(|_| 1 + (rng.next_u32() % 62) as u8).collect() } else { vec![0u8; n * ng] };
    let d_super: Vec<f32> = (0..n * spb_per_row).map(|_| 0.001 + rng.unit() * 0.01).collect();
    let dmin_super: Vec<f32> =
        if dmin_nonzero { (0..n * spb_per_row).map(|_| 0.002 + rng.unit() * 0.02).collect() } else { vec![0f32; n * spb_per_row] };
    KqScaleFixture { sc, mn, d_super, dmin_super }
}

impl KqScaleFixture {
    fn wsm(&self, n: usize, ng: usize) -> Vec<u32> {
        pack_wsm(&self.sc, &self.mn, n, ng)
    }
    fn wd(&self, n: usize, ng: usize) -> Vec<u32> {
        pack_wd(&self.d_super, &self.dmin_super, n, ng.div_ceil(GPS))
    }
}

/// `Gpu::storage_init` is f32-only; `wsm`/`wd` are packed `u32` words, so
/// upload them through `storage` + `write` instead.
fn storage_u32(g: &Gpu, data: &[u32]) -> DeviceBuffer {
    let b = g.storage(data.len() as u64);
    g.write(&b, data);
    b
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// `32/bits` unsigned codes per u32, low bits first - the affine `wq` device
/// layout (`gguf::kquant::pack_codes`'s contract, restated here since this
/// crate does not depend on `gguf`).
fn pack_affine_words(codes: &[i32], bits: u32) -> Vec<u32> {
    let per_word = 32 / bits as usize;
    assert_eq!(codes.len() % per_word, 0);
    let mask = (1u32 << bits) - 1;
    codes
        .chunks_exact(per_word)
        .map(|c| c.iter().enumerate().fold(0u32, |w, (b, &v)| w | ((v as u32 & mask) << (bits * b as u32))))
        .collect()
}

fn rand_unsigned_codes(rng: &mut Lcg, n: usize, bits: u32) -> Vec<i32> {
    let span = 1u32 << bits;
    (0..n).map(|_| (rng.next_u64() % span as u64) as i32).collect()
}

fn rand_pos_f32(rng: &mut Lcg, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| lo + rng.unit() * (hi - lo)).collect()
}

/// One `(x, w)` scenario for one affine dtype, driven through BOTH the
/// façade and a hand-dispatched call to the same underlying kernels.
#[allow(clippy::too_many_arguments)]
fn check_kquant(dtype: Dtype, bits: u32, dyn_kname: &'static str, gemv_kname: &'static str, m: usize, n: usize, k: usize, dmin_nonzero: bool) {
    assert!(k % GROUP == 0, "test shape must keep k a multiple of GROUP (32)");
    let ng = k / GROUP;

    // `kernel_list()` (M12's own builder) already registers BOTH `CODE_BITS`
    // specialisations of `matmul_kq_dyn`/`matmul_kq_gemv` via `interned`, so
    // this just needs to feed it to a real device - the tiled `matmul_kq_dyn`
    // kernel is register-tiled/multi-barrier, so (like `matmul_i8_dyn`) it
    // does not run under the CPU JIT, matching why `crates/kernels/tests/
    // matmul_kq.rs` also builds a `Gpu` directly rather than going through
    // `gpu_core::testgpu::dev` (which may pick the CPU JIT).
    let (dn, _) = interned("matmul_kq_dyn", kernels::MATMUL_KQ_DYN, &[("CODE_BITS", bits)]).unwrap();
    let (gn, _) = interned("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV, &[("CODE_BITS", bits)]).unwrap();
    assert_eq!(dn, dyn_kname, "interned name must match Ops::bind's own literal");
    assert_eq!(gn, gemv_kname, "interned name must match Ops::bind's own literal");
    let g = Gpu::new(kernel_list());
    let ops = Ops::new(g).expect("Ops::new");
    let g = ops.gpu();

    let mut rng = Lcg::new(0xA0000000_u64 ^ (dtype_seed(dtype) << 32) ^ m as u64 ^ ((n as u64) << 16) ^ ((k as u64) << 32));
    let x_h = rand_pos_f32(&mut rng, m * k, -8.0, 8.0);
    let codes_w = rand_unsigned_codes(&mut rng, n * k, bits);
    let sc = build_kq_scale(&mut rng, n, ng, dmin_nonzero);

    let x = g.storage_init("x", &x_h);
    let wq_words = pack_affine_words(&codes_w, bits);
    let wq = g.storage(wq_words.len() as u64);
    g.write(&wq, &wq_words);
    let wsm = storage_u32(g, &sc.wsm(n, ng));
    let wd = storage_u32(g, &sc.wd(n, ng));

    let weight = Weight::KQuant { w: wq, sz: KqScale::Packed { wsm, wd }, n: n as u32, k: k as u32, group: GROUP as u32, bits, affine: true };
    assert_eq!(weight.dtype(), dtype, "Weight::KQuant.dtype() must resolve bits={bits} to {dtype:?}");

    // ---- Path A: the façade ----
    let mut got_steps = Vec::new();
    let act = ops.act_kq(&mut got_steps, &x, 0, m as u32, k as u32);
    let out_got = g.storage((m * n) as u64);
    ops.matmul(&mut got_steps, &weight, &act, &out_got, 0);
    g.submit(&[], &got_steps);
    let got = g.read(&out_got, m * n);

    // ---- Path B: hand-dispatched against the SAME underlying kernels ----
    let (max_abs_row, quant_pack, group_sum) = (idx(g, "max_abs_row"), idx(g, "quant_pack"), idx(g, "quant_group_sum"));
    let scr = I8Scratch::new(g, m as u64, m as u64, &[k as u32]);
    let xgs = g.storage((m * ng) as u64);
    let mut want_steps = Vec::new();
    want_steps.extend(quant_rows_steps(
        g,
        QuantRows { kernels: [max_abs_row, quant_pack], x: &x, sx: &scr.sx, xq: scr.xq_for(k as u32), xgs: Some((group_sum, &xgs)) },
        0,
        m as u32,
        k as u32,
    ));
    let Weight::KQuant { w: wq2, sz: KqScale::Packed { wsm: wsm2, wd: wd2 }, .. } = &weight else { unreachable!() };
    let out_want = g.storage((m * n) as u64);
    let params = [m as u32, k as u32, n as u32];
    let bufs: [&DeviceBuffer; 7] = [scr.xq_for(k as u32), wq2, &scr.sx, wsm2, wd2, &xgs, &out_want];
    if m <= 32 {
        want_steps.push(g.step(idx(g, gemv_kname), &bufs, &params, n as u32 * 64));
    } else {
        let threads = (m as u32).div_ceil(128) * (n as u32).div_ceil(128) * 256;
        want_steps.push(g.step(idx(g, dyn_kname), &bufs, &params, threads));
    }
    g.submit(&[], &want_steps);
    let want = g.read(&out_want, m * n);

    assert_eq!(got, want, "{dtype:?} tier m={m} n={n} k={k} dmin_nonzero={dmin_nonzero}: facade output != hand-dispatched M11 kernel output (bit-for-bit)");
}

fn dtype_seed(dtype: Dtype) -> u64 {
    match dtype {
        Dtype::Q4K => 4,
        Dtype::Q8K => 8,
        _ => 0,
    }
}

/// `m ∈ {8, 192}` covers the GEMV decode regime (`m <= 32`) and the tiled
/// prefill regime (spanning more than one 128x128 tile), for both affine
/// dtypes, with `dmin` nonzero (the min-correction term load-bearing, not
/// coincidentally zero).
#[test]
fn q4k_facade_matches_hand_dispatched_kernels() {
    for &m in &[8usize, 192usize] {
        check_kquant(Dtype::Q4K, 4, kname_kq_dyn_4(), kname_kq_gemv_4(), m, 176, 256, true);
    }
}

#[test]
fn q8k_facade_matches_hand_dispatched_kernels() {
    for &m in &[8usize, 192usize] {
        check_kquant(Dtype::Q8K, 8, kname_kq_dyn_8(), kname_kq_gemv_8(), m, 176, 256, true);
    }
}

/// The literal names `Ops::bind` uses - re-derived via `interned` inside
/// `check_kquant` itself for the actual dispatch, these four just name what
/// this file expects `Ops::bind` to choose, so a drift there fails at THIS
/// assertion rather than as a mystifying kernel-not-found panic deep inside
/// `Ops::matmul`.
fn kname_kq_dyn_4() -> &'static str {
    "matmul_kq_dyn#CODE_BITS=4"
}
fn kname_kq_gemv_4() -> &'static str {
    "matmul_kq_gemv#CODE_BITS=4"
}
fn kname_kq_dyn_8() -> &'static str {
    "matmul_kq_dyn#CODE_BITS=8"
}
fn kname_kq_gemv_8() -> &'static str {
    "matmul_kq_gemv#CODE_BITS=8"
}

/// `Ops::matmul` must refuse LOUDLY, not silently read a zero/missing
/// buffer, when a `Weight::KQuant` is paired with an activation built via
/// `Ops::act` (no `xgs`) or `Ops::act_f32` (no packed activation at all) -
/// the exact "affine weight without an xgs-built activation" mistake M12's
/// own task brief calls out.
#[test]
#[should_panic(expected = "xgs")]
fn kquant_matmul_refuses_an_activation_built_without_xgs() {
    let g = Gpu::new(kernel_list());
    let ops = Ops::new(g).expect("Ops::new");
    let g = ops.gpu();

    let (m, n, k) = (8usize, 32usize, 64usize);
    let x_h = vec![0.1f32; m * k];
    let x = g.storage_init("x", &x_h);
    let codes_w = vec![1i32; n * k];
    let wq_words = pack_affine_words(&codes_w, 4);
    let wq = g.storage(wq_words.len() as u64);
    g.write(&wq, &wq_words);
    let ng = k / GROUP;
    let wsm = storage_u32(g, &vec![0x0101_0101u32; n * ng.div_ceil(2)]);
    let wd = storage_u32(g, &vec![0x3000_3000u32; n * ng.div_ceil(GPS)]);
    let weight = Weight::KQuant { w: wq, sz: KqScale::Packed { wsm, wd }, n: n as u32, k: k as u32, group: GROUP as u32, bits: 4, affine: true };

    let mut s = Vec::new();
    // Built with `Ops::act` (no xgs), not `Ops::act_kq` - must panic.
    let act = ops.act(&mut s, &x, 0, m as u32, k as u32);
    let out = g.storage((m * n) as u64);
    ops.matmul(&mut s, &weight, &act, &out, 0);
}
