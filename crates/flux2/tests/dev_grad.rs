// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device (GPU) FLUX.2 block backward vs the gradchecked host reference.
//!
//! `grad.rs` is finite-difference-gradchecked (`tests/block_grad.rs`); this test
//! confirms the GPU path (`devgrad`) reproduces those gradients with real
//! training kernels - `matmul_dx_reg`/`matmul_dw_reg`, `layernorm_dx` +
//! `film_row_dx`/`film_row_dsb`, `gate_row_dh`/`gate_row_dg`,
//! `attn_bwd_*_bidir`, `silu_bwd_*`, interleaved-RoPE-via-negated-sin and
//! `rms_inv_eps`/`rmsnorm_dw`/`rmsnorm_dx_eps`. Matching the gradchecked host
//! to fp32 tolerance transitively validates the device gradients.
//!
//! The **adapter** gradients are the ones that matter: the device never forms a
//! dense `dW`, it produces `dA`/`dB` from the low-rank intermediates directly.
//! The host side of the comparison therefore runs `Pair::project` on the dense
//! `dW` the reference does produce - the two must agree, and that equality is
//! the whole justification for the cheap path.
//!
//! Both cosine AND rel_l2 are asserted on every tensor: cosine alone is
//! scale-invariant, so a uniformly mis-scaled gradient (a wrong `α/r`, a
//! forgotten epsilon) passes it at 1.000000.
//!
//! Needs a GPU: `BRAIN_DEV_GPU=1`.

use flux2::devgrad::{BlockDev, DoubleDev, SingleDev, N_SITES};
use flux2::grad::{double_backward, double_forward, single_backward, single_forward, Dims, DoubleMods, DoubleW, Mod, SingleW, StreamW};
use model::lora::Pair;

// A tile-friendly tiny config whose every sliced binding offset lands on the
// 256-byte (64-float) storage alignment the backend requires.
const NT: usize = 64;
const NI: usize = 64;
const D: usize = 64;
const NH: usize = 4;
const MLP: usize = 128;
const RANK: usize = 8;
const ALPHA: f32 = RANK as f32;

fn dims() -> Dims {
    Dims { nt: NT, ni: NI, d: D, nh: NH, mlp: MLP }
}

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
}

fn vof(n: usize, r: &mut impl FnMut() -> f32, s: f32) -> Vec<f32> {
    (0..n).map(|_| r() * s).collect()
}
fn gain(n: usize, r: &mut impl FnMut() -> f32, s: f32) -> Vec<f32> {
    (0..n).map(|_| 1.0 + r() * s).collect()
}
fn f64v(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-300)
}

/// Relative L2 of `dev` against `host`, well-defined when `host` is all zero.
fn rel_l2(dev: &[f64], host: &[f64]) -> f64 {
    let nh: f64 = host.iter().map(|x| x * x).sum::<f64>().sqrt();
    let diff: f64 = dev.iter().zip(host).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
    diff / nh.max(1e-12)
}

/// Accumulates the worst (cosine, rel_l2) over every compared tensor.
struct Worst {
    cos: f64,
    rel: f64,
    cos_at: String,
    rel_at: String,
    n: usize,
}

impl Worst {
    fn new() -> Worst {
        Worst { cos: 1.0, rel: 0.0, cos_at: String::new(), rel_at: String::new(), n: 0 }
    }
    fn add(&mut self, name: &str, dev: &[f64], host: &[f64]) {
        assert_eq!(dev.len(), host.len(), "{name}: length {} vs {}", dev.len(), host.len());
        let allzero = host.iter().all(|&v| v == 0.0) && dev.iter().all(|&v| v == 0.0);
        let c = if allzero { 1.0 } else { cosine(dev, host) };
        let r = rel_l2(dev, host);
        if c < self.cos {
            self.cos = c;
            self.cos_at = name.to_string();
        }
        if r > self.rel {
            self.rel = r;
            self.rel_at = name.to_string();
        }
        self.n += 1;
    }
    fn assert_within(&self, cos_floor: f64, rel_ceiling: f64, what: &str) {
        eprintln!("{what}: {} tensors, worst cosine {:.9} ({}), worst rel_l2 {:.3e} ({})", self.n, self.cos, self.cos_at, self.rel, self.rel_at);
        assert!(self.cos > cos_floor, "{what}: worst cosine {:.9} on {} < {cos_floor}", self.cos, self.cos_at);
        assert!(self.rel < rel_ceiling, "{what}: worst rel_l2 {:.3e} on {} > {rel_ceiling:e}", self.rel, self.rel_at);
    }
}

/// One targeted linear's host-side adapter, plus the effective weight it makes.
struct Adapted {
    base: Vec<f32>,
    pair: Pair,
    eff: Vec<f32>,
}

fn adapted(out: usize, inn: usize, r: &mut impl FnMut() -> f32, scale: f32) -> Adapted {
    let base = vof(out * inn, r, 0.15);
    // B is deliberately NON-zero: a fresh adapter has B = 0, which makes the
    // whole up-projection path a no-op and every bug in it invisible.
    let a = vof(RANK * inn, r, 0.3);
    let b = vof(out * RANK, r, 0.3);
    let pair = Pair::from_ab(out, inn, RANK, a, b);
    let mut eff = base.clone();
    pair.delta(scale, &mut eff);
    Adapted { base, pair, eff }
}

fn mods(r: &mut impl FnMut() -> f32) -> Mod<f32> {
    Mod { shift: vof(D, r, 0.3), scale: vof(D, r, 0.3), gate: vof(D, r, 0.6) }
}

fn zero_mod() -> Mod<f32> {
    Mod { shift: vec![0.0; D], scale: vec![0.0; D], gate: vec![0.0; D] }
}

fn to64_mod(m: &Mod<f32>) -> Mod<f64> {
    Mod { shift: f64v(&m.shift), scale: f64v(&m.scale), gate: f64v(&m.gate) }
}

fn rope_tables(n: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let cos = (0..n * half).map(|i| (i as f32 * 0.11).cos()).collect();
    let sin = (0..n * half).map(|i| (i as f32 * 0.11).sin()).collect();
    (cos, sin)
}

fn skip() -> bool {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        brain_testutil::skip_unavailable("set BRAIN_DEV_GPU=1 (needs a GPU) for the FLUX.2 device block-backward parity test");
        return true;
    }
    false
}

/// How the engine under test gets its device. The CPU arm exists because a
/// workgroup-barrier reduction with no barrier-free sibling can return
/// all-zero gradients on `backend-cpu` alone, and this graph is built almost
/// entirely out of workgroup-tiled GEMMs - so the same gate runs on both
/// backends, and the CPU one needs no hardware to reach.
fn engine(n: usize, cpu: bool) -> BlockDev {
    if cpu {
        BlockDev::from_gpu(gpu_core::Gpu::new_cpu(flux2::devgrad::KERNELS), n, D, NH, MLP, RANK)
    } else {
        BlockDev::new(n, D, NH, MLP, RANK)
    }
}

// ---- double block ----

/// One double-block stream: seven adapted linears plus the two QK-norm scales.
struct Stream {
    lin: [Adapted; 7],
    nq: Vec<f32>,
    nk: Vec<f32>,
}

fn stream(r: &mut impl FnMut() -> f32, scale: f32) -> Stream {
    let hd = D / NH;
    Stream {
        lin: [
            adapted(D, D, r, scale),
            adapted(D, D, r, scale),
            adapted(D, D, r, scale),
            adapted(D, D, r, scale),
            adapted(MLP, D, r, scale),
            adapted(MLP, D, r, scale),
            adapted(D, MLP, r, scale),
        ],
        nq: gain(hd, r, 0.1),
        nk: gain(hd, r, 0.1),
    }
}

fn stream_w64(s: &Stream) -> StreamW<f64> {
    StreamW {
        wq: f64v(&s.lin[0].eff),
        wk: f64v(&s.lin[1].eff),
        wv: f64v(&s.lin[2].eff),
        nq: f64v(&s.nq),
        nk: f64v(&s.nk),
        wo: f64v(&s.lin[3].eff),
        w1: f64v(&s.lin[4].eff),
        w3: f64v(&s.lin[5].eff),
        w2: f64v(&s.lin[6].eff),
    }
}

#[test]
fn device_double_block_backward_matches_host() {
    if skip() {
        return;
    }
    double_block_gate(false);
}

/// The same gate on `backend-cpu` - no GPU needed, so it runs in the ordinary
/// test pass.
#[test]
fn cpu_backend_double_block_backward_matches_host() {
    double_block_gate(true);
}

fn double_block_gate(cpu: bool) {
    let dm = dims();
    let (n, hd) = (dm.n(), dm.hd());
    let scale = ALPHA / RANK as f32;
    let mut r = rng(0x51a7_0d1e);
    let img = stream(&mut r, scale);
    let txt = stream(&mut r, scale);
    let sites: [Mod<f32>; N_SITES] = [mods(&mut r), mods(&mut r), mods(&mut r), mods(&mut r), zero_mod(), zero_mod()];
    let x = vof(n * D, &mut r, 1.0);
    let dout = vof(n * D, &mut r, 1.0);
    let (cos, sin) = rope_tables(n, hd);

    // ---- host (f64, the gradchecked reference) on the EFFECTIVE weights ----
    let hw = DoubleW { img: stream_w64(&img), txt: stream_w64(&txt) };
    let hm = DoubleMods { img1: to64_mod(&sites[0]), img2: to64_mod(&sites[1]), txt1: to64_mod(&sites[2]), txt2: to64_mod(&sites[3]) };
    let (_o, cache) = double_forward(dm, &hw, &f64v(&x), &hm, &f64v(&cos), &f64v(&sin));
    let hg = double_backward(dm, &hw, &hm, &cache, &f64v(&dout));

    // ---- device ----
    let eng = engine(n, cpu);
    let mkstream = |s: &Stream| {
        eng.stream(&s.lin[0].base, &s.lin[1].base, &s.lin[2].base, &s.lin[3].base, &s.lin[4].base, &s.lin[5].base, &s.lin[6].base, &s.nq, &s.nk)
    };
    let dev = DoubleDev { img: mkstream(&img), txt: mkstream(&txt) };
    eng.upload_mods(&sites);
    eng.upload_rope(&cos, &sin);
    for (host, dl) in [&img, &txt].iter().zip([&dev.img, &dev.txt]) {
        for (h, d) in host.lin.iter().zip([&dl.wq, &dl.wk, &dl.wv, &dl.wo, &dl.w1, &dl.w3, &dl.w2]) {
            eng.upload_lora(d, &h.pair.a, &h.pair.b, scale);
        }
    }
    let xb = eng.slab(n);
    let doutb = eng.slab(n);
    let dxb = eng.slab(n);
    eng.gpu().write_f32(&xb, &x);
    eng.gpu().write_f32(&doutb, &dout);
    eng.double_backward(&dev, dm, &xb, &doutb, &dxb);

    // ---- compare ----
    let mut w = Worst::new();
    let hstreams = [(&img, &hg.img, &dev.img, "img"), (&txt, &hg.txt, &dev.txt, "txt")];
    for (host, hgs, dl, tag) in hstreams {
        let hdw: [&Vec<f64>; 7] = [&hgs.wq, &hgs.wk, &hgs.wv, &hgs.wo, &hgs.w1, &hgs.w3, &hgs.w2];
        let dls = [&dl.wq, &dl.wk, &dl.wv, &dl.wo, &dl.w1, &dl.w3, &dl.w2];
        for (i, name) in ["wq", "wk", "wv", "wo", "w1", "w3", "w2"].iter().enumerate() {
            let dw32: Vec<f32> = hdw[i].iter().map(|&v| v as f32).collect();
            let (hda, hdb) = host.lin[i].pair.project(&dw32, scale);
            let (dda, ddb) = eng.lin_grads(dls[i], scale);
            w.add(&format!("{tag}.{name}.dA"), &f64v(&dda), &f64v(&hda));
            w.add(&format!("{tag}.{name}.dB"), &f64v(&ddb), &f64v(&hdb));
        }
        let (dnq, dnk) = eng.stream_norm_grads(dl);
        w.add(&format!("{tag}.nq"), &f64v(&dnq), &hgs.nq);
        w.add(&format!("{tag}.nk"), &f64v(&dnk), &hgs.nk);
    }
    let sg = eng.mod_grads();
    for (i, (name, hs)) in [("img1", &hg.img1), ("img2", &hg.img2), ("txt1", &hg.txt1), ("txt2", &hg.txt2)].iter().enumerate() {
        w.add(&format!("{name}.shift"), &f64v(&sg[i].shift), &hs.shift);
        w.add(&format!("{name}.scale"), &f64v(&sg[i].scale), &hs.scale);
        w.add(&format!("{name}.gate"), &f64v(&sg[i].gate), &hs.gate);
    }
    w.add("dx", &f64v(&eng.gpu().read(&dxb, n * D)), &hg.dx);
    w.assert_within(0.9999999, 1e-5, &format!("FLUX.2 {} DOUBLE block backward", if cpu { "backend-cpu" } else { "device" }));
}

// ---- single block ----

#[test]
fn device_single_block_backward_matches_host() {
    if skip() {
        return;
    }
    single_block_gate(false);
}

/// The `backend-cpu` twin of [`device_single_block_backward_matches_host`].
#[test]
fn cpu_backend_single_block_backward_matches_host() {
    single_block_gate(true);
}

fn single_block_gate(cpu: bool) {
    let dm = dims();
    let (n, hd) = (dm.n(), dm.hd());
    let scale = ALPHA / RANK as f32;
    let mut r = rng(0x9911_2233);
    let lin: [Adapted; 7] = [
        adapted(D, D, &mut r, scale),
        adapted(D, D, &mut r, scale),
        adapted(D, D, &mut r, scale),
        adapted(MLP, D, &mut r, scale),
        adapted(MLP, D, &mut r, scale),
        adapted(D, D, &mut r, scale),
        adapted(D, MLP, &mut r, scale),
    ];
    let nq = gain(hd, &mut r, 0.1);
    let nk = gain(hd, &mut r, 0.1);
    let smod = mods(&mut r);
    let sites: [Mod<f32>; N_SITES] = [zero_mod(), zero_mod(), zero_mod(), zero_mod(), smod.clone(), zero_mod()];
    let x = vof(n * D, &mut r, 1.0);
    let dout = vof(n * D, &mut r, 1.0);
    let (cos, sin) = rope_tables(n, hd);

    let hw = SingleW {
        wq: f64v(&lin[0].eff),
        wk: f64v(&lin[1].eff),
        wv: f64v(&lin[2].eff),
        nq: f64v(&nq),
        nk: f64v(&nk),
        w1: f64v(&lin[3].eff),
        w3: f64v(&lin[4].eff),
        wo_a: f64v(&lin[5].eff),
        wo_b: f64v(&lin[6].eff),
    };
    let hm = to64_mod(&smod);
    let (_o, cache) = single_forward(dm, &hw, &f64v(&x), &hm, &f64v(&cos), &f64v(&sin));
    let hg = single_backward(dm, &hw, &hm, &cache, &f64v(&dout));

    let eng = engine(n, cpu);
    let dev: SingleDev = eng.single(&lin[0].base, &lin[1].base, &lin[2].base, &lin[3].base, &lin[4].base, &lin[5].base, &lin[6].base, &nq, &nk);
    eng.upload_mods(&sites);
    eng.upload_rope(&cos, &sin);
    for (h, d) in lin.iter().zip([&dev.wq, &dev.wk, &dev.wv, &dev.w1, &dev.w3, &dev.wo_a, &dev.wo_b]) {
        eng.upload_lora(d, &h.pair.a, &h.pair.b, scale);
    }
    let xb = eng.slab(n);
    let doutb = eng.slab(n);
    let dxb = eng.slab(n);
    eng.gpu().write_f32(&xb, &x);
    eng.gpu().write_f32(&doutb, &dout);
    eng.single_backward(&dev, dm, &xb, &doutb, &dxb);

    let mut w = Worst::new();
    let hdw: [&Vec<f64>; 7] = [&hg.wq, &hg.wk, &hg.wv, &hg.w1, &hg.w3, &hg.wo_a, &hg.wo_b];
    let dls = [&dev.wq, &dev.wk, &dev.wv, &dev.w1, &dev.w3, &dev.wo_a, &dev.wo_b];
    for (i, name) in ["wq", "wk", "wv", "w1", "w3", "wo_a", "wo_b"].iter().enumerate() {
        let dw32: Vec<f32> = hdw[i].iter().map(|&v| v as f32).collect();
        let (hda, hdb) = lin[i].pair.project(&dw32, scale);
        let (dda, ddb) = eng.lin_grads(dls[i], scale);
        w.add(&format!("{name}.dA"), &f64v(&dda), &f64v(&hda));
        w.add(&format!("{name}.dB"), &f64v(&ddb), &f64v(&hdb));
    }
    let (dnq, dnk) = eng.single_norm_grads(&dev);
    w.add("nq", &f64v(&dnq), &hg.nq);
    w.add("nk", &f64v(&dnk), &hg.nk);
    let sg = eng.mod_grads();
    w.add("sgl.shift", &f64v(&sg[4].shift), &hg.m.shift);
    w.add("sgl.scale", &f64v(&sg[4].scale), &hg.m.scale);
    w.add("sgl.gate", &f64v(&sg[4].gate), &hg.m.gate);
    w.add("dx", &f64v(&eng.gpu().read(&dxb, n * D)), &hg.dx);
    w.assert_within(0.9999999, 1e-5, &format!("FLUX.2 {} SINGLE block backward", if cpu { "backend-cpu" } else { "device" }));
}
