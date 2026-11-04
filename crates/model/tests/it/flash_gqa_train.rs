// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The causal GQA flash trio (`flash_attn_causal_gqa` forward with row
//! log-sum-exp, `_bwd_dq`, `_bwd_dkv`) as a TRAINING attention: against the
//! materialised `gqa_fwd`/`gqa_bwd` chain it replaces, at the head widths
//! real models ship (40 and 128 for the qwen3/qwen35 test configs, 256 for
//! the 27B), GQA ratio 6, a ragged `t`, and `b > 1`.
//!
//! These run on the pooled wgpu test device: the CPU backend cannot run a
//! shared-memory barrier kernel (`DeviceCaps::workgroup_reductions` is false
//! there), which is exactly why `FlashGqaIds::supported` is false on it and
//! the mixer falls back to the materialised chain.
//!
//! The `#[ignore]`d tests at the end run the real 27B attention width (24 query
//! heads over 4 kv heads at head_dim 256) on a CUDA device:
//! `BRAIN_BACKEND=cuda gpu cargo test -p brain-model --test flash_gqa_train -- --ignored --nocapture --test-threads=1`.
//!
//! Swedish Embedded AB implements memory-bounded attention training kernels
//! for its clients. If your team needs expertise in long-context fine-tuning
//! on a single GPU then you can procure our services by sending an email to
//! info@swedishembedded.com.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::{flash_gqa_causal_bwd, flash_gqa_causal_fwd, gqa_bwd, gqa_fwd, FlashGqaIds, Gqa, KernelIds, UNREGISTERED};
use model::gqa_mixer::{gqa_mixer_bwd, gqa_mixer_fwd_with, AttnSaved, GqaMixerGrads, GqaMixerIds, GqaMixerShape, GqaMixerWeights};

const KERNELS: &[(&str, &str)] = &[
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("concat2", kernels::CONCAT2),
    ("sigmoid", kernels::SIGMOID),
    ("sigmoid_bwd", kernels::SIGMOID_BWD),
    ("mul", kernels::MUL),
    ("rope2d_partial", kernels::ROPE2D_PARTIAL),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
    ("gqa_bwd_dv", kernels::GQA_BWD_DV),
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ),
    ("gqa_bwd_dk", kernels::GQA_BWD_DK),
    ("flash_attn_causal_gqa", kernels::FLASH_ATTN_CAUSAL_GQA),
    ("flash_attn_causal_gqa_bwd_dq", kernels::FLASH_ATTN_CAUSAL_GQA_BWD_DQ),
    ("flash_attn_causal_gqa_bwd_dkv", kernels::FLASH_ATTN_CAUSAL_GQA_BWD_DKV),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn kernel_ids(g: &Gpu) -> KernelIds {
    KernelIds {
        rmsnorm: idx(g, "rmsnorm"),
        rms_inv: idx(g, "rms_inv"),
        rmsnorm_dx: idx(g, "rmsnorm_dx"),
        rmsnorm_dw: idx(g, "rmsnorm_dw"),
        rope: 0,
        rope_bwd: 0,
        gqa_scores: idx(g, "gqa_scores"),
        gqa_apply: idx(g, "gqa_apply"),
        attn_softmax: idx(g, "attn_softmax"),
        gqa_dscores: idx(g, "gqa_bwd_dscores"),
        gqa_dv: idx(g, "gqa_bwd_dv"),
        gqa_dq: idx(g, "gqa_bwd_dq"),
        gqa_dk: idx(g, "gqa_bwd_dk"),
        silu_mul: 0,
        silu_da: 0,
        silu_db: 0,
        rmsnorm_rows: UNREGISTERED,
        rmsnorm_dx_rows: UNREGISTERED,
    }
}

fn flash_ids(g: &Gpu) -> FlashGqaIds {
    FlashGqaIds { fwd: idx(g, "flash_attn_causal_gqa"), bwd_dq: idx(g, "flash_attn_causal_gqa_bwd_dq"), bwd_dkv: idx(g, "flash_attn_causal_gqa_bwd_dkv"), native: false }
}

fn assert_close(what: &str, got: &[f32], want: &[f32], rel: f32) {
    let scale = want.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
    let worst = got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
    assert!(got.iter().all(|x| x.is_finite()), "{what}: non-finite output");
    assert!(worst <= rel * scale.max(1.0), "{what}: worst abs diff {worst:e} at scale {scale:e}");
}

/// Forward output, `lse`, and `dq`/`dk`/`dv` of the flash trio against the
/// materialised chain. The backward is fed the REFERENCE context so a forward
/// mismatch cannot hide inside a backward one.
#[test]
fn flash_trio_matches_the_materialised_chain_at_every_head_width() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let ids = kernel_ids(&g);
    let flash = flash_ids(&g);
    // (b, t, n_heads, n_kv_heads, head_dim): ratio 6 at 256 (the 27B's own),
    // a non-multiple of every tile edge for t, hd 40 (qwen35 tiny) and 128.
    let cases = [(1u32, 137u32, 12u32, 2u32, 256u32), (2, 70, 6, 1, 40), (1, 137, 4, 2, 128), (1, 19, 6, 3, 256)];
    for (case, &(b, t, n_heads, n_kv_heads, head_dim)) in cases.iter().enumerate() {
        assert!(FlashGqaIds::supported(&g, head_dim), "case {case}: the test device must be able to run the flash trio at head_dim {head_dim}");
        let (hq, hkv) = (n_heads * head_dim, n_kv_heads * head_dim);
        let mut rng = Lcg::new(0x5eed_0000 + case as u64); // seed: fixed per case so a failure reproduces
        let q = g.storage_init("q", &rng.vec_scaled((b * t * hq) as usize, 1.0));
        let k = g.storage_init("k", &rng.vec_scaled((b * t * hkv) as usize, 1.0));
        let v = g.storage_init("v", &rng.vec_scaled((b * t * hkv) as usize, 1.0));
        let d_ctx = g.storage_init("d_ctx", &rng.vec_scaled((b * t * hq) as usize, 1.0));
        let ga = Gqa { b, t, n_heads, n_kv_heads, head_dim };
        let sq = (b * n_heads * t * t) as u64;

        let (scores, probs, d_scores) = (g.storage(sq), g.storage(sq), g.storage(sq));
        let ctx = g.storage((b * t * hq) as u64);
        let (dq_ref, dk_ref, dv_ref) = (g.storage((b * t * hq) as u64), g.storage((b * t * hkv) as u64), g.storage((b * t * hkv) as u64));
        let mut steps = gqa_fwd(&g, &ids, &ga, &q, &k, &v, &scores, &probs, &ctx);
        steps.extend(gqa_bwd(&g, &ids, &ga, &q, &k, &v, &probs, &d_ctx, &d_scores, &dq_ref, &dk_ref, &dv_ref));
        g.submit(&[], &steps);

        let ctx_flash = g.storage((b * t * hq) as u64);
        let lse = g.storage((b * n_heads * t) as u64);
        let dsum = g.storage((b * n_heads * t) as u64);
        let (dq, dk, dv) = (g.storage((b * t * hq) as u64), g.storage((b * t * hkv) as u64), g.storage((b * t * hkv) as u64));
        let mut steps = vec![flash_gqa_causal_fwd(&g, &flash, &ga, &q, &k, &v, &ctx_flash, &lse)];
        steps.extend(flash_gqa_causal_bwd(&g, &flash, &ga, &q, &k, &v, &ctx, &lse, &d_ctx, &dsum, &dq, &dk, &dv));
        g.submit(&[], &steps);

        let tag = format!("case {case} (b={b} t={t} nh={n_heads} nkv={n_kv_heads} hd={head_dim})");
        assert_close(&format!("{tag} ctx"), &g.read(&ctx_flash, (b * t * hq) as usize), &g.read(&ctx, (b * t * hq) as usize), 1e-4);
        assert_close(&format!("{tag} dq"), &g.read(&dq, (b * t * hq) as usize), &g.read(&dq_ref, (b * t * hq) as usize), 1e-4);
        assert_close(&format!("{tag} dk"), &g.read(&dk, (b * t * hkv) as usize), &g.read(&dk_ref, (b * t * hkv) as usize), 1e-4);
        assert_close(&format!("{tag} dv"), &g.read(&dv, (b * t * hkv) as usize), &g.read(&dv_ref, (b * t * hkv) as usize), 1e-4);

        // lse against the host, from the materialised scores.
        let (s_host, lse_got) = (g.read(&scores, sq as usize), g.read(&lse, (b * n_heads * t) as usize));
        for bh in 0..(b * n_heads) as usize {
            for i in 0..t as usize {
                let row = &s_host[(bh * t as usize + i) * t as usize..][..=i];
                let m = row.iter().fold(f32::MIN, |a, &x| a.max(x));
                let want = m + row.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
                let got = lse_got[bh * t as usize + i];
                assert!((got - want).abs() < 1e-4 * want.abs().max(1.0), "{tag} lse bh={bh} i={i}: {got} vs {want}");
            }
        }
    }
}

// ---- the full mixer --------------------------------------------------------

fn mixer_ids(g: &Gpu) -> GqaMixerIds {
    GqaMixerIds {
        kernels: kernel_ids(g),
        concat_split: idx(g, "concat_split"),
        concat2: idx(g, "concat2"),
        sigmoid: idx(g, "sigmoid"),
        sigmoid_bwd: idx(g, "sigmoid_bwd"),
        mul: idx(g, "mul"),
        rope2d_partial: idx(g, "rope2d_partial"),
    }
}

/// Host-side inputs of one mixer run.
#[derive(Clone)]
struct MixerInputs {
    q_full: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    d_out: Vec<f32>,
}

impl MixerInputs {
    fn draw(shape: &GqaMixerShape, rng: &mut Lcg) -> Self {
        let n = (shape.b * shape.t) as usize;
        let (qd, kvd, hd, half) = (shape.qd() as usize, shape.kvd() as usize, shape.head_dim as usize, shape.rotary_half as usize);
        Self {
            q_full: rng.vec_scaled(n * 2 * qd, 1.0),
            k: rng.vec_scaled(n * kvd, 1.0),
            v: rng.vec_scaled(n * kvd, 1.0),
            q_norm: rng.vec_scaled(hd, 1.0).iter().map(|x| 1.0 + 0.3 * x).collect(),
            k_norm: rng.vec_scaled(hd, 1.0).iter().map(|x| 1.0 + 0.3 * x).collect(),
            cos: rng.vec_scaled(n * half, 1.0),
            sin: rng.vec_scaled(n * half, 1.0),
            d_out: rng.vec_scaled(n * qd, 1.0),
        }
    }
}

struct MixerRun {
    out: Vec<f32>,
    d_q_full: Vec<f32>,
    d_k: Vec<f32>,
    d_v: Vec<f32>,
    d_q_norm: Vec<f32>,
    d_k_norm: Vec<f32>,
    flash_saved: bool,
}

/// One forward (and, with `backward`, one backward) of the mixer on `g`.
fn run_mixer(g: &Gpu, shape: &GqaMixerShape, x: &MixerInputs, flash: bool, backward: bool) -> MixerRun {
    let n = shape.b * shape.t;
    let (qd, kvd, hd) = (shape.qd(), shape.kvd(), shape.head_dim);
    let ids = mixer_ids(g);
    let (q_full, k, v) = (g.storage_init("q_full", &x.q_full), g.storage_init("k", &x.k), g.storage_init("v", &x.v));
    let (q_norm, k_norm) = (g.storage_init("q_norm", &x.q_norm), g.storage_init("k_norm", &x.k_norm));
    let (cos, sin) = (g.storage_init("cos", &x.cos), g.storage_init("sin", &x.sin));
    let w = GqaMixerWeights { q_norm: &q_norm, k_norm: &k_norm, cos: &cos, sin: &sin };
    let fids = flash.then(|| flash_ids(g));
    let (out, acts) = gqa_mixer_fwd_with(g, &ids, fids, shape, &w, &q_full, &k, &v, n, backward);
    let mut run = MixerRun { out: g.read(&out, (n * qd) as usize), d_q_full: vec![], d_k: vec![], d_v: vec![], d_q_norm: vec![], d_k_norm: vec![], flash_saved: false };
    if backward {
        let acts = acts.expect("is_train saves activations");
        run.flash_saved = matches!(acts.attn, AttnSaved::Flash { .. });
        let (d_out, gq, gk) = (g.storage_init("d_out", &x.d_out), g.storage_init("gq", &vec![0.0; hd as usize]), g.storage_init("gk", &vec![0.0; hd as usize]));
        let gw = GqaMixerGrads { q_norm: Some(&gq), k_norm: Some(&gk) };
        let (d_q_full, d_k, d_v) = gqa_mixer_bwd(g, &ids, shape, &w, &gw, &acts, &d_out, n);
        run.d_q_full = g.read(&d_q_full, (n * 2 * qd) as usize);
        run.d_k = g.read(&d_k, (n * kvd) as usize);
        run.d_v = g.read(&d_v, (n * kvd) as usize);
        run.d_q_norm = g.read(&gq, hd as usize);
        run.d_k_norm = g.read(&gk, hd as usize);
    }
    run
}

/// The mixer's flash path must be the materialised path, through the gate,
/// QK-norm, partial M-RoPE and their backward - not only the bare kernels -
/// at the real GQA ratio and head width.
#[test]
fn mixer_flash_path_matches_the_materialised_path_at_head_dim_256() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let shape = GqaMixerShape { b: 1, t: 137, n_heads: 12, n_kv_heads: 2, head_dim: 256, rotary_half: 32, rms_eps: 1e-6 };
    let x = MixerInputs::draw(&shape, &mut Lcg::new(0x00fa_57a3)); // seed: fixed so a failure reproduces
    let want = run_mixer(&g, &shape, &x, false, true);
    let got = run_mixer(&g, &shape, &x, true, true);
    assert!(got.flash_saved && !want.flash_saved, "the flash run must save the lse, the reference the probs");
    assert_close("ctx_gated", &got.out, &want.out, 1e-4);
    assert_close("d_q_full", &got.d_q_full, &want.d_q_full, 1e-3);
    assert_close("d_k", &got.d_k, &want.d_k, 1e-3);
    assert_close("d_v", &got.d_v, &want.d_v, 1e-3);
    assert_close("d_q_norm", &got.d_q_norm, &want.d_q_norm, 1e-3);
    assert_close("d_k_norm", &got.d_k_norm, &want.d_k_norm, 1e-3);
}

/// Central finite differences of `L = sum(ctx_gated * d_out)` against the
/// mixer's analytic input gradients, at pairwise-distinct dims (b 2, t 37,
/// 6 query heads over 2 kv heads, head_dim `hd`, rotary_half 17) so a swapped
/// stride or head index cannot cancel.
fn mixer_gradcheck(g: &Gpu, flash: bool, hd: u32) {
    let shape = GqaMixerShape { b: 2, t: 37, n_heads: 6, n_kv_heads: 2, head_dim: hd, rotary_half: 17, rms_eps: 1e-6 };
    let x = MixerInputs::draw(&shape, &mut Lcg::new(0x6ad0_c4ec + hd as u64)); // seed: fixed so a failure reproduces
    let an = run_mixer(g, &shape, &x, flash, true);
    assert_eq!(an.flash_saved, flash);
    let loss = |x: &MixerInputs| -> f64 { run_mixer(g, &shape, x, flash, false).out.iter().zip(&x.d_out).map(|(o, d)| f64::from(*o) * f64::from(*d)).sum() };
    let eps = 1e-2f32;
    let mut rng = Lcg::new(0x5a3b_1e00 + hd as u64); // seed: fixed so a failure reproduces
    let mut checked = 0;
    let mut check = |name: &str, len: usize, analytic: &[f32], pick: &dyn Fn(&mut MixerInputs) -> &mut Vec<f32>| {
        for _ in 0..16 {
            let i = (rng.next_u64() % len as u64) as usize;
            let (mut plus, mut minus) = (x.clone(), x.clone());
            pick(&mut plus)[i] += eps;
            pick(&mut minus)[i] -= eps;
            let fd = ((loss(&plus) - loss(&minus)) / (2.0 * f64::from(eps))) as f32;
            let a = analytic[i];
            assert!((fd - a).abs() <= 2e-2 * a.abs().max(fd.abs()) + 2e-3, "{name}[{i}] (hd {hd}, flash {flash}): finite difference {fd} vs analytic {a}");
            if a.abs() > 1e-2 {
                checked += 1;
            }
        }
    };
    check("q_full", x.q_full.len(), &an.d_q_full, &|m| &mut m.q_full);
    check("k", x.k.len(), &an.d_k, &|m| &mut m.k);
    check("v", x.v.len(), &an.d_v, &|m| &mut m.v);
    assert!(checked >= 24, "only {checked} of 48 sampled gradients were non-trivial: the check would pass on a dead backward");
}

#[test]
fn mixer_flash_path_gradcheck_wide_and_narrow_heads() {
    let g = gpu_core::testgpu::dev(KERNELS);
    for hd in [136u32, 40] {
        mixer_gradcheck(&g, true, hd);
    }
}

/// The CPU backend cannot run the trio, so the training gate must refuse it
/// there and the same mixer must still pass its gradcheck on the chain.
#[test]
fn cpu_backend_falls_back_to_the_materialised_chain_and_stays_correct() {
    let g = Gpu::new_cpu(KERNELS);
    let flash = flash_ids(&g);
    assert!(flash.for_training(&g, 256, 4096).is_none(), "the CPU backend must not be handed the barrier kernels");
    mixer_gradcheck(&g, false, 136);
}

/// The default gate: long enough and supported selects the trio, short stays
/// on the chain that doubles as the oracle.
#[test]
fn training_gate_selects_flash_only_for_long_supported_sequences() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let flash = flash_ids(&g);
    assert!(flash.for_training(&g, 256, FlashGqaIds::MIN_TRAIN_T).is_some());
    assert!(flash.for_training(&g, 256, FlashGqaIds::MIN_TRAIN_T - 1).is_none());
    assert!(flash.for_training(&g, 257, 4096).is_none(), "head_dim above the tile width must fall back");
}

// ---- real width, CUDA ---------------------------------------------------------

/// Deterministic, cheap `[-1, 1)` filler: the 32768-token inputs are 400M
/// floats, too many for the seeded generator the small tests use.
fn hashed(n: usize, salt: u32) -> Vec<f32> {
    (0..n).map(|i| ((i as u32 ^ salt).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0).collect()
}

fn real_width_inputs(shape: &GqaMixerShape) -> MixerInputs {
    let n = (shape.b * shape.t) as usize;
    let (qd, kvd, hd, half) = (shape.qd() as usize, shape.kvd() as usize, shape.head_dim as usize, shape.rotary_half as usize);
    MixerInputs {
        q_full: hashed(n * 2 * qd, 0x1111),
        k: hashed(n * kvd, 0x2222),
        v: hashed(n * kvd, 0x3333),
        q_norm: vec![1.0; hd],
        k_norm: vec![1.0; hd],
        cos: hashed(n * half, 0x4444),
        sin: hashed(n * half, 0x5555),
        d_out: hashed(n * qd, 0x6666),
    }
}

fn real_width_shape(t: u32) -> GqaMixerShape {
    GqaMixerShape { b: 1, t, n_heads: 24, n_kv_heads: 4, head_dim: 256, rotary_half: 32, rms_eps: 1e-6 }
}

/// Peak whole-card memory growth over a window, sampled from `nvidia-smi`
/// (the card is shared, so this is an upper bound on this process's own use).
struct PeakMemory {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    peak: std::sync::Arc<std::sync::atomic::AtomicU64>,
    base: u64,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn card_used_mib() -> u64 {
    let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output().expect("nvidia-smi must run for the memory check");
    String::from_utf8_lossy(&out.stdout).trim().parse().expect("nvidia-smi memory.used")
}

impl PeakMemory {
    fn start() -> Self {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        let (stop, peak) = (std::sync::Arc::new(AtomicBool::new(false)), std::sync::Arc::new(AtomicU64::new(0)));
        let base = card_used_mib();
        let (s, p) = (stop.clone(), peak.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                p.fetch_max(card_used_mib(), Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
        Self { stop, peak, base, thread: Some(thread) }
    }
    /// Peak growth in MiB over the card's use at `start`.
    fn finish(mut self) -> u64 {
        use std::sync::atomic::Ordering;
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().expect("joined once").join().expect("sampler thread");
        self.peak.load(Ordering::Relaxed).saturating_sub(self.base)
    }
}

/// Forward + backward of one mixer layer; returns milliseconds and, when
/// `read_back`, the full outputs (otherwise a 4M-element prefix, enough to see
/// non-finite values without moving gigabytes).
fn timed_layer(g: &Gpu, shape: &GqaMixerShape, x: &MixerInputs, flash: bool, read_back: bool) -> (f64, MixerRun) {
    let n = shape.b * shape.t;
    let (qd, kvd, hd) = (shape.qd(), shape.kvd(), shape.head_dim);
    let ids = mixer_ids(g);
    let (q_full, k, v) = (g.storage_init("q_full", &x.q_full), g.storage_init("k", &x.k), g.storage_init("v", &x.v));
    let (q_norm, k_norm) = (g.storage_init("q_norm", &x.q_norm), g.storage_init("k_norm", &x.k_norm));
    let (cos, sin) = (g.storage_init("cos", &x.cos), g.storage_init("sin", &x.sin));
    let d_out = g.storage_init("d_out", &x.d_out);
    let (gq, gk) = (g.storage_init("gq", &vec![0.0; hd as usize]), g.storage_init("gk", &vec![0.0; hd as usize]));
    g.poll_wait();
    let w = GqaMixerWeights { q_norm: &q_norm, k_norm: &k_norm, cos: &cos, sin: &sin };
    let gw = GqaMixerGrads { q_norm: Some(&gq), k_norm: Some(&gk) };
    let start = std::time::Instant::now();
    let (out, acts) = gqa_mixer_fwd_with(g, &ids, flash.then(|| flash_ids(g)), shape, &w, &q_full, &k, &v, n, true);
    let (d_q_full, d_k, d_v) = gqa_mixer_bwd(g, &ids, shape, &w, &gw, &acts.expect("is_train saves activations"), &d_out, n);
    g.poll_wait();
    let _ = g.read(&d_k, 1); // a read is the sync point on every backend
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let cap = |len: u32| if read_back { len as usize } else { (len as usize).min(1 << 22) };
    let run = MixerRun {
        out: g.read(&out, cap(n * qd)),
        d_q_full: g.read(&d_q_full, cap(n * 2 * qd)),
        d_k: g.read(&d_k, cap(n * kvd)),
        d_v: g.read(&d_v, cap(n * kvd)),
        d_q_norm: g.read(&gq, hd as usize),
        d_k_norm: g.read(&gk, hd as usize),
        flash_saved: flash,
    };
    (ms, run)
}

/// 27B attention width on CUDA: the flash mixer against the materialised one
/// at T = 4096 (where the chain's three 1.6 GiB slabs still fit), then flash
/// alone at 16384 (finite, memory linear in T) and 32768, printing ms per
/// forward+backward layer so a step time can be predicted.
#[test]
#[ignore = "needs a CUDA device and ~20 GiB of it; run through `gpu` with BRAIN_BACKEND=cuda"]
fn real_width_flash_mixer_on_cuda() {
    let g = Gpu::new(KERNELS);
    assert_eq!(gpu_core::backend_name(), "cuda", "this test measures the CUDA backend; set BRAIN_BACKEND=cuda");
    let flash_ids = flash_ids(&g);
    assert!(flash_ids.for_training(&g, 256, 4096).is_some(), "the CUDA device must run the trio at head_dim 256");

    // Warm every pipeline (generated-kernel compilation) outside the timings.
    let warm = real_width_shape(512);
    let xw = real_width_inputs(&warm);
    let _ = timed_layer(&g, &warm, &xw, true, false);
    let _ = timed_layer(&g, &warm, &xw, false, false);

    let shape = real_width_shape(4096);
    let x = real_width_inputs(&shape);
    let (ms_ref, want) = timed_layer(&g, &shape, &x, false, true);
    let (ms_flash, got) = timed_layer(&g, &shape, &x, true, true);
    let (ms_flash2, _) = timed_layer(&g, &shape, &x, true, false);
    if g.set_kernel_timing(true) {
        g.reset_kernel_times();
        let _ = timed_layer(&g, &shape, &x, true, false);
        let mut times = g.kernel_times().unwrap_or_default();
        times.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (name, ms, calls) in times.iter().take(8) {
            eprintln!("t=4096 flash layer device time: {name} {ms:.1} ms over {calls} calls");
        }
        g.set_kernel_timing(false);
    }
    eprintln!("t=4096 materialised {ms_ref:.1} ms, flash {ms_flash:.1} ms (second run {ms_flash2:.1} ms) per fwd+bwd layer");
    assert_close("t=4096 ctx_gated", &got.out, &want.out, 1e-3);
    assert_close("t=4096 d_q_full", &got.d_q_full, &want.d_q_full, 2e-3);
    assert_close("t=4096 d_k", &got.d_k, &want.d_k, 2e-3);
    assert_close("t=4096 d_v", &got.d_v, &want.d_v, 2e-3);
    drop((want, got, x));

    let shape = real_width_shape(16384);
    let x = real_width_inputs(&shape);
    let peak = PeakMemory::start();
    let (ms, run) = timed_layer(&g, &shape, &x, true, true);
    let peak_mib = peak.finish();
    eprintln!("t=16384 flash {ms:.1} ms per fwd+bwd layer, card memory growth {peak_mib} MiB (inputs included)");
    for (name, v) in [("out", &run.out), ("d_q_full", &run.d_q_full), ("d_k", &run.d_k), ("d_v", &run.d_v)] {
        assert!(v.iter().all(|x| x.is_finite()), "t=16384 {name}: non-finite");
        assert!(v.iter().any(|x| x.abs() > 1e-6), "t=16384 {name}: all zero");
    }
    // The chain would need three 24*16384^2*4 B = 24 GiB slabs; the trio keeps
    // O(T*head_dim) buffers, a few hundred KiB per row across the whole layer.
    assert!(peak_mib < 14 * 1024, "t=16384 used {peak_mib} MiB: memory is not O(T)");
    drop((run, x));

    let shape = real_width_shape(32768);
    let x = real_width_inputs(&shape);
    let (ms, run) = timed_layer(&g, &shape, &x, true, false);
    eprintln!("t=32768 flash {ms:.1} ms per fwd+bwd layer");
    for (name, v) in [("out", &run.out), ("d_q_full", &run.d_q_full), ("d_k", &run.d_k), ("d_v", &run.d_v)] {
        assert!(v.iter().all(|x| x.is_finite()), "t=32768 {name}: non-finite prefix");
    }
}
