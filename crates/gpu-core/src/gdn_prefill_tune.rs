// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Measured selection of the Gated DeltaNet serving prefill: the exact form (three-pass TF32 products,
//! `gdn_chunk_prep_tok_tc` + `gdn_chunk_loop_tc`) against the bf16 form (bf16 operands with fp32 accumulation and an fp32
//! state, `gdn_chunk_prep_tok_bf16` + `gdn_chunk_loop_bf16`).
//!
//! Swedish Embedded AB implements self-tuning inference kernels that reach the tensor-core roofline of whatever
//! accelerator they land on. If your team needs expertise in choosing numeric formats for the recurrent layers of a
//! language model by measurement instead of by hand, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Like [`crate::native_tune`] this decides per device by measurement and never by device name: on first use of a device
//! (before the first recorded step, see [`crate::Gpu::prepare_gdn_prefill`]) a synthetic prefill of production width is
//! run through both forms. The bf16 form is admitted only when its output stays within [`ADMIT_ULPS`] bf16 units of the
//! exact form's (the rounding of its own operands is what it is allowed to cost, a layout or indexing defect shows as an
//! error orders of magnitude larger), and it is chosen when it is faster by more than [`TIE`]. A device without both
//! forms, or whose timers are unavailable, keeps the exact form. [`crate::force_candidate`] pins the slot
//! [`SLOT`] to `exact` or `bf16` for A/B measurement and for the tests that need the exact arithmetic.

use crate::provider::cuda::{self, GdnTokenLayout};
use crate::Gpu;

/// The slot name [`crate::force_candidate`] takes.
pub const SLOT: &str = "gdn_prefill";

/// The bf16 form must be this much faster than the exact one to displace it.
const TIE: f64 = 0.97;

/// The admitted error of the bf16 form against the exact one: relative rms of the recurrence output, in units of the
/// bf16 rounding step 2^-9. Measured on this synthetic problem: about 2.1 (4.0e-3); an operand layout defect gives O(1/2^-9).
const ADMIT_ULPS: f64 = 4.0;
const BF16_UNIT: f64 = 1.0 / 512.0;

/// Heads, key-head group, tokens and chunk length of the synthetic prefill: the production head count (so the
/// recurrence fills the device as it does in serving) over a quarter of an 8K round.
const HEADS: u32 = 48;
const GROUP: u32 = 3;
const TOKENS: u32 = 2048;
const CHUNK: u32 = 64;
const DIM: u32 = 128;

/// What the measurement decided, kept per handle.
#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub bf16: bool,
    /// Microseconds of one forward of each form, when both were timed.
    pub exact_us: Option<f64>,
    pub bf16_us: Option<f64>,
    /// Relative rms error of the bf16 form's output against the exact form's.
    pub rel_err: Option<f64>,
}

impl Choice {
    const EXACT: Choice = Choice { bf16: false, exact_us: None, bf16_us: None, rel_err: None };
}

fn lcg(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((*seed >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// The choices this process has measured, by device (adapter, compute capability): several handles on one card share
/// one measurement.
static MEASURED: std::sync::Mutex<Vec<(String, Choice)>> = std::sync::Mutex::new(Vec::new());

/// The choice for `gpu`'s device: measured once per process, on a throwaway handle of the same device so the synthetic
/// problem's buffers are never charged to (or held by) the model's own handle.
pub(crate) fn choose(gpu: &Gpu) -> Choice {
    let cc = gpu.caps().arch.compute_capability;
    if !cc.is_some_and(|cc| cc >= kernels_cuda::MMA_S8_MIN_CC) || kernels_cuda::get("gdn_chunk_loop_bf16").is_none() {
        return Choice::EXACT;
    }
    let key = format!("{}-{cc:?}-{}sm", crate::adapter_info().map(|(d, _)| d).unwrap_or_else(|| gpu.kind().to_owned()), gpu.caps().compute_units.unwrap_or(0));
    let mut measured = MEASURED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, c)) = measured.iter().find(|(k, _)| *k == key) {
        return *c;
    }
    let probe = gpu.new_like(&[]);
    let choice = try_measure(&probe);
    // Timing is off in serving (per-launch events cost real time); the tuner is the only thing that turns it on.
    probe.set_kernel_timing(false);
    let choice = choice.unwrap_or(Choice::EXACT);
    eprintln!(
        "brain: gdn prefill form on {key}: {} (exact {} us, bf16 {} us, bf16 error {})",
        if choice.bf16 { "bf16" } else { "exact" },
        choice.exact_us.map_or("-".into(), |v| format!("{v:.0}")),
        choice.bf16_us.map_or("-".into(), |v| format!("{v:.0}")),
        choice.rel_err.map_or("-".into(), |v| format!("{v:.1e}"))
    );
    measured.push((key, choice));
    choice
}

fn try_measure(gpu: &Gpu) -> Option<Choice> {
    let (h, t, c, group) = (HEADS, TOKENS, CHUNK, GROUP);
    let n_chunks = t / c;
    let (bh, bhc) = (h, h * n_chunks);
    let (hs, ts, cs) = (h as usize, t as usize, c as usize);
    let scale = 1.0 / (DIM as f32).sqrt();
    let layout = GdnTokenLayout { bhc, c_len: c, n_chunks, heads: h, group, head_dim: DIM, scale, eps: 1e-6 };

    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let nkh = hs / group as usize;
    let q = gpu.storage_init("gdn_tune_q", &lcg(&mut s, ts * nkh * 128, 1.0));
    let k = gpu.storage_init("gdn_tune_k", &lcg(&mut s, ts * nkh * 128, 1.0));
    let v = gpu.storage_init("gdn_tune_v", &lcg(&mut s, ts * hs * 128, 1.0));
    let raw_g = gpu.storage_init("gdn_tune_g", &lcg(&mut s, ts * hs, 0.2).iter().map(|x| -x.abs()).collect::<Vec<_>>());
    let beta = gpu.storage_init("gdn_tune_beta", &lcg(&mut s, ts * hs, 0.5).iter().map(|x| x + 0.5).collect::<Vec<_>>());
    let initial = gpu.storage_init("gdn_tune_s0", &lcg(&mut s, bh as usize * 128 * 128, 0.1));
    let len = bhc as usize * cs * 128;
    let (g_cs, exp_g_cs) = (gpu.storage(u64::from(bhc * c)), gpu.storage(u64::from(bhc * c)));
    let (u, w, intra) = (gpu.storage(len as u64), gpu.storage(len as u64), gpu.storage(u64::from(bhc * c * c)));
    let (qcm, kcm) = (gpu.storage(len as u64), gpu.storage(len as u64));
    let out = gpu.storage(len as u64);
    let fin = gpu.storage(bh as u64 * 128 * 128);

    let prep_bufs = [&q, &k, &v, &raw_g, &beta, &g_cs, &exp_g_cs, &u, &w, &intra, &qcm, &kcm];
    let loop_bufs = [&kcm, &qcm, &w, &u, &g_cs, &exp_g_cs, &intra, &initial, &out, &fin];
    let forms = [
        (cuda::gdn_chunk_prep_tok_step(gpu, &prep_bufs, layout), cuda::gdn_chunk_loop_tc_step(gpu, &loop_bufs, bh, c, DIM, DIM, n_chunks, scale, h)),
        (cuda::gdn_chunk_prep_tok_bf16_step(gpu, &prep_bufs, layout), cuda::gdn_chunk_loop_bf16_step(gpu, &loop_bufs, bh, c, DIM, DIM, n_chunks, scale, h)),
    ];
    gpu.set_kernel_timing(true);
    let mut outs: Vec<Vec<f32>> = Vec::new();
    let mut times: Vec<f64> = Vec::new();
    for (prep, recurrence) in forms {
        let steps = [prep?, recurrence?];
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        outs.push(gpu.read(&out, len));
        // The best of a few device-timed forwards (the launches' own event times, not host wall clock).
        let mut best = f64::MAX;
        for _ in 0..4 {
            gpu.reset_kernel_times();
            gpu.submit(&[], &steps);
            gpu.poll_wait();
            let ms: f64 = gpu.kernel_times()?.iter().filter(|r| r.0.starts_with("native:gdn_chunk_")).map(|r| r.1).sum();
            if ms > 0.0 {
                best = best.min(ms * 1e3);
            }
        }
        times.push(best);
    }
    let rms = |a: &[f32]| (a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / a.len() as f64).sqrt();
    let diff: Vec<f32> = outs[1].iter().zip(&outs[0]).map(|(a, b)| a - b).collect();
    let rel_err = rms(&diff) / rms(&outs[0]).max(1e-30);
    let (exact_us, bf16_us) = (times[0], times[1]);
    let timed = exact_us.is_finite() && bf16_us.is_finite() && exact_us < f64::MAX && bf16_us < f64::MAX;
    let admitted = rel_err.is_finite() && rel_err <= ADMIT_ULPS * BF16_UNIT;
    Some(Choice { bf16: timed && admitted && bf16_us < exact_us * TIE, exact_us: timed.then_some(exact_us), bf16_us: timed.then_some(bf16_us), rel_err: Some(rel_err) })
}
