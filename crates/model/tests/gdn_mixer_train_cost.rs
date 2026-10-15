// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What one Gated DeltaNet layer costs to train at the real Qwen3.5-27B width
//! (48 value heads, 16 key heads, key/value head dim 128, conv kernel 4): peak
//! live device bytes, kernel launches and wall time of one forward+backward,
//! one-shot and segmented. This is the measurement that decides how long a
//! sequence a card can train, so it reports the numbers instead of asserting
//! them; the cases are `#[ignore]` (they need a CUDA card and take minutes).
//!
//! Run one with the lock the shared card requires, e.g.
//!   BRAIN_BACKEND=cuda gpu cargo test -p brain-model --test gdn_mixer_train_cost \
//!       -- --ignored --nocapture one_shot_2048
//!
//! The peak is the maximum, over a sampler thread polling the CUDA backend's
//! exact outstanding-allocation counter, of device + pool bytes minus the
//! baseline taken before the layer's inputs were uploaded.
//!
//! Swedish Embedded AB implements solutions for training long-context
//! linear-attention models on a single accelerator for its clients. If your
//! team needs expertise in memory-bounded sequence-model training you can
//! procure our services by sending an email to info@swedishembedded.com.

mod gdn_kit;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use gdn_kit::{mixer_ids, DeviceWeights, Host, KERNELS};
use gpu_core::Gpu;
use model::gdn::GdnShape;
use model::gdn_mixer::{gdn_mixer_bwd, gdn_mixer_fwd, GdnMixerShape};
use model::gdn_segment::{gdn_mixer_seg_bwd, gdn_mixer_seg_fwd, GdnMixerInputs};

fn live_bytes() -> u64 {
    let r = backend_cuda::live::live_resources();
    r.device_bytes + r.pool_bytes + r.managed_bytes
}

/// Polls `live_bytes` until dropped; `peak()` is the maximum seen.
struct PeakSampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl PeakSampler {
    fn start() -> PeakSampler {
        let (stop, peak) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)));
        let (s, p) = (stop.clone(), peak.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                p.fetch_max(live_bytes(), Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        });
        PeakSampler { stop, peak, thread: Some(thread) }
    }
    fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }
}

impl Drop for PeakSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn real_shape(t: u32) -> GdnMixerShape {
    GdnMixerShape { gdn: GdnShape { b: 1, h: 48, t, dk: 128, dv: 128, chunk: 64 }, nkh: 16, conv_kernel: 4, rms_eps: 1e-6 }
}

fn mib(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

/// `segment_tokens == None` runs the one-shot pair, `Some(s)` the segmented one.
fn cost(t: u32, segment_tokens: Option<u32>) {
    assert_eq!(gpu_core::backend_name(), "cuda", "this measurement is CUDA-only (run with BRAIN_BACKEND=cuda)");
    let g = Gpu::new(KERNELS);
    let shape = real_shape(t);
    // seed: synthetic input that defines the measured workload
    let host = Host::random(&shape, 7);
    let ids = mixer_ids(&g);
    let dw = DeviceWeights::upload(&g, &shape, &host);
    let mixed = g.storage_init("mixed_qkv", &host.mixed_qkv);
    let (b, a, z) = (g.storage_init("bproj", &host.bproj), g.storage_init("aproj", &host.aproj), g.storage_init("z", &host.z));
    let d_gated = g.storage_init("d_gated", &host.d_gated);
    let inputs = GdnMixerInputs { mixed_qkv: &mixed, bproj: &b, aproj: &a, z: &z };
    let _ = g.read(&z, 1);
    let baseline = live_bytes();

    let sampler = PeakSampler::start();
    let launches0 = gpu_core::cuda_call_totals().host_launches;
    let t0 = Instant::now();
    let (fwd_peak, fwd_secs, fwd_launches);
    let outputs = match segment_tokens {
        None => {
            let (gated, acts) = gdn_mixer_fwd(&g, &ids, &shape, &dw.weights(), &mixed, &b, &a, &z, t, true);
            let _ = g.read(&gated, 1);
            (fwd_peak, fwd_secs, fwd_launches) = (sampler.peak(), t0.elapsed().as_secs_f64(), gpu_core::cuda_call_totals().host_launches - launches0);
            let acts = acts.expect("acts");
            gdn_mixer_bwd(&g, &ids, &shape, &dw.weights(), &dw.grads(), &acts, &d_gated, t)
        }
        Some(s) => {
            let (gated, bounds) = gdn_mixer_seg_fwd(&g, &ids, &shape, &dw.weights(), &inputs, s);
            let _ = g.read(&gated, 1);
            (fwd_peak, fwd_secs, fwd_launches) = (sampler.peak(), t0.elapsed().as_secs_f64(), gpu_core::cuda_call_totals().host_launches - launches0);
            gdn_mixer_seg_bwd(&g, &ids, &shape, &dw.weights(), &dw.grads(), &inputs, &bounds, &d_gated)
        }
    };
    let _ = g.read(&outputs.3, 1);
    let total_secs = t0.elapsed().as_secs_f64();
    let launches = gpu_core::cuda_call_totals().host_launches - launches0;
    let peak = sampler.peak();
    println!(
        "GDN-COST t={t} segment={segment_tokens:?} fwd_peak_mib={:.0} peak_mib={:.0} peak_kib_per_token={:.1} fwd_launches={fwd_launches} launches={launches} fwd_secs={fwd_secs:.2} secs={total_secs:.2}",
        mib(fwd_peak.saturating_sub(baseline)),
        mib(peak.saturating_sub(baseline)),
        peak.saturating_sub(baseline) as f64 / 1024.0 / t as f64,
    );
}

#[test]
#[ignore = "needs a CUDA card; run via BRAIN_BACKEND=cuda gpu cargo test ... -- --ignored"]
fn one_shot_2048() {
    cost(2048, None);
}

#[test]
#[ignore = "needs a CUDA card; run via BRAIN_BACKEND=cuda gpu cargo test ... -- --ignored"]
fn one_shot_8192() {
    cost(8192, None);
}

#[test]
#[ignore = "needs a CUDA card and ~45 GiB; run via BRAIN_BACKEND=cuda gpu cargo test ... -- --ignored"]
fn one_shot_32768() {
    cost(32768, None);
}

#[test]
#[ignore = "needs a CUDA card; run via BRAIN_BACKEND=cuda gpu cargo test ... -- --ignored"]
fn segmented_8192_by_2048() {
    cost(8192, Some(2048));
}

#[test]
#[ignore = "needs a CUDA card; run via BRAIN_BACKEND=cuda gpu cargo test ... -- --ignored"]
fn segmented_32768_by_2048() {
    cost(32768, Some(2048));
}

#[test]
#[ignore = "needs a CUDA card; run via BRAIN_BACKEND=cuda gpu cargo test ... -- --ignored"]
fn segmented_131072_by_2048() {
    cost(131072, Some(2048));
}
