// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grouped expert GEMM's gate data, shared by the gate of the selected kernel (`it/moe_i8_grouped_native.rs`) and
//! the gate of every candidate (`moe_i8_grouped_candidates.rs`): a routed round on the device, the WGSL tier's output
//! for it, and the f64 value of sampled outputs to hold a candidate that reassociates its sum to.
//!
//! Swedish Embedded AB implements solutions for verifying fast quantised kernels against their exact definition for
//! its clients. If your team needs expertise in sparse mixture-of-experts inference then you can procure our services
//! by sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use gpu_core::{Dispatch, Gpu};

pub const KERNELS: &[(&str, &str)] = &[
    ("moe_i8_grouped", kernels::MOE_I8_GROUPED),
    ("moe_i8_grouped_ref", kernels::MOE_I8_GROUPED),
    ("moe_route_count", kernels::MOE_ROUTE_COUNT),
    ("moe_route_scan", kernels::MOE_ROUTE_SCAN),
    ("moe_route_emit", kernels::MOE_ROUTE_EMIT),
];
pub const K_GROUPED: usize = 0;
pub const K_REF: usize = 1;
const K_COUNT: usize = 2;
const K_SCAN: usize = 3;
const K_EMIT: usize = 4;

pub const MR: u32 = 8;
const WGSL_COLS: u32 = 4;
/// Written around every output so an out-of-place write shows.
const SENTINEL: u32 = 0x7fc0_dead;
/// Unit roundoff of the f32 sum both tiers accumulate in.
const F32_EPS: f64 = 1.0 / 16_777_216.0;
/// Sampled outputs a tolerance check evaluates in f64.
const SAMPLES: usize = 3000;

pub struct Case {
    pub slots: u32,
    pub k: u32,
    pub n: u32,
    pub xdiv: u32,
    pub ne: u32,
    /// Percentage of slots sent to expert 0 (a hot expert, many tiles; the rest thin or empty).
    pub skew: u32,
}

impl Case {
    /// The dispatch's params, `tiles` sized as the host sizes it.
    pub fn params(&self) -> [u32; 5] {
        [self.ne + self.slots.div_ceil(MR), self.k / 4, self.n, self.xdiv, self.ne]
    }
}

fn packed(words: usize, seed: u64) -> Vec<u32> {
    let mut r = Lcg::new(seed);
    (0..words).map(|_| (0..4).fold(0u32, |w, lane| w | u32::from((r.next_u32() % 256) as u8) << (8 * lane))).collect()
}

fn scales(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| 1e-3 + (r.next_u32() % 1000) as f32 * 1e-5 - if r.next_u32().is_multiple_of(7) { 2e-3 } else { 0.0 }).collect()
}

fn upload_u32(gpu: &Gpu, v: &[u32]) -> gpu_core::DeviceBuffer {
    let b = gpu.storage(v.len().max(1) as u64);
    gpu.write(&b, v);
    b
}

fn code(words: &[u32], at: usize) -> i64 {
    i64::from((words[at / 4] >> (8 * (at % 4))) as u8 as i8)
}

/// One routed round: the outputs of the dispatched (selected or forced) kernel and of the WGSL tier, and what an exact
/// evaluation of an output needs.
pub struct Round {
    pub got: Vec<u32>,
    pub want: Vec<u32>,
    ids: Vec<u32>,
    xq: Vec<u32>,
    sx: Vec<f32>,
    wq: Vec<u32>,
    sw: Vec<f32>,
    kg: usize,
    n: usize,
    xdiv: usize,
}

impl Round {
    /// Output `(slot, col)` in f64 from the stored codes and scales, and the sum of its terms' magnitudes: the
    /// definition both tiers compute, `sum_g dot_g * (sw * sx)`.
    fn exact(&self, slot: usize, col: usize) -> (f64, f64) {
        let (kg, ng) = (self.kg, self.kg / 8);
        let (row, xr) = (self.ids[slot] as usize * self.n + col, slot / self.xdiv);
        let (mut total, mut abs) = (0.0f64, 0.0f64);
        for g in 0..ng {
            let dot: i64 = (0..32).map(|i| code(&self.xq, xr * kg * 4 + g * 32 + i) * code(&self.wq, row * kg * 4 + g * 32 + i)).sum();
            let term = dot as f64 * f64::from(self.sw[row * ng + g]) * f64::from(self.sx[xr * ng + g]);
            total += term;
            abs += term.abs();
        }
        (total, abs)
    }

    /// The tolerance gate of a candidate that reassociates the per-group sum (kernels.md F item 10): over sampled
    /// outputs, its error against the f64 definition is at most twice the WGSL tier's own error on the same data, both
    /// in root mean square and at the worst output, never held under the f32 accumulation floor (8 roundoffs of the
    /// terms' magnitude). `Err` says by how much it is not.
    pub fn within_tolerance(&self, seed: u64) -> Result<(f64, f64), String> {
        let total = self.got.len();
        let mut r = Lcg::new(seed);
        let (mut got_sq, mut want_sq, mut got_max, mut want_max, mut floor) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let samples = SAMPLES.min(total);
        for i in 0..samples {
            let at = if samples == total { i } else { r.next_u32() as usize % total };
            let (slot, col) = (at / self.n, at % self.n);
            let (truth, abs) = self.exact(slot, col);
            let (g, w) = ((f64::from(f32::from_bits(self.got[at])) - truth).abs(), (f64::from(f32::from_bits(self.want[at])) - truth).abs());
            if !g.is_finite() {
                return Err(format!("output ({slot}, {col}) is {}", f32::from_bits(self.got[at])));
            }
            got_sq += g * g;
            want_sq += w * w;
            got_max = got_max.max(g);
            want_max = want_max.max(w);
            floor = floor.max(8.0 * F32_EPS * abs);
        }
        let (got_rms, want_rms) = ((got_sq / samples as f64).sqrt(), (want_sq / samples as f64).sqrt());
        if got_rms > (2.0 * want_rms).max(floor) || got_max > (2.0 * want_max).max(floor) {
            return Err(format!("error rms {got_rms:.3e} max {got_max:.3e} against the WGSL tier's rms {want_rms:.3e} max {want_max:.3e} (floor {floor:.3e})"));
        }
        Ok((got_rms / want_rms.max(f64::MIN_POSITIVE), got_max / want_max.max(f64::MIN_POSITIVE)))
    }
}

/// Route `c` on the device, then run the dispatched kernel and the WGSL tier on the same inputs.
pub fn run(gpu: &Gpu, c: &Case, seed: u64) -> Round {
    let (slots, kg, n, xdiv, ne) = (c.slots as usize, (c.k / 4) as usize, c.n as usize, c.xdiv as usize, c.ne as usize);
    let rows = slots.div_ceil(xdiv);
    let mut rng = Lcg::new(seed);
    let ids: Vec<u32> = (0..slots).map(|_| if rng.next_u32() % 100 < c.skew { 0 } else { rng.next_u32() % c.ne }).collect();
    let (xqv, sxv, wqv, swv) = (packed(rows * kg, seed + 1), scales(rows * (kg / 8), seed + 2), packed(ne * n * kg, seed + 3), scales(ne * n * (kg / 8), seed + 4));
    let xq = upload_u32(gpu, &xqv);
    let sx = gpu.storage_init("sx", &sxv);
    let wq = upload_u32(gpu, &wqv);
    let sw = gpu.storage_init("sw", &swv);
    let ids_b = upload_u32(gpu, &ids);
    let counts = gpu.storage(ne as u64);
    let tab = gpu.storage(2 * (ne as u64 + 1));
    let perm = gpu.storage(slots.max(1) as u64);
    gpu.submit(
        &[],
        &[
            gpu.dispatch(K_COUNT, &[&ids_b, &counts], &[c.slots, c.ne], Dispatch::Workgroups(c.ne)),
            gpu.dispatch(K_SCAN, &[&counts, &tab], &[c.ne, MR], Dispatch::Workgroups(1)),
            gpu.dispatch(K_EMIT, &[&ids_b, &tab, &perm], &[c.slots, c.ne], Dispatch::Workgroups(c.ne)),
        ],
    );
    let params = c.params();
    let go = |kind: usize| -> Vec<u32> {
        let out = gpu.storage((slots * n) as u64);
        gpu.write(&out, &vec![SENTINEL; slots * n]);
        gpu.submit(&[], &[gpu.dispatch(kind, &[&xq, &sx, &tab, &perm, &wq, &sw, &out], &params, Dispatch::Workgroups(params[0] * c.n.div_ceil(WGSL_COLS)))]);
        gpu.poll_wait();
        gpu.read(&out, slots * n).iter().map(|f| f.to_bits()).collect()
    };
    let (got, want) = (go(K_GROUPED), go(K_REF));
    Round { got, want, ids, xq: xqv, sx: sxv, wq: wqv, sw: swv, kg, n, xdiv }
}
