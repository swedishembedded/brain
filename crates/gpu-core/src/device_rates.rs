// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What one card does, measured with brain's best kernels and checked before
//! it is timed: the device rows of `brain perf roofline`.
//!
//! Swedish Embedded AB implements accelerator bring-up and validation for its
//! clients. If your team needs to know what a new card reaches with its own
//! kernels, against what the vendor sells it as, you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! Three kinds of row, each with the vendor's published peak beside it where
//! the device is listed (`crate::published_peak`), never as a denominator:
//!
//! * **the silicon's probes** (`crate::roof`, and the tuner's int8 `mma.sync`
//!   probe): register-resident chains that measure an issue rate, not a kernel;
//! * **GEMMs** on brain's best kernels at a large square shape
//!   (`native_tune::square`), every precision the device has a kernel for, each
//!   admitted against an f64 reference first;
//! * **memory and links**: the device memory read, copied and triad-streamed by
//!   native kernels over buffers far larger than any cache, and the host link
//!   kept apart from it - an upload and a readback (driver copies, host-timed)
//!   and the GPU reading host memory directly across the link (a kernel over a
//!   host-resident buffer on the CPU module next to the card). Each stream's
//!   output is checked (a checksum of the read, the copied and triad words
//!   read back) before its time is believed.
//!
//! A row that cannot be measured says why. Kernel rows are timed launch by
//! launch ([`crate::launch_stats`]) and flagged when another context stretched
//! them; probe and stream rows take the fastest launch, a ceiling.

use crate::launch_stats;
use crate::published_peak::Metric;
use crate::{DeviceBuffer, Gpu};
use std::time::Instant;

/// One measured quantity of one card.
#[derive(Clone, Debug)]
pub struct DeviceRate {
    /// `fp32`, `int8-tc`, `hbm-copy`, `c2c-read`, ...
    pub metric: &'static str,
    /// `TFLOP/s`, `TOPS` or `GB/s`.
    pub unit: &'static str,
    pub value: Option<f64>,
    /// Which published figure the value compares with.
    pub peak: Option<Metric>,
    /// The kernel or probe, its shape, the other kernels it beat.
    pub method: String,
    /// How its output was checked, where it has one.
    pub verified: Option<String>,
    /// Whether another context stretched its launches.
    pub noisy: bool,
    /// Why it was not measured.
    pub why_not: Option<String>,
}

impl DeviceRate {
    fn measured(metric: &'static str, unit: &'static str, value: f64, peak: Option<Metric>, method: String) -> DeviceRate {
        DeviceRate { metric, unit, value: Some(value), peak, method, verified: None, noisy: false, why_not: None }
    }

    fn absent(metric: &'static str, unit: &'static str, peak: Option<Metric>, why: String) -> DeviceRate {
        DeviceRate { metric, unit, value: None, peak, method: String::new(), verified: None, noisy: false, why_not: Some(why) }
    }

    fn verified(mut self, how: String) -> DeviceRate {
        self.verified = Some(how);
        self
    }
}

/// Bytes of each stream buffer: far past any device cache.
const STREAM_BYTES: u64 = 1 << 30;
/// Words of the copied and triad buffers read back to check them (and to time the readback).
const READBACK_WORDS: usize = 64 << 20;
/// Kernel launches each stream is timed over, one by one; the fastest is the ceiling.
const STREAM_LAUNCHES: usize = 12;
/// Host copies the upload and readback are timed over; the fastest is kept.
const COPY_REPEATS: usize = 3;

/// The stream kernels' bindings: params, the read buffers, the written one.
const COPY_BINDINGS: &[backend_api::BindKind] = &[backend_api::BindKind::Uniform, backend_api::BindKind::StorageRead, backend_api::BindKind::StorageReadWrite];
const TRIAD_BINDINGS: &[backend_api::BindKind] =
    &[backend_api::BindKind::Uniform, backend_api::BindKind::StorageRead, backend_api::BindKind::StorageRead, backend_api::BindKind::StorageReadWrite];

/// STREAM copy over 16-byte vectors, grid-stride. Params `[vectors]`.
static STREAM_COPY: kernels_cuda::CudaKernel = kernels_cuda::CudaKernel {
    name: "stream_copy",
    op: backend_api::select::Op::MatMul,
    weight: backend_api::select::Dtype::F32,
    by_name: true,
    source: backend_api::ImplSource::Tuned,
    min_cc: kernels_cuda::BASELINE_MIN_CC,
    entry: "brain_stream_copy",
    what: "device memory copy probe: grid-stride 16-byte loads and stores",
    reported: "native:stream_copy",
    block_dim: 256,
    tile: (1, 256),
    shared_bytes: 0,
    programmatic: false,
    src: r#"
extern "C" __global__ void __launch_bounds__(256)
brain_stream_copy(const unsigned int* params, const uint4* __restrict__ src, uint4* __restrict__ dst) {
    const unsigned long long n = params[0];
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}
"#,
};

/// The triad kernel's source, part of the key `crate::roof` persists its bandwidth under.
pub(crate) const STREAM_TRIAD_SOURCE: &str = STREAM_TRIAD.src;

/// STREAM triad `a = b + s * c` over 16-byte vectors of fp32, grid-stride. Params `[vectors, s as f32 bits]`.
static STREAM_TRIAD: kernels_cuda::CudaKernel = kernels_cuda::CudaKernel {
    name: "stream_triad",
    op: backend_api::select::Op::MatMul,
    weight: backend_api::select::Dtype::F32,
    by_name: true,
    source: backend_api::ImplSource::Tuned,
    min_cc: kernels_cuda::BASELINE_MIN_CC,
    entry: "brain_stream_triad",
    what: "device memory triad probe: a = b + s * c over grid-stride 16-byte vectors",
    reported: "native:stream_triad",
    block_dim: 256,
    tile: (1, 256),
    shared_bytes: 0,
    programmatic: false,
    src: r#"
extern "C" __global__ void __launch_bounds__(256)
brain_stream_triad(const unsigned int* params, const float4* __restrict__ b, const float4* __restrict__ c, float4* __restrict__ a) {
    const unsigned long long n = params[0];
    const float s = __uint_as_float(params[1]);
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        const float4 x = b[i], y = c[i];
        a[i] = make_float4(x.x + s * y.x, x.y + s * y.y, x.z + s * y.z, x.w + s * y.w);
    }
}
"#,
};

/// The stream pattern's word `i`: a small integer as fp32, so the triad's `b + 0.5 * c` is exact whatever the
/// device's contraction, and the read probe's checksum (the XOR of every word) is known on the host.
fn pattern(i: usize) -> u32 {
    ((i % 8191) as f32).to_bits()
}

/// Every row of card `dev`: the probes (re-measured when `reprofile`), the square GEMMs, the memory and the links.
/// A few tens of seconds on a large card, most of it the host building the GEMMs' reference inputs.
pub fn measure(dev: &crate::devices::DeviceId, reprofile: bool) -> Vec<DeviceRate> {
    let gpu = Gpu::new_on(dev, &[]);
    let mut rows = probe_rows(&gpu, reprofile);
    let (gemms, missing) = crate::native_tune::square_gemms(dev);
    for metric in ["fp32-gemm", "tf32-tc", "bf16-tc", "fp8-tc", "int8-tc"] {
        let (unit, peak) = gemm_unit(metric);
        rows.push(match gemms.iter().find(|g| g.metric == metric) {
            Some(g) => {
                let side = crate::native_tune::SQUARE_SIDE;
                let others = if g.others.is_empty() { String::new() } else { format!("; beat {}", g.others.join(", ")) };
                let mut r = DeviceRate::measured(metric, unit, g.tera, Some(peak), format!("{} at {side}x{side}x{side}, median of {} launches{others}", g.kernel, g.stats.samples));
                r.noisy = g.stats.contaminated();
                r.verified(g.verified.clone())
            }
            None => DeviceRate::absent(metric, unit, Some(peak), missing.iter().find(|(m, _)| *m == metric).map(|(_, w)| w.clone()).unwrap_or_else(|| "not measured".into())),
        });
    }
    rows.extend(stream_rows(&gpu, dev));
    rows
}

fn gemm_unit(metric: &str) -> (&'static str, Metric) {
    match metric {
        "fp32-gemm" => ("TFLOP/s", Metric::Fp32),
        "tf32-tc" => ("TFLOP/s", Metric::Tf32),
        "bf16-tc" => ("TFLOP/s", Metric::Bf16),
        "fp8-tc" => ("TFLOP/s", Metric::Fp8),
        _ => ("TOPS", Metric::Int8),
    }
}

/// The silicon's probes: fp32 FMA, packed-int8 dot, f16 FMA (`crate::roof`) and the int8 `mma.sync` chain.
fn probe_rows(gpu: &Gpu, reprofile: bool) -> Vec<DeviceRate> {
    let roofs = if reprofile { crate::roof::reprofile(gpu) } else { crate::roof::ensure(gpu) };
    let mut rows = Vec::new();
    match roofs {
        Some(r) => {
            rows.push(DeviceRate::measured("fp32", "TFLOP/s", f64::from(r.gflops) / 1e3, Some(Metric::Fp32), "FMA chain probe (roof_fma), register-resident".into()));
            rows.push(match r.int8_gops {
                Some(v) => DeviceRate::measured("int8-dp4a", "TOPS", f64::from(v) / 1e3, None, "packed-int8 dot probe (roof_dp4a) on the CUDA cores".into()),
                None => DeviceRate::absent("int8-dp4a", "TOPS", None, "the device has no packed-int8 dot-product path".into()),
            });
            rows.push(DeviceRate::measured("mem-roof", "GB/s", f64::from(r.gbs), Some(Metric::Memory), "the profiler's memory roof (`crate::roof`): the native triad over three 256 MiB buffers, device-timed, where the backend runs native kernels, the portable axpy elsewhere".into()));
            rows.push(match r.f16_gflops {
                Some(v) => DeviceRate::measured("f16", "TFLOP/s", f64::from(v) / 1e3, None, "f16 FMA chain probe".into()),
                None => DeviceRate::absent(
                    "f16",
                    "TFLOP/s",
                    None,
                    format!("the portable f16 FMA probe does not execute on the {} backend (f16 tensor-core work is the bf16-tc row)", gpu.kind()),
                ),
            });
        }
        None => rows.push(DeviceRate::absent("fp32", "TFLOP/s", Some(Metric::Fp32), "the roofline probe is off or the backend cannot be probed".into())),
    }
    rows.push(match gpu.set_kernel_timing(true).then(|| crate::native_tune::measure_int8_roof(gpu)).flatten() {
        Some(t) => DeviceRate::measured("int8-mma", "TOPS", f64::from(t), Some(Metric::Int8), "int8 mma.sync m16n8k32 probe on register operands".into()),
        None => DeviceRate::absent("int8-mma", "TOPS", Some(Metric::Int8), "no int8 mma.sync probe compiles or times on this device".into()),
    });
    gpu.set_kernel_timing(false);
    rows
}

/// The device memory (read, copy, triad) and the host link (upload, readback, the GPU reading host memory).
fn stream_rows(gpu: &Gpu, dev: &crate::devices::DeviceId) -> Vec<DeviceRate> {
    const MEM: Option<Metric> = Some(Metric::Memory);
    const LINK: Option<Metric> = Some(Metric::HostLink);
    let words = (STREAM_BYTES / 4) as usize;
    if !gpu.set_kernel_timing(true) {
        let why = "the backend cannot time kernels on this device".to_owned();
        return ["hbm-read", "hbm-copy", "hbm-triad", "h2d", "d2h", "c2c-read"].map(|m| DeviceRate::absent(m, "GB/s", MEM, why.clone())).to_vec();
    }
    let host: Vec<u32> = (0..words).map(pattern).collect();
    let checksum = host.iter().fold(0u32, |a, w| a ^ w);
    let src = gpu.storage(words as u64);
    let mut rows = Vec::new();

    // Upload: the driver's copy of a pageable host vector, host-timed around a drained device.
    let up = (0..COPY_REPEATS)
        .map(|_| {
            let t = Instant::now();
            gpu.write(&src, &host);
            gpu.poll_wait();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min);
    let upload = DeviceRate::measured("h2d", "GB/s", STREAM_BYTES as f64 / up / 1e9, LINK, format!("upload of {} MiB from pageable host memory, host-timed, fastest of {COPY_REPEATS}", STREAM_BYTES >> 20));

    rows.push(read_row(gpu, "hbm-read", MEM, &src, checksum, "device memory"));
    let dst = gpu.storage(words as u64);
    let copy = gpu.native_kernel(&STREAM_COPY, COPY_BINDINGS);
    rows.push(match copy {
        Some(id) => {
            let vectors = (STREAM_BYTES / 16) as u32;
            let steps: Vec<_> = (0..STREAM_LAUNCHES).filter_map(|_| gpu.native_step_raw(id, &[&src, &dst], &[vectors], crate::native_tune::ROOF_BLOCKS)).collect();
            stream_row(gpu, "hbm-copy", &steps, 2.0 * STREAM_BYTES as f64, "copy kernel, grid-stride 16-byte vectors, 1 GiB to 1 GiB", &dst, &|i| host[i])
        }
        None => DeviceRate::absent("hbm-copy", "GB/s", MEM, "the stream copy kernel does not compile on this backend".into()),
    });

    // Readback: the driver's copy back into host memory, which also checks the copy.
    let down = (0..COPY_REPEATS)
        .map(|_| {
            let t = Instant::now();
            let got = gpu.read(&dst, READBACK_WORDS);
            std::hint::black_box(&got);
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min);
    rows.push(upload.verified("the copy of the upload reads back word for word".into()));
    rows.push(DeviceRate::measured("d2h", "GB/s", (READBACK_WORDS * 4) as f64 / down / 1e9, LINK, format!("readback of {} MiB into host memory, host-timed, fastest of {COPY_REPEATS}", (READBACK_WORDS * 4) >> 20)));

    let out = gpu.storage(words as u64);
    rows.push(match gpu.native_kernel(&STREAM_TRIAD, TRIAD_BINDINGS) {
        Some(id) => {
            let vectors = (STREAM_BYTES / 16) as u32;
            let steps: Vec<_> = (0..STREAM_LAUNCHES).filter_map(|_| gpu.native_step_raw(id, &[&src, &dst, &out], &[vectors, 0.5f32.to_bits()], crate::native_tune::ROOF_BLOCKS)).collect();
            stream_row(gpu, "hbm-triad", &steps, 3.0 * STREAM_BYTES as f64, "triad kernel a = b + s * c, grid-stride 16-byte vectors, three 1 GiB arrays", &out, &|i| (f32::from_bits(host[i]) * 1.5).to_bits())
        }
        None => DeviceRate::absent("hbm-triad", "GB/s", MEM, "the stream triad kernel does not compile on this backend".into()),
    });
    drop((dst, out));

    // The GPU reading host memory across the link: a page-locked host buffer first written by a core next to the card,
    // so its pages are on that module's memory and the read crosses only the card's own link. Page-locked, because
    // ordinary host memory a kernel reads over and over is migrated into device memory by a coherent driver, and the
    // probe then reads HBM (knowledge 506).
    let near = crate::devices::host_cores_near(dev.index).unwrap_or_default();
    let placed = backend_cpu::par::with_cores(&near, || {
        let b = gpu.try_alloc_placed("c2c-read probe", STREAM_BYTES, backend_api::AllocPolicy::Pinned)?;
        gpu.write(&b, &host);
        gpu.poll_wait();
        Ok::<DeviceBuffer, String>(b)
    });
    rows.push(match placed {
        Ok(b) => {
            let side = if near.is_empty() { "page-locked host memory (no CPU module is known to sit next to this card)".to_owned() } else { format!("page-locked host memory next to the card ({} cores)", near.len()) };
            read_row(gpu, "c2c-read", LINK, &b, checksum, &side)
        }
        Err(why) => DeviceRate::absent("c2c-read", "GB/s", LINK, format!("no host-resident buffer the GPU reads directly: {why}")),
    });
    gpu.set_kernel_timing(false);
    let order = ["hbm-read", "hbm-copy", "hbm-triad", "h2d", "d2h", "c2c-read"];
    rows.sort_by_key(|r| order.iter().position(|m| *m == r.metric));
    rows
}

/// The device memory's triad bandwidth (GB/s) over three buffers of `bytes` with the native stream kernel, the fastest
/// of launches timed one by one by device events: `crate::roof`'s memory roof where the backend runs native kernels.
/// `None` where it cannot (no native kernel, no device timing). Leaves the handle's timing off.
pub(crate) fn native_triad_gbs(gpu: &Gpu, bytes: u64) -> Option<f32> {
    let id = gpu.native_kernel(&STREAM_TRIAD, TRIAD_BINDINGS)?;
    if !gpu.set_kernel_timing(true) {
        return None;
    }
    let words = bytes / 4;
    let (b, c, a) = (gpu.storage(words), gpu.storage(words), gpu.storage(words));
    let vectors = u32::try_from(bytes / 16).ok()?;
    let steps: Vec<_> = (0..STREAM_LAUNCHES).filter_map(|_| gpu.native_step_raw(id, &[&b, &c, &a], &[vectors, 0.5f32.to_bits()], crate::native_tune::ROOF_BLOCKS)).collect();
    let stats = launch_stats::measure(gpu, &steps);
    gpu.set_kernel_timing(false);
    Some((3.0 * bytes as f64 / (stats?.min_ms * 1e-3) / 1e9) as f32)
}

/// The read probe over `buf` (1 GiB), checked against the host's checksum of its words.
fn read_row(gpu: &Gpu, metric: &'static str, peak: Option<Metric>, buf: &DeviceBuffer, checksum: u32, what: &str) -> DeviceRate {
    let k = &crate::native_tune::ROOF_READ;
    let Some(id) = gpu.native_kernel(k, crate::native_tune::ROOF_BINDINGS) else {
        return DeviceRate::absent(metric, "GB/s", peak, "the read probe does not compile on this backend".into());
    };
    let threads = u64::from(crate::native_tune::ROOF_BLOCKS) * 256;
    let out = gpu.storage(threads);
    let vectors = (STREAM_BYTES / 16) as u32;
    let steps: Vec<_> = (0..STREAM_LAUNCHES).filter_map(|_| gpu.native_step_raw(id, &[buf, &out], &[vectors], crate::native_tune::ROOF_BLOCKS)).collect();
    if steps.is_empty() {
        return DeviceRate::absent(metric, "GB/s", peak, "the backend declined the read probe's launch".into());
    }
    gpu.submit(&[], &steps[..1]);
    gpu.poll_wait();
    let got = gpu.read(&out, threads as usize).iter().fold(0u32, |a, w| a ^ w.to_bits());
    if got != checksum {
        return DeviceRate::absent(metric, "GB/s", peak, format!("the read probe's checksum {got:#010x} is not the data's {checksum:#010x}: the probe did not read every word"));
    }
    match launch_stats::measure(gpu, &steps) {
        Some(s) => DeviceRate::measured(metric, "GB/s", STREAM_BYTES as f64 / (s.min_ms * 1e-3) / 1e9, peak, format!("read probe (16-byte no-allocate loads, grid-stride) over 1 GiB of {what}, fastest of {STREAM_LAUNCHES} launches"))
            .verified("the XOR of every word read equals the host's".into()),
        None => DeviceRate::absent(metric, "GB/s", peak, "no device timing".into()),
    }
}

/// A stream kernel's row: checked on the first [`READBACK_WORDS`] words of `out` against `want`, then the fastest of
/// `steps` (launched one by one) over `bytes` moved a launch.
fn stream_row(gpu: &Gpu, metric: &'static str, steps: &[crate::Step], bytes: f64, what: &str, out: &DeviceBuffer, want: &dyn Fn(usize) -> u32) -> DeviceRate {
    if steps.is_empty() {
        return DeviceRate::absent(metric, "GB/s", Some(Metric::Memory), "the backend declined the stream kernel's launch".into());
    }
    gpu.submit(&[], &steps[..1]);
    gpu.poll_wait();
    if let Some(i) = gpu.read(out, READBACK_WORDS).iter().enumerate().position(|(i, w)| w.to_bits() != want(i)) {
        return DeviceRate::absent(metric, "GB/s", Some(Metric::Memory), format!("the stream kernel wrote a wrong word at {i}"));
    }
    match launch_stats::measure(gpu, steps) {
        Some(s) => DeviceRate::measured(metric, "GB/s", bytes / (s.min_ms * 1e-3) / 1e9, Some(Metric::Memory), format!("{what}, fastest of {} launches", steps.len()))
            .verified(format!("the first {} MiB written read back word for word", (READBACK_WORDS * 4) >> 20)),
        None => DeviceRate::absent(metric, "GB/s", Some(Metric::Memory), "no device timing".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pattern's words are exact small floats, so `b + 0.5 * c` over two copies is `1.5 b` exactly, and its
    /// checksum changes when any one word does.
    #[test]
    fn the_stream_pattern_is_exact_under_the_triad_and_checksummed() {
        for i in [0usize, 1, 8190, 8191, 1 << 20] {
            let v = f32::from_bits(pattern(i));
            assert_eq!(v + 0.5 * v, (v * 1.5), "word {i}");
            assert_eq!(v.fract(), 0.0);
        }
        let words: Vec<u32> = (0..1000).map(pattern).collect();
        let sum = words.iter().fold(0u32, |a, w| a ^ w);
        let mut broken = words.clone();
        broken[7] ^= 1;
        assert_ne!(broken.iter().fold(0u32, |a, w| a ^ w), sum);
    }
}
