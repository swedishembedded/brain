// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Folding device-side LoRA pairs into a model's own weights.
//!
//! [`super::exec::AdapterRuntime`] INJECTS a correction at a linear's output,
//! which suits a model that records its forward per call. A model with one
//! pre-recorded tape (a vision tower differentiated as a whole) needs the
//! adapted weights BEFORE the tape runs instead: it keeps an untouched copy
//! of each targeted weight, folds `W = W0 + scale·B·A` into the live one for
//! the forward, and projects the dense `dW` its backward leaves back onto the
//! two factors.
//!
//! That is seven dispatches over a list of targets - three to fold, four to
//! project - and every model that trains such a tower wrote them itself. This
//! module is the choreography once: the caller supplies the buffers and the
//! kernel ids its own pipeline registers, and keeps its store, its init and
//! its copies.
//!
//! Swedish Embedded AB implements adapter infrastructure that a new
//! pre-recorded model reuses instead of restating, for its clients. If your
//! team needs expertise in parameter-efficient fine-tuning across
//! architectures, you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::{f, DeviceBuffer, Gpu};

/// The four kernels a fold needs, as indices into the caller's pipeline set.
#[derive(Clone, Copy, Debug)]
pub struct FoldKernels {
    /// `x·Wᵀ` - `dB = dW·Aᵀ`.
    pub matmul: usize,
    /// `dYᵀ·X` - `B·A` for the fold, `dA = Bᵀ·dW` for the projection.
    pub matmul_dx: usize,
    /// `dYᵀ·X` accumulating - the projection's `dA`.
    pub matmul_dw: usize,
    /// `y += alpha·x`.
    pub axpy: usize,
}

impl FoldKernels {
    /// Resolve the four by name from a device that has the caller's pipelines
    /// registered; a missing one is named, not a silent no-op.
    pub fn resolve(gpu: &Gpu) -> FoldKernels {
        let need = |name: &str| gpu.kernel_index(name).unwrap_or_else(|| panic!("adapter fold: kernel {name:?} is not registered by this model"));
        FoldKernels { matmul: need("matmul"), matmul_dx: need("matmul_dx"), matmul_dw: need("matmul_dw"), axpy: need("axpy") }
    }
}

/// One linear to fold: its live weight, an untouched copy of the base, and
/// the two factors (`A` is `[rank, inn]`, `B` is `[out, rank]`).
pub struct Target<'a> {
    pub weight: &'a DeviceBuffer,
    pub base: &'a DeviceBuffer,
    pub a: &'a DeviceBuffer,
    pub b: &'a DeviceBuffer,
    pub out: u32,
    pub inn: u32,
}

/// One linear to project: the dense gradient its backward left in `dw`, the
/// factors, and their gradient buffers. A site that is a row slice of a fused
/// weight (q, k and v inside one `[3C, C]` tensor) names its first element in
/// `dw` as `offset`; a whole-tensor site has `offset` 0.
pub struct GradTarget<'a> {
    pub dw: &'a DeviceBuffer,
    pub offset: u64,
    pub a: &'a DeviceBuffer,
    pub b: &'a DeviceBuffer,
    pub ga: &'a DeviceBuffer,
    pub gb: &'a DeviceBuffer,
    pub out: u32,
    pub inn: u32,
}

/// One adapted row slice of a fused weight: the two factors, the slice's `[out, inn]` shape and the index of its
/// first element in the fused tensor (`row0 * row_stride`; see [`super::TargetSpec`]).
pub struct Slice<'a> {
    pub a: &'a DeviceBuffer,
    pub b: &'a DeviceBuffer,
    pub out: u32,
    pub inn: u32,
    pub offset: u64,
}

/// A fused weight (`len` elements), its untouched base copy and the adapted slices inside it. Slices must not
/// overlap; elements outside every slice are the base's.
pub struct FusedTarget<'a> {
    pub weight: &'a DeviceBuffer,
    pub base: &'a DeviceBuffer,
    pub len: u64,
    pub slices: Vec<Slice<'a>>,
}

/// Binding offsets are multiples of this many words on every backend (a 256-byte storage-offset alignment).
const OFFSET_ALIGN_WORDS: u64 = 64;

/// `weight = base`, then `weight[slice] += scale·B·A` for every slice of every fused target. The one clear and
/// copy per fused weight is what lets several sites share it (the whole-tensor [`fold`] clears per target).
///
/// # Panics
/// On a slice that runs past `len` or starts off a 64-word boundary.
pub fn fold_fused(gpu: &Gpu, k: &FoldKernels, targets: &[FusedTarget<'_>], rank: u32, scale: f32) {
    for t in targets {
        gpu.submit(&[t.weight], &[gpu.step(k.axpy, &[t.weight, t.base], &[t.len as u32, f(1.0)], t.len as u32)]);
        for sl in &t.slices {
            let n = sl.out * sl.inn;
            assert!(sl.offset + u64::from(n) <= t.len, "adapter fold: a {}x{} slice at {} runs past a fused weight of {} elements", sl.out, sl.inn, sl.offset, t.len);
            assert!(sl.offset.is_multiple_of(OFFSET_ALIGN_WORDS), "adapter fold: a slice at element {} is not on a {OFFSET_ALIGN_WORDS}-word boundary", sl.offset);
            let ba = gpu.storage(u64::from(n));
            gpu.submit(
                &[],
                &[
                    gpu.step(k.matmul_dx, &[sl.b, sl.a, &ba], &[sl.out, sl.inn, rank, 0], n),
                    gpu.step_sliced(k.axpy, &[t.weight, &ba], &[(sl.offset, 0), (0, 0)], &[n, f(scale)], n),
                ],
            );
        }
    }
}

/// `weight = base + scale·B·A` for every target (the live weight is written
/// from the base copy on each fold, so a refold after an update starts from
/// the untouched weights).
pub fn fold(gpu: &Gpu, k: &FoldKernels, targets: &[Target<'_>], rank: u32, scale: f32) {
    for t in targets {
        let ba = gpu.storage((t.out * t.inn) as u64);
        gpu.submit(
            &[t.weight],
            &[
                gpu.step(k.matmul_dx, &[t.b, t.a, &ba], &[t.out, t.inn, rank, 0], t.out * t.inn),
                gpu.step(k.axpy, &[t.weight, t.base], &[t.out * t.inn, f(1.0)], t.out * t.inn),
                gpu.step(k.axpy, &[t.weight, &ba], &[t.out * t.inn, f(scale)], t.out * t.inn),
            ],
        );
    }
}

/// `gB += scale·dW·Aᵀ` and `gA += scale·Bᵀ·dW` for every target.
pub fn project(gpu: &Gpu, k: &FoldKernels, targets: &[GradTarget<'_>], rank: u32, scale: f32) {
    for t in targets {
        assert!(t.offset.is_multiple_of(OFFSET_ALIGN_WORDS), "adapter projection: a slice at element {} is not on a {OFFSET_ALIGN_WORDS}-word boundary", t.offset);
        let db = gpu.storage((t.out * rank) as u64);
        let da = gpu.storage((rank * t.inn) as u64);
        let at = [(t.offset, 0), (0, 0), (0, 0)];
        let dw_second = [(0, 0), (t.offset, 0), (0, 0)];
        gpu.submit(
            &[&da],
            &[
                gpu.step_sliced(k.matmul, &[t.dw, t.a, &db], &at, &[t.out, t.inn, rank], t.out * rank),
                gpu.step(k.axpy, &[t.gb, &db], &[t.out * rank, f(scale)], t.out * rank),
                gpu.step_sliced(k.matmul_dw, &[t.b, t.dw, &da], &dw_second, &[t.out, t.inn, rank], rank * t.inn),
                gpu.step(k.axpy, &[t.ga, &da], &[rank * t.inn, f(scale)], rank * t.inn),
            ],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::Pair;

    const KERNELS: &[(&str, &str)] = &[("matmul", kernels::MATMUL), ("matmul_dx", kernels::MATMUL_DX), ("matmul_dw", kernels::MATMUL_DW), ("axpy", kernels::AXPY)];

    /// The folded weight is `W0 + s·B·A`, and the projected factors agree with
    /// the host `Pair` they represent.
    #[test]
    fn the_fold_and_projection_equal_the_host_pair() {
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let (out, inn, rank) = (4u32, 6u32, 2u32);
        let scale = 0.5f32;
        let mut rng = data::rng::Lcg::new(data::rng::random_seed());
        let w0: Vec<f32> = rng.vec_scaled((out * inn) as usize, 0.5);
        let a: Vec<f32> = rng.vec_scaled((rank * inn) as usize, 0.5);
        let b: Vec<f32> = rng.vec_scaled((out * rank) as usize, 0.5);
        let dw: Vec<f32> = rng.vec_scaled((out * inn) as usize, 0.5);

        let dev = |v: &[f32]| gpu.storage_init("test", v);
        let (dw_dev, w_dev, w0_dev, a_dev, b_dev) = (dev(&dw), gpu.storage_init("w", &vec![0.0; (out * inn) as usize]), dev(&w0), dev(&a), dev(&b));
        let k = FoldKernels::resolve(&gpu);

        fold(&gpu, &k, &[Target { weight: &w_dev, base: &w0_dev, a: &a_dev, b: &b_dev, out, inn }], rank, scale);
        let want = Pair::from_ab(out as usize, inn as usize, rank as usize, a.clone(), b.clone());
        let mut expected = w0.clone();
        want.delta(scale, &mut expected);
        // The device sums `B·A` in its own order: equal to the host value
        // within fp32 rounding, not bit-for-bit.
        assert!(rel(&gpu.read(&w_dev, (out * inn) as usize), &expected) < 1e-5, "fold must write W0 + s*B*A");

        let (ga, gb) = (vec![0.0f32; (rank * inn) as usize], vec![0.0f32; (out * rank) as usize]);
        let (ga_dev, gb_dev) = (dev(&ga), dev(&gb));
        project(&gpu, &k, &[GradTarget { dw: &dw_dev, offset: 0, a: &a_dev, b: &b_dev, ga: &ga_dev, gb: &gb_dev, out, inn }], rank, scale);
        let (want_da, want_db) = want.project(&dw, scale);
        assert!(rel(&gpu.read(&ga_dev, (rank * inn) as usize), &want_da) < 1e-5, "dA must equal the host projection");
        assert!(rel(&gpu.read(&gb_dev, (out * rank) as usize), &want_db) < 1e-5, "dB must equal the host projection");
    }

    /// Two sites that are row slices of ONE fused weight (q and v of a `[3C, C]` qkv), the third slice un-adapted:
    /// the fold writes the base plus each site's own delta into its rows and leaves the rest, and the projection reads
    /// each site's rows of the fused gradient.
    #[test]
    fn row_slices_of_a_fused_weight_fold_and_project_independently() {
        let gpu = gpu_core::testgpu::dev(KERNELS);
        let (out, inn, rank) = (8u32, 8u32, 2u32);
        let slice = u64::from(out * inn);
        let scale = 0.5f32;
        let mut rng = data::rng::Lcg::new(data::rng::random_seed());
        let fused: Vec<f32> = rng.vec_scaled(3 * slice as usize, 0.5);
        let dw: Vec<f32> = rng.vec_scaled(3 * slice as usize, 0.5);
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..2).map(|_| (rng.vec_scaled((rank * inn) as usize, 0.5), rng.vec_scaled((out * rank) as usize, 0.5))).collect();
        let dev = |v: &[f32]| gpu.storage_init("test", v);
        let (weight, base, dw_dev) = (gpu.storage_init("w", &vec![9.0; fused.len()]), dev(&fused), dev(&dw));
        let bufs: Vec<(DeviceBuffer, DeviceBuffer)> = pairs.iter().map(|(a, b)| (dev(a), dev(b))).collect();
        let k = FoldKernels::resolve(&gpu);
        // Sites at slice 0 (q) and slice 2 (v); slice 1 (k) has no adapter.
        let places = [0u64, 2 * slice];
        let slices = bufs.iter().zip(places).map(|((a, b), offset)| Slice { a, b, out, inn, offset }).collect();
        fold_fused(&gpu, &k, &[FusedTarget { weight: &weight, base: &base, len: fused.len() as u64, slices }], rank, scale);
        let mut expected = fused.clone();
        for ((a, b), off) in pairs.iter().zip(places) {
            Pair::from_ab(out as usize, inn as usize, rank as usize, a.clone(), b.clone()).delta(scale, &mut expected[off as usize..(off + slice) as usize]);
        }
        assert!(rel(&gpu.read(&weight, fused.len()), &expected) < 1e-5, "each slice is base + its own delta, the rest is the base");

        let grads: Vec<(DeviceBuffer, DeviceBuffer)> = pairs.iter().map(|(a, b)| (dev(&vec![0.0; a.len()]), dev(&vec![0.0; b.len()]))).collect();
        let targets: Vec<GradTarget<'_>> = bufs.iter().zip(&grads).zip(places).map(|(((a, b), (ga, gb)), offset)| GradTarget { dw: &dw_dev, offset, a, b, ga, gb, out, inn }).collect();
        project(&gpu, &k, &targets, rank, scale);
        for (((a, b), (ga, gb)), off) in pairs.iter().zip(&grads).zip(places) {
            let (want_da, want_db) = Pair::from_ab(out as usize, inn as usize, rank as usize, a.clone(), b.clone()).project(&dw[off as usize..(off + slice) as usize], scale);
            assert!(rel(&gpu.read(ga, a.len()), &want_da) < 1e-5, "dA of the slice at {off}");
            assert!(rel(&gpu.read(gb, b.len()), &want_db) < 1e-5, "dB of the slice at {off}");
        }
    }

    fn rel(got: &[f32], want: &[f32]) -> f32 {
        let n: f32 = got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        let d: f32 = want.iter().map(|v| v * v).sum::<f32>().sqrt();
        n / d.max(1e-12)
    }
}
