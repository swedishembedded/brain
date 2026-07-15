// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GroupNorm (NCHW) host-side dispatch — spec: `docs/world-models/specs/P1.gn.md`.
//!
//! Composes the six `gn_*` kernels plus the existing `scale_chan` into a
//! forward pass (`gn_stats` → `gn_apply`) and a backward pass
//! (`scale_chan` → `gn_dsum` → `gn_dx`, plus `gn_dgamma`/`gn_dbeta`
//! accumulating into the packed `[gamma||beta]` grad buffer).
//!
//! Layout contracts (spec §2): `gb`/`dgb` are CONCATENATED `[2C]` buffers
//! (`gamma` at `[0,C)`, `beta` at `[C,2C)`); `stats` is `[2*N*G]`
//! (mean|rstd per `(n,g)`); `sums` is `[4*N*G]` (mean|rstd|S1|S2).

use gpu_core::{f, DeviceBuffer, Gpu, Step};

/// Dims + hyperparameters of one GroupNorm site. `eps` is a runtime
/// parameter (DIAMOND uses 1e-5), passed to `gn_stats` as f32 bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GnDims {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub g: u32,
    pub eps: f32,
}

impl GnDims {
    /// Validated constructor. Errors when any of `n,c,h,w,g` is zero, when
    /// `c % g != 0` (kernels assume divisibility), or when `eps <= 0.0`
    /// (zero-variance groups would produce inf/NaN).
    pub fn new(n: u32, c: u32, h: u32, w: u32, g: u32, eps: f32) -> Result<GnDims, String> {
        if n == 0 || c == 0 || h == 0 || w == 0 || g == 0 {
            return Err(format!(
                "GnDims: all dims must be nonzero (n={n}, c={c}, h={h}, w={w}, g={g})"
            ));
        }
        if c % g != 0 {
            return Err(format!(
                "GnDims: C must be divisible by G (c={c}, g={g}, c%g={})",
                c % g
            ));
        }
        // `!(eps > 0.0)` also rejects NaN.
        if !(eps > 0.0) {
            return Err(format!("GnDims: eps must be strictly positive (eps={eps})"));
        }
        Ok(GnDims { n, c, h, w, g, eps })
    }

    /// Total element count `n*c*h*w` (thread count for the per-element kernels).
    pub fn elems(&self) -> u32 {
        self.n * self.c * self.h * self.w
    }

    /// Group count `n*g` (thread count for `gn_stats`/`gn_dsum`).
    pub fn groups(&self) -> u32 {
        self.n * self.g
    }

    /// f32 length of the `stats` buffer: `2*n*g` (mean|rstd per group).
    pub fn stats_len(&self) -> u64 {
        2 * self.n as u64 * self.g as u64
    }

    /// f32 length of the `sums` buffer: `4*n*g` (mean|rstd|S1|S2 per group).
    pub fn sums_len(&self) -> u64 {
        4 * self.n as u64 * self.g as u64
    }

    /// The five u32 dims every `gn_*` kernel shares (spec §4, §8).
    fn dims5(&self) -> [u32; 5] {
        [self.n, self.c, self.h, self.w, self.g]
    }
}

/// DIAMOND group-count convention: `max(1, c / 32)` (u32 division). The
/// CALLER decides the group count; the kernels take whatever `G` they get.
pub fn num_groups(c: u32) -> u32 {
    (c / 32).max(1)
}

/// Kernel-table indices of the GroupNorm family inside the caller's [`Gpu`]
/// (same pattern as `optim::Optim`). `scale_chan` is the EXISTING
/// per-channel scale kernel reused to form `dyg = dy * gamma`.
#[derive(Clone, Copy, Debug)]
pub struct Gn {
    pub stats: usize,
    pub apply: usize,
    pub dgamma: usize,
    pub dbeta: usize,
    pub dsum: usize,
    pub dx: usize,
    pub scale_chan: usize,
}

impl Gn {
    /// `(name, source)` pairs for `Gpu::new`, in the order matching
    /// [`Gn::seq`]: `gn_stats, gn_apply, gn_dgamma, gn_dbeta, gn_dsum,
    /// gn_dx, scale_chan`.
    pub fn kernel_sources() -> [(&'static str, &'static str); 7] {
        [
            ("gn_stats", kernels::GN_STATS),
            ("gn_apply", kernels::GN_APPLY),
            ("gn_dgamma", kernels::GN_DGAMMA),
            ("gn_dbeta", kernels::GN_DBETA),
            ("gn_dsum", kernels::GN_DSUM),
            ("gn_dx", kernels::GN_DX),
            ("scale_chan", kernels::SCALE_CHAN),
        ]
    }

    /// A [`Gn`] whose indices are `0..=6`, matching [`Gn::kernel_sources`].
    pub fn seq() -> Gn {
        Gn { stats: 0, apply: 1, dgamma: 2, dbeta: 3, dsum: 4, dx: 5, scale_chan: 6 }
    }

    /// Forward steps `[gn_stats, gn_apply]` (spec §4.1–4.2). The caller
    /// submits. `stats` (`[2*N*G]`) is written and doubles as the backward
    /// cache; `y` is a fresh SSA output buffer.
    pub fn forward(
        &self,
        gpu: &Gpu,
        d: &GnDims,
        x: &DeviceBuffer,
        gb: &DeviceBuffer,
        stats: &DeviceBuffer,
        y: &DeviceBuffer,
    ) -> Vec<Step> {
        let [n, c, h, w, g] = d.dims5();
        vec![
            // gn_stats: [n, c, h, w, g, f(eps)], N*G threads (spec §4.1).
            gpu.step(self.stats, &[x, stats], &[n, c, h, w, g, f(d.eps)], d.groups()),
            // gn_apply: [n, c, h, w, g], N*C*H*W threads (spec §4.2).
            gpu.step(self.apply, &[x, stats, gb, y], &[n, c, h, w, g], d.elems()),
        ]
    }

    /// Backward steps `[scale_chan, gn_dsum, gn_dx, gn_dgamma, gn_dbeta]`
    /// in the spec §6 order. `stats` is the forward-pass cache; `dyg`
    /// (`[N*C*H*W]`) and `sums` (`[4*N*G]`) are scratch; `dx` is
    /// overwritten; `dgb` (`[2C]`, `[gamma||beta]` layout) is ACCUMULATED
    /// and must be pre-zeroed by the caller unless accumulating.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        gpu: &Gpu,
        d: &GnDims,
        x: &DeviceBuffer,
        gb: &DeviceBuffer,
        stats: &DeviceBuffer,
        dy: &DeviceBuffer,
        dyg: &DeviceBuffer,
        sums: &DeviceBuffer,
        dx: &DeviceBuffer,
        dgb: &DeviceBuffer,
    ) -> Vec<Step> {
        let dims5 = d.dims5();
        let [_n, c, h, w, _g] = dims5;
        vec![
            // scale_chan: dyg = dy * gamma_c. `gb` is passed as the `scale`
            // buffer — the channel index (idx/inner) % c stays in [0, C), so
            // only the gamma half of the concatenated [2C] buffer is read
            // (spec §5). Params [total, c, inner] = [n*c*h*w, c, h*w].
            gpu.step(self.scale_chan, &[dy, gb, dyg], &[d.elems(), c, h * w], d.elems()),
            // gn_dsum: per-(n,g) S1/S2 + copied-through mean/rstd (spec §4.5).
            gpu.step(self.dsum, &[x, dyg, stats, sums], &dims5, d.groups()),
            // gn_dx: per-element input grad, OVERWRITES dx (spec §4.6).
            gpu.step(self.dx, &[x, dyg, sums, dx], &dims5, d.elems()),
            // gn_dgamma / gn_dbeta: per-channel param grads, ACCUMULATE into
            // disjoint halves of dgb (spec §4.3–4.4). Independent of steps
            // 1–3 and of each other.
            gpu.step(self.dgamma, &[x, dy, stats, dgb], &dims5, c),
            gpu.step(self.dbeta, &[dy, dgb], &dims5, c),
        ]
    }
}
