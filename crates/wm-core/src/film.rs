// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FiLM / adaLN modulation host-side dispatch.
//!
//! Three kernel sub-families over a shared `Gpu` kernel table:
//!
//! - **channel** (`film_chan`, `film_chan_dx`, `film_chan_dsb`): NCHW
//!   activations, `y = x*(1+s[n,c]) + b[n,c]`, with `s,b` packed in ONE
//!   buffer `sb[N,2C]` (scale first, shift second per row `n`).
//! - **row** (`film_row`, `film_row_dx`, `film_row_dsb`): `[R,D]` rows,
//!   one `(s,b)` vector per condition group of `rows_per_cond` consecutive
//!   rows (`sb[NC,2D]`, `NC = R/rows_per_cond`) — the per-frame
//!   diffusion-forcing conditioning path.
//! - **gate** (`gate_row`, `gate_row_dh`, `gate_row_dg`): adaLN-Zero gated
//!   residual `y = x + g[k,d]*h`, gate `g[NC,D]`.
//!
//! `s`, `b`, `g` are ACTIVATIONS (outputs of a conditioning projection),
//! so every backward output OVERWRITES a fresh SSA buffer (`=`, never
//! `+=`). **dx of `gate_row` is the identity** (`dx = dy`): there is
//! deliberately no kernel and no step builder for it — reuse `dy` or the
//! existing `add` kernel (spec §1).
//!
use gpu_core::{DeviceBuffer, Gpu, Step};

/// Dims of one channel-FiLM site (NCHW).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmChanDims {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
}

impl FilmChanDims {
    /// Validated constructor. Errors when any of `n,c,h,w` is zero.
    pub fn new(n: u32, c: u32, h: u32, w: u32) -> Result<FilmChanDims, String> {
        if n == 0 || c == 0 || h == 0 || w == 0 {
            return Err(format!(
                "FilmChanDims: all dims must be nonzero (n={n} c={c} h={h} w={w})"
            ));
        }
        Ok(FilmChanDims { n, c, h, w })
    }

    /// Total element count `n*c*h*w` (threads for `film_chan`/`film_chan_dx`).
    pub fn elems(&self) -> u32 {
        self.n * self.c * self.h * self.w
    }

    /// `(n,c)` pair count `n*c` (threads for `film_chan_dsb`).
    pub fn pairs(&self) -> u32 {
        self.n * self.c
    }

    /// f32 length of the packed `sb`/`dsb` buffer: `2*n*c`.
    pub fn sb_len(&self) -> u64 {
        2 * self.n as u64 * self.c as u64
    }

    /// The four u32 params every `film_chan*` kernel shares (spec §4.1–4.3).
    fn dims4(&self) -> [u32; 4] {
        [self.n, self.c, self.h, self.w]
    }
}

/// Dims of one row-FiLM / gate site (`[R,D]` rows, grouped conditioning).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmRowDims {
    pub r: u32,
    pub d: u32,
    pub rows_per_cond: u32,
}

impl FilmRowDims {
    /// Validated constructor. Errors when any field is zero or when
    /// `r % rows_per_cond != 0` (kernels assume divisibility, spec §2.3/§8).
    pub fn new(r: u32, d: u32, rows_per_cond: u32) -> Result<FilmRowDims, String> {
        if r == 0 || d == 0 || rows_per_cond == 0 {
            return Err(format!(
                "FilmRowDims: all dims must be nonzero (r={r} d={d} rows_per_cond={rows_per_cond})"
            ));
        }
        if !r.is_multiple_of(rows_per_cond) {
            return Err(format!(
                "FilmRowDims: r must be divisible by rows_per_cond (r={r} rows_per_cond={rows_per_cond})"
            ));
        }
        Ok(FilmRowDims { r, d, rows_per_cond })
    }

    /// Total element count `r*d` (threads for the per-element kernels).
    pub fn elems(&self) -> u32 {
        self.r * self.d
    }

    /// Condition-group count `NC = r / rows_per_cond`.
    pub fn conds(&self) -> u32 {
        self.r / self.rows_per_cond
    }

    /// `conds()*d` (threads for `film_row_dsb`/`gate_row_dg`).
    pub fn cond_elems(&self) -> u32 {
        self.conds() * self.d
    }

    /// f32 length of the packed `sb`/`dsb` buffer: `2*conds()*d`.
    pub fn sb_len(&self) -> u64 {
        2 * self.conds() as u64 * self.d as u64
    }

    /// f32 length of the gate `g`/`dg` buffer: `conds()*d`.
    pub fn g_len(&self) -> u64 {
        self.conds() as u64 * self.d as u64
    }

    /// The three u32 params every row/gate kernel shares (spec §4.4–4.9).
    fn dims3(&self) -> [u32; 3] {
        [self.r, self.d, self.rows_per_cond]
    }
}

/// Kernel-table indices of the FiLM family inside the caller's [`Gpu`]
/// (same pattern as `optim::Optim`).
#[derive(Clone, Copy, Debug)]
pub struct Film {
    pub chan: usize,
    pub chan_dx: usize,
    pub chan_dsb: usize,
    pub row: usize,
    pub row_dx: usize,
    pub row_dsb: usize,
    pub gate: usize,
    pub gate_dh: usize,
    pub gate_dg: usize,
}

impl Film {
    /// `(name, source)` pairs for `Gpu::new`, in the order matching
    /// [`Film::seq`]: `["film_chan", "film_chan_dx", "film_chan_dsb",
    /// "film_row", "film_row_dx", "film_row_dsb", "gate_row",
    /// "gate_row_dh", "gate_row_dg"]`.
    pub const fn kernel_sources() -> [(&'static str, &'static str); 9] {
        [
            ("film_chan", kernels::FILM_CHAN),
            ("film_chan_dx", kernels::FILM_CHAN_DX),
            ("film_chan_dsb", kernels::FILM_CHAN_DSB),
            ("film_row", kernels::FILM_ROW),
            ("film_row_dx", kernels::FILM_ROW_DX),
            ("film_row_dsb", kernels::FILM_ROW_DSB),
            ("gate_row", kernels::GATE_ROW),
            ("gate_row_dh", kernels::GATE_ROW_DH),
            ("gate_row_dg", kernels::GATE_ROW_DG),
        ]
    }

    /// `Film` with indices `0..=8` matching [`Film::kernel_sources`] order.
    pub fn seq() -> Film {
        Film {
            chan: 0,
            chan_dx: 1,
            chan_dsb: 2,
            row: 3,
            row_dx: 4,
            row_dsb: 5,
            gate: 6,
            gate_dh: 7,
            gate_dg: 8,
        }
    }

    /// `film_chan`: `y = x*(1+s[n,c]) + b[n,c]`. Params `[n,c,h,w]`,
    /// `d.elems()` threads. OVERWRITES `y`. Caller submits.
    pub fn step_chan(
        &self,
        gpu: &Gpu,
        d: &FilmChanDims,
        x: &DeviceBuffer,
        sb: &DeviceBuffer,
        y: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.chan, &[x, sb, y], &d.dims4(), d.elems())
    }

    /// `film_chan_dx`: `dx = dy*(1+s[n,c])`. Params `[n,c,h,w]`,
    /// `d.elems()` threads. OVERWRITES `dx`.
    pub fn step_chan_dx(
        &self,
        gpu: &Gpu,
        d: &FilmChanDims,
        dy: &DeviceBuffer,
        sb: &DeviceBuffer,
        dx: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.chan_dx, &[dy, sb, dx], &d.dims4(), d.elems())
    }

    /// `film_chan_dsb`: `ds[n,c] = Σ_hw dy*x`, `db[n,c] = Σ_hw dy`, both
    /// halves of `dsb[N,2C]`. Params `[n,c,h,w]`, `d.pairs()` threads.
    /// OVERWRITES `dsb`.
    pub fn step_chan_dsb(
        &self,
        gpu: &Gpu,
        d: &FilmChanDims,
        x: &DeviceBuffer,
        dy: &DeviceBuffer,
        dsb: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.chan_dsb, &[x, dy, dsb], &d.dims4(), d.pairs())
    }

    /// `film_row`: `y[r,d] = x*(1+s[g(r),d]) + b[g(r),d]`. Params
    /// `[r,d,rows_per_cond]`, `d.elems()` threads. OVERWRITES `y`.
    pub fn step_row(
        &self,
        gpu: &Gpu,
        d: &FilmRowDims,
        x: &DeviceBuffer,
        sb: &DeviceBuffer,
        y: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.row, &[x, sb, y], &d.dims3(), d.elems())
    }

    /// `film_row_dx`: `dx = dy*(1+s[g(r),d])`. Params `[r,d,rows_per_cond]`,
    /// `d.elems()` threads. OVERWRITES `dx`.
    pub fn step_row_dx(
        &self,
        gpu: &Gpu,
        d: &FilmRowDims,
        dy: &DeviceBuffer,
        sb: &DeviceBuffer,
        dx: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.row_dx, &[dy, sb, dx], &d.dims3(), d.elems())
    }

    /// `film_row_dsb`: per `(cond,d)` sums over the group's rows into both
    /// halves of `dsb[NC,2D]`. Params `[r,d,rows_per_cond]`,
    /// `d.cond_elems()` threads. OVERWRITES `dsb`.
    pub fn step_row_dsb(
        &self,
        gpu: &Gpu,
        d: &FilmRowDims,
        x: &DeviceBuffer,
        dy: &DeviceBuffer,
        dsb: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.row_dsb, &[x, dy, dsb], &d.dims3(), d.cond_elems())
    }

    /// `gate_row`: `y = x + g[gi(r),d]*h`. Params `[r,d,rows_per_cond]`,
    /// `d.elems()` threads. OVERWRITES `y`.
    pub fn step_gate(
        &self,
        gpu: &Gpu,
        d: &FilmRowDims,
        x: &DeviceBuffer,
        g: &DeviceBuffer,
        h: &DeviceBuffer,
        y: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.gate, &[x, g, h, y], &d.dims3(), d.elems())
    }

    /// `gate_row_dh`: `dh = dy*g[gi(r),d]`. Params `[r,d,rows_per_cond]`,
    /// `d.elems()` threads. OVERWRITES `dh`.
    ///
    /// NOTE: the residual grad `dx` of `gate_row` is the IDENTITY
    /// (`dx = dy`) — deliberately no `step_gate_dx` (spec §1).
    pub fn step_gate_dh(
        &self,
        gpu: &Gpu,
        d: &FilmRowDims,
        dy: &DeviceBuffer,
        g: &DeviceBuffer,
        dh: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.gate_dh, &[dy, g, dh], &d.dims3(), d.elems())
    }

    /// `gate_row_dg`: `dg[k,d] = Σ_{rows in cond k} dy*h` over `dg[NC,D]`.
    /// Params `[r,d,rows_per_cond]`, `d.cond_elems()` threads.
    /// OVERWRITES `dg`.
    pub fn step_gate_dg(
        &self,
        gpu: &Gpu,
        d: &FilmRowDims,
        dy: &DeviceBuffer,
        h: &DeviceBuffer,
        dg: &DeviceBuffer,
    ) -> Step {
        gpu.step(self.gate_dg, &[dy, h, dg], &d.dims3(), d.cond_elems())
    }
}
