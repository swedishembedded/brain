// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in full fine-tuning of models larger
// than your accelerator memory then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The parallel primitives of the host update: the CPU scheduler's pool natively (`backend_cpu::par`, the workspace's
//! one rayon seam), the calling thread on wasm, which has no threads. Pieces are contiguous and indexed, so a body that
//! reads only its own piece gives the serial loop's bits whatever the pool size.

/// Elements below which a piece is not worth a task of its own.
pub(crate) const MIN_PIECE_ELEMS: usize = 1 << 16;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use backend_cpu::par::{chunks2_mut, chunks_mut, threads, zip3_mut};

#[cfg(target_arch = "wasm32")]
mod serial {
    pub(crate) fn threads() -> usize {
        1
    }
    pub(crate) fn chunks_mut<T: Send>(buf: &mut [T], chunk_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
        buf.chunks_mut(chunk_len).enumerate().for_each(|(i, c)| f(i, c));
    }
    pub(crate) fn chunks2_mut<T: Send, U: Send>(a: &mut [T], a_len: usize, b: &mut [U], b_len: usize, f: impl Fn(usize, &mut [T], &mut [U]) + Sync) {
        a.chunks_mut(a_len).zip(b.chunks_mut(b_len)).enumerate().for_each(|(i, (x, y))| f(i, x, y));
    }
    pub(crate) fn zip3_mut(a: &mut [f32], b: &mut [f32], c: &mut [f32], d: &[f32], f: impl Fn(&mut f32, &mut f32, &mut f32, &f32) + Sync) {
        for (((x, y), z), w) in a.iter_mut().zip(b).zip(c).zip(d) {
            f(x, y, z, w);
        }
    }
}
#[cfg(target_arch = "wasm32")]
pub(crate) use serial::*;
