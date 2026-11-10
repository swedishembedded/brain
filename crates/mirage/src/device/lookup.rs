// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Row lookup tables resident either as dense f32 or packed (rotated-codebook): the tied token embedding and the
//! engram tables. A packed table is read by `mirage_cq_gather` straight from its indices and norms, so the archive's
//! 2-4 bits per weight are all that is resident.
//!
//! Swedish Embedded AB implements solutions for serving large lookup tables from compressed storage for its
//! clients. If your team needs expertise in memory-lean embedding serving then you can procure our services by
//! sending an email to info@swedishembedded.com.

use gpu_core::{DeviceBuffer, Step};
use model::ops::Weight;

use super::Device;

/// A `[rows, width]` table of rows read by index.
#[derive(Clone)]
pub enum Lookup {
    /// Dense f32 rows.
    Dense(DeviceBuffer),
    /// A resident `Weight::Cq` matrix whose rows are the table rows.
    Packed(Weight),
}

impl Lookup {
    /// The packed table when `w` is a rotated-codebook weight, else the dense `buffer`.
    pub fn of(w: &Weight, dense: impl FnOnce() -> DeviceBuffer) -> Lookup {
        if matches!(w, Weight::Cq { .. }) {
            Lookup::Packed(w.clone())
        } else {
            Lookup::Dense(dense())
        }
    }
}

impl Device {
    /// `out[b, lane, :] = scale * table[tokens[b], :]` for `n` lanes per token.
    pub fn embed_lanes_from(&self, s: &mut Vec<Step>, table: &Lookup, tokens: &DeviceBuffer, out: &DeviceBuffer, (batch, n, d): (u32, u32, u32), scale: f32) {
        match table {
            Lookup::Dense(t) => self.embed_lanes(s, tokens, t, out, (batch, n, d), scale),
            Lookup::Packed(w) => self.packed_rows(s, w, tokens, out, (batch * n, d, n), scale),
        }
    }

    /// `out[b, t, :] = table[rows[b, t], :]` over `tables` tables per sequence; the absent id `0xFFFFFFFF` reads as zeros.
    pub fn gather_rows_from(&self, s: &mut Vec<Step>, table: &Lookup, rows: &DeviceBuffer, out: &DeviceBuffer, (batch, tables, sub): (u32, u32, u32)) {
        match table {
            Lookup::Dense(t) => self.gather_rows(s, rows, t, out, (batch, tables, sub)),
            Lookup::Packed(w) => self.packed_rows(s, w, rows, out, (batch * tables, sub, 1), 1.0),
        }
    }

    fn packed_rows(&self, s: &mut Vec<Step>, table: &Weight, ids: &DeviceBuffer, out: &DeviceBuffer, (count, width, rep): (u32, u32, u32), scale: f32) {
        let Weight::Cq { w, s: norms, book, k, bits, .. } = table else { unreachable!("a packed lookup holds a Cq weight") };
        assert_eq!(*k, width, "the packed table's rows are {k} wide, not {width}");
        assert!(width.is_multiple_of(128), "packed rows hold whole groups of 128 lanes");
        let bufs = [ids, w, norms, book, out];
        let params = [count, width, *bits, rep, scale.to_bits()];
        s.push(self.gpu().dispatch(self.k("mirage_cq_gather"), &bufs, &params, gpu_core::Dispatch::Workgroups(count * (width / 128))));
    }
}
