// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned variants of one model from a single resident
// copy of its weights, including variants that retrain norms and gates in full, for its clients. If your team needs
// expertise in multi-tenant inference engines then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The full tensors of served adapters (`modules_to_save`): one table per tensor the model lets an adapter replace,
//! a row per bank slot and one more for the base's own value.
//!
//! A norm gain or a gate scalar is small, so a batch that mixes rows of different adapters (and base rows) is served
//! by gathering each row's own copy of the tensor from the table: the model records [`SavedBank::gather`] where it
//! would read the base tensor, with the row-to-table-row index [`SavedBank::index`] builds from the slots of the
//! batch's rows. A row of the base, and an adapter that does not carry the tensor, read the base's own row, so the
//! base's tape is unchanged in value. A host mirror of every table answers the models that need a tensor's value on
//! the host (a scalar gate). A tensor too large to gather per row (a whole linear) is not a [`SavedSite`] of a
//! model: [`super::sites::validate_with_saved`] refuses an adapter that carries one, by name.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};

use super::sites::SavedSite;

/// One tensor's table: `(slots + 1) x numel` on the device, mirrored on the host.
struct Table {
    site: SavedSite,
    device: DeviceBuffer,
    host: Vec<f32>,
    /// Whether the base's value arrived ([`SavedBank::set_base`]).
    base_set: bool,
}

/// The tables of one bank (one pipeline stage's worth of tensors).
pub struct SavedBank {
    slots: u32,
    tables: Vec<Table>,
    by_name: HashMap<String, usize>,
    gather: usize,
}

impl SavedBank {
    /// Tables for `sites`, all rows zero until the base's values arrive ([`Self::set_base`]).
    pub fn new(gpu: &Gpu, sites: &[SavedSite], slots: u32) -> Result<SavedBank, String> {
        let gather = if sites.is_empty() { 0 } else { gpu.kernel_index("embed").ok_or("the device does not register the `embed` kernel the saved-tensor gather runs")? };
        let rows = u64::from(slots) + 1;
        let mut tables = Vec::with_capacity(sites.len());
        let mut by_name = HashMap::new();
        for site in sites {
            if by_name.insert(site.name.clone(), tables.len()).is_some() {
                return Err(format!("{}: listed twice among the tensors an adapter may carry in full", site.name));
            }
            let words = rows * site.numel as u64;
            if words * 4 > gpu.max_storage_binding_bytes() {
                return Err(format!("{}: {} slots of {} values exceed one storage binding", site.name, slots, site.numel));
            }
            tables.push(Table { site: site.clone(), device: gpu.storage(words), host: vec![0.0; words as usize], base_set: false });
        }
        Ok(SavedBank { slots, tables, by_name, gather })
    }

    /// Device bytes of the tables of `sites` at `slots` slots.
    pub fn bytes_of(sites: &[SavedSite], slots: u32) -> u64 {
        sites.iter().map(|s| (u64::from(slots) + 1) * s.numel as u64 * 4).sum()
    }

    pub fn bytes(&self) -> u64 {
        self.tables.iter().map(|t| t.host.len() as u64 * 4).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    /// The table index of the tensor `name`, `None` for one this bank does not hold.
    pub fn site_index(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    pub fn numel_of(&self, ix: usize) -> usize {
        self.tables[ix].site.numel
    }

    /// Set the base's value of tensor `name`: the row every base row reads, and the value of every slot until an adapter
    /// that carries the tensor takes it.
    pub fn set_base(&mut self, gpu: &Gpu, name: &str, values: &[f32]) -> Result<(), String> {
        let ix = self.site_index(name).ok_or_else(|| format!("{name}: not a tensor this bank holds"))?;
        let t = &mut self.tables[ix];
        if values.len() != t.site.numel {
            return Err(format!("{name}: the base holds {} values, the bank's table {}", values.len(), t.site.numel));
        }
        for row in t.host.chunks_mut(t.site.numel) {
            row.copy_from_slice(values);
        }
        gpu.write_f32(&t.device, &t.host);
        t.base_set = true;
        Ok(())
    }

    /// Whether the base's value of table `ix` was set ([`Self::set_base`]): until it is, every row holds zeros.
    pub fn base_is_set(&self, ix: usize) -> bool {
        self.tables[ix].base_set
    }

    /// Put an adapter in `slot`: each table's row becomes the adapter's tensor `(table index, values)` when it carries
    /// one, the base's value plus the adapter's delta `(table index, delta)` when it adds one (a merged site's
    /// `(alpha / rank) B A`), the base's value otherwise, so nothing of the slot's previous adapter survives.
    pub fn upload(&mut self, gpu: &Gpu, slot: u32, carried: &[(usize, Vec<f32>)], added: &[(usize, Vec<f32>)]) {
        assert!(slot < self.slots, "SavedBank::upload: slot {slot} of {}", self.slots);
        for (ix, t) in self.tables.iter_mut().enumerate() {
            let n = t.site.numel;
            let mut values: Vec<f32> = t.host[self.slots as usize * n..(self.slots as usize + 1) * n].to_vec();
            if let Some((_, v)) = carried.iter().find(|(i, _)| *i == ix) {
                values.copy_from_slice(v);
            }
            if let Some((_, delta)) = added.iter().find(|(i, _)| *i == ix) {
                values.iter_mut().zip(delta).for_each(|(w, d)| *w += d);
            }
            t.host[slot as usize * n..(slot as usize + 1) * n].copy_from_slice(&values);
            gpu.write_f32_at(&t.device, slot as u64 * n as u64, &values);
        }
    }

    /// The table row a row at `slot` reads: the slot's, or the base's for [`crate::dispatch::LORA_SLOT_NONE`].
    pub fn table_row(&self, slot: u32) -> u32 {
        if slot == crate::dispatch::LORA_SLOT_NONE {
            self.slots
        } else {
            slot
        }
    }

    /// Record the gather of `parts` of tensor `ix` (each `width` values: part `p` is values `p * width ..`) for the table
    /// rows `rows` ([`Self::table_row`]), part-major: `[parts.len(), rows.len(), width]`, part `parts[p]` of table row
    /// `rows[r]` at `(p * rows.len() + r) * width`.
    pub fn gather_parts(&self, gpu: &Gpu, steps: &mut Vec<Step>, ix: usize, rows: &[u32], parts: &[u32], width: u32) -> DeviceBuffer {
        let t = &self.tables[ix];
        let per = t.site.numel as u32 / width;
        assert!(width > 0 && t.site.numel as u32 == per * width && parts.iter().all(|&p| p < per), "SavedBank::gather_parts: parts {parts:?} of width {width} of a tensor of {}", t.site.numel);
        assert!(rows.iter().all(|&r| r <= self.slots), "SavedBank::gather_parts: table rows {rows:?} of {}", self.slots + 1);
        let index: Vec<u32> = parts.iter().flat_map(|&p| rows.iter().map(move |&r| r * per + p)).collect();
        let index_buf = gpu.storage(index.len().max(1) as u64);
        gpu.write(&index_buf, &index);
        let n = index.len() as u32;
        let out = gpu.storage(u64::from(n) * u64::from(width));
        steps.push(gpu.step(self.gather, &[&index_buf, &t.device, &out], &[width, n], n * width));
        out
    }

    /// The table row each row of a batch reads: its slot, or the base's row for a row of the base
    /// ([`crate::dispatch::LORA_SLOT_NONE`]), repeated `repeat` times per row (a per-head norm reads one row per head).
    pub fn index(&self, gpu: &Gpu, slots: &[u32], repeat: u32) -> DeviceBuffer {
        let rows: Vec<u32> = slots.iter().flat_map(|&s| std::iter::repeat_n(if s == crate::dispatch::LORA_SLOT_NONE { self.slots } else { s }, repeat as usize)).collect();
        let buf = gpu.storage(rows.len().max(1) as u64);
        gpu.write(&buf, &rows);
        buf
    }

    /// Record the gather of tensor `ix` for `rows` rows named by `index` (from [`Self::index`]): `[rows, numel]`.
    pub fn gather(&self, gpu: &Gpu, steps: &mut Vec<Step>, ix: usize, index: &DeviceBuffer, rows: u32) -> DeviceBuffer {
        let t = &self.tables[ix];
        let out = gpu.storage(u64::from(rows) * t.site.numel as u64);
        steps.push(gpu.step(self.gather, &[index, &t.device, &out], &[t.site.numel as u32, rows], rows * t.site.numel as u32));
        out
    }

    /// The value of tensor `name` a row at `slot` (or [`crate::dispatch::LORA_SLOT_NONE`]) reads, on the host.
    pub fn value(&self, name: &str, slot: u32) -> Option<&[f32]> {
        let t = &self.tables[self.site_index(name)?];
        let row = if slot == crate::dispatch::LORA_SLOT_NONE { self.slots } else { slot } as usize;
        Some(&t.host[row * t.site.numel..(row + 1) * t.site.numel])
    }
}
