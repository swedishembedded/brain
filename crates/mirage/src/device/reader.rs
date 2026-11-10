// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The text reader and the decoder-memory construction on the device.
//!
//! Windows of every text line of a batch run as one batch of `48 x 320`
//! images; the host cuts the strips, joins the frames and collapses the CTC
//! path (`vision::reader`). Memory construction runs the slot projections on
//! the device; the host supplies the index-driven inputs (box codes, pooled
//! cell ranges, reference-token embeddings), which are small.

use gpu_core::DeviceBuffer;
use model::ops::Weight;

use super::prims::ConvShape;
use super::Device;
use crate::config::{MirageConfig, READER_CLASSES, READER_MIX_KERNEL, READER_STAGES};
use crate::vision::memory::MemoryBuilder;
use crate::vision::reader::{Reader, FRAMES, WINDOW_H, WINDOW_W};

struct DBlock {
    norm: DeviceBuffer,
    mix: DeviceBuffer,
    up: Weight,
    down: Weight,
}

/// The reader weights resident on a device.
pub struct DeviceReader {
    dim: usize,
    w0: DeviceBuffer,
    stage_norm: Vec<DeviceBuffer>,
    dw: Vec<DeviceBuffer>,
    pw: Vec<Weight>,
    pw_out: Vec<usize>,
    blocks: Vec<DBlock>,
    final_norm: DeviceBuffer,
    out: Weight,
    bias: DeviceBuffer,
}

impl DeviceReader {
    /// Upload the host-prepared reader.
    pub fn upload(dev: &Device, r: &Reader) -> DeviceReader {
        DeviceReader {
            dim: r.dim,
            w0: dev.buf(&r.w0),
            stage_norm: r.stage_norm.iter().map(|n| dev.norm_weight(n)).collect(),
            dw: r.dw.iter().map(|w| dev.buf(w)).collect(),
            pw: r.pw.iter().map(|l| dev.weight("reader_pw", &l.wt, l.out, l.inn)).collect(),
            pw_out: r.pw.iter().map(|l| l.out).collect(),
            blocks: r
                .blocks
                .iter()
                .map(|b| DBlock {
                    norm: dev.norm_weight(&b.norm),
                    mix: dev.buf(&b.mix),
                    up: dev.weight("reader_up", &b.up.wt, b.up.out, b.up.inn),
                    down: dev.weight("reader_down", &b.down.wt, b.down.out, b.down.inn),
                })
                .collect(),
            final_norm: dev.norm_weight(&r.final_norm),
            out: dev.weight("reader_out", &r.out.wt, r.out.out, r.out.inn),
            bias: dev.buf(&r.bias),
        }
    }

    /// Logits `[windows, 80, 307]` for `windows` RGB windows (`[windows, 48, 320, 3]` bytes).
    pub fn forward(&self, dev: &Device, rgb: &[u8], windows: usize) -> Vec<f32> {
        assert_eq!(rgb.len(), windows * WINDOW_H * WINDOW_W * 3);
        let x: Vec<f32> = rgb.iter().map(|&p| f32::from(p) / 127.5 - 1.0).collect();
        let xd = dev.buf(&x);
        let mut s = Vec::new();
        let n = windows;
        let c0 = READER_STAGES[0];
        let (mut h, mut w) = (WINDOW_H / 2, WINDOW_W / 2);
        let g0 = ConvShape { batch: n as u32, h: WINDOW_H as u32, w: WINDOW_W as u32, kh: 3, kw: 3, sy: 2, sx: 2 };
        let y0 = dev.alloc(n * h * w * c0);
        dev.conv2d(&mut s, g0, 3, c0 as u32, &xd, &self.w0, &y0);
        let z = dev.zc_new(&mut s, &y0, &self.stage_norm[0], (n * h * w) as u32, c0 as u32);
        let mut cur = dev.silu_new(&mut s, &z, (n * h * w * c0) as u32);
        let mut c = c0;
        for (i, &(sy, sx)) in [(2usize, 2usize), (2, 1), (2, 1)].iter().enumerate() {
            let g = ConvShape { batch: n as u32, h: h as u32, w: w as u32, kh: 3, kw: 3, sy: sy as u32, sx: sx as u32 };
            let (ho, wo) = (g.ho() as usize, g.wo() as usize);
            let d = dev.alloc(n * ho * wo * c);
            dev.dwconv(&mut s, g, c as u32, &cur, &self.dw[i], &d);
            let p = dev.linear_new(&mut s, &self.pw[i], &d, (n * ho * wo) as u32);
            (h, w, c) = (ho, wo, self.pw_out[i]);
            let z = dev.zc_new(&mut s, &p, &self.stage_norm[i + 1], (n * h * w) as u32, c as u32);
            cur = dev.silu_new(&mut s, &z, (n * h * w * c) as u32);
            dev.flush(&mut s);
        }
        assert_eq!((w, c), (FRAMES, self.dim), "the reader ends in {FRAMES} frames of {} channels", self.dim);
        let rows = (n * w) as u32;
        let d = self.dim as u32;
        let mut seq = dev.alloc(n * w * c);
        dev.mean_h(&mut s, &cur, &seq, n as u32, h as u32, w as u32, c as u32);
        for b in &self.blocks {
            let nz = dev.zc_new(&mut s, &seq, &b.norm, rows, d);
            let g = ConvShape { batch: n as u32, h: 1, w: w as u32, kh: 1, kw: READER_MIX_KERNEL as u32, sy: 1, sx: 1 };
            let hh = dev.alloc(n * w * c);
            dev.dwconv(&mut s, g, d, &nz, &b.mix, &hh);
            let uv = dev.linear_new(&mut s, &b.up, &hh, rows);
            let gated = dev.alloc(n * w * c);
            dev.swiglu_rows(&mut s, &uv, &gated, rows, d);
            let down = dev.linear_new(&mut s, &b.down, &gated, rows);
            dev.add(&mut s, &down, &seq, rows * d);
            let _ = &mut seq;
            dev.flush(&mut s);
        }
        let nz = dev.zc_new(&mut s, &seq, &self.final_norm, rows, d);
        let logits = dev.linear_new(&mut s, &self.out, &nz, rows);
        dev.bias_add(&mut s, &logits, &self.bias, rows, READER_CLASSES as u32);
        dev.run_read(&s, &logits, n * w * READER_CLASSES)
    }
}

/// Host-built inputs of the slot rows of a batch, `slots` rows per image.
pub struct SlotInputs {
    /// Slot rows per image (the model's detections plus regions).
    pub slots: usize,
    /// `[batch * slots, head_dim + 64]`.
    pub proj_in: Vec<f32>,
    /// `[batch * slots, dim]`.
    pub pooled: Vec<f32>,
    /// `[batch * slots, dim]`: scaled reference-token embeddings.
    pub embedding: Vec<f32>,
    /// Slots present per image (the rest are padding).
    pub present: Vec<usize>,
}

/// One image's inputs to [`SlotInputs::collect`], all host-side.
pub struct SlotItem<'a> {
    /// Encoder memory `[cells, dim]`.
    pub encoder: &'a [f32],
    /// Dense head input `[side * side, dense_dim]`.
    pub head_in: &'a [f32],
    /// Valid encoder cells.
    pub valid: &'a [bool],
    /// The image's proposal slots.
    pub slots: Vec<crate::vision::memory::Slot>,
}

impl SlotInputs {
    /// Gather the slot inputs of a batch, padding every image to `capacity` slots.
    pub fn collect(m: &MemoryBuilder, items: &[SlotItem], capacity: usize) -> SlotInputs {
        let (mut proj_in, mut pooled, mut embedding, mut present) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for it in items {
            assert!(it.slots.len() <= capacity, "{} slots exceed the capacity {capacity}", it.slots.len());
            let (p, q) = m.slot_inputs(it.encoder, it.valid, it.head_in, &it.slots);
            let pad = capacity - it.slots.len();
            proj_in.extend(p);
            proj_in.extend(std::iter::repeat_n(0.0, pad * m.slot_proj.inn));
            pooled.extend(q);
            pooled.extend(std::iter::repeat_n(0.0, pad * m.dim));
            embedding.extend(m.slot_embedding(&it.slots));
            embedding.extend(std::iter::repeat_n(0.0, pad * m.dim));
            present.push(it.slots.len());
        }
        SlotInputs { slots: capacity, proj_in, pooled, embedding, present }
    }
}

/// Decoder-memory construction weights, resident.
pub struct DeviceMemory {
    dim: usize,
    cells: usize,
    pos: DeviceBuffer,
    slot_proj: Weight,
    slot_pool: Weight,
    slot_norm: DeviceBuffer,
}

/// A batch of decoder memories: `[batch, cells + slots, dim]` rows and their key mask.
pub struct MemoryBatch {
    /// Memory rows.
    pub rows: DeviceBuffer,
    /// `[batch, cells + slots]`: 1 for a live row.
    pub mask: DeviceBuffer,
    /// Rows per image.
    pub stride: usize,
    /// Images.
    pub batch: usize,
}

impl DeviceMemory {
    /// Upload from the host builder.
    pub fn upload(dev: &Device, cfg: &MirageConfig, m: &MemoryBuilder) -> DeviceMemory {
        DeviceMemory {
            dim: cfg.dim,
            cells: cfg.grid() * cfg.grid(),
            pos: dev.buf(&m.position_rows()),
            slot_proj: dev.weight("slot_proj", &m.slot_proj.wt, m.slot_proj.out, m.slot_proj.inn),
            slot_pool: dev.weight("slot_pool", &m.slot_pool.wt, m.slot_pool.out, m.slot_pool.inn),
            slot_norm: dev.norm_weight(&m.slot_norm),
        }
    }

    /// Build the memories of `batch` images from the encoder output `[batch, cells, dim]`.
    pub fn build(&self, dev: &Device, encoder: &DeviceBuffer, valid: &[Vec<bool>], slots: &SlotInputs) -> MemoryBatch {
        let (batch, d, cells, sl) = (valid.len(), self.dim, self.cells, slots.slots);
        let stride = cells + sl;
        let mut s = Vec::new();
        let rows = dev.alloc(batch * stride * d);
        let tmp = dev.alloc(batch * cells * d);
        dev.copy_rows(&mut s, &self.pos, (0, 0), &tmp, (cells as u32, 0), batch as u32, cells as u32, d as u32);
        dev.add(&mut s, encoder, &tmp, (batch * cells * d) as u32);
        dev.copy_rows(&mut s, &tmp, (cells as u32, 0), &rows, (stride as u32, 0), batch as u32, cells as u32, d as u32);
        let sr = (batch * sl) as u32;
        let mut u = dev.linear_new(&mut s, &self.slot_proj, &dev.buf(&slots.proj_in), sr);
        let pooled = dev.linear_new(&mut s, &self.slot_pool, &dev.buf(&slots.pooled), sr);
        dev.add(&mut s, &pooled, &u, sr * d as u32);
        u = dev.zc_new(&mut s, &u, &self.slot_norm, sr, d as u32);
        dev.add(&mut s, &dev.buf(&slots.embedding), &u, sr * d as u32);
        dev.copy_rows(&mut s, &u, (sl as u32, 0), &rows, (stride as u32, cells as u32), batch as u32, sl as u32, d as u32);
        dev.flush(&mut s);
        let mask: Vec<f32> = (0..batch)
            .flat_map(|b| valid[b].iter().map(|&v| f32::from(u8::from(v))).chain((0..sl).map(move |k| f32::from(u8::from(k < slots.present[b])))).collect::<Vec<_>>())
            .collect();
        MemoryBatch { rows, mask: dev.buf(&mask), stride, batch }
    }
}
