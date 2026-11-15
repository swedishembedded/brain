// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements GPU attention kernels that run on devices with strict storage-binding alignment
// for its clients. If your team needs expertise in batched encoder attention on GPUs then you can procure our
// services by sending an email to info@swedishembedded.com.

//! `block::gemm_bidir_fwd` over several spans of one batch. A span after the first starts at `row0 * width` words of
//! its q, k, v and ctx buffers, which for a narrow model (LFM2.5's tiny shape: `d_model` 16, 4 heads, 2 kv heads,
//! `head_dim` 4) is not a multiple of the 64-word storage-binding grain, so that start must never become a binding
//! offset. Recording the dispatches checks that on any backend (the CPU backend applies the same check); on a device
//! that can run the GEMM path, every span of the batched call must equal the same span attended alone from row 0.

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::block::{self, GemmAttnIds};

const PIPES: &[(&str, &str)] = &[
    ("head_pack", kernels::HEAD_PACK),
    ("head_pack_t", kernels::HEAD_PACK_T),
    ("head_unpack", kernels::HEAD_UNPACK),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
];

/// LFM2.5's tiny encoder attention: `d_model` 16 as 4 query heads and 2 kv heads of 4.
const HEADS: u32 = 4;
const HEAD_DIM: u32 = 4;
const GROUP: u32 = 2;
const D_OUT: u32 = HEADS * HEAD_DIM;
const KV_STRIDE: u32 = D_OUT / GROUP;

fn ids(g: &Gpu) -> GemmAttnIds {
    let idx = |name: &str| g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"));
    GemmAttnIds {
        head_pack: idx("head_pack"),
        head_pack_t: idx("head_pack_t"),
        head_unpack: idx("head_unpack"),
        softmax_rows: idx("softmax_rows"),
        matmul: idx("matmul"),
        matmul_reg: idx("matmul_reg3"),
    }
}

/// The buffers one `gemm_bidir_fwd` call over spans of `len` rows (chunked by `chunk`) binds.
struct Bufs {
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    packs: DeviceBuffer,
    ctx_pack: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
}

impl Bufs {
    fn new(g: &Gpu, q: &[f32], k: &[f32], v: &[f32], len: u32, chunk: u32) -> Bufs {
        let (heads, head_words) = (u64::from(HEADS), block::pad64(u64::from(len) * u64::from(HEAD_DIM)));
        let slab = heads * block::pad64(u64::from(chunk.min(len)) * u64::from(len));
        Bufs {
            q: g.storage_init("q", q),
            k: g.storage_init("k", k),
            v: g.storage_init("v", v),
            ctx: g.storage((q.len() as u64).max(1)),
            packs: g.storage(3 * heads * head_words),
            ctx_pack: g.storage(heads * head_words),
            scores: g.storage(slab),
            probs: g.storage(slab),
        }
    }

    /// Records the attention of `spans` and returns its steps; `gemm_bidir_fwd` panics here on a misaligned binding.
    fn record(&self, g: &Gpu, spans: &[(u32, u32)], chunk: u32) -> Vec<gpu_core::Step> {
        let mut steps = Vec::new();
        block::gemm_bidir_fwd(
            g, &ids(g), HEADS, HEAD_DIM, GROUP, &self.q, D_OUT, (&self.k, &self.v), KV_STRIDE, &self.ctx, D_OUT,
            &self.packs, &self.ctx_pack, &self.scores, &self.probs, spans, chunk, false, &mut steps,
        );
        steps
    }
}

/// Host inputs for `batch` spans of `len` rows: q `[rows, D_OUT]`, k and v `[rows, KV_STRIDE]`.
fn inputs(batch: u32, len: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Lcg::new(0x5eed_6e33);
    let rows = (batch * len) as usize;
    let (q_words, kv_words) = (rows * D_OUT as usize, rows * KV_STRIDE as usize);
    (rng.vec_scaled(q_words, 1.0), rng.vec_scaled(kv_words, 1.0), rng.vec_scaled(kv_words, 1.0))
}

fn spans(batch: u32, len: u32) -> Vec<(u32, u32)> {
    (0..batch).map(|i| (i * len, len)).collect()
}

/// The reported shape: batch 2 of 42 rows starts span 2 of q at 42 * 16 words = 2688 bytes; plus 100-row spans in
/// chunks of 64, whose second span starts k and v at 100 * 8 words.
const SHAPES: &[(u32, u32)] = &[(42, 42), (100, 64)];

#[test]
fn batched_spans_record_only_aligned_bindings_on_the_cpu_backend() {
    let g = Gpu::new_cpu(PIPES);
    for &(len, chunk) in SHAPES {
        let (q, k, v) = inputs(2, len);
        let bufs = Bufs::new(&g, &q, &k, &v, len, chunk);
        assert!(!bufs.record(&g, &spans(2, len), chunk).is_empty(), "len {len}: nothing recorded");
    }
}

#[test]
fn every_batched_span_equals_the_same_span_attended_alone() {
    let g = gpu_core::testgpu::dev(PIPES);
    if !g.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("gemm_bidir_fwd parity: the GEMM attention path needs workgroup reductions (a GPU)");
        return;
    }
    for &(len, chunk) in SHAPES {
        let (q, k, v) = inputs(2, len);
        let batched = Bufs::new(&g, &q, &k, &v, len, chunk);
        g.submit(&[], &batched.record(&g, &spans(2, len), chunk));
        let got = g.read(&batched.ctx, q.len());

        for span in 0..2usize {
            let rows = |x: &[f32], width: u32| {
                let w = (len * width) as usize;
                x[span * w..(span + 1) * w].to_vec()
            };
            let alone = Bufs::new(&g, &rows(&q, D_OUT), &rows(&k, KV_STRIDE), &rows(&v, KV_STRIDE), len, chunk);
            g.submit(&[], &alone.record(&g, &spans(1, len), chunk));
            let want = g.read(&alone.ctx, (len * D_OUT) as usize);
            assert!(want.iter().any(|&x| x != 0.0), "len {len}: the span alone attended to nothing");
            assert_eq!(rows(&got, D_OUT), want, "len {len}, chunk {chunk}: span {span} of the batch differs from the span alone");
        }
    }
}
