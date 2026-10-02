// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The native `gqa_decode_prep` kernel (`kernels_cuda`, `cu/gqa_decode_prep.cu`)
//! - the front half of a gated-attention decode step in one launch - against
//! the eight-kernel WGSL chain `gqa_mixer_decode_batched_attend` dispatches, on
//! the same device from the same inputs.
//!
//! Swedish Embedded AB implements bit-exact fused kernels for attention decode.
//! If your team needs expertise in replacing the chain of tiny kernels in front
//! of a decode-time attention with one launch without moving a single output
//! bit, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Byte-identity of everything the step produces: the attention output, the
//! gate half, and BOTH KV pools in full (the kernel writes the new K and V row
//! into the pool; a row in the wrong place, or a stray write next to it, is a
//! wrong cache rather than a wrong number).
//!
//! The pools start RANDOM so an append that lands one row off, or not at all,
//! shows. Heads are the real 256 wide and a 128-wide one, with the real partial
//! rotary (a quarter of the head) and an odd rotary width, so the pass-through
//! tail past the rotated span is exercised too.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::{KernelIds, UNREGISTERED};
use model::gqa_mixer::{gqa_mixer_decode_batched_kv_attend, gqa_mixer_decode_fused_kv_attend, GqaMixerIds, GqaMixerShape, GqaMixerWeights, PagedDecodeBatch};
use model::kv_tier::{KvKernels, KvLayer, KvTier};

const KERNELS: &[(&str, &str)] = &[
    ("rmsnorm", kernels::RMSNORM),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("rope2d_partial", kernels::ROPE2D_PARTIAL),
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
];

/// The chain's pipelines plus every KV tier's kernels (the decode attention the
/// fused prep hands over to).
fn pipelines() -> &'static [(&'static str, &'static str)] {
    static ALL: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    ALL.get_or_init(|| {
        let mut all: Vec<(&'static str, &'static str)> = KERNELS.to_vec();
        all.extend(model::kv_tier::kernel_list(KERNELS));
        all
    })
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn mixer_ids(g: &Gpu) -> GqaMixerIds {
    let u = UNREGISTERED;
    GqaMixerIds {
        kernels: KernelIds {
            rmsnorm: idx(g, "rmsnorm"),
            rms_inv: u,
            rmsnorm_dx: u,
            rmsnorm_dx_rows: u,
            rmsnorm_dw: u,
            rope: u,
            rope_bwd: u,
            gqa_scores: u,
            gqa_apply: u,
            attn_softmax: u,
            gqa_dscores: u,
            gqa_dv: u,
            gqa_dq: u,
            gqa_dk: u,
            silu_mul: u,
            silu_da: u,
            silu_db: u,
            rmsnorm_rows: u,
        },
        concat_split: idx(g, "concat_split"),
        concat2: u,
        sigmoid: u,
        sigmoid_bwd: u,
        mul: u,
        rope2d_partial: idx(g, "rope2d_partial"),
    }
}

fn rnd(r: &mut Lcg, n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|_| r.scaled(s)).collect()
}

fn bits(v: Vec<f32>) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// Whether this device is offered the native kernels: a CUDA device, with
/// `BRAIN_NO_NATIVE_KERNELS` (the A/B switch that pins the WGSL tier) unset.
/// Under it these gates skip, and the "offered only where it can run" test
/// asserts the withholding.
fn is_cuda(g: &Gpu) -> bool {
    g.kind() == "cuda" && g.caps().arch.compute_capability.is_some() && !std::env::var("BRAIN_NO_NATIVE_KERNELS").is_ok_and(|v| v != "0")
}

struct Case {
    nh: u32,
    nkv: u32,
    hd: u32,
    half: u32,
    /// Keys already cached, and the pool's per-sequence capacity.
    ctx_len: u32,
    block_size: u32,
}

fn check(g: &Gpu, c: &Case, seed: u64) {
    let Case { nh, nkv, hd, half, ctx_len, block_size } = *c;
    let shape = GqaMixerShape { b: 1, t: 1, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: half, rms_eps: 1e-6 };
    let (qd, kvd) = (shape.qd(), shape.kvd());
    let mut r = Lcg::new(seed);

    let q_norm = g.storage_init("q_norm", &rnd(&mut r, hd as usize, 1.0));
    let k_norm = g.storage_init("k_norm", &rnd(&mut r, hd as usize, 1.0));
    let cos = g.storage_init("cos", &rnd(&mut r, half as usize, 1.0));
    let sin = g.storage_init("sin", &rnd(&mut r, half as usize, 1.0));
    let w = GqaMixerWeights { q_norm: &q_norm, k_norm: &k_norm, cos: &cos, sin: &sin };

    let q_full = g.storage_init("q_full", &rnd(&mut r, (nh * 2 * hd) as usize, 2.0));
    let k = g.storage_init("k", &rnd(&mut r, kvd as usize, 2.0));
    let v = g.storage_init("v", &rnd(&mut r, kvd as usize, 2.0));

    // One physical block, not the first, backs the sequence (the engine's shape).
    let num_blocks = 3u32;
    let phys = 1u32;
    let pool_words = (num_blocks * block_size * kvd) as usize;
    let pk0 = rnd(&mut r, pool_words, 1.0);
    let pv0 = rnd(&mut r, pool_words, 1.0);
    let blocks = g.storage(1);
    g.write(&blocks, &[phys]);
    let offsets = g.storage(1);
    g.write(&offsets, &[ctx_len]);
    let block_tables = g.storage(1);
    g.write(&block_tables, &[phys]);
    let seq_lens = g.storage(1);
    g.write(&seq_lens, &[ctx_len + 1]);
    let paged = PagedDecodeBatch { blocks: &blocks, offsets: &offsets, block_tables: &block_tables, seq_lens: &seq_lens, block_size, max_bt: 1, cap: block_size };

    let kv = KvKernels::resolve(g, KvTier::F32).expect("the f32 KV kernels are registered");
    let softmax = idx(g, "decode_softmax_batched");
    let layer = || KvLayer::new(g, KvTier::F32, u64::from(num_blocks * block_size), u64::from(kvd), u64::from(hd));
    let (ref_layer, nat_layer) = (layer(), layer());
    for l in [&ref_layer, &nat_layer] {
        g.write_f32(l.k.data(), &pk0);
        g.write_f32(l.v.data(), &pv0);
    }
    let (ctx_ref, gate_ref) = gqa_mixer_decode_batched_kv_attend(g, &mixer_ids(g), &kv, softmax, &shape, &w, &q_full, &k, &v, &ref_layer, 1, &paged);
    let (ctx_nat, gate_nat) = gqa_mixer_decode_fused_kv_attend(g, &kv, softmax, &shape, &w, &q_full, &k, &v, &nat_layer, 1, &paged)
        .expect("the fused GQA decode prep was declined on a CUDA device");
    g.poll_wait();
    let (pk_ref, pv_ref, pk_nat, pv_nat) = (ref_layer.k.data(), ref_layer.v.data(), nat_layer.k.data(), nat_layer.v.data());

    let what = format!("nh={nh} nkv={nkv} hd={hd} half={half} ctx={ctx_len} seed={seed:#x}");
    assert_eq!(bits(g.read(&gate_nat, qd as usize)), bits(g.read(&gate_ref, qd as usize)), "gate half differs: {what}");
    assert_eq!(bits(g.read(&ctx_nat, qd as usize)), bits(g.read(&ctx_ref, qd as usize)), "attention output differs: {what}");
    assert_eq!(bits(g.read(pk_nat, pool_words)), bits(g.read(pk_ref, pool_words)), "K pool differs: {what}");
    assert_eq!(bits(g.read(pv_nat, pool_words)), bits(g.read(pv_ref, pool_words)), "V pool differs: {what}");
}

#[test]
fn the_fused_prep_is_offered_only_where_it_can_run() {
    let g = gpu_core::testgpu::dev(pipelines());
    let shape = GqaMixerShape { b: 1, t: 1, n_heads: 4, n_kv_heads: 2, head_dim: 64, rotary_half: 8, rms_eps: 1e-6 };
    let dummy = g.storage(1024);
    let w = GqaMixerWeights { q_norm: &dummy, k_norm: &dummy, cos: &dummy, sin: &dummy };
    let paged = PagedDecodeBatch { blocks: &dummy, offsets: &dummy, block_tables: &dummy, seq_lens: &dummy, block_size: 4, max_bt: 1, cap: 4 };
    let kv = KvKernels::resolve(&g, KvTier::F32).expect("the f32 KV kernels are registered");
    let softmax = idx(&g, "decode_softmax_batched");
    let layer_in = |tier: KvTier| KvLayer::new(&g, tier, 16, u64::from(shape.kvd()), 64);
    let try_it = |s: &GqaMixerShape, batch: u32, layer: &KvLayer| {
        gqa_mixer_decode_fused_kv_attend(&g, &kv, softmax, s, &w, &dummy, &dummy, &dummy, layer, batch, &paged).is_some()
    };
    let f32_layer = layer_in(KvTier::F32);
    if !is_cuda(&g) {
        assert!(!try_it(&shape, 1, &f32_layer), "only a CUDA device takes a native fused kernel");
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert!(!try_it(&shape, 0, &f32_layer), "an empty batch has nothing to launch");
    assert!(!try_it(&GqaMixerShape { head_dim: 1024, ..shape }, 1, &f32_layer), "a head wider than the block holds");
    assert!(!try_it(&GqaMixerShape { rotary_half: 40, ..shape }, 1, &f32_layer), "a rotated span wider than the head");
    assert!(!try_it(&shape, 1, &layer_in(KvTier::Bf16)), "a compact KV tier keeps the tier's own append");
    assert!(!try_it(&shape, 1, &layer_in(KvTier::Int8)), "a compact KV tier keeps the tier's own append");
}

#[test]
fn the_fused_prep_is_byte_identical_to_the_wgsl_chain() {
    let g = gpu_core::testgpu::dev(pipelines());
    if !is_cuda(&g) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    // (heads, kv heads, head dim, rotary half): the real Qwen3.8-27B attention
    // (24 / 4 / 256 / 32), then 128-wide heads, an odd rotary width, a rotary
    // span that fills the whole head, one head, and a head that is not a
    // multiple of the 256-thread block.
    for (nh, nkv, hd, half) in [(24u32, 4u32, 256u32, 32u32), (6, 2, 128, 16), (4, 2, 64, 7), (2, 1, 64, 32), (1, 1, 32, 4), (3, 1, 96, 10), (4, 2, 512, 64)] {
        for ctx_len in [0u32, 5, 17] {
            check(&g, &Case { nh, nkv, hd, half, ctx_len, block_size: 24 }, 0xa11 ^ u64::from(nh * 31 + hd + half + ctx_len));
        }
    }
}

/// Which sequences the rows of a batch belong to.
#[derive(Clone, Copy, Debug)]
enum Rows {
    /// Every row its own sequence: its own physical block, at its own depth.
    Independent,
    /// Consecutive tokens of ONE sequence - a verify round: one block, offsets
    /// that step by one.
    OneSequence,
}

/// The batch form: `rows` tokens through one launch against the chain's
/// `rows`-row batch, comparing the attention output, the gate and both pools in
/// full. Each row has its own random projections and its own rotary row, so a
/// row read at the wrong stride shows.
fn check_rows(g: &Gpu, rows: u32, layout: Rows, seed: u64) {
    let (nh, nkv, hd, half, block_size) = (24u32, 4u32, 256u32, 32u32, 24u32);
    let shape = GqaMixerShape { b: 1, t: 1, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: half, rms_eps: 1e-6 };
    let (qd, kvd) = (shape.qd(), shape.kvd());
    let mut r = Lcg::new(seed);
    let rn = rows as usize;

    let q_norm = g.storage_init("q_norm", &rnd(&mut r, hd as usize, 1.0));
    let k_norm = g.storage_init("k_norm", &rnd(&mut r, hd as usize, 1.0));
    let cos = g.storage_init("cos", &rnd(&mut r, rn * half as usize, 1.0));
    let sin = g.storage_init("sin", &rnd(&mut r, rn * half as usize, 1.0));
    let w = GqaMixerWeights { q_norm: &q_norm, k_norm: &k_norm, cos: &cos, sin: &sin };
    let q_full = g.storage_init("q_full", &rnd(&mut r, rn * (nh * 2 * hd) as usize, 2.0));
    let k = g.storage_init("k", &rnd(&mut r, rn * kvd as usize, 2.0));
    let v = g.storage_init("v", &rnd(&mut r, rn * kvd as usize, 2.0));

    let num_blocks = rows + 1;
    let pool_words = (num_blocks * block_size * kvd) as usize;
    let (pk0, pv0) = (rnd(&mut r, pool_words, 1.0), rnd(&mut r, pool_words, 1.0));
    let (phys, depth): (Vec<u32>, Vec<u32>) = match layout {
        Rows::Independent => ((0..rows).map(|i| rows - i).collect(), (0..rows).map(|i| 3 + 2 * i).collect()),
        Rows::OneSequence => (vec![1; rn], (0..rows).map(|i| 4 + i).collect()),
    };
    let blocks = g.storage(u64::from(rows));
    g.write(&blocks, &phys);
    let offsets = g.storage(u64::from(rows));
    g.write(&offsets, &depth);
    let block_tables = g.storage(u64::from(rows));
    g.write(&block_tables, &phys);
    let seq_lens = g.storage(u64::from(rows));
    g.write(&seq_lens, &depth.iter().map(|d| d + 1).collect::<Vec<u32>>());
    let paged = PagedDecodeBatch { blocks: &blocks, offsets: &offsets, block_tables: &block_tables, seq_lens: &seq_lens, block_size, max_bt: 1, cap: block_size };

    let kv = KvKernels::resolve(g, KvTier::F32).expect("the f32 KV kernels are registered");
    let softmax = idx(g, "decode_softmax_batched");
    let layer = || KvLayer::new(g, KvTier::F32, u64::from(num_blocks * block_size), u64::from(kvd), u64::from(hd));
    let (ref_layer, nat_layer) = (layer(), layer());
    for l in [&ref_layer, &nat_layer] {
        g.write_f32(l.k.data(), &pk0);
        g.write_f32(l.v.data(), &pv0);
    }
    let (ctx_ref, gate_ref) = gqa_mixer_decode_batched_kv_attend(g, &mixer_ids(g), &kv, softmax, &shape, &w, &q_full, &k, &v, &ref_layer, rows, &paged);
    let (ctx_nat, gate_nat) = gqa_mixer_decode_fused_kv_attend(g, &kv, softmax, &shape, &w, &q_full, &k, &v, &nat_layer, rows, &paged)
        .expect("the fused GQA decode prep was declined on a CUDA device");
    g.poll_wait();

    let what = format!("rows={rows} {layout:?} seed={seed:#x}");
    let n = rn * qd as usize;
    assert_eq!(bits(g.read(&gate_nat, n)), bits(g.read(&gate_ref, n)), "gate half differs: {what}");
    assert_eq!(bits(g.read(&ctx_nat, n)), bits(g.read(&ctx_ref, n)), "attention output differs: {what}");
    assert_eq!(bits(g.read(nat_layer.k.data(), pool_words)), bits(g.read(ref_layer.k.data(), pool_words)), "K pool differs: {what}");
    assert_eq!(bits(g.read(nat_layer.v.data(), pool_words)), bits(g.read(ref_layer.v.data(), pool_words)), "V pool differs: {what}");
}

#[test]
fn a_batch_of_rows_is_byte_identical_to_the_wgsl_chain() {
    let g = gpu_core::testgpu::dev(pipelines());
    if !is_cuda(&g) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    for rows in [2u32, 3, 8] {
        for layout in [Rows::Independent, Rows::OneSequence] {
            check_rows(&g, rows, layout, 0xbeef ^ u64::from(rows));
        }
    }
}
