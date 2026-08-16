// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight-free tiny-config smoke test (porting-playbook §4): every step kind
//! the real encoder dispatches, at toy dims, in well under a second — so
//! buffer-sizing and binding bugs surface here instead of opaquely inside a
//! 24-block run against 19 GB of weights.
//!
//! It also pins the one composition that is pure plumbing rather than learned
//! math: the relative-position bias slab is `embed(bucket_ids, rel_bias)`
//! followed by a `(q,k,H) -> (H,q,k)` permute, so with a rel_bias table of
//! `bucket*10 + head` the resulting `[H, T, T]` buffer must be exactly
//! `bucket(i,j)*10 + h` at every element. That is checked against
//! `hostbias::buckets`, which `tests/parity.rs` in turn gates against the
//! reference's own bucket table.

use std::collections::HashMap;

use t5encoder::config::T5Config;
use t5encoder::model::{T5Encoder, Tap};

/// XXL topology at toy dims. Every width is a multiple of 64 floats, the
/// 256-byte minimum storage-binding offset alignment.
///
/// The dims deliberately BREAK the two coincidences T5-XXL hides — at XXL
/// `heads == d_kv == 64` and `heads * d_kv == 4096 == d_model`, so swapping
/// `heads` for `d_kv` in a Params list, or using `d_model` where the attention
/// inner width belongs, is invisible in the real-weights gate. Here
/// `heads * d_kv = 128 != d_model = 64` and `heads = 2 != d_kv = 64`, which
/// makes every such confusion a size mismatch instead.
fn tiny() -> T5Config {
    T5Config { vocab: 256, d_model: 64, d_ff: 128, d_kv: 64, layers: 2, heads: 2, ..T5Config::xxl() }
}

/// Deterministic small weights: a per-tensor hash keeps every parameter
/// distinct (a constant fill hides transposes and offset bugs).
fn fake_init(cfg: &T5Config) -> HashMap<String, Vec<f32>> {
    cfg.tensor_manifest()
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let seed = name.bytes().fold(7u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
            let v = (0..n)
                .map(|i| {
                    let h = (seed ^ (i as u32).wrapping_mul(2654435761)) % 1000;
                    (h as f32 / 1000.0 - 0.5) * 0.1
                })
                .collect();
            (name, v)
        })
        .collect()
}

/// Force the embedding gather to tile, so the multi-tile `embed_tile` path
/// (sliced bindings, not one whole-table binding) is exercised at toy size too.
///
/// `set_var` mutates process-global state that `block::tile_budget_words`
/// reads, and libtest runs the tests in this binary on CONCURRENT THREADS — a
/// racing `setenv`/`getenv` pair is a real data race, not a theoretical one.
/// The `Once` is what makes it sound: every test in this file calls this as its
/// FIRST action, so a test that arrives while the write is in flight blocks
/// until it completes, and no environment read can overlap the single write.
fn force_tiled_embedding() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the only `set_var` in this binary, serialised by `ONCE`, and every
    // test funnels through here before any code reads the environment.
    ONCE.call_once(|| unsafe { std::env::set_var("BRAIN_TILE_BUDGET_WORDS", "4096") });
}

#[test]
fn tiny_forward_runs_every_step_kind() {
    force_tiled_embedding();
    let cfg = tiny();
    let (b, t) = (2u32, 8u32);
    assert!(model::block::vocab_tiles(cfg.vocab as u64, cfg.d_model as u64).len() > 1);

    let init = fake_init(&cfg);
    let m = T5Encoder::new_on(gpu_core::testgpu::dev(t5encoder::model::PIPELINES), cfg.clone(), b, t, &init);
    let ids: Vec<u32> = (0..b * t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    m.set_tokens(&ids);
    m.forward();

    let n = (b * t) as usize;
    let d = cfg.d_model as usize;
    let finite = |what: &str, v: &[f32]| {
        assert!(v.iter().all(|x| x.is_finite()), "{what}: non-finite value");
        assert!(v.iter().any(|&x| x != 0.0), "{what}: all zero");
    };
    finite("embed", &m.read_x(0));
    for l in 0..cfg.layers as usize {
        for tap in [
            Tap::AttnNorm,
            Tap::Qkv,
            Tap::Ctx,
            Tap::AttnOut,
            Tap::AttnRes,
            Tap::FfNorm,
            Tap::Wi0,
            Tap::Wi1,
            Tap::Gated,
            Tap::FfOut,
        ] {
            finite(&format!("block{l} {tap:?}"), &m.read_block_tap(l, tap));
        }
        finite(&format!("block{l}_out"), &m.read_x(l + 1));
    }
    let hidden = m.read_hidden();
    assert_eq!(hidden.len(), n * d);
    finite("last_hidden_state", &hidden);

    // The residual is a plain add: block output == res + ff_out, exactly.
    let res = m.read_block_tap(0, Tap::AttnRes);
    let ff = m.read_block_tap(0, Tap::FfOut);
    let out = m.read_x(1);
    for i in 0..n * d {
        assert_eq!(out[i], res[i] + ff[i], "residual at {i}");
    }
}

#[test]
fn position_bias_is_the_bucket_gather_permuted() {
    force_tiled_embedding();
    let cfg = tiny();
    let (b, t) = (1u32, 8u32);
    let mut init = fake_init(&cfg);
    // rel_bias[bucket, head] = bucket*10 + head — every element identifiable.
    let heads = cfg.heads as usize;
    init.insert(
        "rel_bias.weight".into(),
        (0..cfg.rel_buckets as usize)
            .flat_map(|bk| (0..heads).map(move |h| (bk * 10 + h) as f32))
            .collect(),
    );
    let m = T5Encoder::new_on(gpu_core::testgpu::dev(t5encoder::model::PIPELINES), cfg.clone(), b, t, &init);
    m.set_tokens(&vec![1u32; (b * t) as usize]);
    m.forward();

    let got = m.read_position_bias();
    let buckets = t5encoder::hostbias::buckets(t, cfg.rel_buckets, cfg.rel_max_distance);
    assert_eq!(got.len(), heads * (t * t) as usize);
    for h in 0..heads {
        for i in 0..t as usize {
            for j in 0..t as usize {
                let want = (buckets[i * t as usize + j] as usize * 10 + h) as f32;
                let idx = (h * t as usize + i) * t as usize + j;
                assert_eq!(got[idx], want, "bias[h={h}, i={i}, j={j}]");
            }
        }
    }
}

// -------------------------------------------------------------- umT5 (Wan)

/// umT5 topology at the same toy dims: per-block relative bias, masked.
fn tiny_umt5() -> T5Config {
    T5Config {
        vocab: 256,
        d_model: 64,
        d_ff: 128,
        d_kv: 64,
        layers: 3,
        heads: 2,
        ..T5Config::umt5_xxl()
    }
}

/// The gate for the delta that is otherwise SILENT: umT5 owns one
/// `T5RelativeEmbedding` per block (`shared_pos=False`), so block `l`'s bias
/// slab must be built from block `l`'s OWN table. With each table filled by
/// `bucket*10 + head + 100*l`, a port that shares block 0's table produces a
/// perfectly plausible slab that is off by a constant per block - which is
/// exactly what makes the bug survive an eyeball and a rough cosine.
#[test]
fn each_block_uses_its_own_relative_position_table() {
    force_tiled_embedding();
    let cfg = tiny_umt5();
    let (b, t) = (1u32, 8u32);
    let heads = cfg.heads as usize;
    let mut init = fake_init(&cfg);
    for l in 0..cfg.layers as usize {
        init.insert(
            cfg.rel_bias_name(l),
            (0..cfg.rel_buckets as usize)
                .flat_map(|bk| (0..heads).map(move |h| (bk * 10 + h + 100 * l) as f32))
                .collect(),
        );
    }
    let m = T5Encoder::new_on(gpu_core::testgpu::dev(t5encoder::model::PIPELINES), cfg.clone(), b, t, &init);
    m.set_tokens(&vec![1u32; (b * t) as usize]);
    m.set_mask(&vec![1u32; (b * t) as usize]);
    m.forward();

    let buckets = t5encoder::hostbias::buckets(t, cfg.rel_buckets, cfg.rel_max_distance);
    for l in 0..cfg.layers as usize {
        let got = m.read_block_bias(l);
        assert_eq!(got.len(), heads * (t * t) as usize);
        for h in 0..heads {
            for i in 0..t as usize {
                for j in 0..t as usize {
                    let want = (buckets[i * t as usize + j] as usize * 10 + h + 100 * l) as f32;
                    let idx = (h * t as usize + i) * t as usize + j;
                    assert_eq!(got[idx], want, "block {l} bias[h={h}, i={i}, j={j}]");
                }
            }
        }
    }
}

/// Masking a right-padded run must be **the same computation** as never having
/// had the padding: nothing in this encoder mixes rows except attention, and
/// the mask removes the pad keys from every query. So a `T=12` run whose last 4
/// keys are masked and a `T=8` run of the same tokens must agree on all 8
/// content rows, and the masked keys contribute a hard zero rather than a small
/// number, so the agreement is BIT-EXACT rather than merely close.
///
/// This is the weight-free gate on the mask. It would catch a mask applied on
/// the query axis, applied after the softmax, or dropped in some blocks - none
/// of which a cosine on a real prompt reliably separates from rounding.
#[test]
fn masking_the_pad_keys_equals_never_having_padded() {
    force_tiled_embedding();
    let cfg = tiny_umt5();
    let (b, keep, pad) = (2u32, 8u32, 4u32);
    let init = fake_init(&cfg);
    let dev = || gpu_core::testgpu::dev(t5encoder::model::PIPELINES);
    let content: Vec<u32> = (0..b * keep).map(|i| (i * 7 + 3) % cfg.vocab).collect();

    let short = T5Encoder::new_on(dev(), cfg.clone(), b, keep, &init);
    short.set_tokens(&content);
    short.set_mask(&vec![1u32; (b * keep) as usize]);
    short.forward();

    let t = keep + pad;
    let mut ids = vec![0u32; (b * t) as usize];
    let mut mask = vec![0u32; (b * t) as usize];
    for r in 0..b as usize {
        for i in 0..keep as usize {
            ids[r * t as usize + i] = content[r * keep as usize + i];
            mask[r * t as usize + i] = 1;
        }
    }
    let long = T5Encoder::new_on(dev(), cfg.clone(), b, t, &init);
    long.set_tokens(&ids);
    long.set_mask(&mask);
    long.forward();

    let d = cfg.d_model as usize;
    let a = short.read_hidden();
    let c = long.read_hidden();
    let mut worst = 0.0f32;
    for r in 0..b as usize {
        for i in 0..keep as usize {
            for k in 0..d {
                let x = a[(r * keep as usize + i) * d + k];
                let y = c[(r * t as usize + i) * d + k];
                worst = worst.max((x - y).abs());
            }
        }
    }
    eprintln!("masked T={t} vs unmasked T={keep}: max_abs {worst:.3e} on the content rows");
    assert_eq!(worst, 0.0, "the key-padding mask is not equivalent to truncation");

    // ...and `read_context` hands the DiT hard zeros at the pad rows, not the
    // encoder's (perfectly nonzero) output there.
    let ctx = long.read_context();
    for r in 0..(b * t) as usize {
        let row = &ctx[r * d..(r + 1) * d];
        if mask[r] == 1 {
            assert!(row.iter().any(|&v| v != 0.0), "content row {r} is zero");
        } else {
            assert!(row.iter().all(|&v| v == 0.0), "pad row {r} is not zeroed");
            assert!(c[r * d..(r + 1) * d].iter().any(|&v| v != 0.0), "pad row {r} was already zero");
        }
    }
}
