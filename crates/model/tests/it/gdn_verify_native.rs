// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding - drafters, verifiers
// and the state rollback a recurrent stack needs - for clients serving large
// models on modest hardware. If your team needs expertise in verifying several
// tokens per sequence through a Gated DeltaNet layer without losing the ability
// to undo a rejection, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The native `gdn_verify_pool` / `gdn_commit_pool` kernels
//! (`cu/gdn_decode.cu`) against plain single-step decoding.
//!
//! A verify launch carries several consecutive rows per sequence of a batch,
//! each sequence from its own pool row. The claim, byte for byte: the layer
//! output of every row, the pool state after each sequence's LAST row, and the
//! state and conv window after every earlier row (the snapshots a rejection
//! restores) are exactly what repeated single decode steps produce - and
//! committing a snapshot puts that state back over the pool row, leaving every
//! other row of the pool alone.
//!
//! The same claim over a pool that stores bf16: plain decode rounds the state and the conv window to bf16 after every
//! token, so a verified row must too (the snapshots are the f32 widening of those bf16 values), and then greedy
//! speculation is token for token plain greedy decoding. And over an f32 pool whose plain decode is the streaming
//! recurrence (not the exact body): the verify runs that recurrence's own body, so each row is that plain step.

use data::rng::{random_seed, Lcg};
use model::gdn::GdnShape;
use model::gdn_mixer::{gdn_mixer_decode_fused, gdn_mixer_decode_pooled_fused, gdn_mixer_decode_pooled_qkvz, GdnMixerShape, GdnMixerWeights, GdnPoolRows, GdnStream};
use model::gdn_verify::{gdn_commit_pooled, gdn_mixer_verify_pooled_fused, gdn_mixer_verify_pooled_qkvz, GdnRowSnapshots, VerifyLayout};

fn rnd(r: &mut Lcg, n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|_| r.scaled(s)).collect()
}

fn bits(v: Vec<f32>) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn is_cuda(g: &gpu_core::Gpu) -> bool {
    g.kind() == "cuda" && g.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

/// Whether this device is offered the GDN fused kernels: a CUDA device at their floor (tensor-core int8
/// and `cp.async`, compute capability 8.0), native kernels not withheld.
fn offers_gdn(g: &gpu_core::Gpu) -> bool {
    is_cuda(g) && g.has_fused(gpu_core::Fused::GdnDecode)
}

fn u32s(g: &gpu_core::Gpu, v: &[u32]) -> gpu_core::DeviceBuffer {
    let b = g.storage(v.len().max(1) as u64);
    g.write(&b, v);
    b
}

/// One verify case: key heads, group, rows per sequence, pool rows, pool row of each sequence.
type GdnCase<'a> = (u32, u32, &'a [u32], u32, &'a [u32]);

/// bf16 rows as words (two to a word, the even element low) and back.
fn round_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn pack_bf16(v: &[f32]) -> Vec<f32> {
    v.chunks(2).map(|p| f32::from_bits(u32::from(round_bf16(p[0])) | u32::from(round_bf16(p[1])) << 16)).collect()
}

fn unpack_bf16(w: &[f32]) -> Vec<f32> {
    w.iter().flat_map(|x| [f32::from_bits(x.to_bits() << 16), f32::from_bits(x.to_bits() & 0xFFFF_0000)]).collect()
}

#[test]
fn verify_rows_and_snapshots_equal_repeated_single_steps_and_commit_restores_them() {
    verify_equals_single_steps(false, false);
}

/// Over bf16 pools the reference is the plain pooled decode (the streaming kernel a bf16 pool takes), one step a row.
#[test]
fn verify_rows_over_bf16_pools_equal_repeated_plain_decode_steps() {
    verify_equals_single_steps(true, true);
}

/// Over f32 pools with the occupancy candidates on (the serving default) the reference is the plain pooled decode, which
/// streams: it agrees with the exact body only to f32 reassociation, so a verify that ran the exact body would drift from the
/// plain decode it must reproduce.
#[test]
fn verify_rows_over_f32_pools_equal_repeated_plain_streaming_steps() {
    verify_equals_single_steps(false, true);
}

/// `stream`: the pool is the serving default (`wide`), and the reference of a row is the plain pooled decode launch;
/// off, the exact body (an f32 pool) is the reference.
fn verify_equals_single_steps(bf16: bool, stream: bool) {
    let g = gpu_core::testgpu::dev(&[]);
    if !offers_gdn(&g) {
        brain_testutil::skip_unavailable("the GDN fused kernels need a CUDA device at their floor (compute capability 8.0)");
        return;
    }
    let (dk, dv, kw) = (128u32, 128u32, 4u32);
    // (key heads, group, rows per sequence, pool rows, pool row of each sequence)
    let cases: [GdnCase; 4] =
        [(1, 1, &[1], 2, &[1]), (2, 3, &[3, 1, 5], 5, &[3, 0, 2]), (4, 2, &[8, 2, 1, 4], 6, &[5, 1, 0, 3]), (16, 3, &[2, 2], 3, &[2, 0])];
    for (case, &(nkh, group, counts, pool_rows, slots)) in cases.iter().enumerate() {
        let nvh = nkh * group;
        let b = counts.len() as u32;
        let n: u32 = counts.iter().sum();
        let shape = GdnMixerShape { gdn: GdnShape { b: 1, h: nvh, t: 1, dk, dv, chunk: 1 }, nkh, conv_kernel: kw, rms_eps: 1e-6 };
        let (value_dim, conv_dim) = (shape.value_dim() as usize, shape.conv_dim() as usize);
        let state_len = (nvh * dk * dv) as usize;
        let hist_len = conv_dim * (kw as usize - 1);
        let ctx = format!("case {case}: nkh={nkh} group={group} counts={counts:?}");

        let mut r = Lcg::new(random_seed());
        let conv_w = g.storage_init("conv_w", &rnd(&mut r, conv_dim * kw as usize, 0.5));
        let a_log = g.storage_init("a_log", &rnd(&mut r, nvh as usize, 0.5));
        let dt_bias = g.storage_init("dt_bias", &rnd(&mut r, nvh as usize, 0.5));
        let norm_w = g.storage_init("norm_w", &rnd(&mut r, dv as usize, 0.5));
        let ones = g.storage_init("ones", &vec![1.0f32; dk as usize]);
        let w = GdnMixerWeights { conv1d_weight: &conv_w, a_log: &a_log, dt_bias: &dt_bias, norm_weight: &norm_w, ones_khd: &ones };

        let nn = n as usize;
        let mixed = rnd(&mut r, nn * conv_dim, 1.0);
        let bp = rnd(&mut r, nn * nvh as usize, 1.0);
        let ap = rnd(&mut r, nn * nvh as usize, 1.0);
        let z = rnd(&mut r, nn * value_dim, 1.0);
        let (state0, hist0) = (rnd(&mut r, pool_rows as usize * state_len, 0.3), rnd(&mut r, pool_rows as usize * hist_len, 0.3));
        // Elements as the pool stores them: f32, or bf16 words (the starting values are made bf16-exact).
        let (state0, hist0) = if bf16 { (unpack_bf16(&pack_bf16(&state0)), unpack_bf16(&pack_bf16(&hist0))) } else { (state0, hist0) };
        let store = |name: &str, v: &[f32]| if bf16 { g.storage_init(name, &pack_bf16(v)) } else { g.storage_init(name, v) };
        let load = |buf: &gpu_core::DeviceBuffer, n: usize| if bf16 { unpack_bf16(&g.read(buf, n / 2)) } else { g.read(buf, n) };

        // Reference: every sequence stepped one row at a time on a private copy
        // of its pool row, recording the state after each row.
        let mut ref_gated = vec![0.0f32; nn * value_dim];
        let mut ref_snap_state = vec![0.0f32; nn * state_len];
        let mut ref_snap_hist = vec![0.0f32; nn * hist_len];
        let mut ref_final = (state0.clone(), hist0.clone());
        let mut off = 0usize;
        let zero_row = u32s(&g, &[0]);
        for (si, &cnt) in counts.iter().enumerate() {
            let slot = slots[si] as usize;
            let st = store("state", &state0[slot * state_len..(slot + 1) * state_len]);
            let hi = store("hist", &hist0[slot * hist_len..(slot + 1) * hist_len]);
            for row in 0..cnt as usize {
                let t = off + row;
                let slice = |v: &[f32], width: usize, name: &str| g.storage_init(name, &v[t * width..(t + 1) * width]);
                let (mixed_t, bp_t, ap_t, z_t) = (slice(&mixed, conv_dim, "mixed"), slice(&bp, nvh as usize, "bp"), slice(&ap, nvh as usize, "ap"), slice(&z, value_dim, "z"));
                let gated = if stream {
                    // The plain streaming decode: the pooled launch a serving pool takes (one sequence, pool row 0).
                    let pool = GdnPoolRows { state: &st, hist: &hi, rows: &zero_row, gather: 0, scatter: 0, fuse: true, bf16, wide: true };
                    gdn_mixer_decode_pooled_fused(&g, &shape, &w, &mixed_t, &bp_t, &ap_t, &z_t, &pool)
                } else {
                    gdn_mixer_decode_fused(&g, &shape, &w, &mixed_t, &bp_t, &ap_t, &z_t, &GdnStream { state: &st, hist: &hi })
                }
                .expect("the fused single-step kernel was declined on a CUDA device");
                ref_gated[t * value_dim..(t + 1) * value_dim].copy_from_slice(&g.read(&gated, value_dim));
                if row + 1 < cnt as usize {
                    ref_snap_state[t * state_len..(t + 1) * state_len].copy_from_slice(&load(&st, state_len));
                    ref_snap_hist[t * hist_len..(t + 1) * hist_len].copy_from_slice(&load(&hi, hist_len));
                }
            }
            ref_final.0[slot * state_len..(slot + 1) * state_len].copy_from_slice(&load(&st, state_len));
            ref_final.1[slot * hist_len..(slot + 1) * hist_len].copy_from_slice(&load(&hi, hist_len));
            off += cnt as usize;
        }

        // The batched verify launch.
        let (pst, phi) = (store("pool_state", &state0), store("pool_hist", &hist0));
        let rows_buf = u32s(&g, slots);
        let offs: Vec<u32> = counts.iter().scan(0u32, |acc, &c| { let o = *acc; *acc += c; Some(o) }).collect();
        let (counts_buf, offs_buf) = (u32s(&g, counts), u32s(&g, &offs));
        let pool = GdnPoolRows { state: &pst, hist: &phi, rows: &rows_buf, gather: 0, scatter: 0, fuse: true, bf16, wide: stream };
        let snaps = GdnRowSnapshots::new(&g, &shape, n);
        let layout = VerifyLayout { rows: &rows_buf, counts: &counts_buf, offs: &offs_buf, b };
        let gated = gdn_mixer_verify_pooled_fused(
            &g,
            &shape,
            &w,
            &g.storage_init("mixed", &mixed),
            &g.storage_init("bp", &bp),
            &g.storage_init("ap", &ap),
            &g.storage_init("z", &z),
            &pool,
            &layout,
            n,
            &snaps,
        )
        .expect("the verify kernel was declined on a CUDA device");
        g.poll_wait();

        assert_eq!(bits(g.read(&gated, nn * value_dim)), bits(ref_gated), "layer outputs differ: {ctx}");
        assert_eq!(bits(load(&pst, pool_rows as usize * state_len)), bits(ref_final.0.clone()), "pool state after the last row differs: {ctx}");
        assert_eq!(bits(load(&phi, pool_rows as usize * hist_len)), bits(ref_final.1.clone()), "pool window after the last row differs: {ctx}");
        let got_snap_state = g.read(&snaps.state, nn * state_len);
        let got_snap_hist = g.read(&snaps.hist, nn * hist_len);
        for (si, &cnt) in counts.iter().enumerate() {
            for row in 0..cnt as usize - 1 {
                let t = offs[si] as usize + row;
                assert_eq!(bits(got_snap_state[t * state_len..(t + 1) * state_len].to_vec()), bits(ref_snap_state[t * state_len..(t + 1) * state_len].to_vec()), "snapshot state of sequence {si} row {row}: {ctx}");
                assert_eq!(bits(got_snap_hist[t * hist_len..(t + 1) * hist_len].to_vec()), bits(ref_snap_hist[t * hist_len..(t + 1) * hist_len].to_vec()), "snapshot window of sequence {si} row {row}: {ctx}");
            }
        }

        // Commit: sequence 0 restores its first snapshot (when it has one), the
        // others are left as the launch left them.
        let mut sel = vec![u32::MAX; counts.len()];
        let mut want_state = ref_final.0.clone();
        let mut want_hist = ref_final.1.clone();
        if counts[0] > 1 {
            sel[0] = offs[0];
            let slot = slots[0] as usize;
            want_state[slot * state_len..(slot + 1) * state_len].copy_from_slice(&ref_snap_state[offs[0] as usize * state_len..(offs[0] as usize + 1) * state_len]);
            want_hist[slot * hist_len..(slot + 1) * hist_len].copy_from_slice(&ref_snap_hist[offs[0] as usize * hist_len..(offs[0] as usize + 1) * hist_len]);
        }
        let sel_buf = u32s(&g, &sel);
        let step = gdn_commit_pooled(&g, &shape, &pst, &phi, bf16, &snaps, &rows_buf, &sel_buf, b).expect("the commit kernel was declined on a CUDA device");
        g.submit(&[], &[step]);
        g.poll_wait();
        assert_eq!(bits(load(&pst, pool_rows as usize * state_len)), bits(want_state), "pool state after commit differs: {ctx}");
        assert_eq!(bits(load(&phi, pool_rows as usize * hist_len)), bits(want_hist), "pool window after commit differs: {ctx}");
    }
}

/// The verify launch reading `in_proj_qkv` and `in_proj_z` from the one `[tokens, conv_dim + value_dim]` product of a fused
/// projection leaves exactly the bits of the launch over the two separate arrays: the gated rows, the pool after the last
/// row, and every snapshot, for bf16 and f32 pools.
#[test]
fn verify_reads_qkv_and_z_in_place_from_one_fused_product() {
    let g = gpu_core::testgpu::dev(&[]);
    if !offers_gdn(&g) {
        brain_testutil::skip_unavailable("the GDN fused kernels need a CUDA device at their floor (compute capability 8.0)");
        return;
    }
    let (dk, dv, kw) = (128u32, 128u32, 4u32);
    let cases: [GdnCase; 3] = [(2, 3, &[3, 1, 5], 5, &[3, 0, 2]), (4, 2, &[8, 2, 1, 4], 6, &[5, 1, 0, 3]), (16, 3, &[2, 4, 3, 1], 5, &[4, 0, 2, 1])];
    for (case, (&(nkh, group, counts, pool_rows, slots), bf16)) in cases.iter().flat_map(|c| [(c, true), (c, false)]).enumerate() {
        let nvh = nkh * group;
        let b = counts.len() as u32;
        let n: u32 = counts.iter().sum();
        let shape = GdnMixerShape { gdn: GdnShape { b: 1, h: nvh, t: 1, dk, dv, chunk: 1 }, nkh, conv_kernel: kw, rms_eps: 1e-6 };
        let (value_dim, conv_dim) = (shape.value_dim() as usize, shape.conv_dim() as usize);
        let state_len = (nvh * dk * dv) as usize;
        let hist_len = conv_dim * (kw as usize - 1);
        let ctx = format!("case {case}: nkh={nkh} group={group} counts={counts:?} bf16={bf16}");
        let mut r = Lcg::new(random_seed());
        let conv_w = g.storage_init("conv_w", &rnd(&mut r, conv_dim * kw as usize, 0.5));
        let a_log = g.storage_init("a_log", &rnd(&mut r, nvh as usize, 0.5));
        let dt_bias = g.storage_init("dt_bias", &rnd(&mut r, nvh as usize, 0.5));
        let norm_w = g.storage_init("norm_w", &rnd(&mut r, dv as usize, 0.5));
        let ones = g.storage_init("ones", &vec![1.0f32; dk as usize]);
        let w = GdnMixerWeights { conv1d_weight: &conv_w, a_log: &a_log, dt_bias: &dt_bias, norm_weight: &norm_w, ones_khd: &ones };
        let nn = n as usize;
        let mixed = rnd(&mut r, nn * conv_dim, 1.0);
        let z = rnd(&mut r, nn * value_dim, 1.0);
        let bp = g.storage_init("bp", &rnd(&mut r, nn * nvh as usize, 1.0));
        let ap = g.storage_init("ap", &rnd(&mut r, nn * nvh as usize, 1.0));
        let qkvz: Vec<f32> = (0..nn).flat_map(|row| mixed[row * conv_dim..(row + 1) * conv_dim].iter().chain(&z[row * value_dim..(row + 1) * value_dim]).copied()).collect();
        let (state0, hist0) = (rnd(&mut r, pool_rows as usize * state_len, 0.3), rnd(&mut r, pool_rows as usize * hist_len, 0.3));
        let (state0, hist0) = if bf16 { (unpack_bf16(&pack_bf16(&state0)), unpack_bf16(&pack_bf16(&hist0))) } else { (state0, hist0) };
        let store = |name: &str, v: &[f32]| if bf16 { g.storage_init(name, &pack_bf16(v)) } else { g.storage_init(name, v) };
        let words = |n: usize| if bf16 { n / 2 } else { n };
        let rows_buf = u32s(&g, slots);
        let offs: Vec<u32> = counts.iter().scan(0u32, |acc, &c| { let o = *acc; *acc += c; Some(o) }).collect();
        let (counts_buf, offs_buf) = (u32s(&g, counts), u32s(&g, &offs));
        let layout = VerifyLayout { rows: &rows_buf, counts: &counts_buf, offs: &offs_buf, b };
        let run = |in_place: bool| {
            let (pst, phi) = (store("pool_state", &state0), store("pool_hist", &hist0));
            let pool = GdnPoolRows { state: &pst, hist: &phi, rows: &rows_buf, gather: 0, scatter: 0, fuse: true, bf16, wide: true };
            let snaps = GdnRowSnapshots::new(&g, &shape, n);
            let gated = if in_place {
                gdn_mixer_verify_pooled_qkvz(&g, &shape, &w, &g.storage_init("qkvz", &qkvz), &bp, &ap, &pool, &layout, n, &snaps)
            } else {
                gdn_mixer_verify_pooled_fused(&g, &shape, &w, &g.storage_init("mixed", &mixed), &bp, &ap, &g.storage_init("z", &z), &pool, &layout, n, &snaps)
            }
            .expect("the verify kernel was declined on a CUDA device");
            g.poll_wait();
            (
                bits(g.read(&gated, nn * value_dim)),
                bits(g.read(&pst, pool_rows as usize * words(state_len))),
                bits(g.read(&phi, pool_rows as usize * words(hist_len))),
                bits(g.read(&snaps.state, nn * state_len)),
                bits(g.read(&snaps.hist, nn * hist_len)),
            )
        };
        let (apart, together) = (run(false), run(true));
        assert_eq!(together.0, apart.0, "gated rows differ: {ctx}");
        assert_eq!(together.1, apart.1, "pool state differs: {ctx}");
        assert_eq!(together.2, apart.2, "pool window differs: {ctx}");
        assert_eq!(together.3, apart.3, "state snapshots differ: {ctx}");
        assert_eq!(together.4, apart.4, "window snapshots differ: {ctx}");
    }
}

/// Back-to-back launches of the streaming decode and verify kernels (bf16 and f32 pools) over a fused projection's rows, at the production head
/// counts and the decode batch widths: an uninstrumented timing run, since a fault of a shared-memory ring showed only under
/// back-to-back launches (`#[ignore]`: a stress run of seconds, then `dmesg | grep -i xid` stays empty).
///
/// ```text
/// BRAIN_DEVICE=gpu0 BRAIN_BACKEND=cuda make test/one P=brain-model T=gdn_verify_native ARGS="--ignored --nocapture qkvz_back_to_back_stress"
/// ```
#[test]
#[ignore]
fn qkvz_back_to_back_stress() {
    let g = gpu_core::testgpu::dev(&[]);
    if !offers_gdn(&g) {
        brain_testutil::skip_unavailable("the GDN fused kernels need a CUDA device at their floor (compute capability 8.0)");
        return;
    }
    let (dk, dv, kw, nkh, group, pool_rows) = (128u32, 128u32, 4u32, 16u32, 3u32, 64u32);
    let nvh = nkh * group;
    let shape = GdnMixerShape { gdn: GdnShape { b: 1, h: nvh, t: 1, dk, dv, chunk: 1 }, nkh, conv_kernel: kw, rms_eps: 1e-6 };
    let (value_dim, conv_dim) = (shape.value_dim() as usize, shape.conv_dim() as usize);
    let (state_len, hist_len) = ((nvh * dk * dv) as usize, conv_dim * (kw as usize - 1));
    let mut r = Lcg::new(random_seed());
    let conv_w = g.storage_init("conv_w", &rnd(&mut r, conv_dim * kw as usize, 0.5));
    let a_log = g.storage_init("a_log", &rnd(&mut r, nvh as usize, 0.5));
    let dt_bias = g.storage_init("dt_bias", &rnd(&mut r, nvh as usize, 0.5));
    let norm_w = g.storage_init("norm_w", &rnd(&mut r, dv as usize, 0.5));
    let ones = g.storage_init("ones", &vec![1.0f32; dk as usize]);
    let w = GdnMixerWeights { conv1d_weight: &conv_w, a_log: &a_log, dt_bias: &dt_bias, norm_weight: &norm_w, ones_khd: &ones };
    let pools: Vec<(bool, gpu_core::DeviceBuffer, gpu_core::DeviceBuffer)> = [true, false]
        .into_iter()
        .map(|bf16| {
            let (state, hist) = (rnd(&mut r, pool_rows as usize * state_len, 0.3), rnd(&mut r, pool_rows as usize * hist_len, 0.3));
            let (state, hist) = if bf16 { (pack_bf16(&state), pack_bf16(&hist)) } else { (state, hist) };
            (bf16, g.storage_init("pool_state", &state), g.storage_init("pool_hist", &hist))
        })
        .collect();
    let slots: Vec<u32> = (0..pool_rows).collect();
    let rows_buf = u32s(&g, &slots);
    for round in 0..40u32 {
        for b in [1u32, 4, 16, 33, 64] {
            let counts: Vec<u32> = (0..b).map(|i| 1 + (i + round) % 4).collect();
            let n: u32 = counts.iter().sum();
            let offs: Vec<u32> = counts.iter().scan(0u32, |acc, &c| { let o = *acc; *acc += c; Some(o) }).collect();
            let (counts_buf, offs_buf) = (u32s(&g, &counts), u32s(&g, &offs));
            let qkvz = g.storage_init("qkvz", &rnd(&mut r, n as usize * (conv_dim + value_dim), 1.0));
            let (bp, ap) = (g.storage_init("bp", &rnd(&mut r, n as usize * nvh as usize, 1.0)), g.storage_init("ap", &rnd(&mut r, n as usize * nvh as usize, 1.0)));
            let snaps = GdnRowSnapshots::new(&g, &shape, n);
            let layout = VerifyLayout { rows: &rows_buf, counts: &counts_buf, offs: &offs_buf, b };
            for (bf16, pst, phi) in &pools {
                let pool = GdnPoolRows { state: pst, hist: phi, rows: &rows_buf, gather: 0, scatter: 0, fuse: true, bf16: *bf16, wide: true };
                for _ in 0..8 {
                    let decode_shape = GdnMixerShape { gdn: GdnShape { b, ..shape.gdn }, ..shape };
                    let _ = gdn_mixer_decode_pooled_qkvz(&g, &decode_shape, &w, &qkvz, &bp, &ap, &pool).expect("streaming decode offered");
                    let _ = gdn_mixer_verify_pooled_qkvz(&g, &shape, &w, &qkvz, &bp, &ap, &pool, &layout, n, &snaps).expect("streaming verify offered");
                }
            }
            g.poll_wait();
        }
    }
}
