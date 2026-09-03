// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M13: the gate for the `matmul_kq_gemv` -> `matmul_kq_gemv_reg` transparent
//! upgrade (`gpu_core::upgrade`), on real hardware. The affine K-quant twin of
//! `i8_gemv_reg_upgrade.rs`, and it holds the same claim for the same reason:
//! every model in this tree that already registers `matmul_kq_gemv#CODE_BITS=
//! {4,8}` (`model::ops::kernel_list`, `qwen3`/`qwen35`/`qwen35moe`'s own
//! pipelines) inherits the register-accumulator substitution with ZERO call-
//! site changes, so there is no place a reviewer would see a tolerance being
//! introduced - the two kernels' raw output BITS are asserted equal.
//!
//! Two `CODE_BITS` specialisations, each with its OWN bucket ladder
//! (`gpu_core::upgrade::UPGRADES` carries two separate rows for exactly this
//! reason - see that module's own doc comment on the `extra` field), so this
//! file drives both independently rather than assuming one implies the other.

use gpu_core::Gpu;
use kernels::template::interned;

/// Deterministic signed activation codes in `[-100, 100]`.
fn rand_i8_codes(seed: u64, n: usize) -> Vec<i8> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| (r.next_u32() % 201) as i32 as i8 - 100).collect()
}

/// Unsigned affine weight codes in `[0, min(2^bits, 32))` - real Q4_K/Q5_K
/// codes never set the top bits of their `CODE_BITS` slot (`matmul_kq_gemv.
/// wgsl`'s own header), so an un-capped generator would corrupt roughly half
/// the `CODE_BITS=8` codes by setting the sign bit `dot4I8Packed` reads.
fn rand_unsigned_codes(seed: u64, n: usize, bits: u32) -> Vec<i32> {
    let span = (1u32 << bits).min(32);
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| (r.next_u32() % span) as i32).collect()
}

fn rand_pos_f32(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| lo + (r.next_u32() % 100_000) as f32 / 100_000.0 * (hi - lo)).collect()
}

fn pack_i8_words(codes: &[i8]) -> Vec<u32> {
    assert_eq!(codes.len() % 4, 0);
    codes.chunks_exact(4).map(|c| c.iter().enumerate().fold(0u32, |w, (b, &v)| w | ((v as u8 as u32) << (8 * b)))).collect()
}

fn pack_affine_words(codes: &[i32], bits: u32) -> Vec<u32> {
    let per_word = 32 / bits as usize;
    assert_eq!(codes.len() % per_word, 0);
    let mask = (1u32 << bits) - 1;
    codes
        .chunks_exact(per_word)
        .map(|c| c.iter().enumerate().fold(0u32, |w, (b, &v)| w | ((v as u32 & mask) << (bits * b as u32))))
        .collect()
}

/// M14: groups per `wd` super-block entry - `matmul_kq_dyn.wgsl`'s own fixed
/// `GPS=8` (256-element GGUF Q4_K/Q5_K super-block / 32-element group).
/// `wd_words_per_row = ceil(ng/GPS)` generalises correctly even when `ng <
/// GPS` (this file's `k=96` shape has `ng=3`), so no shape restriction is
/// needed here beyond `matmul_kq_gemv`'s own `k` a multiple of 32.
const GPS: usize = 8;

fn pack_wsm(sc: &[u8], mn: &[u8], n: usize, ng: usize) -> Vec<u32> {
    let words = ng.div_ceil(2);
    let mut out = vec![0u32; n * words];
    for r in 0..n {
        for g in 0..ng {
            let (w, shift) = (g / 2, if g % 2 == 0 { 0u32 } else { 16u32 });
            out[r * words + w] |= (sc[r * ng + g] as u32) << shift;
            out[r * words + w] |= (mn[r * ng + g] as u32) << (shift + 8);
        }
    }
    out
}

fn pack_wd(d: &[f32], dmin: &[f32], n: usize, spb_per_row: usize) -> Vec<u32> {
    let mut out = vec![0u32; n * spb_per_row];
    for r in 0..n {
        for s in 0..spb_per_row {
            let db = half::f16::from_f32(d[r * spb_per_row + s]).to_bits() as u32;
            let dmb = half::f16::from_f32(dmin[r * spb_per_row + s]).to_bits() as u32;
            out[r * spb_per_row + s] = db | (dmb << 16);
        }
    }
    out
}

/// Random `(sc, mn, d_super, dmin_super)` - `crates/model/tests/matmul_kq.
/// rs`'s own `rand_kq_scale`, restated here (this crate has no dependency on
/// `model`'s test-only helpers).
fn rand_kq_scale(seed: u64, n: usize, ng: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>) {
    let spb = ng.div_ceil(GPS);
    let sc: Vec<u8> = rand_pos_f32(seed ^ 0x54, n * ng, 1.0, 63.0).into_iter().map(|v| v as u8).collect();
    let mn: Vec<u8> = rand_pos_f32(seed ^ 0x55, n * ng, 1.0, 63.0).into_iter().map(|v| v as u8).collect();
    let d_super = rand_pos_f32(seed ^ 0x56, n * spb, 0.001, 0.02);
    let dmin_super = rand_pos_f32(seed ^ 0x57, n * spb, 0.002, 0.04);
    (sc, mn, d_super, dmin_super)
}

fn host_group_sums(codes_x: &[i8], m: usize, k: usize, group: usize) -> Vec<f32> {
    let ng = k / group;
    let mut out = vec![0f32; m * ng];
    for r in 0..m {
        for g in 0..ng {
            let s: i32 = codes_x[r * k + g * group..r * k + (g + 1) * group].iter().map(|&v| v as i32).sum();
            out[r * ng + g] = s as f32;
        }
    }
    out
}

const GROUP: usize = 32;

/// Uploads one affine problem and runs it through kernel `name` (either
/// `matmul_kq_gemv#CODE_BITS=B` or a `matmul_kq_gemv_reg#CODE_BITS=B,MREG=R`
/// bucket - both take the identical `Params{m,k,n}`/binding contract).
#[allow(clippy::too_many_arguments)]
fn run(g: &Gpu, name: &str, m: usize, k: usize, n: usize, bits: u32, seed: u64) -> Vec<f32> {
    let ng = k / GROUP;
    let codes_x = rand_i8_codes(seed ^ 0x51, m * k);
    let sx = rand_pos_f32(seed ^ 0x52, m, 0.3, 1.7);
    let codes_w = rand_unsigned_codes(seed ^ 0x53, n * k, bits);
    let (sc, mn, d_super, dmin_super) = rand_kq_scale(seed, n, ng);

    let xq_words = pack_i8_words(&codes_x);
    let xq = g.storage(xq_words.len() as u64);
    g.write(&xq, &xq_words);
    let wq_words = pack_affine_words(&codes_w, bits);
    let wq = g.storage(wq_words.len() as u64);
    g.write(&wq, &wq_words);
    let sxb = g.storage_init("sx", &sx);
    let wsm_words = pack_wsm(&sc, &mn, n, ng);
    let wsm = g.storage(wsm_words.len() as u64);
    g.write(&wsm, &wsm_words);
    let wd_words = pack_wd(&d_super, &dmin_super, n, ng.div_ceil(GPS));
    let wd = g.storage(wd_words.len() as u64);
    g.write(&wd, &wd_words);
    let xgs = g.storage_init("xgs", &host_group_sums(&codes_x, m, k, GROUP));
    let out = g.storage((m * n) as u64);

    let idx = g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"));
    let params = [m as u32, k as u32, n as u32];
    g.submit(&[], &[g.step(idx, &[&xq, &wq, &sxb, &wsm, &wd, &xgs, &out], &params, n as u32 * 64)]);
    g.poll_wait();
    g.read(&out, m * n)
}

fn gemv_name(kernels_list: &mut Vec<(&'static str, &'static str)>, bits: u32) -> &'static str {
    let (n, s) = interned("matmul_kq_gemv", kernels::MATMUL_KQ_GEMV, &[("CODE_BITS", bits)]).unwrap();
    kernels_list.push((n, s));
    n
}

/// The transparent upgrade is ACTIVE, with the full `MREG` bucket ladder, at
/// EACH `CODE_BITS` independently - without this, the bit-identity test below
/// would pass trivially on a handle where the upgrade never fired.
#[test]
fn the_upgrade_is_active_for_both_code_bits_independently() {
    let mut kernels_list = Vec::new();
    let kq4 = gemv_name(&mut kernels_list, 4);
    let kq8 = gemv_name(&mut kernels_list, 8);
    let g = Gpu::new(&kernels_list);
    if !g.caps().workgroup_reductions || !g.caps().numeric.int8_dot {
        brain_testutil::skip_unavailable("matmul_kq_gemv needs workgroup reductions and a packed int8 dot");
        return;
    }
    let idx4 = g.kernel_index(kq4).unwrap();
    let idx8 = g.kernel_index(kq8).unwrap();
    assert_eq!(
        g.physical_kernel_names(idx4),
        vec![
            "matmul_kq_gemv_reg#CODE_BITS=4,MREG=1",
            "matmul_kq_gemv_reg#CODE_BITS=4,MREG=2",
            "matmul_kq_gemv_reg#CODE_BITS=4,MREG=4",
            "matmul_kq_gemv_reg#CODE_BITS=4,MREG=8",
            "matmul_kq_gemv_reg#CODE_BITS=4,MREG=16",
            "matmul_kq_gemv_reg#CODE_BITS=4,MREG=32",
        ],
        "the CODE_BITS=4 (Q4_K) bucket ladder must be active on a device that selects it"
    );
    assert_eq!(
        g.physical_kernel_names(idx8),
        vec![
            "matmul_kq_gemv_reg#CODE_BITS=8,MREG=1",
            "matmul_kq_gemv_reg#CODE_BITS=8,MREG=2",
            "matmul_kq_gemv_reg#CODE_BITS=8,MREG=4",
            "matmul_kq_gemv_reg#CODE_BITS=8,MREG=8",
            "matmul_kq_gemv_reg#CODE_BITS=8,MREG=16",
            "matmul_kq_gemv_reg#CODE_BITS=8,MREG=32",
        ],
        "the CODE_BITS=8 (Q5_K) bucket ladder must be active independently of CODE_BITS=4"
    );
}

/// BYTE-identical at every row count the kernel supports, both `CODE_BITS` -
/// the two kernels perform the identical operations in the identical order by
/// construction (see `matmul_kq_gemv_reg.wgsl`'s own header), so this is a
/// real `assert_eq!` on the bits, not a tolerance.
#[test]
fn the_register_kernel_is_byte_identical_to_the_plain_gemv() {
    for &bits in &[4u32, 8u32] {
        let mut kernels_list = Vec::new();
        // Registering ONLY the plain kernel is deliberate - the bucket ladder
        // below must arrive via the TRANSPARENT upgrade (`Gpu::new` expands
        // it internally), not a manually assembled list, so this test proves
        // the real end-to-end path a model actually gets.
        let plain = gemv_name(&mut kernels_list, bits);
        let g = Gpu::new(&kernels_list);
        if !g.caps().workgroup_reductions || !g.caps().numeric.int8_dot {
            brain_testutil::skip_unavailable("matmul_kq_gemv_reg needs workgroup reductions and a packed int8 dot");
            return;
        }
        // k spans several 64-wide strides at more than one weight-scale group
        // count; n is not a multiple of anything.
        for (k, n) in [(256usize, 384usize), (96, 129)] {
            for &m in &[1u32, 2, 3, 8, 17, 32] {
                let reg_name = GEMV_MREG_BUCKETS
                    .iter()
                    .find(|&&b| m <= b)
                    .map(|&b| format!("matmul_kq_gemv_reg#CODE_BITS={bits},MREG={b}"))
                    .unwrap();
                let got_plain = run(&g, plain, m as usize, k, n, bits, 0xE0_0000 | (bits as u64) << 8 | m as u64);
                let got_reg = run(&g, &reg_name, m as usize, k, n, bits, 0xE0_0000 | (bits as u64) << 8 | m as u64);
                let bitpat = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(
                    bitpat(&got_plain),
                    bitpat(&got_reg),
                    "CODE_BITS={bits} m={m} k={k} n={n}: matmul_kq_gemv_reg must be BYTE-identical to \
                     matmul_kq_gemv - both form and fold the same per-quad terms in the same order with \
                     the same one-thread-per-group min-correction guard, so a difference here is a real \
                     defect, not rounding"
                );
            }
        }
    }
}

/// Mirrors `gpu_core::upgrade::GEMV_MREG_BUCKETS` (private to that module) -
/// duplicated here rather than exposed, matching `i8_gemv_reg_upgrade.rs`'s
/// own precedent of a test-local `MAX_ROWS` constant for the same reason: an
/// integration test crate cannot see a crate-private `const`.
const GEMV_MREG_BUCKETS: &[u32] = &[1, 2, 4, 8, 16, 32];
