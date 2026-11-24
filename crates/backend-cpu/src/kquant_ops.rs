// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native CPU GEMMs for the packed-quant weight tiers the Cranelift JIT
//! cannot compile (`@cpu no` kernels: `matmul_i8_dyn` and its `#QPG=1`
//! group-16 template, `matmul_q4_dyn_reg`, `matmul_kq_dyn`). Before this
//! module the only int8-family native path was the symmetric group-32 GEMM
//! reached through `register_native`/the hidden if-ladder's *absent* arm —
//! `numeric.int8_dot` stayed `false` on this backend and every quantized
//! checkpoint promoted to fp32 at load (a 2.3 GiB Q4_K_M GGUF became a
//! ~16 GiB fp32 plan that fits no machine). With these routes registered,
//! `int8_dot` is true on the CPU and a K-quant checkpoint serves from its
//! own bytes.
//!
//! Math contracts are the WGSL kernels' own, validated here against the
//! same f64 host oracle `crates/model/tests/it/matmul_kq.rs` uses.

#[cfg(target_arch = "aarch64")]
use crate::fast_ops::dot32_i8_neon;
use crate::fast_ops::{Int8IsaTier, dot32_i8_avx2, dot32_i8_avx512vnni, dot_group_scalar};
use crate::par;

/// One group's 32 weight codes (or 16, at group=16 the second half stays
/// zero) as the 8 packed u32 words the `dot32_i8_*` family takes.
fn pack_words(codes: &[u8]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for (wi, w) in out.iter_mut().enumerate() {
        *w = u32::from_le_bytes([codes[wi * 4], codes[wi * 4 + 1], codes[wi * 4 + 2], codes[wi * 4 + 3]]);
    }
    out
}

fn dot32(tier: Int8IsaTier, x: &[u32], w: &[u32; 8]) -> i32 {
    match tier {
        #[cfg(target_arch = "x86_64")]
        Int8IsaTier::Avx512Vnni => unsafe { dot32_i8_avx512vnni(x, w) },
        #[cfg(target_arch = "x86_64")]
        Int8IsaTier::Avx2 => unsafe { dot32_i8_avx2(x, w) },
        #[cfg(target_arch = "aarch64")]
        Int8IsaTier::Neon => unsafe { dot32_i8_neon(x, w) },
        Int8IsaTier::Scalar => dot_group_scalar(x, w),
    }
}

/// Unpack `count` codes of `bits` bits each, low bits first, from
/// `words[start..]` into `out` (zero-padded past `count`).
fn unpack_codes(words: &[u32], start: usize, count: usize, bits: u32, out: &mut [u8; 32]) {
    let mask = (1u32 << bits) - 1;
    for (i, b) in out.iter_mut().enumerate() {
        *b = if i < count {
            let wi = start + i / (32 / bits as usize);
            let sh = ((i % (32 / bits as usize)) * bits as usize) as u32;
            ((words[wi] >> sh) & mask) as u8
        } else {
            0
        };
    }
}

/// The native CPU analogue of `matmul_i8_dyn.wgsl` (symmetric per-group
/// weight scales, group 32) and of its `#QPG=1` group-16 template: `params`
/// are `[m, kg, n]` with `kg = k/4` packed activation words per row, `sw`
/// holds `k/group` f32 scales per output row, and `group` is 32 or 16.
///
/// `out[m,n] = Σ_g A[m,n,g] · sw[n,g] · sx[m,g]` with `A` the exact integer
/// dot of the group's packed int8 codes on both sides.
#[allow(clippy::too_many_arguments)]
pub fn matmul_i8_dyn_g(xq: &[u32], wq: &[u32], sx: &[f32], sw: &[f32], out: &mut [f32], m: usize, kg: usize, n: usize, group: usize) {
    if m == 0 || n == 0 || kg == 0 {
        return;
    }
    assert!(matches!(group, 16 | 32), "matmul_i8_dyn_g: group={group} is not a symmetric int8 group width");
    let words_per_group = group / 4; // packed activation words per group
    let ng = kg / words_per_group;
    let tier = Int8IsaTier::current();
    let codes_per_word = 32 / group; // weight codes share the activation packing? no:
                                     // weights are packed like activations (4 i8/word)
                                     // for the symmetric family, so a group is
                                     // `group/4` words on both sides.
    let _ = codes_per_word;
    let whole = |mi: usize, orow: &mut [f32]| {
        let xrow = &xq[mi * kg..(mi + 1) * kg];
        let sxrow = &sx[mi * ng..(mi + 1) * ng];
        for ni in 0..n {
            let wrow = &wq[ni * kg..(ni + 1) * kg];
            let swrow = &sw[ni * ng..(ni + 1) * ng];
            let mut acc = 0.0f32;
            for g in 0..ng {
                let off = g * words_per_group;
                let wpg = words_per_group;
                let mut xg8 = [0u32; 8];
                xg8[..wpg].copy_from_slice(&xrow[off..off + wpg]);
                let wg = &wrow[off..off + wpg];
                let a = dot32(tier, &xg8, &pack_words_slice(wg));
                acc += a as f32 * swrow[g] * sxrow[g];
            }
            orow[ni] = acc;
        }
    };
    for mi in 0..m {
        whole(mi, &mut out[mi * n..(mi + 1) * n]);
    }
}

fn pack_words_slice(w: &[u32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    out[..w.len()].copy_from_slice(w);
    out
}

/// The native CPU analogue of `matmul_kq_dyn.wgsl` (the affine K-quant GEMM,
/// GGUF **Q4_K** at `bits=4`, Q5_K at `bits=8`):
///
/// `out[m,n] = Σ_g ( A[m,n,g]·ds[n,g] − dm[n,g]·S[m,g] ) · sx[m,g]`
///
/// * `A[m,n,g]` — exact 32-lane integer dot of the packed int8 activation
///   codes against that group's unsigned weight codes (every code < 2^bits
///   ≤ 128 keeps bit 7 clear, so the signed dot applies unchanged).
/// * `S[m,g]` — signed sum of the group's 32 activation codes, written to
///   `xgs[m, g]` (the caller's `[m, k/32]` scratch; each row's slice is
///   that row's alone).
/// * `ds/dm` decode from `wsm: [n, ceil(k/32/2)]` (two groups' `(sc, m)`
///   byte pairs per word, low half first) and `wd: [n, k/256]` (one f16
///   `(d, dmin)` pair per 256-element super-block) by the same f16
///   bit-pattern decode the WGSL kernel's `kq_ds_dm` performs.
///
/// Buffers follow the WGSL kernel's binding order `[xq, wq, wsm, wd, xgs,
/// sx, out]` with params `[m, k, n]` — RAW logical `k`, not a packed word
/// count. The integer inner sum is order-free; the f32 fold runs in the
/// ascending group order the reference uses.
#[allow(clippy::too_many_arguments)]
pub fn matmul_kq_dyn(
    xq: &[u32], wq: &[u32], wsm: &[u32], wd: &[u32], xgs: &mut [f32], sx: &[f32],
    out: &mut [f32], m: usize, k: usize, n: usize, bits: u32,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    assert!(k % 32 == 0, "matmul_kq_dyn: k={k} is not a multiple of the 32-element weight-scale group");
    assert!(k % 256 == 0, "matmul_kq_dyn: k={k} is not a multiple of the 256-element Q4_K super-block (the wd layout is one (d, dmin) f16 pair per 8 groups)");
    assert!(matches!(bits, 4 | 8), "matmul_kq_dyn: bits={bits} is not a K-quant slot width (4 or 8)");
    let ng = k / 32; // weight-scale groups per row
    let xwords = k / 4; // packed activation words per row
    let wwords = k * bits as usize / 32; // packed weight words per row
    let cpw = 32 / bits as usize; // weight codes per word
    let wsm_r = ng.div_ceil(2);
    let wd_r = ng / 8; // GPS = 8 groups per super-block word
    let tier = Int8IsaTier::current();
    // f16 decode by bit pattern — `kq_ds_dm`'s magic-constant construction.
    fn f16(h: u32) -> f32 {
        let exp = (h >> 10) & 0x1F;
        let sub = f32::from_bits(0x3880_0000 | ((h & 0x3FF) << 13)) - f32::from_bits(0x3880_0000);
        let nor = f32::from_bits((h & 0x7FFF) << 13) * f32::from_bits(0x7780_0000);
        let inn = f32::from_bits(0x7F80_0000 | ((h & 0x3FF) << 13));
        let mag = if exp == 0 { sub } else if exp == 31 { inn } else { nor };
        if h & 0x8000 != 0 { -mag } else { mag }
    }
    let mut one_row = |mi: usize, orow: &mut [f32]| {
        let xrow = &xq[mi * xwords..(mi + 1) * xwords];
        let sxrow = &sx[mi * ng..(mi + 1) * ng];
        let xgrow = &mut xgs[mi * ng..(mi + 1) * ng];
        // S[m,g]: the signed sum of the group's 32 activation codes —
        // `quant_group_sum.wgsl`'s exact integer.
        for g in 0..ng {
            let mut s = 0i32;
            for w in 0..8usize {
                let word = xrow[g * 8 + w];
                s += (word as i8) as i32
                    + ((word >> 8) as i8) as i32
                    + ((word >> 16) as i8) as i32
                    + ((word >> 24) as i8) as i32;
            }
            xgrow[g] = s as f32;
        }
        for ni in 0..n {
            let wrow = &wq[ni * wwords..(ni + 1) * wwords];
            let wsm_row = &wsm[ni * wsm_r..(ni + 1) * wsm_r];
            let wd_row = &wd[ni * wd_r..(ni + 1) * wd_r];
            let mut acc = 0.0f32;
            for g in 0..ng {
                let mut wcode = [0u8; 32];
                unpack_codes(wrow, g * (32 / cpw), 32, bits, &mut wcode);
                let xg = &xrow[g * 8..g * 8 + 8];
                let a = dot32(tier, xg, &pack_words(&wcode));
                let sword = wsm_row[g / 2];
                let shift = if g % 2 == 1 { 16u32 } else { 0u32 };
                let sc = ((sword >> shift) & 0xFF) as f32;
                let mv = ((sword >> (shift + 8)) & 0xFF) as f32;
                let wword = wd_row[g / 8];
                let ds = f16(wword & 0xFFFF) * sc;
                let dm = f16(wword >> 16) * mv;
                acc += (a as f32 * ds - dm * xgrow[g]) * sxrow[g];
            }
            orow[ni] = acc;
        }
    };
    if m <= 1 {
        // Decode regime: the weight planes are read once per token, so the
        // fan-out axis is the output columns (one item per column). The
        // closure below only READS xgs, so the single row's S sums are
        // computed here first.
        for g in 0..ng {
            let mut sm = 0i32;
            for w in 0..8 {
                let word = xq[g * 8 + w];
                sm += (word as i8) as i32
                    + ((word >> 8) as i8) as i32
                    + ((word >> 16) as i8) as i32
                    + ((word >> 24) as i8) as i32;
            }
            xgs[g] = sm as f32;
        }
        par::items_mut(out, 1, ng * 32 * 2, |ni, cell| {
            let wrow = &wq[ni * wwords..(ni + 1) * wwords];
            let wsm_row = &wsm[ni * wsm_r..(ni + 1) * wsm_r];
            let wd_row = &wd[ni * wd_r..(ni + 1) * wd_r];
            let mut acc = 0.0f32;
            for g in 0..ng {
                let mut wcode = [0u8; 32];
                unpack_codes(wrow, g * (32 / cpw), 32, bits, &mut wcode);
                let a = dot32(tier, &xq[g * 8..g * 8 + 8], &pack_words(&wcode));
                let sword = wsm_row[g / 2];
                let shift = if g % 2 == 1 { 16u32 } else { 0u32 };
                let sc = ((sword >> shift) & 0xFF) as f32;
                let mv = ((sword >> (shift + 8)) & 0xFF) as f32;
                let wword = wd_row[g / 8];
                acc += (a as f32 * f16(wword & 0xFFFF) * sc - f16(wword >> 16) * mv * xgs[g]) * sx[g];
            }
            cell[0] = acc;
        });
        return;
    }
    for mi in 0..m {
        one_row(mi, &mut out[mi * n..(mi + 1) * n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The f64 host oracle `matmul_kq.rs` uses, inlined: affine K-quant.
    fn oracle_affine(cx: &[i8], sx: &[f32], cw: &[i32], ds: &[f32], dm: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let ng = k / 32;
        let mut out = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0f64;
                for g in 0..ng {
                    let (mut a, mut s) = (0i64, 0i64);
                    for j in 0..32 {
                        let kk = g * 32 + j;
                        a += i64::from(cw[c * k + kk]) * i64::from(cx[r * k + kk]);
                        s += i64::from(cx[r * k + kk]);
                    }
                    acc += f64::from(sx[r * ng + g]) * (a as f64 * ds[c * ng + g] as f64 - s as f64 * dm[c * ng + g] as f64);
                }
                out[r * n + c] = acc as f32;
            }
        }
        out
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        fn code(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i32
        }
    }

    fn pack_i8_words(codes: &[i8]) -> Vec<u32> {
        codes.chunks(4).map(|c| u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])).collect()
    }

    fn pack_unsigned(codes: &[i32], bits: u32) -> Vec<u32> {
        let per = 32 / bits as usize;
        codes.chunks(per).map(|c| {
            let mut w = 0u32;
            for (i, v) in c.iter().enumerate() {
                w |= (*v as u32) << (i * bits as usize);
            }
            w
        }).collect()
    }

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

    // f16<->f32 by bit pattern (no `half` dependency here): encoder is the
    // standard exponent/mantissa repack, decoder mirrors `kq_ds_dm`'s
    // magic-constant construction. Test values stay in the normal range.
    fn f16_bits(f: f32) -> u32 {
        let b = f.to_bits();
        let sign = ((b >> 16) & 0x8000) as u32;
        let mag = b & 0x7FFF_FFFF;
        if mag == 0 { return sign; }
        let e = ((mag >> 23) as i32) - 127 + 15;
        if e >= 31 { return sign | 0x7C00; }
        if e <= 0 {
            let m = (mag & 0x7F_FFFF) | 0x80_0000;
            return sign | ((m >> (14 - e)) as u32);
        }
        sign | (((e as u32) << 10) | ((mag >> 13) & 0x3FF))
    }
    fn f16_from_bits(h: u32) -> f32 {
        let exp = (h >> 10) & 0x1F;
        let sub = f32::from_bits(0x3880_0000 | ((h & 0x3FF) << 13)) - f32::from_bits(0x3880_0000);
        let nor = f32::from_bits((h & 0x7FFF) << 13) * f32::from_bits(0x7780_0000);
        let inn = f32::from_bits(0x7F80_0000 | ((h & 0x3FF) << 13));
        let mag = if exp == 0 { sub } else if exp == 31 { inn } else { nor };
        if h & 0x8000 != 0 { -mag } else { mag }
    }
    fn pack_wd(d: &[f32], dmin: &[f32], n: usize, words_per_row: usize) -> Vec<u32> {
        let mut out = vec![0u32; n * words_per_row];
        for r in 0..n {
            for s in 0..words_per_row {
                let db = f16_bits(d[r * words_per_row + s]);
                let dmb = f16_bits(dmin[r * words_per_row + s]);
                out[r * words_per_row + s] = db | (dmb << 16);
            }
        }
        out
    }

    #[test]
    fn native_kq_dyn_matches_the_f64_oracle_on_random_blocks() {
        let (m, k, n) = (3usize, 256usize, 5usize);
        let mut rng = Lcg(0x5EED_1234);
        let cx: Vec<i8> = (0..m * k).map(|_| rng.code(-100, 100) as i8).collect();
        let cw: Vec<i32> = (0..n * k).map(|_| rng.code(0, 15)).collect();
        let ng = k / 32;
        let sx: Vec<f32> = (0..m * ng).map(|_| 0.5 + (rng.next() % 1000) as f32 / 1000.0).collect();
        let sc: Vec<u8> = (0..n * ng).map(|_| rng.code(1, 200) as u8).collect();
        let mn: Vec<u8> = (0..n * ng).map(|_| rng.code(0, 60) as u8).collect();
        let spb = ng / 8;
        let d: Vec<f32> = (0..n * spb).map(|_| 0.01 + (rng.next() % 100) as f32 / 100.0).collect();
        let dmin: Vec<f32> = (0..n * spb).map(|_| (rng.next() % 2) as f32 * 0.3).collect();
        let xq = pack_i8_words(&cx);
        let wq = pack_unsigned(&cw, 4);
        let wsm = pack_wsm(&sc, &mn, n, ng);
        let wd = pack_wd(&d, &dmin, n, spb);
        // effective per-group ds/dm from the packed pieces (the test file's own decoding)
        let mut ds = vec![0f32; n * ng];
        let mut dm = vec![0f32; n * ng];
        for c in 0..n {
            for g in 0..ng {
                let wword = wd[c * spb + g / 8];
                let dsv = f16_from_bits(wword & 0xFFFF);
                let dmv = f16_from_bits(wword >> 16);
                ds[c * ng + g] = dsv * sc[c * ng + g] as f32;
                dm[c * ng + g] = dmv * mn[c * ng + g] as f32;
            }
        }
        let want = oracle_affine(&cx, &sx, &cw, &ds, &dm, m, k, n);
        let mut got = vec![0f32; m * n];
        let mut xgs = vec![0f32; m * ng];
        matmul_kq_dyn(&xq, &wq, &wsm, &wd, &mut xgs, &sx, &mut got, m, k, n, 4);
        for i in 0..m * n {
            let scale = want[i].abs().max(1.0);
            assert!((got[i] - want[i]).abs() / scale < 2e-3, "out[{i}] {} vs {} (rel)", got[i], want[i]);
        }
    }

    #[test]
    fn native_symmetric_i8_matches_the_oracle_at_both_group_widths() {
        for group in [32usize, 16usize] {
            let (m, k, n) = (2usize, 64usize, 4usize);
            let mut rng = Lcg(0xC0DE + group as u64);
            let kg = k / 4;
            let ng = k / group;
            let cx: Vec<i8> = (0..m * k).map(|_| rng.code(-120, 120) as i8).collect();
            let cw: Vec<i8> = (0..n * k).map(|_| rng.code(-120, 120) as i8).collect();
            let sx: Vec<f32> = (0..m * ng).map(|_| 0.5 + (rng.next() % 500) as f32 / 500.0).collect();
            let sw: Vec<f32> = (0..n * ng).map(|_| 0.5 + (rng.next() % 500) as f32 / 500.0).collect();
            let xq = pack_i8_words(&cx);
            let wq = pack_i8_words(&cw);
            let mut got = vec![0f32; m * n];
            matmul_i8_dyn_g(&xq, &wq, &sx, &sw, &mut got, m, kg, n, group);
            for r in 0..m {
                for c in 0..n {
                    let mut want = 0f64;
                    for g in 0..ng {
                        let mut a = 0i64;
                        for j in 0..group {
                            let kk = g * group + j;
                            a += i64::from(cw[c * k + kk]) * i64::from(cx[r * k + kk]);
                        }
                        want += a as f64 * sw[c * ng + g] as f64 * sx[r * ng + g] as f64;
                    }
                    let got_v = got[r * n + c];
                    assert!((got_v - want as f32).abs() / (want.abs() as f32).max(1.0) < 1e-5, "group={group} [{r},{c}] {got_v} vs {want}");
                }
            }
        }
    }
}
