// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The ops the plan runs on the CPU between NPU segments, in place on native16
//! tensors.
//!
//! Every loop works on whole 16-byte atoms (one pixel's 16 lanes), so on a
//! Cortex-A7 built with NEON each atom is one vector operation. Padding lanes
//! past a tensor's real channels are processed too; nothing reads them.

use crate::layout::{Shape, LANES};

type Atom = [u8; LANES];

fn atoms(bytes: &[u8]) -> &[Atom] {
    bytes.as_chunks::<LANES>().0
}

fn atoms_mut(bytes: &mut [u8]) -> &mut [Atom] {
    bytes.as_chunks_mut::<LANES>().0
}

/// Fixed-point fraction bits of [`add`]'s operand scales.
pub const ADD_FRAC_BITS: u32 = 16;

/// `out = round(((a - za) * scale_a + (b - zb) * scale_b) / 2^16) + zo`,
/// saturated to int8. The scales are the operands' scales over the output's,
/// in Q16.
#[allow(clippy::too_many_arguments)]
pub fn add(a: &[u8], b: &[u8], out: &mut [u8], za: i8, zb: i8, zo: i8, scale_a: i32, scale_b: i32) {
    let (za, zb, zo) = (i32::from(za), i32::from(zb), i32::from(zo));
    let half = 1i32 << (ADD_FRAC_BITS - 1);
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        let v = (((i32::from(x as i8) - za) * scale_a + (i32::from(y as i8) - zb) * scale_b + half) >> ADD_FRAC_BITS) + zo;
        *o = v.clamp(-128, 127) as i8 as u8;
    }
}

fn max_into(acc: &mut Atom, v: &Atom) {
    for (a, &b) in acc.iter_mut().zip(v) {
        *a = (*a as i8).max(b as i8) as u8;
    }
}

/// Square max pooling of one tensor `si` -> `so`, separably: a horizontal
/// pass into `scratch` (resized as needed), then a vertical one. Padding never
/// wins the max.
pub fn max_pool(input: &[u8], si: Shape, output: &mut [u8], so: Shape, kernel: usize, stride: usize, pad: usize, scratch: &mut Vec<Atom>) {
    if (kernel, stride, pad) == (3, 1, 1) {
        return max_pool_3x3_same(input, si, output, scratch);
    }
    let window = |o: usize, len: usize| (o * stride).saturating_sub(pad)..(o * stride + kernel - pad).min(len);
    scratch.resize(si.h * so.w, [0; LANES]);
    for plane in 0..si.planes() {
        let src = atoms(&input[plane * si.plane_bytes()..][..si.h * si.w * LANES]);
        for y in 0..si.h {
            let row = &src[y * si.w..(y + 1) * si.w];
            for ox in 0..so.w {
                let mut m = [i8::MIN as u8; LANES];
                for v in &row[window(ox, si.w)] {
                    max_into(&mut m, v);
                }
                scratch[y * so.w + ox] = m;
            }
        }
        let dst = atoms_mut(&mut output[plane * so.plane_bytes()..][..so.h * so.w * LANES]);
        for oy in 0..so.h {
            for ox in 0..so.w {
                let mut m = [i8::MIN as u8; LANES];
                for y in window(oy, si.h) {
                    max_into(&mut m, &scratch[y * so.w + ox]);
                }
                dst[oy * so.w + ox] = m;
            }
        }
    }
}

/// The 3x3, stride-1, padding-1 pool (SPPF's), without window bookkeeping:
/// each output takes its neighbours directly, edges clamped.
fn max_pool_3x3_same(input: &[u8], s: Shape, output: &mut [u8], scratch: &mut Vec<Atom>) {
    let (h, w) = (s.h, s.w);
    scratch.resize(h * w, [0; LANES]);
    let max3 = |a: &Atom, b: &Atom, c: &Atom| {
        let mut m = *a;
        max_into(&mut m, b);
        max_into(&mut m, c);
        m
    };
    for plane in 0..s.planes() {
        let src = atoms(&input[plane * s.plane_bytes()..][..h * w * LANES]);
        for y in 0..h {
            let row = &src[y * w..(y + 1) * w];
            for x in 0..w {
                scratch[y * w + x] = max3(&row[x.saturating_sub(1)], &row[x], &row[(x + 1).min(w - 1)]);
            }
        }
        let dst = atoms_mut(&mut output[plane * s.plane_bytes()..][..h * w * LANES]);
        for y in 0..h {
            let (up, down) = (y.saturating_sub(1), (y + 1).min(h - 1));
            for x in 0..w {
                dst[y * w + x] = max3(&scratch[up * w + x], &scratch[y * w + x], &scratch[down * w + x]);
            }
        }
    }
}

/// Nearest-neighbour 2x upsampling: each input row's atoms doubled into an
/// output row, which is then copied to the row below it.
pub fn upsample2x(input: &[u8], si: Shape, output: &mut [u8], so: Shape) {
    for plane in 0..si.planes() {
        let src = atoms(&input[plane * si.plane_bytes()..][..si.h * si.w * LANES]);
        let dst = atoms_mut(&mut output[plane * so.plane_bytes()..][..so.h * so.w * LANES]);
        for y in 0..si.h {
            let (top, bottom) = dst[2 * y * so.w..(2 * y + 2) * so.w].split_at_mut(so.w);
            for (pair, atom) in top.as_chunks_mut::<2>().0.iter_mut().zip(&src[y * si.w..(y + 1) * si.w]) {
                *pair = [*atom, *atom];
            }
            bottom.copy_from_slice(top);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(s: Shape, f: impl Fn(usize, usize, usize) -> i8) -> Vec<u8> {
        let hwc: Vec<i8> = (0..s.h * s.w * s.c).map(|i| f(i / s.c / s.w, i / s.c % s.w, i % s.c)).collect();
        let mut buf = vec![0u8; s.bytes()];
        s.pack(&hwc, &mut buf);
        buf
    }

    #[test]
    fn max_pool_matches_the_direct_definition() {
        let si = Shape::new(7, 6, 20);
        let input = tensor(si, |y, x, c| ((y * 37 + x * 11 + c * 5) % 256) as i8);
        for (k, s, p) in [(3, 1, 1), (2, 2, 0), (3, 2, 1)] {
            let so = Shape::new((si.h + 2 * p - k) / s + 1, (si.w + 2 * p - k) / s + 1, si.c);
            let mut out = vec![0u8; so.bytes()];
            max_pool(&input, si, &mut out, so, k, s, p, &mut Vec::new());
            let got = so.unpack(&out);
            let x = si.unpack(&input);
            for oy in 0..so.h {
                for ox in 0..so.w {
                    for c in 0..si.c {
                        let mut want = i8::MIN;
                        for y in (oy * s) as isize - p as isize..(oy * s + k) as isize - p as isize {
                            for xx in (ox * s) as isize - p as isize..(ox * s + k) as isize - p as isize {
                                if (0..si.h as isize).contains(&y) && (0..si.w as isize).contains(&xx) {
                                    want = want.max(x[(y as usize * si.w + xx as usize) * si.c + c]);
                                }
                            }
                        }
                        assert_eq!(got[(oy * so.w + ox) * so.c + c], want, "k{k} s{s} p{p} at ({oy},{ox},{c})");
                    }
                }
            }
        }
    }

    #[test]
    fn upsample_repeats_every_pixel_twice_each_way() {
        let si = Shape::new(3, 4, 18);
        let so = Shape::new(6, 8, 18);
        let input = tensor(si, |y, x, c| (y * 40 + x * 7 + c) as i8);
        let mut out = vec![0u8; so.bytes()];
        upsample2x(&input, si, &mut out, so);
        let (x, got) = (si.unpack(&input), so.unpack(&out));
        for y in 0..so.h {
            for xx in 0..so.w {
                for c in 0..so.c {
                    assert_eq!(got[(y * so.w + xx) * so.c + c], x[((y / 2) * si.w + xx / 2) * si.c + c]);
                }
            }
        }
    }
}
