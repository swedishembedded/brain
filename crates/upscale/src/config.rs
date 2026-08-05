// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RRDBNet shape, DERIVED from the checkpoint rather than hardcoded.
//!
//! Real-ESRGAN ships several variants over one architecture (`x4plus` at
//! `num_block 23`, `x4plus_anime_6B` at 6, `x2plus` at `scale 2`), and they
//! differ only in numbers every one of which is recoverable from the tensor
//! shapes. Reading them is strictly better than a table of magic constants that
//! silently mis-builds the next variant someone points at it.

use std::collections::HashMap;

/// One `RRDBNet`'s shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RrdbConfig {
    /// Input/output image channels (`3`).
    pub in_channels: u32,
    pub out_channels: u32,
    /// Trunk width — `conv_first`'s output (`num_feat`, 64).
    pub num_feat: u32,
    /// Growth width inside a dense block (`num_grow_ch`, 32).
    pub num_grow_ch: u32,
    /// Number of `RRDB` blocks in the trunk (`num_block`, 23 for x4plus).
    pub num_block: u32,
    /// Spatial upscale, `4` or `2`. Counted from how many `conv_up*` the
    /// checkpoint carries: the reference does one nearest-2x per `conv_up`.
    pub scale: u32,
}

impl RrdbConfig {
    /// Derive from `name -> shape`. Returns `Err` naming the first thing that
    /// does not look like an RRDBNet, so a wrong checkpoint fails HERE rather
    /// than as a shape mismatch 700 tensors later.
    pub fn from_tensors(shapes: &HashMap<String, Vec<usize>>) -> Result<RrdbConfig, String> {
        let get = |n: &str| shapes.get(n).ok_or_else(|| format!("rrdb: no `{n}` in checkpoint"));

        // conv_first: [num_feat, in_channels, 3, 3]
        let cf = get("conv_first.weight")?;
        if cf.len() != 4 || cf[2] != 3 || cf[3] != 3 {
            return Err(format!("rrdb: conv_first.weight is {cf:?}, expected [F, Cin, 3, 3]"));
        }
        let num_feat = cf[0] as u32;
        let in_channels = cf[1] as u32;

        // conv_last: [out_channels, num_feat, 3, 3]
        let cl = get("conv_last.weight")?;
        let out_channels = cl[0] as u32;

        // A dense block's first conv is [num_grow_ch, num_feat, 3, 3].
        let g = get("body.0.rdb1.conv1.weight")?;
        let num_grow_ch = g[0] as u32;
        if g[1] as u32 != num_feat {
            return Err(format!(
                "rrdb: body.0.rdb1.conv1 takes {} channels but conv_first emits {num_feat}",
                g[1]
            ));
        }

        // num_block: the highest `body.<i>.` index, +1. Counted from the keys so
        // a 6-block anime checkpoint derives 6 without a second code path.
        let num_block = shapes
            .keys()
            .filter_map(|k| k.strip_prefix("body.")?.split('.').next()?.parse::<u32>().ok())
            .max()
            .ok_or("rrdb: no `body.<i>.*` tensors")?
            + 1;

        // scale: one nearest-2x per conv_up. x4plus has conv_up1+conv_up2 -> 4;
        // x2plus has conv_up1 alone -> 2.
        let ups = (1..)
            .take_while(|i| shapes.contains_key(&format!("conv_up{i}.weight")))
            .count() as u32;
        if ups == 0 {
            return Err("rrdb: no `conv_up1.weight` — not an RRDBNet".into());
        }
        let scale = 1 << ups;

        Ok(RrdbConfig { in_channels, out_channels, num_feat, num_grow_ch, num_block, scale })
    }

    /// Every tensor the forward reads, in graph order. `import::validate` checks
    /// the checkpoint against exactly this list, so a missing or mis-shaped
    /// weight is named at load rather than mis-bound at dispatch.
    pub fn param_list(&self) -> Vec<(String, Vec<usize>)> {
        let (f, g) = (self.num_feat as usize, self.num_grow_ch as usize);
        let mut v: Vec<(String, Vec<usize>)> = Vec::new();
        let conv = |name: String, cout: usize, cin: usize, out: &mut Vec<(String, Vec<usize>)>| {
            out.push((format!("{name}.weight"), vec![cout, cin, 3, 3]));
            out.push((format!("{name}.bias"), vec![cout]));
        };
        conv("conv_first".into(), f, self.in_channels as usize, &mut v);
        for b in 0..self.num_block as usize {
            for r in 1..=3 {
                for c in 1..=5 {
                    // Dense growth: conv<c> sees the block input plus every
                    // earlier conv's output.
                    let cin = f + (c - 1) * g;
                    let cout = if c == 5 { f } else { g };
                    conv(format!("body.{b}.rdb{r}.conv{c}"), cout, cin, &mut v);
                }
            }
        }
        conv("conv_body".into(), f, f, &mut v);
        for i in 1..=self.scale.trailing_zeros() as usize {
            conv(format!("conv_up{i}"), f, f, &mut v);
        }
        conv("conv_hr".into(), f, f, &mut v);
        conv("conv_last".into(), self.out_channels as usize, f, &mut v);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x4plus_shapes() -> HashMap<String, Vec<usize>> {
        let cfg = RrdbConfig {
            in_channels: 3,
            out_channels: 3,
            num_feat: 64,
            num_grow_ch: 32,
            num_block: 23,
            scale: 4,
        };
        cfg.param_list().into_iter().collect()
    }

    /// `param_list` and `from_tensors` are inverses: the shapes the config says
    /// it needs must derive back to the same config. That is what makes
    /// "derived, not hardcoded" checkable rather than a claim in a comment.
    #[test]
    fn derive_round_trips_through_param_list() {
        let want = RrdbConfig {
            in_channels: 3,
            out_channels: 3,
            num_feat: 64,
            num_grow_ch: 32,
            num_block: 23,
            scale: 4,
        };
        assert_eq!(RrdbConfig::from_tensors(&x4plus_shapes()).unwrap(), want);
    }

    /// The anime variant differs ONLY in num_block, and the x2 variant only in
    /// how many conv_up it carries — both must derive, or the "several variants
    /// over one architecture" claim is false.
    #[test]
    fn the_other_released_variants_derive_too() {
        let anime = RrdbConfig { num_block: 6, ..RrdbConfig::from_tensors(&x4plus_shapes()).unwrap() };
        let s: HashMap<_, _> = anime.param_list().into_iter().collect();
        assert_eq!(RrdbConfig::from_tensors(&s).unwrap().num_block, 6);

        let x2 = RrdbConfig { scale: 2, ..anime };
        let s: HashMap<_, _> = x2.param_list().into_iter().collect();
        let got = RrdbConfig::from_tensors(&s).unwrap();
        assert_eq!((got.scale, got.num_block), (2, 6));
    }

    /// A checkpoint that is not an RRDBNet must be named at derive time.
    #[test]
    fn a_foreign_checkpoint_is_rejected_by_name() {
        let mut s = x4plus_shapes();
        s.remove("conv_up1.weight");
        assert!(RrdbConfig::from_tensors(&s).unwrap_err().contains("conv_up1"));

        let empty: HashMap<String, Vec<usize>> = HashMap::new();
        assert!(RrdbConfig::from_tensors(&empty).unwrap_err().contains("conv_first"));
    }

    /// The dense widths are the whole point of the block: conv<c> must see the
    /// block input plus every earlier conv's output, so a growth-width bug is a
    /// shape mismatch here rather than a silently wrong picture.
    #[test]
    fn dense_block_input_widths_grow_by_num_grow_ch() {
        let cfg = RrdbConfig::from_tensors(&x4plus_shapes()).unwrap();
        let s: HashMap<_, _> = cfg.param_list().into_iter().collect();
        for (c, want_cin) in [(1, 64), (2, 96), (3, 128), (4, 160), (5, 192)] {
            let k = format!("body.0.rdb1.conv{c}.weight");
            assert_eq!(s[&k][1], want_cin, "{k} input width");
        }
        assert_eq!(s["body.0.rdb1.conv5.weight"][0], 64, "conv5 returns to the trunk width");
    }
}
