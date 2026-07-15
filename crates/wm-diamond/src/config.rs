// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DIAMOND denoiser configuration (the Atari-100k `InnerModel` of
//! resources/world-models/repos/diamond, NeurIPS 2024). Parameter names match
//! the reference `state_dict` with the `denoiser.inner_model.` prefix
//! stripped, so the importer is a prefix strip + full-coverage match.

/// Architecture of the conditional EDM UNet denoiser.
#[derive(Clone, Debug, PartialEq)]
pub struct DiamondConfig {
    pub img_channels: u32,
    /// Number of conditioning context frames (and actions).
    pub num_steps_conditioning: u32,
    pub cond_channels: u32,
    /// Residual blocks per level (down path; up path gets `+1`).
    pub depths: Vec<u32>,
    /// Channels per level (Atari: all 64).
    pub channels: Vec<u32>,
    /// Self-attention per level in d/u blocks (Atari: none; mid always has it).
    pub attn_depths: Vec<bool>,
    pub num_actions: u32,
    /// Frame height/width (Atari: 64; must be divisible by 2^(levels-1)).
    pub h: u32,
    pub w: u32,
    /// EDM sigma_data.
    pub sigma_data: f32,
    /// Offset-noise sigma folded into the conditioners.
    pub sigma_offset_noise: f32,
}

impl DiamondConfig {
    /// The published Atari-100k configuration (config/agent/default.yaml).
    pub fn atari(num_actions: u32) -> DiamondConfig {
        DiamondConfig {
            img_channels: 3,
            num_steps_conditioning: 4,
            cond_channels: 256,
            depths: vec![2, 2, 2, 2],
            channels: vec![64, 64, 64, 64],
            attn_depths: vec![false, false, false, false],
            num_actions,
            h: 64,
            w: 64,
            sigma_data: 0.5,
            sigma_offset_noise: 0.3,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\"arch\":\"diamond\",\"img_channels\":{},\"num_steps_conditioning\":{},",
                "\"cond_channels\":{},\"depths\":{:?},\"channels\":{:?},\"attn_depths\":{:?},",
                "\"num_actions\":{},\"h\":{},\"w\":{},\"sigma_data\":{},\"sigma_offset_noise\":{}}}"
            ),
            self.img_channels,
            self.num_steps_conditioning,
            self.cond_channels,
            self.depths,
            self.channels,
            self.attn_depths,
            self.num_actions,
            self.h,
            self.w,
            self.sigma_data,
            self.sigma_offset_noise,
        )
    }

    pub fn from_json(s: &str) -> Result<DiamondConfig, String> {
        let v: serde_json::Value =
            serde_json::from_str(s).map_err(|e| format!("diamond config: {e}"))?;
        let u = |k: &str| -> Result<u32, String> {
            v[k].as_u64().map(|x| x as u32).ok_or_else(|| format!("diamond config: missing {k}"))
        };
        let f = |k: &str| -> Result<f32, String> {
            v[k].as_f64().map(|x| x as f32).ok_or_else(|| format!("diamond config: missing {k}"))
        };
        let arr_u = |k: &str| -> Result<Vec<u32>, String> {
            v[k].as_array()
                .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect())
                .ok_or_else(|| format!("diamond config: missing {k}"))
        };
        let arr_b = |k: &str| -> Result<Vec<bool>, String> {
            v[k].as_array()
                .map(|a| a.iter().filter_map(|x| x.as_bool()).collect())
                .ok_or_else(|| format!("diamond config: missing {k}"))
        };
        Ok(DiamondConfig {
            img_channels: u("img_channels")?,
            num_steps_conditioning: u("num_steps_conditioning")?,
            cond_channels: u("cond_channels")?,
            depths: arr_u("depths")?,
            channels: arr_u("channels")?,
            attn_depths: arr_b("attn_depths")?,
            num_actions: u("num_actions")?,
            h: u("h")?,
            w: u("w")?,
            sigma_data: f("sigma_data")?,
            sigma_offset_noise: f("sigma_offset_noise")?,
        })
    }

    /// Levels in the UNet.
    pub fn levels(&self) -> usize {
        self.channels.len()
    }

    /// The full expected tensor-name list (reference names, prefix stripped),
    /// with shapes — the importer validates FULL coverage against this.
    pub fn param_list(&self) -> Vec<(String, Vec<usize>)> {
        let cc = self.cond_channels as usize;
        let ic = self.img_channels as usize;
        let na = self.num_actions as usize;
        let nsc = self.num_steps_conditioning as usize;
        let mut out: Vec<(String, Vec<usize>)> = vec![
            ("noise_emb.weight".into(), vec![1, cc / 2]),
            ("act_emb.0.weight".into(), vec![na, cc / nsc]),
            ("cond_proj.0.weight".into(), vec![cc, cc]),
            ("cond_proj.0.bias".into(), vec![cc]),
            ("cond_proj.2.weight".into(), vec![cc, cc]),
            ("cond_proj.2.bias".into(), vec![cc]),
        ];
        let c0 = self.channels[0] as usize;
        out.push(("conv_in.weight".into(), vec![c0, (nsc + 1) * ic, 3, 3]));
        out.push(("conv_in.bias".into(), vec![c0]));

        let resblock = |out: &mut Vec<(String, Vec<usize>)>,
                        prefix: String,
                        cin: usize,
                        cout: usize,
                        attn: bool| {
            if cin != cout {
                out.push((format!("{prefix}.proj.weight"), vec![cout, cin, 1, 1]));
                out.push((format!("{prefix}.proj.bias"), vec![cout]));
            }
            out.push((format!("{prefix}.norm1.linear.weight"), vec![2 * cin, cc]));
            out.push((format!("{prefix}.norm1.linear.bias"), vec![2 * cin]));
            out.push((format!("{prefix}.conv1.weight"), vec![cout, cin, 3, 3]));
            out.push((format!("{prefix}.conv1.bias"), vec![cout]));
            out.push((format!("{prefix}.norm2.linear.weight"), vec![2 * cout, cc]));
            out.push((format!("{prefix}.norm2.linear.bias"), vec![2 * cout]));
            out.push((format!("{prefix}.conv2.weight"), vec![cout, cout, 3, 3]));
            out.push((format!("{prefix}.conv2.bias"), vec![cout]));
            if attn {
                out.push((format!("{prefix}.attn.norm.norm.weight"), vec![cout]));
                out.push((format!("{prefix}.attn.norm.norm.bias"), vec![cout]));
                out.push((format!("{prefix}.attn.qkv_proj.weight"), vec![3 * cout, cout, 1, 1]));
                out.push((format!("{prefix}.attn.qkv_proj.bias"), vec![3 * cout]));
                out.push((format!("{prefix}.attn.out_proj.weight"), vec![cout, cout, 1, 1]));
                out.push((format!("{prefix}.attn.out_proj.bias"), vec![cout]));
            }
        };

        let n_lv = self.levels();
        for i in 0..n_lv {
            let c1 = self.channels[i.saturating_sub(1)] as usize;
            let c2 = self.channels[i] as usize;
            let n = self.depths[i] as usize;
            let attn = self.attn_depths[i];
            // Down: [c1] + [c2]*(n-1) -> [c2]*n.
            for r in 0..n {
                let cin = if r == 0 { c1 } else { c2 };
                resblock(&mut out, format!("unet.d_blocks.{i}.resblocks.{r}"), cin, c2, attn);
            }
        }
        // Mid: 2 resblocks at channels[last], always attention.
        let cl = *self.channels.last().unwrap() as usize;
        for r in 0..2 {
            resblock(&mut out, format!("unet.mid_blocks.resblocks.{r}"), cl, cl, true);
        }
        // Up: module order is REVERSED (deepest first). u_blocks[j] corresponds
        // to level i = n_lv-1-j with [2*c2]*n + [c1+c2] -> [c2]*n + [c1].
        for j in 0..n_lv {
            let i = n_lv - 1 - j;
            let c1 = self.channels[i.saturating_sub(1)] as usize;
            let c2 = self.channels[i] as usize;
            let n = self.depths[i] as usize;
            let attn = self.attn_depths[i];
            for r in 0..=n {
                let (cin, cout) =
                    if r < n { (2 * c2, c2) } else { (c1 + c2, c1) };
                resblock(&mut out, format!("unet.u_blocks.{j}.resblocks.{r}"), cin, cout, attn);
            }
        }
        // Down/upsample convs (index 0 is identity in the reference).
        for i in 1..n_lv {
            let c = self.channels[i - 1] as usize;
            out.push((format!("unet.downsamples.{i}.conv.weight"), vec![c, c, 3, 3]));
            out.push((format!("unet.downsamples.{i}.conv.bias"), vec![c]));
        }
        for i in 1..n_lv {
            // reversed(channels[:-1]) — for uniform-channel configs this is c.
            let c = self.channels[n_lv - 1 - i] as usize;
            out.push((format!("unet.upsamples.{i}.conv.weight"), vec![c, c, 3, 3]));
            out.push((format!("unet.upsamples.{i}.conv.bias"), vec![c]));
        }
        out.push(("norm_out.norm.weight".into(), vec![c0]));
        out.push(("norm_out.norm.bias".into(), vec![c0]));
        out.push(("conv_out.weight".into(), vec![ic, c0, 3, 3]));
        out.push(("conv_out.bias".into(), vec![ic]));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_atari_param_list_matches_reference_count_and_key_tensors() {
        // The Breakout checkpoint has exactly 236 denoiser tensors.
        let cfg = DiamondConfig::atari(4);
        let list = cfg.param_list();
        assert_eq!(list.len(), 236);
        let find = |n: &str| list.iter().find(|(k, _)| k == n).map(|(_, s)| s.clone());
        assert_eq!(find("conv_in.weight"), Some(vec![64, 15, 3, 3]));
        assert_eq!(find("unet.d_blocks.0.resblocks.0.norm1.linear.weight"), Some(vec![128, 256]));
        // Up resblocks concatenate a skip: norm1 over 128 channels.
        assert_eq!(find("unet.u_blocks.0.resblocks.0.norm1.linear.weight"), Some(vec![256, 256]));
        assert_eq!(find("unet.u_blocks.0.resblocks.0.proj.weight"), Some(vec![64, 128, 1, 1]));
        assert_eq!(
            find("unet.mid_blocks.resblocks.1.attn.qkv_proj.weight"),
            Some(vec![192, 64, 1, 1])
        );
        assert_eq!(find("unet.upsamples.3.conv.weight"), Some(vec![64, 64, 3, 3]));
        assert_eq!(find("conv_out.weight"), Some(vec![3, 64, 3, 3]));
        // No down resblock has a proj (64 -> 64 everywhere).
        assert!(find("unet.d_blocks.0.resblocks.0.proj.weight").is_none());
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg = DiamondConfig::atari(4);
        let back = DiamondConfig::from_json(&cfg.to_json()).unwrap();
        assert_eq!(cfg, back);
    }
}
