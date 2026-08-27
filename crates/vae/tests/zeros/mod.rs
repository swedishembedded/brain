// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Zero tensors at the shapes a VAE graph asks for, so it can be built - and
//! therefore measured - without a checkpoint. Shapes only: nothing here runs,
//! and a wrong length would show up as a wrong binding rather than as a
//! plausible wrong number.

#![allow(dead_code)]

use vae::blocks::Tensors;
use vae::VaeConfig;

fn conv(t: &mut Tensors, p: &str, cin: u32, cout: u32, k: u32) {
    put(t, format!("{p}.weight"), vec![cout as usize, cin as usize, k as usize, k as usize]);
    put(t, format!("{p}.bias"), vec![cout as usize]);
}

fn gnorm(t: &mut Tensors, p: &str, c: u32) {
    put(t, format!("{p}.weight"), vec![c as usize]);
    put(t, format!("{p}.bias"), vec![c as usize]);
}

fn resnet(t: &mut Tensors, p: &str, cin: u32, cout: u32) {
    gnorm(t, &format!("{p}.norm1"), cin);
    conv(t, &format!("{p}.conv1"), cin, cout, 3);
    gnorm(t, &format!("{p}.norm2"), cout);
    conv(t, &format!("{p}.conv2"), cout, cout, 3);
    if cin != cout {
        conv(t, &format!("{p}.conv_shortcut"), cin, cout, 1);
    }
}

/// diffusers stores each attention projection as a 1x1 conv, so the element
/// count is `c*c` whichever way the checkpoint spells the shape.
fn attn(t: &mut Tensors, p: &str, c: u32) {
    gnorm(t, &format!("{p}.group_norm"), c);
    for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
        conv(t, &format!("{p}.{leaf}"), c, c, 1);
    }
}

fn put(t: &mut Tensors, name: String, shape: Vec<usize>) {
    let n: usize = shape.iter().product();
    t.insert(name, (shape, vec![0.0f32; n]));
}

/// Every tensor `VaeDecoder::from_diffusers` reads.
pub fn decoder(cfg: &VaeConfig) -> Tensors {
    let mut t = Tensors::new();
    let zc = cfg.latent_channels;
    let rc = cfg.reversed_channels();
    let mid_c = *cfg.block_out_channels.last().expect("block_out_channels");
    if cfg.use_post_quant_conv {
        conv(&mut t, "post_quant_conv", zc, zc, 1);
    }
    conv(&mut t, "decoder.conv_in", zc, mid_c, 3);
    resnet(&mut t, "decoder.mid_block.resnets.0", mid_c, mid_c);
    if cfg.mid_block_add_attention {
        attn(&mut t, "decoder.mid_block.attentions.0", mid_c);
    }
    resnet(&mut t, "decoder.mid_block.resnets.1", mid_c, mid_c);
    let mut prev = mid_c;
    for (i, &out_c) in rc.iter().enumerate() {
        for r in 0..cfg.layers_per_block + 1 {
            let cin = if r == 0 { prev } else { out_c };
            resnet(&mut t, &format!("decoder.up_blocks.{i}.resnets.{r}"), cin, out_c);
        }
        if i < rc.len() - 1 {
            conv(&mut t, &format!("decoder.up_blocks.{i}.upsamplers.0.conv"), out_c, out_c, 3);
        }
        prev = out_c;
    }
    gnorm(&mut t, "decoder.conv_norm_out", prev);
    conv(&mut t, "decoder.conv_out", prev, cfg.out_channels, 3);
    t
}

/// Every tensor `VaeEncoder::from_diffusers` reads.
pub fn encoder(cfg: &VaeConfig) -> Tensors {
    let mut t = Tensors::new();
    let ch = &cfg.block_out_channels;
    conv(&mut t, "encoder.conv_in", cfg.in_channels, ch[0], 3);
    let mut prev = ch[0];
    for (i, &out_c) in ch.iter().enumerate() {
        for r in 0..cfg.layers_per_block {
            let cin = if r == 0 { prev } else { out_c };
            resnet(&mut t, &format!("encoder.down_blocks.{i}.resnets.{r}"), cin, out_c);
        }
        prev = out_c;
        if i < ch.len() - 1 {
            conv(&mut t, &format!("encoder.down_blocks.{i}.downsamplers.0.conv"), out_c, out_c, 3);
        }
    }
    let mid_c = *ch.last().expect("block_out_channels");
    resnet(&mut t, "encoder.mid_block.resnets.0", mid_c, mid_c);
    if cfg.mid_block_add_attention {
        attn(&mut t, "encoder.mid_block.attentions.0", mid_c);
    }
    resnet(&mut t, "encoder.mid_block.resnets.1", mid_c, mid_c);
    gnorm(&mut t, "encoder.conv_norm_out", mid_c);
    let moments = 2 * cfg.latent_channels;
    conv(&mut t, "encoder.conv_out", mid_c, moments, 3);
    if cfg.use_quant_conv {
        conv(&mut t, "quant_conv", moments, moments, 1);
    }
    t
}
