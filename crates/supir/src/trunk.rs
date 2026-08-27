// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `GLVControl`: the ~1.24B control trunk, over the PUBLIC
//! `sdxlunet::model::Rec` - composition, not a `ControlNetConfig`
//! generalisation. A vanilla ControlNet's 8-layer PIXEL embedder, its output
//! zero-convs and `scale_chan`, and its single conditioning-image input
//! share no code with a trunk whose hint is already a latent and which
//! returns raw hidden states with no output convs at all - generalising
//! `ControlNetConfig` to cover both would be a config union sharing no
//! code, not "one implementation".
//!
//! ```text
//! h = xt                                        # the noisy latent
//! guided_hint = input_hint_block(LQ_latent)      # ZERO-INIT 3x3 conv, 4 -> c0
//! h = conv_in(h) + guided_hint                   # hint added ONLY after conv_in
//! (h, hs, ch, cw) = down_path(h)                 # Rec::down_path, unchanged
//! h = mid_block(h)                                # Rec::mid_block, unchanged
//! return hs ++ [h]                                # 10 raw hidden states
//! ```
//!
//! The trunk's own `time_embed`/`label_emb` and every down/mid block tensor
//! are recorded through the SAME [`sdxlunet::model::Rec`] methods the frozen
//! backbone uses (`Rec::conditioning`, `Rec::down_path`, `Rec::mid_block`) -
//! under [`Rec::set_prefix`]'s `"control_model."` prefix, so this module adds
//! no block and no kernel: one zero-init conv plus the hint-add is the whole
//! delta over a plain UNet down+mid walk.

use gpu_core::DeviceBuffer;
use sdxlunet::config::UNetConfig;
use sdxlunet::model::Rec;
use vae::blocks::skipfuse::Map;

use crate::config::HINT_EMBEDDER;

/// Record `GLVControl`'s forward into `r` and return its 10 raw hidden
/// states (`hs`, push order: the 9 down-path entries then the mid output) -
/// [`crate::adaptors::Adaptors`]'s per-join control tensors.
///
/// `prefix` must be the SAME string already installed on `r` via
/// [`Rec::set_prefix`] (normally `"control_model."`) - `Rec`'s own
/// `conditioning`/`down_path`/`mid_block` read that field internally, but
/// [`vae::blocks::Builder::conv`] (which `conv_in` and the hint embedder go
/// through directly, bypassing `Rec`) does not, so this function needs it
/// spelled out rather than reaching for a private field.
#[allow(clippy::too_many_arguments)]
pub fn record(
    r: &mut Rec<'_>,
    cfg: &UNetConfig,
    prefix: &str,
    h: u32,
    w: u32,
    enc_in: &DeviceBuffer,
    hint_in: &DeviceBuffer,
    sample_in: &DeviceBuffer,
    temb_in: &DeviceBuffer,
    aug_in: &DeviceBuffer,
) -> Vec<Map> {
    r.conditioning(cfg, temb_in, aug_in);

    let c0 = cfg.block_out_channels[0];
    let cin = r.blocks().conv(&format!("{prefix}conv_in"), cfg.in_channels, c0, 3, 1, h, w, sample_in);
    r.blocks().tap(format!("{prefix}conv_in"), &cin, c0 * h * w);
    let hint = r.blocks().conv(&format!("{prefix}{HINT_EMBEDDER}"), 4, c0, 3, 1, h, w, hint_in);
    r.blocks().tap(format!("{prefix}hint"), &hint, c0 * h * w);
    // Upstream's forward loop adds the hint after the FIRST `input_blocks`
    // entry only, and `input_blocks[0]` IS `conv_in` (a bare 3x3 conv, no
    // resnet) in the upstream numbering - so the add happens here, before
    // any down-path resnet sees the running state.
    let x = r.blocks().add(c0 * h * w, &cin, &hint);
    r.blocks().free((c0 * h * w) as u64, cin);
    r.blocks().free((c0 * h * w) as u64, hint);
    r.blocks().tap(format!("{prefix}hs0"), &x, c0 * h * w);

    let (hh, skips, ch, cw) = r.down_path(cfg, h, w, enc_in, &x);
    let mid = r.mid_block(cfg, ch, cw, enc_in, &hh);
    let cmid = *cfg.block_out_channels.last().expect("levels >= 1");

    let mut hs: Vec<Map> = skips.into_iter().map(|(buf, c, h, w)| Map { buf, c, h, w }).collect();
    r.blocks().tap(format!("{prefix}hs_mid"), &mid, cmid * ch * cw);
    hs.push(Map { buf: mid, c: cmid, h: ch, w: cw });
    hs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny-config smoke test (porting.md §4): records a full trunk forward
    /// at [`UNetConfig::tiny`] with synthetic weights and asserts every
    /// hidden state is finite and the right shape. No real checkpoint
    /// needed.
    #[test]
    fn tiny_trunk_forward_is_finite_and_the_right_shapes() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let cfg = UNetConfig::tiny();
        let prefix = "control_model.";
        let manifest = crate::config::trunk_manifest(&cfg);
        assert!(manifest.iter().any(|(n, _)| n == &format!("{prefix}{HINT_EMBEDDER}.weight")));
        let tensors = crate::init::init_weights_for(&manifest, 3);
        let gpu = gpu_core::testgpu::dev(&crate::model::KERNELS);
        let mut r = Rec::new(&gpu, &cfg, &tensors, 9, false);
        r.set_prefix(prefix);

        let (h, w) = (16u32, 16u32);
        let sample_in = gpu.storage((cfg.in_channels * h * w) as u64);
        let hint_in = gpu.storage((4 * h * w) as u64);
        let enc_in = gpu.storage((9 * cfg.cross_attention_dim) as u64);
        let temb_in = gpu.storage(cfg.block_out_channels[0] as u64);
        let aug_in = gpu.storage(cfg.projection_class_embeddings_input_dim as u64);
        gpu.write_f32(&sample_in, &vec![0.1f32; (cfg.in_channels * h * w) as usize]);
        gpu.write_f32(&hint_in, &vec![0.2f32; (4 * h * w) as usize]);
        gpu.write_f32(&enc_in, &vec![0.3f32; (9 * cfg.cross_attention_dim) as usize]);

        let hs = record(&mut r, &cfg, prefix, h, w, &enc_in, &hint_in, &sample_in, &temb_in, &aug_in);
        assert_eq!(hs.len(), cfg.skip_stack().len() + 1);

        let shapes = cfg.skip_shapes(h, w);
        for (i, (c, sh, sw)) in shapes.into_iter().enumerate() {
            assert_eq!((hs[i].c, hs[i].h, hs[i].w), (c, sh, sw), "hs[{i}]");
        }
        let cmid = *cfg.block_out_channels.last().unwrap();
        assert_eq!(hs.last().unwrap().c, cmid);

        let bufs: Vec<(gpu_core::DeviceBuffer, usize)> =
            hs.iter().map(|m| (m.buf.clone(), (m.c * m.h * m.w) as usize)).collect();
        let (steps, _taps) = r.into_blocks().finish();
        gpu.submit(&[], &steps);
        for (buf, n) in &bufs {
            let v = gpu.read(buf, *n);
            assert!(v.iter().all(|x| x.is_finite()), "a trunk hidden state is non-finite");
        }
    }
}
