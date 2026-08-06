// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `AutoencoderKL` decoder → Intel NPU, as an ONNX graph.
//!
//! This is the decoder half of the VAE every latent image model in the tree
//! ends with — Z-Image, FLUX, SDXL, and the VQGAN stack CodeFormer sits on. It
//! is the piece worth putting on an NPU first: it runs ONCE per generation
//! (unlike the DiT/UNet, which runs per step), it is small enough to fit, and
//! it is pure conv + GroupNorm + SiLU + nearest-Resize + one attention.
//!
//! **`use_post_quant_conv` is honoured and defaults to TRUE**, because the
//! reference's `AutoencoderKL.__init__` does and a config.json only carries the
//! keys it overrides. Dropping it is a decode that stays in a plausible `[-1,1]`
//! range and is UNCORRELATED with the truth — `docs/lessons.md` #16, which cost
//! a day of looking at a fox that had the right shape and unusable colour.
//!
//! GroupNorm comes from [`crate::topo::TopoBase::group_norm`], which is
//! decomposed (opset 13 has no `GroupNormalization`) and gated NUMERICALLY
//! against a host implementation in `tests/group_norm_onnx.rs`.
//!
//! Validation here is structural — see `tests/vae_onnx.rs` — plus that numerical
//! GroupNorm gate underneath. End-to-end numbers against `crates/vae` need
//! hardware.

use onnx::{GraphBuilder, Node};

use crate::topo::TopoBase;
use crate::topology::WeightSource;

/// The per-architecture leaf tensor names, mirroring `vae::blocks::BlockNames`
/// without depending on that crate.
///
/// diffusers and VQGAN are the SAME graph with different leaf names — a
/// resnet's projection shortcut is `conv_shortcut` in one and `conv_out` in the
/// other, an attention's projections `to_q/to_k/to_v/to_out.0` over a
/// `group_norm` versus `q/k/v/proj_out` over a `norm`. Parameterising them is
/// what stops `vqgan_topology` being a copy of this file (the shared block
/// builder makes exactly this choice for the same reason).
#[derive(Clone, Copy, Debug)]
pub struct TopoNames {
    pub shortcut: &'static str,
    pub attn_norm: &'static str,
    pub attn_q: &'static str,
    pub attn_k: &'static str,
    pub attn_v: &'static str,
    pub attn_proj: &'static str,
    /// Prefix the decoder's tensors sit under (`decoder` for both today).
    pub decoder: &'static str,
}

impl TopoNames {
    /// diffusers `AutoencoderKL` naming.
    pub const fn diffusers() -> TopoNames {
        TopoNames {
            shortcut: "conv_shortcut",
            attn_norm: "group_norm",
            attn_q: "to_q",
            attn_k: "to_k",
            attn_v: "to_v",
            attn_proj: "to_out.0",
            decoder: "decoder",
        }
    }
    /// `basicsr` VQGAN naming (CodeFormer's autoencoder).
    pub const fn vqgan() -> TopoNames {
        TopoNames {
            shortcut: "conv_out",
            attn_norm: "norm",
            attn_q: "q",
            attn_k: "k",
            attn_v: "v",
            attn_proj: "proj_out",
            decoder: "decoder",
        }
    }
}

/// The decoder's shape, mirroring `vae::VaeConfig` without depending on it
/// (the `npu` crate stays free of model crates).
#[derive(Clone, Debug)]
pub struct VaeTopo {
    /// Leaf names — [`TopoNames::diffusers`] or [`TopoNames::vqgan`].
    pub names: TopoNames,
    pub latent_channels: u32,
    pub out_channels: u32,
    /// Encoder channel schedule, low→high res. The decoder walks it REVERSED.
    pub block_out_channels: Vec<u32>,
    pub layers_per_block: u32,
    pub norm_num_groups: u32,
    pub norm_eps: f32,
    pub mid_block_add_attention: bool,
    /// The 1x1 `post_quant_conv` before `conv_in`. See the module docs — this
    /// defaults to TRUE in the reference and dropping it is silent corruption.
    pub use_post_quant_conv: bool,
    /// Latent grid the graph is built for.
    pub lh: u32,
    pub lw: u32,
}

impl VaeTopo {
    /// Decoder channel schedule, high→low res.
    fn reversed(&self) -> Vec<u32> {
        let mut c = self.block_out_channels.clone();
        c.reverse();
        c
    }
    /// One nearest-2x per up-block except the last.
    pub fn upscale(&self) -> u32 {
        1 << (self.block_out_channels.len() as u32 - 1)
    }
}

struct Vae<'a> {
    b: TopoBase<'a>,
}

impl<'a> Vae<'a> {
    /// `k`x`k` same-padded conv with bias.
    fn conv(&mut self, p: &str, cout: u32, cin: u32, k: u32, w: &dyn WeightSource, x: &str) -> String {
        let (wn, bn) = (format!("{p}.weight"), format!("{p}.bias"));
        let pad = (k / 2) as i64;
        self.b.f32(&wn, &[cout as i64, cin as i64, k as i64, k as i64], w.get(&wn));
        self.b.f32(&bn, &[cout as i64], w.get(&bn));
        let out = self.b.tmp("conv");
        self.b.g.add(
            Node::new("Conv", &[x, &wn, &bn], &[&out])
                .attr_ints("kernel_shape", &[k as i64, k as i64])
                .attr_ints("pads", &[pad, pad, pad, pad])
                .attr_ints("strides", &[1, 1]),
        );
        out
    }

    fn gn(&mut self, p: &str, c: u32, h: u32, w_: u32, t: &VaeTopo, w: &dyn WeightSource, x: &str) -> String {
        let (gn, bn) = (format!("{p}.weight"), format!("{p}.bias"));
        let (g, b) = (w.get(&gn), w.get(&bn));
        self.b.group_norm(
            x,
            1,
            c as usize,
            h as usize,
            w_ as usize,
            t.norm_num_groups as usize,
            &gn,
            g,
            &bn,
            b,
            t.norm_eps,
        )
    }

    /// diffusers `ResnetBlock2D` without temb: the shortcut is a 1x1 conv named
    /// `conv_shortcut` when the widths differ, else the identity.
    #[allow(clippy::too_many_arguments)]
    fn resnet(
        &mut self,
        p: &str,
        cin: u32,
        cout: u32,
        h: u32,
        w_: u32,
        t: &VaeTopo,
        w: &dyn WeightSource,
        x: &str,
    ) -> String {
        let n1 = self.gn(&format!("{p}.norm1"), cin, h, w_, t, w, x);
        let s1 = self.b.silu_t(&n1);
        let c1 = self.conv(&format!("{p}.conv1"), cout, cin, 3, w, &s1);
        let n2 = self.gn(&format!("{p}.norm2"), cout, h, w_, t, w, &c1);
        let s2 = self.b.silu_t(&n2);
        let c2 = self.conv(&format!("{p}.conv2"), cout, cout, 3, w, &s2);
        let skip = if cin != cout {
            self.conv(&format!("{p}.{}", t.names.shortcut), cout, cin, 1, w, x)
        } else {
            x.to_string()
        };
        self.b.add_t(&c2, &skip)
    }

    /// Single-head self-attention over the spatial grid, `head_dim = C`, scale
    /// `C^-0.5`, residual added to the PRE-norm input.
    #[allow(clippy::too_many_arguments)]
    fn attn(&mut self, p: &str, c: u32, h: u32, w_: u32, t: &VaeTopo, w: &dyn WeightSource, x: &str) -> String {
        let hw = (h * w_) as i64;
        let n = self.gn(&format!("{p}.{}", t.names.attn_norm), c, h, w_, t, w, x);
        // [1,C,H,W] -> [1,HW,C] so the projections are plain MatMuls.
        let flat_shape = self.b.tmp("attn_shape_flat");
        self.b.i64(&flat_shape, &[3], vec![1, c as i64, hw]);
        let f = self.b.reshape(&n, &flat_shape);
        let rows = self.b.transpose(&f, &[0, 2, 1]);

        let proj = |s: &mut Self, name: &str, src: &str| -> String {
            let wn = format!("{p}.{name}.weight");
            let bn = format!("{p}.{name}.bias");
            // diffusers stores [out,in]; ONNX MatMul wants [in,out].
            let raw = w.get(&wn);
            let mut tr = vec![0.0f32; raw.len()];
            for o in 0..c as usize {
                for i in 0..c as usize {
                    tr[i * c as usize + o] = raw[o * c as usize + i];
                }
            }
            s.b.f32(&wn, &[c as i64, c as i64], tr);
            s.b.f32(&bn, &[c as i64], w.get(&bn));
            let m = s.b.matmul(src, &wn);
            s.b.add_t(&m, &bn)
        };
        let q = proj(self, t.names.attn_q, &rows);
        let k = proj(self, t.names.attn_k, &rows);
        let v = proj(self, t.names.attn_v, &rows);

        let kt = self.b.transpose(&k, &[0, 2, 1]);
        let scores = self.b.matmul(&q, &kt);
        let scale = self.b.tmp("attn_scale");
        self.b.f32(&scale, &[1], vec![1.0 / (c as f32).sqrt()]);
        let scaled = self.b.mul_t(&scores, &scale);
        let probs = self.b.softmax(&scaled, -1);
        let ctx = self.b.matmul(&probs, &v);
        let out = proj(self, t.names.attn_proj, &ctx);

        // back to [1,C,H,W] and the residual onto the PRE-norm input.
        let back = self.b.transpose(&out, &[0, 2, 1]);
        let nchw_shape = self.b.tmp("attn_shape_nchw");
        self.b.i64(&nchw_shape, &[4], vec![1, c as i64, h as i64, w_ as i64]);
        let r = self.b.reshape(&back, &nchw_shape);
        self.b.add_t(x, &r)
    }

    /// Nearest 2x, the form OpenVINO accepts (explicit `scales`, empty `roi`).
    fn upsample2(&mut self, x: &str) -> String {
        let roi = self.b.tmp("roi");
        self.b.f32(&roi, &[0], vec![]);
        let scales = self.b.tmp("scales");
        self.b.f32(&scales, &[4], vec![1.0, 1.0, 2.0, 2.0]);
        let out = self.b.tmp("resize");
        self.b.g.add(
            Node::new("Resize", &[x, &roi, &scales], &[&out])
                .attr_str("mode", "nearest")
                .attr_str("nearest_mode", "floor")
                .attr_str("coordinate_transformation_mode", "asymmetric"),
        );
        out
    }
}

/// Build the decoder forward into `g`: latent `[1, zc, lh, lw]` → image.
pub fn build_vae_decoder_graph(t: &VaeTopo, w: &dyn WeightSource, g: &mut GraphBuilder) {
    let s = t.upscale() as i64;
    g.input_f32("latent", &[1, t.latent_channels as i64, t.lh as i64, t.lw as i64]);
    g.output_f32("image", &[1, t.out_channels as i64, t.lh as i64 * s, t.lw as i64 * s]);

    let rev = t.reversed();
    let mid_c = rev[0];
    let mut m = Vae { b: TopoBase::new(g) };

    // post_quant_conv (1x1) -> conv_in. Skipping the first is silent corruption.
    let z = if t.use_post_quant_conv {
        m.conv("post_quant_conv", t.latent_channels, t.latent_channels, 1, w, "latent")
    } else {
        "latent".to_string()
    };
    let dec = t.names.decoder;
    let mut x = m.conv(&format!("{dec}.conv_in"), mid_c, t.latent_channels, 3, w, &z);

    let (mut ch, mut cw) = (t.lh, t.lw);
    x = m.resnet(&format!("{dec}.mid_block.resnets.0"), mid_c, mid_c, ch, cw, t, w, &x);
    if t.mid_block_add_attention {
        x = m.attn(&format!("{dec}.mid_block.attentions.0"), mid_c, ch, cw, t, w, &x);
    }
    x = m.resnet(&format!("{dec}.mid_block.resnets.1"), mid_c, mid_c, ch, cw, t, w, &x);

    let mut cin = mid_c;
    for (i, &out_c) in rev.iter().enumerate() {
        for r in 0..=t.layers_per_block {
            x = m.resnet(&format!("{dec}.up_blocks.{i}.resnets.{r}"), cin, out_c, ch, cw, t, w, &x);
            cin = out_c;
        }
        if i + 1 < rev.len() {
            x = m.upsample2(&x);
            ch *= 2;
            cw *= 2;
            x = m.conv(&format!("{dec}.up_blocks.{i}.upsamplers.0.conv"), out_c, out_c, 3, w, &x);
        }
    }

    // Head: GroupNorm -> SiLU -> conv_out, writing the graph output.
    let n = m.gn(&format!("{dec}.conv_norm_out"), cin, ch, cw, t, w, &x);
    let a = m.b.silu_t(&n);
    let (wn, bn) = (format!("{dec}.conv_out.weight"), format!("{dec}.conv_out.bias"));
    m.b.f32(&wn, &[t.out_channels as i64, cin as i64, 3, 3], w.get(&wn));
    m.b.f32(&bn, &[t.out_channels as i64], w.get(&bn));
    m.b.g.add(
        Node::new("Conv", &[&a, &wn, &bn], &["image"])
            .attr_ints("kernel_shape", &[3, 3])
            .attr_ints("pads", &[1, 1, 1, 1])
            .attr_ints("strides", &[1, 1]),
    );
}
