// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build CosyVoice 2's `HiFTGenerator` conv trunk (`hift::decode`, from
//! `conv_pre` through `conv_post`) as an ONNX graph, for OpenVINO/Intel-NPU
//! compilation. Architecturally the same shape as [`crate::codec_topology`]'s
//! SEANet decoder (upsample stages + dilated resblocks), but non-causal
//! (symmetric padding, matching `HiFTGenerator`, not CosyVoice 3's
//! `CausalHiFTGenerator`) and with plain single-parameter Snake activations
//! (`x + (alpha+eps)^-1 * sin(alpha*x)^2`, `audio::snake::snake1d_ref`'s
//! formula) rather than the codec's two-parameter SnakeBeta.
//!
//! **Scope, a deliberate judgment call**: only the conv/resblock/upsample
//! trunk goes on the graph. The STFT of the NSF excitation (`s_stft`) is a
//! graph INPUT, computed host-side exactly as `hift::decode` computes it
//! internally (`stft_center`), and the ISTFT that turns this graph's
//! `magnitude`/`phase` outputs back into a waveform stays host too. Neither is
//! a weight-bearing part of the architecture - both are fixed DSP transforms
//! with no learned parameter - and this crate's own established convention
//! already keeps non-NPU-friendly, weight-free pre/post-processing off the
//! graph rather than reinventing it as ONNX ops with no precedent anywhere in
//! this crate (`crate::topology`'s own module doc: "DFL decode + NMS stay on
//! the host"). `f0_predictor` (a small 5-conv trunk feeding a NSF harmonic
//! source, not the resblock trunk this milestone measured as CosyVoice 2's
//! actual per-kernel-kind cost) and the NSF sine-generation math (elementwise,
//! and genuinely dependent on a host RNG draw) are left on the host for the
//! same reason: cheap relative to the conv trunk, and the noise draw already
//! has to cross the host/device boundary as a wired-through input either way
//! (exactly what `hift::forward_seeded` already does for the pure-Rust path).
//!
//! Swedish Embedded AB implements solutions for exporting from-scratch neural
//! vocoders to NPU-deployable ONNX graphs for its clients. If your team needs
//! a HiFi-GAN-family vocoder's conv trunk offloaded to an Intel NPU, you can
//! procure our services by sending an email to info@swedishembedded.com.

use cosyvoice::hift_config::{HiftConfig, RESBLOCK_DILATIONS};
use cosyvoice::hift_import::{ConvW, HiftWeights, ResBlockW};
use onnx::builder::GraphBuilder;
use onnx::conv::ConvTranspose1d;
use onnx::graph::Node;

use crate::topo::TopoBase;

struct HiftTopo<'a> {
    b: TopoBase<'a>,
}

impl<'a> std::ops::Deref for HiftTopo<'a> {
    type Target = TopoBase<'a>;
    fn deref(&self) -> &Self::Target {
        &self.b
    }
}
impl<'a> std::ops::DerefMut for HiftTopo<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.b
    }
}

impl<'a> HiftTopo<'a> {
    /// Plain `Conv1d` (symmetric padding, dilation, no groups). `name` doubles
    /// as the initializer prefix and the node name - each conv in this trunk
    /// is emitted exactly once, so no idempotency check is needed here (unlike
    /// `crate::topo::TopoBase::f32`, which several models call from inside a
    /// per-timestep loop with a REUSED name).
    #[allow(clippy::too_many_arguments)]
    fn conv(&mut self, name: &str, x: &str, w: &ConvW, cin: usize, cout: usize, k: usize, stride: usize, pad: usize, dilation: usize) -> String {
        let wname = format!("{name}.weight");
        self.g.init_f32(&wname, &[cout as i64, cin as i64, k as i64], w.weight.clone());
        let bname = format!("{name}.bias");
        self.g.init_f32(&bname, &[cout as i64], w.bias.clone());
        let out = self.tmp(name);
        self.g.add(
            Node::new("Conv", &[x, &wname, &bname], &[&out])
                .name(name)
                .attr_ints("kernel_shape", &[k as i64])
                .attr_ints("strides", &[stride as i64])
                .attr_ints("pads", &[pad as i64, pad as i64])
                .attr_ints("dilations", &[dilation as i64])
                .attr_int("group", 1),
        );
        out
    }

    /// Plain `ConvTranspose1d`, symmetric padding on both sides (`pad_begin ==
    /// pad_end == pad`) - CosyVoice 2's non-causal upsample stage, unlike the
    /// codec's causal `pads=[0, K-stride]`.
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose(&mut self, name: &str, x: &str, w: &ConvW, cin: usize, cout: usize, l: usize, k: usize, stride: usize, pad: usize) -> (String, usize) {
        let c = ConvTranspose1d { cin, cout, l, k, stride, pad_begin: pad, pad_end: pad, dilation: 1, groups: 1, output_padding: 0 };
        let lo = c.l_out();
        let out = self.tmp(name);
        self.g.conv_transpose1d(name, x, &out, w.weight.clone(), Some(w.bias.clone()), &c);
        (out, lo)
    }

    fn leaky_relu(&mut self, x: &str, slope: f32) -> String {
        let out = self.tmp("lrelu");
        self.g.add(Node::new("LeakyRelu", &[x], &[&out]).attr_float("alpha", slope));
        out
    }

    /// Plain (non-Beta) Snake over NCL `[1,C,L]`: `x + (alpha+eps)^-1 *
    /// sin(alpha*x)^2`, per-channel `alpha` - `audio::snake::snake1d_ref`'s
    /// exact formula (distinct from `crate::codec_topology::snake`'s
    /// two-parameter SnakeBeta).
    fn snake(&mut self, x: &str, name: &str, alpha: &[f32], c: usize) -> String {
        let aname = format!("{name}.alpha");
        self.g.init_f32(&aname, &[1, c as i64, 1], alpha.to_vec());
        if !self.has("snake_eps") {
            self.f32("snake_eps", &[1], vec![1e-9]);
        }
        let ax = self.mul(x, &aname);
        let sn = self.unary("Sin", &ax);
        let sn2 = self.mul_t(&sn, &sn);
        let denom = self.add(&aname, "snake_eps");
        let recip = self.unary("Reciprocal", &denom);
        let term = self.mul_t(&sn2, &recip);
        self.add_t(x, &term)
    }

    /// `ResBlock.forward`: 3 sequential `(Snake -> conv1[dilation] -> Snake ->
    /// conv2[dilation=1]) + x` branches, the SAME `x` threaded through all
    /// three (matching `hift::resblock_forward`'s doc, not a chained residual
    /// stack).
    fn resblock(&mut self, x_in: &str, prefix: &str, rb: &ResBlockW, c: usize, k: usize) -> String {
        let mut x = x_in.to_string();
        // `idx` indexes several parallel arrays (`RESBLOCK_DILATIONS`,
        // `rb.alpha{1,2}`, `rb.convs{1,2}`) - clippy's `needless_range_loop`
        // heuristic only sees the first use (matches `hift::resblock_forward`'s
        // own `#[allow]` for the identical shape).
        #[allow(clippy::needless_range_loop)]
        for idx in 0..3usize {
            let d = RESBLOCK_DILATIONS[idx] as usize;
            let pad1 = (k - 1) * d / 2;
            let xt = self.snake(&x, &format!("{prefix}.act1.{idx}"), &rb.alpha1[idx], c);
            let xt = self.conv(&format!("{prefix}.convs1.{idx}"), &xt, &rb.convs1[idx], c, c, k, 1, pad1, d);
            let xt = self.snake(&xt, &format!("{prefix}.act2.{idx}"), &rb.alpha2[idx], c);
            let pad2 = (k - 1) / 2;
            let xt = self.conv(&format!("{prefix}.convs2.{idx}"), &xt, &rb.convs2[idx], c, c, k, 1, pad2, 1);
            x = self.add_t(&x, &xt);
        }
        x
    }

    /// Slice NCL `[1,C,L]` along the length axis (axis 2) -> `[1,C,end-start]`.
    fn slice_ncl(&mut self, x: &str, start: i64, end: i64) -> String {
        let s = self.tmp("sl_s");
        let e = self.tmp("sl_e");
        self.g.init_i64(&s, &[1], vec![start]);
        self.g.init_i64(&e, &[1], vec![end]);
        if !self.has("axis2_const") {
            self.g.init_i64("axis2_const", &[1], vec![2]);
        }
        let o = self.tmp("slc");
        self.g.add(Node::new("Slice", &[x, &s, &e, "axis2_const"], &[&o]));
        o
    }

    /// Concat two NCL `[1,C,*]` tensors along the length axis.
    fn concat_ncl(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("cat");
        self.g.add(Node::new("Concat", &[a, b], &[&o]).attr_int("axis", 2));
        o
    }
}

/// Build the HiFT conv-trunk graph into `g`. `t_mel` is the fixed mel frame
/// count; `n_frames_s` is the fixed frame count of the (host-precomputed)
/// excitation STFT `s_stft` the caller will feed in - see this module's doc
/// for why that tensor is a graph input rather than computed in-graph.
///
/// Inputs: `mel:[1,in_channels,t_mel]`, `s_stft:[1,source_stft_channels,
/// n_frames_s]`. Outputs: `magnitude`/`phase:[1,stft_bins,L]` each
/// (pre-ISTFT, matching `hift::DecodeOutput`'s two fields of the same name) -
/// `L` is returned so the caller can declare the exact output shape / drive
/// the host-side ISTFT.
pub fn build_hift_decode_graph(cfg: &HiftConfig, w: &HiftWeights, t_mel: usize, n_frames_s: usize, g: &mut GraphBuilder) -> usize {
    let base = cfg.base_channels as usize;
    let bins = cfg.stft_bins() as usize;
    let src_ch = cfg.source_stft_channels() as usize;
    let mut tp = HiftTopo { b: TopoBase::new(g) };

    tp.g.input_f32("mel", &[1, cfg.in_channels as i64, t_mel as i64]);
    tp.g.input_f32("s_stft", &[1, src_ch as i64, n_frames_s as i64]);

    let mut x = tp.conv("conv_pre", "mel", &w.conv_pre, cfg.in_channels as usize, base, 7, 1, 3, 1);
    let mut c = base;
    let mut l = t_mel;
    let down_strides = cfg.source_downsample_strides();

    // `i` indexes several parallel arrays (`down_strides`,
    // `cfg.upsample_{rates,kernel_sizes}`, `w.ups`, `w.source_downs`,
    // `w.source_resblocks`, `w.resblocks`) - clippy's `needless_range_loop`
    // heuristic only sees the first use (matches `hift::decode`'s own
    // `#[allow]` for the identical shape).
    #[allow(clippy::needless_range_loop)]
    for i in 0..3usize {
        let u = cfg.upsample_rates[i] as usize;
        let k = cfg.upsample_kernel_sizes[i] as usize;
        let pad = (k - u) / 2;
        let cout = base / (1 << (i + 1));

        let xa = tp.leaky_relu(&x, cfg.lrelu_slope);
        let (mut xu, mut lu) = tp.conv_transpose(&format!("ups.{i}"), &xa, &w.ups[i], c, cout, l, k, u, pad);
        c = cout;

        if i == 2 {
            // ReflectionPad1d((1, 0)): prepend column `mirror` (mirrors
            // `hift::decode`'s manual mirror-pad exactly).
            let mirror = 1usize.min(lu - 1);
            let mirror_col = tp.slice_ncl(&xu, mirror as i64, mirror as i64 + 1);
            xu = tp.concat_ncl(&mirror_col, &xu);
            lu += 1;
        }
        l = lu;

        let stride_i = down_strides[i] as usize;
        let (kd, padd) = if stride_i == 1 { (1usize, 0usize) } else { (stride_i * 2, stride_i / 2) };
        let si_raw = tp.conv(&format!("source_downs.{i}"), "s_stft", &w.source_downs[i], src_ch, c, kd, stride_i, padd, 1);
        let src_k = cfg.source_resblock_kernel_sizes[i] as usize;
        let si = tp.resblock(&si_raw, &format!("source_resblocks.{i}"), &w.source_resblocks[i], c, src_k);
        xu = tp.add_t(&xu, &si);
        x = xu;

        let mut acc: Option<String> = None;
        for j in 0..3usize {
            let k3 = cfg.resblock_kernel_sizes[j] as usize;
            let r = tp.resblock(&x, &format!("resblocks.{}", i * 3 + j), &w.resblocks[i * 3 + j], c, k3);
            acc = Some(match acc {
                None => r,
                Some(a) => tp.add_t(&a, &r),
            });
        }
        let third = tp.tmp("third_const");
        tp.g.init_f32(&third, &[1], vec![1.0 / 3.0]);
        x = tp.mul(&acc.expect("3 resblock kernel sizes"), &third);
    }

    let xa = tp.leaky_relu(&x, 0.01); // F.leaky_relu(x) default slope, NOT cfg.lrelu_slope.
    let post_ch = 2 * bins;
    let post = tp.conv("conv_post", &xa, &w.conv_post, c, post_ch, 7, 1, 3, 1);

    let mag_raw = tp.slice_ncl(&post, 0, bins as i64);
    let magnitude = tp.unary("Exp", &mag_raw);
    let phase_raw = tp.slice_ncl(&post, bins as i64, post_ch as i64);
    let phase = tp.unary("Sin", &phase_raw);

    tp.node("Identity", &[&magnitude], "magnitude");
    tp.node("Identity", &[&phase], "phase");
    tp.g.output_f32("magnitude", &[1, bins as i64, l as i64]);
    tp.g.output_f32("phase", &[1, bins as i64, l as i64]);
    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use cosyvoice::hift_import::{ConvW as CW, F0PredictorW, HiftWeights as HW, ResBlockW as RBW};

    fn rand_conv(seed: &mut u64, cout: usize, cin: usize, k: usize) -> CW {
        let mut next = || {
            *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
        };
        CW { weight: (0..cout * cin * k).map(|_| next()).collect(), bias: (0..cout).map(|_| next()).collect() }
    }

    /// `ConvTranspose1d` weight layout is `[Cin, Cout, K]` (ONNX/brain's
    /// shared convention) with a `[Cout]` bias - the reverse of
    /// [`rand_conv`]'s plain-`Conv1d` `[Cout, Cin, K]` layout, so `ups[i]`
    /// needs its own fixture builder rather than a relabeled call.
    fn rand_convtr(seed: &mut u64, cin: usize, cout: usize, k: usize) -> CW {
        let mut next = || {
            *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
        };
        CW { weight: (0..cin * cout * k).map(|_| next()).collect(), bias: (0..cout).map(|_| next()).collect() }
    }

    fn rand_resblock(seed: &mut u64, c: usize, k: usize) -> RBW {
        RBW {
            convs1: [rand_conv(seed, c, c, k), rand_conv(seed, c, c, k), rand_conv(seed, c, c, k)],
            convs2: [rand_conv(seed, c, c, k), rand_conv(seed, c, c, k), rand_conv(seed, c, c, k)],
            alpha1: [vec![0.1; c], vec![0.1; c], vec![0.1; c]],
            alpha2: [vec![0.1; c], vec![0.1; c], vec![0.1; c]],
        }
    }

    /// A tiny fixture with the real config's architecture but random weights -
    /// small enough to build in a fraction of a second, and to run through
    /// OpenVINO CPU in the numerical parity test.
    fn tiny_weights(cfg: &HiftConfig) -> HW {
        let mut seed = 42u64;
        let base = cfg.base_channels as usize;
        HW {
            f0_predictor: F0PredictorW {
                condnet: [
                    rand_conv(&mut seed, base, cfg.in_channels as usize, 3),
                    rand_conv(&mut seed, base, base, 3),
                    rand_conv(&mut seed, base, base, 3),
                    rand_conv(&mut seed, base, base, 3),
                    rand_conv(&mut seed, base, base, 3),
                ],
                classifier_w: vec![0.01; base],
                classifier_b: 0.0,
            },
            m_source_linear_w: vec![0.1; cfg.harmonics() as usize],
            m_source_linear_b: 0.0,
            conv_pre: rand_conv(&mut seed, base, cfg.in_channels as usize, 7),
            ups: [
                rand_convtr(&mut seed, base, base / 2, cfg.upsample_kernel_sizes[0] as usize),
                rand_convtr(&mut seed, base / 2, base / 4, cfg.upsample_kernel_sizes[1] as usize),
                rand_convtr(&mut seed, base / 4, base / 8, cfg.upsample_kernel_sizes[2] as usize),
            ],
            source_downs: [
                rand_conv(&mut seed, base / 2, cfg.source_stft_channels() as usize, 30),
                rand_conv(&mut seed, base / 4, cfg.source_stft_channels() as usize, 6),
                rand_conv(&mut seed, base / 8, cfg.source_stft_channels() as usize, 1),
            ],
            source_resblocks: [
                rand_resblock(&mut seed, base / 2, cfg.source_resblock_kernel_sizes[0] as usize),
                rand_resblock(&mut seed, base / 4, cfg.source_resblock_kernel_sizes[1] as usize),
                rand_resblock(&mut seed, base / 8, cfg.source_resblock_kernel_sizes[2] as usize),
            ],
            resblocks: [
                rand_resblock(&mut seed, base / 2, cfg.resblock_kernel_sizes[0] as usize),
                rand_resblock(&mut seed, base / 2, cfg.resblock_kernel_sizes[1] as usize),
                rand_resblock(&mut seed, base / 2, cfg.resblock_kernel_sizes[2] as usize),
                rand_resblock(&mut seed, base / 4, cfg.resblock_kernel_sizes[0] as usize),
                rand_resblock(&mut seed, base / 4, cfg.resblock_kernel_sizes[1] as usize),
                rand_resblock(&mut seed, base / 4, cfg.resblock_kernel_sizes[2] as usize),
                rand_resblock(&mut seed, base / 8, cfg.resblock_kernel_sizes[0] as usize),
                rand_resblock(&mut seed, base / 8, cfg.resblock_kernel_sizes[1] as usize),
                rand_resblock(&mut seed, base / 8, cfg.resblock_kernel_sizes[2] as usize),
            ],
            conv_post: rand_conv(&mut seed, cfg.source_stft_channels() as usize, base / 8, 7),
        }
    }

    /// Structural check (always runs, no OpenVINO/NPU needed): the graph
    /// builds, declares the right I/O, and its op mix matches the
    /// architecture - the same "always-on" tier `tests/wm_onnx.rs` uses for
    /// DIAMOND.
    #[test]
    fn builds_a_structurally_correct_graph() {
        let cfg = HiftConfig::cosyvoice2();
        let w = tiny_weights(&cfg);
        let t_mel = 20usize;
        // n_frames_s must be >= the full trunk length after 3 upsample
        // stages for source_downs' conv to have a defined output - compute it
        // the same way `hift::decode`'s own call to `stft_center` would,
        // scaled down for the tiny fixture (exact value does not matter here,
        // only "large enough").
        let n_frames_s = t_mel * cfg.upsample_rates.iter().product::<u32>() as usize / cfg.hop_len as usize + 4;
        let mut g = onnx::GraphBuilder::new("hift_decode_test");
        let l = build_hift_decode_graph(&cfg, &w, t_mel, n_frames_s, &mut g);
        assert!(l > t_mel, "the trunk must upsample: l={l} t_mel={t_mel}");

        let graph = g.graph();
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.inputs[0].name, "mel");
        assert_eq!(graph.inputs[1].name, "s_stft");
        assert_eq!(graph.outputs.len(), 2);
        assert_eq!(graph.outputs[0].name, "magnitude");
        assert_eq!(graph.outputs[1].name, "phase");
        assert_eq!(graph.outputs[0].dims, vec![1, cfg.stft_bins() as i64, l as i64]);

        let count = |op: &str| graph.nodes.iter().filter(|n| n.op_type == op).count();
        // 1 conv_pre + 3 source_downs + (9 trunk + 3 source) resblocks * 3
        // branches * 2 convs each + 1 conv_post.
        assert_eq!(count("Conv"), 1 + 3 + 12 * 3 * 2 + 1, "unexpected Conv count");
        assert_eq!(count("ConvTranspose"), 3, "one ConvTranspose per upsample stage");
        // 12 resblocks * 3 branches * 2 Snake calls each (one Sin per Snake), plus the phase Sin.
        assert_eq!(count("Sin"), 12 * 3 * 2 + 1, "one Sin per Snake call, plus the phase Sin");
        assert_eq!(count("Exp"), 1, "the magnitude Exp");
        assert_eq!(count("LeakyRelu"), 3 + 1, "one per upsample stage plus the final activation");

        let bytes = g.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn same_output_names_are_never_duplicated() {
        // A structural regression this port has hit before elsewhere in this
        // crate (kernel/graph naming collisions silently produce a wrong
        // graph): every node output name must be unique.
        let cfg = HiftConfig::cosyvoice2();
        let w = tiny_weights(&cfg);
        let mut g = onnx::GraphBuilder::new("hift_decode_dupe_check");
        let t_mel = 12usize;
        let n_frames_s = t_mel * cfg.upsample_rates.iter().product::<u32>() as usize / cfg.hop_len as usize + 4;
        build_hift_decode_graph(&cfg, &w, t_mel, n_frames_s, &mut g);
        let mut seen = std::collections::HashSet::new();
        for n in &g.graph().nodes {
            for o in &n.outputs {
                assert!(seen.insert(o.clone()), "duplicate output tensor name: {o}");
            }
        }
    }
}
