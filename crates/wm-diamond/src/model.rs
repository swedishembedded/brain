// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The DIAMOND conditional EDM UNet as a pre-recorded brain kernel graph.
//!
//! Inference-only (fine-tuning lands with the training unit): weights are
//! uploaded once, the whole forward is recorded once as a `Vec<Step>`, and a
//! denoise call is: host-compute cond → write ~44 AdaGroupNorm gamma/beta
//! buffers + the noisy input → ONE submit → read the model output. Per-sigma
//! scalars therefore never re-record the graph.
//!
//! AdaGroupNorm maps onto the stock `gn_stats`+`gn_apply` pair by folding the
//! conditioning into the gamma/beta buffer: gamma = 1+scale, beta = shift
//! (see cond::AdaGnSite). The 8x8 mid-block self-attention composes
//! `nchw_nlc` + fused-qkv bidirectional attention + `nlc_nchw`.

use crate::cond::{AdaGnSite, CondNet};
use crate::config::DiamondConfig;
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use std::collections::HashMap;

// Kernel-table indices (order matches KERNELS).
//
// `conv_bias` and the serial `gn_stats` used to sit at slots 0 and 1, carrying
// `#[allow(dead_code)]` markers: registered on every device this model ever
// built, compiled by every backend, and dispatched by nothing. The register
// -tiled `conv_bias_reg` replaced the first and the two-stage
// `gn_part`/`gn_stats2` pair replaced the second, and neither removal reached
// the table. The `#[allow]` is what let them sit there — see the slot test at
// the bottom of this file, which is now what keeps this list honest.
const K_GN_APPLY: usize = 0;
const K_SILU: usize = 1;
const K_ADD2: usize = 2;
const K_CONCAT2: usize = 3;
const K_UPSAMPLE2: usize = 4;
const K_NCHW_NLC: usize = 5;
const K_NLC_NCHW: usize = 6;
const K_ATTN_SCORES: usize = 7;
const K_ATTN_SOFTMAX: usize = 8;
const K_ATTN_APPLY: usize = 9;
const K_SCALE_ROW: usize = 10;
const K_EDM_MIX: usize = 11;
const K_EDM_WRAP: usize = 12;
const K_GN_PART: usize = 13;
const K_GN_STATS2: usize = 14;
const K_CONV_BIAS_REG: usize = 15;

const KERNELS: [(&str, &str); 16] = [
    ("gn_apply", kernels::GN_APPLY),
    ("silu", kernels::SILU),
    ("add2", kernels::ADD2),
    ("concat2", kernels::CONCAT2),
    ("upsample2", kernels::UPSAMPLE2),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("scale_row", kernels::SCALE_ROW),
    ("edm_mix", kernels::EDM_MIX),
    ("edm_wrap", kernels::EDM_WRAP),
    ("gn_part", kernels::GN_PART),
    ("gn_stats2", kernels::GN_STATS2),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
];

const GN_EPS: f32 = 1e-5;
/// Partial-reduction width for gn_part (threads per (n,g) group).
const GN_P: u32 = 64;
const ATTN_HEAD_DIM: u32 = 8;

/// Host tensors by (stripped) reference name.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

pub struct DiamondUNet {
    pub gpu: Gpu,
    pub cfg: DiamondConfig,
    cond: CondNet,
    /// AdaGroupNorm sites in graph order with their device gamma/beta buffers.
    adagn: Vec<(AdaGnSite, DeviceBuffer)>,
    steps: Vec<Step>,
    /// Full denoise-iteration step list: c_in scale -> UNet -> EDM wrap ->
    /// Euler mix -> copy back into `x_state`. Submitted once per sigma; all
    /// per-sigma scalars live in the tiny coef buffers below.
    loop_steps: Vec<Step>,
    x_in: DeviceBuffer,
    obs_in: DeviceBuffer,
    y_out: DeviceBuffer,
    /// Device-resident sampler state (x_t across denoise steps).
    x_state: DeviceBuffer,
    /// Quantized denoised output of one iteration (the final frame).
    denoised: DeviceBuffer,
    cin_buf: DeviceBuffer,
    wrap_coef: DeviceBuffer,
    euler_ab: DeviceBuffer,
    /// Named intermediate buffers for parity debugging (name, buffer, len).
    taps: Vec<(String, DeviceBuffer, usize)>,
}

/// Graph-construction state.
struct Builder<'a> {
    gpu: &'a Gpu,
    t: &'a Tensors,
    cc: usize,
    steps: Vec<Step>,
    adagn: Vec<(AdaGnSite, DeviceBuffer)>,
    taps: Vec<(String, DeviceBuffer, usize)>,
}

impl<'a> Builder<'a> {
    fn dev(&self, name: &str) -> DeviceBuffer {
        let (_, data) =
            self.t.get(name).unwrap_or_else(|| panic!("diamond: missing tensor {name}"));
        self.gpu.storage_init(name, data)
    }

    fn host(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        self.t
            .get(name)
            .unwrap_or_else(|| panic!("diamond: missing tensor {name}"))
            .clone()
    }

    fn act(&self, len: u64) -> DeviceBuffer {
        self.gpu.storage(len)
    }

    fn tap(&mut self, name: String, buf: &DeviceBuffer, len: u32) {
        self.taps.push((name, buf.clone(), len as usize));
    }

    fn push(&mut self, step: Step) {
        self.steps.push(step);
    }

    /// conv (+bias) `prefix.{weight,bias}`: x[cin,h,w] -> y[cout,ho,wo].
    fn conv(
        &mut self,
        prefix: &str,
        cin: u32,
        cout: u32,
        k: u32,
        stride: u32,
        pad: u32,
        h: u32,
        w: u32,
        x: &DeviceBuffer,
    ) -> (DeviceBuffer, u32, u32) {
        let ho = (h + 2 * pad - k) / stride + 1;
        let wo = (w + 2 * pad - k) / stride + 1;
        let wgt = self.dev(&format!("{prefix}.weight"));
        let bias = self.dev(&format!("{prefix}.bias"));
        let y = self.act((cout * ho * wo) as u64);
        // Register-tiled conv (8 output channels x 4 positions per invocation).
        let threads = cout.div_ceil(8) * (ho * wo).div_ceil(4);
        self.push(self.gpu.step(
            K_CONV_BIAS_REG,
            &[x, &wgt, &bias, &y],
            &[1, cin, h, w, cout, k, stride, pad, ho, wo],
            threads,
        ));
        (y, ho, wo)
    }

    /// GroupNorm with a DYNAMIC (conditioned) gamma/beta buffer.
    fn adagn(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let (wsh, wdat) = self.host(&format!("{prefix}.linear.weight"));
        let (_, bdat) = self.host(&format!("{prefix}.linear.bias"));
        assert_eq!(wsh, vec![2 * c as usize, self.cc]);
        let gb = self.gpu.storage(2 * c as u64);
        self.adagn.push((AdaGnSite { w: wdat, b: bdat, c: c as usize }, gb.clone()));
        self.gn_with_gb(c, h, w, x, &gb)
    }

    /// GroupNorm with a STATIC affine gamma/beta from `prefix.{weight,bias}`.
    fn affine_gn(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let (_, gamma) = self.host(&format!("{prefix}.weight"));
        let (_, beta) = self.host(&format!("{prefix}.bias"));
        let mut gbv = gamma;
        gbv.extend_from_slice(&beta);
        let gb = self.gpu.storage_init(prefix, &gbv);
        self.gn_with_gb(c, h, w, x, &gb)
    }

    fn gn_with_gb(
        &mut self,
        c: u32,
        h: u32,
        w: u32,
        x: &DeviceBuffer,
        gb: &DeviceBuffer,
    ) -> DeviceBuffer {
        let g = wm_core::gn::num_groups(c);
        let stats = self.act(2 * g as u64);
        let y = self.act((c * h * w) as u64);
        // Parallel two-stage reduction (gn_part -> gn_stats2): the serial
        // per-group gn_stats was measured at 77% of GPU frame time (2-4
        // invocations looping 131k elements on one EU lane each).
        let part = self.act(2 * g as u64 * GN_P as u64);
        self.push(self.gpu.step(
            K_GN_PART,
            &[x, &part],
            &[1, c, h, w, g, GN_P],
            g * GN_P,
        ));
        self.push(self.gpu.step(
            K_GN_STATS2,
            &[&part, &stats],
            &[1, c, h, w, g, GN_P, f(GN_EPS)],
            g,
        ));
        self.push(self.gpu.step(
            K_GN_APPLY,
            &[x, &stats, gb, &y],
            &[1, c, h, w, g],
            c * h * w,
        ));
        y
    }

    fn silu(&mut self, n: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act(n as u64);
        self.push(self.gpu.step(K_SILU, &[x, &y], &[n], n));
        y
    }

    fn add(&mut self, n: u32, a: &DeviceBuffer, b: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act(n as u64);
        self.push(self.gpu.step(K_ADD2, &[a, b, &y], &[n], n));
        y
    }

    fn concat(
        &mut self,
        ca: u32,
        cb: u32,
        h: u32,
        w: u32,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
    ) -> DeviceBuffer {
        let y = self.act(((ca + cb) * h * w) as u64);
        self.push(self.gpu.step(
            K_CONCAT2,
            &[a, b, &y],
            &[1, ca, cb, h, w],
            (ca + cb) * h * w,
        ));
        y
    }

    fn upsample(&mut self, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act((c * 2 * h * 2 * w) as u64);
        self.push(self.gpu.step(K_UPSAMPLE2, &[x, &y], &[1, c, h, w], c * 4 * h * w));
        y
    }

    /// Mid-block self-attention (norm -> qkv 1x1 -> bidir attention -> out 1x1
    /// -> residual add).
    fn attn(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let t = h * w;
        let heads = (c / ATTN_HEAD_DIM).max(1);
        let normed = self.affine_gn(&format!("{prefix}.norm.norm"), c, h, w, x);
        self.tap(format!("{prefix}.norm"), &normed, c * h * w);
        let (qkv_chw, _, _) =
            self.conv(&format!("{prefix}.qkv_proj"), c, 3 * c, 1, 1, 0, h, w, &normed);
        self.tap(format!("{prefix}.qkv_proj"), &qkv_chw, 3 * c * h * w);
        // Reshape the conv output for attention: NCHW -> [T, 3C].
        let qkv = self.act((3 * c * t) as u64);
        self.push(self.gpu.step(
            K_NCHW_NLC,
            &[&qkv_chw, &qkv],
            &[3 * c * t, 3 * c, t],
            3 * c * t,
        ));
        self.tap(format!("{prefix}.qkv_rows"), &qkv, 3 * c * t);
        let scores = self.act((heads * t * t) as u64);
        self.push(self.gpu.step(
            K_ATTN_SCORES,
            &[&qkv, &scores],
            &[1, heads, t, ATTN_HEAD_DIM, 3 * c, 0, c],
            heads * t * t,
        ));
        self.tap(format!("{prefix}.scores"), &scores, heads * t * t);
        let probs = self.act((heads * t * t) as u64);
        self.push(self.gpu.step(
            K_ATTN_SOFTMAX,
            &[&scores, &probs],
            &[1, heads, t],
            heads * t,
        ));
        self.tap(format!("{prefix}.probs"), &probs, heads * t * t);
        let attn_out = self.act((t * c) as u64);
        // NOTE binding order: attn_apply_bidir takes (probs, qkv, out).
        self.push(self.gpu.step(
            K_ATTN_APPLY,
            &[&probs, &qkv, &attn_out],
            &[1, heads, t, ATTN_HEAD_DIM, 3 * c, 2 * c, c],
            heads * t * ATTN_HEAD_DIM,
        ));
        self.tap(format!("{prefix}.attn_rows"), &attn_out, t * c);
        let attn_chw = self.act((c * t) as u64);
        self.push(self.gpu.step(
            K_NLC_NCHW,
            &[&attn_out, &attn_chw],
            &[c * t, c, t],
            c * t,
        ));
        self.tap(format!("{prefix}.pre_out_proj"), &attn_chw, c * h * w);
        let (proj, _, _) = self.conv(&format!("{prefix}.out_proj"), c, c, 1, 1, 0, h, w, &attn_chw);
        self.tap(format!("{prefix}.out_proj"), &proj, c * h * w);
        // Reference quirk: SelfAttention2d reassigns x = norm(x) BEFORE the
        // residual — the skip connection adds the NORMED tensor, not the
        // block input (blocks.py::SelfAttention2d.forward).
        self.add(c * h * w, &normed, &proj)
    }

    /// One reference ResBlock: r = proj(x); y = conv2(silu(norm2(conv1(silu(
    /// norm1(x)))))) + r; then optional attention.
    #[allow(clippy::too_many_arguments)]
    fn resblock(
        &mut self,
        prefix: &str,
        cin: u32,
        cout: u32,
        attn: bool,
        h: u32,
        w: u32,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let r = if cin != cout {
            let (p, _, _) = self.conv(&format!("{prefix}.proj"), cin, cout, 1, 1, 0, h, w, x);
            p
        } else {
            x.clone()
        };
        let n1 = self.adagn(&format!("{prefix}.norm1"), cin, h, w, x);
        self.tap(format!("{prefix}.norm1"), &n1, cin * h * w);
        let s1 = self.silu(cin * h * w, &n1);
        let (c1, _, _) = self.conv(&format!("{prefix}.conv1"), cin, cout, 3, 1, 1, h, w, &s1);
        self.tap(format!("{prefix}.conv1"), &c1, cout * h * w);
        let n2 = self.adagn(&format!("{prefix}.norm2"), cout, h, w, &c1);
        self.tap(format!("{prefix}.norm2"), &n2, cout * h * w);
        let s2 = self.silu(cout * h * w, &n2);
        let (c2, _, _) = self.conv(&format!("{prefix}.conv2"), cout, cout, 3, 1, 1, h, w, &s2);
        self.tap(format!("{prefix}.conv2"), &c2, cout * h * w);
        let y = self.add(cout * h * w, &c2, &r);
        let out = if attn {
            self.attn(&format!("{prefix}.attn"), cout, h, w, &y)
        } else {
            y
        };
        self.tap(prefix.to_string(), &out, cout * h * w);
        out
    }
}

impl DiamondUNet {
    /// Build from host tensors on the given device ("cpu" | "gpu" default).
    pub fn new(cfg: DiamondConfig, tensors: &Tensors, device: Option<&str>) -> DiamondUNet {
        let gpu = match device {
            Some("cpu") => Gpu::new_cpu(&KERNELS),
            Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
            _ => Gpu::new(&KERNELS),
        };
        let cc = cfg.cond_channels as usize;

        // Host conditioning net.
        let get = |n: &str| -> Vec<f32> {
            tensors.get(n).unwrap_or_else(|| panic!("diamond: missing tensor {n}")).1.clone()
        };
        let cond = CondNet {
            cond_channels: cc,
            num_steps_conditioning: cfg.num_steps_conditioning as usize,
            fourier_w: get("noise_emb.weight"),
            act_emb: get("act_emb.0.weight"),
            num_actions: cfg.num_actions as usize,
            mlp0_w: get("cond_proj.0.weight"),
            mlp0_b: get("cond_proj.0.bias"),
            mlp2_w: get("cond_proj.2.weight"),
            mlp2_b: get("cond_proj.2.bias"),
        };

        let mut b =
            Builder { gpu: &gpu, t: tensors, cc, steps: vec![], adagn: vec![], taps: vec![] };

        let ic = cfg.img_channels;
        let (h0, w0) = (cfg.h, cfg.w);
        let nsc = cfg.num_steps_conditioning;
        let n_lv = cfg.levels();

        // Inputs: rescaled obs [nsc*ic,h,w] and c_in-scaled noisy [ic,h,w].
        let obs_in = gpu.storage((nsc * ic * h0 * w0) as u64);
        let x_in = gpu.storage((ic * h0 * w0) as u64);

        // conv_in over cat(obs, noisy) — concat then conv.
        let cat = b.concat(nsc * ic, ic, h0, w0, &obs_in, &x_in);
        b.tap("cat".into(), &cat, (nsc + 1) * ic * h0 * w0);
        let c0 = cfg.channels[0];
        let (mut x, _, _) =
            b.conv("conv_in", (nsc + 1) * ic, c0, 3, 1, 1, h0, w0, &cat);
        b.tap("conv_in".into(), &x, c0 * h0 * w0);

        // Down path. Skips per level: (x_down, rb outputs...).
        let mut hw = (h0, w0);
        let mut d_skips: Vec<Vec<(DeviceBuffer, u32)>> = vec![]; // (buf, channels)
        for i in 0..n_lv {
            let c1 = cfg.channels[i.saturating_sub(1)];
            let c2 = cfg.channels[i];
            // downsamples[i]: identity for i==0, stride-2 conv otherwise.
            if i > 0 {
                let (y, nh, nw) = b.conv(
                    &format!("unet.downsamples.{i}.conv"),
                    c1,
                    c1,
                    3,
                    2,
                    1,
                    hw.0,
                    hw.1,
                    &x,
                );
                x = y;
                hw = (nh, nw);
            }
            let mut level: Vec<(DeviceBuffer, u32)> = vec![(x.clone(), c1)];
            let n = cfg.depths[i];
            for r in 0..n {
                let cin = if r == 0 { c1 } else { c2 };
                x = b.resblock(
                    &format!("unet.d_blocks.{i}.resblocks.{r}"),
                    cin,
                    c2,
                    cfg.attn_depths[i],
                    hw.0,
                    hw.1,
                    &x,
                );
                level.push((x.clone(), c2));
            }
            d_skips.push(level);
        }

        // Mid.
        let cl = *cfg.channels.last().unwrap();
        for r in 0..2 {
            x = b.resblock(
                &format!("unet.mid_blocks.resblocks.{r}"),
                cl,
                cl,
                true,
                hw.0,
                hw.1,
                &x,
            );
        }

        // Up path: u_blocks[j] pairs with d_skips[n_lv-1-j], skips reversed.
        for j in 0..n_lv {
            let i = n_lv - 1 - j;
            let c1 = cfg.channels[i.saturating_sub(1)];
            let c2 = cfg.channels[i];
            if j > 0 {
                // upsamples[j]: nearest x2 then 3x3 conv.
                let cx = c2; // uniform-channel configs: output of previous level is c2
                let up = b.upsample(cx, hw.0, hw.1, &x);
                hw = (hw.0 * 2, hw.1 * 2);
                let (y, _, _) = b.conv(
                    &format!("unet.upsamples.{j}.conv"),
                    cx,
                    cx,
                    3,
                    1,
                    1,
                    hw.0,
                    hw.1,
                    &up,
                );
                x = y;
            }
            let skips = &d_skips[i];
            let n = cfg.depths[i] as usize;
            for r in 0..=n {
                let (skip, skip_c) = &skips[n - r]; // reversed order
                let xc = if r == 0 { c2 } else { cfg.channels[i] };
                let cat = b.concat(xc, *skip_c, hw.0, hw.1, &x, skip);
                let (cin, cout) = if r < n { (2 * c2, c2) } else { (c1 + c2, c1) };
                debug_assert_eq!(cin, xc + skip_c);
                x = b.resblock(
                    &format!("unet.u_blocks.{j}.resblocks.{r}"),
                    cin,
                    cout,
                    cfg.attn_depths[i],
                    hw.0,
                    hw.1,
                    &cat,
                );
            }
        }

        // Head: affine GroupNorm -> SiLU -> conv_out.
        let hn = b.affine_gn("norm_out.norm", c0, hw.0, hw.1, &x);
        let hs = b.silu(c0 * hw.0 * hw.1, &hn);
        let (y_out, _, _) = b.conv("conv_out", c0, ic, 3, 1, 1, hw.0, hw.1, &hs);

        assert_eq!(hw, (h0, w0), "UNet did not return to input resolution");

        // On-device sampler ring: scale_row(c_in) feeds the UNet, edm_wrap
        // quantizes, edm_mix advances x, scale_row(1) copies back into
        // x_state (no in-dispatch aliasing; each pass is elementwise).
        let n_px = ic * h0 * w0;
        let x_state = gpu.storage(n_px as u64);
        let denoised = gpu.storage(n_px as u64);
        let x_next = gpu.storage(n_px as u64);
        let cin_buf = gpu.storage(1);
        let one_buf = gpu.storage_init("wm.one", &[1.0]);
        let wrap_coef = gpu.storage(2);
        let euler_ab = gpu.storage(2);
        let mut loop_steps: Vec<Step> = Vec::with_capacity(b.steps.len() + 4);
        loop_steps.push(gpu.step(K_SCALE_ROW, &[&x_state, &cin_buf, &x_in], &[n_px, n_px], n_px));
        loop_steps.extend(b.steps.iter().cloned());
        loop_steps.push(gpu.step(K_EDM_WRAP, &[&x_state, &y_out, &wrap_coef, &denoised], &[n_px], n_px));
        loop_steps.push(gpu.step(K_EDM_MIX, &[&x_state, &denoised, &euler_ab, &x_next], &[n_px, n_px], n_px));
        loop_steps.push(gpu.step(K_SCALE_ROW, &[&x_next, &one_buf, &x_state], &[n_px, n_px], n_px));

        // End the builder's borrow of `gpu` before moving it into the model.
        let Builder { steps, adagn, taps, gpu: _, t: _, cc: _ } = b;

        DiamondUNet {
            gpu,
            cfg,
            cond,
            adagn,
            steps,
            loop_steps,
            x_in,
            obs_in,
            y_out,
            x_state,
            denoised,
            cin_buf,
            wrap_coef,
            euler_ab,
            taps,
        }
    }

    /// Upload the rescaled context (obs / sigma_data), once per frame.
    pub fn set_context(&self, obs_rescaled: &[f32]) {
        self.gpu.write_f32(&self.obs_in, obs_rescaled);
    }

    /// Wall-clock per-kernel profile of one forward: runs the recorded steps
    /// ONE SUBMIT EACH (adds per-submit overhead — ranking, not gospel; the
    /// production path is a single submit) and aggregates ms by kernel.
    /// Returns (kernel name, total ms, dispatch count) sorted by time.
    pub fn profile_forward(
        &self,
        noisy_scaled: &[f32],
        c_noise: f32,
        actions: &[u32],
    ) -> Vec<(&'static str, f64, u32)> {
        let cond = self.cond.cond(c_noise, actions);
        for (site, gb) in &self.adagn {
            self.gpu.write_f32(gb, &site.gb(&cond));
        }
        self.gpu.write_f32(&self.x_in, noisy_scaled);
        let mut agg: std::collections::HashMap<usize, (f64, u32)> = Default::default();
        // The kernel index comes off the step itself: `Gpu::step` attaches
        // `StepMeta { kernel, params, threads }` on every backend, recording the
        // CALLER's kind (not the upgraded one), which is exactly what this used
        // to keep in a `kinds: Vec<usize>` running parallel to `steps`. A
        // parallel vec can only ever agree with or drift from the thing it
        // parallels, and drift here mislabels the profile rather than failing.
        for step in &self.steps {
            let kind = step.meta().expect("Gpu::step attaches StepMeta").kernel;
            let t0 = std::time::Instant::now();
            self.gpu.submit(&[], std::slice::from_ref(step));
            self.gpu.poll_wait();
            let e = agg.entry(kind).or_insert((0.0, 0));
            e.0 += t0.elapsed().as_secs_f64() * 1e3;
            e.1 += 1;
        }
        let mut out: Vec<(&'static str, f64, u32)> =
            agg.into_iter().map(|(k, (ms, n))| (KERNELS[k].0, ms, n)).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        out
    }

    /// Read a named intermediate tap after a forward (parity debugging).
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        self.taps
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, buf, len)| self.gpu.read(buf, *len))
    }

    /// Tap names in graph order.
    pub fn tap_names(&self) -> Vec<String> {
        self.taps.iter().map(|(n, _, _)| n.clone()).collect()
    }

    /// Full on-device denoising of one frame: upload the unit-noise init once,
    /// then per sigma write only the tiny coef buffers + the conditioned
    /// gamma/betas and submit the WHOLE iteration (c_in scale -> UNet -> wrap
    /// -> Euler) — x never leaves the device; ONE readback per frame.
    /// `sigmas` is the Karras schedule incl. trailing 0.
    pub fn denoise_frame(&self, x0: &[f32], sigmas: &[f32], actions: &[u32]) -> Vec<f32> {
        let cfg = &self.cfg;
        self.gpu.write_f32(&self.x_state, x0);
        for i in 0..sigmas.len() - 1 {
            let sigma = sigmas[i];
            let next = sigmas[i + 1];
            let cs = crate::cond::conditioners(sigma, cfg.sigma_data, cfg.sigma_offset_noise);
            let cond = self.cond.cond(cs.c_noise, actions);
            for (site, gb) in &self.adagn {
                self.gpu.write_f32(gb, &site.gb(&cond));
            }
            self.gpu.write_f32(&self.cin_buf, &[cs.c_in]);
            self.gpu.write_f32(&self.wrap_coef, &[cs.c_skip, cs.c_out]);
            let dt = next - sigma;
            // Euler: x' = (1 + dt/sigma)*x - (dt/sigma)*denoised.
            self.gpu.write_f32(&self.euler_ab, &[1.0 + dt / sigma, -dt / sigma]);
            self.gpu.submit(&[], &self.loop_steps);
        }
        let n = (cfg.img_channels * cfg.h * cfg.w) as usize;
        // The final Euler step (sigma_next = 0) lands exactly on `denoised`.
        self.gpu.read(&self.denoised, n)
    }

    /// One inner-model forward: F(c_in*x, c_noise, obs, act).
    /// `noisy_scaled` is the c_in-scaled noisy frame [ic*h*w].
    pub fn forward(&self, noisy_scaled: &[f32], c_noise: f32, actions: &[u32]) -> Vec<f32> {
        let cond = self.cond.cond(c_noise, actions);
        for (site, gb) in &self.adagn {
            self.gpu.write_f32(gb, &site.gb(&cond));
        }
        self.gpu.write_f32(&self.x_in, noisy_scaled);
        self.gpu.submit(&[], &self.steps);
        let n = (self.cfg.img_channels * self.cfg.h * self.cfg.w) as usize;
        self.gpu.read(&self.y_out, n)
    }
}

#[cfg(test)]
mod kernel_slots {
    /// Every slot constant must name the kernel actually sitting at that index.
    ///
    /// This did not exist while `conv_bias` and `gn_stats` sat at slots 0 and 1
    /// long after `conv_bias_reg` and `gn_part`/`gn_stats2` replaced them —
    /// they were dispatched by nothing and compiled by every backend on every
    /// device init, and two `#[allow(dead_code)]` markers are what let them
    /// stay. Removing a slot renumbers every one after it, and a stale constant
    /// then dispatches the WRONG pipeline: silently different numbers if the
    /// bind-group arities happen to match, a panic if they do not
    /// (`docs/lessons.md` #13). This test is what makes that a red build.
    #[test]
    fn every_slot_constant_names_its_kernel() {
        for (slot, name) in [
            (super::K_GN_APPLY, "gn_apply"),
            (super::K_SILU, "silu"),
            (super::K_ADD2, "add2"),
            (super::K_CONCAT2, "concat2"),
            (super::K_UPSAMPLE2, "upsample2"),
            (super::K_NCHW_NLC, "nchw_nlc"),
            (super::K_NLC_NCHW, "nlc_nchw"),
            (super::K_ATTN_SCORES, "attn_scores_bidir"),
            (super::K_ATTN_SOFTMAX, "attn_softmax_bidir"),
            (super::K_ATTN_APPLY, "attn_apply_bidir"),
            (super::K_SCALE_ROW, "scale_row"),
            (super::K_EDM_MIX, "edm_mix"),
            (super::K_EDM_WRAP, "edm_wrap"),
            (super::K_GN_PART, "gn_part"),
            (super::K_GN_STATS2, "gn_stats2"),
            (super::K_CONV_BIAS_REG, "conv_bias_reg"),
        ] {
            assert_eq!(super::KERNELS[slot].0, name, "slot {slot}");
        }
        // ...and the table has nothing in it the list above does not cover, so
        // a kernel that stops being dispatched cannot quietly stay registered.
        assert_eq!(super::KERNELS.len(), 16);
        assert!(super::KERNELS.iter().all(|(n, s)| !n.is_empty() && !s.is_empty()));
    }
}
