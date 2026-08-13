// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build the DIAMOND conditional EDM UNet denoiser (the *inner model* of
//! `diamond::model::DiamondUNet`) as an fp32 ONNX graph for OpenVINO
//! whole-graph compilation, plus the [`WmSession`] that compiles/runs it.
//! Pure Rust to *produce* the graph — no NPU / OpenVINO needed.
//!
//! The graph starts AFTER the host conditioning (Fourier + action embedding +
//! cond MLP stay on the host, see `diamond::cond`) and ends at the inner
//! model output `F` — the EDM wrap (c_skip/c_out + quantize) and the Euler
//! step stay host-side too. Inputs:
//!   - `noisy_scaled` `[1,ic,H,W]`   (c_in pre-multiplied on the host)
//!   - `obs_rescaled` `[1,nsc*ic,H,W]` (obs / sigma_data)
//!   - `cond`         `[1,cond_channels]`
//! Output: `model_out` `[1,ic,H,W]`.
//!
//! Semantics mirror `diamond::model` 1:1:
//!   - AdaGroupNorm = non-affine GroupNorm (eps 1e-5 INSIDE the sqrt), then
//!     `y = xhat*(1+scale) + shift` with `[scale||shift] = Gemm(cond)` per site.
//!   - SiLU = Sigmoid+Mul; skip concat = `Concat(x, skip)` axis 1; the up path
//!     upsample = nearest x2 `Resize` + 3x3 conv; downsample = stride-2 conv.
//!   - Mid/attn: fused qkv 1x1 conv, 8-dim heads, softmax over keys, out_proj
//!     1x1 conv, and the residual adds the NORMED input (reference quirk).
//!
//! brain-npu does not depend on brain-wm-diamond: [`WmUnetConfig`] mirrors the
//! graph-relevant fields of `diamond::DiamondConfig` and the host glue in
//! `diamond::npu` converts.

use std::collections::HashMap;

use onnx::builder::GraphBuilder;
use onnx::graph::Node;

/// UNet architecture parameters (see `diamond::DiamondConfig`).
#[derive(Clone, Debug)]
pub struct WmUnetConfig {
    pub img_channels: u32,
    /// Number of conditioning context frames.
    pub num_steps_conditioning: u32,
    pub cond_channels: u32,
    /// Residual blocks per level (down path; up path gets `+1`).
    pub depths: Vec<u32>,
    pub channels: Vec<u32>,
    pub attn_depths: Vec<bool>,
    pub h: u32,
    pub w: u32,
}

/// Host tensors by (stripped) reference name: `name -> (shape, data)` — the
/// same shape as `diamond::Tensors`.
pub type W = HashMap<String, (Vec<usize>, Vec<f32>)>;

const GN_EPS: f32 = 1e-5;
/// Fixed attention head dim (matches `diamond::model::ATTN_HEAD_DIM`).
const ATTN_HEAD_DIM: i64 = 8;

fn num_groups(c: i64) -> i64 {
    (c / 32).max(1)
}

/// Assemble the DIAMOND UNet inner-model graph into `g` from checkpoint
/// tensors `w` (reference names, `denoiser.inner_model.` prefix stripped).
pub fn build_diamond_graph(cfg: &WmUnetConfig, w: &W, g: &mut GraphBuilder) {
    let ic = cfg.img_channels as i64;
    let nsc = cfg.num_steps_conditioning as i64;
    let cc = cfg.cond_channels as i64;
    let (h0, w0) = (cfg.h as i64, cfg.w as i64);
    let n_lv = cfg.channels.len();

    let mut tp = Topo { b: crate::topo::TopoBase::new(g), w, cc };
    tp.g.input_f32("noisy_scaled", &[1, ic, h0, w0]);
    tp.g.input_f32("obs_rescaled", &[1, nsc * ic, h0, w0]);
    tp.g.input_f32("cond", &[1, cc]);
    tp.g.init_f32("wm_one", &[1], vec![1.0]);
    tp.g.init_f32("wm_gn_eps", &[1], vec![GN_EPS]);
    tp.g.init_f32("wm_attn_scale", &[1], vec![1.0 / (ATTN_HEAD_DIM as f32).sqrt()]);

    // conv_in over cat(obs, noisy) — obs FIRST (model.rs concat order).
    let cat = tp.concat2("obs_rescaled", "noisy_scaled");
    let mut x = tp.conv("conv_in", &cat, 1, 1);

    // Down path. Skips per level: (x_down, resblock outputs...).
    let mut hw = (h0, w0);
    let mut d_skips: Vec<Vec<String>> = vec![];
    for i in 0..n_lv {
        let c1 = cfg.channels[i.saturating_sub(1)] as i64;
        let c2 = cfg.channels[i] as i64;
        // downsamples[i]: identity for i==0, stride-2 conv otherwise.
        if i > 0 {
            x = tp.conv(&format!("unet.downsamples.{i}.conv"), &x, 2, 1);
            hw = ((hw.0 + 1) / 2, (hw.1 + 1) / 2);
        }
        let mut level = vec![x.clone()];
        for r in 0..cfg.depths[i] {
            let cin = if r == 0 { c1 } else { c2 };
            x = tp.resblock(
                &format!("unet.d_blocks.{i}.resblocks.{r}"),
                &x,
                cin,
                c2,
                cfg.attn_depths[i],
                hw.0,
                hw.1,
            );
            level.push(x.clone());
        }
        d_skips.push(level);
    }

    // Mid: 2 resblocks, always attention.
    let cl = *cfg.channels.last().unwrap() as i64;
    for r in 0..2 {
        x = tp.resblock(&format!("unet.mid_blocks.resblocks.{r}"), &x, cl, cl, true, hw.0, hw.1);
    }

    // Up path: u_blocks[j] pairs with d_skips[n_lv-1-j], skips reversed.
    for j in 0..n_lv {
        let i = n_lv - 1 - j;
        let c1 = cfg.channels[i.saturating_sub(1)] as i64;
        let c2 = cfg.channels[i] as i64;
        if j > 0 {
            // upsamples[j]: nearest x2 then 3x3 conv.
            let up = tp.upsample(&x);
            hw = (hw.0 * 2, hw.1 * 2);
            x = tp.conv(&format!("unet.upsamples.{j}.conv"), &up, 1, 1);
        }
        let skips = &d_skips[i];
        let n = cfg.depths[i] as usize;
        for r in 0..=n {
            let skip = &skips[n - r]; // reversed order
            let cat = tp.concat2(&x, skip);
            let (cin, cout) = if r < n { (2 * c2, c2) } else { (c1 + c2, c1) };
            x = tp.resblock(
                &format!("unet.u_blocks.{j}.resblocks.{r}"),
                &cat,
                cin,
                cout,
                cfg.attn_depths[i],
                hw.0,
                hw.1,
            );
        }
    }
    assert_eq!(hw, (h0, w0), "UNet did not return to input resolution");

    // Head: affine GroupNorm -> SiLU -> conv_out.
    let c0 = cfg.channels[0] as i64;
    let hn = tp.affine_gn("norm_out.norm", &x, c0, hw.0, hw.1);
    let hs = tp.silu(&hn);
    let y = tp.conv("conv_out", &hs, 1, 1);
    tp.node("Identity", &[&y], "model_out");
    tp.g.output_f32("model_out", &[1, ic, h0, w0]);
}

/// Graph-construction state.
struct Topo<'a> {
    b: crate::topo::TopoBase<'a>,
    w: &'a W,
    /// cond_channels.
    cc: i64,
}

// Identical DSL helpers live on `TopoBase` (crate::topo); dialect-specific ones
// (tagged unary, model emitters) stay here.
impl<'a> std::ops::Deref for Topo<'a> {
    type Target = crate::topo::TopoBase<'a>;
    fn deref(&self) -> &Self::Target { &self.b }
}
impl<'a> std::ops::DerefMut for Topo<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.b }
}

impl Topo<'_> {


    fn host(&self, name: &str) -> &(Vec<usize>, Vec<f32>) {
        self.w.get(name).unwrap_or_else(|| panic!("diamond onnx: missing tensor {name}"))
    }

    /// Register checkpoint tensor `name` as an initializer with its own shape.
    fn init(&mut self, name: &str) {
        if self.has(name) {
            return;
        }
        let (shape, data) = self.host(name).clone();
        let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        self.g.init_f32(name, &dims, data);
    }

    /// Register checkpoint tensor `name` under `alias` with explicit `dims`
    /// (e.g. a `[C]` affine vector broadcast as `[1,C,1,1]`).
    fn init_as(&mut self, name: &str, alias: &str, dims: &[i64]) {
        if self.has(alias) {
            return;
        }
        let data = self.host(name).1.clone();
        self.g.init_f32(alias, dims, data);
    }


    fn unary(&mut self, op: &str, x: &str, tag: &str) -> String {
        let o = self.tmp(tag);
        self.node(op, &[x], &o);
        o
    }

    fn binary(&mut self, op: &str, a: &str, b: &str, tag: &str) -> String {
        let o = self.tmp(tag);
        self.node(op, &[a, b], &o);
        o
    }

    fn add(&mut self, a: &str, b: &str) -> String {
        self.binary("Add", a, b, "add")
    }

    fn mul(&mut self, a: &str, b: &str) -> String {
        self.binary("Mul", a, b, "mul")
    }

    fn reshape_to(&mut self, x: &str, shape: &[i64]) -> String {
        let sname = format!("shape_{}", self.n + 1);
        self.g.init_i64(&sname, &[shape.len() as i64], shape.to_vec());
        self.binary("Reshape", x, &sname, "rs")
    }

    fn transpose(&mut self, x: &str, perm: &[i64]) -> String {
        let o = self.tmp("tr");
        self.g.add(Node::new("Transpose", &[x], &[&o]).attr_ints("perm", perm));
        o
    }

    /// Mean over axis 2 (keepdims) — the per-group reduction of GroupNorm.
    fn mean2(&mut self, x: &str) -> String {
        let o = self.tmp("gn_mean");
        self.g.add(Node::new("ReduceMean", &[x], &[&o]).attr_ints("axes", &[2]).attr_int("keepdims", 1));
        o
    }

    /// `Concat(a, b)` along the channel axis (axis 1).
    fn concat2(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("cat");
        self.g.add(Node::new("Concat", &[a, b], &[&o]).attr_int("axis", 1));
        o
    }

    /// 2D conv (+bias) `prefix.{weight,bias}`; kernel size read off the
    /// checkpoint weight shape `[cout,cin,k,k]` (== ONNX `Conv` layout).
    fn conv(&mut self, prefix: &str, x: &str, stride: i64, pad: i64) -> String {
        let wname = format!("{prefix}.weight");
        let k = self.host(&wname).0[2] as i64;
        self.init(&wname);
        let bname = format!("{prefix}.bias");
        self.init(&bname);
        let o = self.tmp("conv");
        self.g.add(
            Node::new("Conv", &[x, &wname, &bname], &[&o])
                .name(prefix)
                .attr_ints("kernel_shape", &[k, k])
                .attr_ints("strides", &[stride, stride])
                .attr_ints("pads", &[pad, pad, pad, pad])
                .attr_ints("dilations", &[1, 1])
                .attr_int("group", 1),
        );
        o
    }

    fn silu(&mut self, x: &str) -> String {
        let s = self.unary("Sigmoid", x, "sig");
        self.mul(x, &s)
    }

    /// Non-affine GroupNorm core: `xhat = (x - mean_g) / sqrt(var_g + eps)`
    /// over channel groups (eps INSIDE the sqrt, matching brain's gn kernels).
    fn gn_core(&mut self, x: &str, c: i64, h: i64, w: i64) -> String {
        let g = num_groups(c);
        let xg = self.reshape_to(x, &[1, g, (c / g) * h * w]);
        let mean = self.mean2(&xg);
        let xc = self.binary("Sub", &xg, &mean, "gn_xc");
        let sq = self.mul(&xc, &xc);
        let var = self.mean2(&sq);
        let ve = self.add(&var, "wm_gn_eps");
        let std = self.unary("Sqrt", &ve, "gn_std");
        let xn = self.binary("Div", &xc, &std, "gn_div");
        self.reshape_to(&xn, &[1, c, h, w])
    }

    /// AdaGroupNorm: `[scale||shift] = Gemm(cond, W^T)+b`, then
    /// `y = xhat*(1+scale) + shift` (scale/shift broadcast `[1,C,1,1]`).
    fn adagn(&mut self, prefix: &str, x: &str, c: i64, h: i64, w: i64) -> String {
        let wname = format!("{prefix}.linear.weight");
        let bname = format!("{prefix}.linear.bias");
        assert_eq!(
            self.host(&wname).0,
            vec![2 * c as usize, self.cc as usize],
            "{wname}: unexpected shape"
        );
        self.init(&wname);
        self.init(&bname);
        let ss = self.tmp("adagn_ss");
        self.g.add(
            Node::new("Gemm", &["cond", &wname, &bname], &[&ss])
                .name(&format!("{prefix}/Gemm"))
                .attr_int("transB", 1),
        );
        let sc = self.tmp("adagn_scale");
        let sh = self.tmp("adagn_shift");
        self.g.add(Node::new("Split", &[&ss], &[&sc, &sh]).attr_int("axis", 1));
        let sc = self.reshape_to(&sc, &[1, c, 1, 1]);
        let sh = self.reshape_to(&sh, &[1, c, 1, 1]);
        let gamma = self.add(&sc, "wm_one"); // 1 + scale
        let xhat = self.gn_core(x, c, h, w);
        let scaled = self.mul(&xhat, &gamma);
        self.add(&scaled, &sh)
    }

    /// GroupNorm with a STATIC affine gamma/beta from `prefix.{weight,bias}`.
    fn affine_gn(&mut self, prefix: &str, x: &str, c: i64, h: i64, w: i64) -> String {
        let gname = format!("{prefix}.weight.c111");
        let bname = format!("{prefix}.bias.c111");
        self.init_as(&format!("{prefix}.weight"), &gname, &[1, c, 1, 1]);
        self.init_as(&format!("{prefix}.bias"), &bname, &[1, c, 1, 1]);
        let xhat = self.gn_core(x, c, h, w);
        let scaled = self.mul(&xhat, &gname);
        self.add(&scaled, &bname)
    }

    /// Nearest x2 upsample = ONNX `Resize` (asymmetric/floor — matches
    /// brain's `upsample2` integer doubling; same emission as the YOLO neck).
    fn upsample(&mut self, x: &str) -> String {
        let scales = self.tmp("up_scales");
        self.g.init_f32(&scales, &[4], vec![1.0, 1.0, 2.0, 2.0]);
        let o = self.tmp("up");
        // Resize inputs: (X, roi, scales). roi is skipped via an empty input "".
        self.g.add(
            Node::new("Resize", &[x, "", &scales], &[&o])
                .attr_str("mode", "nearest")
                .attr_str("coordinate_transformation_mode", "asymmetric")
                .attr_str("nearest_mode", "floor"),
        );
        o
    }

    /// Self-attention (norm -> fused qkv 1x1 -> bidirectional attention ->
    /// out 1x1 -> residual). The residual adds the NORMED input (reference
    /// quirk, `blocks.py::SelfAttention2d.forward`).
    fn attn(&mut self, prefix: &str, x: &str, c: i64, h: i64, w: i64) -> String {
        let t = h * w;
        let heads = (c / ATTN_HEAD_DIM).max(1);
        assert_eq!(heads * ATTN_HEAD_DIM, c, "{prefix}: channels not divisible into 8-dim heads");
        let normed = self.affine_gn(&format!("{prefix}.norm.norm"), x, c, h, w);
        let qkv = self.conv(&format!("{prefix}.qkv_proj"), &normed, 1, 0); // [1,3C,H,W]
        // Channel layout is (s, head, dim): s in {q,k,v}, within each the
        // kernel reads head h at offset h*8+d — exactly this reshape.
        let qkv = self.reshape_to(&qkv, &[3, heads, ATTN_HEAD_DIM, t]);
        let qkv = self.transpose(&qkv, &[0, 1, 3, 2]); // [3,heads,T,hd]
        let q = self.tmp("attn_q");
        let k = self.tmp("attn_k");
        let v = self.tmp("attn_v");
        self.g.add(Node::new("Split", &[&qkv], &[&q, &k, &v]).attr_int("axis", 0));
        let kt = self.transpose(&k, &[0, 1, 3, 2]); // [1,heads,hd,T]
        let scores = self.binary("MatMul", &q, &kt, "attn_scores"); // [1,heads,T,T]
        let scores = self.mul(&scores, "wm_attn_scale"); // / sqrt(hd)
        let probs = self.tmp("attn_probs");
        self.g.add(Node::new("Softmax", &[&scores], &[&probs]).attr_int("axis", 3));
        let ctx = self.binary("MatMul", &probs, &v, "attn_ctx"); // [1,heads,T,hd]
        let ctx = self.transpose(&ctx, &[0, 1, 3, 2]); // [1,heads,hd,T]
        let ctx = self.reshape_to(&ctx, &[1, c, h, w]);
        let proj = self.conv(&format!("{prefix}.out_proj"), &ctx, 1, 0);
        self.add(&normed, &proj)
    }

    /// One reference ResBlock: `r = proj(x); y = conv2(silu(norm2(conv1(silu(
    /// norm1(x)))))) + r`; then optional attention.
    #[allow(clippy::too_many_arguments)]
    fn resblock(
        &mut self,
        prefix: &str,
        x: &str,
        cin: i64,
        cout: i64,
        attn: bool,
        h: i64,
        w: i64,
    ) -> String {
        let r = if cin != cout {
            self.conv(&format!("{prefix}.proj"), x, 1, 0)
        } else {
            x.to_string()
        };
        let n1 = self.adagn(&format!("{prefix}.norm1"), x, cin, h, w);
        let s1 = self.silu(&n1);
        let c1 = self.conv(&format!("{prefix}.conv1"), &s1, 1, 1);
        let n2 = self.adagn(&format!("{prefix}.norm2"), &c1, cout, h, w);
        let s2 = self.silu(&n2);
        let c2 = self.conv(&format!("{prefix}.conv2"), &s2, 1, 1);
        let y = self.add(&c2, &r);
        if attn {
            self.attn(&format!("{prefix}.attn"), &y, cout, h, w)
        } else {
            y
        }
    }
}

// ---------------------------------------------------------------------------
// WmSession — compile + run the exported graph via OpenVINO. Real on x86_64
// linux/windows (runtime-linked, like crate::openvino::real); a stub with the
// same API reports Unsupported elsewhere.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")))]
mod wm_session {
    use crate::openvino::{NpuConfig, NpuDevice, NpuError, PerfHint};
    use openvino::{Core, DeviceType, ElementType, RwPropertyKey, Shape, Tensor};
    use std::path::Path;

    fn matches_base(id: &str, base: &str) -> bool {
        id == base || id.starts_with(&format!("{base}."))
    }

    fn dev_to_ov(d: NpuDevice) -> DeviceType<'static> {
        match d {
            NpuDevice::Npu => DeviceType::NPU,
            NpuDevice::Cpu => DeviceType::CPU,
            NpuDevice::Gpu => DeviceType::GPU,
            NpuDevice::Auto => DeviceType::Other("AUTO".into()),
        }
    }

    /// Resolve the requested device against what's present (NPU -> GPU -> CPU
    /// when `allow_fallback`), same policy as `crate::openvino::real`.
    fn resolve(cfg: &NpuConfig, avail: &[String]) -> Result<DeviceType<'static>, NpuError> {
        if matches!(cfg.device, NpuDevice::Auto) {
            return Ok(dev_to_ov(cfg.device));
        }
        let base = cfg.device.ov_str();
        if avail.iter().any(|d| matches_base(d, base)) {
            return Ok(dev_to_ov(cfg.device));
        }
        if cfg.allow_fallback {
            for (cand, dev) in [("GPU", DeviceType::GPU), ("CPU", DeviceType::CPU)] {
                if avail.iter().any(|d| matches_base(d, cand)) {
                    eprintln!(
                        "brain wm npu: requested device {base} not available; falling back to {cand}"
                    );
                    return Ok(dev);
                }
            }
        }
        Err(NpuError::DeviceUnavailable(format!(
            "{base} not in available OpenVINO devices {avail:?}"
        )))
    }

    /// Best-effort OpenVINO properties (perf hint + compile cache).
    fn apply_props(core: &mut Core, device: &DeviceType<'static>, cfg: &NpuConfig) {
        let hint = match cfg.perf_hint {
            PerfHint::Latency => "LATENCY",
            PerfHint::Throughput => "THROUGHPUT",
        };
        let mut set = |key: RwPropertyKey, val: &str| {
            if let Err(e) = core.set_property(device, &key, val) {
                eprintln!(
                    "brain wm npu: set_property {} = {val} failed ({e:?}); ignoring",
                    key.as_ref()
                );
            }
        };
        set(RwPropertyKey::HintPerformanceMode, hint);
        if let Some(dir) = &cfg.cache_dir {
            if let Some(s) = dir.to_str() {
                set(RwPropertyKey::CacheDir, s);
            }
        }
    }

    /// A compiled DIAMOND UNet inner-model graph: three named f32 inputs
    /// (`noisy_scaled` `[1,ic,H,W]`, `obs_rescaled` `[1,nsc*ic,H,W]`, `cond`
    /// `[1,cc]`) -> `model_out` `[1,ic,H,W]`. The sampler stays host-side.
    pub struct WmSession {
        // `core` must outlive the compiled model / request (owns the plugin).
        _core: Core,
        request: openvino::InferRequest,
        img: [i64; 4],
        obs: [i64; 4],
        cond: [i64; 2],
        device: String,
    }

    impl WmSession {
        /// Compile from an ONNX file for the configured device. Input shapes
        /// come from the caller (known from the DIAMOND config).
        #[allow(clippy::too_many_arguments)]
        pub fn load_path(
            onnx_path: &Path,
            cfg: &NpuConfig,
            img_c: usize,
            ctx_c: usize,
            h: usize,
            w: usize,
            cond_c: usize,
        ) -> Result<Self, NpuError> {
            let bytes = std::fs::read(onnx_path)
                .map_err(|e| NpuError::Other(format!("read {}: {e}", onnx_path.display())))?;
            Self::load_bytes(&bytes, cfg, img_c, ctx_c, h, w, cond_c)
        }

        /// Compile ONNX bytes directly (e.g. an in-memory fp32 export).
        #[allow(clippy::too_many_arguments)]
        pub fn load_bytes(
            bytes: &[u8],
            cfg: &NpuConfig,
            img_c: usize,
            ctx_c: usize,
            h: usize,
            w: usize,
            cond_c: usize,
        ) -> Result<Self, NpuError> {
            // available_devices() both probes the runtime (RuntimeNotFound on
            // a machine without OpenVINO) and, in the real impl, makes the pip
            // wheel's libraries discoverable in-process before Core::new.
            let avail = crate::openvino::available_devices()?;
            let mut core =
                Core::new().map_err(|e| NpuError::RuntimeNotFound(format!("{e:?}")))?;
            let device = resolve(cfg, &avail)?;
            apply_props(&mut core, &device, cfg);
            let model = core
                .read_model_from_buffer(bytes, None)
                .map_err(|e| NpuError::Other(format!("read_model (ONNX): {e:?}")))?;
            let mut compiled = core.compile_model(&model, device.to_owned()).map_err(|e| {
                NpuError::Other(format!("compile_model on {}: {e:?}", device.as_ref()))
            })?;
            let request = compiled
                .create_infer_request()
                .map_err(|e| NpuError::Other(format!("create_infer_request: {e:?}")))?;
            Ok(WmSession {
                _core: core,
                request,
                img: [1, img_c as i64, h as i64, w as i64],
                obs: [1, ctx_c as i64, h as i64, w as i64],
                cond: [1, cond_c as i64],
                device: device.as_ref().to_string(),
            })
        }

        /// The OpenVINO device the model was compiled for (e.g. "NPU").
        pub fn device(&self) -> &str {
            &self.device
        }

        fn set_f32(
            &mut self,
            holders: &mut Vec<Tensor>,
            name: &str,
            dims: &[i64],
            data: &[f32],
        ) -> Result<(), NpuError> {
            let want: i64 = dims.iter().product();
            if data.len() as i64 != want {
                return Err(NpuError::Other(format!(
                    "{name}: expected {want} f32 (shape {dims:?}), got {}",
                    data.len()
                )));
            }
            let shape = Shape::new(dims).map_err(|e| NpuError::Other(format!("{e:?}")))?;
            let mut t = Tensor::new(ElementType::F32, &shape)
                .map_err(|e| NpuError::Other(format!("{e:?}")))?;
            t.get_data_mut::<f32>()
                .map_err(|e| NpuError::Other(format!("{e:?}")))?
                .copy_from_slice(data);
            self.request
                .set_tensor(name, &t)
                .map_err(|e| NpuError::Other(format!("set {name}: {e:?}")))?;
            holders.push(t);
            Ok(())
        }

        /// One inner-model forward: `F(noisy_scaled, obs_rescaled, cond)`.
        pub fn run(
            &mut self,
            noisy_scaled: &[f32],
            obs_rescaled: &[f32],
            cond: &[f32],
        ) -> Result<Vec<f32>, NpuError> {
            let (img, obs, cnd) = (self.img, self.obs, self.cond);
            let mut holders: Vec<Tensor> = Vec::with_capacity(3);
            self.set_f32(&mut holders, "noisy_scaled", &img, noisy_scaled)?;
            self.set_f32(&mut holders, "obs_rescaled", &obs, obs_rescaled)?;
            self.set_f32(&mut holders, "cond", &cnd, cond)?;
            self.request.infer().map_err(|e| NpuError::Other(format!("infer: {e:?}")))?;
            let t = self
                .request
                .get_tensor("model_out")
                .map_err(|e| NpuError::Other(format!("get model_out: {e:?}")))?;
            let out = t.get_data::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.to_vec();
            drop(holders);
            Ok(out)
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))))]
mod wm_session {
    use crate::openvino::{NpuConfig, NpuError};
    use std::path::Path;

    fn unsupported<T>() -> Result<T, NpuError> {
        Err(NpuError::Unsupported(std::env::consts::ARCH.to_string()))
    }

    /// A compiled DIAMOND UNet graph. Never constructible on unsupported targets.
    pub struct WmSession {
        _priv: (),
    }

    impl WmSession {
        #[allow(clippy::too_many_arguments)]
        pub fn load_path(
            _onnx_path: &Path,
            _cfg: &NpuConfig,
            _img_c: usize,
            _ctx_c: usize,
            _h: usize,
            _w: usize,
            _cond_c: usize,
        ) -> Result<Self, NpuError> {
            unsupported()
        }
        #[allow(clippy::too_many_arguments)]
        pub fn load_bytes(
            _bytes: &[u8],
            _cfg: &NpuConfig,
            _img_c: usize,
            _ctx_c: usize,
            _h: usize,
            _w: usize,
            _cond_c: usize,
        ) -> Result<Self, NpuError> {
            unsupported()
        }
        pub fn device(&self) -> &str {
            ""
        }
        pub fn run(
            &mut self,
            _noisy_scaled: &[f32],
            _obs_rescaled: &[f32],
            _cond: &[f32],
        ) -> Result<Vec<f32>, NpuError> {
            unsupported()
        }
    }
}

pub use wm_session::WmSession;
