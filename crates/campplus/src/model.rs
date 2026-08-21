// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The CAM++ forward graph: a 2D-conv `FCM` stem feeding a D-TDNN with
//! context-aware masking (`CAMLayer`).
//!
//! # Dispatch choices, and why
//!
//! * **`FCM` (`head.*`) runs as `conv3d` with a dummy `T=1` axis**, not
//!   `conv2d`: this stem downsamples FREQUENCY only (stride `(2,1)`), which
//!   no 2D conv kernel in `crates/kernels` supports (`conv2d`/`conv2d_gd` all
//!   take one scalar `stride`). `conv3d`'s NCTHW layout at `T=1` is
//!   bit-identical to NCHW and it happens to carry a fused bias - see
//!   `crate::import`'s module doc.
//! * **`xvector.*` runs as `audio::conv::Conv1d` + `conv1d_fwd`**, with bias
//!   added separately via `add_chan_inplace` (the kernel has no bias input).
//! * **The `FCM -> xvector.tdnn` reshape (`(C,Hf,T) -> (C*Hf,T)`) is not a
//!   dispatch at all.** Both sides are row-major NCHW with `W=T`, so merging
//!   the channel and frequency axes is exactly how the buffer already reads
//!   in memory - the same buffer is simply reinterpreted at a different
//!   `(cin, l)`.
//! * **`CAMLayer`'s context (`x.mean(-1) + seg_pooling(x)`) runs on the
//!   HOST**, once per layer (52 times per forward). `seg_pooling` is a FIXED
//!   100-frame window with `ceil_mode` (PyTorch/ONNX `AvgPool1d(100, 100,
//!   ceil_mode=True)`: the tail window divides by however many real frames
//!   it covers, not by 100), which is a different rule from every pooling
//!   kernel in `crates/kernels` - `avgpool2d.wgsl` is an ADAPTIVE box pool
//!   (equal-sized windows over the whole extent), not a fixed-stride one. A
//!   CAM++ forward runs once per reference clip (not per token, not
//!   per-timestep of the LM this feeds), so a host round trip here costs
//!   nothing a real caller would notice, and it is the same idiom
//!   `vision::BatchNorm` already uses to pack eval-mode running stats -
//!   composing over inventing a kernel whose one caller is not a hot path.
//! * **The final `StatsPool` (`cat([mean, std])`) and `dense` + affine-free
//!   BatchNorm1d run on the HOST too** - once per forward, over a
//!   `[1024] -> [192]` vector. `std` is the UNBIASED estimator
//!   (`torch.std(unbiased=True)`, divisor `L-1`), confirmed against the
//!   ONNX graph's own `Mul(var, N) / (N-1)` chain (`crate::import`'s
//!   `dump.py` node trace), not assumed from the summary architecture.
//!
//! Swedish Embedded AB implements solutions for from-scratch, dependency-light
//! neural network inference on constrained and embedded targets for its
//! clients. If your team needs expertise in porting speech/audio models to a
//! from-scratch GPU/CPU engine, you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::collections::HashMap;

use audio::conv::{conv1d_fwd, Conv1d, ConvKernels};
use gpu_core::{DeviceBuffer, Gpu};

use crate::config::CampplusConfig;
use crate::import::bn_schedule;
use onnx::walk::Tensors;

/// Every kernel this crate dispatches, by name (resolved by [`kernel`] -
/// a bare pipeline index means nothing outside the list that declared it).
pub const PIPELINES: &[(&str, &str)] = &[
    ("conv3d", kernels::CONV3D),
    ("conv1d", kernels::CONV1D),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("bn_eval", kernels::BN_EVAL),
    ("relu_inplace", kernels::RELU_INPLACE),
    ("sigmoid", kernels::SIGMOID),
    ("mul", kernels::MUL),
    ("concat2", kernels::CONCAT2),
    ("add2", kernels::ADD2),
];

fn kernel(name: &str) -> usize {
    PIPELINES.iter().position(|(n, _)| *n == name).unwrap_or_else(|| panic!("kernel `{name}` is not in campplus::PIPELINES"))
}

fn dget<'a>(w: &'a HashMap<String, DeviceBuffer>, k: &str) -> &'a DeviceBuffer {
    w.get(k).unwrap_or_else(|| panic!("campplus: missing device tensor {k}"))
}

fn hget<'a>(t: &'a Tensors, k: &str) -> &'a [f32] {
    &t.get(k).unwrap_or_else(|| panic!("campplus: missing tensor {k}")).1
}

fn pack2(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(2 * a.len());
    for i in 0..a.len() {
        v.push(a[i]);
        v.push(b[i]);
    }
    v
}

fn relu_inplace(gpu: &Gpu, x: &DeviceBuffer, n: u32) {
    let s = gpu.step(kernel("relu_inplace"), &[x], &[n], n);
    gpu.submit(&[], &[s]);
}

fn sigmoid(gpu: &Gpu, x: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let y = gpu.storage(n as u64);
    let s = gpu.step(kernel("sigmoid"), &[x, &y], &[n], n);
    gpu.submit(&[], &[s]);
    y
}

fn mul(gpu: &Gpu, a: &DeviceBuffer, b: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let y = gpu.storage(n as u64);
    let s = gpu.step(kernel("mul"), &[a, b, &y], &[n], n);
    gpu.submit(&[], &[s]);
    y
}

fn add2(gpu: &Gpu, a: &DeviceBuffer, b: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let y = gpu.storage(n as u64);
    let s = gpu.step(kernel("add2"), &[a, b, &y], &[n], n);
    gpu.submit(&[], &[s]);
    y
}

/// A per-channel `[N=1, C, T=1]` -> `[N=1, C, L]` broadcast add, in place -
/// `bn_eval`'s bias companion for a plain `Conv1d` bias.
fn add_chan_inplace(gpu: &Gpu, y: &DeviceBuffer, bias: &DeviceBuffer, c: u32, l: u32) {
    let s = gpu.step(kernel("add_chan_inplace"), &[y, bias], &[c * l, c, l], c * l);
    gpu.submit(&[], &[s]);
}

/// `y = conv1d(x, w) [+ bias]`. Returns `(y, out_len)`.
#[allow(clippy::too_many_arguments)]
fn conv1d(
    gpu: &Gpu,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    bias: Option<&DeviceBuffer>,
    cin: u32,
    l: u32,
    cout: u32,
    k: u32,
    stride: u32,
    pad: u32,
    dilation: u32,
) -> (DeviceBuffer, u32) {
    let lo = Conv1d::out_len(l, k, stride, pad, pad, dilation);
    let c = Conv1d { n: 1, cin, l, cout, k, stride, pad, dilation, groups: 1, lo };
    let y = gpu.storage((cout * lo) as u64);
    let kk = ConvKernels { fwd: kernel("conv1d"), dx: usize::MAX, dw: usize::MAX };
    let s = conv1d_fwd(gpu, &kk, &c, x, w, &y);
    gpu.submit(&[], &[s]);
    if let Some(b) = bias {
        add_chan_inplace(gpu, &y, b, cout, lo);
    }
    (y, lo)
}

/// Eval-mode BatchNorm over `[C, L]` (bn_eval's NCHW with `H=L, W=1`),
/// optionally fusing a ReLU (`act=1`).
fn bn_eval(gpu: &Gpu, x: &DeviceBuffer, mv: &DeviceBuffer, gb: &DeviceBuffer, c: u32, l: u32, act: u32) -> DeviceBuffer {
    let n = c * l;
    let y = gpu.storage(n as u64);
    let s = gpu.step(kernel("bn_eval"), &[x, mv, gb, &y], &[1, c, l, 1, act], n);
    gpu.submit(&[], &[s]);
    y
}

fn concat2(gpu: &Gpu, a: &DeviceBuffer, b: &DeviceBuffer, ca: u32, cb: u32, l: u32) -> DeviceBuffer {
    let n = (ca + cb) * l;
    let y = gpu.storage(n as u64);
    let s = gpu.step(kernel("concat2"), &[a, b, &y], &[1, ca, cb, l, 1], n);
    gpu.submit(&[], &[s]);
    y
}

/// An NCHW-with-`N=1` shape for the `FCM` 2D stem.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Shape3 {
    c: u32,
    h: u32,
    w: u32,
}
impl Shape3 {
    fn numel(&self) -> u32 {
        self.c * self.h * self.w
    }
}

/// `conv3d` with a dummy singleton `T` axis - see the module doc for why this
/// stands in for a genuinely-asymmetric-stride `conv2d`.
#[allow(clippy::too_many_arguments)]
fn conv3d_2d(
    gpu: &Gpu,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    bias: &DeviceBuffer,
    shape: Shape3,
    cout: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
) -> (DeviceBuffer, Shape3) {
    let ho = (shape.h + 2 * ph - kh) / sh + 1;
    let wo = (shape.w + 2 * pw - kw) / sw + 1;
    let out_shape = Shape3 { c: cout, h: ho, w: wo };
    let y = gpu.storage(out_shape.numel() as u64);
    let params = [1, shape.c, 1, shape.h, shape.w, cout, 1, kh, kw, 1, sh, sw, 0, ph, pw, 1, 1, ho, wo];
    let s = gpu.step(kernel("conv3d"), &[x, w, bias, &y], &params, out_shape.numel());
    gpu.submit(&[], &[s]);
    (y, out_shape)
}

/// One `BasicResBlock`: `relu(conv2(relu(conv1(x))) + shortcut(x))`, the
/// shortcut present (and strided) exactly when `stride_h != 1`.
fn basic_block(
    gpu: &Gpu,
    w: &HashMap<String, DeviceBuffer>,
    prefix: &str,
    x: &DeviceBuffer,
    shape: Shape3,
    cout: u32,
    stride_h: u32,
) -> (DeviceBuffer, Shape3) {
    let (h1, s1) = conv3d_2d(
        gpu,
        x,
        dget(w, &format!("{prefix}.conv1.weight")),
        dget(w, &format!("{prefix}.conv1.bias")),
        shape,
        cout,
        3,
        3,
        stride_h,
        1,
        1,
        1,
    );
    relu_inplace(gpu, &h1, s1.numel());
    let (h2, s2) = conv3d_2d(
        gpu,
        &h1,
        dget(w, &format!("{prefix}.conv2.weight")),
        dget(w, &format!("{prefix}.conv2.bias")),
        s1,
        cout,
        3,
        3,
        1,
        1,
        1,
        1,
    );

    let sc_buf;
    let ident: &DeviceBuffer = if stride_h != 1 {
        let (b, sshape) = conv3d_2d(
            gpu,
            x,
            dget(w, &format!("{prefix}.shortcut.weight")),
            dget(w, &format!("{prefix}.shortcut.bias")),
            shape,
            cout,
            1,
            1,
            stride_h,
            1,
            0,
            0,
        );
        debug_assert_eq!(sshape, s2, "campplus: {prefix} shortcut shape does not match the main path");
        sc_buf = b;
        &sc_buf
    } else {
        x
    };
    let out = add2(gpu, &h2, ident, s2.numel());
    relu_inplace(gpu, &out, s2.numel());
    (out, s2)
}

/// The `FCM` 2D stem: `(1, feat_dim, T) -> (fcm_out_c, T)`. The trailing
/// reshape is a reinterpretation, not a dispatch - see the module doc.
fn fcm_forward(gpu: &Gpu, w: &HashMap<String, DeviceBuffer>, x0: &DeviceBuffer, t: u32, cfg: &CampplusConfig) -> (DeviceBuffer, u32) {
    let shape0 = Shape3 { c: 1, h: cfg.feat_dim, w: t };
    let (h, s) = conv3d_2d(gpu, x0, dget(w, "head.conv1.weight"), dget(w, "head.conv1.bias"), shape0, cfg.fcm_channels, 3, 3, 1, 1, 1, 1);
    relu_inplace(gpu, &h, s.numel());

    let mut x = h;
    let mut shape = s;
    for li in 0..2usize {
        for bi in 0..2usize {
            let prefix = format!("head.layer{}.{}", li + 1, bi);
            let stride_h = if bi == 0 { 2 } else { 1 };
            let (o, os) = basic_block(gpu, w, &prefix, &x, shape, cfg.fcm_channels, stride_h);
            x = o;
            shape = os;
        }
    }

    let (h2, s2) = conv3d_2d(gpu, &x, dget(w, "head.conv2.weight"), dget(w, "head.conv2.bias"), shape, cfg.fcm_channels, 3, 3, 2, 1, 1, 1);
    relu_inplace(gpu, &h2, s2.numel());
    assert_eq!(s2.h, cfg.fcm_freq_out(), "campplus: FCM frequency output does not match feat_dim/8");
    // (C, Hf, T), row-major, IS (C*Hf, T) row-major already - no dispatch.
    (h2, s2.c * s2.h)
}

/// `CAMLayer`'s context: `x.mean(-1, keepdim=True) + seg_pooling(x)`,
/// `seg_pooling` a fixed `seg_len`-window `ceil_mode` average (the tail
/// window divides by its real element count). Computed on the host - see the
/// module doc.
fn cam_context(gpu: &Gpu, x: &DeviceBuffer, c: u32, l: u32, seg_len: u32) -> DeviceBuffer {
    let (c_us, l_us, seg) = (c as usize, l as usize, seg_len as usize);
    let xs = gpu.read(x, c_us * l_us);
    let n_segs = l_us.div_ceil(seg);
    let mut out = vec![0f32; c_us * l_us];
    for ch in 0..c_us {
        let row = &xs[ch * l_us..(ch + 1) * l_us];
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / l_us as f64;
        let out_row = &mut out[ch * l_us..(ch + 1) * l_us];
        for s in 0..n_segs {
            let start = s * seg;
            let end = (start + seg).min(l_us);
            let seg_mean = row[start..end].iter().map(|&v| v as f64).sum::<f64>() / (end - start) as f64;
            for v in out_row.iter_mut().take(end).skip(start) {
                *v = (mean + seg_mean) as f32;
            }
        }
    }
    let buf = gpu.storage((c_us * l_us) as u64);
    gpu.write_f32(&buf, &out);
    buf
}

/// One `CAMDenseTDNNLayer`: `BN+ReLU -> linear1 -> ReLU -> CAMLayer`. Returns
/// the `[cam_out, l]` layer output the caller concatenates onto the running
/// `x`.
fn cam_dense_layer(
    gpu: &Gpu,
    w: &HashMap<String, DeviceBuffer>,
    prefix: &str,
    x: &DeviceBuffer,
    cin: u32,
    l: u32,
    cfg: &CampplusConfig,
    dilation: u32,
) -> DeviceBuffer {
    let h0 = bn_eval(gpu, x, dget(w, &format!("{prefix}.nonlinear1.mv")), dget(w, &format!("{prefix}.nonlinear1.gb")), cin, l, 1);
    let (h1, _) = conv1d(
        gpu,
        &h0,
        dget(w, &format!("{prefix}.linear1.weight")),
        Some(dget(w, &format!("{prefix}.linear1.bias"))),
        cin,
        l,
        cfg.tdnn_out,
        1,
        1,
        0,
        1,
    );
    relu_inplace(gpu, &h1, cfg.tdnn_out * l);

    let (y_local, _) = conv1d(
        gpu,
        &h1,
        dget(w, &format!("{prefix}.cam.linear_local.weight")),
        None,
        cfg.tdnn_out,
        l,
        cfg.cam_out,
        3,
        1,
        dilation,
        dilation,
    );
    let ctx = cam_context(gpu, &h1, cfg.tdnn_out, l, cfg.seg_len);
    let (c1, _) = conv1d(
        gpu,
        &ctx,
        dget(w, &format!("{prefix}.cam.linear1.weight")),
        Some(dget(w, &format!("{prefix}.cam.linear1.bias"))),
        cfg.tdnn_out,
        l,
        cfg.cam_mid,
        1,
        1,
        0,
        1,
    );
    relu_inplace(gpu, &c1, cfg.cam_mid * l);
    let (c2, _) = conv1d(
        gpu,
        &c1,
        dget(w, &format!("{prefix}.cam.linear2.weight")),
        Some(dget(w, &format!("{prefix}.cam.linear2.bias"))),
        cfg.cam_mid,
        l,
        cfg.cam_out,
        1,
        1,
        0,
        1,
    );
    let sig = sigmoid(gpu, &c2, cfg.cam_out * l);
    mul(gpu, &y_local, &sig, cfg.cam_out * l)
}

/// One `CAMDenseTDNNBlock`: `cfg.block_layers[b]` layers, each concatenated
/// onto the running `x` (DenseNet-style).
fn dtdnn_block(gpu: &Gpu, w: &HashMap<String, DeviceBuffer>, b: usize, x: DeviceBuffer, cin0: u32, l: u32, cfg: &CampplusConfig) -> (DeviceBuffer, u32) {
    let dilation = cfg.block_dilation[b];
    let mut x = x;
    let mut cin = cin0;
    for i in 0..cfg.block_layers[b] as usize {
        let prefix = format!("xvector.block{}.tdnnd{}", b + 1, i + 1);
        let layer_out = cam_dense_layer(gpu, w, &prefix, &x, cin, l, cfg, dilation);
        x = concat2(gpu, &x, &layer_out, cin, cfg.cam_out, l);
        cin += cfg.cam_out;
    }
    (x, cin)
}

/// One `TransitLayer`: `BN+ReLU -> Conv1d(bias=False)`. The LAST transit
/// (`last=true`) absorbed `out_nonlinear`'s trailing `BN+ReLU` (see
/// `crate::import`), so it carries a bias and needs its own ReLU afterward.
fn transit(gpu: &Gpu, w: &HashMap<String, DeviceBuffer>, idx: usize, x: &DeviceBuffer, cin: u32, l: u32, cfg: &CampplusConfig, last: bool) -> (DeviceBuffer, u32) {
    let prefix = format!("xvector.transit{}", idx + 1);
    let h = bn_eval(gpu, x, dget(w, &format!("{prefix}.nonlinear.mv")), dget(w, &format!("{prefix}.nonlinear.gb")), cin, l, 1);
    let cout = cfg.transit_out_c(idx);
    let bias = if last { Some(dget(w, &format!("{prefix}.linear.bias"))) } else { None };
    let (y, _) = conv1d(gpu, &h, dget(w, &format!("{prefix}.linear.weight")), bias, cin, l, cout, 1, 1, 0, 1);
    if last {
        relu_inplace(gpu, &y, cout * l);
    }
    (y, cout)
}

/// The inference-only CAM++ speaker encoder.
pub struct Campplus {
    gpu: Gpu,
    cfg: CampplusConfig,
    w: HashMap<String, DeviceBuffer>,
    dense_w: Vec<f32>,
    dense_mean: Vec<f32>,
    dense_var: Vec<f32>,
}

impl Campplus {
    /// Build on a shared device handle from an imported tensor map
    /// (`crate::import::import_dir`/`import_campplus`).
    pub fn new(gpu: Gpu, cfg: CampplusConfig, weights: &Tensors) -> Campplus {
        let manifest = cfg.tensor_manifest();
        assert_eq!(
            weights.len(),
            manifest.len(),
            "campplus: checkpoint has {} tensors, the config manifest expects {}",
            weights.len(),
            manifest.len()
        );
        for (name, shape) in &manifest {
            let (got_shape, data) = weights.get(name).unwrap_or_else(|| panic!("campplus: checkpoint is missing {name}"));
            assert_eq!(got_shape, shape, "campplus: {name} shape {got_shape:?}, expected {shape:?}");
            let n: usize = shape.iter().product();
            assert_eq!(data.len(), n, "campplus: {name} has {} values, expected {n}", data.len());
        }

        let mut w: HashMap<String, DeviceBuffer> = HashMap::with_capacity(weights.len() + 2 * bn_schedule(&cfg).len());
        for (name, (_, data)) in weights {
            w.insert(name.clone(), gpu.storage_init(name, data));
        }
        // Pre-pack every standalone BatchNorm's eval-mode stats into the
        // interleaved `mv`/`gb` layout `bn_eval` wants, once - not per forward.
        for plan in bn_schedule(&cfg) {
            let g = hget(weights, &format!("{}.weight", plan.prefix));
            let b = hget(weights, &format!("{}.bias", plan.prefix));
            let m = hget(weights, &format!("{}.running_mean", plan.prefix));
            let v = hget(weights, &format!("{}.running_var", plan.prefix));
            let mv = pack2(m, v);
            let gb = pack2(g, b);
            w.insert(format!("{}.mv", plan.prefix), gpu.storage_init(&format!("{}.mv", plan.prefix), &mv));
            w.insert(format!("{}.gb", plan.prefix), gpu.storage_init(&format!("{}.gb", plan.prefix), &gb));
        }

        let dense_w = hget(weights, "xvector.dense.linear.weight").to_vec();
        let dense_mean = hget(weights, "xvector.dense.nonlinear.running_mean").to_vec();
        let dense_var = hget(weights, "xvector.dense.nonlinear.running_var").to_vec();
        Campplus { gpu, cfg, w, dense_w, dense_mean, dense_var }
    }

    pub fn config(&self) -> &CampplusConfig {
        &self.cfg
    }
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// `fbank` is `[t, feat_dim]` row-major (kaldi-style, time-major - the
    /// golden's own `[346, 80]` layout). Returns the `embedding_size`-d
    /// x-vector.
    pub fn forward(&self, fbank: &[f32], t: u32) -> Vec<f32> {
        let (feat, cfg) = (self.cfg.feat_dim, &self.cfg);
        assert_eq!(fbank.len(), (t * feat) as usize, "campplus: fbank must be [{t}, {feat}]");

        // permute (t, feat) -> (feat, t): the FCM stem's own H axis is
        // frequency, so this is the one host transpose the whole forward
        // needs (everything after is either GPU-dispatched or already in the
        // right layout).
        let (tu, fu) = (t as usize, feat as usize);
        let mut xt = vec![0f32; tu * fu];
        for ti in 0..tu {
            for f in 0..fu {
                xt[f * tu + ti] = fbank[ti * fu + f];
            }
        }
        let x0 = self.gpu.storage(xt.len() as u64);
        self.gpu.write_f32(&x0, &xt);

        let (fcm_out, fcm_c) = fcm_forward(&self.gpu, &self.w, &x0, t, cfg);
        assert_eq!(fcm_c, cfg.fcm_out_c());

        let (tdnn_h, l0) = conv1d(
            &self.gpu,
            &fcm_out,
            dget(&self.w, "xvector.tdnn.linear.weight"),
            Some(dget(&self.w, "xvector.tdnn.linear.bias")),
            fcm_c,
            t,
            cfg.tdnn_out,
            5,
            2,
            2,
            1,
        );
        relu_inplace(&self.gpu, &tdnn_h, cfg.tdnn_out * l0);

        let mut x = tdnn_h;
        let mut cin = cfg.tdnn_out;
        let l = l0;
        for b in 0..3usize {
            let (xb, cb) = dtdnn_block(&self.gpu, &self.w, b, x, cin, l, cfg);
            let (xt2, ct2) = transit(&self.gpu, &self.w, b, &xb, cb, l, cfg, b == 2);
            x = xt2;
            cin = ct2;
        }
        assert_eq!(cin, cfg.transit_out_c(2));

        self.stats_dense_bn(&x, cin, l)
    }

    /// `StatsPool` (`cat([mean, std(unbiased)])`) then `dense` (a bias-free
    /// `Conv1d` over a length-1 sequence == a matmul) then the final
    /// `affine=False` BatchNorm1d - all on the host over a `[stats_out_c] ->
    /// [embedding_size]` vector. See the module doc.
    fn stats_dense_bn(&self, x: &DeviceBuffer, c: u32, l: u32) -> Vec<f32> {
        let (c_us, l_us) = (c as usize, l as usize);
        assert!(l_us > 1, "campplus: unbiased std needs at least 2 frames, got {l_us}");
        let xs = self.gpu.read(x, c_us * l_us);
        let mut stats = vec![0f64; 2 * c_us];
        for ch in 0..c_us {
            let row = &xs[ch * l_us..(ch + 1) * l_us];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / l_us as f64;
            let var = row.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / (l_us as f64 - 1.0);
            stats[ch] = mean;
            stats[c_us + ch] = var.sqrt();
        }

        let e = self.cfg.embedding_size as usize;
        let din = 2 * c_us;
        assert_eq!(self.dense_w.len(), e * din);
        let mut out = vec![0f64; e];
        for (o, out_o) in out.iter_mut().enumerate() {
            let row = &self.dense_w[o * din..(o + 1) * din];
            *out_o = row.iter().zip(&stats).map(|(&wv, &sv)| wv as f64 * sv).sum();
        }
        let eps = self.cfg.bn_eps as f64;
        out.iter()
            .enumerate()
            .map(|(o, &v)| ((v - self.dense_mean[o] as f64) / (self.dense_var[o] as f64 + eps).sqrt()) as f32)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel the model dispatches must actually be registered in
    /// [`PIPELINES`], or the first forward panics at a dispatch instead of at
    /// build time.
    #[test]
    fn every_dispatched_kernel_is_registered() {
        for name in ["conv3d", "conv1d", "add_chan_inplace", "bn_eval", "relu_inplace", "sigmoid", "mul", "concat2", "add2"] {
            assert!(PIPELINES.iter().any(|(n, _)| *n == name), "{name} missing from campplus::PIPELINES");
            let _ = kernel(name);
        }
    }

    /// A SHRUNKEN encoder of the same shape, built from manifest-derived
    /// weights and run end to end. Every architectural feature is present: an
    /// asymmetric-stride `FCM` residual stem, all three D-TDNN blocks (12/24/16
    /// dilation 1/2/2), `CAMLayer`'s seg-pooled context at a short (non-trivial
    /// `ceil_mode`) sequence length, and the final unbiased-std stats pool. A
    /// kernel missing from [`PIPELINES`] panics here by NAME.
    #[test]
    fn the_whole_graph_runs_with_every_kernel_it_dispatches_registered() {
        let cfg = CampplusConfig {
            feat_dim: 16,
            embedding_size: 6,
            fcm_channels: 4,
            tdnn_out: 8,
            growth: 3,
            cam_mid: 5,
            cam_out: 3,
            block_layers: [2, 2, 2],
            block_dilation: [1, 2, 2],
            seg_len: 5,
            bn_eps: 1e-5,
        };
        let weights: Tensors = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, shape)| {
                let k: usize = shape.iter().product();
                // running_var at 1 and gamma/weight at 1: a zero variance
                // divides by sqrt(eps) and a zero gamma erases the signal,
                // either of which would make a wrongly-wired graph look the
                // same as a right one.
                let v = if n.ends_with(".running_var") || n.ends_with(".nonlinear1.weight") || n.ends_with(".nonlinear.weight") {
                    1.0
                } else if n.ends_with(".running_mean") {
                    0.0
                } else {
                    0.02
                };
                (n, (shape, vec![v; k]))
            })
            .collect();

        let gpu = gpu_core::testgpu::dev(PIPELINES);
        let m = Campplus::new(gpu, cfg.clone(), &weights);
        let t = 23u32; // not a multiple of seg_len after the tdnn stride-2 downsample
        let fbank = vec![0.1f32; (t * cfg.feat_dim) as usize];
        let emb = m.forward(&fbank, t);

        assert_eq!(emb.len(), cfg.embedding_size as usize);
        assert!(emb.iter().all(|v| v.is_finite()), "{emb:?}");
        assert!(emb.iter().any(|v| v.abs() > 0.0), "the embedding is all zero");
    }
}
