// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DIAMOND UNet ONNX export tests.
//!
//! Structural (always runs, no backend/OpenVINO): build a tiny DiamondConfig
//! with deterministic random weights, export the fp32 ONNX inner model, and
//! assert the graph decodes with the right inputs/outputs and op counts
//! computed from the architecture — this validates the wiring/shapes without
//! hardware.
//!
//! Numerical parity vs brain's own `DiamondUNet` (cpu device) forward runs the
//! exported graph through OpenVINO and is gated like the other parity tests
//! (BRAIN_OV_PROBE needs OpenVINO installed; MOE_SKIP_GPU_TESTS skips the
//! backend-dependent reference).

use wm_diamond::{DiamondConfig, Tensors};

/// The fixture architecture: 2 levels, attention on the deep level (so the
/// export exercises downsample/upsample, skip concats, AdaGN, and attention).
fn tiny_cfg() -> DiamondConfig {
    DiamondConfig {
        img_channels: 3,
        num_steps_conditioning: 2,
        cond_channels: 16,
        depths: vec![1, 1],
        channels: vec![8, 8],
        attn_depths: vec![false, true],
        num_actions: 4,
        h: 8,
        w: 8,
        sigma_data: 0.5,
        sigma_offset_noise: 0.3,
    }
}

/// Deterministic small random weights covering the full param_list.
fn rand_tensors(cfg: &DiamondConfig, seed: u64) -> Tensors {
    let mut s = seed;
    let mut next = move || {
        // SplitMix64 -> roughly U(-0.2, 0.2).
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
    };
    cfg.param_list()
        .into_iter()
        .map(|(n, shape)| {
            let numel: usize = shape.iter().product();
            (n, (shape, (0..numel).map(|_| next()).collect::<Vec<f32>>()))
        })
        .collect()
}

/// Expected op counts, computed from the architecture (mirrors the builder's
/// accounting, independent of its code path).
struct Expected {
    total_nodes: usize,
    conv: usize,
    gemm: usize,
    split: usize,
    softmax: usize,
    matmul: usize,
    resize: usize,
    concat: usize,
    weight_inits: usize,
}

fn expected(cfg: &DiamondConfig) -> Expected {
    let n_lv = cfg.channels.len();
    let down_rb: usize = cfg.depths.iter().map(|&d| d as usize).sum();
    let up_rb: usize = cfg.depths.iter().map(|&d| d as usize + 1).sum();
    let rb = down_rb + 2 + up_rb; // + mid
    let list = cfg.param_list();
    let count_suffix =
        |suf: &str| list.iter().filter(|(n, _)| n.ends_with(suf)).count();
    let attn = count_suffix(".attn.qkv_proj.weight"); // attention sites
    let projs = count_suffix(".proj.weight"); // channel-changing resblocks
    let adagn = 2 * rb;
    // Node accounting per building block:
    //   gn_core: Reshape,ReduceMean,Sub,Mul,ReduceMean,Add,Sqrt,Div,Reshape = 9
    //   adagn:   Gemm,Split,2xReshape,Add(1+scale),gn_core,Mul,Add       = 16
    //   affine:  gn_core,Mul,Add                                          = 11
    //   silu:    Sigmoid,Mul                                              = 2
    //   attn:    affine + conv,Reshape,Transpose,Split,Transpose,MatMul,
    //            Mul,Softmax,MatMul,Transpose,Reshape,conv,Add            = 24
    //   resblock (no attn/proj): 2*adagn + 2*silu + 2*conv + Add          = 39
    let total_nodes = rb * 39 + projs + attn * 24
        + 2                    // conv_in: Concat + Conv
        + (n_lv - 1)           // downsample convs
        + (n_lv - 1) * 2       // upsamples: Resize + Conv
        + up_rb                // up-path skip Concats
        + 11 + 2 + 1 + 1; // head: affine_gn + SiLU + conv_out + Identity
    Expected {
        total_nodes,
        conv: 1 + 2 * rb + projs + 2 * attn + 2 * (n_lv - 1) + 1,
        gemm: adagn,
        split: adagn + attn,
        softmax: attn,
        matmul: 2 * attn,
        resize: n_lv - 1,
        concat: 1 + up_rb,
        // Every UNet tensor except the 6 host-side conditioning tensors
        // becomes an initializer (affine GN vectors are re-registered under a
        // .c111 alias, still one initializer each).
        weight_inits: list.len() - 6,
    }
}

fn dims_of(v: &onnx::onnx::ValueInfoProto) -> Vec<i64> {
    v.r#type
        .as_ref()
        .and_then(|t| t.tensor_type.as_ref())
        .and_then(|t| t.shape.as_ref())
        .map(|s| s.dim.iter().map(|d| d.dim_value).collect())
        .unwrap_or_default()
}

#[test]
fn wm_onnx_graph_is_well_formed() {
    let cfg = tiny_cfg();
    let tensors = rand_tensors(&cfg, 7);
    let bytes = wm_diamond::npu::build_onnx_bytes(&cfg, &tensors);
    assert!(bytes.len() > 1000, "onnx export suspiciously small: {} bytes", bytes.len());

    let model = onnx::decode_model(&bytes).expect("export must decode as a valid ONNX ModelProto");
    let g = model.graph.expect("model has a graph");

    // Inputs / output: names AND static shapes.
    let input = |name: &str| {
        g.input
            .iter()
            .find(|v| v.name == name)
            .unwrap_or_else(|| panic!("missing input {name}"))
    };
    assert_eq!(dims_of(input("noisy_scaled")), vec![1, 3, 8, 8]);
    assert_eq!(dims_of(input("obs_rescaled")), vec![1, 6, 8, 8]);
    assert_eq!(dims_of(input("cond")), vec![1, 16]);
    assert_eq!(g.input.len(), 3, "exactly the three model inputs");
    assert_eq!(g.output.len(), 1);
    assert_eq!(g.output[0].name, "model_out");
    assert_eq!(dims_of(&g.output[0]), vec![1, 3, 8, 8]);

    // Op histogram vs architecture-derived expectations.
    let mut ops: std::collections::HashMap<&str, usize> = Default::default();
    for n in &g.node {
        *ops.entry(n.op_type.as_str()).or_insert(0) += 1;
    }
    let e = expected(&cfg);
    let count = |op: &str| ops.get(op).copied().unwrap_or(0);
    assert_eq!(count("Conv"), e.conv, "Conv count");
    assert_eq!(count("Gemm"), e.gemm, "Gemm (AdaGN sites)");
    assert_eq!(count("Split"), e.split, "Split (AdaGN chunks + qkv)");
    assert_eq!(count("Softmax"), e.softmax, "Softmax (attention sites)");
    assert_eq!(count("MatMul"), e.matmul, "MatMul (qk^T + probs*v)");
    assert_eq!(count("Resize"), e.resize, "Resize (upsamples)");
    assert_eq!(count("Concat"), e.concat, "Concat (conv_in + skips)");
    assert_eq!(g.node.len(), e.total_nodes, "total node count");

    // Every checkpoint tensor used by the graph is an initializer (plus the
    // shape/scalar helpers on top).
    assert!(
        g.initializer.len() >= e.weight_inits,
        "{} initializers < {} UNet weight tensors",
        g.initializer.len(),
        e.weight_inits
    );
}

/// Numerical parity: run the exported graph through OpenVINO (CPU device,
/// fallback allowed) and compare the inner-model output F to brain's own
/// `DiamondUNet` (cpu device) forward on identical inputs. Gated like
/// glm_onnx_matches_brain_forward.
#[test]
fn wm_onnx_matches_brain_forward() {
    if std::env::var("BRAIN_OV_PROBE").is_err() || std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    use npu::openvino::{NpuConfig, NpuDevice};
    use npu::wm_topology::WmSession;

    let cfg = tiny_cfg();
    let tensors = rand_tensors(&cfg, 7);
    let n = (cfg.img_channels * cfg.h * cfg.w) as usize;
    let nobs = n * cfg.num_steps_conditioning as usize;

    // Fixed, non-trivial inputs.
    let noisy: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.37).sin()) * 0.8).collect();
    let obs: Vec<f32> = (0..nobs).map(|i| ((i as f32 * 0.11).cos()) * 0.9).collect();
    let actions: Vec<u32> = vec![1, 3];
    let c_noise = 0.1f32;

    // Reference: brain's own engine on the cpu device.
    let unet = wm_diamond::DiamondUNet::new(cfg.clone(), &tensors, Some("cpu"));
    unet.set_context(&obs);
    let reference = unet.forward(&noisy, c_noise, &actions);

    // Exported graph through OpenVINO; cond computed by the same host path.
    let bytes = wm_diamond::npu::build_onnx_bytes(&cfg, &tensors);
    let mut sess = WmSession::load_bytes(
        &bytes,
        &NpuConfig { device: NpuDevice::Cpu, allow_fallback: true, ..Default::default() },
        cfg.img_channels as usize,
        (cfg.num_steps_conditioning * cfg.img_channels) as usize,
        cfg.h as usize,
        cfg.w as usize,
        cfg.cond_channels as usize,
    )
    .expect("compile DIAMOND UNet on OpenVINO CPU");
    let cond = cond_of(&cfg, &tensors, c_noise, &actions);
    let got = sess.run(&noisy, &obs, &cond).expect("run DIAMOND UNet");

    assert_eq!(got.len(), reference.len(), "output length");
    let mut max_abs = 0f32;
    for (a, b) in reference.iter().zip(&got) {
        max_abs = max_abs.max((a - b).abs());
    }
    eprintln!("DIAMOND ONNX vs brain: max_abs={max_abs:.6} (device {})", sess.device());
    assert!(max_abs < 1e-3, "inner-model output mismatch too large: {max_abs}");
}

/// Compile + run the exported graph on the actual Intel **NPU** device (with
/// fallback) via the full host glue (`DiamondNpu::forward`) and check parity
/// vs brain. Reports the device it landed on. Gated like the CPU parity test.
#[test]
fn wm_onnx_runs_on_npu() {
    if std::env::var("BRAIN_OV_PROBE").is_err() || std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    use npu::openvino::{NpuConfig, NpuDevice};

    let cfg = tiny_cfg();
    let tensors = rand_tensors(&cfg, 7);
    let n = (cfg.img_channels * cfg.h * cfg.w) as usize;
    let nobs = n * cfg.num_steps_conditioning as usize;
    let noisy: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.37).sin()) * 0.8).collect();
    let obs: Vec<f32> = (0..nobs).map(|i| ((i as f32 * 0.11).cos()) * 0.9).collect();
    let actions: Vec<u32> = vec![1, 3];
    let c_noise = 0.1f32;

    let unet = wm_diamond::DiamondUNet::new(cfg.clone(), &tensors, Some("cpu"));
    unet.set_context(&obs);
    let reference = unet.forward(&noisy, c_noise, &actions);

    let dir = std::env::temp_dir().join(format!("brain-wm-onnx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let opath = dir.join("diamond-tiny.onnx");
    std::fs::write(&opath, wm_diamond::npu::build_onnx_bytes(&cfg, &tensors)).unwrap();

    let mut dn = wm_diamond::npu::DiamondNpu::new(
        cfg.clone(),
        &tensors,
        opath.to_str().unwrap(),
        &NpuConfig { device: NpuDevice::Npu, allow_fallback: true, ..Default::default() },
    )
    .expect("compile DIAMOND UNet (NPU or fallback)");
    let got = dn.forward(&noisy, c_noise, &actions, &obs).expect("run DIAMOND UNet");
    std::fs::remove_dir_all(&dir).ok();

    let mut max_abs = 0f32;
    for (a, b) in reference.iter().zip(&got) {
        max_abs = max_abs.max((a - b).abs());
    }
    eprintln!("DIAMOND ONNX on {}: max_abs={max_abs:.6}", dn.device());
    // The NPU computes in fp16 internally — allow a wider (but still tight
    // for a [-1,1]-ranged image model) tolerance than the CPU fp32 run.
    let tol = if dn.device().starts_with("NPU") { 1e-2 } else { 1e-3 };
    assert!(max_abs < tol, "inner-model output mismatch too large on {}: {max_abs}", dn.device());
}

/// Host-side cond vector, mirroring `wm_diamond::cond::CondNet` construction.
fn cond_of(cfg: &DiamondConfig, tensors: &Tensors, c_noise: f32, actions: &[u32]) -> Vec<f32> {
    let get = |n: &str| tensors[n].1.clone();
    let net = wm_diamond::cond::CondNet {
        cond_channels: cfg.cond_channels as usize,
        num_steps_conditioning: cfg.num_steps_conditioning as usize,
        fourier_w: get("noise_emb.weight"),
        act_emb: get("act_emb.0.weight"),
        num_actions: cfg.num_actions as usize,
        mlp0_w: get("cond_proj.0.weight"),
        mlp0_b: get("cond_proj.0.bias"),
        mlp2_w: get("cond_proj.2.weight"),
        mlp2_b: get("cond_proj.2.bias"),
    };
    net.cond(c_noise, actions)
}
