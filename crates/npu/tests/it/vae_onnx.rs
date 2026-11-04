// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `AutoencoderKL` decoder → ONNX, gated for what is checkable without an NPU.
//!
//! The GroupNorm underneath is gated NUMERICALLY (`group_norm_onnx.rs`); this
//! file gates the assembly — op set, the schedule the config implies, and the
//! two things that are silently wrong rather than loudly broken:
//!
//! * **`post_quant_conv` present when the config says so.** Dropping it leaves a
//!   decode in a plausible `[-1,1]` range and UNCORRELATED with the truth.
//!   On the NPU that would be found by someone looking
//!   at a wrong picture on other hardware.
//! * **The channel schedule is REVERSED.** The decoder walks
//!   `block_out_channels` high→low; walking it forward builds a net with the
//!   right op counts and the wrong widths.

use std::collections::HashMap;

use npu::topology::WeightSource;
use npu::vae_topology::{build_vae_decoder_graph, VaeTopo};

/// Shape-correct zeros, so the graph's structure is testable with no checkpoint.
/// Panics on an unknown name, which makes a typo'd tensor a test failure rather
/// than a zero-filled buffer.
struct Zeros(HashMap<String, Vec<i64>>);

impl WeightSource for Zeros {
    fn get(&self, name: &str) -> Vec<f32> {
        let d = self
            .0
            .get(name)
            .unwrap_or_else(|| panic!("graph asked for an unknown tensor `{name}`"));
        vec![0.0; d.iter().product::<i64>() as usize]
    }
}

impl Zeros {
    fn for_topo(t: &VaeTopo) -> Zeros {
        let mut m: HashMap<String, Vec<i64>> = HashMap::new();
        let conv = |p: &str, cout: u32, cin: u32, k: i64, m: &mut HashMap<String, Vec<i64>>| {
            m.insert(format!("{p}.weight"), vec![cout as i64, cin as i64, k, k]);
            m.insert(format!("{p}.bias"), vec![cout as i64]);
        };
        let norm = |p: &str, c: u32, m: &mut HashMap<String, Vec<i64>>| {
            m.insert(format!("{p}.weight"), vec![c as i64]);
            m.insert(format!("{p}.bias"), vec![c as i64]);
        };
        let names = t.names;
        let resnet = |p: &str, cin: u32, cout: u32, m: &mut HashMap<String, Vec<i64>>| {
            norm(&format!("{p}.norm1"), cin, m);
            conv(&format!("{p}.conv1"), cout, cin, 3, m);
            norm(&format!("{p}.norm2"), cout, m);
            conv(&format!("{p}.conv2"), cout, cout, 3, m);
            if cin != cout {
                conv(&format!("{p}.{}", names.shortcut), cout, cin, 1, m);
            }
        };

        let rev: Vec<u32> = t.block_out_channels.iter().rev().copied().collect();
        let mid = rev[0];
        if t.use_post_quant_conv {
            conv("post_quant_conv", t.latent_channels, t.latent_channels, 1, &mut m);
        }
        conv("decoder.conv_in", mid, t.latent_channels, 3, &mut m);
        resnet("decoder.mid_block.resnets.0", mid, mid, &mut m);
        if t.mid_block_add_attention {
            let p = "decoder.mid_block.attentions.0";
            norm(&format!("{p}.{}", names.attn_norm), mid, &mut m);
            for n in [names.attn_q, names.attn_k, names.attn_v, names.attn_proj] {
                m.insert(format!("{p}.{n}.weight"), vec![mid as i64, mid as i64]);
                m.insert(format!("{p}.{n}.bias"), vec![mid as i64]);
            }
        }
        resnet("decoder.mid_block.resnets.1", mid, mid, &mut m);

        let mut cin = mid;
        for (i, &out_c) in rev.iter().enumerate() {
            for r in 0..=t.layers_per_block {
                resnet(&format!("decoder.up_blocks.{i}.resnets.{r}"), cin, out_c, &mut m);
                cin = out_c;
            }
            if i + 1 < rev.len() {
                conv(&format!("decoder.up_blocks.{i}.upsamplers.0.conv"), out_c, out_c, 3, &mut m);
            }
        }
        norm("decoder.conv_norm_out", cin, &mut m);
        conv("decoder.conv_out", t.out_channels, cin, 3, &mut m);
        Zeros(m)
    }
}

/// SDXL's VAE at a tiny latent: 4 latent channels, four blocks, 2 layers each.
/// Widths all differ so a schedule walked the wrong way cannot pass.
fn sdxl_tiny() -> VaeTopo {
    VaeTopo {
        names: npu::vae_topology::TopoNames::diffusers(),
        latent_channels: 4,
        out_channels: 3,
        block_out_channels: vec![32, 64, 96, 128],
        layers_per_block: 2,
        norm_num_groups: 16,
        norm_eps: 1e-6,
        mid_block_add_attention: true,
        use_post_quant_conv: true,
        lh: 8,
        lw: 8,
    }
}

fn build(t: &VaeTopo) -> onnx::GraphBuilder {
    let mut g = onnx::GraphBuilder::new("vae_dec");
    build_vae_decoder_graph(t, &Zeros::for_topo(t), &mut g);
    g
}

fn op_counts(g: &onnx::GraphBuilder) -> HashMap<String, usize> {
    let mut c: HashMap<String, usize> = HashMap::new();
    for n in &g.graph().nodes {
        *c.entry(n.op_type.clone()).or_default() += 1;
    }
    c
}

#[test]
fn the_graph_uses_only_opset_13_ops() {
    let t = sdxl_tiny();
    let counts = op_counts(&build(&t));
    let allowed = [
        "Conv", "Resize", "Add", "Mul", "Sub", "Div", "Sqrt", "ReduceMean", "Reshape", "Transpose",
        "MatMul", "Softmax", "Sigmoid",
    ];
    for op in counts.keys() {
        assert!(allowed.contains(&op.as_str()), "unexpected op `{op}`");
    }
    assert!(!counts.contains_key("GroupNormalization"), "opset 18 op would not load at 13");
    eprintln!("vae decoder ops: {counts:?}");
}

/// The conv count follows from the schedule, so a block walked the wrong way or
/// a missing upsampler is arithmetic.
#[test]
fn the_conv_count_matches_the_schedule() {
    let t = sdxl_tiny();
    let counts = op_counts(&build(&t));
    let nb = t.block_out_channels.len();
    let lpb = t.layers_per_block as usize;

    // post_quant_conv + conv_in + conv_out.
    let mut want = 3;
    // mid: two resnets, each conv1+conv2 (widths equal -> no shortcut).
    want += 4;
    // up blocks: (lpb+1) resnets each, +1 shortcut wherever the width changes.
    // Widths change on the FIRST resnet of every block after the first.
    want += nb * (lpb + 1) * 2 + (nb - 1);
    // one upsampler conv per block except the last.
    want += nb - 1;
    assert_eq!(counts["Conv"], want, "Conv count");

    // One nearest-neighbour doubling per block except the last.
    assert_eq!(counts["Resize"], nb - 1);
    // One attention -> 4 projections + 2 matmuls, 1 softmax.
    assert_eq!(counts["Softmax"], 1);
}

/// THE SILENT ONE. `post_quant_conv` defaults to true in the reference; dropping
/// it is a decode that keeps its range and loses its meaning.
#[test]
fn post_quant_conv_is_emitted_when_the_config_asks_for_it() {
    let with = op_counts(&build(&sdxl_tiny()));
    let without = op_counts(&build(&VaeTopo { use_post_quant_conv: false, ..sdxl_tiny() }));
    assert_eq!(
        with["Conv"],
        without["Conv"] + 1,
        "the post_quant_conv 1x1 must be exactly one extra Conv"
    );
}

/// The decoder walks the channel schedule REVERSED. Building with a schedule
/// whose widths all differ, the first up-block must be the WIDEST.
#[test]
fn the_channel_schedule_is_walked_high_to_low() {
    let t = sdxl_tiny();
    let g = build(&t);
    let init: HashMap<&str, Vec<i64>> =
        g.graph().initializers.iter().map(|i| (i.name.as_str(), i.dims.clone())).collect();
    // conv_in emits the WIDEST channel count (the reversed schedule's head).
    let widest = *t.block_out_channels.iter().max().unwrap() as i64;
    assert_eq!(init["decoder.conv_in.weight"][0], widest, "conv_in must emit the widest width");
    // The last up-block's conv2 emits the NARROWEST.
    let narrowest = *t.block_out_channels.iter().min().unwrap() as i64;
    let last = t.block_out_channels.len() - 1;
    let k = format!("decoder.up_blocks.{last}.resnets.0.conv2.weight");
    assert_eq!(init[k.as_str()][0], narrowest, "the last up-block must emit the narrowest width");
}

#[test]
fn the_export_round_trips_with_the_declared_shapes() {
    let t = sdxl_tiny();
    let bytes = build(&t).finish();
    let m = onnx::decode_model(&bytes).expect("valid ONNX");
    let g = m.graph.expect("graph");
    assert!(g.input.iter().any(|v| v.name == "latent"));
    assert!(g.output.iter().any(|v| v.name == "image"));
    // 4 blocks -> an eightfold upscale.
    assert_eq!(t.upscale(), 8);
}

/// The VQGAN autoencoder is the SAME graph under different leaf names, so it
/// must build from this topology with `TopoNames::vqgan()` rather than a copied
/// module. Asserts the op counts are IDENTICAL — only the names differ.
#[test]
fn the_vqgan_naming_builds_the_same_graph() {
    let d = sdxl_tiny();
    let v = VaeTopo { names: npu::vae_topology::TopoNames::vqgan(), ..sdxl_tiny() };
    let (cd, cv) = (op_counts(&build(&d)), op_counts(&build(&v)));
    assert_eq!(cd, cv, "diffusers and VQGAN naming must yield the same graph shape");

    // ...and the names really did change, or the test above is vacuous.
    let g = build(&v);
    let init: Vec<&str> = g.graph().initializers.iter().map(|i| i.name.as_str()).collect();
    assert!(init.iter().any(|n| n.contains(".q.weight")), "VQGAN uses `q`, not `to_q`");
    assert!(!init.iter().any(|n| n.contains(".to_q.")), "no diffusers names should survive");
}
