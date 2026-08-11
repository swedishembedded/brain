// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL / Qwen3-Omni ViT vision tower + main `PatchMerger` as an
//! OpenVINO-compilable ONNX graph (**single-merger path only** — see the
//! DeepStack note below). Op-for-op with `qwenvl::encoder::VisionEncoder::encode`
//! + `PatchMerger::merge` (main merger, pre-shuffle LayerNorm), same split as
//! [`crate::qwen_asr_topology`]: patch packing (data-dependent, host-side) and
//! the learned pos-embed's bilinear resample stay off-graph; the patch-embed
//! matmul onward is in-graph.
//!
//! Standard pre-LN ViT block, no QK-norm / no LayerScale, fused QKV, 2-D vision
//! RoPE, **tanh-GELU** MLP (`gelu_pytorch_tanh` — distinct from the audio
//! tower's erf-GELU), full (unmasked) attention over the whole image (one
//! image per graph, no windowing). RoPE table is baked as a duplicated
//! `[n, head_dim]` initializer (see [`vision_rope_tables_host`]'s doc) so the
//! in-graph rotation is the same half-split `x·cos + rotate_half(x)·sin` op
//! sequence [`crate::codec_topology`]'s transformer already uses — table
//! values differ (h/w split vs. 1-D), not the op graph.
//!
//! **DeepStack is out of scope**: `omni::mm::encode_image`'s own host path
//! does not exercise the 3 DeepStack taps / per-tap mergers either (see its
//! doc comment on `deepstack_merger.*` weights) — this graph mirrors the code
//! actually served, not a gap unique to the NPU path. This is a named scope
//! limitation, not an oversight.

use onnx::{GraphBuilder, Node};

use crate::nemotron_topology::{add_t, layernorm_onnx, linear_nb, reshape, transpose};
use crate::qwen_asr_topology::{gelu_erf, linear_bias, slice_cols};
use crate::topology::WeightSource;

/// Config subset the ViT head needs (mirrors `qwenvl::config::VisionConfig`,
/// duplicated rather than depending on the `qwenvl` crate — same
/// self-contained-topology convention `codec_topology.rs` follows for its own
/// RoPE table math).
#[derive(Clone, Copy, Debug)]
pub struct VitTopo {
    pub depth: u32,
    pub hidden: u32,
    pub num_heads: u32,
    pub intermediate: u32,
    pub out_hidden: u32,
    pub merge: u32,
    pub eps: f32,
    pub rope_theta: f32,
}

impl VitTopo {
    pub fn head_dim(&self) -> u32 {
        self.hidden / self.num_heads
    }
}

/// Per-patch `(h, w)` grid positions in spatial-merge-block order — ports
/// `qwenvl::vision::vision_position_ids` (duplicated, see module doc).
fn vision_position_ids_host(hp: u32, wp: u32, merge: u32) -> Vec<(u32, u32)> {
    assert!(hp.is_multiple_of(merge) && wp.is_multiple_of(merge), "grid must be a multiple of merge size");
    let mut out = Vec::with_capacity((hp * wp) as usize);
    for bh in 0..hp / merge {
        for bw in 0..wp / merge {
            for ih in 0..merge {
                for iw in 0..merge {
                    out.push((bh * merge + ih, bw * merge + iw));
                }
            }
        }
    }
    out
}

/// Build the 2-D vision-RoPE `[n, head_dim]` cos/sin tables, duplicated across
/// the two rotate-half halves — ports `qwenvl::vision::vision_rope_tables`
/// (which returns `[n, head_dim/2]`), pre-duplicated here so the in-graph
/// rotation can use the plain 1-D `x·cos + rotate_half(x)·sin` op sequence:
/// the `rope2d` WGSL kernel indexes both `x1=buf[d]` and `x2=buf[d+half]` at
/// the SAME table row `d` (`crates/kernels/wgsl/rope2d.wgsl`), which is
/// exactly what a duplicated-halves full-width table reproduces without a
/// dedicated 2-D-RoPE ONNX op.
fn vision_rope_tables_host(positions: &[(u32, u32)], head_dim: u32, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = (head_dim / 2) as usize; // rotary dim
    let quarter = half / 2; // freqs per axis (h, w)
    let mut cos = vec![0f32; positions.len() * head_dim as usize];
    let mut sin = vec![0f32; positions.len() * head_dim as usize];
    for (t, &(h, w)) in positions.iter().enumerate() {
        let base = t * head_dim as usize;
        for j in 0..quarter {
            let inv = theta.powf(-2.0 * j as f32 / half as f32);
            let (ah, aw) = (h as f32 * inv, w as f32 * inv);
            let (ch, sh) = (ah.cos(), ah.sin());
            let (cw, sw) = (aw.cos(), aw.sin());
            // First half: [h-freqs, w-freqs] (length `half`); second half: the
            // SAME values again (rotate_half pairs (d, d+half) share angle d).
            cos[base + j] = ch;
            sin[base + j] = sh;
            cos[base + quarter + j] = cw;
            sin[base + quarter + j] = sw;
            cos[base + half + j] = ch;
            sin[base + half + j] = sh;
            cos[base + half + quarter + j] = cw;
            sin[base + half + quarter + j] = sw;
        }
    }
    (cos, sin)
}

/// Bilinear-resample the learned `side×side` pos-embed table onto a
/// `grid_h×grid_w` patch grid (merge-block order) — ports
/// `qwenvl::vision::pos_embed_bilinear` fused with the gather/weighted-sum it
/// feeds, since the export only needs the resampled `[n, hidden]` result
/// (baked as an initializer, not a graph op — same reasoning as the window
/// mask in [`crate::qwen_asr_topology::build_qwen_asr_head`]).
fn pos_embed_resampled_host(pos_table: &[f32], hidden: usize, grid_h: u32, grid_w: u32, merge: u32, side: u32) -> Vec<f32> {
    assert!(side >= 1 && grid_h.is_multiple_of(merge) && grid_w.is_multiple_of(merge));
    let lin = |i: u32, n: u32| -> f32 {
        if n <= 1 { 0.0 } else { i as f32 * (side as f32 - 1.0) / (n as f32 - 1.0) }
    };
    let mut out = Vec::with_capacity((grid_h * grid_w) as usize * hidden);
    for bh in 0..grid_h / merge {
        for bw in 0..grid_w / merge {
            for ih in 0..merge {
                for iw in 0..merge {
                    let (hi, wi) = (bh * merge + ih, bw * merge + iw);
                    let (hg, wg) = (lin(hi, grid_h), lin(wi, grid_w));
                    let (hf, wf) = (hg.floor() as u32, wg.floor() as u32);
                    let (hc, wc) = ((hf + 1).min(side - 1), (wf + 1).min(side - 1));
                    let (hfr, wfr) = (hg - hf as f32, wg - wf as f32);
                    let idx = [hf * side + wf, hf * side + wc, hc * side + wf, hc * side + wc];
                    let wts = [(1.0 - hfr) * (1.0 - wfr), (1.0 - hfr) * wfr, hfr * (1.0 - wfr), hfr * wfr];
                    for c in 0..hidden {
                        let mut acc = 0f32;
                        for k in 0..4 {
                            acc += pos_table[idx[k] as usize * hidden + c] * wts[k];
                        }
                        out.push(acc);
                    }
                }
            }
        }
    }
    out
}

/// tanh-GELU (`gelu_pytorch_tanh`): `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
fn gelu_tanh(g: &mut GraphBuilder, x: &str, tag: &str) -> String {
    let x2 = format!("{tag}.x2");
    g.add(Node::new("Mul", &[x, x], &[&x2]).name(&format!("{tag}.x2")));
    let x3 = format!("{tag}.x3");
    g.add(Node::new("Mul", &[&x2, x], &[&x3]).name(&format!("{tag}.x3")));
    let cn = format!("{tag}.c");
    g.init_f32(&cn, &[1], vec![0.044715]);
    let cx3 = format!("{tag}.cx3");
    g.add(Node::new("Mul", &[&x3, &cn], &[&cx3]).name(&format!("{tag}.cx3")));
    let inner = format!("{tag}.inner");
    g.add(Node::new("Add", &[x, &cx3], &[&inner]).name(&format!("{tag}.inner")));
    let kn = format!("{tag}.k");
    g.init_f32(&kn, &[1], vec![(2.0f32 / std::f32::consts::PI).sqrt()]);
    let scaled = format!("{tag}.scaled");
    g.add(Node::new("Mul", &[&inner, &kn], &[&scaled]).name(&format!("{tag}.scaled")));
    let th = format!("{tag}.tanh");
    g.add(Node::new("Tanh", &[&scaled], &[&th]).name(&format!("{tag}.tanh")));
    let one = format!("{tag}.one");
    g.init_f32(&one, &[1], vec![1.0]);
    let onep = format!("{tag}.onep");
    g.add(Node::new("Add", &[&th, &one], &[&onep]).name(&format!("{tag}.onep")));
    let half = format!("{tag}.half");
    g.init_f32(&half, &[1], vec![0.5]);
    let hx = format!("{tag}.hx");
    g.add(Node::new("Mul", &[x, &half], &[&hx]).name(&format!("{tag}.hx")));
    let o = format!("{tag}.gelu");
    g.add(Node::new("Mul", &[&hx, &onep], &[&o]).name(&format!("{tag}.gelu")));
    o
}

/// Apply the baked `[n, head_dim]` cos/sin tables to `x` `[n, heads, head_dim]`
/// (broadcast over `heads`): `x·cos + rotate_half(x)·sin`.
fn rope2d_apply(g: &mut GraphBuilder, x: &str, cos_name: &str, sin_name: &str, half: u32, tag: &str) -> String {
    let (s0, s1, ax) = (format!("{tag}.lo0"), format!("{tag}.hi0"), format!("{tag}.ax"));
    g.init_i64(&s0, &[1], vec![0]);
    g.init_i64(&s1, &[1], vec![half as i64]);
    g.init_i64(&ax, &[1], vec![2]);
    let x1 = format!("{tag}.x1");
    g.add(Node::new("Slice", &[x, &s0, &s1, &ax], &[&x1]).name(&format!("{tag}.slice1")));
    let hd2 = format!("{tag}.hi1");
    g.init_i64(&hd2, &[1], vec![2 * half as i64]);
    let x2 = format!("{tag}.x2");
    g.add(Node::new("Slice", &[x, &s1, &hd2, &ax], &[&x2]).name(&format!("{tag}.slice2")));
    let nx2 = format!("{tag}.neg");
    g.add(Node::new("Neg", &[&x2], &[&nx2]).name(&format!("{tag}.neg")));
    let rot = format!("{tag}.rot");
    g.add(Node::new("Concat", &[&nx2, &x1], &[&rot]).name(&format!("{tag}.concat")).attr_int("axis", 2));
    let a = format!("{tag}.a");
    g.add(Node::new("Mul", &[x, cos_name], &[&a]).name(&format!("{tag}.a")));
    let b = format!("{tag}.b");
    g.add(Node::new("Mul", &[&rot, sin_name], &[&b]).name(&format!("{tag}.b")));
    let o = format!("{tag}.roped");
    g.add(Node::new("Add", &[&a, &b], &[&o]).name(&format!("{tag}.add")));
    o
}

/// One pre-LN ViT block: fused-QKV attention (2-D RoPE on q/k, full
/// attention, no QK-norm/LayerScale) + tanh-GELU MLP.
fn vit_block_onnx(g: &mut GraphBuilder, topo: &VitTopo, w: &dyn WeightSource, cos: &str, sin: &str, x: &str, blk: u32, n: u32) -> String {
    let (c, heads, hd, ffn) = (topo.hidden, topo.num_heads, topo.head_dim(), topo.intermediate);
    let half = hd / 2;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let tag = format!("vblk{blk}");
    let p = format!("blocks.{blk}");

    let ln1 = layernorm_onnx(g, w, x, &format!("{p}.norm1.weight"), &format!("{p}.norm1.bias"), c, topo.eps, &format!("{tag}.n1"));
    let qkv = linear_bias(g, w, &ln1, &format!("{p}.qkv.weight"), &format!("{p}.qkv.bias"), 3 * c, c, &format!("{tag}.qkv"));
    let q = slice_cols(g, &qkv, 0, c, &format!("{tag}.q"));
    let k = slice_cols(g, &qkv, c, 2 * c, &format!("{tag}.k"));
    let v = slice_cols(g, &qkv, 2 * c, 3 * c, &format!("{tag}.v"));
    let qh = reshape(g, &q, &[n as i64, heads as i64, hd as i64], &format!("{tag}.qh.r"));
    let kh = reshape(g, &k, &[n as i64, heads as i64, hd as i64], &format!("{tag}.kh.r"));
    let vh = reshape(g, &v, &[n as i64, heads as i64, hd as i64], &format!("{tag}.vh.r"));
    let qh = rope2d_apply(g, &qh, cos, sin, half, &format!("{tag}.qrope"));
    let kh = rope2d_apply(g, &kh, cos, sin, half, &format!("{tag}.krope"));
    let qh = transpose(g, &qh, &[1, 0, 2], &format!("{tag}.qh.t")); // [heads,n,hd]
    let kh = transpose(g, &kh, &[1, 0, 2], &format!("{tag}.kh.t"));
    let vh = transpose(g, &vh, &[1, 0, 2], &format!("{tag}.vh.t"));
    let kt = transpose(g, &kh, &[0, 2, 1], &format!("{tag}.kt"));
    let sc = format!("{tag}.sc");
    g.add(Node::new("MatMul", &[&qh, &kt], &[&sc]).name(&format!("{tag}.sc")));
    let scn = format!("{tag}.sck");
    g.init_f32(&scn, &[1], vec![scale]);
    let scaled = format!("{tag}.scaled");
    g.add(Node::new("Mul", &[&sc, &scn], &[&scaled]).name(&format!("{tag}.scaled")));
    let probs = format!("{tag}.probs");
    g.add(Node::new("Softmax", &[&scaled], &[&probs]).name(&format!("{tag}.softmax")).attr_int("axis", -1));
    let ctxh = format!("{tag}.ctxh");
    g.add(Node::new("MatMul", &[&probs, &vh], &[&ctxh]).name(&format!("{tag}.ctx")));
    let ctx_tp = transpose(g, &ctxh, &[1, 0, 2], &format!("{tag}.ctxtp"));
    let ctx = reshape(g, &ctx_tp, &[n as i64, c as i64], &format!("{tag}.ctxflat"));
    let attn = linear_bias(g, w, &ctx, &format!("{p}.proj.weight"), &format!("{p}.proj.bias"), c, c, &format!("{tag}.proj"));
    let x1 = add_t(g, x, &attn, &format!("{tag}.res1"));

    let ln2 = layernorm_onnx(g, w, &x1, &format!("{p}.norm2.weight"), &format!("{p}.norm2.bias"), c, topo.eps, &format!("{tag}.n2"));
    let h1 = linear_bias(g, w, &ln2, &format!("{p}.fc1.weight"), &format!("{p}.fc1.bias"), ffn, c, &format!("{tag}.fc1"));
    let act = gelu_tanh(g, &h1, &format!("{tag}.act"));
    let h2 = linear_bias(g, w, &act, &format!("{p}.fc2.weight"), &format!("{p}.fc2.bias"), c, ffn, &format!("{tag}.fc2"));
    add_t(g, &x1, &h2, &format!("{tag}.res2"))
}

/// The main `PatchMerger` (pre-shuffle LayerNorm, in-graph): `[n, in_dim]` ViT
/// features -> `[n/merge², out_dim]` visual tokens. `LayerNorm(in_dim)` per
/// patch, reshape (free, merge-block order) to `[n/merge², in_dim·merge²]`,
/// `Linear -> erf-GELU -> Linear`.
fn patch_merger_onnx(g: &mut GraphBuilder, w: &dyn WeightSource, x: &str, in_dim: u32, merge: u32, out_dim: u32, n: u32, eps: f32, out_name: &str) {
    let ln = layernorm_onnx(g, w, x, "merger.ln.weight", "merger.ln.bias", in_dim, eps, "mrg.ln");
    let merged = in_dim * merge * merge;
    let mrows = n / (merge * merge);
    let xr = reshape(g, &ln, &[mrows as i64, merged as i64], "mrg.reshape");
    let h1 = linear_bias(g, w, &xr, "merger.fc1.weight", "merger.fc1.bias", merged, merged, "mrg.fc1");
    let act = gelu_erf(g, &h1, "mrg.act");
    let mm = linear_nb(g, w, &act, "merger.fc2.weight", out_dim, merged, "mrg.fc2");
    let bn = "merger.fc2.b";
    g.init_f32(bn, &[out_dim as i64], w.get("merger.fc2.bias"));
    g.add(Node::new("Add", &[&mm, bn], &[out_name]).name("mrg.out"));
}

/// The vision head, fused end to end: packed patches `[n, patch_vec]` (named
/// `input_name`) -> patch-embed -> `depth`× ViT block -> main `PatchMerger` ->
/// `visual_embeds [n/merge², out_hidden]` (named `out_name`). `w` must carry
/// `patch_embed.{weight,bias}`, `blocks.{b}.*`, `merger.{ln,fc1,fc2}.*`, and
/// `pos_embed` (the raw learned table, resampled here at build time — see
/// [`pos_embed_resampled_host`]). `pos_side` is the learned table's square
/// side (`√num_position_embeddings`); `patch_vec` is
/// `in_channels·temporal_patch_size·patch_size²`.
#[allow(clippy::too_many_arguments)]
pub fn build_vit_head(g: &mut GraphBuilder, topo: &VitTopo, w: &dyn WeightSource, grid_h: u32, grid_w: u32, pos_side: u32, patch_vec: u32, input_name: &str, out_name: &str) {
    let n = grid_h * grid_w;
    let c = topo.hidden;

    // patch-embed: [n,patch_vec] @ patch_embed.weight^T + bias
    let pe = linear_bias(g, w, input_name, "patch_embed.weight", "patch_embed.bias", c, patch_vec, "vpe");

    // host-resampled learned pos-embed, baked as an initializer (fixed grid_h/grid_w per graph).
    let pos_table = w.get("pos_embed");
    let pos_resampled = pos_embed_resampled_host(&pos_table, c as usize, grid_h, grid_w, topo.merge, pos_side);
    let posn = "vit.pos";
    g.init_f32(posn, &[n as i64, c as i64], pos_resampled);
    let mut cur = add_t(g, &pe, posn, "vpe.add_pos");

    // 2-D vision-RoPE tables, baked (fixed grid per graph).
    let positions = vision_position_ids_host(grid_h, grid_w, topo.merge);
    let (cos, sin) = vision_rope_tables_host(&positions, topo.head_dim(), topo.rope_theta);
    g.init_f32("vit.rope.cos", &[n as i64, 1, topo.head_dim() as i64], cos);
    g.init_f32("vit.rope.sin", &[n as i64, 1, topo.head_dim() as i64], sin);

    for b in 0..topo.depth {
        cur = vit_block_onnx(g, topo, w, "vit.rope.cos", "vit.rope.sin", &cur, b, n);
    }
    // No post-block norm (the PatchMerger's own LayerNorm is the final norm).
    patch_merger_onnx(g, w, &cur, c, topo.merge, topo.out_hidden, n, topo.eps, out_name);
}
