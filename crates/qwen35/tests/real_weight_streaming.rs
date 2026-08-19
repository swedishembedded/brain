// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M10: real-weight streaming parity, one decoder layer at a time, against
//! `tools/goldens/qwen35_dump_real_layer_reference.py`'s own dump of the REAL
//! `transformers.models.qwen3_5.Qwen3_5DecoderLayer` forward at the real
//! `Qwen/Qwen3.8-27B-FP8` checkpoint. Covers both mixer types: layer 0
//! (Gated DeltaNet) and layers 3/63 (gated GQA, first and last full-attention
//! layer).
//!
//! **RAM discipline** (the reason this test exists in this shape, not as a
//! full `Qwen35` build): `qwen35::import::import_layer` streams exactly ONE
//! `layers-{l}.safetensors` shard via `checkpoint::mmap::MmapSafetensors`,
//! dequantizing only that layer's own FP8 tensors - never the whole 30.9 GB
//! checkpoint, never a full-model `HashMap` (~108 GB dequantized, far past
//! any RAM budget this box or the task's own 16 GB ceiling could hold). This
//! test drives the ALREADY-HOISTED `model::gdn_mixer`/`model::gqa_mixer`
//! (M12) directly on a standalone `Gpu`, never constructing a `Qwen35`
//! instance at all - the model struct's own construction path still assumes
//! every layer's weights are present, which a real 27B/18 GiB-RAM box cannot
//! do (a recorded gap, not something this test works around).
//!
//! Self-skips loudly (never silently) without `BRAIN_QWEN35_DIR` (the
//! downloaded checkpoint directory) or the matching golden file. Regenerate a
//! layer's golden with:
//!
//! ```text
//! python tools/goldens/qwen35_dump_real_layer_reference.py \
//!     --dir "$BRAIN_QWEN35_DIR" --layer L
//! ```
//!
//! Run with (each layer is several hundred MB of real 27B weights):
//!
//! ```text
//! BRAIN_QWEN35_DIR=/path/to/Qwen3.8-27B-FP8 \
//!     cargo test -p brain-qwen35 --test real_weight_streaming -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use checkpoint::mmap::MmapSafetensors;
use gpu_core::Gpu;
use model::block::{rmsnorm_fwd, swiglu_fwd, KernelIds};
use model::gdn::{GdnBwdIds, GdnIds, GdnShape};
use model::gdn_mixer::{gdn_mixer_fwd, GdnMixerIds, GdnMixerShape, GdnMixerWeights};
use model::gqa_mixer::{gqa_mixer_fwd, GqaMixerIds, GqaMixerShape, GqaMixerWeights};
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::import::import_layer;

/// Every kernel this test's own layer-forward replay dispatches: the mixer
/// internals' own set (mirrors `crates/model/tests/gdn_mixer_equivalence.rs`)
/// plus `matmul`/`add2` for the projections/MLP/residual this test does by
/// hand (no LoRA, no int8 - a plain fp32 replay, matching `crates/qwen35`'s
/// own `layer_gdn_fwd`/`layer_gqa_fwd` minus the LoRA branch).
const KERNELS: &[(&str, &str)] = &[
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rmsnorm_dx", kernels::RMSNORM_DX),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("gqa_bwd_dscores", kernels::GQA_BWD_DSCORES),
    ("gqa_bwd_dv", kernels::GQA_BWD_DV),
    ("gqa_bwd_dq", kernels::GQA_BWD_DQ),
    ("gqa_bwd_dk", kernels::GQA_BWD_DK),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("conv1d", kernels::CONV1D),
    ("conv1d_dx", kernels::CONV1D_DX),
    ("conv1d_dw", kernels::CONV1D_DW),
    ("bmm", kernels::BMM),
    ("bmm_acc", kernels::BMM_ACC),
    ("gdn_chunk_cumsum_step", kernels::GDN_CHUNK_CUMSUM_STEP),
    ("gdn_decay_mask", kernels::GDN_DECAY_MASK),
    ("gdn_mask_strict_lower", kernels::GDN_MASK_STRICT_LOWER),
    ("gdn_ut_step", kernels::GDN_UT_STEP),
    ("gdn_add_identity", kernels::GDN_ADD_IDENTITY),
    ("scale_row", kernels::SCALE_ROW),
    ("gdn_row_scale_off", kernels::GDN_ROW_SCALE_OFF),
    ("gdn_decay_scale", kernels::GDN_DECAY_SCALE),
    ("gdn_state_decay", kernels::GDN_STATE_DECAY),
    ("exp", kernels::EXP),
    ("sub", kernels::SUB),
    ("mul", kernels::MUL),
    ("region_copy", kernels::REGION_COPY),
    ("splice_add", kernels::SPLICE_ADD),
    ("row_dot", kernels::ROW_DOT),
    ("scale_add", kernels::SCALE_ADD),
    ("gdn_chunk_reverse_cumsum_step", kernels::GDN_CHUNK_REVERSE_CUMSUM_STEP),
    ("gdn_ut_bwd_dattn0", kernels::GDN_UT_BWD_DATTN0),
    ("gdn_ut_bwd_dtmat", kernels::GDN_UT_BWD_DTMAT),
    ("gdn_mask_strict_lower_bwd", kernels::GDN_MASK_STRICT_LOWER_BWD),
    ("gdn_decay_mask_bwd", kernels::GDN_DECAY_MASK_BWD),
    ("gdn_decay_scale_bwd", kernels::GDN_DECAY_SCALE_BWD),
    ("gdn_decay_scale_bwd_last", kernels::GDN_DECAY_SCALE_BWD_LAST),
    ("gdn_state_decay_bwd_dscale", kernels::GDN_STATE_DECAY_BWD_DSCALE),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("silu", kernels::SILU),
    ("silu_bwd", kernels::SILU_BWD),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("concat2", kernels::CONCAT2),
    ("l2norm_scale", kernels::L2NORM_SCALE),
    ("l2norm_scale_dx", kernels::L2NORM_SCALE_DX),
    ("sigmoid", kernels::SIGMOID),
    ("sigmoid_bwd", kernels::SIGMOID_BWD),
    ("gdn_decay_gate", kernels::GDN_DECAY_GATE),
    ("gdn_decay_gate_bwd", kernels::GDN_DECAY_GATE_BWD),
    ("kv_expand", kernels::KV_EXPAND),
    ("kv_expand_bwd", kernels::KV_EXPAND_BWD),
    ("gdn_layout_permute", kernels::GDN_LAYOUT_PERMUTE),
    ("bias_grad", kernels::BIAS_GRAD),
    ("rope2d_partial", kernels::ROPE2D_PARTIAL),
    ("matmul", kernels::MATMUL),
    ("add2", kernels::ADD2),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn kernel_ids(g: &Gpu) -> KernelIds {
    KernelIds {
        rmsnorm: idx(g, "rmsnorm"),
        rms_inv: idx(g, "rms_inv"),
        rmsnorm_dx: idx(g, "rmsnorm_dx"),
        rmsnorm_dw: idx(g, "rmsnorm_dw"),
        rope: idx(g, "rmsnorm"),
        rope_bwd: idx(g, "rmsnorm"),
        gqa_scores: idx(g, "gqa_scores"),
        gqa_apply: idx(g, "gqa_apply"),
        attn_softmax: idx(g, "attn_softmax"),
        gqa_dscores: idx(g, "gqa_bwd_dscores"),
        gqa_dv: idx(g, "gqa_bwd_dv"),
        gqa_dq: idx(g, "gqa_bwd_dq"),
        gqa_dk: idx(g, "gqa_bwd_dk"),
        silu_mul: idx(g, "silu_mul"),
        silu_da: idx(g, "silu_bwd_da"),
        silu_db: idx(g, "silu_bwd_db"),
    }
}

fn gdn_mixer_ids(g: &Gpu) -> GdnMixerIds {
    GdnMixerIds {
        kernels: kernel_ids(g),
        conv: audio::conv::ConvKernels { fwd: idx(g, "conv1d"), dx: idx(g, "conv1d_dx"), dw: idx(g, "conv1d_dw") },
        chunk: GdnIds {
            bmm: idx(g, "bmm"),
            bmm_acc: idx(g, "bmm_acc"),
            cumsum_step: idx(g, "gdn_chunk_cumsum_step"),
            decay_mask: idx(g, "gdn_decay_mask"),
            mask_strict_lower: idx(g, "gdn_mask_strict_lower"),
            ut_step: idx(g, "gdn_ut_step"),
            add_identity: idx(g, "gdn_add_identity"),
            row_scale: idx(g, "scale_row"),
            row_scale_off: idx(g, "gdn_row_scale_off"),
            decay_scale: idx(g, "gdn_decay_scale"),
            state_decay: idx(g, "gdn_state_decay"),
            exp: idx(g, "exp"),
            sub: idx(g, "sub"),
            mul: idx(g, "mul"),
            region_copy: idx(g, "region_copy"),
        },
        chunk_bwd: GdnBwdIds {
            splice_add: idx(g, "splice_add"),
            row_dot: idx(g, "row_dot"),
            scale_add: idx(g, "scale_add"),
            reverse_cumsum_step: idx(g, "gdn_chunk_reverse_cumsum_step"),
            ut_bwd_dattn0: idx(g, "gdn_ut_bwd_dattn0"),
            ut_bwd_dtmat: idx(g, "gdn_ut_bwd_dtmat"),
            mask_strict_lower_bwd: idx(g, "gdn_mask_strict_lower_bwd"),
            decay_mask_bwd: idx(g, "gdn_decay_mask_bwd"),
            decay_scale_bwd: idx(g, "gdn_decay_scale_bwd"),
            decay_scale_bwd_last: idx(g, "gdn_decay_scale_bwd_last"),
            state_decay_bwd_dscale: idx(g, "gdn_state_decay_bwd_dscale"),
        },
        nlc_nchw: idx(g, "nlc_nchw"),
        nchw_nlc: idx(g, "nchw_nlc"),
        silu: idx(g, "silu"),
        silu_bwd: idx(g, "silu_bwd"),
        concat_split: idx(g, "concat_split"),
        concat2: idx(g, "concat2"),
        l2norm_scale: idx(g, "l2norm_scale"),
        l2norm_scale_dx: idx(g, "l2norm_scale_dx"),
        sigmoid: idx(g, "sigmoid"),
        sigmoid_bwd: idx(g, "sigmoid_bwd"),
        gdn_decay_gate: idx(g, "gdn_decay_gate"),
        gdn_decay_gate_bwd: idx(g, "gdn_decay_gate_bwd"),
        kv_expand: idx(g, "kv_expand"),
        kv_expand_bwd: idx(g, "kv_expand_bwd"),
        gdn_layout_permute: idx(g, "gdn_layout_permute"),
        mul: idx(g, "mul"),
        bias_grad: idx(g, "bias_grad"),
    }
}

fn gqa_mixer_ids(g: &Gpu) -> GqaMixerIds {
    GqaMixerIds {
        kernels: kernel_ids(g),
        concat_split: idx(g, "concat_split"),
        concat2: idx(g, "concat2"),
        sigmoid: idx(g, "sigmoid"),
        sigmoid_bwd: idx(g, "sigmoid_bwd"),
        mul: idx(g, "mul"),
        rope2d_partial: idx(g, "rope2d_partial"),
    }
}

fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("BRAIN_QWEN35_DIR").map(PathBuf::from)
}

fn golden_dir(l: usize) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden/qwen35").join(format!("real_layer_{l}"))
}

fn run_layer_parity(l: usize) {
    let Some(dir) = checkpoint_dir() else {
        brain_testutil::skip("BRAIN_QWEN35_DIR unset (set it to a downloaded Qwen/Qwen3.8-27B-FP8 dir to run this - see this file's own doc)");
        return;
    };
    let shard = dir.join(format!("layers-{l}.safetensors"));
    if !shard.exists() {
        brain_testutil::skip_unavailable(&format!("{} not present under BRAIN_QWEN35_DIR", shard.display()));
        return;
    }
    let golden_path = golden_dir(l).join("layer.safetensors");
    if !golden_path.exists() {
        brain_testutil::skip(&format!(
            "{} missing - run: python tools/goldens/qwen35_dump_real_layer_reference.py --dir {} --layer {l}",
            golden_path.display(),
            dir.display()
        ));
        return;
    }

    brain_testutil::mem(&format!("layer {l}: before import"));
    let cfg = Qwen35Config::qwen38_27b();
    let ty = cfg.layer_types()[l];
    let d = cfg.d_model;

    let reader = MmapSafetensors::open(&shard).unwrap_or_else(|e| panic!("open {}: {e}", shard.display()));
    let w = import_layer(&reader, &cfg, l, 128).unwrap_or_else(|e| panic!("import_layer({l}): {e}"));
    drop(reader);
    brain_testutil::mem(&format!("layer {l}: after import_layer"));

    let golden = MmapSafetensors::open(&golden_path).unwrap_or_else(|e| panic!("open {}: {e}", golden_path.display()));
    let t = golden.shape("x_in").unwrap()[0] as u32;
    let x_in = golden.tensor_f32("x_in").unwrap();
    let want_out = golden.tensor_f32("out").unwrap();
    assert_eq!(x_in.len(), (t * d) as usize);
    assert_eq!(want_out.len(), (t * d) as usize);

    let g = Gpu::new_cpu(KERNELS);
    let p = |s: &str| format!("blocks.{l}.{s}");
    let wbuf = |name: &str| g.storage_init(name, &w[name]);

    let x = g.storage_init("x_in", &x_in);
    let ln1w = wbuf(&p("ln1.weight"));
    let xn1 = g.storage((t * d) as u64);
    g.submit(&[], &[rmsnorm_fwd(&g, &kernel_ids(&g), &x, &ln1w, &xn1, d, t)]);

    let mixer_out = match ty {
        LayerType::Linear => {
            let conv_dim = cfg.linear_conv_dim();
            let value_dim = cfg.linear_value_dim();
            let nvh = cfg.linear_num_value_heads;
            let khd = cfg.linear_key_head_dim;
            let vhd = cfg.linear_value_head_dim;

            let mixed_qkv = g.storage((t * conv_dim) as u64);
            let bproj = g.storage((t * nvh) as u64);
            let aproj = g.storage((t * nvh) as u64);
            let z = g.storage((t * value_dim) as u64);
            let qkvw = wbuf(&p("linear_attn.in_proj_qkv.weight"));
            let bw = wbuf(&p("linear_attn.in_proj_b.weight"));
            let aw = wbuf(&p("linear_attn.in_proj_a.weight"));
            let zw = wbuf(&p("linear_attn.in_proj_z.weight"));
            g.submit(
                &[],
                &[
                    g.step(idx(&g, "matmul"), &[&xn1, &qkvw, &mixed_qkv], &[t, d, conv_dim], t * conv_dim),
                    g.step(idx(&g, "matmul"), &[&xn1, &bw, &bproj], &[t, d, nvh], t * nvh),
                    g.step(idx(&g, "matmul"), &[&xn1, &aw, &aproj], &[t, d, nvh], t * nvh),
                    g.step(idx(&g, "matmul"), &[&xn1, &zw, &z], &[t, d, value_dim], t * value_dim),
                ],
            );

            let shape = GdnMixerShape {
                gdn: GdnShape { b: 1, h: nvh, t, dk: khd, dv: vhd, chunk: model::gdn::gdn_chunk_size(t) },
                nkh: cfg.linear_num_key_heads,
                conv_kernel: cfg.linear_conv_kernel_dim,
            };
            let ones_khd = g.storage_init("ones_khd", &vec![1.0f32; khd as usize]);
            let weights = GdnMixerWeights {
                conv1d_weight: &wbuf(&p("linear_attn.conv1d.weight")),
                a_log: &wbuf(&p("linear_attn.A_log")),
                dt_bias: &wbuf(&p("linear_attn.dt_bias")),
                norm_weight: &wbuf(&p("linear_attn.norm.weight")),
                ones_khd: &ones_khd,
            };
            let (gated, _acts) = gdn_mixer_fwd(&g, &gdn_mixer_ids(&g), &shape, &weights, &mixed_qkv, &bproj, &aproj, &z, t, false);

            let out = g.storage((t * d) as u64);
            let outw = wbuf(&p("linear_attn.out_proj.weight"));
            g.submit(&[], &[g.step(idx(&g, "matmul"), &[&gated, &outw, &out], &[t, value_dim, d], t * d)]);
            out
        }
        LayerType::Full => {
            let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
            let (qpd, kvd) = (cfg.q_proj_dim(), cfg.kv_dim());

            let q_full = g.storage((t * qpd) as u64);
            let k = g.storage((t * kvd) as u64);
            let v = g.storage((t * kvd) as u64);
            let qw = wbuf(&p("self_attn.q_proj.weight"));
            let kw = wbuf(&p("self_attn.k_proj.weight"));
            let vw = wbuf(&p("self_attn.v_proj.weight"));
            g.submit(
                &[],
                &[
                    g.step(idx(&g, "matmul"), &[&xn1, &qw, &q_full], &[t, d, qpd], t * qpd),
                    g.step(idx(&g, "matmul"), &[&xn1, &kw, &k], &[t, d, kvd], t * kvd),
                    g.step(idx(&g, "matmul"), &[&xn1, &vw, &v], &[t, d, kvd], t * kvd),
                ],
            );

            let positions: Vec<[u32; 3]> = (0..t).map(|ti| [ti, ti, ti]).collect();
            let (cos, sin) = qwen3vl::mrope::mrope_tables(&positions, cfg.mrope_section, cfg.rotary_dim(), cfg.rope_theta);
            let cos = g.storage_init("cos", &cos);
            let sin = g.storage_init("sin", &sin);
            let shape = GqaMixerShape { b: 1, t, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: cfg.rotary_dim() / 2 };
            let weights = GqaMixerWeights {
                q_norm: &wbuf(&p("self_attn.q_norm.weight")),
                k_norm: &wbuf(&p("self_attn.k_norm.weight")),
                cos: &cos,
                sin: &sin,
            };
            let (ctx_gated, _acts) = gqa_mixer_fwd(&g, &gqa_mixer_ids(&g), &shape, &weights, &q_full, &k, &v, t, false);

            let out = g.storage((t * d) as u64);
            let ow = wbuf(&p("self_attn.o_proj.weight"));
            g.submit(&[], &[g.step(idx(&g, "matmul"), &[&ctx_gated, &ow, &out], &[t, shape.qd(), d], t * d)]);
            out
        }
    };

    let h1 = g.storage((t * d) as u64);
    g.submit(&[], &[g.step(idx(&g, "add2"), &[&x, &mixer_out, &h1], &[t * d], t * d)]);

    let ln2w = wbuf(&p("ln2.weight"));
    let xn2 = g.storage((t * d) as u64);
    g.submit(&[], &[rmsnorm_fwd(&g, &kernel_ids(&g), &h1, &ln2w, &xn2, d, t)]);

    let ff = cfg.intermediate_size;
    let gatew = wbuf(&p("mlp.gate.weight"));
    let upw = wbuf(&p("mlp.up.weight"));
    let downw = wbuf(&p("mlp.down.weight"));
    let gate_pre = g.storage((t * ff) as u64);
    let up = g.storage((t * ff) as u64);
    g.submit(
        &[],
        &[
            g.step(idx(&g, "matmul"), &[&xn2, &gatew, &gate_pre], &[t, d, ff], t * ff),
            g.step(idx(&g, "matmul"), &[&xn2, &upw, &up], &[t, d, ff], t * ff),
        ],
    );
    let h_act = g.storage((t * ff) as u64);
    g.submit(&[], &[swiglu_fwd(&g, &kernel_ids(&g), &gate_pre, &up, &h_act, t * ff)]);
    let down = g.storage((t * d) as u64);
    g.submit(&[], &[g.step(idx(&g, "matmul"), &[&h_act, &downw, &down], &[t, ff, d], t * d)]);

    let out = g.storage((t * d) as u64);
    g.submit(&[], &[g.step(idx(&g, "add2"), &[&h1, &down, &out], &[t * d], t * d)]);

    let got = g.read(&out, (t * d) as usize);
    let (cos, max_abs) = brain_testutil::parity::compare(&got, &want_out);
    let rel = brain_testutil::parity::rel_l2(&got, &want_out);
    brain_testutil::mem(&format!("layer {l}: after forward"));
    eprintln!("layer {l} ({ty:?}): cosine={cos:.9} rel_l2={rel:.6} max_abs={max_abs:.4}");
    assert!(got.iter().all(|v| v.is_finite()), "layer {l}: brain forward produced a non-finite value");
    assert!(cos > 0.999, "layer {l}: cosine={cos:.9} too low (want > 0.999)");
    assert!(rel < 0.05, "layer {l}: rel_l2={rel:.6} too high (want < 0.05)");
}

#[test]
#[ignore]
fn layer_0_gated_delta_net_matches_the_real_reference() {
    run_layer_parity(0);
}

#[test]
#[ignore]
fn layer_3_gated_gqa_matches_the_real_reference() {
    run_layer_parity(3);
}

#[test]
#[ignore]
fn layer_63_gated_gqa_matches_the_real_reference() {
    run_layer_parity(63);
}

/// A row-value spot check on `model.language_model.embed_tokens.weight`/
/// `lm_head.weight` (both plain BF16, never FP8, `[vocab, d_model]`,
/// ~2.5 GB each) - the third M10 bullet ("embedding/lm_head spot checks in
/// isolation"). Reads only the SPECIFIC rows
/// `tools/goldens/qwen35_dump_embed_lm_head_rows.py` dumped, via
/// `with_tensor_chunks` (bounded extra host allocation, `O(d_model)` per
/// row, never `O(vocab*d_model)` for the whole table) - never a full-table
/// decode of either 2.5 GB tensor. Trades wall-clock for that RAM bound:
/// `with_tensor_chunks` decodes sequentially from offset 0, so reaching the
/// last of ~250K rows genuinely walks every row before it (~3 minutes on
/// this box) - the right trade for a correctness spot check that runs on
/// demand, not in a hot loop.
#[test]
#[ignore]
fn embed_and_lm_head_rows_match_the_real_reference() {
    let Some(dir) = checkpoint_dir() else {
        brain_testutil::skip("BRAIN_QWEN35_DIR unset (set it to a downloaded Qwen/Qwen3.8-27B-FP8 dir to run this - see this file's own doc)");
        return;
    };
    let shard = dir.join("outside.safetensors");
    if !shard.exists() {
        brain_testutil::skip_unavailable(&format!("{} not present under BRAIN_QWEN35_DIR", shard.display()));
        return;
    }
    let golden_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden/qwen35/embed_lm_head/rows.safetensors");
    if !golden_path.exists() {
        brain_testutil::skip(&format!(
            "{} missing - run: python tools/goldens/qwen35_dump_embed_lm_head_rows.py --dir {}",
            golden_path.display(),
            dir.display()
        ));
        return;
    }

    brain_testutil::mem("embed/lm_head: before");
    let cfg = Qwen35Config::qwen38_27b();
    let d = cfg.d_model as usize;

    let golden = MmapSafetensors::open(&golden_path).unwrap_or_else(|e| panic!("open {}: {e}", golden_path.display()));
    let token_ids: Vec<u32> = golden.tensor_f32("token_ids").unwrap().into_iter().map(|v| v as u32).collect();
    let want_embed = golden.tensor_f32("embed_rows").unwrap();
    let want_head = golden.tensor_f32("lm_head_rows").unwrap();

    let reader = MmapSafetensors::open(&shard).unwrap_or_else(|e| panic!("open {}: {e}", shard.display()));
    let extract_rows = |name: &str, rows: &[u32]| -> Vec<f32> {
        let mut out = vec![0f32; rows.len() * d];
        let found = reader.with_tensor_chunks(name, d, &mut |off, chunk| {
            let row = off as usize / d;
            if let Some(pos) = rows.iter().position(|&r| r as usize == row) {
                out[pos * d..(pos + 1) * d].copy_from_slice(chunk);
            }
        });
        assert!(found, "{name} not present in {}", shard.display());
        out
    };
    let got_embed = extract_rows("model.language_model.embed_tokens.weight", &token_ids);
    let got_head = extract_rows("lm_head.weight", &token_ids);
    brain_testutil::mem("embed/lm_head: after");

    let (cos_e, max_e) = brain_testutil::parity::compare(&got_embed, &want_embed);
    let (cos_h, max_h) = brain_testutil::parity::compare(&got_head, &want_head);
    eprintln!("embed_tokens rows {token_ids:?}: cosine={cos_e:.9} max_abs={max_e:.6}");
    eprintln!("lm_head rows {token_ids:?}: cosine={cos_h:.9} max_abs={max_h:.6}");
    // BF16 decode is a deterministic bit-widening on both sides - this
    // should be an exact match, not just cosine-close; the tolerance here
    // only covers the two f64-accumulated reductions' own rounding.
    assert!(cos_e > 0.999999, "embed_tokens rows diverged: cosine={cos_e:.9}");
    assert!(cos_h > 0.999999, "lm_head rows diverged: cosine={cos_h:.9}");
    assert!(max_e < 1e-3, "embed_tokens rows diverged: max_abs={max_e:.6}");
    assert!(max_h < 1e-3, "lm_head rows diverged: max_abs={max_h:.6}");
}
