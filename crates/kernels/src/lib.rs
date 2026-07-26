// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Raw WGSL compute kernels — the single source of truth for brain's GPU
//! engine. fp32-only, core-compute-only (single bind group, <=4 storage
//! buffers/kernel, no atomics/subgroups/f16) so the same text runs on old
//! desktop GPUs and on WebGPU in the browser.
//!
//! Workgroup size is `@workgroup_size(64)` everywhere except the register-tiled
//! GEMMs (`matmul_reg*`), which need 256 invocations to hold a 128x128 output
//! tile. Backends read each kernel's declared size via
//! `backend_api::workgroup_size_of`, so a kernel's WGSL is the only place its
//! size is written down — but a kernel that departs from 64 must also
//! reconstruct its flat invocation id with its own size.
//!
//! Each `.wgsl` file under `wgsl/` is embedded as a `pub const` (UPPER_SNAKE of
//! the file stem) and indexed by name via [`src`]. This crate is pure string
//! data: no GPU, no std beyond core, safe for every target including wasm32.

/// `wgsl/adamw.wgsl`
pub const ADAMW: &str = include_str!("../wgsl/adamw.wgsl");
/// `wgsl/add.wgsl`
pub const ADD: &str = include_str!("../wgsl/add.wgsl");
/// `wgsl/add2.wgsl`
pub const ADD2: &str = include_str!("../wgsl/add2.wgsl");
/// `wgsl/add_chan_bcast.wgsl`
pub const ADD_CHAN_BCAST: &str = include_str!("../wgsl/add_chan_bcast.wgsl");
/// `wgsl/add_chan_bcast_dv.wgsl`
pub const ADD_CHAN_BCAST_DV: &str = include_str!("../wgsl/add_chan_bcast_dv.wgsl");
/// `wgsl/add_chan_inplace.wgsl`
pub const ADD_CHAN_INPLACE: &str = include_str!("../wgsl/add_chan_inplace.wgsl");
/// `wgsl/add_index_mask.wgsl`
pub const ADD_INDEX_MASK: &str = include_str!("../wgsl/add_index_mask.wgsl");
/// `wgsl/add_inplace.wgsl`
pub const ADD_INPLACE: &str = include_str!("../wgsl/add_inplace.wgsl");
/// `wgsl/attention.wgsl`
pub const ATTENTION: &str = include_str!("../wgsl/attention.wgsl");
/// `wgsl/attn_apply.wgsl`
pub const ATTN_APPLY: &str = include_str!("../wgsl/attn_apply.wgsl");
/// `wgsl/attn_apply_bidir.wgsl`
pub const ATTN_APPLY_BIDIR: &str = include_str!("../wgsl/attn_apply_bidir.wgsl");
/// `wgsl/attn_apply_cross.wgsl`
pub const ATTN_APPLY_CROSS: &str = include_str!("../wgsl/attn_apply_cross.wgsl");
/// `wgsl/attn_apply_full.wgsl`
pub const ATTN_APPLY_FULL: &str = include_str!("../wgsl/attn_apply_full.wgsl");
/// `wgsl/attn_bwd_dbias.wgsl`
pub const ATTN_BWD_DBIAS: &str = include_str!("../wgsl/attn_bwd_dbias.wgsl");
/// `wgsl/attn_bwd_dk.wgsl`
pub const ATTN_BWD_DK: &str = include_str!("../wgsl/attn_bwd_dk.wgsl");
/// `wgsl/attn_bwd_dk_bias.wgsl`
pub const ATTN_BWD_DK_BIAS: &str = include_str!("../wgsl/attn_bwd_dk_bias.wgsl");
/// `wgsl/attn_bwd_dk_bidir.wgsl`
pub const ATTN_BWD_DK_BIDIR: &str = include_str!("../wgsl/attn_bwd_dk_bidir.wgsl");
/// `wgsl/attn_bwd_dk_cross.wgsl`
pub const ATTN_BWD_DK_CROSS: &str = include_str!("../wgsl/attn_bwd_dk_cross.wgsl");
/// `wgsl/attn_bwd_dq.wgsl`
pub const ATTN_BWD_DQ: &str = include_str!("../wgsl/attn_bwd_dq.wgsl");
/// `wgsl/attn_bwd_dq_bias.wgsl`
pub const ATTN_BWD_DQ_BIAS: &str = include_str!("../wgsl/attn_bwd_dq_bias.wgsl");
/// `wgsl/attn_bwd_dq_bidir.wgsl`
pub const ATTN_BWD_DQ_BIDIR: &str = include_str!("../wgsl/attn_bwd_dq_bidir.wgsl");
/// `wgsl/attn_bwd_dq_cross.wgsl`
pub const ATTN_BWD_DQ_CROSS: &str = include_str!("../wgsl/attn_bwd_dq_cross.wgsl");
/// `wgsl/attn_bwd_dscores.wgsl`
pub const ATTN_BWD_DSCORES: &str = include_str!("../wgsl/attn_bwd_dscores.wgsl");
/// `wgsl/attn_bwd_dscores_bidir.wgsl`
pub const ATTN_BWD_DSCORES_BIDIR: &str = include_str!("../wgsl/attn_bwd_dscores_bidir.wgsl");
/// `wgsl/attn_bwd_dscores_cross.wgsl`
pub const ATTN_BWD_DSCORES_CROSS: &str = include_str!("../wgsl/attn_bwd_dscores_cross.wgsl");
/// `wgsl/attn_bwd_dv.wgsl`
pub const ATTN_BWD_DV: &str = include_str!("../wgsl/attn_bwd_dv.wgsl");
/// `wgsl/attn_bwd_dv_bidir.wgsl`
pub const ATTN_BWD_DV_BIDIR: &str = include_str!("../wgsl/attn_bwd_dv_bidir.wgsl");
/// `wgsl/attn_bwd_dv_cross.wgsl`
pub const ATTN_BWD_DV_CROSS: &str = include_str!("../wgsl/attn_bwd_dv_cross.wgsl");
/// `wgsl/attn_scores.wgsl`
pub const ATTN_SCORES: &str = include_str!("../wgsl/attn_scores.wgsl");
/// `wgsl/attn_scores_bidir.wgsl`
pub const ATTN_SCORES_BIDIR: &str = include_str!("../wgsl/attn_scores_bidir.wgsl");
/// `wgsl/attn_scores_bidir_bias.wgsl`
pub const ATTN_SCORES_BIDIR_BIAS: &str = include_str!("../wgsl/attn_scores_bidir_bias.wgsl");
/// `wgsl/attn_scores_causal_bias.wgsl`
pub const ATTN_SCORES_CAUSAL_BIAS: &str = include_str!("../wgsl/attn_scores_causal_bias.wgsl");
/// `wgsl/attn_scores_cross.wgsl`
pub const ATTN_SCORES_CROSS: &str = include_str!("../wgsl/attn_scores_cross.wgsl");
/// `wgsl/attn_scores_full.wgsl`
pub const ATTN_SCORES_FULL: &str = include_str!("../wgsl/attn_scores_full.wgsl");
/// `wgsl/attn_scores_masked.wgsl`
pub const ATTN_SCORES_MASKED: &str = include_str!("../wgsl/attn_scores_masked.wgsl");
/// `wgsl/attn_scores_qk.wgsl`
pub const ATTN_SCORES_QK: &str = include_str!("../wgsl/attn_scores_qk.wgsl");
/// `wgsl/attn_softmax.wgsl`
pub const ATTN_SOFTMAX: &str = include_str!("../wgsl/attn_softmax.wgsl");
/// `wgsl/attn_softmax_bidir.wgsl`
pub const ATTN_SOFTMAX_BIDIR: &str = include_str!("../wgsl/attn_softmax_bidir.wgsl");
/// `wgsl/attn_softmax_cross.wgsl`
pub const ATTN_SOFTMAX_CROSS: &str = include_str!("../wgsl/attn_softmax_cross.wgsl");
/// `wgsl/attn_softmax_full.wgsl`
pub const ATTN_SOFTMAX_FULL: &str = include_str!("../wgsl/attn_softmax_full.wgsl");
/// `wgsl/attn_softmax_masked.wgsl`
pub const ATTN_SOFTMAX_MASKED: &str = include_str!("../wgsl/attn_softmax_masked.wgsl");
/// `wgsl/avgpool2d.wgsl`
pub const AVGPOOL2D: &str = include_str!("../wgsl/avgpool2d.wgsl");
/// `wgsl/avgpool2d_dx.wgsl`
pub const AVGPOOL2D_DX: &str = include_str!("../wgsl/avgpool2d_dx.wgsl");
/// `wgsl/axpy.wgsl`
pub const AXPY: &str = include_str!("../wgsl/axpy.wgsl");
/// `wgsl/bce_logits.wgsl`
pub const BCE_LOGITS: &str = include_str!("../wgsl/bce_logits.wgsl");
/// `wgsl/bce_logits_grad.wgsl`
pub const BCE_LOGITS_GRAD: &str = include_str!("../wgsl/bce_logits_grad.wgsl");
/// `wgsl/bias_add.wgsl`
pub const BIAS_ADD: &str = include_str!("../wgsl/bias_add.wgsl");
/// `wgsl/bias_grad.wgsl`
pub const BIAS_GRAD: &str = include_str!("../wgsl/bias_grad.wgsl");
/// `wgsl/bn_dbeta.wgsl`
pub const BN_DBETA: &str = include_str!("../wgsl/bn_dbeta.wgsl");
/// `wgsl/bn_dgamma.wgsl`
pub const BN_DGAMMA: &str = include_str!("../wgsl/bn_dgamma.wgsl");
/// `wgsl/bn_dstats.wgsl`
pub const BN_DSTATS: &str = include_str!("../wgsl/bn_dstats.wgsl");
/// `wgsl/bn_dx.wgsl`
pub const BN_DX: &str = include_str!("../wgsl/bn_dx.wgsl");
/// `wgsl/bn_eval.wgsl`
pub const BN_EVAL: &str = include_str!("../wgsl/bn_eval.wgsl");
/// `wgsl/bn_running.wgsl`
pub const BN_RUNNING: &str = include_str!("../wgsl/bn_running.wgsl");
/// `wgsl/bn_stats.wgsl`
pub const BN_STATS: &str = include_str!("../wgsl/bn_stats.wgsl");
/// `wgsl/bn_train.wgsl`
pub const BN_TRAIN: &str = include_str!("../wgsl/bn_train.wgsl");
/// `wgsl/broadcast_add_hw.wgsl`
pub const BROADCAST_ADD_HW: &str = include_str!("../wgsl/broadcast_add_hw.wgsl");
/// `wgsl/broadcast_add_hw_da.wgsl`
pub const BROADCAST_ADD_HW_DA: &str = include_str!("../wgsl/broadcast_add_hw_da.wgsl");
/// `wgsl/bsq_quantize.wgsl`
pub const BSQ_QUANTIZE: &str = include_str!("../wgsl/bsq_quantize.wgsl");
/// `wgsl/ce_grad.wgsl`
pub const CE_GRAD: &str = include_str!("../wgsl/ce_grad.wgsl");
/// `wgsl/ce_grad_masked.wgsl`
pub const CE_GRAD_MASKED: &str = include_str!("../wgsl/ce_grad_masked.wgsl");
/// `wgsl/ce_grad_stats.wgsl`
pub const CE_GRAD_STATS: &str = include_str!("../wgsl/ce_grad_stats.wgsl");
/// `wgsl/ce_stats.wgsl`
pub const CE_STATS: &str = include_str!("../wgsl/ce_stats.wgsl");
/// `wgsl/ce_value.wgsl`
pub const CE_VALUE: &str = include_str!("../wgsl/ce_value.wgsl");
/// `wgsl/ce_value_masked.wgsl`
pub const CE_VALUE_MASKED: &str = include_str!("../wgsl/ce_value_masked.wgsl");
/// `wgsl/chan_place.wgsl`
pub const CHAN_PLACE: &str = include_str!("../wgsl/chan_place.wgsl");
/// `wgsl/ciou.wgsl`
pub const CIOU: &str = include_str!("../wgsl/ciou.wgsl");
/// `wgsl/ciou_grad.wgsl`
pub const CIOU_GRAD: &str = include_str!("../wgsl/ciou_grad.wgsl");
/// `wgsl/clip_coef.wgsl`
pub const CLIP_COEF: &str = include_str!("../wgsl/clip_coef.wgsl");
/// `wgsl/concat2.wgsl`
pub const CONCAT2: &str = include_str!("../wgsl/concat2.wgsl");
/// `wgsl/concat_split.wgsl`
pub const CONCAT_SPLIT: &str = include_str!("../wgsl/concat_split.wgsl");
/// `wgsl/conv1d.wgsl`
pub const CONV1D: &str = include_str!("../wgsl/conv1d.wgsl");
/// `wgsl/conv1d_dw.wgsl`
pub const CONV1D_DW: &str = include_str!("../wgsl/conv1d_dw.wgsl");
/// `wgsl/conv1d_dx.wgsl`
pub const CONV1D_DX: &str = include_str!("../wgsl/conv1d_dx.wgsl");
/// `wgsl/conv2d.wgsl`
pub const CONV2D: &str = include_str!("../wgsl/conv2d.wgsl");
/// `wgsl/conv2d_dw.wgsl`
pub const CONV2D_DW: &str = include_str!("../wgsl/conv2d_dw.wgsl");
/// `wgsl/conv2d_dx.wgsl`
pub const CONV2D_DX: &str = include_str!("../wgsl/conv2d_dx.wgsl");
/// `wgsl/conv2d_gd.wgsl`
pub const CONV2D_GD: &str = include_str!("../wgsl/conv2d_gd.wgsl");
/// `wgsl/conv2d_gd_dw.wgsl`
pub const CONV2D_GD_DW: &str = include_str!("../wgsl/conv2d_gd_dw.wgsl");
/// `wgsl/conv2d_gd_dx.wgsl`
pub const CONV2D_GD_DX: &str = include_str!("../wgsl/conv2d_gd_dx.wgsl");
/// `wgsl/conv2d_gd_reg.wgsl`
pub const CONV2D_GD_REG: &str = include_str!("../wgsl/conv2d_gd_reg.wgsl");
/// `wgsl/conv2d_tiled.wgsl`
pub const CONV2D_TILED: &str = include_str!("../wgsl/conv2d_tiled.wgsl");
/// `wgsl/conv_act.wgsl`
pub const CONV_ACT: &str = include_str!("../wgsl/conv_act.wgsl");
/// `wgsl/conv_act_reg.wgsl`
pub const CONV_ACT_REG: &str = include_str!("../wgsl/conv_act_reg.wgsl");
/// `wgsl/conv_act_tiled.wgsl`
pub const CONV_ACT_TILED: &str = include_str!("../wgsl/conv_act_tiled.wgsl");
/// `wgsl/conv_bias.wgsl`
pub const CONV_BIAS: &str = include_str!("../wgsl/conv_bias.wgsl");
/// `wgsl/conv_bias_reg.wgsl`
pub const CONV_BIAS_REG: &str = include_str!("../wgsl/conv_bias_reg.wgsl");
/// `wgsl/conv_epilogue.wgsl`
pub const CONV_EPILOGUE: &str = include_str!("../wgsl/conv_epilogue.wgsl");
/// `wgsl/convex_upsample.wgsl`
pub const CONVEX_UPSAMPLE: &str = include_str!("../wgsl/convex_upsample.wgsl");
/// `wgsl/convex_upsample_dd.wgsl`
pub const CONVEX_UPSAMPLE_DD: &str = include_str!("../wgsl/convex_upsample_dd.wgsl");
/// `wgsl/convex_upsample_dmask.wgsl`
pub const CONVEX_UPSAMPLE_DMASK: &str = include_str!("../wgsl/convex_upsample_dmask.wgsl");
/// `wgsl/convtr1d.wgsl`
pub const CONVTR1D: &str = include_str!("../wgsl/convtr1d.wgsl");
/// `wgsl/convtr1d_dw.wgsl`
pub const CONVTR1D_DW: &str = include_str!("../wgsl/convtr1d_dw.wgsl");
/// `wgsl/convtr1d_dx.wgsl`
pub const CONVTR1D_DX: &str = include_str!("../wgsl/convtr1d_dx.wgsl");
/// `wgsl/crop2d.wgsl`
pub const CROP2D: &str = include_str!("../wgsl/crop2d.wgsl");
/// `wgsl/dfl_decode.wgsl`
pub const DFL_DECODE: &str = include_str!("../wgsl/dfl_decode.wgsl");
/// `wgsl/dfl_grad.wgsl`
pub const DFL_GRAD: &str = include_str!("../wgsl/dfl_grad.wgsl");
/// `wgsl/dfl_loss.wgsl`
pub const DFL_LOSS: &str = include_str!("../wgsl/dfl_loss.wgsl");
/// `wgsl/dfl_loss_grad.wgsl`
pub const DFL_LOSS_GRAD: &str = include_str!("../wgsl/dfl_loss_grad.wgsl");
/// `wgsl/dwconv3d.wgsl`
pub const DWCONV3D: &str = include_str!("../wgsl/dwconv3d.wgsl");
/// `wgsl/dwconv3d_dw.wgsl`
pub const DWCONV3D_DW: &str = include_str!("../wgsl/dwconv3d_dw.wgsl");
/// `wgsl/dwconv3d_dx.wgsl`
pub const DWCONV3D_DX: &str = include_str!("../wgsl/dwconv3d_dx.wgsl");
/// `wgsl/edm_mix.wgsl`
pub const EDM_MIX: &str = include_str!("../wgsl/edm_mix.wgsl");
/// `wgsl/edm_wrap.wgsl`
pub const EDM_WRAP: &str = include_str!("../wgsl/edm_wrap.wgsl");
/// `wgsl/emb_bwd.wgsl`
pub const EMB_BWD: &str = include_str!("../wgsl/emb_bwd.wgsl");
/// `wgsl/embed.wgsl`
pub const EMBED: &str = include_str!("../wgsl/embed.wgsl");
/// `wgsl/embed_tile.wgsl`
pub const EMBED_TILE: &str = include_str!("../wgsl/embed_tile.wgsl");
/// `wgsl/expert_counts.wgsl`
pub const EXPERT_COUNTS: &str = include_str!("../wgsl/expert_counts.wgsl");
/// `wgsl/film_chan.wgsl`
pub const FILM_CHAN: &str = include_str!("../wgsl/film_chan.wgsl");
/// `wgsl/film_chan_dsb.wgsl`
pub const FILM_CHAN_DSB: &str = include_str!("../wgsl/film_chan_dsb.wgsl");
/// `wgsl/film_chan_dx.wgsl`
pub const FILM_CHAN_DX: &str = include_str!("../wgsl/film_chan_dx.wgsl");
/// `wgsl/film_row.wgsl`
pub const FILM_ROW: &str = include_str!("../wgsl/film_row.wgsl");
/// `wgsl/film_row_dsb.wgsl`
pub const FILM_ROW_DSB: &str = include_str!("../wgsl/film_row_dsb.wgsl");
/// `wgsl/film_row_dx.wgsl`
pub const FILM_ROW_DX: &str = include_str!("../wgsl/film_row_dx.wgsl");
/// `wgsl/gate_row.wgsl`
pub const GATE_ROW: &str = include_str!("../wgsl/gate_row.wgsl");
/// `wgsl/gate_row_dg.wgsl`
pub const GATE_ROW_DG: &str = include_str!("../wgsl/gate_row_dg.wgsl");
/// `wgsl/gate_row_dh.wgsl`
pub const GATE_ROW_DH: &str = include_str!("../wgsl/gate_row_dh.wgsl");
/// `wgsl/gelu.wgsl`
pub const GELU: &str = include_str!("../wgsl/gelu.wgsl");
/// `wgsl/gelu_bwd.wgsl`
pub const GELU_BWD: &str = include_str!("../wgsl/gelu_bwd.wgsl");
/// `wgsl/gelu_erf.wgsl`
pub const GELU_ERF: &str = include_str!("../wgsl/gelu_erf.wgsl");
/// `wgsl/gelu_erf_bwd.wgsl`
pub const GELU_ERF_BWD: &str = include_str!("../wgsl/gelu_erf_bwd.wgsl");
/// `wgsl/gn_apply.wgsl`
pub const GN_APPLY: &str = include_str!("../wgsl/gn_apply.wgsl");
/// `wgsl/gn_dbeta.wgsl`
pub const GN_DBETA: &str = include_str!("../wgsl/gn_dbeta.wgsl");
/// `wgsl/gn_dgamma.wgsl`
pub const GN_DGAMMA: &str = include_str!("../wgsl/gn_dgamma.wgsl");
/// `wgsl/gn_dsum.wgsl`
pub const GN_DSUM: &str = include_str!("../wgsl/gn_dsum.wgsl");
/// `wgsl/gn_dx.wgsl`
pub const GN_DX: &str = include_str!("../wgsl/gn_dx.wgsl");
/// `wgsl/gn_part.wgsl`
pub const GN_PART: &str = include_str!("../wgsl/gn_part.wgsl");
/// `wgsl/gn_stats.wgsl`
pub const GN_STATS: &str = include_str!("../wgsl/gn_stats.wgsl");
/// `wgsl/gn_stats2.wgsl`
pub const GN_STATS2: &str = include_str!("../wgsl/gn_stats2.wgsl");
/// `wgsl/gqa_apply.wgsl`
pub const GQA_APPLY: &str = include_str!("../wgsl/gqa_apply.wgsl");
/// `wgsl/gqa_bwd_dk.wgsl`
pub const GQA_BWD_DK: &str = include_str!("../wgsl/gqa_bwd_dk.wgsl");
/// `wgsl/gqa_bwd_dq.wgsl`
pub const GQA_BWD_DQ: &str = include_str!("../wgsl/gqa_bwd_dq.wgsl");
/// `wgsl/gqa_bwd_dscores.wgsl`
pub const GQA_BWD_DSCORES: &str = include_str!("../wgsl/gqa_bwd_dscores.wgsl");
/// `wgsl/gqa_bwd_dv.wgsl`
pub const GQA_BWD_DV: &str = include_str!("../wgsl/gqa_bwd_dv.wgsl");
/// `wgsl/gqa_scores.wgsl`
pub const GQA_SCORES: &str = include_str!("../wgsl/gqa_scores.wgsl");
/// `wgsl/grad_scale.wgsl`
pub const GRAD_SCALE: &str = include_str!("../wgsl/grad_scale.wgsl");
/// `wgsl/grad_scale_buf.wgsl`
pub const GRAD_SCALE_BUF: &str = include_str!("../wgsl/grad_scale_buf.wgsl");
/// `wgsl/gradnorm_sq.wgsl`
pub const GRADNORM_SQ: &str = include_str!("../wgsl/gradnorm_sq.wgsl");
/// `wgsl/im2col.wgsl`
pub const IM2COL: &str = include_str!("../wgsl/im2col.wgsl");
/// `wgsl/l2norm_scale.wgsl`
pub const L2NORM_SCALE: &str = include_str!("../wgsl/l2norm_scale.wgsl");
/// `wgsl/l2norm_scale_dg.wgsl`
pub const L2NORM_SCALE_DG: &str = include_str!("../wgsl/l2norm_scale_dg.wgsl");
/// `wgsl/l2norm_scale_dx.wgsl`
pub const L2NORM_SCALE_DX: &str = include_str!("../wgsl/l2norm_scale_dx.wgsl");
/// `wgsl/layernorm.wgsl`
pub const LAYERNORM: &str = include_str!("../wgsl/layernorm.wgsl");
/// `wgsl/layernorm_dbeta.wgsl`
pub const LAYERNORM_DBETA: &str = include_str!("../wgsl/layernorm_dbeta.wgsl");
/// `wgsl/layernorm_dgamma.wgsl`
pub const LAYERNORM_DGAMMA: &str = include_str!("../wgsl/layernorm_dgamma.wgsl");
/// `wgsl/layernorm_dx.wgsl`
pub const LAYERNORM_DX: &str = include_str!("../wgsl/layernorm_dx.wgsl");
/// `wgsl/leaky_relu.wgsl`
pub const LEAKY_RELU: &str = include_str!("../wgsl/leaky_relu.wgsl");
/// `wgsl/leaky_relu_bwd.wgsl`
pub const LEAKY_RELU_BWD: &str = include_str!("../wgsl/leaky_relu_bwd.wgsl");
/// `wgsl/ln_head.wgsl`
pub const LN_HEAD: &str = include_str!("../wgsl/ln_head.wgsl");
/// `wgsl/ln_head_dgb.wgsl`
pub const LN_HEAD_DGB: &str = include_str!("../wgsl/ln_head_dgb.wgsl");
/// `wgsl/ln_head_dx.wgsl`
pub const LN_HEAD_DX: &str = include_str!("../wgsl/ln_head_dx.wgsl");
/// `wgsl/ln_stats.wgsl`
pub const LN_STATS: &str = include_str!("../wgsl/ln_stats.wgsl");
/// `wgsl/masked_l1.wgsl`
pub const MASKED_L1: &str = include_str!("../wgsl/masked_l1.wgsl");
/// `wgsl/masked_l1_grad.wgsl`
pub const MASKED_L1_GRAD: &str = include_str!("../wgsl/masked_l1_grad.wgsl");
/// `wgsl/matmul.wgsl`
pub const MATMUL: &str = include_str!("../wgsl/matmul.wgsl");
/// `wgsl/matmul_dw.wgsl`
pub const MATMUL_DW: &str = include_str!("../wgsl/matmul_dw.wgsl");
/// `wgsl/matmul_dw_reg.wgsl`
pub const MATMUL_DW_REG: &str = include_str!("../wgsl/matmul_dw_reg.wgsl");
/// `wgsl/matmul_dx.wgsl`
pub const MATMUL_DX: &str = include_str!("../wgsl/matmul_dx.wgsl");
/// `wgsl/matmul_dx_reg.wgsl`
pub const MATMUL_DX_REG: &str = include_str!("../wgsl/matmul_dx_reg.wgsl");
/// `wgsl/matmul_i8.wgsl`
pub const MATMUL_I8: &str = include_str!("../wgsl/matmul_i8.wgsl");
/// `wgsl/matmul_reg.wgsl`
pub const MATMUL_REG: &str = include_str!("../wgsl/matmul_reg.wgsl");
/// `wgsl/matmul_reg2.wgsl`
pub const MATMUL_REG2: &str = include_str!("../wgsl/matmul_reg2.wgsl");
/// `wgsl/matmul_rows.wgsl`
pub const MATMUL_ROWS: &str = include_str!("../wgsl/matmul_rows.wgsl");
/// `wgsl/matmul_tile.wgsl`
pub const MATMUL_TILE: &str = include_str!("../wgsl/matmul_tile.wgsl");
/// `wgsl/matmul_tiled.wgsl`
pub const MATMUL_TILED: &str = include_str!("../wgsl/matmul_tiled.wgsl");
/// `wgsl/maxpool5.wgsl`
pub const MAXPOOL5: &str = include_str!("../wgsl/maxpool5.wgsl");
/// `wgsl/maxpool5_dx.wgsl`
pub const MAXPOOL5_DX: &str = include_str!("../wgsl/maxpool5_dx.wgsl");
/// `wgsl/mla_bwd_dk_pass.wgsl`
pub const MLA_BWD_DK_PASS: &str = include_str!("../wgsl/mla_bwd_dk_pass.wgsl");
/// `wgsl/mla_bwd_dk_rope.wgsl`
pub const MLA_BWD_DK_ROPE: &str = include_str!("../wgsl/mla_bwd_dk_rope.wgsl");
/// `wgsl/mla_bwd_dq_pass.wgsl`
pub const MLA_BWD_DQ_PASS: &str = include_str!("../wgsl/mla_bwd_dq_pass.wgsl");
/// `wgsl/mla_bwd_dq_rope.wgsl`
pub const MLA_BWD_DQ_ROPE: &str = include_str!("../wgsl/mla_bwd_dq_rope.wgsl");
/// `wgsl/mla_index_scores.wgsl`
pub const MLA_INDEX_SCORES: &str = include_str!("../wgsl/mla_index_scores.wgsl");
/// `wgsl/mla_scores.wgsl`
pub const MLA_SCORES: &str = include_str!("../wgsl/mla_scores.wgsl");
/// `wgsl/mse_grad.wgsl`
pub const MSE_GRAD: &str = include_str!("../wgsl/mse_grad.wgsl");
/// `wgsl/mse_grad_w.wgsl`
pub const MSE_GRAD_W: &str = include_str!("../wgsl/mse_grad_w.wgsl");
/// `wgsl/mse_value.wgsl`
pub const MSE_VALUE: &str = include_str!("../wgsl/mse_value.wgsl");
/// `wgsl/mse_value_w.wgsl`
pub const MSE_VALUE_W: &str = include_str!("../wgsl/mse_value_w.wgsl");
/// `wgsl/mul.wgsl`
pub const MUL: &str = include_str!("../wgsl/mul.wgsl");
/// `wgsl/nchw_nlc.wgsl`
pub const NCHW_NLC: &str = include_str!("../wgsl/nchw_nlc.wgsl");
/// `wgsl/nlc_nchw.wgsl`
pub const NLC_NCHW: &str = include_str!("../wgsl/nlc_nchw.wgsl");
/// `wgsl/pack_qkv.wgsl`
pub const PACK_QKV: &str = include_str!("../wgsl/pack_qkv.wgsl");
/// `wgsl/pad2d.wgsl`
pub const PAD2D: &str = include_str!("../wgsl/pad2d.wgsl");
/// `wgsl/pixel_shuffle.wgsl`
pub const PIXEL_SHUFFLE: &str = include_str!("../wgsl/pixel_shuffle.wgsl");
/// `wgsl/pixel_shuffle_dx.wgsl`
pub const PIXEL_SHUFFLE_DX: &str = include_str!("../wgsl/pixel_shuffle_dx.wgsl");
/// `wgsl/pos_add.wgsl`
pub const POS_ADD: &str = include_str!("../wgsl/pos_add.wgsl");
/// `wgsl/pos_bwd.wgsl`
pub const POS_BWD: &str = include_str!("../wgsl/pos_bwd.wgsl");
/// `wgsl/region_copy.wgsl`
pub const REGION_COPY: &str = include_str!("../wgsl/region_copy.wgsl");
/// `wgsl/relu_inplace.wgsl`
pub const RELU_INPLACE: &str = include_str!("../wgsl/relu_inplace.wgsl");
/// `wgsl/resize_bilinear.wgsl`
pub const RESIZE_BILINEAR: &str = include_str!("../wgsl/resize_bilinear.wgsl");
/// `wgsl/resize_bilinear_dx.wgsl`
pub const RESIZE_BILINEAR_DX: &str = include_str!("../wgsl/resize_bilinear_dx.wgsl");
/// `wgsl/resize_nearest.wgsl`
pub const RESIZE_NEAREST: &str = include_str!("../wgsl/resize_nearest.wgsl");
/// `wgsl/resize_nearest_dx.wgsl`
pub const RESIZE_NEAREST_DX: &str = include_str!("../wgsl/resize_nearest_dx.wgsl");
/// `wgsl/rms_inv.wgsl`
pub const RMS_INV: &str = include_str!("../wgsl/rms_inv.wgsl");
/// `wgsl/rmsnorm.wgsl`
pub const RMSNORM: &str = include_str!("../wgsl/rmsnorm.wgsl");
/// `wgsl/rmsnorm_dw.wgsl`
pub const RMSNORM_DW: &str = include_str!("../wgsl/rmsnorm_dw.wgsl");
/// `wgsl/rmsnorm_dx.wgsl`
pub const RMSNORM_DX: &str = include_str!("../wgsl/rmsnorm_dx.wgsl");
/// `wgsl/rmsnorm_eps.wgsl`
pub const RMSNORM_EPS: &str = include_str!("../wgsl/rmsnorm_eps.wgsl");
/// `wgsl/rope.wgsl`
pub const ROPE: &str = include_str!("../wgsl/rope.wgsl");
/// `wgsl/rope2d.wgsl`
pub const ROPE2D: &str = include_str!("../wgsl/rope2d.wgsl");
/// `wgsl/rope_base.wgsl`
pub const ROPE_BASE: &str = include_str!("../wgsl/rope_base.wgsl");
/// `wgsl/rope_base_bwd.wgsl`
pub const ROPE_BASE_BWD: &str = include_str!("../wgsl/rope_base_bwd.wgsl");
/// `wgsl/rope_interleave_table.wgsl`
pub const ROPE_INTERLEAVE_TABLE: &str = include_str!("../wgsl/rope_interleave_table.wgsl");
/// `wgsl/rope_neox.wgsl`
pub const ROPE_NEOX: &str = include_str!("../wgsl/rope_neox.wgsl");
/// `wgsl/rope_sub.wgsl`
pub const ROPE_SUB: &str = include_str!("../wgsl/rope_sub.wgsl");
/// `wgsl/rope_train.wgsl`
pub const ROPE_TRAIN: &str = include_str!("../wgsl/rope_train.wgsl");
/// `wgsl/rope_train_bwd.wgsl`
pub const ROPE_TRAIN_BWD: &str = include_str!("../wgsl/rope_train_bwd.wgsl");
/// `wgsl/router_bwd.wgsl`
pub const ROUTER_BWD: &str = include_str!("../wgsl/router_bwd.wgsl");
/// `wgsl/router_bwd_sigmoid.wgsl`
pub const ROUTER_BWD_SIGMOID: &str = include_str!("../wgsl/router_bwd_sigmoid.wgsl");
/// `wgsl/router_gate.wgsl`
pub const ROUTER_GATE: &str = include_str!("../wgsl/router_gate.wgsl");
/// `wgsl/router_gate_sigmoid.wgsl`
pub const ROUTER_GATE_SIGMOID: &str = include_str!("../wgsl/router_gate_sigmoid.wgsl");
/// `wgsl/router_gate_train.wgsl`
pub const ROUTER_GATE_TRAIN: &str = include_str!("../wgsl/router_gate_train.wgsl");
/// `wgsl/scale_add.wgsl`
pub const SCALE_ADD: &str = include_str!("../wgsl/scale_add.wgsl");
/// `wgsl/scale_add_dexp.wgsl`
pub const SCALE_ADD_DEXP: &str = include_str!("../wgsl/scale_add_dexp.wgsl");
/// `wgsl/scale_add_dgate.wgsl`
pub const SCALE_ADD_DGATE: &str = include_str!("../wgsl/scale_add_dgate.wgsl");
/// `wgsl/scale_chan.wgsl`
pub const SCALE_CHAN: &str = include_str!("../wgsl/scale_chan.wgsl");
/// `wgsl/scale_chan_dg.wgsl`
pub const SCALE_CHAN_DG: &str = include_str!("../wgsl/scale_chan_dg.wgsl");
/// `wgsl/scale_row.wgsl`
pub const SCALE_ROW: &str = include_str!("../wgsl/scale_row.wgsl");
/// `wgsl/scan_add.wgsl`
pub const SCAN_ADD: &str = include_str!("../wgsl/scan_add.wgsl");
/// `wgsl/scan_block.wgsl`
pub const SCAN_BLOCK: &str = include_str!("../wgsl/scan_block.wgsl");
/// `wgsl/sigmoid.wgsl`
pub const SIGMOID: &str = include_str!("../wgsl/sigmoid.wgsl");
/// `wgsl/sigmoid_bwd.wgsl`
pub const SIGMOID_BWD: &str = include_str!("../wgsl/sigmoid_bwd.wgsl");
/// `wgsl/silu.wgsl`
pub const SILU: &str = include_str!("../wgsl/silu.wgsl");
/// `wgsl/silu_bwd.wgsl`
pub const SILU_BWD: &str = include_str!("../wgsl/silu_bwd.wgsl");
/// `wgsl/silu_bwd_da.wgsl`
pub const SILU_BWD_DA: &str = include_str!("../wgsl/silu_bwd_da.wgsl");
/// `wgsl/silu_bwd_db.wgsl`
pub const SILU_BWD_DB: &str = include_str!("../wgsl/silu_bwd_db.wgsl");
/// `wgsl/silu_gate.wgsl`
pub const SILU_GATE: &str = include_str!("../wgsl/silu_gate.wgsl");
/// `wgsl/silu_mul.wgsl`
pub const SILU_MUL: &str = include_str!("../wgsl/silu_mul.wgsl");
/// `wgsl/snake_beta.wgsl`
pub const SNAKE_BETA: &str = include_str!("../wgsl/snake_beta.wgsl");
/// `wgsl/softmax_k.wgsl`
pub const SOFTMAX_K: &str = include_str!("../wgsl/softmax_k.wgsl");
/// `wgsl/softmax_k_dx.wgsl`
pub const SOFTMAX_K_DX: &str = include_str!("../wgsl/softmax_k_dx.wgsl");
/// `wgsl/sort_hist.wgsl`
pub const SORT_HIST: &str = include_str!("../wgsl/sort_hist.wgsl");
/// `wgsl/sort_scatter.wgsl`
pub const SORT_SCATTER: &str = include_str!("../wgsl/sort_scatter.wgsl");
/// `wgsl/splat_bwd_count.wgsl`
pub const SPLAT_BWD_COUNT: &str = include_str!("../wgsl/splat_bwd_count.wgsl");
/// `wgsl/splat_bwd_emit.wgsl`
pub const SPLAT_BWD_EMIT: &str = include_str!("../wgsl/splat_bwd_emit.wgsl");
/// `wgsl/splat_bwd_keys.wgsl`
pub const SPLAT_BWD_KEYS: &str = include_str!("../wgsl/splat_bwd_keys.wgsl");
/// `wgsl/splat_emit.wgsl`
pub const SPLAT_EMIT: &str = include_str!("../wgsl/splat_emit.wgsl");
/// `wgsl/splat_grad_reduce.wgsl`
pub const SPLAT_GRAD_REDUCE: &str = include_str!("../wgsl/splat_grad_reduce.wgsl");
/// `wgsl/splat_naive.wgsl`
pub const SPLAT_NAIVE: &str = include_str!("../wgsl/splat_naive.wgsl");
/// `wgsl/splat_pack_rgba8.wgsl`
pub const SPLAT_PACK_RGBA8: &str = include_str!("../wgsl/splat_pack_rgba8.wgsl");
/// `wgsl/splat_project.wgsl`
pub const SPLAT_PROJECT: &str = include_str!("../wgsl/splat_project.wgsl");
/// `wgsl/splat_project_bwd.wgsl`
pub const SPLAT_PROJECT_BWD: &str = include_str!("../wgsl/splat_project_bwd.wgsl");
/// `wgsl/splat_rasterize.wgsl`
pub const SPLAT_RASTERIZE: &str = include_str!("../wgsl/splat_rasterize.wgsl");
/// `wgsl/splat_tile_count.wgsl`
pub const SPLAT_TILE_COUNT: &str = include_str!("../wgsl/splat_tile_count.wgsl");
/// `wgsl/splat_tile_ranges.wgsl`
pub const SPLAT_TILE_RANGES: &str = include_str!("../wgsl/splat_tile_ranges.wgsl");
/// `wgsl/splat_unpack.wgsl`
pub const SPLAT_UNPACK: &str = include_str!("../wgsl/splat_unpack.wgsl");
/// `wgsl/tanh_act.wgsl`
pub const TANH_ACT: &str = include_str!("../wgsl/tanh_act.wgsl");
/// `wgsl/tanh_act_bwd.wgsl`
pub const TANH_ACT_BWD: &str = include_str!("../wgsl/tanh_act_bwd.wgsl");
/// `wgsl/topk_mask.wgsl`
pub const TOPK_MASK: &str = include_str!("../wgsl/topk_mask.wgsl");
/// `wgsl/upsample2.wgsl`
pub const UPSAMPLE2: &str = include_str!("../wgsl/upsample2.wgsl");
/// `wgsl/upsample2_dx.wgsl`
pub const UPSAMPLE2_DX: &str = include_str!("../wgsl/upsample2_dx.wgsl");
/// `wgsl/vq_argmax_dot.wgsl`
pub const VQ_ARGMAX_DOT: &str = include_str!("../wgsl/vq_argmax_dot.wgsl");
/// `wgsl/vq_argmin.wgsl`
pub const VQ_ARGMIN: &str = include_str!("../wgsl/vq_argmin.wgsl");
/// `wgsl/weighted_gap.wgsl`
pub const WEIGHTED_GAP: &str = include_str!("../wgsl/weighted_gap.wgsl");
/// `wgsl/weighted_gap_dm.wgsl`
pub const WEIGHTED_GAP_DM: &str = include_str!("../wgsl/weighted_gap_dm.wgsl");
/// `wgsl/weighted_gap_dx.wgsl`
pub const WEIGHTED_GAP_DX: &str = include_str!("../wgsl/weighted_gap_dx.wgsl");

/// Every kernel as `(name, source)`, name = file stem.
pub const ALL: &[(&str, &str)] = &[
    ("adamw", ADAMW),
    ("add", ADD),
    ("add2", ADD2),
    ("add_chan_bcast", ADD_CHAN_BCAST),
    ("add_chan_bcast_dv", ADD_CHAN_BCAST_DV),
    ("add_chan_inplace", ADD_CHAN_INPLACE),
    ("add_index_mask", ADD_INDEX_MASK),
    ("add_inplace", ADD_INPLACE),
    ("attention", ATTENTION),
    ("attn_apply", ATTN_APPLY),
    ("attn_apply_bidir", ATTN_APPLY_BIDIR),
    ("attn_apply_cross", ATTN_APPLY_CROSS),
    ("attn_apply_full", ATTN_APPLY_FULL),
    ("attn_bwd_dbias", ATTN_BWD_DBIAS),
    ("attn_bwd_dk", ATTN_BWD_DK),
    ("attn_bwd_dk_bias", ATTN_BWD_DK_BIAS),
    ("attn_bwd_dk_bidir", ATTN_BWD_DK_BIDIR),
    ("attn_bwd_dk_cross", ATTN_BWD_DK_CROSS),
    ("attn_bwd_dq", ATTN_BWD_DQ),
    ("attn_bwd_dq_bias", ATTN_BWD_DQ_BIAS),
    ("attn_bwd_dq_bidir", ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dq_cross", ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dscores", ATTN_BWD_DSCORES),
    ("attn_bwd_dscores_bidir", ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dscores_cross", ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dv", ATTN_BWD_DV),
    ("attn_bwd_dv_bidir", ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dv_cross", ATTN_BWD_DV_CROSS),
    ("attn_scores", ATTN_SCORES),
    ("attn_scores_bidir", ATTN_SCORES_BIDIR),
    ("attn_scores_bidir_bias", ATTN_SCORES_BIDIR_BIAS),
    ("attn_scores_causal_bias", ATTN_SCORES_CAUSAL_BIAS),
    ("attn_scores_cross", ATTN_SCORES_CROSS),
    ("attn_scores_full", ATTN_SCORES_FULL),
    ("attn_scores_masked", ATTN_SCORES_MASKED),
    ("attn_scores_qk", ATTN_SCORES_QK),
    ("attn_softmax", ATTN_SOFTMAX),
    ("attn_softmax_bidir", ATTN_SOFTMAX_BIDIR),
    ("attn_softmax_cross", ATTN_SOFTMAX_CROSS),
    ("attn_softmax_full", ATTN_SOFTMAX_FULL),
    ("attn_softmax_masked", ATTN_SOFTMAX_MASKED),
    ("avgpool2d", AVGPOOL2D),
    ("avgpool2d_dx", AVGPOOL2D_DX),
    ("axpy", AXPY),
    ("bce_logits", BCE_LOGITS),
    ("bce_logits_grad", BCE_LOGITS_GRAD),
    ("bias_add", BIAS_ADD),
    ("bias_grad", BIAS_GRAD),
    ("bn_dbeta", BN_DBETA),
    ("bn_dgamma", BN_DGAMMA),
    ("bn_dstats", BN_DSTATS),
    ("bn_dx", BN_DX),
    ("bn_eval", BN_EVAL),
    ("bn_running", BN_RUNNING),
    ("bn_stats", BN_STATS),
    ("bn_train", BN_TRAIN),
    ("broadcast_add_hw", BROADCAST_ADD_HW),
    ("broadcast_add_hw_da", BROADCAST_ADD_HW_DA),
    ("bsq_quantize", BSQ_QUANTIZE),
    ("ce_grad", CE_GRAD),
    ("ce_grad_masked", CE_GRAD_MASKED),
    ("ce_grad_stats", CE_GRAD_STATS),
    ("ce_stats", CE_STATS),
    ("ce_value", CE_VALUE),
    ("ce_value_masked", CE_VALUE_MASKED),
    ("chan_place", CHAN_PLACE),
    ("ciou", CIOU),
    ("ciou_grad", CIOU_GRAD),
    ("clip_coef", CLIP_COEF),
    ("concat2", CONCAT2),
    ("concat_split", CONCAT_SPLIT),
    ("conv1d", CONV1D),
    ("conv1d_dw", CONV1D_DW),
    ("conv1d_dx", CONV1D_DX),
    ("conv2d", CONV2D),
    ("conv2d_dw", CONV2D_DW),
    ("conv2d_dx", CONV2D_DX),
    ("conv2d_gd", CONV2D_GD),
    ("conv2d_gd_dw", CONV2D_GD_DW),
    ("conv2d_gd_dx", CONV2D_GD_DX),
    ("conv2d_gd_reg", CONV2D_GD_REG),
    ("conv2d_tiled", CONV2D_TILED),
    ("conv_act", CONV_ACT),
    ("conv_act_reg", CONV_ACT_REG),
    ("conv_act_tiled", CONV_ACT_TILED),
    ("conv_bias", CONV_BIAS),
    ("conv_bias_reg", CONV_BIAS_REG),
    ("conv_epilogue", CONV_EPILOGUE),
    ("convex_upsample", CONVEX_UPSAMPLE),
    ("convex_upsample_dd", CONVEX_UPSAMPLE_DD),
    ("convex_upsample_dmask", CONVEX_UPSAMPLE_DMASK),
    ("convtr1d", CONVTR1D),
    ("convtr1d_dw", CONVTR1D_DW),
    ("convtr1d_dx", CONVTR1D_DX),
    ("crop2d", CROP2D),
    ("dfl_decode", DFL_DECODE),
    ("dfl_grad", DFL_GRAD),
    ("dfl_loss", DFL_LOSS),
    ("dfl_loss_grad", DFL_LOSS_GRAD),
    ("dwconv3d", DWCONV3D),
    ("dwconv3d_dw", DWCONV3D_DW),
    ("dwconv3d_dx", DWCONV3D_DX),
    ("edm_mix", EDM_MIX),
    ("edm_wrap", EDM_WRAP),
    ("emb_bwd", EMB_BWD),
    ("embed", EMBED),
    ("embed_tile", EMBED_TILE),
    ("expert_counts", EXPERT_COUNTS),
    ("film_chan", FILM_CHAN),
    ("film_chan_dsb", FILM_CHAN_DSB),
    ("film_chan_dx", FILM_CHAN_DX),
    ("film_row", FILM_ROW),
    ("film_row_dsb", FILM_ROW_DSB),
    ("film_row_dx", FILM_ROW_DX),
    ("gate_row", GATE_ROW),
    ("gate_row_dg", GATE_ROW_DG),
    ("gate_row_dh", GATE_ROW_DH),
    ("gelu", GELU),
    ("gelu_bwd", GELU_BWD),
    ("gelu_erf", GELU_ERF),
    ("gelu_erf_bwd", GELU_ERF_BWD),
    ("gn_apply", GN_APPLY),
    ("gn_dbeta", GN_DBETA),
    ("gn_dgamma", GN_DGAMMA),
    ("gn_dsum", GN_DSUM),
    ("gn_dx", GN_DX),
    ("gn_part", GN_PART),
    ("gn_stats", GN_STATS),
    ("gn_stats2", GN_STATS2),
    ("gqa_apply", GQA_APPLY),
    ("gqa_bwd_dk", GQA_BWD_DK),
    ("gqa_bwd_dq", GQA_BWD_DQ),
    ("gqa_bwd_dscores", GQA_BWD_DSCORES),
    ("gqa_bwd_dv", GQA_BWD_DV),
    ("gqa_scores", GQA_SCORES),
    ("grad_scale", GRAD_SCALE),
    ("grad_scale_buf", GRAD_SCALE_BUF),
    ("gradnorm_sq", GRADNORM_SQ),
    ("im2col", IM2COL),
    ("l2norm_scale", L2NORM_SCALE),
    ("l2norm_scale_dg", L2NORM_SCALE_DG),
    ("l2norm_scale_dx", L2NORM_SCALE_DX),
    ("layernorm", LAYERNORM),
    ("layernorm_dbeta", LAYERNORM_DBETA),
    ("layernorm_dgamma", LAYERNORM_DGAMMA),
    ("layernorm_dx", LAYERNORM_DX),
    ("leaky_relu", LEAKY_RELU),
    ("leaky_relu_bwd", LEAKY_RELU_BWD),
    ("ln_head", LN_HEAD),
    ("ln_head_dgb", LN_HEAD_DGB),
    ("ln_head_dx", LN_HEAD_DX),
    ("ln_stats", LN_STATS),
    ("masked_l1", MASKED_L1),
    ("masked_l1_grad", MASKED_L1_GRAD),
    ("matmul", MATMUL),
    ("matmul_dw", MATMUL_DW),
    ("matmul_dw_reg", MATMUL_DW_REG),
    ("matmul_dx", MATMUL_DX),
    ("matmul_dx_reg", MATMUL_DX_REG),
    ("matmul_i8", MATMUL_I8),
    ("matmul_reg", MATMUL_REG),
    ("matmul_reg2", MATMUL_REG2),
    ("matmul_rows", MATMUL_ROWS),
    ("matmul_tile", MATMUL_TILE),
    ("matmul_tiled", MATMUL_TILED),
    ("maxpool5", MAXPOOL5),
    ("maxpool5_dx", MAXPOOL5_DX),
    ("mla_bwd_dk_pass", MLA_BWD_DK_PASS),
    ("mla_bwd_dk_rope", MLA_BWD_DK_ROPE),
    ("mla_bwd_dq_pass", MLA_BWD_DQ_PASS),
    ("mla_bwd_dq_rope", MLA_BWD_DQ_ROPE),
    ("mla_index_scores", MLA_INDEX_SCORES),
    ("mla_scores", MLA_SCORES),
    ("mse_grad", MSE_GRAD),
    ("mse_grad_w", MSE_GRAD_W),
    ("mse_value", MSE_VALUE),
    ("mse_value_w", MSE_VALUE_W),
    ("mul", MUL),
    ("nchw_nlc", NCHW_NLC),
    ("nlc_nchw", NLC_NCHW),
    ("pack_qkv", PACK_QKV),
    ("pad2d", PAD2D),
    ("pixel_shuffle", PIXEL_SHUFFLE),
    ("pixel_shuffle_dx", PIXEL_SHUFFLE_DX),
    ("pos_add", POS_ADD),
    ("pos_bwd", POS_BWD),
    ("region_copy", REGION_COPY),
    ("relu_inplace", RELU_INPLACE),
    ("resize_bilinear", RESIZE_BILINEAR),
    ("resize_bilinear_dx", RESIZE_BILINEAR_DX),
    ("resize_nearest", RESIZE_NEAREST),
    ("resize_nearest_dx", RESIZE_NEAREST_DX),
    ("rms_inv", RMS_INV),
    ("rmsnorm", RMSNORM),
    ("rmsnorm_dw", RMSNORM_DW),
    ("rmsnorm_dx", RMSNORM_DX),
    ("rmsnorm_eps", RMSNORM_EPS),
    ("rope", ROPE),
    ("rope2d", ROPE2D),
    ("rope_base", ROPE_BASE),
    ("rope_base_bwd", ROPE_BASE_BWD),
    ("rope_interleave_table", ROPE_INTERLEAVE_TABLE),
    ("rope_neox", ROPE_NEOX),
    ("rope_sub", ROPE_SUB),
    ("rope_train", ROPE_TRAIN),
    ("rope_train_bwd", ROPE_TRAIN_BWD),
    ("router_bwd", ROUTER_BWD),
    ("router_bwd_sigmoid", ROUTER_BWD_SIGMOID),
    ("router_gate", ROUTER_GATE),
    ("router_gate_sigmoid", ROUTER_GATE_SIGMOID),
    ("router_gate_train", ROUTER_GATE_TRAIN),
    ("scale_add", SCALE_ADD),
    ("scale_add_dexp", SCALE_ADD_DEXP),
    ("scale_add_dgate", SCALE_ADD_DGATE),
    ("scale_chan", SCALE_CHAN),
    ("scale_chan_dg", SCALE_CHAN_DG),
    ("scale_row", SCALE_ROW),
    ("scan_add", SCAN_ADD),
    ("scan_block", SCAN_BLOCK),
    ("sigmoid", SIGMOID),
    ("sigmoid_bwd", SIGMOID_BWD),
    ("silu", SILU),
    ("silu_bwd", SILU_BWD),
    ("silu_bwd_da", SILU_BWD_DA),
    ("silu_bwd_db", SILU_BWD_DB),
    ("silu_gate", SILU_GATE),
    ("silu_mul", SILU_MUL),
    ("snake_beta", SNAKE_BETA),
    ("softmax_k", SOFTMAX_K),
    ("softmax_k_dx", SOFTMAX_K_DX),
    ("sort_hist", SORT_HIST),
    ("sort_scatter", SORT_SCATTER),
    ("splat_bwd_count", SPLAT_BWD_COUNT),
    ("splat_bwd_emit", SPLAT_BWD_EMIT),
    ("splat_bwd_keys", SPLAT_BWD_KEYS),
    ("splat_emit", SPLAT_EMIT),
    ("splat_grad_reduce", SPLAT_GRAD_REDUCE),
    ("splat_naive", SPLAT_NAIVE),
    ("splat_pack_rgba8", SPLAT_PACK_RGBA8),
    ("splat_project", SPLAT_PROJECT),
    ("splat_project_bwd", SPLAT_PROJECT_BWD),
    ("splat_rasterize", SPLAT_RASTERIZE),
    ("splat_tile_count", SPLAT_TILE_COUNT),
    ("splat_tile_ranges", SPLAT_TILE_RANGES),
    ("splat_unpack", SPLAT_UNPACK),
    ("tanh_act", TANH_ACT),
    ("tanh_act_bwd", TANH_ACT_BWD),
    ("topk_mask", TOPK_MASK),
    ("upsample2", UPSAMPLE2),
    ("upsample2_dx", UPSAMPLE2_DX),
    ("vq_argmax_dot", VQ_ARGMAX_DOT),
    ("vq_argmin", VQ_ARGMIN),
    ("weighted_gap", WEIGHTED_GAP),
    ("weighted_gap_dm", WEIGHTED_GAP_DM),
    ("weighted_gap_dx", WEIGHTED_GAP_DX),
];

/// Look up a kernel's WGSL source by file stem (e.g. `"matmul"`).
///
/// # Panics
/// Panics if `name` is not a known kernel — callers reference kernels by
/// compile-time-known names, so an unknown name is a programmer error.
pub fn src(name: &str) -> &'static str {
    ALL.iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| *s)
        .unwrap_or_else(|| panic!("unknown WGSL kernel: {name:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn src_roundtrips() {
        for (n, s) in ALL { assert_eq!(src(n), *s); }
    }
}
