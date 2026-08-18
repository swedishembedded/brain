// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The LTX-2.5 NA ("diffusion") video VAE decoder - `NADiffusionDecoder` /
//! `ltx_core.model.video_vae.diffusion_video_decoder.DiffusionVideoDecoder`,
//! the convolution-free sibling of [`crate::vae3d`]'s conv decoder. Ported
//! from `scratchpad/reference/ltxv/packages/ltx-core/src/ltx_core/model/
//! video_vae/{diffusion_video_decoder.py,transformer/*}`, cross-checked
//! against the real `ltx-2.5-video-vae-bf16.safetensors` header/metadata
//! (NOT the `-conv-` file [`crate::vae3d`] uses - a single file that carries
//! BOTH a conv `encoder.*` and a `decoder.*` tree that is THIS module's
//! architecture: `decoder.det_stages.*`/`decoder.diff_blocks.*`, not
//! `decoder.up_blocks.*`).
//!
//! ## Two structurally different pieces, staged in that order
//!
//! **Stages 1-4** (`forward_context`): a plain pre-norm transformer -
//! `x + NA(RMSNorm(x))`, then `x + SwiGLU(RMSNorm(x))` - at each stage's own
//! channel width and NA kernel size, four times, each followed by a
//! `LinearPixelShuffleUpsample`. No timestep, no diffusion - this is
//! deterministic feature upsampling into a "context" volume.
//!
//! **Stage 5** (`forward_diff`): 8 `CombinedDiffusionNABlock`s that denoise
//! patchified noised pixels `x_t`, guided by the (fixed, never mutated)
//! stage-4 context via AdaLN-Zero scale/shift injected FRESH every block
//! (context_proj) - **ungated** residuals (`x + attn(...)`, `x +
//! swiglu(...)`, no `*gate` multiply anywhere). The real checkpoint's
//! `default_num_inference_steps=1` + `model_output_type="x0"` collapse the
//! usual multi-step Euler sampling loop to exactly ONE forward at `t=1.0`
//! (pure noise) whose output IS the final pixel prediction directly - no
//! Euler integration step is needed at all for this checkpoint (see
//! `forward_diff`'s doc for the exact reasoning, checked against the real
//! metadata, not assumed from the general architecture description).
//!
//! ## The "gate" chunks: confirmed dead at the FORWARD level, not folded here
//!
//! `AdaLNZero.NUM_CHUNKS = 7` (`scale_msa, shift_msa, gate_msa, scale_mlp,
//! shift_mlp, gate_mlp, gate_ctx`) and `scale_shift_table` is `[7, dim]` in
//! the real checkpoint - but `DiffusionNABlock._modulation` (`blocks.py`)
//! only reads chunks 0,1,3,4 (`scale_msa, shift_msa, scale_mlp, shift_mlp`);
//! chunks 2, 5, 6 are computed (the list comprehension runs over all 7) and
//! then discarded via `_` unpacking. `combined/attn.py`'s `full` and
//! `combined/mlp.py`'s `residual_mlp` both add the raw block output straight
//! to the residual (`x + attn.proj(out)`, `x + swiglu_tiled(...)`) - there is
//! no gate multiply ANYWHERE in `CombinedDiffusionNABlock`'s actual forward
//! path. The class docstring's "legacy static gates are folded into Linear
//! weights at load time" describes upstream checkpoint HISTORY (an older
//! gated formulation whose gate scalars were pre-multiplied into
//! `attn.proj`/`mlp.w_down`/`context_proj` before this checkpoint was
//! produced) - it is NOT a transformation this importer performs: the real
//! checkpoint's `attn.proj.weight` etc. already reflect it, and the 7-row
//! `scale_shift_table` is read/imported as-is (7 rows in, 4 used, exactly
//! mirroring `_modulation`). No "fold gate into linear" step exists in this
//! file because none is needed - verified by reading `_modulation` and both
//! residual functions directly, not inferred from the docstring alone.
//!
//! ## RoPE: interleaved pairs, not the main DiT's split/rotate-half
//!
//! `rope_math.rot_abs_axis_impl` pairs ADJACENT channels `(2j, 2j+1)` per
//! axis-chunk (`crates/kernels/wgsl/rope_interleave_table.wgsl`'s own
//! convention, unlike `crate::rope`'s split-style construction for the main
//! DiT) - so this module reuses `rope_interleave_table` directly, no
//! permutation bridging needed. `head_dim=64` splits `(16, 24, 24)` across
//! (T, H, W) via `default_rope_dim_split` (below), each axis independently
//! absolute-RoPE'd (`base=10000`) then concatenated along the channel axis -
//! genuinely simpler than `crate::rope::ltx_rope_tables`'s band-major/
//! axis-minor/front-padded construction (a DIFFERENT upstream module,
//! `ltx_core.model.transformer.rope`, not this one). The rotation is
//! IDENTICAL across every head (broadcast, unlike the main DiT's per-head
//! different sub-table), so one shared `[nq, head_dim/2]` table serves every
//! head in a single `rope_interleave_table` dispatch.
//!
//! ## NATTEN windowing: border SHIFTS inward, never masks
//!
//! `fallback_na/eager.py`'s `_window_bounds` (this port's correctness
//! oracle - the reference's own CPU/no-natten-library fallback, so it needs
//! no compiled dependency to read as ground truth) computes, per query index
//! `i` on one axis of length `L` with kernel `K`:
//! ```text
//! half = K / 2          (integer division)
//! lo   = L - K
//! start = clamp(i - half, 0, lo)
//! ```
//! and the window is ALWAYS the full `K` positions `[start, start+K)` - never
//! shrunk, never masked. `crates/kernels/wgsl/na3d_scores.wgsl`/
//! `na3d_apply.wgsl` implement exactly this per-axis formula in all three
//! axes independently (this repo's own kernel-checklist worked example of
//! this exact bug class: an off-by-one here degrades quality without
//! crashing). [`na_decoder::tests::window_bounds_matches_the_eager_oracle`]
//! pins the formula against hand-computed tables at several `(length,
//! kernel, query)` triples, independent of any real-weight run.
//!
//! ## Existing kernels checked and refuted before writing
//! ## `na3d_scores`/`na3d_apply`
//!
//! `gqa_scores_win`/`attn_decode_scores_win` are 1-D SEQUENCE-POSITION
//! windows (`key j lives iff i-window < j <= i`, a causal band along one
//! axis) - fundamentally different semantics from a 3-axis SPATIAL
//! neighborhood with a symmetric, non-causal, inward-shifting window on
//! every axis independently. Neither kernel's `Params` (nor any masked-
//! attention kernel in this repo) can express NATTEN's per-axis shift
//! without a full rewrite, so a genuinely new kernel pair was warranted -
//! `attn_softmax_cross` (a layout-generic `[rows, cols]` row softmax) is
//! reused UNCHANGED for the softmax step in between, since NATTEN's window
//! is always fully dense (no masking, see above), which is exactly what a
//! plain row softmax computes.
//!
//! ## Channel sub-order traps (both checked against source, not precedent)
//!
//! `LinearPixelShuffleUpsample`'s rearrange (`layers.py`:
//! `'b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c'`) uses the SAME
//! `(c, p1=T, p2=H, p3=W)` sub-order as `vae::blocks3d`'s internal
//! `space_to_depth`/`depth_to_space` (height before width) - genuinely
//! DIFFERENT from `crate::patchify`'s outer-pixel-boundary convention
//! (`ops.py`'s `(c p r q)`, width before height), which is a real, doc'd
//! trap from the M2 port (see `crate::patchify`'s own module doc). This
//! decoder needs BOTH: `crates/kernels/wgsl/pixel_shuffle3d_cl.wgsl`
//! (new - channels-last, height-before-width) for the four upsample stages,
//! and `crate::patchify::{patchify,unpatchify}` UNCHANGED (channel-first,
//! width-before-height) for the outer noised-pixel boundary, since
//! `diffusion_video_decoder.py` imports `patchify`/`unpatchify` from
//! `ltx_core.model.video_vae.ops` - the exact same functions
//! `crate::vae3d`'s conv decoder already uses, not a new convention.
//!
//! ## What this module does NOT implement
//!
//! Tiling (`diffusion_tiling.py`'s overlapping-tile trapezoidal blend),
//! multi-step Euler sampling (moot for this real checkpoint, see above), the
//! CHUNKED/BLACKWELL_DSL block variants (`w_chunks>1`, deferred stage-4
//! upsample) - this module always runs the COMBINED pathway
//! (`CombinedDiffusionNABlock`, full-volume attention, `w_chunks=1`), the
//! same pathway `DiffVAEMode.COMBINED_COMPILE` selects in production (minus
//! `torch.compile`, irrelevant to a Rust port). `forward_context`/
//! `forward_diff` re-upload their weights on every call rather than caching
//! a resident decoder object - a correctness-milestone simplification
//! (performance comes after parity, this whole port's own established
//! discipline), not a numerical one.

use checkpoint::safetensors::StTensor;
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use vae::blocks::Tensors;

use crate::import::validate_manifest;

// --------------------------------------------------------------- kernels

const K_MATMUL: usize = 0;
const K_BIAS_ADD: usize = 1;
const K_RMSNORM_EPS: usize = 2;
const K_SILU_MUL: usize = 3;
const K_MUL: usize = 4;
const K_ADD2: usize = 5;
const K_NA3D_SCORES: usize = 6;
const K_ATTN_SOFTMAX_CROSS: usize = 7;
const K_NA3D_APPLY: usize = 8;
const K_ROPE_INTERLEAVE_TABLE: usize = 9;
const K_PIXEL_SHUFFLE3D_CL: usize = 10;

/// Every kernel this module dispatches - all pre-existing except the three
/// documented in this module's header (`na3d_scores`/`na3d_apply`/
/// `pixel_shuffle3d_cl`).
pub const KERNELS: [(&str, &str); 11] = [
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("silu_mul", kernels::SILU_MUL),
    ("mul", kernels::MUL),
    ("add2", kernels::ADD2),
    ("na3d_scores", kernels::NA3D_SCORES),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("na3d_apply", kernels::NA3D_APPLY),
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
    ("pixel_shuffle3d_cl", kernels::PIXEL_SHUFFLE3D_CL),
];

/// Open a device for this module's kernel table. `None` takes brain's default.
pub fn open_device(device: Option<&str>) -> Gpu {
    match device {
        Some("cpu") => Gpu::new_cpu(&KERNELS),
        Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
        _ => Gpu::new(&KERNELS),
    }
}

// --------------------------------------------------------------- config

/// The real LTX-2.5 NA decoder's shape configuration - every field
/// transcribed from the real `ltx-2.5-video-vae-bf16.safetensors` header's
/// `config.vae.decoder` metadata (`_class_name: "NADiffusionDecoder"`), not
/// from the reference module's own (superseded-checkpoint) class defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct NaDecoderConfig {
    pub in_channels: u32,
    pub out_channels: u32,
    pub patch_size: u32,
    pub head_dim: u32,
    /// 5 entries: `stage_channels[0..4]` are the 4 deterministic stages'
    /// widths, `stage_channels[4]` is both the context width AND (since the
    /// real config has no `stage5_channels` override) stage 5's own width.
    pub stage_channels: [u32; 5],
    /// 5 entries: `stage_depths[0..4]` are the deterministic stages' block
    /// counts, `stage_depths[4]` is the diffusion stage's block count.
    pub stage_depths: [u32; 5],
    /// The 4 deterministic stages' own NA kernel sizes
    /// (`stage_kernels[0..4]` of the real config - `stage_kernels[4]`
    /// exists in the checkpoint metadata but is never read by the
    /// deterministic stages; the diffusion stage uses [`stage5_kernel`]
    /// instead, a SEPARATE field, per `diffusion_video_decoder.py`'s own
    /// constructor).
    pub stage_kernels: [[u32; 3]; 4],
    /// 4 entries: `(stride, out_channels_reduction_factor)` per upsample,
    /// between stage `i` and `i+1`.
    pub upsamples: [([u32; 3], u32); 4],
    pub stage5_kernel: [u32; 3],
    pub t_emb_dim: u32,
    pub timestep_scale_multiplier: f32,
    pub default_num_inference_steps: u32,
}

impl NaDecoderConfig {
    /// The real checkpoint's config, transcribed from
    /// `ltx-2.5-video-vae-bf16.safetensors`'s embedded
    /// `config.vae.decoder` metadata verbatim.
    pub fn ltx25() -> NaDecoderConfig {
        NaDecoderConfig {
            in_channels: 128,
            out_channels: 3,
            patch_size: 4,
            head_dim: 64,
            stage_channels: [2048, 1024, 512, 512, 256],
            stage_depths: [4, 6, 4, 2, 8],
            stage_kernels: [[3, 7, 7], [3, 7, 7], [3, 5, 5], [3, 5, 5]],
            upsamples: [([1, 2, 2], 2), ([2, 1, 1], 2), ([2, 2, 2], 1), ([2, 2, 2], 2)],
            stage5_kernel: [11, 11, 11],
            t_emb_dim: 384,
            timestep_scale_multiplier: 1000.0,
            default_num_inference_steps: 1,
        }
    }

    /// Stage-4 output / stage-5 context width AND (no `stage5_channels`
    /// override in the real config) stage 5's own working width.
    pub fn context_channels(&self) -> u32 {
        self.stage_channels[4]
    }

    /// `out_channels * patch_size^2` - the patchified noised-pixel channel
    /// count `conv_in_x_t`/`conv_out` operate at.
    pub fn noised_pixel_channels(&self) -> u32 {
        self.out_channels * self.patch_size * self.patch_size
    }

    /// SwiGLU hidden width at channel count `dim`: `(dim*4 + 15) // 16 * 16`
    /// (`mlp_ratio=4.0`, always already a multiple of 16 at this
    /// checkpoint's real widths, confirmed against every real
    /// `mlp.w_gate.weight` shape in the header).
    fn hidden(dim: u32) -> u32 {
        (dim * 4).div_ceil(16) * 16
    }

    /// Every tensor this decoder reads, checkpoint-side name + shape (the
    /// FUSED `attn.qkv.{weight,bias}` layout the real file actually ships,
    /// not the post-split `to_q/to_k/to_v` names [`import_na_decoder`]'s
    /// internal map uses - see that function's doc). `decoder.type_emb` is
    /// deliberately excluded: a real tensor in the checkpoint with zero
    /// references anywhere in the reference source tree (grepped, not
    /// assumed) - dead weight, same "noted, not implemented" treatment
    /// `crate::upsampler`'s dead `rational_resampler` config field got.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut m: Vec<(String, Vec<usize>)> = Vec::new();
        let c0 = self.stage_channels[0];
        let c5 = self.context_channels();
        let noised = self.noised_pixel_channels();

        m.push(("decoder.conv_in.weight".into(), vec![c0 as usize, self.in_channels as usize]));
        m.push(("decoder.conv_in.bias".into(), vec![c0 as usize]));
        m.push(("decoder.conv_in_x_t.weight".into(), vec![c5 as usize, noised as usize]));
        m.push(("decoder.conv_in_x_t.bias".into(), vec![c5 as usize]));
        m.push(("decoder.conv_out.weight".into(), vec![noised as usize, c5 as usize]));
        m.push(("decoder.conv_out.bias".into(), vec![noised as usize]));
        m.push(("decoder.norm_out.weight".into(), vec![c5 as usize]));
        m.push(("decoder.shared_adaln.proj.weight".into(), vec![7 * c5 as usize, self.t_emb_dim as usize]));
        m.push(("decoder.shared_adaln.proj.bias".into(), vec![7 * c5 as usize]));
        m.push(("decoder.t_embedder.mlp.0.weight".into(), vec![self.t_emb_dim as usize, 256]));
        m.push(("decoder.t_embedder.mlp.0.bias".into(), vec![self.t_emb_dim as usize]));
        m.push(("decoder.t_embedder.mlp.2.weight".into(), vec![self.t_emb_dim as usize, self.t_emb_dim as usize]));
        m.push(("decoder.t_embedder.mlp.2.bias".into(), vec![self.t_emb_dim as usize]));

        let attn_block = |m: &mut Vec<(String, Vec<usize>)>, prefix: &str, dim: u32| {
            m.push((format!("{prefix}.attn.qkv.weight"), vec![3 * dim as usize, dim as usize]));
            m.push((format!("{prefix}.attn.qkv.bias"), vec![3 * dim as usize]));
            m.push((format!("{prefix}.attn.proj.weight"), vec![dim as usize, dim as usize]));
            m.push((format!("{prefix}.attn.proj.bias"), vec![dim as usize]));
            m.push((format!("{prefix}.attn.q_norm.weight"), vec![self.head_dim as usize]));
            m.push((format!("{prefix}.attn.k_norm.weight"), vec![self.head_dim as usize]));
            let hidden = Self::hidden(dim) as usize;
            m.push((format!("{prefix}.mlp.w_gate.weight"), vec![hidden, dim as usize]));
            m.push((format!("{prefix}.mlp.w_up.weight"), vec![hidden, dim as usize]));
            m.push((format!("{prefix}.mlp.w_down.weight"), vec![dim as usize, hidden]));
            m.push((format!("{prefix}.norm1.weight"), vec![dim as usize]));
            m.push((format!("{prefix}.norm2.weight"), vec![dim as usize]));
        };

        for stage in 0..4usize {
            let dim = self.stage_channels[stage];
            for i in 0..self.stage_depths[stage] {
                attn_block(&mut m, &format!("decoder.det_stages.{stage}.{i}"), dim);
            }
            let (stride, reduction) = self.upsamples[stage];
            let proj_out = dim * stride[0] * stride[1] * stride[2] / reduction;
            m.push((format!("decoder.upsamples.{stage}.proj.weight"), vec![proj_out as usize, dim as usize]));
            m.push((format!("decoder.upsamples.{stage}.proj.bias"), vec![proj_out as usize]));
        }

        for i in 0..self.stage_depths[4] {
            let prefix = format!("decoder.diff_blocks.{i}");
            attn_block(&mut m, &prefix, c5);
            m.push((format!("{prefix}.context_proj.weight"), vec![c5 as usize, c5 as usize]));
            m.push((format!("{prefix}.context_proj.bias"), vec![c5 as usize]));
            m.push((format!("{prefix}.scale_shift_table"), vec![7, c5 as usize]));
        }

        m.push(("per_channel_statistics.mean-of-means".into(), vec![self.in_channels as usize]));
        m.push(("per_channel_statistics.std-of-means".into(), vec![self.in_channels as usize]));
        m
    }
}

// ---------------------------------------------------------------- import

/// Import the NA decoder from a real `ltx-2.5-video-vae-bf16.safetensors`
/// tensor list: filter to `decoder.*`/`per_channel_statistics.*` (silently
/// dropping `encoder.*` - out of scope, [`crate::vae3d`]'s own encoder
/// already covers that architecture from the OTHER checkpoint file) and
/// `decoder.type_emb` (dead, see [`NaDecoderConfig::tensor_manifest`]'s
/// doc), validate two-way coverage against the checkpoint-side manifest,
/// then split every fused `attn.qkv.{weight,bias}` into `to_q`/`to_k`/`to_v`
/// on the host (porting.md SS2: "split fused weights on the host at import
/// time... every device matmul then reads a whole buffer") - the RETURNED
/// map uses the post-split names, which is what every function below reads.
pub fn import_na_decoder(tensors: Vec<StTensor>, cfg: &NaDecoderConfig) -> Result<Tensors, String> {
    let map: Tensors = tensors
        .into_iter()
        .filter_map(|t| {
            let StTensor { name, shape, data } = t;
            if name == "decoder.type_emb" {
                return None;
            }
            if name.starts_with("decoder.") || name.starts_with("per_channel_statistics.") {
                Some((name, (shape, data)))
            } else {
                None
            }
        })
        .collect();
    let map = validate_manifest(map, &cfg.tensor_manifest(), "NA decoder")?;
    Ok(split_fused_qkv(map, cfg))
}

fn split_fused_qkv(mut map: Tensors, cfg: &NaDecoderConfig) -> Tensors {
    let mut prefixes: Vec<String> = Vec::new();
    for stage in 0..4usize {
        for i in 0..cfg.stage_depths[stage] {
            prefixes.push(format!("decoder.det_stages.{stage}.{i}"));
        }
    }
    for i in 0..cfg.stage_depths[4] {
        prefixes.push(format!("decoder.diff_blocks.{i}"));
    }
    for p in prefixes {
        let (wshape, wdata) = map.remove(&format!("{p}.attn.qkv.weight")).unwrap_or_else(|| panic!("na decoder import: missing {p}.attn.qkv.weight"));
        let (_bshape, bdata) = map.remove(&format!("{p}.attn.qkv.bias")).unwrap_or_else(|| panic!("na decoder import: missing {p}.attn.qkv.bias"));
        let dim = wshape[1];
        assert_eq!(wshape[0], 3 * dim, "{p}.attn.qkv.weight shape {wshape:?} is not [3*dim, dim]");
        for (i, part) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            let wslice = wdata[i * dim * dim..(i + 1) * dim * dim].to_vec();
            let bslice = bdata[i * dim..(i + 1) * dim].to_vec();
            map.insert(format!("{p}.attn.{part}.weight"), (vec![dim, dim], wslice));
            map.insert(format!("{p}.attn.{part}.bias"), (vec![dim], bslice));
        }
    }
    map
}

fn tget<'a>(w: &'a Tensors, name: &str) -> &'a [f32] {
    &w.get(name).unwrap_or_else(|| panic!("na_decoder: missing weight {name}")).1
}

// ------------------------------------------------------------------ RoPE

/// `ltx_core...transformer.rope_math.default_rope_dim_split` - split
/// `head_dim` across (T, H, W) RoPE chunks. At `head_dim=64` (this
/// checkpoint's only value): `(16, 24, 24)`.
fn default_rope_dim_split(head_dim: u32) -> (u32, u32, u32) {
    assert_eq!(head_dim % 8, 0, "default_rope_dim_split: head_dim {head_dim} must be a multiple of 8");
    let mut d_t = (head_dim / 4) / 2 * 2;
    let mut d_hw = (head_dim - d_t) / 2;
    if !d_hw.is_multiple_of(2) {
        d_t -= 2;
        d_hw = (head_dim - d_t) / 2;
    }
    (d_t, d_hw, d_hw)
}

/// `1 / base^(i/dim)` for `i` in `0, 2, .., dim-2` - `rope_math.rope_inv_freqs`.
fn rope_inv_freqs(dim: u32, base: f64) -> Vec<f32> {
    (0..dim).step_by(2).map(|i| (1.0 / base.powf(i as f64 / dim as f64)) as f32).collect()
}

/// Shared (head-independent) `[nq, head_dim/2]` interleaved-pair RoPE table,
/// query-flat-index-major (matching `na3d_scores.wgsl`'s own query decode:
/// `qi = (qt*h+qh)*w+qw`) - the per-axis absolute-RoPE construction
/// `rope_math.rot_abs_axis_impl`/`_apply_nested_full_volume_rope` compute,
/// with `num_tiles` collapsed to its mathematically-equivalent direct form
/// (the reference's own W-axis tiling is a Dynamo memory-chunking device,
/// not a semantic change - see `crate::rope`'s sibling module for the same
/// observation on the main DiT's construction).
struct NaRope {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half: u32,
}

fn na_rope_table(t: u32, h: u32, w: u32, head_dim: u32) -> NaRope {
    let (dt, dh, dw) = default_rope_dim_split(head_dim);
    let inv_t = rope_inv_freqs(dt, 10000.0);
    let inv_h = rope_inv_freqs(dh, 10000.0);
    let inv_w = rope_inv_freqs(dw, 10000.0);
    let (ft, fh, fw) = (inv_t.len(), inv_h.len(), inv_w.len());
    let half = ft + fh + fw;
    assert_eq!(half as u32 * 2, head_dim, "rope split {ft}+{fh}+{fw} does not fill head_dim/2={}", head_dim / 2);

    let nq = (t * h * w) as usize;
    let mut cos = vec![0f32; nq * half];
    let mut sin = vec![0f32; nq * half];
    for qt in 0..t {
        for qh in 0..h {
            for qw in 0..w {
                let qi = ((qt * h + qh) * w + qw) as usize;
                let base = qi * half;
                for (j, &iv) in inv_t.iter().enumerate() {
                    let ang = qt as f32 * iv;
                    cos[base + j] = ang.cos();
                    sin[base + j] = ang.sin();
                }
                for (j, &iv) in inv_h.iter().enumerate() {
                    let ang = qh as f32 * iv;
                    cos[base + ft + j] = ang.cos();
                    sin[base + ft + j] = ang.sin();
                }
                for (j, &iv) in inv_w.iter().enumerate() {
                    let ang = qw as f32 * iv;
                    cos[base + ft + fh + j] = ang.cos();
                    sin[base + ft + fh + j] = ang.sin();
                }
            }
        }
    }
    NaRope { cos, sin, half: half as u32 }
}

// -------------------------------------------------------------- GPU steps

fn upload(gpu: &Gpu, data: &[f32]) -> DeviceBuffer {
    let b = gpu.storage(data.len() as u64);
    gpu.write_f32(&b, data);
    b
}

fn linear_step(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, b: Option<&DeviceBuffer>, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
    s.push(gpu.step(K_MATMUL, &[x, w, out], &[m, k, n], m * n));
    if let Some(b) = b {
        s.push(gpu.step(K_BIAS_ADD, &[out, b], &[m, n], m * n));
    }
}

fn rmsnorm_step(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32, eps: f32) {
    s.push(gpu.step(K_RMSNORM_EPS, &[x, w, out], &[dim, rows, f(eps)], rows));
}

/// `out = x*(1+scale) + shift`, `scale`/`shift` already row-broadcast to
/// `x`'s own `[rows,dim]` shape by the caller (host-repeated - stage 5's
/// modulation is one vector per forward, not per token, see
/// [`forward_diff`]'s doc).
fn modulate(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, scale_b: &DeviceBuffer, shift_b: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let tmp = gpu.storage(n as u64);
    s.push(gpu.step(K_MUL, &[x, scale_b, &tmp], &[n], n));
    let out = gpu.storage(n as u64);
    s.push(gpu.step(K_ADD2, &[&tmp, shift_b, &out], &[n], n));
    out
}

fn broadcast_rows(v: &[f32], rows: u32) -> Vec<f32> {
    let mut out = vec![0f32; v.len() * rows as usize];
    for r in 0..rows as usize {
        out[r * v.len()..r * v.len() + v.len()].copy_from_slice(v);
    }
    out
}

fn broadcast_rows_plus1(v: &[f32], rows: u32) -> Vec<f32> {
    let v1: Vec<f32> = v.iter().map(|x| 1.0 + x).collect();
    broadcast_rows(&v1, rows)
}

/// One `NeighborhoodAttention3D`'s weights, uploaded. `scale = head_dim
/// **-0.5` is folded into the uploaded `q_norm` weight (RMSNorm-then-scale
/// is linear in a uniform per-vector scalar - the exact same fold this
/// port's Gemma-4 milestone used for its own `scaling` field), so no
/// separate device scale step is needed.
struct NaAttnWeights {
    wq: DeviceBuffer,
    bq: DeviceBuffer,
    wk: DeviceBuffer,
    bk: DeviceBuffer,
    wv: DeviceBuffer,
    bv: DeviceBuffer,
    wo: DeviceBuffer,
    bo: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
}

fn upload_attn(gpu: &Gpu, w: &Tensors, prefix: &str, head_dim: u32) -> NaAttnWeights {
    let scale = (head_dim as f32).powf(-0.5);
    let q_norm_scaled: Vec<f32> = tget(w, &format!("{prefix}.q_norm.weight")).iter().map(|v| v * scale).collect();
    NaAttnWeights {
        wq: upload(gpu, tget(w, &format!("{prefix}.to_q.weight"))),
        bq: upload(gpu, tget(w, &format!("{prefix}.to_q.bias"))),
        wk: upload(gpu, tget(w, &format!("{prefix}.to_k.weight"))),
        bk: upload(gpu, tget(w, &format!("{prefix}.to_k.bias"))),
        wv: upload(gpu, tget(w, &format!("{prefix}.to_v.weight"))),
        bv: upload(gpu, tget(w, &format!("{prefix}.to_v.bias"))),
        wo: upload(gpu, tget(w, &format!("{prefix}.proj.weight"))),
        bo: upload(gpu, tget(w, &format!("{prefix}.proj.bias"))),
        q_norm: upload(gpu, &q_norm_scaled),
        k_norm: upload(gpu, tget(w, &format!("{prefix}.k_norm.weight"))),
    }
}

/// One windowed self-attention call: QKV projections, per-head RMSNorm
/// (`q_norm`'s weight already carries the `scale` fold), shared
/// interleaved-pair RoPE (identical across heads), the windowed
/// scores/softmax/apply trio, output projection. `x` is `[nq, dim]`
/// channels-last (query-flat-index-major); returns the same shape.
#[allow(clippy::too_many_arguments)]
fn na_attention(
    gpu: &Gpu,
    s: &mut Vec<Step>,
    w: &NaAttnWeights,
    x: &DeviceBuffer,
    nq: u32,
    dim: u32,
    heads: u32,
    head_dim: u32,
    t: u32,
    h: u32,
    wd: u32,
    kt: u32,
    kh: u32,
    kw: u32,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    half: u32,
    eps: f32,
) -> DeviceBuffer {
    let dimtot = nq * dim;
    let q_pre = gpu.storage(dimtot as u64);
    let k_pre = gpu.storage(dimtot as u64);
    let v = gpu.storage(dimtot as u64);
    linear_step(gpu, s, x, &w.wq, Some(&w.bq), &q_pre, nq, dim, dim);
    linear_step(gpu, s, x, &w.wk, Some(&w.bk), &k_pre, nq, dim, dim);
    linear_step(gpu, s, x, &w.wv, Some(&w.bv), &v, nq, dim, dim);

    // Per-head RMSNorm: [nq, heads, head_dim] read as (nq*heads) rows of
    // width head_dim - the SAME contiguous buffer, just a different row count.
    let q = gpu.storage(dimtot as u64);
    let k = gpu.storage(dimtot as u64);
    rmsnorm_step(gpu, s, &q_pre, &w.q_norm, &q, head_dim, nq * heads, eps);
    rmsnorm_step(gpu, s, &k_pre, &w.k_norm, &k, head_dim, nq * heads, eps);

    let q_rot = gpu.storage(dimtot as u64);
    let k_rot = gpu.storage(dimtot as u64);
    s.push(gpu.step(K_ROPE_INTERLEAVE_TABLE, &[&q, cos, sin, &q_rot], &[nq, heads, head_dim, half], nq * heads * half));
    s.push(gpu.step(K_ROPE_INTERLEAVE_TABLE, &[&k, cos, sin, &k_rot], &[nq, heads, head_dim, half], nq * heads * half));

    let window = kt * kh * kw;
    let scores = gpu.storage((heads * nq * window) as u64);
    s.push(gpu.step(K_NA3D_SCORES, &[&q_rot, &k_rot, &scores], &[t, h, wd, heads, head_dim, kt, kh, kw], heads * nq * window));
    let probs = gpu.storage((heads * nq * window) as u64);
    s.push(gpu.step(K_ATTN_SOFTMAX_CROSS, &[&scores, &probs], &[1, heads, nq, window], heads * nq));
    let ctx = gpu.storage(dimtot as u64);
    s.push(gpu.step(K_NA3D_APPLY, &[&probs, &v, &ctx], &[t, h, wd, heads, head_dim, kt, kh, kw], nq * heads * head_dim));

    let out = gpu.storage(dimtot as u64);
    linear_step(gpu, s, &ctx, &w.wo, Some(&w.bo), &out, nq, dim, dim);
    out
}

/// `w_down(silu(w_gate(x)) * w_up(x))` - `SwiGLU`'s forward, no bias
/// anywhere (matches the real checkpoint: `mlp.w_{gate,up,down}` carry no
/// `.bias` tensor).
#[allow(clippy::too_many_arguments)]
fn swiglu(gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, w_gate: &DeviceBuffer, w_up: &DeviceBuffer, w_down: &DeviceBuffer, nq: u32, dim: u32, hidden: u32) -> DeviceBuffer {
    let gate = gpu.storage((nq * hidden) as u64);
    let up = gpu.storage((nq * hidden) as u64);
    linear_step(gpu, s, x, w_gate, None, &gate, nq, dim, hidden);
    linear_step(gpu, s, x, w_up, None, &up, nq, dim, hidden);
    let act = gpu.storage((nq * hidden) as u64);
    s.push(gpu.step(K_SILU_MUL, &[&gate, &up, &act], &[nq * hidden], nq * hidden));
    let down = gpu.storage((nq * dim) as u64);
    linear_step(gpu, s, &act, w_down, None, &down, nq, hidden, dim);
    down
}

// ---------------------------------------------------------- det NA block

struct DetBlockWeights {
    attn: NaAttnWeights,
    norm1: DeviceBuffer,
    norm2: DeviceBuffer,
    w_gate: DeviceBuffer,
    w_up: DeviceBuffer,
    w_down: DeviceBuffer,
    hidden: u32,
}

fn upload_det_block(gpu: &Gpu, w: &Tensors, prefix: &str, dim: u32, head_dim: u32) -> DetBlockWeights {
    DetBlockWeights {
        attn: upload_attn(gpu, w, &format!("{prefix}.attn"), head_dim),
        norm1: upload(gpu, tget(w, &format!("{prefix}.norm1.weight"))),
        norm2: upload(gpu, tget(w, &format!("{prefix}.norm2.weight"))),
        w_gate: upload(gpu, tget(w, &format!("{prefix}.mlp.w_gate.weight"))),
        w_up: upload(gpu, tget(w, &format!("{prefix}.mlp.w_up.weight"))),
        w_down: upload(gpu, tget(w, &format!("{prefix}.mlp.w_down.weight"))),
        hidden: NaDecoderConfig::hidden(dim),
    }
}

/// `NABlock.forward`: `x = x + attn(norm1(x)); x = x + swiglu(norm2(x))` -
/// plain pre-norm, no AdaLN modulation.
#[allow(clippy::too_many_arguments)]
fn det_block_forward(gpu: &Gpu, x: &[f32], w: &DetBlockWeights, nq: u32, dim: u32, heads: u32, head_dim: u32, t: u32, h: u32, wd: u32, kernel: [u32; 3], rope: &NaRope, eps: f32) -> Vec<f32> {
    let x_buf = upload(gpu, x);
    let cos = upload(gpu, &rope.cos);
    let sin = upload(gpu, &rope.sin);
    let dimtot = nq * dim;
    let mut s: Vec<Step> = Vec::new();

    let n1 = gpu.storage(dimtot as u64);
    rmsnorm_step(gpu, &mut s, &x_buf, &w.norm1, &n1, dim, nq, eps);
    let attn_out = na_attention(gpu, &mut s, &w.attn, &n1, nq, dim, heads, head_dim, t, h, wd, kernel[0], kernel[1], kernel[2], &cos, &sin, rope.half, eps);
    let x1 = gpu.storage(dimtot as u64);
    s.push(gpu.step(K_ADD2, &[&x_buf, &attn_out, &x1], &[dimtot], dimtot));

    let n2 = gpu.storage(dimtot as u64);
    rmsnorm_step(gpu, &mut s, &x1, &w.norm2, &n2, dim, nq, eps);
    let mlp_out = swiglu(gpu, &mut s, &n2, &w.w_gate, &w.w_up, &w.w_down, nq, dim, w.hidden);
    let x2 = gpu.storage(dimtot as u64);
    s.push(gpu.step(K_ADD2, &[&x1, &mlp_out, &x2], &[dimtot], dimtot));

    gpu.submit(&[], &s);
    gpu.read(&x2, dimtot as usize)
}

// ------------------------------------------------------ LinearPixelShuffleUpsample

/// `LinearPixelShuffleUpsample.forward`: `proj` (Linear, channel expand)
/// then `pixel_shuffle3d_cl` (rearrange), then (only when `stride[0]==2` and
/// `drop_leading`) crop the duplicate leading frame the shuffle produces.
#[allow(clippy::too_many_arguments)]
fn upsample_forward(gpu: &Gpu, x: &[f32], t: u32, h: u32, w: u32, cin: u32, w_proj: &[f32], b_proj: &[f32], stride: [u32; 3], reduction: u32, drop_leading: bool) -> (Vec<f32>, u32, u32, u32, u32) {
    let nq = t * h * w;
    let proj_out = cin * stride[0] * stride[1] * stride[2] / reduction;
    let cout = proj_out / (stride[0] * stride[1] * stride[2]);

    let gpu_x = upload(gpu, x);
    let wp = upload(gpu, w_proj);
    let bp = upload(gpu, b_proj);
    let mut s: Vec<Step> = Vec::new();
    let proj = gpu.storage((nq * proj_out) as u64);
    linear_step(gpu, &mut s, &gpu_x, &wp, Some(&bp), &proj, nq, cin, proj_out);

    let (t2, h2, w2) = (t * stride[0], h * stride[1], w * stride[2]);
    let shuffled = gpu.storage((t2 * h2 * w2 * cout) as u64);
    s.push(gpu.step(K_PIXEL_SHUFFLE3D_CL, &[&proj, &shuffled], &[t, h, w, cout, stride[0], stride[1], stride[2]], t2 * h2 * w2 * cout));

    gpu.submit(&[], &s);
    let y = gpu.read(&shuffled, (t2 * h2 * w2 * cout) as usize);

    if stride[0] == 2 && drop_leading {
        let per_frame = (h2 * w2 * cout) as usize;
        (y[per_frame..].to_vec(), t2 - 1, h2, w2, cout)
    } else {
        (y, t2, h2, w2, cout)
    }
}

// ------------------------------------------------------------ host layout

/// `[C,T,H,W] -> [T,H,W,C]` (channel-first to channels-last).
fn chw_to_hwc(x: &[f32], c: u32, t: u32, h: u32, w: u32) -> Vec<f32> {
    let (c, t, h, w) = (c as usize, t as usize, h as usize, w as usize);
    assert_eq!(x.len(), c * t * h * w);
    let mut out = vec![0f32; c * t * h * w];
    for ci in 0..c {
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    out[((ti * h + hi) * w + wi) * c + ci] = x[((ci * t + ti) * h + hi) * w + wi];
                }
            }
        }
    }
    out
}

/// `[T,H,W,C] -> [C,T,H,W]` (channels-last to channel-first).
fn hwc_to_chw(x: &[f32], c: u32, t: u32, h: u32, w: u32) -> Vec<f32> {
    let (c, t, h, w) = (c as usize, t as usize, h as usize, w as usize);
    assert_eq!(x.len(), c * t * h * w);
    let mut out = vec![0f32; c * t * h * w];
    for ti in 0..t {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    out[((ci * t + ti) * h + hi) * w + wi] = x[((ti * h + hi) * w + wi) * c + ci];
                }
            }
        }
    }
    out
}

// ------------------------------------------------------------------- API

/// One deterministic stage: `NA(RMSNorm)` blocks, then
/// `LinearPixelShuffleUpsample` - `_run_det_stage`. `x` is channels-last
/// `[t*h*w, dim]` at `cfg.stage_channels[stage]`'s own width; returns the
/// upsampled result at `cfg.stage_channels[stage+1]` (equivalently
/// [`NaDecoderConfig::context_channels`] when `stage==3`).
fn run_det_stage(gpu: &Gpu, weights: &Tensors, cfg: &NaDecoderConfig, stage: usize, x: &[f32], t: u32, h: u32, w: u32) -> (Vec<f32>, u32, u32, u32, u32) {
    let dim = cfg.stage_channels[stage];
    let heads = dim / cfg.head_dim;
    let kernel = cfg.stage_kernels[stage];
    assert!(t >= kernel[0] && h >= kernel[1] && w >= kernel[2], "run_det_stage: stage {stage} volume ({t},{h},{w}) below kernel {kernel:?}");
    let rope = na_rope_table(t, h, w, cfg.head_dim);
    let nq = t * h * w;
    let mut x = x.to_vec();
    for i in 0..cfg.stage_depths[stage] {
        let prefix = format!("decoder.det_stages.{stage}.{i}");
        let bw = upload_det_block(gpu, weights, &prefix, dim, cfg.head_dim);
        x = det_block_forward(gpu, &x, &bw, nq, dim, heads, cfg.head_dim, t, h, w, kernel, &rope, 1e-6);
    }
    let (stride, reduction) = cfg.upsamples[stage];
    let w_proj = tget(weights, &format!("decoder.upsamples.{stage}.proj.weight")).to_vec();
    let b_proj = tget(weights, &format!("decoder.upsamples.{stage}.proj.bias")).to_vec();
    upsample_forward(gpu, &x, t, h, w, dim, &w_proj, &b_proj, stride, reduction, true)
}

/// Stages 1-3 (0-indexed 0..3, `_run_det_stage` for each) - `un_normalize`,
/// `conv_in`, then the first 3 deterministic stages. `latent`:
/// `[in_channels,T,H,W]` channel-first, NORMALIZED (the VAE encoder's own
/// output convention, matching [`crate::vae3d::LtxVaeEncoder::encode`]).
/// `T,H,W` must each be `>=` stage 0's own kernel (`cfg.stage_kernels[0]`) -
/// every later stage's own floor is smaller after the fixed upsample.
///
/// Returns the stage-4 INPUT feature (before stage 4's own blocks/upsample),
/// channels-last `[t3,h3,w3,cfg.stage_channels[3]]`.
pub fn forward_stages_1_to_3(gpu: &Gpu, weights: &Tensors, cfg: &NaDecoderConfig, latent: &[f32], t: u32, h: u32, w: u32) -> (Vec<f32>, u32, u32, u32) {
    let c_in = cfg.in_channels;
    assert_eq!(latent.len(), (c_in * t * h * w) as usize, "forward_stages_1_to_3: latent has {} values, expected {}", latent.len(), c_in * t * h * w);
    let (kt0, kh0, kw0) = (cfg.stage_kernels[0][0], cfg.stage_kernels[0][1], cfg.stage_kernels[0][2]);
    assert!(t >= kt0 && h >= kh0 && w >= kw0, "forward_stages_1_to_3: (T,H,W)=({t},{h},{w}) below stage-0 kernel {:?}", cfg.stage_kernels[0]);

    // PerChannelStatistics.un_normalize: x*std + mean, per input channel.
    let mean = tget(weights, "per_channel_statistics.mean-of-means");
    let std = tget(weights, "per_channel_statistics.std-of-means");
    let hw = (t * h * w) as usize;
    let mut denorm = vec![0f32; latent.len()];
    for ci in 0..c_in as usize {
        for i in 0..hw {
            denorm[ci * hw + i] = latent[ci * hw + i] * std[ci] + mean[ci];
        }
    }
    let x_hwc = chw_to_hwc(&denorm, c_in, t, h, w);

    // conv_in: ChannelLinear(in_channels -> stage_channels[0]).
    let dim0 = cfg.stage_channels[0];
    let mut x = {
        let nq = t * h * w;
        let gx = upload(gpu, &x_hwc);
        let wc = upload(gpu, tget(weights, "decoder.conv_in.weight"));
        let bc = upload(gpu, tget(weights, "decoder.conv_in.bias"));
        let out = gpu.storage((nq * dim0) as u64);
        let mut s: Vec<Step> = Vec::new();
        linear_step(gpu, &mut s, &gx, &wc, Some(&bc), &out, nq, c_in, dim0);
        gpu.submit(&[], &s);
        gpu.read(&out, (nq * dim0) as usize)
    };

    let (mut ct, mut ch, mut cw) = (t, h, w);
    for stage in 0..3usize {
        let (nx, nt, nh, nw, _nc) = run_det_stage(gpu, weights, cfg, stage, &x, ct, ch, cw);
        x = nx;
        ct = nt;
        ch = nh;
        cw = nw;
    }
    (x, ct, ch, cw)
}

/// Stage 4 (0-indexed 3) - the 4th deterministic stage + its upsample,
/// producing the final stage-5 context. `x`: stage-4 INPUT feature, channels-
/// last `[t,h,w,cfg.stage_channels[3]]` (from [`forward_stages_1_to_3`]).
///
/// Returns `(context, t4, h4, w4)`, channels-last `[t4,h4,w4,
/// cfg.context_channels()]`.
pub fn forward_stage_4(gpu: &Gpu, weights: &Tensors, cfg: &NaDecoderConfig, x: &[f32], t: u32, h: u32, w: u32) -> (Vec<f32>, u32, u32, u32) {
    let (context, t4, h4, w4, _c) = run_det_stage(gpu, weights, cfg, 3, x, t, h, w);
    (context, t4, h4, w4)
}

/// Convenience composition of [`forward_stages_1_to_3`] + [`forward_stage_4`]
/// - the full deterministic "context" path, stages 1-4.
pub fn forward_context(gpu: &Gpu, weights: &Tensors, cfg: &NaDecoderConfig, latent: &[f32], t: u32, h: u32, w: u32) -> (Vec<f32>, u32, u32, u32) {
    let (x, t3, h3, w3) = forward_stages_1_to_3(gpu, weights, cfg, latent, t, h, w);
    forward_stage_4(gpu, weights, cfg, &x, t3, h3, w3)
}

// ------------------------------------------------------------ diff block

struct DiffBlockWeights {
    attn: NaAttnWeights,
    norm1: DeviceBuffer,
    norm2: DeviceBuffer,
    w_gate: DeviceBuffer,
    w_up: DeviceBuffer,
    w_down: DeviceBuffer,
    hidden: u32,
    context_proj_w: DeviceBuffer,
    context_proj_b: DeviceBuffer,
    /// Host copy - combined with the shared per-forward modulation via
    /// [`dit::adaln::add_table`] BEFORE upload (see [`forward_diff`]).
    scale_shift_table: Vec<f32>,
}

fn upload_diff_block(gpu: &Gpu, w: &Tensors, prefix: &str, dim: u32, head_dim: u32) -> DiffBlockWeights {
    DiffBlockWeights {
        attn: upload_attn(gpu, w, &format!("{prefix}.attn"), head_dim),
        norm1: upload(gpu, tget(w, &format!("{prefix}.norm1.weight"))),
        norm2: upload(gpu, tget(w, &format!("{prefix}.norm2.weight"))),
        w_gate: upload(gpu, tget(w, &format!("{prefix}.mlp.w_gate.weight"))),
        w_up: upload(gpu, tget(w, &format!("{prefix}.mlp.w_up.weight"))),
        w_down: upload(gpu, tget(w, &format!("{prefix}.mlp.w_down.weight"))),
        hidden: NaDecoderConfig::hidden(dim),
        context_proj_w: upload(gpu, tget(w, &format!("{prefix}.context_proj.weight"))),
        context_proj_b: upload(gpu, tget(w, &format!("{prefix}.context_proj.bias"))),
        scale_shift_table: tget(w, &format!("{prefix}.scale_shift_table")).to_vec(),
    }
}

/// `CombinedDiffusionNABlock.forward_combined`: context injection (`x = x +
/// context_proj(context)`, this block's OWN weights, `context` unchanged
/// across every block), modulated NA attention residual, modulated SwiGLU
/// residual - both ungated (see this module's header). `scale_msa` etc. are
/// this block's own COMBINED modulation (shared per-forward vector + this
/// block's `scale_shift_table` row, already summed by the caller), each
/// `[dim]` - broadcast to every one of `nq` rows before the device `modulate`
/// call, since stage 5's timestep is ONE scalar per forward, not per token.
#[allow(clippy::too_many_arguments)]
fn diff_block_forward(
    gpu: &Gpu,
    x: &[f32],
    context: &[f32],
    w: &DiffBlockWeights,
    nq: u32,
    dim: u32,
    ctx_c: u32,
    heads: u32,
    head_dim: u32,
    t: u32,
    h: u32,
    wd: u32,
    kernel: [u32; 3],
    rope: &NaRope,
    scale_msa: &[f32],
    shift_msa: &[f32],
    scale_mlp: &[f32],
    shift_mlp: &[f32],
    eps: f32,
) -> Vec<f32> {
    let x_buf = upload(gpu, x);
    let ctx_buf = upload(gpu, context);
    let cos = upload(gpu, &rope.cos);
    let sin = upload(gpu, &rope.sin);
    let dimtot = nq * dim;
    let mut s: Vec<Step> = Vec::new();

    let ctx_proj = gpu.storage(dimtot as u64);
    linear_step(gpu, &mut s, &ctx_buf, &w.context_proj_w, Some(&w.context_proj_b), &ctx_proj, nq, ctx_c, dim);
    let x0 = gpu.storage(dimtot as u64);
    s.push(gpu.step(K_ADD2, &[&x_buf, &ctx_proj, &x0], &[dimtot], dimtot));

    let n1 = gpu.storage(dimtot as u64);
    rmsnorm_step(gpu, &mut s, &x0, &w.norm1, &n1, dim, nq, eps);
    let scale_msa_b = upload(gpu, &broadcast_rows_plus1(scale_msa, nq));
    let shift_msa_b = upload(gpu, &broadcast_rows(shift_msa, nq));
    let y1 = modulate(gpu, &mut s, &n1, &scale_msa_b, &shift_msa_b, dimtot);
    let attn_out = na_attention(gpu, &mut s, &w.attn, &y1, nq, dim, heads, head_dim, t, h, wd, kernel[0], kernel[1], kernel[2], &cos, &sin, rope.half, eps);
    let x1 = gpu.storage(dimtot as u64);
    s.push(gpu.step(K_ADD2, &[&x0, &attn_out, &x1], &[dimtot], dimtot));

    let n2 = gpu.storage(dimtot as u64);
    rmsnorm_step(gpu, &mut s, &x1, &w.norm2, &n2, dim, nq, eps);
    let scale_mlp_b = upload(gpu, &broadcast_rows_plus1(scale_mlp, nq));
    let shift_mlp_b = upload(gpu, &broadcast_rows(shift_mlp, nq));
    let y2 = modulate(gpu, &mut s, &n2, &scale_mlp_b, &shift_mlp_b, dimtot);
    let mlp_out = swiglu(gpu, &mut s, &y2, &w.w_gate, &w.w_up, &w.w_down, nq, dim, w.hidden);
    let x2 = gpu.storage(dimtot as u64);
    s.push(gpu.step(K_ADD2, &[&x1, &mlp_out, &x2], &[dimtot], dimtot));

    gpu.submit(&[], &s);
    gpu.read(&x2, dimtot as usize)
}

fn linear_row(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(b.len(), out_dim);
    (0..out_dim)
        .map(|o| {
            let wr = &w[o * in_dim..o * in_dim + in_dim];
            b[o] + x.iter().zip(wr).map(|(a, c)| a * c).sum::<f32>()
        })
        .collect()
}

/// Stage 5: 8 `CombinedDiffusionNABlock`s over patchified noised pixels
/// `x_t`, guided by the (unchanging) stage-4 `context`, at the checkpoint's
/// real `default_num_inference_steps=1` + `model_output_type="x0"`.
///
/// ## Why this is a single forward, not a sampling loop
///
/// `_decode_pixels`: `timestep = linspace(1.0, 1.0/N, N)` with `N=1` is the
/// one-element tensor `[1.0]`; `single_step_x0 = (N==1 and
/// model_output_type=="x0")` is `True` for this checkpoint, so
/// `_decode_one_tile`'s loop `for i in range(N-1)` runs ZERO iterations and
/// falls straight to `t_now = timestep[:,-1]` (`=1.0`), one
/// `forward_diff_step` call, then `if model_output_type=="x0": return
/// model_out` - no Euler integration step at all. So this function IS the
/// complete "sampling loop" for this real checkpoint, not an approximation
/// of one: `x_t` at `t=1.0` (pure noise) goes in, the x0 pixel prediction
/// comes out directly. (A checkpoint with `N>1` or `model_output_type="v"`
/// would need the Euler recurrence `crate::pipeline`'s DiT loop already
/// implements the general shape of - out of scope here since this real
/// checkpoint never takes that path.)
///
/// `context`: `[t,h,w,cfg.context_channels()]` channels-last (from
/// [`forward_context`]). `x_t`: `[3,t,h*patch_size,w*patch_size]`
/// channel-first pixel-space noise, `N(0,1)` distributed (the reference's
/// own `torch.randn` init) - caller-supplied rather than reproduced from
/// upstream RNG, per this port's own "replay a captured input" precedent
/// (`crates/ltxv/src/pipeline.rs`'s M4 design). Returns the x0 pixel
/// prediction, same shape as `x_t`.
#[allow(clippy::too_many_arguments)]
pub fn forward_diff(gpu: &Gpu, weights: &Tensors, cfg: &NaDecoderConfig, context: &[f32], t: u32, h: u32, w: u32, x_t: &[f32]) -> Vec<f32> {
    assert_eq!(cfg.default_num_inference_steps, 1, "forward_diff only implements the real checkpoint's single-step x0 case (see this function's doc)");
    let dim5 = cfg.context_channels();
    let heads = dim5 / cfg.head_dim;
    let p = cfg.patch_size;
    let noised_c = cfg.noised_pixel_channels();
    let (hpx, wpx) = (h * p, w * p);
    assert_eq!(context.len(), (t * h * w * dim5) as usize, "forward_diff: context has {} values, expected {}", context.len(), t * h * w * dim5);
    assert_eq!(x_t.len(), (3 * t * hpx * wpx) as usize, "forward_diff: x_t has {} values, expected {}", x_t.len(), 3 * t * hpx * wpx);
    let kernel = cfg.stage5_kernel;
    assert!(t >= kernel[0] && h >= kernel[1] && w >= kernel[2], "forward_diff: context volume ({t},{h},{w}) below stage5 kernel {kernel:?}");

    // patchify (ops.py's convention, width-before-height - crate::patchify,
    // UNCHANGED from crate::vae3d's own outer-boundary use) then to
    // channels-last for conv_in_x_t.
    let patched_chw = crate::patchify::patchify(x_t, 3, t as usize, hpx as usize, wpx as usize, p as usize, p as usize);
    let x_hwc = chw_to_hwc(&patched_chw, noised_c, t, h, w);

    let nq = t * h * w;
    let mut x = {
        let gx = upload(gpu, &x_hwc);
        let wc = upload(gpu, tget(weights, "decoder.conv_in_x_t.weight"));
        let bc = upload(gpu, tget(weights, "decoder.conv_in_x_t.bias"));
        let out = gpu.storage((nq * dim5) as u64);
        let mut s: Vec<Step> = Vec::new();
        linear_step(gpu, &mut s, &gx, &wc, Some(&bc), &out, nq, noised_c, dim5);
        gpu.submit(&[], &s);
        gpu.read(&out, (nq * dim5) as usize)
    };

    // t_embedder + shared_adaln: ONE scalar timestep (t=1.0) for the whole
    // forward - PixArt sinusoid -> Linear -> SiLU -> Linear (dit::timestep,
    // the shared shape every scalar-timestep-conditioned model in this repo
    // uses), then AdaLNZero.proj(silu(t_emb)) -> 7 chunks of width dim5.
    let scaled_t = cfg.timestep_scale_multiplier * 1.0;
    let t_emb = dit::timestep::pixart_timestep_embed(
        scaled_t,
        256,
        tget(weights, "decoder.t_embedder.mlp.0.weight"),
        tget(weights, "decoder.t_embedder.mlp.0.bias"),
        cfg.t_emb_dim as usize,
        tget(weights, "decoder.t_embedder.mlp.2.weight"),
        tget(weights, "decoder.t_embedder.mlp.2.bias"),
        cfg.t_emb_dim as usize,
        10000.0,
    );
    let t_emb_silu = model::hostmath::silu_slice(&t_emb);
    let shared_mod = linear_row(&t_emb_silu, tget(weights, "decoder.shared_adaln.proj.weight"), tget(weights, "decoder.shared_adaln.proj.bias"), 7 * dim5 as usize);

    let rope = na_rope_table(t, h, w, cfg.head_dim);
    let d = dim5 as usize;
    for i in 0..cfg.stage_depths[4] {
        let prefix = format!("decoder.diff_blocks.{i}");
        let bw = upload_diff_block(gpu, weights, &prefix, dim5, cfg.head_dim);
        // AdaLNZero row order: 0=scale_msa,1=shift_msa,2=gate_msa(unused),
        // 3=scale_mlp,4=shift_mlp,5=gate_mlp(unused),6=gate_ctx(unused).
        let combined = dit::adaln::add_table(&shared_mod, &bw.scale_shift_table, 1, 7 * d);
        let scale_msa = &combined[0..d];
        let shift_msa = &combined[d..2 * d];
        let scale_mlp = &combined[3 * d..4 * d];
        let shift_mlp = &combined[4 * d..5 * d];
        x = diff_block_forward(gpu, &x, context, &bw, nq, dim5, dim5, heads, cfg.head_dim, t, h, w, kernel, &rope, scale_msa, shift_msa, scale_mlp, shift_mlp, 1e-6);
    }

    let y = {
        let x_buf = upload(gpu, &x);
        let nw = upload(gpu, tget(weights, "decoder.norm_out.weight"));
        let mut s: Vec<Step> = Vec::new();
        let n_out = gpu.storage((nq * dim5) as u64);
        rmsnorm_step(gpu, &mut s, &x_buf, &nw, &n_out, dim5, nq, 1e-6);
        let wc = upload(gpu, tget(weights, "decoder.conv_out.weight"));
        let bc = upload(gpu, tget(weights, "decoder.conv_out.bias"));
        let out = gpu.storage((nq * noised_c) as u64);
        linear_step(gpu, &mut s, &n_out, &wc, Some(&bc), &out, nq, dim5, noised_c);
        gpu.submit(&[], &s);
        gpu.read(&out, (nq * noised_c) as usize)
    };

    let y_chw = hwc_to_chw(&y, noised_c, t, h, w);
    crate::patchify::unpatchify(&y_chw, 3, t as usize, h as usize, w as usize, p as usize, p as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_config_shapes() {
        let cfg = NaDecoderConfig::ltx25();
        assert_eq!(cfg.context_channels(), 256);
        assert_eq!(cfg.noised_pixel_channels(), 48);
        assert_eq!(NaDecoderConfig::hidden(2048), 8192);
        assert_eq!(NaDecoderConfig::hidden(256), 1024);
    }

    /// The manifest matches the real checkpoint's decoder tensor count
    /// exactly: 309 `decoder.*` tensors (310 minus the dead `type_emb`) + 2
    /// `per_channel_statistics.*` = 311, counted directly off the real
    /// header (`python3 -c "... startswith('decoder.')/... == 310"`)
    /// before this manifest was written, not derived from it.
    #[test]
    fn manifest_counts_the_shipped_checkpoint() {
        let cfg = NaDecoderConfig::ltx25();
        let m = cfg.tensor_manifest();
        assert_eq!(m.len(), 311, "NA decoder manifest has {} tensors, expected 311", m.len());
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate tensor name in the manifest");
        assert!(!names.contains("decoder.type_emb"), "type_emb is dead weight and must not appear in the manifest");
        assert!(names.contains("decoder.diff_blocks.7.scale_shift_table"));
        assert!(names.contains("decoder.det_stages.3.1.attn.qkv.weight"));

        let get = |n: &str| m.iter().find(|(k, _)| k == n).unwrap().1.clone();
        assert_eq!(get("decoder.conv_in.weight"), vec![2048, 128]);
        assert_eq!(get("decoder.conv_in_x_t.weight"), vec![256, 48]);
        assert_eq!(get("decoder.diff_blocks.0.attn.qkv.weight"), vec![768, 256]);
        assert_eq!(get("decoder.diff_blocks.0.scale_shift_table"), vec![7, 256]);
        assert_eq!(get("decoder.upsamples.0.proj.weight"), vec![4096, 2048]);
        assert_eq!(get("decoder.upsamples.3.proj.weight"), vec![2048, 512]);
        assert_eq!(get("decoder.shared_adaln.proj.weight"), vec![1792, 384]);
    }

    /// The importer's two-way coverage, on a synthetic (zero-filled)
    /// checkpoint at the manifest's own shapes - a missing/extra tensor is
    /// caught by name.
    #[test]
    fn import_validates_both_directions() {
        let cfg = NaDecoderConfig::ltx25();
        let manifest = cfg.tensor_manifest();
        let build = |skip: Option<&str>, add: Option<&str>| -> Vec<StTensor> {
            let mut v: Vec<StTensor> = manifest
                .iter()
                .filter(|(n, _)| Some(n.as_str()) != skip)
                .map(|(n, s)| StTensor { name: n.clone(), shape: s.clone(), data: vec![0.0; s.iter().product()] })
                .collect();
            if let Some(a) = add {
                v.push(StTensor { name: a.into(), shape: vec![1], data: vec![0.0] });
            }
            v
        };

        let w = import_na_decoder(build(None, None), &cfg).expect("full manifest imports");
        // Post-qkv-split count: every attn block's fused qkv (2 tensors)
        // becomes 6 (to_q/to_k/to_v * weight/bias) - +4 per block, 24 blocks
        // total (16 det + 8 diff).
        assert_eq!(w.len(), manifest.len() + 24 * 4);

        // A source tensor from the OTHER checkpoint's encoder must be
        // silently dropped, not counted as coverage.
        let mut with_encoder = build(None, None);
        with_encoder.push(StTensor { name: "encoder.conv_in.conv.weight".into(), shape: vec![1], data: vec![0.0] });
        let w2 = import_na_decoder(with_encoder, &cfg).expect("encoder.* silently ignored");
        assert_eq!(w2.len(), w.len());

        // The dead decoder.type_emb tensor must also be silently dropped.
        let mut with_type_emb = build(None, None);
        with_type_emb.push(StTensor { name: "decoder.type_emb".into(), shape: vec![128], data: vec![0.0; 128] });
        let w3 = import_na_decoder(with_type_emb, &cfg).expect("decoder.type_emb silently ignored");
        assert_eq!(w3.len(), w.len());

        let e = import_na_decoder(build(Some("decoder.conv_out.bias"), None), &cfg).unwrap_err();
        assert!(e.contains("decoder.conv_out.bias"), "{e}");

        let e = import_na_decoder(build(None, Some("decoder.det_stages.99.0.norm1.weight")), &cfg).unwrap_err();
        assert!(e.contains("unused source tensors"), "{e}");
    }

    /// `default_rope_dim_split(64) == (16, 24, 24)` - the real checkpoint's
    /// only `head_dim` value, pinned by hand against the reference's own
    /// formula (`(64//4)//2*2=16`, `(64-16)/2=24`, `24%2==0` so no
    /// adjustment needed).
    #[test]
    fn rope_split_matches_reference_at_head_dim_64() {
        assert_eq!(default_rope_dim_split(64), (16, 24, 24));
    }

    /// `cos^2+sin^2 == 1` everywhere in the NA RoPE table - the same
    /// structural invariant `crate::rope`'s own table test uses.
    #[test]
    fn rope_table_is_unit_rotations() {
        let r = na_rope_table(3, 5, 5, 64);
        assert_eq!(r.half, 32);
        assert_eq!(r.cos.len(), 3 * 5 * 5 * 32);
        for (c, sv) in r.cos.iter().zip(&r.sin) {
            let dev = (*c as f64 * *c as f64 + *sv as f64 * *sv as f64 - 1.0).abs();
            assert!(dev < 1e-5, "cos^2+sin^2 deviates by {dev}");
        }
    }

    /// NATTEN's own per-axis window-bounds formula
    /// (`fallback_na/eager.py::_window_bounds`, this port's correctness
    /// oracle - see this module's header), hand-verified at several
    /// `(length, kernel, query)` triples INDEPENDENT of any real-weight run:
    /// the window is always `kernel`-wide and SHIFTS inward at a boundary
    /// rather than clamping+masking down to a smaller width.
    fn window_start(length: i32, kernel: i32, query: i32) -> i32 {
        let half = kernel / 2;
        let lo = length - kernel;
        (query - half).clamp(0, lo)
    }

    #[test]
    fn window_bounds_matches_the_eager_oracle() {
        // length=9, kernel=7: lo=2, half=3. Interior queries clamp at the
        // edges (0 and 2), never producing a start outside [0, lo] - the
        // window is ALWAYS exactly `kernel`=7 wide, whatever the query.
        let cases: [(i32, i32, i32, i32); 9] = [
            (9, 7, 0, 0),
            (9, 7, 1, 0),
            (9, 7, 2, 0),
            (9, 7, 3, 0),
            (9, 7, 4, 1),
            (9, 7, 5, 2),
            (9, 7, 6, 2),
            (9, 7, 7, 2),
            (9, 7, 8, 2),
        ];
        for (length, kernel, query, want) in cases {
            let got = window_start(length, kernel, query);
            assert_eq!(got, want, "length={length} kernel={kernel} query={query}: start {got}, want {want}");
            assert!(got >= 0 && got + kernel <= length, "window [{got},{}) escapes [0,{length})", got + kernel);
        }

        // kernel == length: every query's window is the whole axis (start
        // always 0) - the trivial no-shift case.
        for query in 0..7 {
            assert_eq!(window_start(7, 7, query), 0);
        }

        // Odd/even kernel parity both exercised: kernel=11 (odd, half=5) at
        // length=13 (this port's stage-5 golden volume).
        assert_eq!(window_start(13, 11, 0), 0);
        assert_eq!(window_start(13, 11, 6), 1);
        assert_eq!(window_start(13, 11, 12), 2);
    }
}
