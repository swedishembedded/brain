// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The SDXL `UNet2DConditionModel` forward graph.
//!
//! **Composition, not new math.** Every convolutional stage comes from the
//! shared [`vae::blocks::Builder`] (conv / GroupNorm / SiLU / add /
//! nearest-2× upsample, and its NCHW↔NLC permutations); every transformer stage
//! comes from `model::block` (`layernorm_fwd`, `pick_gemm`, `flash_bidir_fwd`,
//! `bidir_fwd`) and the existing attention kernels. **This crate adds no kernel
//! and no block** — the measured claim that put phase 4b in
//! `docs/imaging/plan.md`, now discharged.
//!
//! ```text
//! emb   = time_embedding(sinusoid(t)) + add_embedding([pooled ‖ sinusoid(time_ids)])
//! h     = conv_in(x)                                       ; skips = [h]
//! down i: for j in 0..layers:  h = resnet(h, emb) [; h = transformer(h, enc)]
//!                              skips.push(h)
//!         if not last:         h = conv_s2(h) ; skips.push(h)
//! mid   : h = resnet(h, emb) ; h = transformer(h, enc) ; h = resnet(h, emb)
//! up   i: for j in 0..=layers: h = resnet(cat(h, skips.pop()), emb)
//!                              [; h = transformer(h, enc)]
//!         if not last:         h = conv3(upsample2(h))
//! out   = conv_out(silu(conv_norm_out(h)))
//! ```
//!
//! ### The five conventions this graph pins, each verified against diffusers
//! 1. **The timestep is ADDED, not scale-shifted.** SDXL ships
//!    `resnet_time_scale_shift: "default"`, which in `ResnetBlock2D.forward` is
//!    `hidden = conv1(...) + time_emb_proj(silu(temb))[:, :, None, None]`
//!    *before* `norm2` — a per-channel broadcast add (`add_chan_bcast`), not
//!    `film_chan`. The `"scale_shift"` variant exists in diffusers and is a
//!    different model; SDXL is not it.
//! 2. **The added conditioning is `[pooled ‖ time_ids]`**, pooled first — see
//!    [`crate::hostemb::added_cond`].
//! 3. **`attention_head_dim` is a head COUNT** — see [`crate::config`].
//! 4. **Two GroupNorm epsilons in one graph**: 1e-5 in the resnets and
//!    `conv_norm_out`, 1e-6 inside every `Transformer2DModel`.
//! 5. **GEGLU is `hidden * gelu(gate)`** with `hidden` the FIRST half of the
//!    projection's output row. The halves are interleaved per row, so the
//!    projection is SPLIT into two weights at import (see [`crate::import`])
//!    and the activation is `gelu_erf` + `mul` — the composition
//!    `mul.wgsl`'s own header prescribes. `geglu_shift` is a different
//!    function (`gelu(h)·(g+1)`, Moondream's) and does not apply.
//!
//! ### Attention
//! Self-attention over the `H·W` spatial positions uses
//! `block::flash_bidir_fwd` wherever the device reports
//! `DeviceCaps::workgroup_reductions`, because the materialised score slab is
//! `heads·T²` — 671 MB at SDXL's native 1024² (T = 4096, 10 heads), which is
//! both a needless allocation and close to the P40's 2047 MiB per-binding
//! limit. On a device without workgroup reductions (the CPU JIT) it falls back
//! to the materialised `attn_*_bidir` trio, and the slabs are only allocated on
//! that path. Cross-attention is the `attn_*_cross` trio: its slab is
//! `heads·T·77`, which does not have that problem.

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block;
use vae::blocks::{BlockNames, Builder, Tensors};

use crate::config::{BlockKind, UNetConfig, N_TIME_IDS, TRANSFORMER_NORM_EPS};
use crate::hostemb;

/// Kernel slots after the shared `vae::blocks` set. Adding one means bumping
/// [`KERNELS`]'s length; the const constructor checks it at compile time.
const K_LAYERNORM: usize = vae::blocks::NEXT_SLOT;
const K_MATMUL: usize = vae::blocks::NEXT_SLOT + 1;
const K_BIAS_ADD: usize = vae::blocks::NEXT_SLOT + 2;
const K_GELU_ERF: usize = vae::blocks::NEXT_SLOT + 3;
const K_MUL: usize = vae::blocks::NEXT_SLOT + 4;
const K_XSCORES: usize = vae::blocks::NEXT_SLOT + 5;
const K_XSOFTMAX: usize = vae::blocks::NEXT_SLOT + 6;
const K_XAPPLY: usize = vae::blocks::NEXT_SLOT + 7;
const K_FLASH: usize = vae::blocks::NEXT_SLOT + 8;
const K_FLASH_SPLIT: usize = vae::blocks::NEXT_SLOT + 9;
const K_ADD_CHAN: usize = vae::blocks::NEXT_SLOT + 10;

/// The tiled GEMM every `nn.Linear` here dispatches: the ONE the shared block
/// set already registers. This crate used to register its own `matmul_reg2`
/// beside it and send every linear there — the slower of two kernels it was
/// already carrying (`docs/lessons.md` #17).
const K_MATMUL_REG: usize = vae::blocks::MATMUL_REG3_SLOT;
// `layernorm_rows` occupies the last slot and is resolved BY NAME through
// `block::LayerNormIds::resolve_fwd`, never indexed directly.
const N_EXTRA: usize = 12;

/// This model's kernel set: the shared block kernels (slots `0..NEXT_SLOT`,
/// copied — never restated — by [`vae::blocks::kernels_with`]) then the extras
/// above. The shared set's `attn_*_bidir` trio is reused here for the
/// CPU-fallback self-attention rather than registered twice; its slots come
/// from [`vae::blocks::ATTN_BIDIR_SLOTS`], never from a literal.
pub const KERNELS: [(&str, &str); vae::blocks::NEXT_SLOT + N_EXTRA] = kernel_set();

const fn kernel_set() -> [(&'static str, &'static str); vae::blocks::NEXT_SLOT + N_EXTRA] {
    let mut k = vae::blocks::kernels_with::<{ vae::blocks::NEXT_SLOT + N_EXTRA }>();
    k[K_LAYERNORM] = ("layernorm", kernels::LAYERNORM);
    k[K_MATMUL] = ("matmul", kernels::MATMUL);
    k[K_BIAS_ADD] = ("bias_add", kernels::BIAS_ADD);
    k[K_GELU_ERF] = ("gelu_erf", kernels::GELU_ERF);
    k[K_MUL] = ("mul", kernels::MUL);
    k[K_XSCORES] = ("attn_scores_cross", kernels::ATTN_SCORES_CROSS);
    k[K_XSOFTMAX] = ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS);
    k[K_XAPPLY] = ("attn_apply_cross", kernels::ATTN_APPLY_CROSS);
    k[K_FLASH] = ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR);
    k[K_FLASH_SPLIT] = ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT);
    k[K_ADD_CHAN] = ("add_chan_bcast", kernels::ADD_CHAN_BCAST);
    k[vae::blocks::NEXT_SLOT + 11] = ("layernorm_rows", kernels::LAYERNORM_ROWS);
    k
}

/// One entry of the down path's skip stack: `(buffer, channels, h, w)`.
pub type Skip = (DeviceBuffer, u32, u32, u32);

/// Words needed by ONE materialised self-attention slab (`scores` and `probs`
/// are the same size) on a device without workgroup reductions.
///
/// Sized over the levels that actually RECORD a transformer, not all of them.
/// Taking the max over all levels is quadratically wrong at the finest one:
/// SDXL's level 0 is `DownBlock2D`/`UpBlock2D` (no attention at all), yet its
/// `T = H·W` is 16x the next level's, so including it sizes the pair at
/// 5·(H·W)² instead of 10·(H·W/4)² — 8x too big, i.e. 10.7 GB of never-bound
/// slab at a 128×128 latent.
///
/// `with_up = false` is the **ControlNet** shape: a trainable copy of the
/// down + mid blocks has no up path at all, and `cfg.up_block_types` may be
/// empty, so the up term must not even be indexed.
pub fn attn_slab_words(cfg: &UNetConfig, h: u32, w: u32, with_up: bool) -> u64 {
    let levels = cfg.levels();
    let max_t = (h as u64) * (w as u64);
    let mut words = 0u64;
    for l in 0..levels {
        // The mid block's transformer always runs, at the coarsest level.
        let used = l == levels - 1
            || cfg.down_block_types[l] == BlockKind::CrossAttn
            || (with_up && cfg.up_block_types[levels - 1 - l] == BlockKind::CrossAttn);
        if !used {
            continue;
        }
        let t = max_t >> (2 * l);
        words = words.max(cfg.attention_heads[l] as u64 * t * t);
    }
    words
}

/// Graph-recording state: the shared block builder plus the handful of things
/// only the transformer half needs.
///
/// **Public because a ControlNet is a trainable copy of exactly these blocks.**
/// `crates/controlnet` records `conditioning` → `conv_in` → [`Rec::down_path`]
/// → [`Rec::mid_block`] with the same tensor prefixes and the same tap names as
/// [`Unet::new`] does, so the two graphs are one implementation rather than a
/// copy that can drift (AGENTS.md "one implementation"). Nothing here is
/// SDXL-specific beyond `UNetConfig` itself.
pub struct Rec<'a> {
    b: Builder<'a>,
    ln: block::LayerNormIds,
    flash: block::FlashIds,
    /// The device runs workgroup-cooperative kernels (flash attention).
    coop: bool,
    /// Materialised self-attention slabs, allocated ONLY on the non-cooperative
    /// path (see the module header).
    slab: Option<(DeviceBuffer, DeviceBuffer)>,
    t_enc: u32,
    /// An optional extra term added into every cross-attention context, before
    /// the block's shared `to_out` — the consumer half of
    /// `model::attninject::CrossAttnInject`.
    ///
    /// It cannot be a pre-supplied input the way a control residual is: the
    /// adapter attends the per-site QUERY, which only exists inside
    /// [`Rec::transformer_block`]. So the graph calls it there, at the point
    /// that tensor is live.
    inject: Option<&'a dyn model::attninject::CrossAttnInject>,
    /// Which cross-attention site the recorder is on. Counted in graph order,
    /// which is the order the released `ip_adapter` keys are numbered in.
    site: usize,
    /// `silu(emb)` — computed once by [`Rec::conditioning`], because
    /// `ResnetBlock2D` applies the SAME nonlinearity to the SAME `temb` in
    /// every one of the 17 resnets. `None` until then, so a resnet recorded
    /// before the conditioning chain panics by name instead of binding a
    /// placeholder buffer.
    temb_act: Option<DeviceBuffer>,
}

impl<'a> Rec<'a> {
    /// A recorder over `tensors` for a graph with `t_enc` text tokens.
    ///
    /// `slab_words` sizes the materialised self-attention slabs and is only
    /// consulted on a device without workgroup reductions — see
    /// [`attn_slab_words`].
    pub fn new(
        gpu: &'a Gpu,
        cfg: &UNetConfig,
        tensors: &'a Tensors,
        t_enc: u32,
        slab_words: u64,
        taps: bool,
    ) -> Rec<'a> {
        let coop = gpu.caps().workgroup_reductions;
        let slab =
            if coop { None } else { Some((gpu.storage(slab_words.max(1)), gpu.storage(slab_words.max(1)))) };
        let b = Builder::new(gpu, tensors, cfg.norm_eps, cfg.norm_num_groups, BlockNames::diffusers(), taps);
        let ln = block::LayerNormIds::resolve_fwd(gpu, K_LAYERNORM);
        let flash = block::FlashIds { bidir: K_FLASH, split: Some(K_FLASH_SPLIT) };
        Rec { b, ln, flash, coop, slab, t_enc, inject: None, site: 0, temb_act: None }
    }

    /// The underlying block builder — conv / GroupNorm / SiLU / add / upsample.
    pub fn blocks(&mut self) -> &mut Builder<'a> {
        &mut self.b
    }

    /// Consume the recorder, yielding the builder so the caller can `finish()`.
    pub fn into_blocks(self) -> Builder<'a> {
        self.b
    }

    /// `y[m, n] = x[m, k] · W[n, k]ᵀ (+ b[n])`, the shape every diffusers
    /// `nn.Linear` has. `bias = false` covers `to_q/to_k/to_v`, which SDXL
    /// ships bias-free.
    pub fn linear(&mut self, prefix: &str, m: u32, k: u32, n: u32, bias: bool, x: &DeviceBuffer) -> DeviceBuffer {
        let w = self.b.dev(&format!("{prefix}.weight"));
        let y = self.b.act((m as u64) * (n as u64));
        let (kind, threads) = block::pick_gemm(m as usize, n as usize, K_MATMUL, K_MATMUL_REG, false);
        let g = self.b.gpu();
        self.b.push_step(g.step(kind, &[x, &w, &y], &[m, k, n], threads));
        if bias {
            let bb = self.b.dev(&format!("{prefix}.bias"));
            let g = self.b.gpu();
            self.b.push_step(g.step(K_BIAS_ADD, &[&y, &bb], &[m, n], m * n));
        }
        y
    }

    pub fn layernorm(&mut self, prefix: &str, rows: u32, d: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let (gamma, beta) =
            (self.b.dev(&format!("{prefix}.weight")), self.b.dev(&format!("{prefix}.bias")));
        let y = self.b.act((rows as u64) * (d as u64));
        let g = self.b.gpu();
        // LayerNorm eps is `norm_eps` on the BasicTransformerBlock, which
        // diffusers defaults to 1e-5 and SDXL does not override.
        let step = block::layernorm_fwd(g, &self.ln, x, &gamma, &beta, &y, d, rows, 1e-5);
        self.b.push_step(step);
        y
    }

    /// SDXL's conditioning chain: `time_embedding(sinusoid(t)) +
    /// add_embedding([pooled ‖ time_ids sinusoids])`, then the single
    /// `silu(emb)` every resnet consumes.
    ///
    /// Must be recorded before any [`Rec::resnet`]. Byte-for-byte the same
    /// modules (and the same tap names) in `UNet2DConditionModel` and in
    /// `ControlNetModel` — which is why it lives here and not in either.
    pub fn conditioning(&mut self, cfg: &UNetConfig, temb_in: &DeviceBuffer, aug_in: &DeviceBuffer) {
        let (c0, te) = (cfg.block_out_channels[0], cfg.time_embed_dim);
        let t1 = self.linear("time_embedding.linear_1", 1, c0, te, true, temb_in);
        self.b.tap("time_embedding.linear_1".into(), &t1, te);
        let t1a = self.b.silu(te, &t1);
        self.b.free(te as u64, t1);
        let temb = self.linear("time_embedding.linear_2", 1, te, te, true, &t1a);
        self.b.tap("time_embedding".into(), &temb, te);
        self.b.free(te as u64, t1a);

        let a1 = self.linear(
            "add_embedding.linear_1",
            1,
            cfg.projection_class_embeddings_input_dim,
            te,
            true,
            aug_in,
        );
        self.b.tap("add_embedding.linear_1".into(), &a1, te);
        let a1a = self.b.silu(te, &a1);
        self.b.free(te as u64, a1);
        let aug = self.linear("add_embedding.linear_2", 1, te, te, true, &a1a);
        self.b.tap("add_embedding".into(), &aug, te);
        self.b.free(te as u64, a1a);

        let emb = self.b.add(te, &temb, &aug);
        self.b.free(te as u64, temb);
        self.b.free(te as u64, aug);
        // `ResnetBlock2D` applies `nonlinearity(temb)` before its own linear;
        // the argument is identical in all 17 resnets, so it is hoisted here.
        let act = self.b.silu(te, &emb);
        self.b.free(te as u64, emb);
        self.temb_act = Some(act);
    }

    /// The down path: `layers_per_block` resnets (each optionally followed by a
    /// spatial transformer) per level, plus a stride-2 downsampler between
    /// levels. Returns the running hidden state, the skip/residual stack in
    /// PUSH order (`x` first), and the final spatial size.
    ///
    /// `x` is `conv_in`'s output — the ControlNet adds its conditioning
    /// embedding to that *before* calling, which is the only difference between
    /// the two down paths and the reason this takes it as an argument.
    pub fn down_path(
        &mut self,
        cfg: &UNetConfig,
        h: u32,
        w: u32,
        enc: &DeviceBuffer,
        x: &DeviceBuffer,
    ) -> (DeviceBuffer, Vec<Skip>, u32, u32) {
        let levels = cfg.levels();
        let c0 = cfg.block_out_channels[0];
        let mut hh = x.clone();
        let mut skips: Vec<Skip> = vec![(x.clone(), c0, h, w)];
        let (mut ch, mut cw) = (h, w);
        let mut prev = c0;
        for i in 0..levels {
            let cout = cfg.block_out_channels[i];
            for j in 0..cfg.layers_per_block {
                let cin = if j == 0 { prev } else { cout };
                hh = self.resnet(
                    &format!("down_blocks.{i}.resnets.{j}"),
                    &format!("down{i}.resnet{j}"),
                    cin,
                    cout,
                    ch,
                    cw,
                    cfg.time_embed_dim,
                    &hh,
                );
                if cfg.down_block_types[i] == BlockKind::CrossAttn {
                    hh = self.transformer(
                        &format!("down_blocks.{i}.attentions.{j}"),
                        &format!("down{i}.attn{j}"),
                        cfg,
                        i,
                        ch,
                        cw,
                        enc,
                        &hh,
                    );
                }
                skips.push((hh.clone(), cout, ch, cw));
            }
            if i + 1 < levels {
                // `Downsample2D(use_conv, padding=1)`: a symmetric-pad stride-2
                // 3x3 conv. NOT the VAE's asymmetric `F.pad(x,(0,1,0,1))` +
                // `padding=0` form that `Builder::conv_down` implements — the
                // two differ by a half-pixel shift in every feature.
                let next = self.b.conv_s(
                    &format!("down_blocks.{i}.downsamplers.0.conv"),
                    cout,
                    cout,
                    3,
                    2,
                    1,
                    ch,
                    cw,
                    ch / 2,
                    cw / 2,
                    &hh,
                );
                ch /= 2;
                cw /= 2;
                hh = next;
                self.b.tap(format!("down{i}.downsample0"), &hh, cout * ch * cw);
                skips.push((hh.clone(), cout, ch, cw));
            }
            prev = cout;
        }
        (hh, skips, ch, cw)
    }

    /// `UNetMidBlock2DCrossAttn`: resnet → spatial transformer → resnet, at the
    /// coarsest level.
    pub fn mid_block(
        &mut self,
        cfg: &UNetConfig,
        ch: u32,
        cw: u32,
        enc: &DeviceBuffer,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let cmid = *cfg.block_out_channels.last().expect("levels >= 1");
        let te = cfg.time_embed_dim;
        let mut hh = self.resnet("mid_block.resnets.0", "mid.resnet0", cmid, cmid, ch, cw, te, x);
        hh = self.transformer("mid_block.attentions.0", "mid.attn0", cfg, cfg.levels() - 1, ch, cw, enc, &hh);
        self.resnet("mid_block.resnets.1", "mid.resnet1", cmid, cmid, ch, cw, te, &hh)
    }

    /// `ResnetBlock2D` with `time_embedding_norm = "default"`.
    #[allow(clippy::too_many_arguments)]
    pub fn resnet(
        &mut self,
        prefix: &str,
        tap: &str,
        cin: u32,
        cout: u32,
        h: u32,
        w: u32,
        temb_dim: u32,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let (nin, nout) = ((cin * h * w) as u64, (cout * h * w) as u64);
        let (shortcut, owned) = if cin != cout {
            (self.b.conv(&format!("{prefix}.conv_shortcut"), cin, cout, 1, 0, h, w, x), true)
        } else {
            (x.clone(), false)
        };
        if owned {
            self.b.tap(format!("{tap}.conv_shortcut"), &shortcut, cout * h * w);
        }
        let n1 = self.b.gn(&format!("{prefix}.norm1"), cin, h, w, x);
        self.b.tap(format!("{tap}.norm1"), &n1, cin * h * w);
        let s1 = self.b.silu(cin * h * w, &n1);
        self.b.free(nin, n1);
        let c1 = self.b.conv(&format!("{prefix}.conv1"), cin, cout, 3, 1, h, w, &s1);
        self.b.tap(format!("{tap}.conv1"), &c1, cout * h * w);
        self.b.free(nin, s1);

        // temb -> [1, cout], broadcast over H*W and added to every channel.
        let temb_act =
            self.temb_act.clone().expect("Rec::conditioning must be recorded before any resnet");
        let tp = self.linear(&format!("{prefix}.time_emb_proj"), 1, temb_dim, cout, true, &temb_act);
        self.b.tap(format!("{tap}.time_emb_proj"), &tp, cout);
        let c1t = self.b.act(nout);
        let g = self.b.gpu();
        // `add_chan_bcast` Params: [N, C, HW]; bufs [x, v[N*C], y].
        self.b.push_step(g.step(K_ADD_CHAN, &[&c1, &tp, &c1t], &[1, cout, h * w], cout * h * w));
        self.b.tap(format!("{tap}.temb_add"), &c1t, cout * h * w);
        self.b.free(nout, c1);
        self.b.free(cout as u64, tp);

        let n2 = self.b.gn(&format!("{prefix}.norm2"), cout, h, w, &c1t);
        self.b.tap(format!("{tap}.norm2"), &n2, cout * h * w);
        self.b.free(nout, c1t);
        let s2 = self.b.silu(cout * h * w, &n2);
        self.b.free(nout, n2);
        let c2 = self.b.conv(&format!("{prefix}.conv2"), cout, cout, 3, 1, h, w, &s2);
        self.b.tap(format!("{tap}.conv2"), &c2, cout * h * w);
        self.b.free(nout, s2);
        let out = self.b.add(cout * h * w, &shortcut, &c2);
        self.b.free(nout, c2);
        if owned {
            self.b.free(nout, shortcut);
        }
        self.b.tap(tap.to_string(), &out, cout * h * w);
        out
    }

    /// One `BasicTransformerBlock` over `[T, c]` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_block(
        &mut self,
        prefix: &str,
        tap: &str,
        c: u32,
        t: u32,
        heads: u32,
        hd: u32,
        cross_dim: u32,
        enc: &DeviceBuffer,
        h: &DeviceBuffer,
    ) -> DeviceBuffer {
        let n = (t as u64) * (c as u64);
        let t_enc = self.t_enc;

        // ---- self-attention ------------------------------------------------
        let n1 = self.layernorm(&format!("{prefix}.norm1"), t, c, h);
        self.b.tap(format!("{tap}.norm1"), &n1, t * c);
        let qkv = self.linear(&format!("{prefix}.attn1.qkv"), t, c, 3 * c, false, &n1);
        self.b.free(n, n1);
        let ctx = self.b.act(n);
        self.self_attention(heads, hd, c, t, &qkv, &ctx);
        self.b.free(3 * n, qkv);
        let ao = self.linear(&format!("{prefix}.attn1.to_out"), t, c, c, true, &ctx);
        self.b.tap(format!("{tap}.attn1"), &ao, t * c);
        self.b.free(n, ctx);
        let h1 = self.b.add(t * c, &ao, h);
        self.b.free(n, ao);

        // ---- cross-attention to the text encoding ---------------------------
        let n2 = self.layernorm(&format!("{prefix}.norm2"), t, c, &h1);
        self.b.tap(format!("{tap}.norm2"), &n2, t * c);
        let q = self.linear(&format!("{prefix}.attn2.to_q"), t, c, c, false, &n2);
        self.b.free(n, n2);
        // k and v come from one fused [2c, cross_dim] weight, so `kv` is
        // exactly the `[t_enc, 2c]` layout `attn_*_cross` expects.
        let kv = self.linear(&format!("{prefix}.attn2.kv"), t_enc, cross_dim, 2 * c, false, enc);
        let ctx2 = self.b.act(n);
        self.cross_attention(heads, hd, c, t, &q, &kv, &ctx2);
        // The adapter attends the SAME queries, so `q` must outlive the text
        // attention when one is installed.
        let q_for_inject = q.clone();
        if self.inject.is_none() {
            self.b.free(n, q);
        }
        self.b.free((t_enc as u64) * 2 * (c as u64), kv);
        // The decoupled term goes in HERE — on the context, before the shared
        // `to_out`, which is where IPAttnProcessor puts it. Adding after
        // `to_out`, or concatenating the adapter's tokens onto the text tokens,
        // both run and both produce a plausible image.
        if let Some(inj) = self.inject {
            let k = self.site;
            let gpu = self.b.gpu();
            let mut extra = Vec::new();
            inj.inject(&mut extra, gpu, k, &q_for_inject, &ctx2, t, c);
            for st in extra {
                self.b.push_step(st);
            }
            self.b.tap(format!("{tap}.attn2_injected"), &ctx2, t * c);
        }
        self.site += 1;
        if self.inject.is_some() {
            self.b.free(n, q_for_inject.clone());
        }
        let ao2 = self.linear(&format!("{prefix}.attn2.to_out"), t, c, c, true, &ctx2);
        self.b.tap(format!("{tap}.attn2"), &ao2, t * c);
        self.b.free(n, ctx2);
        let h2 = self.b.add(t * c, &ao2, &h1);
        self.b.free(n, ao2);
        self.b.free(n, h1);

        // ---- GEGLU feed-forward ---------------------------------------------
        let inner = 4 * c;
        let ni = (t as u64) * (inner as u64);
        let n3 = self.layernorm(&format!("{prefix}.norm3"), t, c, &h2);
        self.b.tap(format!("{tap}.norm3"), &n3, t * c);
        let hidden = self.linear(&format!("{prefix}.ff.hidden"), t, c, inner, true, &n3);
        let gate = self.linear(&format!("{prefix}.ff.gate"), t, c, inner, true, &n3);
        self.b.free(n, n3);
        let act = self.b.act(ni);
        let g = self.b.gpu();
        self.b.push_step(g.step(K_GELU_ERF, &[&gate, &act], &[t * inner], t * inner));
        self.b.free(ni, gate);
        let gated = self.b.act(ni);
        let g = self.b.gpu();
        self.b.push_step(g.step(K_MUL, &[&hidden, &act, &gated], &[t * inner], t * inner));
        self.b.tap(format!("{tap}.ff_geglu"), &gated, t * inner);
        self.b.free(ni, hidden);
        self.b.free(ni, act);
        let ff = self.linear(&format!("{prefix}.ff.out"), t, inner, c, true, &gated);
        self.b.tap(format!("{tap}.ff"), &ff, t * c);
        self.b.free(ni, gated);
        let out = self.b.add(t * c, &ff, &h2);
        self.b.free(n, ff);
        self.b.free(n, h2);
        self.b.tap(tap.to_string(), &out, t * c);
        out
    }

    /// Multi-head self-attention over a fused `[T, 3c]` buffer.
    pub fn self_attention(&mut self, heads: u32, hd: u32, c: u32, t: u32, qkv: &DeviceBuffer, ctx: &DeviceBuffer) {
        let g = self.b.gpu();
        let mut steps: Vec<Step> = Vec::new();
        if self.coop {
            block::flash_bidir_fwd(
                g,
                self.flash,
                heads,
                hd,
                c,
                qkv,
                3 * c,
                0,
                c,
                2 * c,
                ctx,
                &[(0, t)],
                &mut steps,
            );
        } else {
            let (scores, probs) = self.slab.clone().expect("non-cooperative device allocates the slabs");
            let a = block::Bidir {
                b: 1,
                t,
                n_heads: heads,
                head_dim: hd,
                stride: 3 * c,
                q_off: 0,
                k_off: c,
                v_off: 2 * c,
            };
            // Forward only: the backward slots are `usize::MAX` sentinels so a
            // future reverse pass panics loudly instead of dispatching a
            // silently-wrong pipeline (the `clip::EvaVision` convention).
            let (k_scores, k_softmax, k_apply) = vae::blocks::ATTN_BIDIR_SLOTS;
            let ids = block::BidirIds {
                scores: k_scores,
                softmax: k_softmax,
                apply: k_apply,
                dscores: usize::MAX,
                dv: usize::MAX,
                dq: usize::MAX,
                dk: usize::MAX,
            };
            steps.extend(block::bidir_fwd(g, &ids, &a, qkv, &scores, &probs, ctx));
        }
        for s in steps {
            self.b.push_step(s);
        }
    }

    /// Cross-attention: `T` queries over `t_enc` text keys/values.
    #[allow(clippy::too_many_arguments)]
    pub fn cross_attention(
        &mut self,
        heads: u32,
        hd: u32,
        c: u32,
        t: u32,
        q: &DeviceBuffer,
        kv: &DeviceBuffer,
        ctx: &DeviceBuffer,
    ) {
        let te = self.t_enc;
        let slab = (heads as u64) * (t as u64) * (te as u64);
        let scores = self.b.act(slab);
        let probs = self.b.act(slab);
        let g = self.b.gpu();
        // `attn_scores_cross`  Params: [bsz, heads, t_dec, t_enc, head_dim, q_stride, kv_stride, q_off, k_off]
        self.b.push_step(g.step(
            K_XSCORES,
            &[q, kv, &scores],
            &[1, heads, t, te, hd, c, 2 * c, 0, 0],
            heads * t * te,
        ));
        // `attn_softmax_cross` Params: [bsz, heads, t_dec, t_enc]
        self.b.push_step(g.step(K_XSOFTMAX, &[&scores, &probs], &[1, heads, t, te], heads * t));
        // `attn_apply_cross`   Params: [bsz, heads, t_dec, t_enc, head_dim, kv_stride, v_off, d_model]
        self.b.push_step(g.step(
            K_XAPPLY,
            &[&probs, kv, ctx],
            &[1, heads, t, te, hd, 2 * c, c, c],
            heads * t * hd,
        ));
        self.b.free(slab, scores);
        self.b.free(slab, probs);
    }

    /// `Transformer2DModel` (`use_linear_projection = true`).
    #[allow(clippy::too_many_arguments)]
    pub fn transformer(
        &mut self,
        prefix: &str,
        tap: &str,
        cfg: &UNetConfig,
        level: usize,
        h: u32,
        w: u32,
        enc: &DeviceBuffer,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let c = cfg.block_out_channels[level];
        let (heads, hd) = (cfg.attention_heads[level], cfg.head_dim(level));
        let t = h * w;
        let n = (t as u64) * (c as u64);

        self.b.set_eps(TRANSFORMER_NORM_EPS);
        let norm = self.b.gn(&format!("{prefix}.norm"), c, h, w, x);
        self.b.set_eps(cfg.norm_eps);
        self.b.tap(format!("{tap}.norm"), &norm, c * t);

        // NCHW -> [T, C] rows, THEN proj_in — diffusers permutes first when
        // `use_linear_projection`, and the other order is a different graph
        // when proj_in is a conv.
        let rows = self.b.nchw_to_rows(c, t, &norm);
        self.b.free(n, norm);
        assert!(cfg.use_linear_projection, "conv proj_in/proj_out is not implemented (SD 1.5)");
        let mut hh = self.linear(&format!("{prefix}.proj_in"), t, c, c, true, &rows);
        self.b.tap(format!("{tap}.proj_in"), &hh, t * c);
        self.b.free(n, rows);

        for k in 0..cfg.transformer_layers_per_block[level] {
            let next = self.transformer_block(
                &format!("{prefix}.transformer_blocks.{k}"),
                &format!("{tap}.tb{k}"),
                c,
                t,
                heads,
                hd,
                cfg.cross_attention_dim,
                enc,
                &hh,
            );
            self.b.free(n, hh);
            hh = next;
        }

        let po = self.linear(&format!("{prefix}.proj_out"), t, c, c, true, &hh);
        self.b.tap(format!("{tap}.proj_out"), &po, t * c);
        self.b.free(n, hh);
        let chw = self.b.rows_to_nchw(c, t, &po);
        self.b.free(n, po);
        let out = self.b.add(c * t, x, &chw);
        self.b.free(n, chw);
        self.b.tap(tap.to_string(), &out, c * t);
        out
    }

    /// `torch.cat([hidden, skip], dim=1)` — the up path's skip join.
    pub fn concat_channels(
        &mut self,
        ca: u32,
        cb: u32,
        h: u32,
        w: u32,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
    ) -> DeviceBuffer {
        self.b.concat(ca, cb, h, w, a, b)
    }
}

/// A recorded SDXL UNet at one latent resolution and one text-token count.
pub struct Unet {
    gpu: Gpu,
    cfg: UNetConfig,
    hw: (u32, u32),
    t_enc: u32,
    sample_in: DeviceBuffer,
    enc_in: DeviceBuffer,
    temb_in: DeviceBuffer,
    aug_in: DeviceBuffer,
    /// Control-residual inputs, one per injection point in `control_shapes`
    /// order. Empty unless the graph was built by [`Unet::new_controlled`].
    control_in: Vec<DeviceBuffer>,
    control_shapes: Vec<(u32, u32, u32)>,
    /// Cross-attention sites the recorder emitted — the number an injection
    /// adapter must serve. Counted from the graph, never from a formula.
    sites: usize,
    out: DeviceBuffer,
    steps: Vec<Step>,
    taps: Vec<(String, DeviceBuffer, usize)>,
}

impl Unet {
    /// How many cross-attention sites this backbone recorded — what
    /// [`Unet::new_injected`]'s adapter must serve.
    pub fn cross_attention_sites(&self) -> usize {
        self.sites
    }

    /// Record the graph for a `h × w` latent and `t_enc` text tokens.
    ///
    /// `taps` records every stage output for the parity ladder; it pins buffers
    /// and therefore disables the activation pool, so a production build passes
    /// `false` (and `crates/unet/tests/parity.rs` gates the two against each
    /// other bit-for-bit).
    pub fn new(gpu: Gpu, cfg: UNetConfig, tensors: &Tensors, h: u32, w: u32, t_enc: u32, taps: bool) -> Unet {
        Unet::new_controlled(gpu, cfg, tensors, h, w, t_enc, taps, false)
    }

    /// [`Unet::new`], plus (when `control`) a set of device inputs the graph
    /// adds as **control residuals** — the consumer half of the ControlNet
    /// seam. One buffer per entry of [`UNetConfig::skip_stack`], plus one for
    /// the mid block, in that order; write them with [`Unet::run_with_control`]
    /// and read their shapes from [`Unet::control_shapes`].
    ///
    /// The residual is added to the SKIP COPY, not to the running hidden state
    /// — diffusers adds it to `down_block_res_samples`, which only the up path
    /// consumes. Adding it to `hh` as well would double-count it through the
    /// next down block, and every shape still matches, so nothing would fail.
    /// The mid residual *is* added to the running state, because there the
    /// residual list and the hidden state are the same tensor.
    #[allow(clippy::too_many_arguments)]
    /// [`Unet::new_controlled`], plus an adapter that contributes an extra term
    /// to every cross-attention context.
    ///
    /// This is a DIFFERENT seam from `control`, and deliberately so. A control
    /// residual is a pre-supplied device input, because it depends on nothing
    /// the backbone computes. A cross-attention adapter attends the per-site
    /// QUERY, which only exists inside the transformer block — so it is called
    /// during recording, at the point that tensor is live, through
    /// `model::attninject::CrossAttnInject`.
    ///
    /// The adapter's site count is checked against the number of cross-attention
    /// blocks the recorder ACTUALLY emitted — not against a formula over the
    /// config, which could disagree with the graph. A checkpoint built for a
    /// different UNet therefore fails at construction with two numbers rather
    /// than mid-forward with a shape.
    #[allow(clippy::too_many_arguments)]
    pub fn new_injected(
        gpu: Gpu,
        cfg: UNetConfig,
        tensors: &Tensors,
        h: u32,
        w: u32,
        t_enc: u32,
        taps: bool,
        control: bool,
        inject: &dyn model::attninject::CrossAttnInject,
    ) -> Unet {
        // The adapter dispatches on OUR device, so it can only use kernels this
        // Gpu was built with. Check now, naming the missing one, rather than
        // letting the adapter's own resolve panic mid-record.
        for (name, _) in inject.kernels() {
            assert!(
                gpu.kernel_index(name).is_some(),
                "unet: the injection adapter needs the `{name}` kernel, but this Gpu was not built with it — \
                 construct it from the union of unet::KERNELS and the adapter's (see model::attninject)"
            );
        }
        Unet::build(gpu, cfg, tensors, h, w, t_enc, taps, control, Some(inject))
    }

    pub fn new_controlled(
        gpu: Gpu,
        cfg: UNetConfig,
        tensors: &Tensors,
        h: u32,
        w: u32,
        t_enc: u32,
        taps: bool,
        control: bool,
    ) -> Unet {
        Unet::build(gpu, cfg, tensors, h, w, t_enc, taps, control, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        gpu: Gpu,
        cfg: UNetConfig,
        tensors: &Tensors,
        h: u32,
        w: u32,
        t_enc: u32,
        taps: bool,
        control: bool,
        inject: Option<&dyn model::attninject::CrossAttnInject>,
    ) -> Unet {
        let levels = cfg.levels();
        let scale = 1u32 << (levels - 1);
        assert!(
            h.is_multiple_of(scale) && w.is_multiple_of(scale),
            "unet: latent {h}x{w} is not a multiple of the {scale}x downscale"
        );
        let c0 = cfg.block_out_channels[0];
        let te = cfg.time_embed_dim;

        let sample_in = gpu.storage((cfg.in_channels * h * w) as u64);
        let enc_in = gpu.storage((t_enc * cfg.cross_attention_dim) as u64);
        let temb_in = gpu.storage(c0 as u64);
        let aug_in = gpu.storage(cfg.projection_class_embeddings_input_dim as u64);

        // Slabs, sized for the worst level that actually records a transformer.
        let s_words = attn_slab_words(&cfg, h, w, true);
        let mut r = Rec::new(&gpu, &cfg, tensors, t_enc, s_words, taps);
        r.inject = inject;

        r.conditioning(&cfg, &temb_in, &aug_in);

        // ---- conv_in + down path ---------------------------------------------
        let cin = r.b.conv("conv_in", cfg.in_channels, c0, 3, 1, h, w, &sample_in);
        r.b.tap("conv_in".into(), &cin, c0 * h * w);
        let (mut hh, mut skips, mut ch, mut cw) = r.down_path(&cfg, h, w, &enc_in, &cin);
        let mut prev = *cfg.block_out_channels.last().expect("levels >= 1");

        // ---- mid --------------------------------------------------------------
        hh = r.mid_block(&cfg, ch, cw, &enc_in, &hh);

        // ---- control residuals -------------------------------------------------
        let control_shapes: Vec<(u32, u32, u32)> =
            skips.iter().map(|&(_, c, sh, sw)| (c, sh, sw)).chain([(prev, ch, cw)]).collect();
        let control_in: Vec<DeviceBuffer> = if control {
            control_shapes.iter().map(|&(c, sh, sw)| gpu.storage((c * sh * sw) as u64)).collect()
        } else {
            Vec::new()
        };
        if control {
            for (k, (buf, c, sh, sw)) in skips.iter_mut().enumerate() {
                let n = *c * *sh * *sw;
                *buf = r.b.add(n, buf, &control_in[k]);
            }
            let n = prev * ch * cw;
            hh = r.b.add(n, &hh, &control_in[skips.len()]);
        }

        // ---- up path -----------------------------------------------------------
        for i in 0..levels {
            let level = levels - 1 - i;
            let cout = cfg.block_out_channels[level];
            for j in 0..=cfg.layers_per_block {
                let (skip, cskip, sh, sw) =
                    skips.pop().expect("the skip stack is sized by UNetConfig::skip_stack");
                assert_eq!((sh, sw), (ch, cw), "up{i}.resnet{j}: skip is {sh}x{sw}, hidden is {ch}x{cw}");
                let cin = prev + cskip;
                let cat = r.concat_channels(prev, cskip, ch, cw, &hh, &skip);
                r.b.tap(format!("up{i}.cat{j}"), &cat, cin * ch * cw);
                let next = r.resnet(
                    &format!("up_blocks.{i}.resnets.{j}"),
                    &format!("up{i}.resnet{j}"),
                    cin,
                    cout,
                    ch,
                    cw,
                    te,
                    &cat,
                );
                r.b.free((cin as u64) * (ch as u64) * (cw as u64), cat);
                hh = next;
                if cfg.up_block_types[i] == BlockKind::CrossAttn {
                    hh = r.transformer(
                        &format!("up_blocks.{i}.attentions.{j}"),
                        &format!("up{i}.attn{j}"),
                        &cfg,
                        level,
                        ch,
                        cw,
                        &enc_in,
                        &hh,
                    );
                }
                prev = cout;
            }
            if i + 1 < levels {
                let up = r.b.upsample(cout, ch, cw, &hh);
                r.b.tap(format!("up{i}.nearest2x"), &up, cout * 4 * ch * cw);
                ch *= 2;
                cw *= 2;
                hh = r.b.conv(&format!("up_blocks.{i}.upsamplers.0.conv"), cout, cout, 3, 1, ch, cw, &up);
                r.b.free((cout as u64) * (ch as u64) * (cw as u64), up);
                r.b.tap(format!("up{i}.upsample0"), &hh, cout * ch * cw);
            }
        }
        assert!(skips.is_empty(), "unet: {} skip tensors left unconsumed", skips.len());

        // ---- head ---------------------------------------------------------------
        let no = r.b.gn("conv_norm_out", c0, ch, cw, &hh);
        r.b.tap("conv_norm_out".into(), &no, c0 * ch * cw);
        let sa = r.b.silu(c0 * ch * cw, &no);
        let out = r.b.conv("conv_out", c0, cfg.out_channels, 3, 1, ch, cw, &sa);
        r.b.tap("conv_out".into(), &out, cfg.out_channels * ch * cw);

        // The graph is the authority on how many cross-attention sites exist —
        // a formula over the config could disagree with what was recorded.
        if let Some(inj) = inject {
            assert_eq!(
                r.site,
                inj.sites(),
                "unet: recorded {} cross-attention sites but the adapter serves {}",
                r.site,
                inj.sites()
            );
        }
        let sites = r.site;
        let (steps, taps) = r.into_blocks().finish();
        Unet {
            sites,
            gpu,
            cfg,
            hw: (h, w),
            t_enc,
            sample_in,
            enc_in,
            temb_in,
            aug_in,
            control_in,
            control_shapes,
            out,
            steps,
            taps,
        }
    }

    /// `(channels, h, w)` of every control-residual injection point, in the
    /// order [`Unet::run_with_control`] expects: the whole
    /// [`UNetConfig::skip_stack`] (finest first), then the mid block.
    /// Available whether or not the graph was built with `control`.
    pub fn control_shapes(&self) -> &[(u32, u32, u32)] {
        &self.control_shapes
    }

    /// Does the RECORDED graph read control residuals (i.e. was it built by
    /// [`Unet::new_controlled`] with `control = true`)? [`Unet::control_shapes`]
    /// describes the points either way, so this is the question a caller has to
    /// ask before [`Unet::run_with_control`].
    pub fn accepts_control(&self) -> bool {
        !self.control_in.is_empty()
    }

    /// [`Unet::run`] with a ControlNet's residuals added at every injection
    /// point. Requires a graph built by [`Unet::new_controlled`] with
    /// `control = true`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_with_control(
        &self,
        sample: &[f32],
        timestep: f32,
        enc: &[f32],
        pooled: &[f32],
        time_ids: &[f32],
        control: &[Vec<f32>],
    ) -> Vec<f32> {
        assert!(!self.control_in.is_empty(), "unet: graph was not built with control inputs");
        assert_eq!(
            control.len(),
            self.control_in.len(),
            "unet: {} control residuals, graph has {} injection points",
            control.len(),
            self.control_in.len()
        );
        for (k, (buf, (c, h, w))) in self.control_in.iter().zip(&self.control_shapes).enumerate() {
            let want = (c * h * w) as usize;
            assert_eq!(control[k].len(), want, "unet: control residual {k} is {} values, want {want}", control[k].len());
            self.gpu.write_f32(buf, &control[k]);
        }
        self.run(sample, timestep, enc, pooled, time_ids)
    }

    pub fn config(&self) -> &UNetConfig {
        &self.cfg
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// One denoising evaluation.
    ///
    /// * `sample` — `[in_channels · H · W]`, NCHW, batch 1.
    /// * `timestep` — the discrete timestep (fractional values are allowed; the
    ///   sinusoid is defined on the reals and Euler schedules produce them).
    /// * `enc` — `[t_enc · cross_attention_dim]`, the CONCATENATED penultimate
    ///   hidden states of CLIP-L and OpenCLIP-bigG.
    /// * `pooled` — `[pooled_dim]`, bigG's projected pooled output.
    /// * `time_ids` — the six micro-conditioning values, in diffusers' order.
    pub fn run(&self, sample: &[f32], timestep: f32, enc: &[f32], pooled: &[f32], time_ids: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        let (h, w) = self.hw;
        assert_eq!(sample.len(), (c.in_channels * h * w) as usize, "unet: sample size");
        assert_eq!(enc.len(), (self.t_enc * c.cross_attention_dim) as usize, "unet: encoder_hidden_states size");
        assert_eq!(pooled.len(), c.pooled_dim() as usize, "unet: pooled text size");
        assert_eq!(time_ids.len(), N_TIME_IDS as usize, "unet: time_ids must be {N_TIME_IDS} values");

        let temb = model::hostmath::timestep_embedding(
            timestep,
            c.block_out_channels[0] as usize,
            c.flip_sin_to_cos,
            c.freq_shift as f64,
            10_000.0,
        );
        let aug = hostemb::added_cond(
            pooled,
            time_ids,
            c.addition_time_embed_dim,
            c.flip_sin_to_cos,
            c.freq_shift,
        );
        self.gpu.write_f32(&self.sample_in, sample);
        self.gpu.write_f32(&self.enc_in, enc);
        self.gpu.write_f32(&self.temb_in, &temb);
        self.gpu.write_f32(&self.aug_in, &aug);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, (c.out_channels * h * w) as usize)
    }

    /// A recorded stage output (only when the model was built with `taps`).
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        let (_, buf, len) = self.taps.iter().find(|(n, _, _)| n == name)?;
        Some(self.gpu.read(buf, *len))
    }

    pub fn tap_names(&self) -> Vec<&str> {
        self.taps.iter().map(|(n, _, _)| n.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{BlockKind, UNetConfig};

    /// Every appended slot holds the kernel its constant is named for, and no
    /// slot is left empty. `kernels_with` zero-fills, so an off-by-one in the
    /// `NEXT_SLOT + k` arithmetic produces an empty entry that only fails at
    /// pipeline-creation time, inside a 2158-step graph.
    #[test]
    fn appended_slots_hold_their_named_kernels() {
        for (slot, name) in [
            (super::K_LAYERNORM, "layernorm"),
            (super::K_MATMUL, "matmul"),
            (super::K_MATMUL_REG, "matmul_reg3"),
            (super::K_BIAS_ADD, "bias_add"),
            (super::K_GELU_ERF, "gelu_erf"),
            (super::K_MUL, "mul"),
            (super::K_XSCORES, "attn_scores_cross"),
            (super::K_XSOFTMAX, "attn_softmax_cross"),
            (super::K_XAPPLY, "attn_apply_cross"),
            (super::K_FLASH, "flash_attn_bidir"),
            (super::K_FLASH_SPLIT, "flash_attn_bidir_split"),
            (super::K_ADD_CHAN, "add_chan_bcast"),
        ] {
            assert_eq!(super::KERNELS[slot].0, name, "slot {slot}");
        }
        assert!(super::KERNELS.iter().all(|(n, s)| !n.is_empty() && !s.is_empty()));
    }

    /// The non-cooperative self-attention path indexes `vae::blocks`' own trio
    /// through [`vae::blocks::ATTN_BIDIR_SLOTS`]. Assert those slots really do
    /// name the bidir kernels — a wrong index there is invisible on a GPU
    /// (which takes the flash path) and dispatches the wrong pipeline on the
    /// CPU JIT.
    #[test]
    fn the_bidir_attention_trio_is_at_the_exported_slots() {
        let (s, m, a) = vae::blocks::ATTN_BIDIR_SLOTS;
        assert_eq!(super::KERNELS[s].0, "attn_scores_bidir");
        assert_eq!(super::KERNELS[m].0, "attn_softmax_bidir");
        assert_eq!(super::KERNELS[a].0, "attn_apply_bidir");
    }

    /// The self-attention slab is sized over the levels that actually record a
    /// transformer, not all of them. SDXL's level 0 has no attention and 16x
    /// the token count of level 1, so including it is an 8x over-allocation —
    /// 10.7 GB of never-bound buffer at a 128x128 latent.
    #[test]
    fn slab_is_sized_over_attention_levels_only() {
        let cfg = UNetConfig::sdxl_base();
        let (h, w) = (128u64, 128u64);
        let levels = cfg.levels();
        let mut used_max = 0u64;
        let mut all_max = 0u64;
        for l in 0..levels {
            let t = (h * w) >> (2 * l);
            let words = cfg.attention_heads[l] as u64 * t * t;
            all_max = all_max.max(words);
            if l == levels - 1
                || cfg.down_block_types[l] == BlockKind::CrossAttn
                || cfg.up_block_types[levels - 1 - l] == BlockKind::CrossAttn
            {
                used_max = used_max.max(words);
            }
        }
        assert_eq!(cfg.down_block_types[0], BlockKind::Plain);
        assert_eq!(cfg.up_block_types[levels - 1], BlockKind::Plain);
        assert_eq!(all_max / used_max, 8, "{all_max} vs {used_max}");
    }
}
