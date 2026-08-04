// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The VQGAN forward graph: encoder → codebook assignment → generator.
//!
//! Composition, not re-implementation. Every convolutional block comes from
//! the shared [`vae::blocks::Builder`] (conv / GroupNorm+SiLU resnet /
//! single-head spatial attention / asymmetric-pad strided downsample /
//! nearest-2× upsample) selected with [`BlockNames::vqgan`]; the codebook
//! search is the existing `vq_argmin` kernel dispatched through
//! [`wm_core::vq::Vq`]; the code lookup is the existing `embed` gather. This
//! crate adds no kernel and no block.
//!
//! Two deliberate departures from `crates/vae`'s `AutoencoderKL` schedule,
//! both verified against the reference source:
//!
//! 1. **The heads have no activation.** VQGAN is `GroupNorm → Conv2d`
//!    (`vqgan_arch.py:265` and `:313`); the diffusers VAE is
//!    `GroupNorm → SiLU → conv_out`. Reusing the VAE head would insert a
//!    spurious SiLU.
//! 2. **Attention is not mid-block-only.** At `attn_resolutions`, an
//!    `AttnBlock` follows *every* residual block, plus the mid triple — the
//!    encoder gets them at indices 17/19/21 and the generator at 2/5/7.
//!
//! Upsampling is nearest-2× + `Conv2d(k3,s1,p1)`, **not** a transposed
//! convolution, so `convtr2d` / `vision::blocks::ConvTranspose` are not on
//! this model's path.
//!
//! ## Buffer/step layout (SSA)
//!
//! Each stage writes a fresh buffer (the shape a hand-written backward will
//! cache). Two submits with one host round-trip between them, because the
//! assignment is an integer the `embed` gather needs as `u32` and because it
//! is the natural seam for CodeFormer, whose transformer *replaces* the argmin:
//!
//! ```text
//! encode_steps : img_in → enc blocks → z[emb,lh,lw] → z_flat[T,emb] → packed[2T]
//!   (host)     : packed → indices u32[T] → idx_in
//! decode_steps : [gather] idx_in,codebook → rows[T,emb] → z_q[emb,lh,lw]
//!                [gen]    z_q → gen blocks → out[3,H,W]
//! ```
//!
//! `decode_steps[gather_end..]` is the generator alone, so a caller that has a
//! latent already (CodeFormer's fused features) writes `z_q` and submits that
//! suffix.

use gpu_core::{DeviceBuffer, Gpu, Step};
use vae::blocks::{BlockNames, Builder, Tensors};

use crate::config::{Block, VqganConfig};

/// Kernel slots after the shared block set.
const K_VQ_ARGMIN: usize = vae::blocks::NEXT_SLOT;
const K_VQ_ARGMAX_DOT: usize = vae::blocks::NEXT_SLOT + 1;
const K_EMBED: usize = vae::blocks::NEXT_SLOT + 2;

/// This model's kernel set: the shared block kernels (slots `0..NEXT_SLOT`,
/// copied — never restated — by [`vae::blocks::kernels_with`]), then the two
/// VQ assignment kernels in [`wm_core::vq::Vq::kernel_sources`] order, then the
/// `embed` gather.
pub const KERNELS: [(&str, &str); vae::blocks::NEXT_SLOT + 3] = kernel_set();

const fn kernel_set() -> [(&'static str, &'static str); vae::blocks::NEXT_SLOT + 3] {
    let mut k = vae::blocks::kernels_with::<{ vae::blocks::NEXT_SLOT + 3 }>();
    k[K_VQ_ARGMIN] = ("vq_argmin", kernels::VQ_ARGMIN);
    k[K_VQ_ARGMAX_DOT] = ("vq_argmax_dot", kernels::VQ_ARGMAX_DOT);
    k[K_EMBED] = ("embed", kernels::EMBED);
    k
}

/// The shared VQ dispatch helper bound to this crate's slots.
fn vq() -> wm_core::vq::Vq {
    wm_core::vq::Vq { argmin: K_VQ_ARGMIN, argmax_dot: K_VQ_ARGMAX_DOT }
}

/// Record the nearest-codebook assignment: `rows[m,d]` against `cb[k,d]` →
/// `packed[2m]` (`packed[2i]` = `f32(argmin_k)`, `packed[2i+1]` = the minimum
/// squared L2 distance; ties resolve to the lowest `k`).
///
/// `vq_argmin`'s `Params` is `{ m, k, d }` with one invocation per query. This
/// is the ONE dispatch site — [`Vqgan`] and [`Codebook`] both come here.
pub fn record_assign(
    b: &mut Builder,
    cb: &DeviceBuffer,
    m: u32,
    k: u32,
    d: u32,
    rows: &DeviceBuffer,
) -> DeviceBuffer {
    let packed = b.act(2 * m as u64);
    let step = vq().step_argmin(b.gpu(), m, k, d, rows, cb, &packed);
    b.push_step(step);
    packed
}

/// Record the codebook gather: `idx[m]` (u32) → `rows[m,d]`. `embed`'s `Params`
/// is `{ d_model, seq_len }`, one invocation per output element, computing
/// `x[t,c] = emb[token[t], c]` — the reference's `get_codebook_feat` one-hot
/// matmul without materialising the one-hot.
pub fn record_lookup(
    b: &mut Builder,
    cb: &DeviceBuffer,
    m: u32,
    d: u32,
    idx: &DeviceBuffer,
) -> DeviceBuffer {
    let rows = b.act((m * d) as u64);
    let step = b.gpu().step(K_EMBED, &[idx, cb, &rows], &[d, m], m * d);
    b.push_step(step);
    rows
}

/// Split the packed `vq_argmin` output into indices and minimum distances.
fn unpack(packed: &[f32]) -> (Vec<u32>, Vec<f32>) {
    (wm_core::vq::indices(packed), packed.chunks_exact(2).map(|c| c[1]).collect())
}

/// The quantizer on its own: a resident codebook plus the assignment and
/// lookup graphs for a fixed query count.
///
/// CodeFormer's inference path never runs [`Vqgan::encode`]'s argmin — its
/// transformer predicts the code indices and calls `get_codebook_feat`, i.e.
/// [`Codebook::lookup`] — so the two halves are separately usable.
pub struct Codebook {
    gpu: Gpu,
    k: u32,
    d: u32,
    m: u32,
    z_in: DeviceBuffer,
    packed: DeviceBuffer,
    idx_in: DeviceBuffer,
    rows_out: DeviceBuffer,
    assign_steps: Vec<Step>,
    lookup_steps: Vec<Step>,
}

impl Codebook {
    /// `embedding` is `[k, d]` row-major (`quantize.embedding.weight`); `m` is
    /// the number of queries the graphs are built for.
    pub fn new(gpu: Gpu, embedding: &[f32], k: u32, d: u32, m: u32) -> Codebook {
        assert_eq!(
            embedding.len(),
            (k * d) as usize,
            "codebook has {} values, expected {}",
            embedding.len(),
            k * d
        );
        let cb = gpu.storage_init("quantize.embedding.weight", embedding);
        let z_in = gpu.storage((m * d) as u64);
        let idx_in = gpu.storage(m as u64);
        let empty = Tensors::new();

        let mut b = Builder::new(&gpu, &empty, 1e-6, 32, BlockNames::vqgan(), false);
        let packed = record_assign(&mut b, &cb, m, k, d, &z_in);
        let (assign_steps, _) = b.finish();

        let mut b = Builder::new(&gpu, &empty, 1e-6, 32, BlockNames::vqgan(), false);
        let rows_out = record_lookup(&mut b, &cb, m, d, &idx_in);
        let (lookup_steps, _) = b.finish();

        Codebook { gpu, k, d, m, z_in, packed, idx_in, rows_out, assign_steps, lookup_steps }
    }

    /// Assign `rows` (`[m, d]`, row-major) to the nearest code; returns
    /// `(indices, min squared distance)`.
    pub fn assign(&self, rows: &[f32]) -> (Vec<u32>, Vec<f32>) {
        let n = (self.m * self.d) as usize;
        assert_eq!(rows.len(), n, "codebook: {} values, expected {n}", rows.len());
        let bits: Vec<u32> = rows.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&self.z_in, &bits);
        self.gpu.submit(&[], &self.assign_steps);
        unpack(&self.gpu.read(&self.packed, 2 * self.m as usize))
    }

    /// Gather `indices` into `[m, d]` rows (`get_codebook_feat`).
    pub fn lookup(&self, indices: &[u32]) -> Vec<f32> {
        assert_eq!(indices.len(), self.m as usize, "codebook: {} indices", indices.len());
        assert!(
            indices.iter().all(|&i| i < self.k),
            "codebook: index out of range for {} codes",
            self.k
        );
        self.gpu.write(&self.idx_in, indices);
        self.gpu.submit(&[], &self.lookup_steps);
        self.gpu.read(&self.rows_out, (self.m * self.d) as usize)
    }
}

/// Result of one `image → codes → image` pass.
pub struct Reconstruction {
    /// Nearest codebook index per latent position, row-major over `[lh, lw]`.
    pub indices: Vec<u32>,
    /// The squared L2 distance to that code, same order.
    pub min_dist: Vec<f32>,
    /// Reconstructed image `[out_channels, H, W]`, row-major.
    pub image: Vec<f32>,
}

/// A VQGAN autoencoder graph for a fixed input size, weights resident.
pub struct Vqgan {
    gpu: Gpu,
    cfg: VqganConfig,
    /// Input image size and the latent grid it produces.
    hw: (u32, u32),
    lhw: (u32, u32),
    encode_steps: Vec<Step>,
    decode_steps: Vec<Step>,
    /// Index in `decode_steps` where the generator starts (the codebook gather
    /// occupies everything before it).
    gather_end: usize,
    img_in: DeviceBuffer,
    z: DeviceBuffer,
    packed: DeviceBuffer,
    idx_in: DeviceBuffer,
    z_q: DeviceBuffer,
    out: DeviceBuffer,
    taps: Vec<(String, DeviceBuffer, usize)>,
}

impl Vqgan {
    /// Build both graphs for an input image `[in_channels, h, w]` on `gpu`
    /// (which MUST have been created with [`KERNELS`]) and upload the weights.
    ///
    /// `taps` records every block output for stage-by-stage parity; it pins
    /// every activation (the builder's buffer pool is disabled), so leave it
    /// off outside tests.
    pub fn new(cfg: VqganConfig, tensors: &Tensors, h: u32, w: u32, gpu: Gpu, taps: bool) -> Vqgan {
        let scale = cfg.downscale();
        assert!(
            h.is_multiple_of(scale) && w.is_multiple_of(scale),
            "vqgan: input {h}x{w} is not a multiple of the {scale}x downscale"
        );
        let (lh, lw) = (h / scale, w / scale);
        let t = lh * lw;
        let emb = cfg.emb_dim;

        // The codebook is read by both graphs; upload it once.
        let codebook = {
            let name = "quantize.embedding.weight";
            let (_, data) = tensors.get(name).unwrap_or_else(|| panic!("vqgan: missing {name}"));
            gpu.storage_init(name, data)
        };

        // ---- encoder + assignment ------------------------------------------
        let img_in = gpu.storage((cfg.in_channels * h * w) as u64);
        let mut b =
            Builder::new(&gpu, tensors, cfg.norm_eps, cfg.norm_groups, BlockNames::vqgan(), taps);
        let z = run_blocks(&mut b, "encoder", &cfg.encoder_blocks(), 0, h, w, &img_in).0;

        // z[emb, lh, lw] → z_flat[T, emb]: the quantizer's `z.permute(0,2,3,1)
        // .view(-1, emb_dim)`, which is exactly the NCHW→NLC permutation.
        let z_flat = b.nchw_to_rows(emb, t, &z);
        b.tap("z_flat".into(), &z_flat, emb * t);
        let packed = record_assign(&mut b, &codebook, t, cfg.codebook_size, emb, &z_flat);
        b.free((emb * t) as u64, z_flat);
        let (encode_steps, mut all_taps) = b.finish();

        // ---- codebook gather + generator ------------------------------------
        let idx_in = gpu.storage(t as u64);
        let mut b =
            Builder::new(&gpu, tensors, cfg.norm_eps, cfg.norm_groups, BlockNames::vqgan(), taps);
        let rows = record_lookup(&mut b, &codebook, t, emb, &idx_in);
        let z_q = b.rows_to_nchw(emb, t, &rows);
        b.free((t * emb) as u64, rows);
        b.tap("z_q".into(), &z_q, emb * t);
        let gather_end = b.n_steps();
        let (out, (oh, ow)) = run_blocks(&mut b, "generator", &cfg.generator_blocks(), 0, lh, lw, &z_q);
        assert_eq!((oh, ow), (h, w), "vqgan: generator output {oh}x{ow} != input {h}x{w}");
        let (decode_steps, dec_taps) = b.finish();
        all_taps.extend(dec_taps);

        Vqgan {
            gpu,
            cfg,
            hw: (h, w),
            lhw: (lh, lw),
            encode_steps,
            decode_steps,
            gather_end,
            img_in,
            z,
            packed,
            idx_in,
            z_q,
            out,
            taps: all_taps,
        }
    }

    pub fn config(&self) -> &VqganConfig {
        &self.cfg
    }

    /// The latent grid `(lh, lw)` for the configured input size.
    pub fn latent_size(&self) -> (u32, u32) {
        self.lhw
    }

    /// Encode an image `[in_channels·H·W]` (row-major NCHW, batch 1) and assign
    /// each latent position to its nearest code. Returns `(indices, min_dist)`;
    /// the continuous latent stays on the device (see [`Vqgan::latent`]).
    pub fn encode(&self, image: &[f32]) -> (Vec<u32>, Vec<f32>) {
        let n = (self.cfg.in_channels * self.hw.0 * self.hw.1) as usize;
        assert_eq!(image.len(), n, "vqgan: image has {} values, expected {n}", image.len());
        let bits: Vec<u32> = image.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&self.img_in, &bits);
        self.gpu.submit(&[], &self.encode_steps);
        let t = (self.lhw.0 * self.lhw.1) as usize;
        unpack(&self.gpu.read(&self.packed, 2 * t))
    }

    /// Gather `indices` from the codebook and run the generator.
    pub fn decode(&self, indices: &[u32]) -> Vec<f32> {
        let t = (self.lhw.0 * self.lhw.1) as usize;
        assert_eq!(indices.len(), t, "vqgan: {} indices, expected {t}", indices.len());
        // The `embed` gather indexes `codebook[idx * d + c]` with NO bounds
        // check — brain compiles its shaders with `ShaderRuntimeChecks`
        // *unchecked*, so an out-of-range code reads past the buffer instead of
        // trapping. `decode` is public and the CodeFormer follow-up feeds it
        // transformer-predicted indices, so validate here exactly as
        // [`Codebook::lookup`] does.
        if let Some(&bad) = indices.iter().find(|&&i| i >= self.cfg.codebook_size) {
            panic!("vqgan: code index {bad} out of range for {} codes", self.cfg.codebook_size);
        }
        self.gpu.write(&self.idx_in, indices);
        self.gpu.submit(&[], &self.decode_steps);
        self.gpu.read(&self.out, self.out_len())
    }

    /// Run the generator on a caller-supplied latent `[emb_dim·lh·lw]`,
    /// skipping the codebook gather — the seam CodeFormer's controllable
    /// feature transformation plugs into.
    pub fn generate(&self, z_q: &[f32]) -> Vec<f32> {
        let n = (self.cfg.emb_dim * self.lhw.0 * self.lhw.1) as usize;
        assert_eq!(z_q.len(), n, "vqgan: latent has {} values, expected {n}", z_q.len());
        let bits: Vec<u32> = z_q.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&self.z_q, &bits);
        self.gpu.submit(&[], &self.decode_steps[self.gather_end..]);
        self.gpu.read(&self.out, self.out_len())
    }

    /// Full `image → codes → image` pass.
    pub fn reconstruct(&self, image: &[f32]) -> Reconstruction {
        let (indices, min_dist) = self.encode(image);
        let image = self.decode(&indices);
        Reconstruction { indices, min_dist, image }
    }

    /// The continuous encoder output `[emb_dim·lh·lw]` from the last
    /// [`Vqgan::encode`].
    pub fn latent(&self) -> Vec<f32> {
        let n = (self.cfg.emb_dim * self.lhw.0 * self.lhw.1) as usize;
        self.gpu.read(&self.z, n)
    }

    /// The quantized latent `[emb_dim·lh·lw]` from the last [`Vqgan::decode`].
    pub fn quantized(&self) -> Vec<f32> {
        let n = (self.cfg.emb_dim * self.lhw.0 * self.lhw.1) as usize;
        self.gpu.read(&self.z_q, n)
    }

    fn out_len(&self) -> usize {
        (self.cfg.out_channels * self.hw.0 * self.hw.1) as usize
    }

    /// Read a recorded intermediate by name (`encoder.blocks.7`,
    /// `generator.blocks.2.norm1`, `z_flat`, `z_q`, …). `None` unless the model
    /// was built with `taps = true`.
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        self.taps.iter().find(|(n, _, _)| n == name).map(|(_, b, len)| self.gpu.read(b, *len))
    }

    /// Every recorded tap name (parity-test diagnostics).
    pub fn tap_names(&self) -> Vec<&str> {
        self.taps.iter().map(|(n, _, _)| n.as_str()).collect()
    }

    /// The device the graphs were built on (profiling / benches).
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The recorded encode and decode dispatch sequences (profiling / benches).
    pub fn steps(&self) -> (&[Step], &[Step]) {
        (&self.encode_steps, &self.decode_steps)
    }
}

/// Record a **contiguous run** of a flat `nn.ModuleList` of [`Block`]s, tapping
/// each block's output under `{net}.blocks.{start + i}`. Returns the final
/// buffer and its spatial size.
///
/// `start` is the global index of `blocks[0]`, so a caller that needs to reach
/// *between* two blocks records the list in segments and keeps the checkpoint's
/// positional tensor names correct. That is how `crates/restore` walks the same
/// encoder/generator with CodeFormer's feature taps and controllable-feature
/// transformation spliced in, rather than owning a second copy of this loop.
/// [`Vqgan`] passes `start = 0` and the whole list.
///
/// Buffer ownership: `input` and the returned buffer are the CALLER's — neither
/// is returned to the builder's activation pool, so a segment's output stays
/// valid for as long as the caller needs it (CodeFormer holds four encoder
/// features live across the entire generator).
pub fn run_blocks(
    b: &mut Builder,
    net: &str,
    blocks: &[Block],
    start: usize,
    h: u32,
    w: u32,
    input: &DeviceBuffer,
) -> (DeviceBuffer, (u32, u32)) {
    let (mut hh, mut ww) = (h, w);
    let mut x = input.clone();
    let mut xlen = 0u64; // length of `x` when we own it (0 = caller-owned)
    for (i, blk) in blocks.iter().enumerate() {
        let p = format!("{net}.blocks.{}", start + i);
        // `resnet` and `attn` tap their own output under `p`; the others do not.
        let (nx, self_tapped) = match *blk {
            Block::Conv { cin, cout } => (b.conv(&p, cin, cout, 3, 1, hh, ww, &x), false),
            Block::Res { cin, cout } => (b.resnet(&p, cin, cout, hh, ww, &x), true),
            Block::Attn { c } => (b.attn(&p, c, hh, ww, &x), true),
            Block::Down { c } => {
                let y = b.conv_down(&format!("{p}.conv"), c, hh, ww, &x);
                hh /= 2;
                ww /= 2;
                (y, false)
            }
            Block::Up { c } => {
                let up = b.upsample(c, hh, ww, &x);
                hh *= 2;
                ww *= 2;
                let y = b.conv(&format!("{p}.conv"), c, c, 3, 1, hh, ww, &up);
                b.free((c * hh * ww) as u64, up);
                (y, false)
            }
            // The VQGAN head is GroupNorm → Conv2d: no activation between them.
            Block::Norm { c } => (b.gn(&p, c, hh, ww, &x), false),
        };
        let prev = std::mem::replace(&mut x, nx);
        if xlen != 0 {
            b.free(xlen, prev);
        }
        xlen = (blk.out_channels() * hh * ww) as u64;
        if !self_tapped {
            b.tap(p, &x, xlen as u32);
        }
    }
    (x, (hh, ww))
}

#[cfg(test)]
mod tests {
    /// The two VQ slots must stay the pair `wm_core::vq::Vq` expects, in its
    /// order — a swapped pair is silently wrong, not a crash.
    #[test]
    fn vq_slots_match_the_shared_kernel_sources() {
        let src = wm_core::vq::Vq::kernel_sources();
        assert_eq!(super::KERNELS[super::K_VQ_ARGMIN], src[0]);
        assert_eq!(super::KERNELS[super::K_VQ_ARGMAX_DOT], src[1]);
    }

    /// The shared block builder addresses slots `0..NEXT_SLOT` by position.
    #[test]
    fn shared_block_slots_are_copied_verbatim() {
        assert_eq!(super::KERNELS[..vae::blocks::NEXT_SLOT], vae::blocks::KERNELS[..]);
        assert_eq!(super::KERNELS[super::K_EMBED].0, "embed");
    }
}
