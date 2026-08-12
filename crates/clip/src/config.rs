// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reference configurations for the CLIP encoder family, plus the canonical
//! tensor manifests the importers validate against.
//!
//! Three consumers, one graph each:
//!   * **CLIP-L** (SDXL `text_encoder`, FLUX.1's pooled conditioning) - 12x768,
//!     `quick_gelu`.
//!   * **OpenCLIP-bigG/14** (SDXL `text_encoder_2`) - 32x1280, exact-erf `gelu`,
//!     plus a `text_projection`.
//!   * **EVA02-CLIP-L/14@336** (PuLID's identity tower) - the image tower.
//!
//! Every number here was read off the reference `config.json` / checkpoint and
//! is asserted against `testdata/clip/manifest.json` by the parity test.

/// MLP activation. CLIP-L uses OpenAI's sigmoid approximation, bigG the exact
/// erf form - they differ by ~1e-2 and are NOT interchangeable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TextAct {
    /// `x * sigmoid(1.702 x)` (`quick_gelu.wgsl`).
    QuickGelu,
    /// `0.5 x (1 + erf(x/sqrt 2))` (`gelu_erf.wgsl`) - transformers' `"gelu"`.
    GeluErf,
}

/// One CLIP text tower.
#[derive(Clone, Debug)]
pub struct ClipTextConfig {
    pub hidden: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub heads: u32,
    pub max_positions: u32,
    pub vocab: u32,
    pub act: TextAct,
    pub eps: f32,
    /// `text_projection` output width, when the checkpoint carries one
    /// (`CLIPTextModelWithProjection`). `None` = no projection (CLIP-L as SDXL
    /// and FLUX.1 consume it).
    pub projection: Option<u32>,
    /// BOS / EOS ids written by the tokenizer.
    pub bos_id: u32,
    pub eos_id: u32,
    /// Padding id - a property of the *encoder*, not the vocabulary: SDXL's
    /// `tokenizer` pads with `<|endoftext|>` (49407) and `tokenizer_2` pads with
    /// `"!"` (0). Both share vocab.json/merges.txt.
    pub pad_id: u32,
}

impl ClipTextConfig {
    /// SDXL `text_encoder` = OpenAI CLIP ViT-L/14's text tower.
    pub fn clip_l() -> ClipTextConfig {
        ClipTextConfig {
            hidden: 768,
            intermediate: 3072,
            layers: 12,
            heads: 12,
            max_positions: 77,
            vocab: 49408,
            act: TextAct::QuickGelu,
            eps: 1e-5,
            projection: None,
            bos_id: 49406,
            eos_id: 49407,
            pad_id: 49407,
        }
    }

    /// SDXL `text_encoder_2` = OpenCLIP-bigG/14's text tower, as HF's
    /// `CLIPTextModelWithProjection` (q/k/v already split; an open_clip-native
    /// checkpoint carries a FUSED `in_proj_weight` and must be split at import).
    pub fn openclip_bigg() -> ClipTextConfig {
        ClipTextConfig {
            hidden: 1280,
            intermediate: 5120,
            layers: 32,
            heads: 20,
            max_positions: 77,
            vocab: 49408,
            act: TextAct::GeluErf,
            eps: 1e-5,
            projection: Some(1280),
            bos_id: 49406,
            eos_id: 49407,
            pad_id: 0,
        }
    }

    pub fn head_dim(&self) -> u32 {
        self.hidden / self.heads
    }

    /// The 0-based encoder layer whose OUTPUT is diffusers' `hidden_states[-2]`
    /// - the sequence embedding SDXL actually conditions on.
    ///
    /// Settled empirically in step 1 and re-asserted by the parity test:
    /// `hidden_states[0]` is the embedding output, `hidden_states[k]` is the
    /// output of `encoder.layers[k-1]`, and the whole tuple is **pre**
    /// `final_layer_norm`. So `hidden_states[-2] == hidden_states[layers-1] ==
    /// output of layers[layers-2]`, and it is **not** layer-normed:
    ///   * bigG (32 layers) -> layer **30**; layer 31 and the final LayerNorm
    ///     are unused for the sequence embedding.
    ///   * CLIP-L (12 layers) -> layer **10**.
    pub fn penultimate_layer(&self) -> u32 {
        self.layers - 2
    }

    /// Canonical brain-side tensor manifest: `(name, shape)` for every
    /// parameter the model binds, in a stable order. The importer validates
    /// **both** directions against this.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let h = self.hidden as usize;
        let i = self.intermediate as usize;
        let mut v: Vec<(String, Vec<usize>)> = vec![
            ("tok.weight".into(), vec![self.vocab as usize, h]),
            ("pos.weight".into(), vec![self.max_positions as usize, h]),
        ];
        for l in 0..self.layers {
            let p = format!("blocks.{l}");
            v.push((format!("{p}.ln1.weight"), vec![h]));
            v.push((format!("{p}.ln1.bias"), vec![h]));
            v.push((format!("{p}.qkv.weight"), vec![3 * h, h]));
            v.push((format!("{p}.qkv.bias"), vec![3 * h]));
            v.push((format!("{p}.proj.weight"), vec![h, h]));
            v.push((format!("{p}.proj.bias"), vec![h]));
            v.push((format!("{p}.ln2.weight"), vec![h]));
            v.push((format!("{p}.ln2.bias"), vec![h]));
            v.push((format!("{p}.fc1.weight"), vec![i, h]));
            v.push((format!("{p}.fc1.bias"), vec![i]));
            v.push((format!("{p}.fc2.weight"), vec![h, i]));
            v.push((format!("{p}.fc2.bias"), vec![h]));
        }
        v.push(("final_norm.weight".into(), vec![h]));
        v.push(("final_norm.bias".into(), vec![h]));
        if let Some(p) = self.projection {
            v.push(("text_projection.weight".into(), vec![p as usize, h]));
        }
        v
    }
}

/// EVA02-CLIP image tower. **Not a vanilla ViT** - see [`EvaVisionConfig`]'s
/// module notes and `crates/clip/src/model.rs`.
#[derive(Clone, Debug)]
pub struct EvaVisionConfig {
    pub image_size: u32,
    pub patch: u32,
    pub width: u32,
    pub layers: u32,
    pub heads: u32,
    /// SwiGLU hidden width - `int(width * 2.6667)` = 2730, NOT `4*width`.
    pub mlp_hidden: u32,
    /// `head` output width (the CLIP joint-embedding dim).
    pub embed_dim: u32,
    pub eps: f32,
    /// RoPE frequency interpolation: the tower was pretrained at a
    /// `pt_seq_len`-square grid and runs at `grid()`; `t = arange(g)/g*pt`.
    pub pt_seq_len: u32,
    pub rope_theta: f32,
    /// OpenAI CLIP normalization constants (for the preprocessing seam; the
    /// parity test replays the reference's `pixel_values` directly). Written
    /// verbatim as the reference states them - see [`EvaVisionConfig::eva02_l336`].
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl EvaVisionConfig {
    /// `EVA02-CLIP-L-14-336` - the checkpoint PuLID ships against.
    // `mean`/`std` are OpenAI CLIP's published constants, transcribed digit for
    // digit from the reference preprocessing config. Two of them carry one more
    // decimal than f32 can represent, which clippy flags; truncating would make
    // them stop matching the source they were copied from for no numerical gain,
    // so the literals stay and the lint is silenced HERE only.
    #[allow(clippy::excessive_precision)]
    pub fn eva02_l336() -> EvaVisionConfig {
        EvaVisionConfig {
            image_size: 336,
            patch: 14,
            width: 1024,
            layers: 24,
            heads: 16,
            mlp_hidden: 2730,
            embed_dim: 768,
            eps: 1e-6,
            pt_seq_len: 16,
            rope_theta: 10000.0,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711],
        }
    }

    /// Patch-grid side (24 for 336/14).
    pub fn grid(&self) -> u32 {
        self.image_size / self.patch
    }
    pub fn num_patches(&self) -> u32 {
        self.grid() * self.grid()
    }
    /// Token count including the cls token.
    pub fn seq_len(&self) -> u32 {
        self.num_patches() + 1
    }
    pub fn head_dim(&self) -> u32 {
        self.width / self.heads
    }
    /// Rotary pairs per head (`head_dim/2`).
    pub fn rope_half(&self) -> u32 {
        self.head_dim() / 2
    }

    /// The blocks PuLID taps for `id_vit_hidden` (their OUTPUTS).
    pub const PULID_TAPS: [u32; 5] = [3, 7, 11, 15, 19];

    /// **Head-channel permutation, half-split <- interleaved.**
    ///
    /// EVA's RoPE rotates the *interleaved* pairs `(2j, 2j+1)` of each head;
    /// brain's table-driven `rope2d.wgsl` rotates *half-split* pairs
    /// `(d, d+half)` sharing angle index `d`. Rather than add a fourth RoPE
    /// kernel, the q and k projection ROWS are permuted once at import so that
    /// half-split index `d` holds the reference's channel `perm[d]`:
    ///   `perm[d] = 2d` for `d < half`, `perm[d] = 2(d-half)+1` otherwise.
    ///
    /// This is exact, not an approximation: attention only ever consumes
    /// `q . k` summed over the head axis, so permuting q and k identically
    /// leaves every score unchanged, and v / `proj` are untouched. The parity
    /// test proves it by un-permuting the tapped post-RoPE q and k and
    /// comparing against the reference tensors in reference channel order.
    pub fn head_perm(&self) -> Vec<usize> {
        let half = self.rope_half() as usize;
        (0..2 * half)
            .map(|d| if d < half { 2 * d } else { 2 * (d - half) + 1 })
            .collect()
    }

    /// `[num_patches, rope_half]` cos/sin tables for `rope2d.wgsl`, in the
    /// permuted (half-split) channel order [`head_perm`] establishes.
    ///
    /// Reference (`eva_clip/rope.py::VisionRotaryEmbeddingFast`), recomputed
    /// rather than imported - the checkpoint's `freqs_cos`/`freqs_sin` buffers
    /// are fp16 leftovers the reference itself discards:
    ///   `freqs[k] = theta^(-2k/dim)`, `dim = head_dim/2`, `k in [0, dim/2)`
    ///   `t[i]     = i/grid * pt_seq_len`
    ///   token `y*grid + x` gets `t[y]*freqs` on the first half of the head and
    ///   `t[x]*freqs` on the second half.
    pub fn rope_tables(&self) -> (Vec<f32>, Vec<f32>) {
        let half = self.rope_half() as usize; // 32 = channels rotated per pair set
        let quarter = half / 2; // 16 = frequencies per axis
        let g = self.grid() as usize;
        let dim = half as f32; // VisionRotaryEmbeddingFast's `dim`
        let freqs: Vec<f32> = (0..quarter)
            .map(|k| self.rope_theta.powf(-(2.0 * k as f32) / dim))
            .collect();
        let t: Vec<f32> =
            (0..g).map(|i| i as f32 / g as f32 * self.pt_seq_len as f32).collect();
        let mut cos = vec![0.0f32; g * g * half];
        let mut sin = vec![0.0f32; g * g * half];
        for y in 0..g {
            for x in 0..g {
                let row = (y * g + x) * half;
                for k in 0..quarter {
                    let ah = t[y] * freqs[k];
                    let aw = t[x] * freqs[k];
                    cos[row + k] = ah.cos();
                    sin[row + k] = ah.sin();
                    cos[row + quarter + k] = aw.cos();
                    sin[row + quarter + k] = aw.sin();
                }
            }
        }
        (cos, sin)
    }

    /// Canonical brain-side tensor manifest (see
    /// [`ClipTextConfig::tensor_manifest`]).
    ///
    /// `blocks.N.qkv.*` is FUSED `[3W, W]` / `[3W]`: the checkpoint's three
    /// separate `q_proj`/`k_proj`/`v_proj` weights concatenated, with the k
    /// third of the bias set to zero (the reference's k linear genuinely has no
    /// bias - `F.linear(x, k_proj.weight, bias=None)`).
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let w = self.width as usize;
        let m = self.mlp_hidden as usize;
        let p = self.patch as usize;
        let mut v: Vec<(String, Vec<usize>)> = vec![
            ("cls_token".into(), vec![w]),
            ("pos_embed".into(), vec![self.seq_len() as usize, w]),
            ("patch.weight".into(), vec![w, 3, p, p]),
            ("patch.bias".into(), vec![w]),
        ];
        for l in 0..self.layers {
            let b = format!("blocks.{l}");
            v.push((format!("{b}.norm1.weight"), vec![w]));
            v.push((format!("{b}.norm1.bias"), vec![w]));
            v.push((format!("{b}.qkv.weight"), vec![3 * w, w]));
            v.push((format!("{b}.qkv.bias"), vec![3 * w]));
            v.push((format!("{b}.inner_ln.weight"), vec![w]));
            v.push((format!("{b}.inner_ln.bias"), vec![w]));
            v.push((format!("{b}.proj.weight"), vec![w, w]));
            v.push((format!("{b}.proj.bias"), vec![w]));
            v.push((format!("{b}.norm2.weight"), vec![w]));
            v.push((format!("{b}.norm2.bias"), vec![w]));
            v.push((format!("{b}.w1.weight"), vec![m, w]));
            v.push((format!("{b}.w1.bias"), vec![m]));
            v.push((format!("{b}.w2.weight"), vec![m, w]));
            v.push((format!("{b}.w2.bias"), vec![m]));
            v.push((format!("{b}.ffn_ln.weight"), vec![m]));
            v.push((format!("{b}.ffn_ln.bias"), vec![m]));
            v.push((format!("{b}.w3.weight"), vec![w, m]));
            v.push((format!("{b}.w3.bias"), vec![w]));
        }
        v.push(("norm.weight".into(), vec![w]));
        v.push(("norm.bias".into(), vec![w]));
        v.push(("head.weight".into(), vec![self.embed_dim as usize, w]));
        v.push(("head.bias".into(), vec![self.embed_dim as usize]));
        v
    }
}

/// A **vanilla CLIP-L/14 image tower** - the one DeepSeek-OCR's DeepEncoder
/// puts behind its SAM branch.
///
/// Deliberately NOT a second copy of a config shape that already exists: the
/// eight numbers that describe the tower are
/// [`gguf::deepseek_ocr_vision::ClipConfig`], read off the real mmproj by
/// `gguf::deepseek_ocr_vision::config_from_gguf`, and this type wraps that value
/// rather than restating its fields. The only thing it adds is the MLP
/// activation, which the mmproj carries one level up (`clip.use_gelu`, on
/// [`gguf::deepseek_ocr_vision::DeepseekOcrVisionConfig`]) because it is shared
/// with the SAM branch.
///
/// **`use_gelu == false` means quick-GELU**, not "no activation" - the flag
/// names the *exact-erf* variant, so the DeepSeek-OCR checkpoint (which sets it
/// false) runs `quick_gelu`. See that module's `DeepseekOcrVisionConfig`
/// documentation; [`ClipVisionConfig::from_gguf`] is the one place that
/// inversion is written down.
///
/// Unlike [`EvaVisionConfig`] this is an ordinary pre-LN ViT - no LayerScale, no
/// QK-norm, no RoPE, no subln, one GELU-activated MLP pair - which is exactly
/// what `model::vit::vit_block_fwd_cached` / `vit_block_bwd` express, so
/// `crate::model::ClipVision` composes those instead of carrying its own block
/// graph.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipVisionConfig {
    /// The checkpoint-side shape, owned by `crates/gguf`.
    pub shape: gguf::deepseek_ocr_vision::ClipConfig,
    /// MLP activation (`quick_gelu` for every released CLIP-L; `gelu_erf` is
    /// the OpenCLIP-style alternative the same graph runs unchanged).
    pub act: TextAct,
}

/// The epsilon DeepSeek-OCR's CLIP body actually runs its LayerNorms at.
///
/// Not a read of the checkpoint - see [`ClipVisionConfig::from_gguf`] for why
/// the file's own `clip.vision.attention.layer_norm_epsilon` (1e-6) is the SAM
/// tower's number and never reaches these blocks.
pub const DEEPSEEK_OCR_CLIP_EPS: f32 = 1e-5;

impl ClipVisionConfig {
    /// The CLIP-L/14@224 tower of DeepSeek-OCR's DeepEncoder: 24 x 1024, 16
    /// heads, MLP 4096, patch 14, 257 learned positions.
    ///
    /// The numbers are the ones [`Self::from_gguf`] derives from the real
    /// `mmproj-DeepSeek-OCR-Q8_0.gguf`, asserted equal to it by
    /// `tests/deepseek_ocr_real.rs` when that file is present; `ffn_hidden` is
    /// 4096 from the tensor shapes, NOT the file's
    /// `clip.vision.feed_forward_length = 64` (a converter bug - see the
    /// `gguf::deepseek_ocr_vision` module doc).
    ///
    /// **Two fields were wrong here until the real header was read**, both
    /// recorded below because each was a plausible-looking guess:
    ///
    ///  * `act` was `QuickGelu`, on the belief that this file sets
    ///    `clip.use_gelu = false`. It does not: the shipped mmproj carries
    ///    `clip.use_gelu = true`, so the inversion in [`Self::from_gguf`]
    ///    selects the exact-erf form. (What the ORIGINAL CLIP-L uses is
    ///    quick-GELU, and llama.cpp's own reading of `use_gelu = true` is its
    ///    `FFN_GELU`, the tanh approximation - so all three candidates are in
    ///    play and no CLIP-internal reference tap exists to settle it. This
    ///    tracks the file's own flag, which is the only defensible default.)
    ///  * `layer_norm_eps` is 1e-5 and that is NOT the file's key - see
    ///    [`Self::from_gguf`].
    pub fn deepseek_ocr() -> ClipVisionConfig {
        ClipVisionConfig {
            shape: gguf::deepseek_ocr_vision::ClipConfig {
                d_model: 1024,
                n_layers: 24,
                n_heads: 16,
                ffn_hidden: 4096,
                patch_size: 14,
                image_size: 224,
                n_positions: 257,
                layer_norm_eps: DEEPSEEK_OCR_CLIP_EPS,
            },
            act: TextAct::GeluErf,
        }
    }

    /// Adopt an imported mmproj config. **The activation inversion lives here**:
    /// `use_gelu == false` selects quick-GELU (this file sets it *true*, so it
    /// gets the exact-erf form).
    ///
    /// **The LayerNorm epsilon is deliberately NOT the file's key.** The shipped
    /// mmproj carries `clip.vision.attention.layer_norm_epsilon = 1e-6`, but
    /// that value never reaches the reference's CLIP blocks: llama.cpp reads it
    /// into `hparams.eps` and then, in the `PROJECTOR_TYPE_DEEPSEEKOCR` branch
    /// that runs afterwards, overwrites it with `1e-5` - with the comment "the
    /// CLIP/ViT body runs its layernorms at 1e-5 (the SAM stage uses 1e-6)".
    /// The SAM tower's `1e-6` is likewise a hardcoded constant in its graph
    /// builder, not a read of this key. Taking the key at face value here would
    /// silently run the CLIP tower at the SAM tower's epsilon, so the same
    /// override is applied - once, here, where a reader of an imported config
    /// will find it.
    pub fn from_gguf(cfg: &gguf::deepseek_ocr_vision::DeepseekOcrVisionConfig) -> ClipVisionConfig {
        let mut shape = cfg.clip.clone();
        shape.layer_norm_eps = DEEPSEEK_OCR_CLIP_EPS;
        ClipVisionConfig { shape, act: if cfg.use_gelu { TextAct::GeluErf } else { TextAct::QuickGelu } }
    }

    pub fn d_model(&self) -> u32 {
        self.shape.d_model
    }
    pub fn layers(&self) -> u32 {
        self.shape.n_layers
    }
    pub fn heads(&self) -> u32 {
        self.shape.n_heads
    }
    pub fn head_dim(&self) -> u32 {
        self.shape.d_model / self.shape.n_heads
    }
    pub fn mlp_hidden(&self) -> u32 {
        self.shape.ffn_hidden
    }
    pub fn patch(&self) -> u32 {
        self.shape.patch_size
    }
    pub fn image_size(&self) -> u32 {
        self.shape.image_size
    }
    pub fn eps(&self) -> f32 {
        self.shape.layer_norm_eps
    }
    /// Learned position rows in the checkpoint (class token + native patches).
    pub fn n_positions(&self) -> u32 {
        self.shape.n_positions
    }
    /// The patch-grid side the checkpoint's position table was trained at
    /// (`image_size / patch_size`) - 16 for CLIP-L/14@224.
    pub fn native_grid(&self) -> u32 {
        self.shape.image_size / self.shape.patch_size
    }
    pub fn native_patches(&self) -> u32 {
        self.native_grid() * self.native_grid()
    }

    /// Canonical brain-side tensor manifest (see
    /// [`ClipTextConfig::tensor_manifest`]).
    ///
    /// Names and ORDER are `gguf::deepseek_ocr_vision`'s `param_list()` entries
    /// with the `vision.clip.` prefix stripped, so the importer is a prefix
    /// strip and the two lists cannot drift - asserted by
    /// `manifest_matches_the_gguf_param_list`.
    ///
    /// Two absences are real, not oversights:
    ///   * **no `patch_embed.bias`** - CLIP's patch embedding is
    ///     `Conv2d(3, D, k, stride=k, bias=False)`.
    ///   * **no post-LayerNorm** - DeepSeek-OCR consumes the last block's
    ///     hidden states directly, so the mmproj carries `pre_norm` only.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let d = self.d_model() as usize;
        let ff = self.mlp_hidden() as usize;
        let p = self.patch() as usize;
        let mut v: Vec<(String, Vec<usize>)> = vec![
            ("class_embed".into(), vec![d]),
            ("patch_embed.weight".into(), vec![d, 3, p, p]),
            ("pos_embed".into(), vec![self.n_positions() as usize, d]),
            ("pre_norm.weight".into(), vec![d]),
            ("pre_norm.bias".into(), vec![d]),
        ];
        for l in 0..self.layers() {
            let b = format!("blocks.{l}");
            v.push((format!("{b}.norm1.weight"), vec![d]));
            v.push((format!("{b}.norm1.bias"), vec![d]));
            v.push((format!("{b}.attn.qkv.weight"), vec![3 * d, d]));
            v.push((format!("{b}.attn.qkv.bias"), vec![3 * d]));
            v.push((format!("{b}.attn.proj.weight"), vec![d, d]));
            v.push((format!("{b}.attn.proj.bias"), vec![d]));
            v.push((format!("{b}.norm2.weight"), vec![d]));
            v.push((format!("{b}.norm2.bias"), vec![d]));
            v.push((format!("{b}.mlp.fc1.weight"), vec![ff, d]));
            v.push((format!("{b}.mlp.fc1.bias"), vec![ff]));
            v.push((format!("{b}.mlp.fc2.weight"), vec![d, ff]));
            v.push((format!("{b}.mlp.fc2.bias"), vec![d]));
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_manifest_counts_match_the_reference_checkpoints() {
        // CLIP-L: 2 embeddings + 12 layers x 12 tensors + 2 final-LN = 148.
        let l = ClipTextConfig::clip_l();
        assert_eq!(l.tensor_manifest().len(), 2 + 12 * 12 + 2);
        assert_eq!(l.penultimate_layer(), 10);
        // bigG: + text_projection.
        let g = ClipTextConfig::openclip_bigg();
        assert_eq!(g.tensor_manifest().len(), 2 + 32 * 12 + 2 + 1);
        assert_eq!(g.penultimate_layer(), 30);
    }

    #[test]
    fn eva_manifest_and_shapes() {
        let c = EvaVisionConfig::eva02_l336();
        assert_eq!(c.grid(), 24);
        assert_eq!(c.seq_len(), 577);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.rope_half(), 32);
        // 4 stem + 24 blocks x 18 + norm(2) + head(2).
        assert_eq!(c.tensor_manifest().len(), 4 + 24 * 18 + 2 + 2);
    }

    #[test]
    fn head_perm_is_a_permutation_pairing_half_split_with_interleaved() {
        let c = EvaVisionConfig::eva02_l336();
        let p = c.head_perm();
        let half = c.rope_half() as usize;
        assert_eq!(p.len(), 2 * half);
        let mut seen = vec![false; 2 * half];
        for &x in &p {
            assert!(!seen[x], "head_perm is not injective");
            seen[x] = true;
        }
        // half-split partner (d, d+half) == interleaved pair (2d, 2d+1)
        for d in 0..half {
            assert_eq!(p[d], 2 * d);
            assert_eq!(p[d + half], 2 * d + 1);
        }
    }

    /// The vision manifest IS `gguf::deepseek_ocr_vision`'s CLIP half, prefix
    /// stripped - names, order AND element counts. This is the gate that keeps
    /// [`crate::import::import_deepseek_ocr_vision`] a prefix strip instead of a
    /// second, drifting name table.
    #[test]
    fn manifest_matches_the_gguf_param_list() {
        use gguf::deepseek_ocr_vision::{DeepseekOcrVisionConfig, SamConfig};

        let cfg = ClipVisionConfig::deepseek_ocr();
        // The SAM half is irrelevant here but `param_list` needs a whole config;
        // one tiny block keeps it cheap and its entries are filtered out below.
        let full = DeepseekOcrVisionConfig {
            sam: SamConfig {
                d_model: 8,
                n_layers: 1,
                n_heads: 2,
                ffn_hidden: 16,
                patch_size: 2,
                grid: 4,
                window_size: 2,
                global_attn_layers: vec![],
                neck_channels: 4,
                compress_mid: 4,
                compress_out: 4,
            },
            clip: cfg.shape.clone(),
            projector_in: cfg.d_model() + 4,
            projection_dim: 12,
            image_mean: vec![0.5; 3],
            image_std: vec![0.5; 3],
            use_gelu: false,
            scale_factor: 1,
        };
        let want: Vec<(String, usize)> = full
            .param_list()
            .into_iter()
            .filter_map(|(n, k)| n.strip_prefix("vision.clip.").map(|s| (s.to_string(), k)))
            .collect();
        let got: Vec<(String, usize)> =
            cfg.tensor_manifest().into_iter().map(|(n, s)| (n, s.iter().product())).collect();
        assert_eq!(got, want, "vision manifest must be the gguf CLIP half, prefix stripped");
        // 5 stem + 24 blocks x 12.
        assert_eq!(got.len(), 5 + 24 * 12);
    }

    /// `use_gelu` is INVERTED relative to what its name suggests, and the
    /// DeepSeek-OCR mmproj sets it false. A future flip would silently change
    /// every MLP in the tower, so it is asserted rather than commented.
    #[test]
    fn use_gelu_false_selects_quick_gelu() {
        use gguf::deepseek_ocr_vision::{DeepseekOcrVisionConfig, SamConfig};
        let sam = SamConfig {
            d_model: 8,
            n_layers: 1,
            n_heads: 2,
            ffn_hidden: 16,
            patch_size: 2,
            grid: 4,
            window_size: 2,
            global_attn_layers: vec![],
            neck_channels: 4,
            compress_mid: 4,
            compress_out: 4,
        };
        let base = ClipVisionConfig::deepseek_ocr();
        let mk = |use_gelu: bool| DeepseekOcrVisionConfig {
            sam: sam.clone(),
            clip: base.shape.clone(),
            projector_in: base.d_model() + 4,
            projection_dim: 12,
            image_mean: vec![0.5; 3],
            image_std: vec![0.5; 3],
            use_gelu,
            scale_factor: 1,
        };
        assert_eq!(ClipVisionConfig::from_gguf(&mk(false)).act, TextAct::QuickGelu);
        assert_eq!(ClipVisionConfig::from_gguf(&mk(true)).act, TextAct::GeluErf);
        // The SHIPPED mmproj sets `clip.use_gelu = true` (read off the real
        // header, not assumed), so the documented preset must be the erf arm.
        // `tests/deepseek_ocr_real.rs` asserts the preset against the real file.
        assert_eq!(ClipVisionConfig::deepseek_ocr().act, TextAct::GeluErf);
    }

    /// The file's `layer_norm_epsilon` is the SAM tower's 1e-6 and the reference
    /// overrides it for the CLIP body - see [`ClipVisionConfig::from_gguf`]. A
    /// config that simply adopted the key would run 1e-6 and nothing else would
    /// notice.
    #[test]
    fn clip_eps_overrides_the_files_key() {
        use gguf::deepseek_ocr_vision::{DeepseekOcrVisionConfig, SamConfig};
        let base = ClipVisionConfig::deepseek_ocr();
        let mut clip = base.shape.clone();
        clip.layer_norm_eps = 1e-6; // what the shipped file carries
        let cfg = DeepseekOcrVisionConfig {
            sam: SamConfig {
                d_model: 8,
                n_layers: 1,
                n_heads: 2,
                ffn_hidden: 16,
                patch_size: 2,
                grid: 4,
                window_size: 2,
                global_attn_layers: vec![],
                neck_channels: 4,
                compress_mid: 4,
                compress_out: 4,
            },
            clip,
            projector_in: base.d_model() + 4,
            projection_dim: 12,
            image_mean: vec![0.5; 3],
            image_std: vec![0.5; 3],
            use_gelu: true,
            scale_factor: 1,
        };
        assert_eq!(ClipVisionConfig::from_gguf(&cfg).eps(), super::DEEPSEEK_OCR_CLIP_EPS);
        assert_ne!(super::DEEPSEEK_OCR_CLIP_EPS, 1e-6, "the override must not be a no-op");
    }

    #[test]
    fn vision_geometry() {
        let c = ClipVisionConfig::deepseek_ocr();
        assert_eq!(c.native_grid(), 16);
        assert_eq!(c.native_patches(), 256);
        assert_eq!(c.n_positions(), 1 + c.native_patches());
        assert_eq!(c.head_dim(), 64);
    }

    #[test]
    fn rope_tables_match_the_reference_construction_at_a_sampled_token() {
        let c = EvaVisionConfig::eva02_l336();
        let (cos, sin) = c.rope_tables();
        let half = c.rope_half() as usize;
        let g = c.grid() as usize;
        assert_eq!(cos.len(), g * g * half);
        // token (y=3, x=5), first h frequency and first w frequency
        let (y, x) = (3usize, 5usize);
        let row = (y * g + x) * half;
        let t = |i: usize| i as f32 / g as f32 * c.pt_seq_len as f32;
        assert!((cos[row] - (t(y) * 1.0).cos()).abs() < 1e-6);
        assert!((sin[row + half / 2] - (t(x) * 1.0).sin()).abs() < 1e-6);
        // last frequency: theta^(-(2*15)/32)
        let f15 = c.rope_theta.powf(-30.0 / 32.0);
        assert!((cos[row + 15] - (t(y) * f15).cos()).abs() < 1e-6);
    }
}
