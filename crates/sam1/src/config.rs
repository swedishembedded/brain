// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The SAM-1 / ViTDet tower's shape, and the tensor manifest it implies.
//!
//! The manifest is the **same name set** `brain-gguf`'s `deepseek_ocr_vision`
//! module already emits for the `vision.sam.*` half of the mmproj file, so an
//! import is a 1:1 name match with no translation table (see [`crate::import`]).

use gguf::deepseek_ocr_vision::SamConfig;

/// Everything that varies between SAM towers.
///
/// ## Relationship to [`gguf::deepseek_ocr_vision::SamConfig`]
///
/// That struct is the checkpoint's view and is deliberately reused as the
/// import seam ([`SamViTConfig::from`]); this one is the *graph's* view and
/// differs in exactly three ways, each because the graph can express something
/// the real checkpoint happens not to use:
///
///  * `grid_h`/`grid_w` and `window_h`/`window_w` are independent, where the
///    checkpoint's are square. A square-only config cannot express a tiny
///    fixture whose two axes are different numbers, and "the two axes are the
///    same number" is exactly the coincidence that hides an h/w swap in the
///    decomposed relative-position bias.
///  * [`Self::rel_pos_table_rows`] can override the canonical `2*extent - 1`
///    table height, which is what makes `get_rel_pos`'s resample run at all.
///  * `eps` and [`Self::attn_chunk`] are graph parameters with no GGUF key.
#[derive(Debug, Clone, PartialEq)]
pub struct SamViTConfig {
    pub d_model: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub ffn_hidden: u32,
    /// Patch-embed conv kernel == stride.
    pub patch_size: u32,
    /// Patch grid, in patches.
    pub grid_h: u32,
    pub grid_w: u32,
    /// Local-attention window, in patches. The grid is zero-padded up to a
    /// multiple of it (`model::vit::WindowPlan::padded`), never cropped.
    pub window_h: u32,
    pub window_w: u32,
    /// Blocks using **global** (unwindowed) attention; the rest are windowed.
    pub global_attn_layers: Vec<u32>,
    /// Neck width -- both the `1x1` and the `3x3` conv run at it.
    pub neck_channels: u32,
    /// The two stride-2 compressor conv widths.
    pub compress_mid: u32,
    pub compress_out: u32,
    /// LayerNorm epsilon, shared by the block norms and the two `LayerNorm2d`s.
    pub eps: f32,
    /// Query rows per attention dispatch (`model::block::chunked_bidir_fwd`'s
    /// `chunk`).
    ///
    /// **This is an alignment-constrained number, not a tuning knob.** That
    /// builder binds `qkv` sliced at `(row0 + q0) * 3 * d_model` floats and
    /// `ctx` at `(row0 + q0) * d_model`, and a storage-binding offset must be a
    /// multiple of 64 floats (256 B) or wgpu rejects the bind group. So both
    /// `attn_chunk * d_model` and every span's `row0 * d_model` must be
    /// multiples of 64. [`SamViTConfig::check_bindable`] is the loud failure.
    pub attn_chunk: u32,
    /// Per-block learned relative-position table heights, `(rows_h, rows_w)`.
    ///
    /// Empty means the canonical `2 * extent - 1` for every block, which is
    /// what the DeepSeek-OCR checkpoint carries and what makes `get_rel_pos`'s
    /// resample the identity. A non-empty override models a table trained at a
    /// *different* window/grid extent -- the case real SAM-1 hits whenever it is
    /// run at a resolution other than the one it was trained at, and the only
    /// way to exercise the two-tap interpolation from a fixture.
    pub rel_pos_table_rows: Vec<(u32, u32)>,
}

impl SamViTConfig {
    pub fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }
    /// Patch tokens, in row-major grid order.
    pub fn rows(&self) -> u32 {
        self.grid_h * self.grid_w
    }
    pub fn image_h(&self) -> u32 {
        self.grid_h * self.patch_size
    }
    pub fn image_w(&self) -> u32 {
        self.grid_w * self.patch_size
    }
    pub fn is_global(&self, l: u32) -> bool {
        self.global_attn_layers.contains(&l)
    }
    /// The `(h, w)` extent ONE attention span covers in block `l`: the window
    /// for a windowed block, the whole grid for a global one. This is the
    /// `(qh, qw) == (kh, kw)` of `model::vit::RelPos`.
    pub fn attn_extent(&self, l: u32) -> (u32, u32) {
        if self.is_global(l) {
            (self.grid_h, self.grid_w)
        } else {
            (self.window_h, self.window_w)
        }
    }
    /// Learned relative-position table heights for block `l`.
    pub fn rel_pos_rows(&self, l: u32) -> (u32, u32) {
        if let Some(&rc) = self.rel_pos_table_rows.get(l as usize) {
            return rc;
        }
        let (h, w) = self.attn_extent(l);
        (2 * h - 1, 2 * w - 1)
    }
    /// Compressor output grid -- two stride-2 `3x3` convs at pad 1.
    pub fn compress_grid(&self) -> (u32, u32) {
        let half = |n: u32| (n + 2 - 3) / 2 + 1;
        (half(half(self.grid_h)), half(half(self.grid_w)))
    }

    /// Panic with the numbers in scope if this config cannot be dispatched.
    ///
    /// Two invariants, both paid for by the kernels' own ABI rather than by
    /// taste; see [`Self::attn_chunk`] for the binding-alignment one.
    pub fn check_bindable(&self) {
        assert!(
            self.n_heads > 0 && self.d_model.is_multiple_of(self.n_heads),
            "d_model {} not divisible by n_heads {}",
            self.d_model,
            self.n_heads
        );
        let c = self.d_model as u64;
        assert!(
            (self.attn_chunk as u64 * c).is_multiple_of(64),
            "attn_chunk {} x d_model {} = {} floats is not 64-float (256 B) aligned; \
             chunk must be a multiple of {}",
            self.attn_chunk,
            self.d_model,
            self.attn_chunk as u64 * c,
            64 / gcd(c, 64)
        );
        // Every span's first row, over every block. Windowed spans start at
        // multiples of the window's token count; a global block has one span
        // at row 0, which is trivially aligned.
        let win = (self.window_h * self.window_w) as u64;
        assert!(
            (win * c).is_multiple_of(64),
            "a {}x{} window is {win} rows and d_model is {}, so span row0 x d_model = {} floats \
             is not 64-float (256 B) aligned; d_model must be a multiple of {}",
            self.window_h,
            self.window_w,
            self.d_model,
            win * c,
            64 / gcd(win, 64)
        );
    }

    /// The canonical tensor manifest: every parameter and its element count,
    /// under the SAME names `gguf::deepseek_ocr_vision` maps the mmproj onto.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let (d, ff) = (self.d_model as usize, self.ffn_hidden as usize);
        let p = self.patch_size as usize;
        let mut out: Vec<(String, usize)> = Vec::new();
        out.push(("vision.sam.patch_embed.weight".into(), d * 3 * p * p));
        out.push(("vision.sam.patch_embed.bias".into(), d));
        out.push(("vision.sam.pos_embed".into(), self.rows() as usize * d));
        for l in 0..self.n_layers {
            let b = |leaf: &str| format!("vision.sam.blocks.{l}.{leaf}");
            let (rh, rw) = self.rel_pos_rows(l);
            let hd = self.head_dim() as usize;
            out.push((b("norm1.weight"), d));
            out.push((b("norm1.bias"), d));
            out.push((b("attn.qkv.weight"), 3 * d * d));
            out.push((b("attn.qkv.bias"), 3 * d));
            out.push((b("attn.proj.weight"), d * d));
            out.push((b("attn.proj.bias"), d));
            out.push((b("attn.rel_pos_h"), rh as usize * hd));
            out.push((b("attn.rel_pos_w"), rw as usize * hd));
            out.push((b("norm2.weight"), d));
            out.push((b("norm2.bias"), d));
            out.push((b("mlp.fc1.weight"), ff * d));
            out.push((b("mlp.fc1.bias"), ff));
            out.push((b("mlp.fc2.weight"), d * ff));
            out.push((b("mlp.fc2.bias"), d));
        }
        let n = self.neck_channels as usize;
        out.push(("vision.sam.neck.conv1.weight".into(), n * d));
        out.push(("vision.sam.neck.norm1.weight".into(), n));
        out.push(("vision.sam.neck.norm1.bias".into(), n));
        out.push(("vision.sam.neck.conv2.weight".into(), n * n * 3 * 3));
        out.push(("vision.sam.neck.norm2.weight".into(), n));
        out.push(("vision.sam.neck.norm2.bias".into(), n));
        out.push(("vision.sam.compress.conv1.weight".into(), self.compress_mid as usize * n * 3 * 3));
        out.push(("vision.sam.compress.conv2.weight".into(), self.compress_out as usize * self.compress_mid as usize * 3 * 3));
        out
    }

    /// The real DeepSeek-OCR / SAM ViT-B tower, at its native 1024² input.
    ///
    /// Every number is the one `gguf::deepseek_ocr_vision::config_from_gguf`
    /// reads off the shipped mmproj (verified against the real file by
    /// `crate::import`'s coverage test when the checkpoint is present). It is
    /// here for documentation and for a shape-only smoke test; running it needs
    /// weights.
    pub fn deepseek_ocr() -> SamViTConfig {
        SamViTConfig {
            d_model: 768,
            n_layers: 12,
            n_heads: 12,
            ffn_hidden: 3072,
            patch_size: 16,
            grid_h: 64,
            grid_w: 64,
            window_h: 14,
            window_w: 14,
            global_attn_layers: vec![2, 5, 8, 11],
            neck_channels: 256,
            compress_mid: 512,
            compress_out: 1024,
            eps: 1e-6,
            attn_chunk: 256,
            rel_pos_table_rows: Vec::new(),
        }
    }

    /// The gradient-check fixture. See this crate's `tests/gradcheck.rs` header
    /// for what each number is chosen to break.
    ///
    /// The SAM geometry (patch 2, grid 13x7, window 4x3, two blocks -- one
    /// windowed, one global -- with table heights `(7, 11)` and `(15, 13)`)
    /// mirrors the checkpoint-free golden dumper's own tiny SAM sub-fixture, so
    /// the two are comparable later. `d_model` deliberately does NOT: the
    /// golden uses 10, and 10 is un-dispatchable here (see
    /// [`Self::check_bindable`] -- a 12-row window forces `16 | d_model`).
    pub fn tiny() -> SamViTConfig {
        SamViTConfig {
            d_model: 16, // 2 heads x head_dim 8
            n_layers: 2,
            n_heads: 2,
            ffn_hidden: 17,
            patch_size: 2,
            grid_h: 13,
            grid_w: 7,
            window_h: 4,
            window_w: 3,
            global_attn_layers: vec![1],
            neck_channels: 6,
            compress_mid: 9,
            compress_out: 11,
            eps: 1e-6,
            attn_chunk: 4,
            // block 0 (windowed 4x3): h identity (2*4-1 == 7), w DOWNsample
            // (11 > 2*3-1); block 1 (global 13x7): h UPsample (15 < 2*13-1),
            // w identity (13 == 2*7-1). All three `get_rel_pos` cases, once.
            rel_pos_table_rows: vec![(7, 11), (15, 13)],
        }
    }
}

/// Greatest common divisor -- only used to phrase [`SamViTConfig::check_bindable`].
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a.max(1)
    } else {
        gcd(b, a % b)
    }
}

/// The checkpoint's square-grid view -> the graph's view.
///
/// One-to-one on every field the GGUF carries; the two axis pairs both take the
/// checkpoint's single value, and the table heights stay canonical (which is
/// what that file has).
impl From<&SamConfig> for SamViTConfig {
    fn from(s: &SamConfig) -> SamViTConfig {
        SamViTConfig {
            d_model: s.d_model,
            n_layers: s.n_layers,
            n_heads: s.n_heads,
            ffn_hidden: s.ffn_hidden,
            patch_size: s.patch_size,
            grid_h: s.grid,
            grid_w: s.grid,
            window_h: s.window_size,
            window_w: s.window_size,
            global_attn_layers: s.global_attn_layers.clone(),
            neck_channels: s.neck_channels,
            compress_mid: s.compress_mid,
            compress_out: s.compress_out,
            eps: 1e-6,
            attn_chunk: 256,
            rel_pos_table_rows: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest this crate builds must be exactly the `vision.sam.*` half
    /// of what the mmproj loader emits -- same names, same element counts. If
    /// the two ever drift, import stops being a 1:1 name match and nothing else
    /// would notice.
    #[test]
    fn manifest_matches_the_gguf_loaders_sam_half() {
        let full = gguf::deepseek_ocr_vision::DeepseekOcrVisionConfig {
            sam: SamConfig {
                d_model: 768,
                n_layers: 12,
                n_heads: 12,
                ffn_hidden: 3072,
                patch_size: 16,
                grid: 64,
                window_size: 14,
                global_attn_layers: vec![2, 5, 8, 11],
                neck_channels: 256,
                compress_mid: 512,
                compress_out: 1024,
            },
            clip: gguf::deepseek_ocr_vision::ClipConfig {
                d_model: 1024,
                n_layers: 24,
                n_heads: 16,
                ffn_hidden: 4096,
                patch_size: 14,
                image_size: 224,
                n_positions: 257,
                layer_norm_eps: 1e-6,
            },
            projector_in: 2048,
            projection_dim: 1280,
            image_mean: vec![0.5; 3],
            image_std: vec![0.5; 3],
            use_gelu: false,
            scale_factor: 1,
        };
        let ours = SamViTConfig::from(&full.sam);
        assert_eq!(ours, SamViTConfig::deepseek_ocr(), "the documented preset must equal the derived config");
        let theirs: Vec<(String, usize)> =
            full.param_list().into_iter().filter(|(n, _)| n.starts_with("vision.sam.")).collect();
        assert_eq!(ours.param_list(), theirs);
    }

    #[test]
    fn tiny_geometry_is_what_the_fixture_claims() {
        let c = SamViTConfig::tiny();
        c.check_bindable();
        assert_eq!(c.rows(), 91);
        assert_eq!((c.image_h(), c.image_w()), (26, 14));
        assert_eq!(c.attn_extent(0), (4, 3));
        assert_eq!(c.attn_extent(1), (13, 7));
        assert_eq!(c.rel_pos_rows(0), (7, 11));
        assert_eq!(c.rel_pos_rows(1), (15, 13));
        assert_eq!(c.compress_grid(), (4, 2));
        // The three `get_rel_pos` cases, spelled out against `2*extent - 1`.
        assert_eq!(c.rel_pos_rows(0).0, 2 * 4 - 1, "block 0 h must be the identity case");
        assert!(c.rel_pos_rows(0).1 > 2 * 3 - 1, "block 0 w must downsample");
        assert!(c.rel_pos_rows(1).0 < 2 * 13 - 1, "block 1 h must upsample");
        assert_eq!(c.rel_pos_rows(1).1, 2 * 7 - 1, "block 1 w must be the identity case");
    }

    /// The real tower's compressor is the documented `grid/4`.
    #[test]
    fn real_compressor_quarters_the_grid() {
        assert_eq!(SamViTConfig::deepseek_ocr().compress_grid(), (16, 16));
        SamViTConfig::deepseek_ocr().check_bindable();
    }

    /// The golden dumper's `sam_embed = 10` is NOT dispatchable through
    /// `chunked_bidir_fwd` at a 12-row window: `row0 * d_model` lands at 120
    /// floats, which is not a multiple of 64. Pinned so the next reader does not
    /// spend an afternoon on it.
    #[test]
    fn a_width_of_ten_is_rejected_loudly() {
        let cfg = SamViTConfig { d_model: 10, n_heads: 2, attn_chunk: 4, ..SamViTConfig::tiny() };
        let e = std::panic::catch_unwind(|| cfg.check_bindable()).unwrap_err();
        let msg = e.downcast_ref::<String>().map(String::as_str).unwrap_or("");
        assert!(msg.contains("64-float"), "expected an alignment complaint, got {msg:?}");
    }
}
