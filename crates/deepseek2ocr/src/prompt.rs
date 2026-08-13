// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The **decoder-side prompt**: the real tokenizer, DeepSeek-OCR's reserved
//! token ids, and the id sequence one image + its text turns into.
//!
//! [`crate::rows`] answers "which decoder row carries which *vector*". This
//! module answers the other half: which *token id* stands in that row before
//! the splice overwrites its embedding, and where the run starts.
//!
//! ## The one fact that decides the whole layout
//!
//! **Every image row is the same token id.** The newline and view-separator
//! rows do NOT get reserved ids of their own - they are `<image>` (128815) like
//! every projector row, and what makes them different is purely the *vector*
//! written over them (`vision.image_newline` / `vision.view_separator`, two
//! learned rows the mmproj ships). Checked twice, independently:
//!
//! * the reference implementation builds the run as
//!   `([image_token_id] * n + [image_token_id]) * n` then `+= [image_token_id]`
//!   - one id, `n` token rows each followed by a newline row, then the
//!   separator - with `image_token_id = 128815` written as a literal;
//!   its `self.image_newline` / `self.view_seperator` are `nn.Parameter`s,
//!   not vocabulary;
//! * the shipped `DeepSeek-OCR-Q8_0.gguf` vocabulary contains no
//!   newline/separator token to resolve: its 830 CONTROL entries are BOS/EOS/
//!   pad, 800 `<｜place▁holder▁no▁N｜>`, the FIM/chat/tool markers, `<image>`,
//!   the five grounding markers and four HTML table tags. That list is
//!   exhaustive and it is pinned by `tests/prompt_real.rs`.
//!
//! So [`ImageTokens`] resolves ONE string and points all three [`Src`] kinds at
//! it. It is still written as a three-field mapping rather than a single id,
//! because that is the shape a caller needs and the shape that stays honest if
//! a later checkpoint ever does split them.
//!
//! ## Scope: one contiguous run, global view only
//!
//! `deepseek2::DeepseekV2::enable_mm_splice` takes ONE `(row0, n_rows)` run,
//! so this builds the global (overview) view only -
//! [`ViewGrid::global_only`], 273 rows at the real `tokens_per_side = 16`.
//! Gundam/multi-tile prompts are a `rows`-level layout this decoder cannot
//! splice yet.
//!
//! **`n_rows` is the whole block (273), not the 256 projector rows.** The 17
//! newline/separator rows sit *inside* the run, so whoever fills the splice
//! buffer must interleave the two learned vectors at the rows
//! [`RowPlan::rows`] marks. That is what [`crate::layout::RowGather`] does, and
//! [`crate::DeepseekOcr::new_with_prompt`] is the entry point that takes a
//! [`Prompt`] from here, sizes `enable_mm_splice` at ITS `n_rows` and fills the
//! block through the gather. (`DeepseekOcr::new`/`new_split` still splice the
//! 256 projector rows contiguously - the golden fixture's own scope - so both
//! shapes exist on purpose; see `crate::model`'s header.)

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;

use crate::rows::{row_plan, RowPlan, Src, ViewGrid};

/// The image placeholder. One id for every image row - token, newline AND
/// separator (see this module's header).
pub const IMAGE: &str = "<image>";
/// Beginning of sequence. The reference prepends it to every prompt
/// (`tokenized_str = [bos_id] + tokenized_str`), and the GGUF agrees
/// (`tokenizer.ggml.add_bos_token = true`).
pub const BOS: &str = "<｜begin▁of▁sentence｜>";
/// End of sequence - the stop token a decode loop watches for.
pub const EOS: &str = "<｜end▁of▁sentence｜>";
/// Turns on DeepSeek-OCR's grounding mode: the model then emits
/// [`REF_OPEN`]/[`REF_CLOSE`] + [`DET_OPEN`]/[`DET_CLOSE`] spans.
pub const GROUNDING: &str = "<|grounding|>";
/// Opens a grounded text reference.
pub const REF_OPEN: &str = "<|ref|>";
/// Closes a grounded text reference.
pub const REF_CLOSE: &str = "<|/ref|>";
/// Opens a grounded detection box list.
pub const DET_OPEN: &str = "<|det|>";
/// Closes a grounded detection box list.
pub const DET_CLOSE: &str = "<|/det|>";

/// Load the LM's own tokenizer out of a GGUF's `tokenizer.ggml.*` KV.
///
/// The checkpoint declares `model = "gpt2"` (byte-level BPE) and
/// `pre = "deepseek-v3"`, which is exactly what
/// [`QwenBpe::from_gguf`] handles - see its doc for how far the pre-tokenizer
/// is reproduced and what is known to still differ. Only the KV block is read;
/// the mmap is dropped on return, so this costs the header, not the 3.1 GB.
pub fn tokenizer_from_gguf(path: &str) -> Result<QwenBpe, String> {
    let mg = checkpoint::gguf::MmapGguf::open(path)?;
    let gt = mg.tokenizer().ok_or_else(|| format!("{path}: no embedded tokenizer.ggml.* KV"))?;
    QwenBpe::from_gguf(&gt)
}

/// The decoder-side id of each [`Src`] kind.
///
/// All three are the same id in this checkpoint. The struct exists so a caller
/// writes `t.id_for(src)` instead of hardcoding that coincidence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ImageTokens {
    /// A projector-output row.
    pub image: u32,
    /// A row carrying the learned `image_newline` vector.
    pub newline: u32,
    /// A row carrying the learned `view_separator` vector.
    pub separator: u32,
}

impl ImageTokens {
    /// Resolve the image placeholder by its literal content. `Err` when the
    /// tokenizer is not this model's (a Qwen vocab has no `<image>`), which is
    /// the failure worth catching loudly - a missing special silently BPEs into
    /// `<`, `image`, `>` and the splice then lands on text rows.
    pub fn resolve(tok: &QwenBpe) -> Result<ImageTokens, String> {
        let image = tok.special_id(IMAGE).ok_or_else(|| format!("this tokenizer has no reserved {IMAGE:?} token"))?;
        Ok(ImageTokens { image, newline: image, separator: image })
    }

    /// The id that stands in a row of the given kind.
    pub fn id_for(&self, src: Src) -> u32 {
        match src {
            Src::Projector(_) => self.image,
            Src::Newline => self.newline,
            Src::Separator => self.separator,
        }
    }

    /// True when `id` is one of the three (i.e. the row belongs to the image
    /// block), which is what a test asserts over the spliced window.
    pub fn contains(&self, id: u32) -> bool {
        id == self.image || id == self.newline || id == self.separator
    }
}

/// One assembled prompt: the ids to decode over, and where the image sits.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Prompt {
    /// `BOS ++ text_before ++ image rows ++ text_after`.
    pub ids: Vec<u32>,
    /// First decoder row of the image block.
    pub row0: u32,
    /// Rows in the image block - `plan.len()`, newline/separator rows
    /// INCLUDED. See this module's header.
    pub n_rows: u32,
    /// The row layout those `n_rows` follow: which of them is a projector row,
    /// and which projector row it is.
    pub plan: RowPlan,
}

impl Prompt {
    /// The `(row0, n_rows)` pair `DeepseekV2::enable_mm_splice` /
    /// `DeepseekOcr::new_split` take.
    pub fn image_run(&self) -> (u32, u32) {
        (self.row0, self.n_rows)
    }
    /// Sequence length - what the decoder instance must be sized for.
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Build the full prompt for a single image plus the text around it.
///
/// `text_before` is what precedes the image (usually empty - the reference's
/// own prompts start with `<image>`); `text_after` is the instruction, e.g.
/// `"\n<|grounding|>Convert the document to markdown."`. Both are encoded as
/// ordinary text, so any reserved marker spelled in them (`<|grounding|>`,
/// `<|ref|>`, …) is matched atomically by the tokenizer's own special table -
/// they need no special-casing here.
///
/// `tokens_per_side` is the projector's token grid side, i.e.
/// `DeepseekOcrConfig::token_grid().0` (16 for the real 1024² view, giving 273
/// image rows). [`BOS`] is prepended, matching both the reference
/// implementation and the file's `add_bos_token`.
pub fn build_prompt(tok: &QwenBpe, text_before: &str, text_after: &str, tokens_per_side: u32) -> Result<Prompt, String> {
    let img = ImageTokens::resolve(tok)?;
    let bos = tok.special_id(BOS).ok_or_else(|| format!("this tokenizer has no reserved {BOS:?} token"))?;
    // A stray `<image>` in the TEXT would encode to the placeholder id outside
    // the spliced window, where no embedding is ever written over it - a silent
    // wrong-embedding bug. The reference splits its prompt on that marker; this
    // builder takes the two sides of that split, so the marker itself must be
    // gone by now.
    for (side, text) in [("text_before", text_before), ("text_after", text_after)] {
        if text.contains(IMAGE) {
            return Err(format!("{side} still contains {IMAGE:?}: pass the text on either side of the image, not the marker"));
        }
    }

    let before = tok.encode(text_before);
    let after = tok.encode(text_after);
    let plan = row_plan(tokens_per_side, ViewGrid::global_only());

    let mut ids = Vec::with_capacity(1 + before.len() + plan.len() + after.len());
    ids.push(bos);
    ids.extend_from_slice(&before);
    let row0 = ids.len() as u32;
    ids.extend(plan.rows.iter().map(|s| img.id_for(*s)));
    let n_rows = ids.len() as u32 - row0;
    ids.extend_from_slice(&after);

    // The invariants a caller relies on, asserted rather than assumed: the
    // image block is exactly the plan, it is ONE contiguous decoder run (which
    // is all `enable_mm_splice` can take), and no text id leaked into it.
    assert_eq!(n_rows as usize, plan.len(), "the image block must be the row plan, row for row");
    assert!(
        ids[row0 as usize..(row0 + n_rows) as usize].iter().all(|&i| img.contains(i)),
        "a non-image id leaked into the spliced window"
    );
    // `runs()` is the PROJECTOR-side mapping inside that one decoder run; the
    // newline rows break it into `tokens_per_side` runs of `tokens_per_side`,
    // and together they must cover every projector row exactly once.
    let runs = plan.runs();
    assert_eq!(
        runs.iter().map(|(_, _, n)| n).sum::<u32>(),
        plan.projector_rows(),
        "the projector runs do not cover the encoder's output"
    );

    Ok(Prompt { ids, row0, n_rows, plan })
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::gguf::GgufTokenizer;

    /// A synthetic gpt2-scheme GGUF tokenizer carrying this model's reserved
    /// strings - the fast lane for everything above that does not need the
    /// 3.1 GB checkpoint. Ids are deliberately NOT the real ones, so a test
    /// that happens to hardcode 128815 fails here.
    fn toy() -> QwenBpe {
        let gt = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("deepseek-v3".into()),
            tokens: vec![BOS.into(), EOS.into(), IMAGE.into(), GROUNDING.into(), "a".into(), "b".into(), "ab".into()],
            merges: vec!["a b".into()],
            token_types: vec![3, 3, 3, 3, 1, 1, 1],
            bos: Some(0),
            eos: Some(1),
            unk: None,
            pad: None,
        };
        QwenBpe::from_gguf(&gt).expect("toy tokenizer")
    }

    #[test]
    fn the_three_row_kinds_share_one_id() {
        let t = toy();
        let img = ImageTokens::resolve(&t).unwrap();
        assert_eq!(img.image, t.special_id(IMAGE).unwrap());
        assert_eq!((img.newline, img.separator), (img.image, img.image));
        assert_eq!(img.id_for(Src::Projector(7)), img.image);
        assert_eq!(img.id_for(Src::Newline), img.image);
        assert_eq!(img.id_for(Src::Separator), img.image);
        assert!(img.contains(img.image));
        assert!(!img.contains(img.image + 1));
    }

    /// A tokenizer without the reserved marker is refused at resolve time, not
    /// silently BPE'd into `<`, `image`, `>`.
    #[test]
    fn a_tokenizer_without_the_marker_is_refused() {
        let gt = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("qwen2".into()),
            tokens: vec!["<|endoftext|>".into(), "a".into()],
            merges: Vec::new(),
            token_types: vec![3, 1],
            bos: Some(0),
            eos: Some(0),
            unk: None,
            pad: None,
        };
        let t = QwenBpe::from_gguf(&gt).unwrap();
        let e = ImageTokens::resolve(&t).unwrap_err();
        assert!(e.contains("<image>"), "{e}");
        assert!(build_prompt(&t, "", "", 2).is_err());
    }

    /// The assembled sequence: BOS, the text before, the whole row plan, the
    /// text after - with the image block contiguous and starting where the
    /// returned `row0` says.
    #[test]
    fn the_layout_is_bos_text_image_text() {
        let t = toy();
        let img = ImageTokens::resolve(&t).unwrap();
        let g = 3;
        let p = build_prompt(&t, "ab", "a", g).unwrap();

        let before = t.encode("ab");
        let after = t.encode("a");
        assert_eq!(p.len(), 1 + before.len() + row_plan(g, ViewGrid::global_only()).len() + after.len());
        assert_eq!(p.n_rows, g * (g + 1) + 1, "the global view is g*(g+1) + 1 rows");
        assert_eq!(p.row0, 1 + before.len() as u32);
        assert_eq!(p.image_run(), (p.row0, p.n_rows));

        assert_eq!(p.ids[0], t.special_id(BOS).unwrap());
        assert_eq!(&p.ids[1..p.row0 as usize], &before[..]);
        assert_eq!(&p.ids[(p.row0 + p.n_rows) as usize..], &after[..]);
        assert!(p.ids[p.row0 as usize..(p.row0 + p.n_rows) as usize].iter().all(|&i| i == img.image));
        // The rows outside the block are text: the splice must not touch them.
        assert!(!img.contains(p.ids[0]));
        assert!(p.ids[1..p.row0 as usize].iter().all(|&i| !img.contains(i)));

        // The plan travels with the prompt, so a caller can build the 273-row
        // embedding block without recomputing the layout.
        assert_eq!(p.plan.projector_rows(), g * g);
        assert_eq!(p.plan.rows[g as usize], Src::Newline);
        assert_eq!(*p.plan.rows.last().unwrap(), Src::Separator);
    }

    /// The marker belongs to the row plan, not to the text: leaving it in the
    /// text would put a placeholder id outside the spliced window, where its
    /// embedding is never overwritten.
    #[test]
    fn the_marker_may_not_be_left_in_the_text() {
        let t = toy();
        let e = build_prompt(&t, "", "a<image>b", 2).unwrap_err();
        assert!(e.contains("text_after") && e.contains("<image>"), "{e}");
        assert!(build_prompt(&t, "<image>", "", 2).is_err());
    }

    /// The real geometry, checkpoint-free: 16 tokens per side is 273 rows, of
    /// which 256 are projector output and 17 are the two learned vectors.
    #[test]
    fn the_real_geometry_is_273_rows_of_which_256_are_projector_rows() {
        let t = toy();
        let p = build_prompt(&t, "", "a", 16).unwrap();
        assert_eq!((p.n_rows, p.plan.projector_rows(), p.plan.special_rows()), (273, 256, 17));
        assert_eq!(p.row0, 1, "no text before the image: the block starts right after BOS");
        assert_eq!(p.len(), 1 + 273 + t.encode("a").len());
        // One decoder run, 16 projector runs inside it (a newline breaks the
        // projector-side contiguity, never the decoder-side one).
        assert_eq!(p.plan.runs().len(), 16);
        assert!(p.plan.runs().iter().all(|(_, _, n)| *n == 16));
    }
}
