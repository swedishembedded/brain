// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Quantization-exactness gate (Phase 4, step 2 of the real-weight parity
//! ladder): does `checkpoint::gguf`'s Q8_0 dequant produce EXACTLY the
//! values the GGUF spec says it should, on a HANDFUL of REAL tensors from
//! the real 22B LTX-2.5 distilled checkpoint?
//!
//! This is deliberately separate from port parity (`dit_parity.rs`'s
//! `ltxv_real_dit_tiny_layers_matches_reference`): a bug in EITHER dequant
//! or the port's op sequence could otherwise look like "close but not
//! exact" cosine drift, with no way to tell which one to fix. This test
//! isolates the FIRST failure mode and needs no forward pass at all - it reads a few
//! tensors' raw bytes and checks arithmetic, nothing else.
//!
//! Method: [`checkpoint::gguf::MmapGguf::raw_tensor_bytes`] returns the
//! UNDECODED on-disk block bytes for a named tensor; this file re-derives
//! the expected fp32 values with [`independent_deq_q8_0`], a from-spec
//! reimplementation written directly against the GGUF Q8_0 block layout (34
//! bytes: a little-endian f16 scale, then 32 signed int8 values,
//! `value = qs[i] as i8 as f32 * scale`) - NOT by calling
//! `checkpoint::gguf`'s own (private) `deq_q8_0` function, so this really is
//! two independent computations of the same quantity (porting.md §1's
//! self-validation discipline), compared bit-for-bit against
//! [`checkpoint::gguf::MmapGguf::tensor`]'s dequantized output (which DOES
//! go through the crate's own `deq_q8_0`).
//!
//! Real tensors chosen to cover both magnitude classes named in the task:
//! `patchify_proj.weight` (small, and per `crate::int8::is_never_quantized`
//! actually ships F32 in this checkpoint, not Q8_0 - included anyway as a
//! trivial "raw bytes == tensor() bytes" sanity check on the OTHER dtype
//! this reader must also get exactly right) plus two real Q8_0 2D
//! projections from `transformer_blocks.0` (`attn1.to_q`, `ff.net.2`) and
//! one from a late block (`transformer_blocks.47.attn1.to_v`) - covering the
//! first, an early, and the last block, per lesson #4 (don't let block 0
//! alone hide an off-by-block-index bug).
//!
//! Real-file-gated: skips loudly (`BRAIN_REQUIRE_FIXTURES=1` upgrades the
//! skip to a failure) via [`brain_testutil::skip`], the same convention
//! `crates/wan/tests/gguf_direct_real.rs` and every other real-weight test
//! in this crate already use. Pure numerical check on a few tensors' raw
//! bytes - no model forward, no dequant of the full 23.6 GB file, so this
//! runs in well under a second once the file is mmapped.

use checkpoint::gguf::MmapGguf;
use checkpoint::TensorSource;

const REPO: &str = "Lightricks/LTX-2.5";

/// The real Q8_0 DiT GGUF's path: `BRAIN_LTXV_DIT` if set, else the first
/// `*Q8_0*.gguf` under the model store's `Lightricks/LTX-2.5/` directory
/// (deliberately NOT "any `.gguf` lying around" - a `Q4_K_M` sibling may
/// also be present per this task's own briefing, and this test's from-spec
/// dequant is written for Q8_0's block layout specifically).
fn gguf_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_LTXV_DIT") {
        if !p.is_empty() {
            return Some(p);
        }
    }
    let dir = brain_testutil::model_dir(REPO)?;
    let mut found: Vec<String> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("Q8_0") && n.ends_with(".gguf")))
        // Discriminate on the file's OWN declared architecture, not on
        // its name. The model store legitimately holds several Q8_0
        // GGUFs for one repo - the DiT and, since the text encoder was
        // quantized too, Gemma-4 - and a name glob picked whichever
        // sorted first, which surfaced as an architecture mismatch deep
        // inside an importer rather than as "no fixture here".
        .filter(|p| {
            checkpoint::gguf::MmapGguf::open(&p.to_string_lossy())
                .ok()
                .and_then(|g| g.kv().get("general.architecture").and_then(|v| v.as_str()).map(str::to_string))
                .as_deref()
                == Some(ltxv::import::GGUF_ARCHITECTURE)
        })
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    found.sort();
    found.into_iter().next()
}

/// `f16` (ggml_half) little-endian bytes at `b[i..i+2]` -> f32. Reimplemented
/// here rather than imported, so this really is a SEPARATE decode path from
/// `checkpoint::gguf`'s own (private) `f16` helper.
fn f16_le(b: &[u8], i: usize) -> f32 {
    half::f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// From-spec GGUF Q8_0 dequant: 34-byte blocks, `[f16 scale][32 x int8]`,
/// `value = qs[j] as i8 as f32 * scale` - the GGML `block_q8_0` layout,
/// independently re-derived from the format definition (not copied from
/// `checkpoint::gguf::deq_q8_0`'s source, though the two necessarily agree
/// if both are correct - that agreement is exactly what this test checks).
fn independent_deq_q8_0(raw: &[u8], numel: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(numel);
    for blk in raw.chunks_exact(34) {
        let scale = f16_le(blk, 0);
        for j in 0..32 {
            out.push(blk[2 + j] as i8 as f32 * scale);
        }
    }
    out.truncate(numel);
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Assert `checkpoint::gguf`'s own dequant of `name` matches this test's
/// independent from-spec Q8_0 dequant EXACTLY (Q8_0 dequant is `int8 * f16`,
/// no accumulation, so bit-exact agreement is the right bar, not a cosine
/// tolerance - see this file's own doc).
fn assert_q8_0_exact(mg: &MmapGguf, name: &str) {
    let dtype = mg.dtype(name).unwrap_or_else(|| panic!("{name}: unknown dtype"));
    assert_eq!(dtype, "Q8_0", "{name}: expected this checkpoint to store it as Q8_0, got {dtype}");

    let (raw, ty) = mg.raw_tensor_bytes(name).unwrap_or_else(|| panic!("{name}: missing from the real header"));
    let numel = mg.numel(name).unwrap();
    let want = independent_deq_q8_0(raw, numel);

    let got = mg.tensor(name).unwrap_or_else(|| panic!("{name}: missing")).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let m = max_abs(&got, &want);
    eprintln!("{name}: ggml_type={ty}, n={numel}, max_abs(checkpoint::gguf vs independent) = {m:.3e}");
    assert_eq!(got, want, "{name}: checkpoint::gguf::deq_q8_0 disagrees with an independent from-spec Q8_0 dequant (max_abs {m:.3e})");
}

#[test]
fn deq_q8_0_matches_an_independent_from_spec_dequant_on_real_tensors() {
    let Some(path) = gguf_path() else {
        brain_testutil::skip(&format!("set BRAIN_LTXV_DIT to a real {REPO} distilled Q8_0 GGUF (none in the model store)"));
        return;
    };
    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    assert_eq!(mg.kv().get("general.architecture").and_then(|v| v.as_str()), Some("ltxv"));

    // Three real Q8_0 2D projections: block 0 (two different attn/ff
    // families) and block 47 (the LAST block - lesson #4, don't let block 0
    // alone hide an off-by-block-index bug in how names are built).
    for name in ["transformer_blocks.0.attn1.to_q.weight", "transformer_blocks.0.ff.net.2.weight", "transformer_blocks.47.attn1.to_v.weight"] {
        assert_q8_0_exact(&mg, name);
    }

    // `patchify_proj.weight` ships F32 in this checkpoint (it is on
    // `crate::int8::is_never_quantized`'s substring list) - a trivial
    // sanity check that `raw_tensor_bytes` and `tensor()` agree on the
    // OTHER dtype this reader must decode correctly too, at max_abs == 0.0
    // (F32 decode is a straight byte reinterpretation, no arithmetic to get
    // wrong, so exact-zero is the right bar here, not a tolerance).
    let name = "patchify_proj.weight";
    let dtype = mg.dtype(name).unwrap();
    assert_eq!(dtype, "F32", "{name}: expected F32 per crate::int8::is_never_quantized, got {dtype}");
    let (raw, ty) = mg.raw_tensor_bytes(name).unwrap();
    assert_eq!(ty, 0, "F32 ggml type id must be 0");
    let numel = mg.numel(name).unwrap();
    let want: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    assert_eq!(want.len(), numel);
    let got = mg.tensor(name).unwrap().unwrap();
    assert_eq!(got, want, "{name}: F32 raw-byte reinterpretation must match tensor() exactly");
    eprintln!("{name}: ggml_type=F32, n={numel}, max_abs = 0.0e0 (exact)");
}
