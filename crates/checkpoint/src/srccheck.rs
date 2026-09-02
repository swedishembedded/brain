// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structural enforcement that every [`crate::TensorSource`] read path agrees.
//!
//! Swedish Embedded AB implements checkpoint-loading correctness tooling for
//! its clients. If your team needs expertise in catching a silently-wrong
//! model import - a transform applied on one read path but not another -
//! then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! [`TensorSource`] has four ways to read the same tensor: [`with_tensor`],
//! [`with_tensor_chunks`], [`raw_words`] and [`raw_blocks`]. A source that
//! transforms VALUES on read (`qwen35::int8_gguf_resident::SsmALogFix`
//! un-negates-and-un-exponentiates one leaf llama.cpp stores pre-applied) has
//! to apply that transform on every one of them, or refuse the ones it
//! cannot - a zero-copy path that forgets is a silent wrong-weights bug, not
//! a crash. The concrete incident this guards against: importing a checkpoint
//! whose on-disk `ssm_a` leaf is stored as `-exp(A_log)` verbatim through a
//! path that skipped the un-negate-and-un-exponentiate step once produced a
//! decay gate up to 260x too strong - fluent-looking output, no error
//! anywhere, and a model that quietly stopped integrating context.
//!
//! [`assert_read_paths_agree`] is the mechanical version of "did I remember
//! to fix every path": decode a named tensor through whichever of the four
//! the source offers (declining is always a valid answer for `raw_words`/
//! `raw_blocks` - only *disagreeing* is a bug) and assert they produce
//! identical values. A future wrapper type should call this in its own test
//! suite over a source where the paths COULD disagree (a plain in-memory
//! `HashMap` never zero-copies to anything but itself, so the bug is only
//! reachable over a real quantized GGUF - see
//! `qwen35::int8_gguf_resident::tests::raw_blocks_never_lends_the_
//! untransformed_a_log_blocks` for the shape).
//!
//! [`with_tensor`]: crate::TensorSource::with_tensor
//! [`with_tensor_chunks`]: crate::TensorSource::with_tensor_chunks
//! [`raw_words`]: crate::TensorSource::raw_words
//! [`raw_blocks`]: crate::TensorSource::raw_blocks

use crate::TensorSource;

/// Decode `name` (`numel` elements) through every read path `src` offers and
/// assert they agree bit-for-bit. `with_tensor` and `with_tensor_chunks` are
/// required to agree (every source must implement `with_tensor`, and the
/// chunked path's default literally IS `with_tensor` under the hood - see its
/// doc - so a disagreement there means an override broke that identity).
/// `raw_words` and `raw_blocks` are each checked ONLY when the source offers
/// them (returning `None` is always a valid, non-buggy answer for either).
///
/// Panics naming which path disagreed, and with which values, rather than a
/// bare `assert_eq!` - the whole point of this function is to be run once per
/// source type in that type's own test suite and read by a human when it goes
/// red.
pub fn assert_read_paths_agree(src: &dyn TensorSource, name: &str, numel: usize) {
    let mut via_with_tensor: Option<Vec<f32>> = None;
    let found = src.with_tensor(name, &mut |d| via_with_tensor = Some(d.to_vec()));
    assert!(found, "srccheck: '{name}' not found via with_tensor");
    let via_with_tensor = via_with_tensor.expect("with_tensor reported found, so the callback ran");
    assert_eq!(
        via_with_tensor.len(),
        numel,
        "srccheck: '{name}' with_tensor produced {} elements, caller expected {numel}",
        via_with_tensor.len()
    );

    let mut via_chunks = vec![0.0f32; numel];
    let mut covered = 0usize;
    // A small, non-power-of-two chunk size: large enough that a handful of
    // chunks cover a real tensor, small enough that a source which secretly
    // ignores `max_elems` and hands back one giant chunk is still exercised
    // through the same reassembly logic as a genuinely-chunked one.
    let chunk_found = src.with_tensor_chunks(name, 7, &mut |off, chunk| {
        let off = off as usize;
        via_chunks[off..off + chunk.len()].copy_from_slice(chunk);
        covered += chunk.len();
    });
    assert!(chunk_found, "srccheck: '{name}' not found via with_tensor_chunks");
    assert_eq!(covered, numel, "srccheck: '{name}' with_tensor_chunks covered {covered} of {numel} elements - gap or overlap");
    assert_eq!(via_chunks, via_with_tensor, "srccheck: '{name}' with_tensor_chunks disagrees with with_tensor");

    if let Some(words) = src.raw_words(name) {
        let via_words: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();
        assert_eq!(via_words, via_with_tensor, "srccheck: '{name}' raw_words disagrees with with_tensor");
    }

    if let Some((layout, bytes)) = src.raw_blocks(name) {
        assert_eq!(
            layout.numel, numel,
            "srccheck: '{name}' raw_blocks reports numel {}, caller expected {numel}",
            layout.numel
        );
        // Through the SAME `dequantize` every `MmapGguf::with_tensor` call
        // uses (`crate::gguf::dequantize` is the one decoder both paths
        // share) - this is precisely what catches a wrapper that applies a
        // transform on `with_tensor` but forgot to refuse (or reapply it)
        // here: `with_tensor` would report the transformed values while this
        // reconstructs the PRE-transform ones from the raw blocks, and the
        // two would disagree.
        let via_blocks = crate::gguf::dequantize(layout.ty.id(), bytes, numel)
            .unwrap_or_else(|e| panic!("srccheck: '{name}' raw_blocks dequant failed: {e}"));
        assert_eq!(via_blocks, via_with_tensor, "srccheck: '{name}' raw_blocks disagrees with with_tensor");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A source where every path genuinely agrees passes cleanly - the
    /// non-buggy baseline this function must not false-positive on.
    #[test]
    fn agreeing_paths_pass() {
        let mut m: HashMap<String, Vec<f32>> = HashMap::new();
        m.insert("w".to_string(), vec![1.0, -2.5, 3.25, 0.0]);
        assert_read_paths_agree(&m, "w", 4);
    }

    /// A `raw_words` override that disagrees with `with_tensor` must be
    /// caught, by name - proves this is a real check, not a tautology that
    /// always passes because every implementor happens to agree by
    /// construction.
    #[test]
    #[should_panic(expected = "raw_words disagrees")]
    fn a_disagreeing_raw_words_is_caught() {
        struct Lying(Vec<f32>);
        impl TensorSource for Lying {
            fn with_tensor(&self, _name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
                f(&self.0);
                true
            }
            fn raw_words(&self, _name: &str) -> Option<&[u32]> {
                // Deliberately wrong: lends a DIFFERENT value than with_tensor.
                static WRONG: [u32; 1] = [0u32];
                Some(&WRONG)
            }
        }
        assert_read_paths_agree(&Lying(vec![1.0]), "w", 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod gguf_backed {
        use super::*;
        use crate::gguf::{MmapGguf, T_Q8_0};
        use crate::gguf_write::{write, TensorOut};

        /// The real target: an `MmapGguf`-backed source, where `raw_blocks`
        /// is actually exercised (a plain `HashMap` never has one).
        #[test]
        fn a_real_gguf_source_agrees_on_every_path() {
            let block = crate::quant::quantize_par(T_Q8_0, &[3.0f32; 32]).unwrap();
            let path = std::env::temp_dir().join(format!("brain-srccheck-{}.gguf", std::process::id()));
            let path = path.to_str().unwrap().to_string();
            write(&path, &[], &[TensorOut { name: "w".to_string(), shape: vec![32], ty: T_Q8_0, data: block }], 32).unwrap();
            let mg = MmapGguf::open(&path).unwrap();
            assert_read_paths_agree(&mg, "w", 32);
            std::fs::remove_file(&path).ok();
        }
    }
}
