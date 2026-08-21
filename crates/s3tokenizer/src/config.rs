// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `AudioEncoderV2` + FSQ head hyperparameters, and the tensor manifest they
//! imply.
//!
//! v3 (`speech_tokenizer_v3.onnx`, a 12-layer MinMo encoder) is a recorded
//! follow-up - see the crate module doc's "Status" line. This config is v2
//! only.

use onnx::walk::Manifest;

/// S3Tokenizer v2's `AudioEncoderV2` + FSQ head, read verbatim from
/// `xingchensong/S3Tokenizer`'s `s3tokenizer/model_v2.py` `ModelConfig`
/// defaults.
#[derive(Clone, Copy, Debug)]
pub struct S3TokenizerConfig {
    pub n_mels: u32,
    pub n_audio_state: u32,
    pub n_audio_head: u32,
    pub n_audio_layer: u32,
    /// `3**8`, the FSQ codebook size (8 dims, 3 levels each).
    pub n_codebook_size: u32,
}

impl S3TokenizerConfig {
    /// CosyVoice 2's `speech_tokenizer_v2.onnx`.
    pub fn v2() -> S3TokenizerConfig {
        S3TokenizerConfig {
            n_mels: 128,
            n_audio_state: 1280,
            n_audio_head: 20,
            n_audio_layer: 6,
            n_codebook_size: 3u32.pow(8),
        }
    }

    pub fn head_dim(&self) -> u32 {
        self.n_audio_state / self.n_audio_head
    }

    /// The MLP's hidden width - `ResidualAttentionBlock` always widens 4x.
    pub fn mlp_dim(&self) -> u32 {
        self.n_audio_state * 4
    }

    /// FSQ dimensionality: `n_codebook_size == fsq_levels ** fsq_dims`, with
    /// `fsq_levels = 3` fixed by `FSQCodebook(level=3)` - so `fsq_dims` is
    /// `log3(n_codebook_size)`, computed rather than hardcoded so a
    /// misconfigured `n_codebook_size` fails loudly instead of silently
    /// projecting to the wrong width.
    pub fn fsq_dims(&self) -> u32 {
        let mut n = self.n_codebook_size;
        let mut dims = 0u32;
        while n > 1 {
            assert!(n.is_multiple_of(3), "S3TokenizerConfig: n_codebook_size {} is not a power of 3", self.n_codebook_size);
            n /= 3;
            dims += 1;
        }
        dims
    }

    /// Every tensor the encoder + FSQ head read, with the shape the ONNX
    /// graph itself stores it at.
    ///
    /// Linear weights are the one surprise: `speech_tokenizer_v2.onnx` traces
    /// each `torch.nn.Linear` as a bare `MatMul(x, W)` node (not `Gemm` with
    /// `transB`), which only works if the exporter pre-transposed `W` to
    /// `[in, out]` - the opposite of `model::hostmath`'s `[out, in]`
    /// `matvec`/`linear_rows` convention. This manifest states the
    /// checkpoint's own shape ([`crate::import`] binds positionally against
    /// exactly this); [`crate::model::S3TokenizerWeights::from_tensors`] does
    /// the `[in, out] -> [out, in]` transpose once, at load time, so nothing
    /// downstream repeats it per forward.
    pub fn tensor_manifest(&self) -> Manifest {
        let (d, h, mel) = (self.n_audio_state as usize, self.mlp_dim() as usize, self.n_mels as usize);
        let mut m: Manifest = vec![
            ("conv1.weight".into(), vec![d, mel, 3]),
            ("conv1.bias".into(), vec![d]),
            ("conv2.weight".into(), vec![d, d, 3]),
            ("conv2.bias".into(), vec![d]),
        ];
        for b in 0..self.n_audio_layer as usize {
            let p = format!("blocks.{b}");
            m.push((format!("{p}.attn_ln.weight"), vec![d]));
            m.push((format!("{p}.attn_ln.bias"), vec![d]));
            // ONNX MatMul weight layout: [in, out] - see the doc comment above.
            m.push((format!("{p}.attn.query.weight"), vec![d, d]));
            m.push((format!("{p}.attn.query.bias"), vec![d]));
            m.push((format!("{p}.attn.key.weight"), vec![d, d])); // no bias (Whisper convention)
            m.push((format!("{p}.attn.value.weight"), vec![d, d]));
            m.push((format!("{p}.attn.value.bias"), vec![d]));
            m.push((format!("{p}.attn.out.weight"), vec![d, d]));
            m.push((format!("{p}.attn.out.bias"), vec![d]));
            m.push((format!("{p}.attn.fsmn_block.weight"), vec![d, 1, 31])); // depthwise, no bias
            m.push((format!("{p}.mlp_ln.weight"), vec![d]));
            m.push((format!("{p}.mlp_ln.bias"), vec![d]));
            m.push((format!("{p}.mlp.fc1.weight"), vec![d, h]));
            m.push((format!("{p}.mlp.fc1.bias"), vec![h]));
            m.push((format!("{p}.mlp.fc2.weight"), vec![h, d]));
            m.push((format!("{p}.mlp.fc2.bias"), vec![d]));
        }
        let fsq = self.fsq_dims() as usize;
        m.push(("quantizer.project_down.weight".into(), vec![d, fsq]));
        m.push(("quantizer.project_down.bias".into(), vec![fsq]));
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 102 = the real `speech_tokenizer_v2.onnx`'s initializer count
    /// (`dump_graph resources/cosyvoice/weights/speech_tokenizer_v2.onnx`):
    /// 4 (conv1/conv2) + 6*16 (blocks) + 2 (project_down).
    #[test]
    fn manifest_counts_match_the_released_graph() {
        assert_eq!(S3TokenizerConfig::v2().tensor_manifest().len(), 102);
    }

    #[test]
    fn manifest_names_are_unique() {
        let m = S3TokenizerConfig::v2().tensor_manifest();
        let mut names: Vec<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate canonical tensor name");
    }

    #[test]
    fn fsq_dims_matches_the_configured_codebook_size() {
        assert_eq!(S3TokenizerConfig::v2().fsq_dims(), 8);
    }

    #[test]
    fn head_dim_and_mlp_dim_match_the_reference() {
        let cfg = S3TokenizerConfig::v2();
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.mlp_dim(), 5120);
    }
}
