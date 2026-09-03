// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training-convergence smoke: the LoRA fine-tuning loop this session added
//! (`crate::finetune::run`'s per-step shape: `Qwen3Vl::zero_grads` ->
//! `Qwen3Vl::forward` -> `Qwen3Vl::backward` -> `Qwen3Vl::adamw_step`) actually
//! *optimizes*, over a HANDFUL of distinct synthetic image/caption pairs, not
//! just one memorized example. Mirrors `fastvlm::train_smoke`'s own shape
//! (tiny random model, no checkpoint, no tokenizer, no real image codec) but
//! composed through the real vision tower + PatchMerger + M-RoPE splice, not
//! just the bare decoder `fastvlm::train_smoke` exercises via
//! `write_img_embeds` directly.

#[cfg(test)]
mod tests {
    use data::rng::Rng;
    use qwen3::{IGNORE, LoraCfg, QwenConfig};

    use crate::config::VisionConfig;
    use crate::model::{DecoderBuild, Qwen3Vl};

    const IMG: u32 = 7;

    /// Three distinct synthetic "images" (random patch pixels) each paired
    /// with a distinct short caption (fixed token ids - no tokenizer
    /// involved, same simplification `fastvlm::train_smoke` makes).
    struct Sample {
        pixels: Vec<f32>,
        tokens: Vec<u32>,
        targets: Vec<u32>,
    }

    #[test]
    fn lora_finetune_loop_decreases_loss_over_several_captioned_images() {
        std::env::set_var("BRAIN_DEVICE", "cpu");

        let vcfg = VisionConfig {
            depth: 2,
            hidden: 16,
            num_heads: 2,
            intermediate: 32,
            patch_size: 2,
            temporal_patch_size: 1,
            spatial_merge_size: 2,
            num_position_embeddings: 16,
            out_hidden_size: 20,
            in_channels: 2,
            deepstack_indexes: vec![],
            tokens_per_second: 2,
        };
        let dcfg = QwenConfig {
            vocab: 18,
            block_size: 16,
            n_layers: 2,
            d_model: 20,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 8, // even: M-RoPE pairs head_dim/2 cos/sin channels
            d_ff: 32,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            max_position_embeddings: 16,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            // Same target set `crate::finetune::run` actually trains with
            // (attn + MLP projections), not the attn-only
            // `LoraCfg::attn` gradcheck shortcut - this smoke test's job is
            // proving the production loop converges, so it should have the
            // production loop's own capacity.
            lora: Some(LoraCfg { rank: 8, alpha: 16.0, targets: crate::finetune::lora_targets() }),
        };

        // --- vision + merger weights (frozen; a fixed random tower) ---
        let (c, pv, mlp) = (vcfg.hidden as usize, vcfg.patch_vec_dim() as usize, vcfg.intermediate as usize);
        let mut rng = Rng::new(1);
        let rand_vec = |n: usize, rng: &mut Rng| -> Vec<f32> { (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect() };
        let mut vweights = std::collections::HashMap::new();
        vweights.insert("patch_embed.weight".to_string(), rand_vec(c * pv, &mut rng));
        vweights.insert("patch_embed.bias".to_string(), rand_vec(c, &mut rng));
        vweights.insert("pos_embed".to_string(), rand_vec(vcfg.num_position_embeddings as usize * c, &mut rng));
        for b in 0..vcfg.depth {
            vweights.insert(format!("blocks.{b}.norm1.weight"), vec![1.0; c]);
            vweights.insert(format!("blocks.{b}.norm1.bias"), rand_vec(c, &mut rng));
            vweights.insert(format!("blocks.{b}.qkv.weight"), rand_vec(3 * c * c, &mut rng));
            vweights.insert(format!("blocks.{b}.qkv.bias"), rand_vec(3 * c, &mut rng));
            vweights.insert(format!("blocks.{b}.proj.weight"), rand_vec(c * c, &mut rng));
            vweights.insert(format!("blocks.{b}.proj.bias"), rand_vec(c, &mut rng));
            vweights.insert(format!("blocks.{b}.norm2.weight"), vec![1.0; c]);
            vweights.insert(format!("blocks.{b}.norm2.bias"), rand_vec(c, &mut rng));
            vweights.insert(format!("blocks.{b}.fc1.weight"), rand_vec(mlp * c, &mut rng));
            vweights.insert(format!("blocks.{b}.fc1.bias"), rand_vec(mlp, &mut rng));
            vweights.insert(format!("blocks.{b}.fc2.weight"), rand_vec(c * mlp, &mut rng));
            vweights.insert(format!("blocks.{b}.fc2.bias"), rand_vec(c, &mut rng));
        }
        let merged = c * 4;
        let mut mweights = std::collections::HashMap::new();
        mweights.insert("ln.weight".to_string(), vec![1.0; c]);
        mweights.insert("ln.bias".to_string(), rand_vec(c, &mut rng));
        mweights.insert("fc1.weight".to_string(), rand_vec(merged * merged, &mut rng));
        mweights.insert("fc1.bias".to_string(), rand_vec(merged, &mut rng));
        mweights.insert("fc2.weight".to_string(), rand_vec(20 * merged, &mut rng));
        mweights.insert("fc2.bias".to_string(), rand_vec(20, &mut rng));

        let dweights = qwen3::init_weights(&dcfg, 2); // seeds .lora_a/.lora_b too (B=0)

        // Fixed prompt shape shared by every sample: 1 text + 4 image (2x2
        // merged) + 3 distinct caption tokens.
        let tokens_prefix = vec![1u32, IMG, IMG, IMG, IMG];
        let (image_row0, n_visual) = (1u32, 4u32);
        let seq_len = 8u32; // prefix(5) + 3 caption tokens

        let model = Qwen3Vl::new(
            vcfg.clone(),
            dcfg,
            vweights,
            mweights,
            vec![],
            &dweights,
            seq_len,
            IMG,
            image_row0,
            n_visual,
            [2, 1, 1],
            DecoderBuild::Batched,
        );

        // Three distinct captioned "images": different pixels AND different
        // captions, so the model must actually condition on both, not just
        // memorize one fixed target.
        let pv_total = (16 * vcfg.patch_vec_dim()) as usize;
        let captions: [[u32; 3]; 3] = [[8, 9, 10], [11, 12, 13], [14, 15, 16]];
        let samples: Vec<Sample> = captions
            .iter()
            .enumerate()
            .map(|(i, cap)| {
                let mut prng = Rng::new(100 + i as u64);
                let pixels: Vec<f32> = (0..pv_total).map(|_| prng.next_f32() - 0.5).collect();
                let mut tokens = tokens_prefix.clone();
                tokens.extend_from_slice(cap);
                let mut targets = vec![IGNORE; seq_len as usize];
                // target[i] predicts tokens[i+1]: the last prefix token
                // predicts cap[0], then cap[0]->cap[1], cap[1]->cap[2].
                targets[tokens_prefix.len() - 1] = cap[0];
                targets[tokens_prefix.len()] = cap[1];
                targets[tokens_prefix.len() + 1] = cap[2];
                Sample { pixels, tokens, targets }
            })
            .collect();

        let avg_loss = |m: &Qwen3Vl| -> f32 {
            let sum: f32 = samples.iter().map(|s| m.forward(&s.tokens, &s.targets, (4, 4), &s.pixels)).sum();
            sum / samples.len() as f32
        };
        let loss0 = avg_loss(&model);

        for step in 0..800u32 {
            let s = &samples[step as usize % samples.len()];
            model.zero_grads();
            model.forward(&s.tokens, &s.targets, (4, 4), &s.pixels);
            model.backward();
            model.adamw_step(step + 1, 1e-2, 0.0, Some(1.0), 1.0);
        }
        let loss1 = avg_loss(&model);

        eprintln!("qwen3vl LoRA finetune: avg loss {loss0:.4} -> {loss1:.4}");
        assert!(loss0 > 0.5, "expected a non-trivial initial loss, got {loss0}");
        // A real, honest bar for THIS test's shape, not an overfit-to-zero
        // claim: a from-scratch random tiny decoder, a from-scratch random
        // rank-8 LoRA (its `A` starts near-random and only moves as `B`
        // moves off its zero init - see `crate::model`'s
        // `lora_delta_gradient_matches_finite_difference` doc), cycling
        // across THREE distinct image/caption pairs sharing one fixed prompt
        // shape. That is a materially harder optimization landscape than
        // `fastvlm::train_smoke`'s single memorized example under FULL
        // fine-tuning (which this crate's own `Qwen3Vl` composite also
        // reaches near-zero loss on - confirmed manually while designing
        // this test, not asserted here since that is not what LoRA ships).
        // A double-digit percent drop is what proves the loop - `zero_grads`
        // -> `forward` -> `backward` -> `adamw_step`, `crate::finetune::run`'s
        // own per-step shape - is a real optimizer over a real gradient, not
        // a no-op; demanding near-zero loss here would instead be testing
        // this ONE random seed's local-minimum luck.
        assert!(loss1 < loss0 * 0.9, "loss should decrease under LoRA finetuning across {} samples: {loss0} -> {loss1}", samples.len());
    }
}
