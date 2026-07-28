// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training-convergence smoke: the VLM finetune loop (image-splice → decoder →
//! backward → AdamW) actually *optimizes*. Overfits a tiny decoder on a single
//! image→caption example and checks the loss collapses toward zero — empirical
//! confirmation, on top of the gradient checks, that the whole training loop
//! (not just the gradients) drives learning. Self-contained (random tiny model,
//! no checkpoint).

#[cfg(test)]
mod tests {
    use data::rng::Rng;
    use qwen::{init_weights, Qwen, QwenConfig, IGNORE};

    #[test]
    fn vlm_finetune_overfits_image_caption() {
        std::env::set_var("BRAIN_DEVICE", "cpu");
        let cfg = QwenConfig::tiny(); // vocab 23, d_model 16, 2 layers
        let init = init_weights(&cfg, 1);
        let (t, n_img, d) = (8u32, 3u32, cfg.d_model);

        let mut qwen = Qwen::new(cfg.clone(), 1, t, &init);
        qwen.enable_mm_splice(1, n_img); // image tokens occupy rows [1, 1+n_img)

        // A single fixed "image" (random projected embeds) + caption to memorize.
        let mut rng = Rng::new(2);
        let img: Vec<f32> = (0..(n_img * d) as usize).map(|_| (rng.next_f32() - 0.5) * 0.5).collect();
        // tokens: bos + 3 image placeholders + 4 caption tokens.
        let tokens = vec![0u32, 5, 5, 5, 7, 11, 17, 3];
        // targets (predict-next): bos + image rows unsupervised; the last image row
        // predicts the first caption token (image→caption), then the caption chain.
        let targets = vec![IGNORE, IGNORE, IGNORE, 7, 11, 17, 3, 9];

        let loss_at = |q: &Qwen| -> f32 {
            q.set_batch(&tokens, &targets);
            q.write_img_embeds(&img);
            q.forward()
        };
        let loss0 = loss_at(&qwen);

        // Finetune loop: zero_grads → set_batch → (splice) forward → backward → AdamW.
        for step in 0..300u32 {
            qwen.zero_grads();
            qwen.set_batch(&tokens, &targets);
            qwen.write_img_embeds(&img);
            qwen.forward();
            qwen.backward();
            qwen.adamw_step(step + 1, 3e-3, 0.0, None, 1.0);
        }
        let loss1 = loss_at(&qwen);

        eprintln!("VLM overfit: loss {loss0:.4} → {loss1:.4}");
        assert!(loss0 > 1.0, "expected a non-trivial initial loss, got {loss0}");
        assert!(loss1 < 0.05, "loss should collapse under overfitting: {loss0} → {loss1}");
    }
}
