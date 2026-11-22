// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements image-text fine-tuning of multimodal mixture-of-experts models for its clients. If
// your team needs expertise in training a vision tower and a looped decoder together from a published release then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Image-text training: the vision tower's features spliced into the decoder's embedding input, and the decoder's
//! gradient at those rows carried back into the tower.
//!
//! One step ([`ImageTextTrainer::step`]): the tower encodes the images for training
//! ([`VisionTower::encode_train`]), the features fill the image-token positions of the row
//! ([`crate::vision::splice_embeds`], the reference's `masked_scatter` after the embedding norm), the decoder runs its
//! forward and backward over the spliced rows ([`Inkling2Train::set_image_rows`]), the gradient at the image rows is
//! taken out of the embedding gradient ([`model::serve::PromptEmbeds::take_row_gradients`]: the placeholder token's own
//! embedding row gets none of it) and the tower differentiates it ([`VisionTower::backward`]). The tower is frozen by
//! default; its adapters (and the merger's) train when the tower was built with them, under the same optimizer
//! settings as the decoder (each part clipped to its own norm). Both run on the decoder's device.

use gpu_core::Gpu;
use imaging::RegionImage;
use model::Adam;
use optim::Optim;

use super::Inkling2Train;
use crate::vision::VisionTower;

/// A decoder and a vision tower trained together on image-text rows.
pub struct ImageTextTrainer {
    decoder: Inkling2Train,
    tower: VisionTower,
    /// The tower adapters' optimizer and the tower's own trained tensors' (each records its store's update once).
    lora_opt: Optim,
    base_opt: Optim,
}

impl ImageTextTrainer {
    /// Train `decoder` with `tower`'s features spliced in. The tower must be built on the decoder's device
    /// ([`Inkling2Train::gpu`]) with [`VisionTower::new_trainable`] (frozen: `|_| false` and no adapters, or with
    /// adapters); its output width must be the decoder's.
    pub fn new(decoder: Inkling2Train, tower: VisionTower) -> Result<ImageTextTrainer, String> {
        if tower.out_width() != decoder.config().hidden as usize {
            return Err(format!("inkling2: the tower's features are {} wide, the decoder's embeddings {}", tower.out_width(), decoder.config().hidden));
        }
        let g = decoder.gpu();
        let need = |name: &str| g.kernel_index(name).ok_or_else(|| format!("inkling2: kernel {name} is not registered"));
        let optim = || -> Result<Optim, String> { Ok(Optim::new(need("adamw")?, need("gradnorm_sq")?, need("grad_scale")?, need("clip_coef")?, need("grad_scale_buf")?)) };
        let (lora_opt, base_opt) = (optim()?, optim()?);
        Ok(ImageTextTrainer { decoder, tower, lora_opt, base_opt })
    }

    pub fn decoder(&self) -> &Inkling2Train {
        &self.decoder
    }

    pub fn tower(&self) -> &VisionTower {
        &self.tower
    }

    fn g(&self) -> &Gpu {
        self.decoder.gpu()
    }

    /// Zero the decoder's and the tower's gradients.
    pub fn zero_grads(&self) {
        model::Model::zero_grads(&self.decoder);
        self.tower.zero_grads(self.g());
    }

    /// One forward and backward of a batch whose `tokens` (`[b * t]`, every image already expanded to one
    /// `image_token` per region) hold the regions of `images` in order: the loss, every gradient added.
    pub fn step(&self, tokens: &[u32], targets: &[u32], images: &[RegionImage], image_token: u32) -> Result<f32, String> {
        let g = self.g();
        let (features, pass) = self.tower.encode_train(g, images)?;
        let embeds = crate::vision::splice_embeds(tokens, image_token, &features)?;
        self.decoder.set_image_rows(Some(embeds))?;
        model::Model::set_batch(&self.decoder, model::Batch::Lm { tokens, targets });
        let loss = model::Model::forward(&self.decoder);
        model::Model::backward(&self.decoder);
        let d_rows = self.decoder.take_image_gradients().ok_or("inkling2: the backward left no image-row gradient")?;
        self.decoder.set_image_rows(None)?;
        self.tower.backward(g, pass, &d_rows)?;
        Ok(loss)
    }

    /// One AdamW step of the decoder's trained tensors and the tower's adapters (when it has them), each clipped to
    /// `clip` over its own gradients.
    pub fn update(&self, t: u32, lr: f32, wd: f32, adam: Adam, clip: Option<f32>) {
        model::Model::adamw_step(&self.decoder, t, lr, wd, adam, clip, 1.0);
        if let Some(store) = self.tower.lora_store() {
            self.lora_opt.step(self.g(), store, t, lr, wd, adam, clip, 1.0);
        }
        if !self.tower.store().opt_params().is_empty() {
            self.base_opt.step(self.g(), self.tower.store(), t, lr, wd, adam, clip, 1.0);
        }
    }
}
