// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The adapter a Ripple LoRA run produces: written in brain's adapter file, read back for serving or folding, and
//! exported as a PEFT adapter directory the `peft` library applies to the Hugging Face checkpoint.
//!
//! Swedish Embedded AB implements solutions for moving fine-tuned adapters between training stacks, serving
//! engines and the ecosystem's tools without a converter per model family, for its clients. If your team needs
//! expertise in parameter-efficient fine-tuning interchange then you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! The file is the shared one (`model::lora::device_adapter`): a model card naming the base, and per site
//! `{base weight}.lora_a` `[rank, in]` and `.lora_b` `[out, rank]`; the tensor naming lives in `model::adapter`.
//! Ripple's weight names already are the Hugging Face ones, so a PEFT module path is a base weight name without its
//! `.weight`.

use std::path::Path;

pub use model::lora::device_adapter::{fold_adapter_into, parse_adapter, read_adapter, AdapterSite, DeviceAdapter};

use super::Ripple;

/// The family tag an adapter file's card carries.
pub const FAMILY: &str = "ricursive";

impl Ripple {
    /// Write only this model's adapter factors (never the base) to `path`, with a card naming `card_id`, the
    /// `base_id` it was trained on and the dataset, and the run that produced it when `training` is given.
    pub fn save_adapter(&self, path: &str, card_id: &str, base_id: &str, dataset_id: Option<&str>, training: Option<checkpoint::st::TrainingProvenance>) -> std::io::Result<()> {
        let lora = self.cfg.lora.as_ref().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "ricursive: this model was not built with an adapter"))?;
        model::lora::device_adapter::save_lora_with_lineage(path, self, lora, card_id, base_id, FAMILY, dataset_id, training)
    }

    /// Replace this model's adapter factors by those of `adapter` (to continue training an adapter, or to evaluate
    /// one): every pair is checked against the model's sites and its rank must be the model's.
    pub fn load_adapter(&self, adapter: &DeviceAdapter) -> Result<(), String> {
        use model::adapter::sites::LoraSites;
        let lora = self.cfg.lora.as_ref().ok_or("ricursive: this model was not built with an adapter")?;
        if adapter.rank != lora.rank {
            return Err(format!("ricursive: the adapter has rank {}, this model trains rank {}", adapter.rank, lora.rank));
        }
        model::adapter::sites::validate_with_saved(adapter, &self.cfg.lora_sites(), &self.cfg.trained_in_full()).map_err(|e| format!("ricursive load_adapter: {e}"))?;
        for pair in &adapter.sites {
            let (a, b) = model::adapter::device::lora_names(&pair.base);
            if !self.ps.weight.contains_key(&a) {
                return Err(format!("ricursive load_adapter: this model's adapter does not target {}", pair.base));
            }
            self.gpu.write_f32(self.ps.w(&a), &pair.a);
            self.gpu.write_f32(self.ps.w(&b), &pair.b);
        }
        for t in &adapter.saved {
            self.gpu.write_f32(self.ps.w(&t.base), &t.data);
        }
        Ok(())
    }
}

/// Write the adapter file at `adapter_path` as a PEFT adapter directory in `out`: `adapter_model.safetensors` under
/// PEFT's names and an `adapter_config.json` with the rank, `lora_alpha` and target modules, so
/// `peft.PeftModel.from_pretrained` over the Hugging Face checkpoint applies exactly `(alpha / rank) B A`. `base_model`
/// names that checkpoint; `None` takes the base the adapter's own card records.
/// Whether `adapter` fits a model of configuration `cfg`: its rank is positive, and every pair names a linear this
/// model has (a layer past its depth, `g_proj` on a model without the gate and any name that is no adapter site are
/// refused naming the pair), and every full tensor it carries is one the engine serves per row
/// ([`super::TrainConfig::saved_sites`]: the norm gains and the value-residual gates). The same check the engine makes
/// before it uploads a factor.
pub fn check_fits(cfg: &crate::RicursiveConfig, adapter: &DeviceAdapter) -> Result<(), String> {
    use model::adapter::sites::LoraSites;
    if adapter.rank == 0 {
        return Err("ricursive: the adapter has rank 0".into());
    }
    let cfg = super::TrainConfig::new(cfg.clone(), 1);
    model::adapter::sites::validate_with_saved(adapter, &cfg.lora_sites(), &cfg.saved_sites()).map_err(|e| format!("ricursive: {e}"))
}

pub fn export_peft(adapter_path: &str, out: &Path, base_model: Option<&str>) -> Result<(), String> {
    use model::adapter::peft::{adapter_pairs, PeftAdapter};
    let module = |base: &str| base.strip_suffix(".weight").map(str::to_string);
    let (loras, base) = adapter_pairs(adapter_path, &module)?;
    let base_model = base_model.map(str::to_string).or(base).ok_or_else(|| format!("{adapter_path}: no base model given and none on its card"))?;
    // PEFT's `modules_to_save` holds whole modules: a norm is one (`<module>.modules_to_save.default.weight`). A bare
    // parameter (the value-residual gate) has no PEFT form, so an adapter carrying one is refused rather than exported
    // without it.
    let (mut modules_to_save, mut saved) = (Vec::new(), Vec::new());
    for t in read_adapter(adapter_path).map_err(|e| format!("{adapter_path}: {e}"))?.saved {
        let module = module(&t.base).ok_or_else(|| format!("{adapter_path}: {} is a bare parameter, which PEFT's modules_to_save cannot hold; the brain adapter file carries it, a PEFT export cannot", t.base))?;
        saved.push((format!("{module}.modules_to_save.default.weight"), vec![t.data.len() as u64], t.data));
        modules_to_save.push(module);
    }
    PeftAdapter { base_model, task_type: Some("CAUSAL_LM".into()), loras, modules_to_save, saved }.write(out)
}
