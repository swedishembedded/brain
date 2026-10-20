// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for loading, folding and serving
// fine-tuned adapters of any method across model families, for its clients. If
// your team needs expertise in parameter-efficient fine-tuning infrastructure
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A saved device adapter of any kind, read back and folded into a host base.
//!
//! [`crate::lora::device_adapter`] reads and folds the LoRA files every
//! serving path consumes. This module is the kind-general reader: the card names
//! the kind, the tensors are the kind's own, and the linears' dimensions come
//! from the file's `sites` record (a Kronecker factor carries no shape of its
//! own). A base is folded in place; LoKr's product is added element by element,
//! never as a dense temporary.

use std::collections::HashMap;

use super::device::param_name;
use super::kind::{lokr_factors, DeviceKind, IA3_D, LOKR_W1, LOKR_W2};
use crate::lora::device_adapter::fold_delta;

/// A saved adapter file: card fields, per-linear dimensions and tensors.
pub struct AdapterFile {
    pub kind: DeviceKind,
    pub rank: u32,
    pub alpha: f32,
    pub targets: Vec<String>,
    /// `(out, in)` of every linear the file adapts, when it records them.
    pub dims: HashMap<String, (usize, usize)>,
    pub tensors: HashMap<String, Vec<f32>>,
}

fn invalid(origin: &str, what: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{origin}: {what}"))
}

impl AdapterFile {
    pub fn read(path: &str) -> std::io::Result<AdapterFile> {
        let st = checkpoint::st::load_safetensors(path)?;
        let card = st.card().ok_or_else(|| invalid(path, "has no ModelCard".into()))?;
        let a = card.adapter.as_ref().ok_or_else(|| invalid(path, "its card has no adapter descriptor".into()))?;
        let kind = DeviceKind::from_name(&a.kind).ok_or_else(|| invalid(path, format!("adapter kind {:?} is not one of lora, ia3, lokr", a.kind)))?;
        let rank = a.rank.ok_or_else(|| invalid(path, "adapter has no rank".into()))?;
        let dims = st
            .config()
            .get("sites")
            .and_then(|s| s.as_object())
            .map(|o| o.iter().filter_map(|(k, v)| Some((k.clone(), (v.get(0)?.as_u64()? as usize, v.get(1)?.as_u64()? as usize)))).collect())
            .unwrap_or_default();
        Ok(AdapterFile { kind, rank, alpha: a.alpha.unwrap_or(rank as f32), targets: a.targets.clone().unwrap_or_default(), dims, tensors: st.tensors })
    }

    /// The factor the kind's product is scaled by: `alpha / rank`.
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// The base weights the file adapts, sorted.
    pub fn bases(&self) -> Vec<String> {
        let first = self.kind.suffixes()[0];
        let mut bases: Vec<String> = self.tensors.keys().filter_map(|n| n.strip_suffix(first)).map(str::to_string).collect();
        bases.sort();
        bases
    }

    fn tensor(&self, base: &str, suffix: &str) -> Result<&Vec<f32>, String> {
        self.tensors.get(&param_name(base, suffix)).ok_or_else(|| format!("{base}: the adapter has no {suffix} tensor"))
    }

    /// Fold the adapter into `base` (name -> row-major data), in place. Every
    /// target is checked before any weight is written, so a refused adapter
    /// leaves `base` untouched.
    pub fn fold_into(&self, base: &mut HashMap<String, Vec<f32>>) -> Result<(), String> {
        let bases = self.bases();
        if bases.is_empty() {
            return Err("the adapter holds no tensors of its kind".into());
        }
        let scale = self.scale();
        let mut plan: Vec<(String, Fold)> = Vec::new();
        for name in &bases {
            let w = base.get(name).ok_or_else(|| format!("base has no weight named {name}"))?;
            let fold = match self.kind {
                DeviceKind::Lora => {
                    let (a, b) = (self.tensor(name, super::kind::LORA_A)?, self.tensor(name, super::kind::LORA_B)?);
                    let r = self.rank as usize;
                    let (inn, out) = (a.len() / r, b.len() / r);
                    if out * inn != w.len() {
                        return Err(format!("{name}: adapter is [{out}, {inn}] but the base weight holds {} values", w.len()));
                    }
                    Fold::Lora
                }
                DeviceKind::Ia3 => {
                    let d = self.tensor(name, IA3_D)?;
                    if d.is_empty() || w.len() % d.len() != 0 {
                        return Err(format!("{name}: {} channel scales do not divide the base weight's {} values", d.len(), w.len()));
                    }
                    Fold::Ia3
                }
                DeviceKind::Lokr => {
                    let (w1, w2) = (self.tensor(name, LOKR_W1)?, self.tensor(name, LOKR_W2)?);
                    let (out, inn) = *self.dims.get(name).ok_or_else(|| format!("{name}: the file does not record this linear's dimensions"))?;
                    let (a, c, b, d) = lokr_factors(out, inn);
                    if out * inn != w.len() || w1.len() != a * b || w2.len() != c * d {
                        return Err(format!("{name}: LoKr factors {} and {} do not fit a [{out}, {inn}] weight", w1.len(), w2.len()));
                    }
                    Fold::Lokr { out, inn }
                }
            };
            plan.push((name.clone(), fold));
        }
        for (name, fold) in plan {
            let w = base.get_mut(&name).expect("validated above");
            match fold {
                Fold::Lora => fold_delta(w, self.tensor(&name, super::kind::LORA_A)?, self.tensor(&name, super::kind::LORA_B)?, self.rank as usize, scale),
                Fold::Ia3 => {
                    let d = self.tensor(&name, IA3_D)?;
                    let inn = w.len() / d.len();
                    for (o, row) in w.chunks_mut(inn).enumerate() {
                        let g = 1.0 + d[o];
                        row.iter_mut().for_each(|v| *v *= g);
                    }
                }
                Fold::Lokr { out, inn } => {
                    let (w1, w2) = (self.tensor(&name, LOKR_W1)?, self.tensor(&name, LOKR_W2)?);
                    let (a, c, b, d) = lokr_factors(out, inn);
                    for i in 0..a {
                        for p in 0..c {
                            let row = &mut w[(i * c + p) * inn..(i * c + p + 1) * inn];
                            for j in 0..b {
                                let s1 = scale * w1[i * b + j];
                                if s1 == 0.0 {
                                    continue;
                                }
                                for q in 0..d {
                                    row[j * d + q] += s1 * w2[p * d + q];
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

enum Fold {
    Lora,
    Ia3,
    Lokr { out: usize, inn: usize },
}
