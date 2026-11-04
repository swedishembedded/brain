// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The quantizable linears of a model, from the model's own declaration.

use model::Model;

/// One stored weight tensor a tier applies to: the unit a precision is chosen
/// for. Several adapter sites can share one tensor (a fused projection whose
/// rows or columns several linears read); they are one site here, because
/// one stored tensor has one storage tier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site {
    /// The stored tensor's full name, as [`Model::read_weight`] knows it.
    pub name: String,
    /// Rows (outputs) and the contraction width of the stored tensor.
    pub n: usize,
    pub k: usize,
    pub layer: Option<usize>,
    /// The linears this tensor holds (`"wq"`, `"up"`, ...), in declaration order.
    pub roles: Vec<&'static str>,
}

impl Site {
    pub fn numel(&self) -> usize {
        self.n * self.k
    }
}

/// The model's quantizable tensors, from [`Model::adapter_sites`]: every
/// linear a model declares, grouped by the tensor that stores it. The stored
/// width is the site's row stride, and the row count follows from the stored
/// element count, which must be a whole number of rows.
///
/// A model that declares no linear sites has nothing this search can place,
/// and is refused by name rather than treated as an empty plan.
pub fn sites<M: Model>(m: &M) -> Result<Vec<Site>, String> {
    let declared = m.adapter_sites();
    if declared.is_empty() {
        return Err("the model declares no linear sites (Model::adapter_sites), so there is nothing to place a tier on".into());
    }
    let mut out: Vec<Site> = Vec::new();
    for site in declared {
        if let Some(existing) = out.iter_mut().find(|s| s.name == site.name) {
            if existing.k != site.spec.row_stride {
                return Err(format!("{}: its linears disagree on the stored width ({} and {})", site.name, existing.k, site.spec.row_stride));
            }
            existing.roles.push(site.leaf);
            continue;
        }
        let k = site.spec.row_stride;
        let numel = m.read_weight(&site.name).len();
        if k == 0 || numel == 0 || !numel.is_multiple_of(k) {
            return Err(format!("{}: {numel} stored elements are not whole rows of width {k}", site.name));
        }
        out.push(Site { name: site.name, n: numel / k, k, layer: site.layer, roles: vec![site.leaf] });
    }
    Ok(out)
}
