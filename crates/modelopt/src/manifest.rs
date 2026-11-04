// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The quantizable linears of a model, from the model's own declaration.

use model::Model;

/// One stored weight tensor: its full name, as [`Model::read_weight`] knows
/// it, its rows (outputs) and its contraction width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub n: usize,
    pub k: usize,
}

/// The unit a storage tier is chosen for: one stored tensor, or the tensors
/// the deployed engine fuses into one weight ([`Model::coupled_tensors`]),
/// which must share a tier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site {
    /// The first tensor's name: what reports call the site.
    pub name: String,
    pub tensors: Vec<Tensor>,
    pub layer: Option<usize>,
    /// The linears the site holds (`"wq"`, `"up"`, ...), in declaration order.
    pub roles: Vec<&'static str>,
}

impl Site {
    pub fn numel(&self) -> usize {
        self.tensors.iter().map(|t| t.n * t.k).sum()
    }
}

/// The model's quantizable tensors, from [`Model::linear_sites`] (grouped by
/// the tensor that stores each linear: the stored width is the site's row
/// stride, and the row count follows from the stored element count), then
/// joined into the groups [`Model::coupled_tensors`] declares.
///
/// A model that declares no linear sites has nothing a search can place, and a
/// coupled group naming a tensor that is not a linear site is a declaration
/// error; both are refused by name rather than treated as an empty plan.
pub fn sites<M: Model>(m: &M) -> Result<Vec<Site>, String> {
    let declared = m.linear_sites();
    if declared.is_empty() {
        return Err("the model declares no linear sites (Model::linear_sites), so there is nothing to place a tier on".into());
    }
    let mut out: Vec<Site> = Vec::new();
    for site in declared {
        if let Some(existing) = out.iter_mut().find(|s| s.name == site.name) {
            if existing.tensors[0].k != site.spec.row_stride {
                return Err(format!("{}: its linears disagree on the stored width ({} and {})", site.name, existing.tensors[0].k, site.spec.row_stride));
            }
            existing.roles.push(site.leaf);
            continue;
        }
        let k = site.spec.row_stride;
        let numel = m.read_weight(&site.name).len();
        if k == 0 || numel == 0 || !numel.is_multiple_of(k) {
            return Err(format!("{}: {numel} stored elements are not whole rows of width {k}", site.name));
        }
        out.push(Site { name: site.name.clone(), tensors: vec![Tensor { name: site.name, n: numel / k, k }], layer: site.layer, roles: vec![site.leaf] });
    }
    for group in m.coupled_tensors() {
        let Some((first, rest)) = group.split_first() else { continue };
        let position = |out: &[Site], name: &str| out.iter().position(|s| s.name == name).ok_or_else(|| format!("coupled tensor {name} is not a linear site of the model, or is in two groups"));
        position(&out, first)?;
        for name in rest {
            if name == first {
                return Err(format!("coupled group lists {name} twice"));
            }
            let joined = out.remove(position(&out, name)?);
            let at = position(&out, first)?;
            out[at].tensors.extend(joined.tensors);
            out[at].roles.extend(joined.roles);
        }
    }
    Ok(out)
}
