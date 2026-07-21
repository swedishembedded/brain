// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The forecasting **input**: a `Panel` of named, role-tagged series.
//!
//! One shape spans every case the models in this project consume:
//! - univariate — one item, one `target` variate;
//! - multivariate / OHLCV — one item, several variates (OHLCV is just six named
//!   variates, not a special case);
//! - covariate-informed — extra variates tagged `PastCovariate` /
//!   `KnownFuture` / `Static`;
//! - cross-sectional — many items in one panel (a universe of instruments).
//!
//! The `role` taxonomy is exactly Chronos-2's masking scheme, and it degrades
//! cleanly: a model that ignores covariates simply drops the non-`Target`
//! variates (and advertises `supports_covariates = false` in its
//! [`Capabilities`](crate::Capabilities)).

/// What a variate contributes to a forecast, and how much of it is observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// A series to be forecast. History observed, future predicted.
    Target,
    /// Observed up to the origin only; informs the forecast but is never scored
    /// and has no known future (e.g. realised volume, order-flow imbalance).
    PastCovariate,
    /// Observed for both history *and* future (e.g. day-of-week, holiday flags,
    /// scheduled earnings/expiry dates). Supplied via [`Variate::future`].
    KnownFuture,
    /// A single value per item, constant over time (e.g. sector id, exchange).
    Static,
}

impl Role {
    /// Wire tag.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Target => "target",
            Role::PastCovariate => "past_covariate",
            Role::KnownFuture => "known_future",
            Role::Static => "static",
        }
    }

    /// Parse a wire tag.
    pub fn parse(s: &str) -> Option<Role> {
        Some(match s {
            "target" => Role::Target,
            "past_covariate" | "past" => Role::PastCovariate,
            "known_future" | "future" => Role::KnownFuture,
            "static" => Role::Static,
            _ => return None,
        })
    }
}

/// Whether a variate's values are real-valued or category ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Real-valued (prices, returns, volume, sentiment).
    Continuous,
    /// Integer category ids in `0..cardinality` (holiday type, regime label).
    Categorical,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Continuous => "continuous",
            Kind::Categorical => "categorical",
        }
    }
    pub fn parse(s: &str) -> Option<Kind> {
        Some(match s {
            "continuous" => Kind::Continuous,
            "categorical" => Kind::Categorical,
            _ => return None,
        })
    }
}

/// One named series within an [`Item`].
#[derive(Clone, Debug, PartialEq)]
pub struct Variate {
    /// Column name (`"close"`, `"volume"`, `"is_earnings"`). Unique within an
    /// item.
    pub name: String,
    /// What this series contributes.
    pub role: Role,
    /// Value type.
    pub kind: Kind,
    /// Context values, one per historical timestep (length = context length).
    pub data: Vec<f32>,
    /// For [`Role::KnownFuture`]: the values over the forecast horizon. `None`
    /// for every other role. Length must equal the request horizon.
    pub future: Option<Vec<f32>>,
    /// Optional observed-mask (`1.0` observed, `0.0` missing), parallel to
    /// `data`. `None` means all observed.
    pub observed: Option<Vec<f32>>,
    /// For [`Kind::Categorical`]: number of distinct categories.
    pub cardinality: Option<u32>,
}

impl Variate {
    /// A plain continuous target series.
    pub fn target(name: impl Into<String>, data: Vec<f32>) -> Variate {
        Variate {
            name: name.into(),
            role: Role::Target,
            kind: Kind::Continuous,
            data,
            future: None,
            observed: None,
            cardinality: None,
        }
    }

    /// Number of observed (context) timesteps.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// One entity (instrument) and all its variates. All target variates in an item
/// share the same time index; cross-sectional panels hold many items.
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    /// Identifier echoed back on the forecast (a ticker, an instrument id).
    pub item_id: String,
    /// The series for this item.
    pub variates: Vec<Variate>,
}

impl Item {
    pub fn new(item_id: impl Into<String>, variates: Vec<Variate>) -> Item {
        Item { item_id: item_id.into(), variates }
    }

    /// The target variates, in declaration order.
    pub fn targets(&self) -> impl Iterator<Item = &Variate> {
        self.variates.iter().filter(|v| v.role == Role::Target)
    }

    /// Look up a variate by name.
    pub fn variate(&self, name: &str) -> Option<&Variate> {
        self.variates.iter().find(|v| v.name == name)
    }

    /// Context length for this item — the length of its first target series.
    /// Zero if the item has no targets.
    pub fn context_len(&self) -> usize {
        self.targets().next().map(|v| v.len()).unwrap_or(0)
    }
}

/// The full forecasting input: a set of items sharing a sampling frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct Panel {
    /// Sampling frequency (`"1d"`, `"1min"`, or an ISO-8601 duration). Free-form;
    /// used for calendar covariates and echoed on the result.
    pub freq: String,
    /// RFC-3339 timestamp of the first context step, enabling calendar features.
    /// Optional — models that do not use wall-clock time ignore it.
    pub start: Option<String>,
    /// The entities to forecast.
    pub items: Vec<Item>,
}

impl Panel {
    /// A single-item panel from a set of variates.
    pub fn single(
        freq: impl Into<String>,
        item_id: impl Into<String>,
        variates: Vec<Variate>,
    ) -> Panel {
        Panel { freq: freq.into(), start: None, items: vec![Item::new(item_id, variates)] }
    }

    /// Maximum context length across all items — the longest window any model
    /// must accommodate. Used to reject over-length requests early.
    pub fn max_context_len(&self) -> usize {
        self.items.iter().map(|it| it.context_len()).max().unwrap_or(0)
    }

    /// True if any item carries a covariate variate (non-target).
    pub fn has_covariates(&self) -> bool {
        self.items
            .iter()
            .any(|it| it.variates.iter().any(|v| v.role != Role::Target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_and_kind_wire_tags_roundtrip() {
        for r in [Role::Target, Role::PastCovariate, Role::KnownFuture, Role::Static] {
            assert_eq!(Role::parse(r.as_str()), Some(r));
        }
        for k in [Kind::Continuous, Kind::Categorical] {
            assert_eq!(Kind::parse(k.as_str()), Some(k));
        }
        assert_eq!(Role::parse("nonsense"), None);
    }

    #[test]
    fn ohlcv_is_just_named_variates_not_a_special_case() {
        let p = Panel::single(
            "1d",
            "AAPL",
            vec![
                Variate::target("open", vec![1.0, 2.0, 3.0]),
                Variate::target("high", vec![1.5, 2.5, 3.5]),
                Variate::target("low", vec![0.5, 1.5, 2.5]),
                Variate::target("close", vec![1.2, 2.2, 3.2]),
                Variate {
                    name: "volume".into(),
                    role: Role::PastCovariate,
                    kind: Kind::Continuous,
                    data: vec![100.0, 110.0, 120.0],
                    future: None,
                    observed: None,
                    cardinality: None,
                },
            ],
        );
        assert_eq!(p.items[0].targets().count(), 4);
        assert!(p.has_covariates());
        assert_eq!(p.max_context_len(), 3);
    }

    #[test]
    fn known_future_carries_a_future_path() {
        let v = Variate {
            name: "is_earnings".into(),
            role: Role::KnownFuture,
            kind: Kind::Categorical,
            data: vec![0.0, 0.0, 1.0],
            future: Some(vec![0.0, 0.0]),
            observed: None,
            cardinality: Some(2),
        };
        assert_eq!(v.future.as_ref().unwrap().len(), 2);
        assert_eq!(v.cardinality, Some(2));
    }

    #[test]
    fn context_len_reads_the_first_target() {
        let it = Item::new(
            "X",
            vec![
                Variate {
                    name: "vix".into(),
                    role: Role::PastCovariate,
                    kind: Kind::Continuous,
                    data: vec![1.0; 10],
                    future: None,
                    observed: None,
                    cardinality: None,
                },
                Variate::target("close", vec![1.0; 7]),
            ],
        );
        // context_len is the first *target*, not the first variate.
        assert_eq!(it.context_len(), 7);
    }
}
