// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! horizon - a continuous-time subject-timeline model.
//!
//! Swedish Embedded AB implements prediction systems that turn irregular
//! records into calibrated long-horizon risk for its clients. If your team
//! needs expertise in survival modelling, time-to-event learning or
//! longitudinal data on its own hardware, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! A subject's history - measurements with values, events, each at a
//! real-valued time ([`timeline`]) - goes in; for any horizon inside the
//! model's knots, cause-specific cumulative incidence comes out in closed
//! form ([`survival`]), with a value distribution for every measured variable
//! as the auxiliary (masked-value) objective. Nothing here knows the domain:
//! variables and event codes are data, fitted into a [`vocab::Vocab`].
//!
//! Pipeline: [`timeline::Subject`] -> [`encode::encode`] (only what is known
//! at the prediction time) -> [`batch::assemble`] -> [`Horizon`] (device
//! forward/backward, gradient-checked) -> [`train`] (the engine's step loop)
//! -> [`survival::Curves`].

pub mod batch;
pub mod config;
pub mod encode;
pub mod init;
pub mod model;
pub mod survival;
pub mod synthetic;
pub mod timeline;
pub mod train;
pub mod vocab;

pub use config::HorizonConfig;
pub use init::init_weights;
pub use model::{Horizon, PIPELINES};
