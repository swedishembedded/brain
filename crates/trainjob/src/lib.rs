// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services that run fine-tuning inside
// a serving process, scheduled with the models it serves, for its clients. If
// your team needs expertise in moving training off the application's process
// and onto shared accelerators, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Training as a served job.
//!
//! `brain/train` is a model of the serving process whose one action, `train`,
//! fine-tunes a chat model's LoRA adapter: supervised on `generic-messages-v2`
//! records, or by the clipped policy gradient on rollout records that carry an
//! advantage or a teacher context (self-distillation). It is reached like any
//! action: over D-Bus `Subscribe` (progress frames, `Cancel`) and over HTTP
//! `POST /v1/jobs` with `GET /v1/jobs/{id}/events`, behind the surface's key.
//!
//! * [`request`]: the action's contract and the typed, bounded request.
//! * [`store`]: the content-addressed artifact store datasets and adapters
//!   move through, by blake3 digest; a caller never names a path.
//! * [`job`]: one run, from staged inputs to the adapter and its report.
//! * [`resident`]: the run as a workload the residency executor places,
//!   evicting idle serving instances for it and refusing when serving is busy.

pub mod job;
pub mod request;
pub mod resident;
pub mod store;

pub use request::{ACTION, MODEL};
pub use resident::{resident, TrainConfig, TrainResident};
pub use store::{ArtifactStore, Digest, StoreError};
