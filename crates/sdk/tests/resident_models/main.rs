// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for validating speech and language
// models on the hardware they deploy to for its clients. If your team needs
// expertise in on-device model integration then you can procure our services
// by sending an email to info@swedishembedded.com.

//! The SDK's specs that load a real checkpoint from the model store onto the
//! process's own device (Qwen3-TTS, Qwen3 with a speech projector), linked as a
//! binary of their own. The chat specs in `tests/it` select the CPU through
//! `Device`, which the SDK applies to the whole process as the CLI does; beside
//! them, a checkpoint whose bf16 kernels need a GPU was built on the CPU backend
//! instead. Here nothing selects a device, so every spec runs where the process
//! does. `brain_testutil::env_lock` serialises them: each holds a model of
//! several GB.

mod speech_ingress;
mod tts_policy;
mod tts_profile;
mod tts_resident;
