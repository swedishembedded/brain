<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 484. Every installed model of a brain-format family was served under the one id its family writes on the card

**Lesson:** A served model's name is where the store keeps it, not a field a writer copies into every file. Two models
must never be able to collide by construction.

## What happened

`serving::model_dir::discover` registers a brain checkpoint under its card's id. Ouro writes the card id `brain/ouro`
for every checkpoint it saves, so the second Ouro the training service installed (`install_as`, `local/<name>`) was
skipped as a duplicate id, and the first answered to `brain/ouro` rather than to the name it was installed as.

## What changed

A model under the reserved `local` vendor (what `install_as` writes) is served under its store reference,
`local/<name>`; `serving::model_dir::tests::an_installed_model_is_served_under_its_installed_name` holds two installed
checkpoints with one card id to their two names.
