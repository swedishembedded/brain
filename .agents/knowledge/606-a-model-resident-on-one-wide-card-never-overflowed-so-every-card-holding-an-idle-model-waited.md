<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 606. A model resident on one wide card never overflowed, so every card holding an idle model waited

**Lesson:** "a model goes where it is loaded; a card where it is not takes it only when no resident card has a free
slot" keeps two models from swapping cards, but on a wide model it is a one-card ceiling. Observed 2026-11-27 on a
16-card fleet, every card serving every adapter: a LoRA evaluation of Qwen3-4B with 264 concurrent trials ran on the
one card where the 4B was resident (72% busy) while nine cards holding the idle 27B had nothing in flight. The 4B's
card advertised hundreds of decode slots, so "no free slot" never happened.

**How to apply:** an overflow rule keyed on "full" needs a load bound that scales with the card's width (as #543 for
prefix affinity). The router (`apiserve::route`) now spills a model onto a non-resident card when every resident card
carries at least `FleetConfig::spill_load` requests for it (default 8; first an eighth of its slots, replaced by
an absolute count once advertised slots proved nominal, #607) and that card
is completely idle: nothing of the router's in flight and its last health showing no job running or waiting. A card
busy with another model is never a spill target, so a card changes model only while nothing runs on it, which keeps
it thrash-free. Spec: `crates/apiserve/tests/it/route.rs::a_loaded_model_spills_onto_an_idle_upstream_but_never_onto_a_busy_one`.
