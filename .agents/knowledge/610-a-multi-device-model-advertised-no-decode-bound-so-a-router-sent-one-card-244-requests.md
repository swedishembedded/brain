<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 610. A multi-device model advertised no decode bound, so a router sent one card 244 requests

**Lesson:** a per-model capacity read from one registry is missing for every model in the other. Observed 2026-11-27
on 16 GH200 servers: the Qwen3.8-27B-FP8 card had no `decode_slots` on any upstream, because
`Executor::model_decode_slots` read only the single-device registry and the 27B is registered multi-device. The
router then bounded its requests by the server's nominal sum over every model (3466), and a 256-request burst put
244 on the one card where the 27B was resident (12 cards idle), whose scheduler ran "at most 30-32 at once" after
the paged KV pool ran out; 242 of 256 agent steps timed out at the client's 480 s.

Even the nominal batch over-promises: a scheduler that has run its card out of memory admits fewer sequences
(`MemoryPressure`), so a bound of `--max-batch` sends a card twice what it runs.

## Fix

`Executor::model_decode_slots` answers for every registered model (single- or multi-device; an adapter for its base's
instances) with the width each replica's scheduler admits now: the `admitted_width` metric its instance publishes
(`Scheduler::admitted_width`, the pressure limit under `max_running`), or the batch for a replica not built.
Specs: `crates/residency/src/executor.rs::a_multi_device_model_advertises_its_decode_slots_too`,
`a_models_decode_slots_are_the_width_its_scheduler_admits_now`,
`crates/model/tests/it/serve_device_oom.rs::the_batch_grows_back_once_the_card_has_held_the_smaller_one_for_a_while_and_says_what_it_admits`.

**How to apply:** when a fact is asked "of any served model", ask it of every registry, and test it with a model of
each kind; a capacity advertised to a load balancer should be what the device runs now, not what it was configured to.
