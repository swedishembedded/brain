<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 622. A job that holds a served adapter blocks every other job of the training service

**Lesson:** the first design served a stored adapter for as long as a `serve_adapter` job ran (cancel to withdraw),
one job per served id. The training service declares no batch, so the executor runs one of its jobs per replica (one
per device): the second job, the candidate beside the anchor, waited behind the first until the HTTP call timed out
(606 s). Lending the device back while the job waited (`DeviceLease::lent`) did not change that: the job still held
the service's one slot. A served adapter rides on its base's bank and needs no slot of its own.

**Fix:** `serve_adapter` offers the stored file under the caller's id and returns at once; it stays served until
`withdraw_adapter`, and the bank's capacity bounds how many are served. Two adapters are served side by side in 10 s
(spec: `trainjob` `it::stream::a_stored_version_is_served_on_another_server_under_a_name_the_caller_chooses`). The
stored file does not carry its policy version, so the caller names it (`version`).
