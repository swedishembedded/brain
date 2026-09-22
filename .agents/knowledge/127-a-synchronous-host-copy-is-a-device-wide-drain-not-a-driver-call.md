<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 127. A synchronous host copy is a device-wide drain, not a driver call

Moving a parameter upload from record time into submit time looks like a
scheduling detail and is a serialisation. A synchronous host-to-device copy
runs on the legacy default stream, and the legacy stream is ordered against
every blocking stream in the context - so one per dispatch does not cost a
driver call each, it drains the device between every pair of dispatches.
Measured: eight back-to-back dispatches of one kernel took an order of
magnitude longer that way than with the uploads already done.

The fix is to enqueue the copy on the same stream the dispatches use, which
obliges the source to be page-locked and to stay untouched until the copy
runs - so a pool with an explicit "returned once the device has drained"
rule, not a `Vec` handed to an async copy.

The general shape: on an API with a legacy-default-stream rule, "synchronous"
does not mean "blocks this one operation". It means a global ordering point,
and a per-dispatch ordering point is a per-dispatch stall.
