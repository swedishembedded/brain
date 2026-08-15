# npu - roadmap

## Real bug: `npu_model_parity` test hangs under a full `make test` run, not seen when tests run to completion in isolation

`crates/cli/tests/npu_model_parity.rs` hung during a full `make test` run on
this box (real OpenVINO NPU compiler libraries ARE present -
`libopenvino_intel_npu_compiler.so` under `/usr/lib/x86_64-linux-gnu/` - so
its `_when_openvino_available` tests run for real rather than skipping): 2 of
7 tests passed (`resident_forecast::tests::f32_codec_roundtrips_with_shape`,
`resident_forecast::tests::forecast_manifests_are_well_formed`), then it hung
for 26+ minutes on one of the five remaining NPU/OpenVINO parity tests
(`chronos2_npu_model_builds_and_parity_ref_matches_core_forward`,
`chronos2_npu_graph_output_matches_parity_ref_when_openvino_available`,
`fincast_npu_model_builds_and_parity_ref_matches_core_forward_amask`,
`fincast_npu_graph_output_matches_parity_ref_when_openvino_available`, or
`forecast_residents_advertise_npu_and_never_panic_on_npu_activation` - stdout
was buffered per-test so the exact one that hung was not identified before
it was killed). Same signature as the `kernel_timing` hang documented in
`.agents/roadmap/backend-vulkan.md`: exactly one thread pinned at 100% CPU
(`/proc/<pid>/task/*/stat` showed one `R`, the rest `S`) while the other
tests' threads sat idle - consistent with a busy-poll wait on a completion
that never signals, not a slow-but-progressing computation. Killed with
SIGKILL to unblock the gate rather than let it run to the 2400s suite
timeout.

**Not chased further** in this pass (time-boxed in favor of finishing the
README verification gate) - worth checking whether this is the SAME root
cause as the `kernel_timing` hang (some shared device-serialization
assumption broken when many GPU/NPU-touching test binaries run concurrently
under `make test`'s default `--test-threads=8` across a full-workspace run)
or a distinct NPU/OpenVINO-specific issue. Re-running just this one test
binary in isolation, and re-running the FULL suite a second time back-to-back
(the way `kernel_timing`'s hang did NOT reproduce on a clean immediate
re-run), would be the fastest way to tell "isolated-binary bug" from
"cross-process contention" apart here too.

**Practical impact**: same as `backend-vulkan.md`'s entry - `make test` is
not currently reliable end-to-end on this box on a single attempt; a second,
clean run (no other GPU/NPU work competing) is sometimes needed to get a
green result.
