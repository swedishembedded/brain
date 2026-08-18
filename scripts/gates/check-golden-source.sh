#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Every golden dumper must record WHICH checkpoint it dumped from.
#
# A golden dump is tensors plus a claim - "this is what the reference produced"
# - and the claim only means something together with the checkpoint that
# produced it. Pair a dump with a different tier of the same architecture and
# the suite either dies deep in the importer with a tensor-shape error, or
# compares against the wrong reference and certifies a meaningless number, or
# prints to stderr and returns, which cargo reports as a PASS. All three have
# happened here.
#
# tools/goldens/golden_source.py writes the `source` block that closes it, and
# brain_testutil::golden::Source enforces it on the reading side. This gate is
# the third leg: a NEW dumper that writes a manifest without going through the
# shared helper is refused, so the convention cannot quietly stop being one.
#
# Dumpers that predate the convention are grandfathered by name below, exactly
# like the clippy ratchet: the list may only ever shrink, and shrinking it is a
# re-run of that dumper against its checkpoint. Nothing may be added to it.
#
# Usage: scripts/gates/check-golden-source.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

# Dumpers whose manifests predate `source_block`. MAY ONLY SHRINK.
GRANDFATHERED=$(
  cat <<'EOF'
arcface_dump_reference.py
asr_dump_reference.py
chronos2_dump_kf_reference.py
chronos2_dump_mv_reference.py
chronos2_dump_reference.py
clip_dump_reference.py
codeformer_dump_reference.py
codeformer_restore_dump_reference.py
controlnet_dump_reference.py
deepseek_ocr_convert_llamacpp_dump.py
deepseek_ocr_dump_reference.py
fastvlm_caption_dump_reference.py
fastvlm_decoder_dump_reference.py
fastvlm_vision_dump_reference.py
fincast_dump_reference.py
flux1_dump_reference.py
flux2_dump_reference.py
instantid_dump_reference.py
lfm2_dump_reference.py
moondream3_decoder_dump_reference.py
pulid_dump_reference.py
qwen3omnimoe_dump_generate.py
qwen3omnimoe_dump_reference.py
qwen3vl_decoder_dump_reference.py
qwen_encoder_dump_reference.py
rrdbnet_dump_reference.py
s3dit_block_dump_reference.py
s3dit_model_dump_reference.py
s3dit_real_512_dump_reference.py
s3dit_real_dump_reference.py
sam2_dump_reference.py
scrfd_dump_reference.py
sdxl_dump_vae_decode.py
sdxlunet_dump_reference.py
splat_dump_gradcheck.py
t5encoder_dump_reference.py
vae_dump_reference.py
vit_dump_gradcheck.py
wan_dit_dump_reference.py
wan_schedule_dump_reference.py
wan_t5_dump_reference.py
wan_vae_dump_reference.py
worldmirror2_check_onnx.py
worldmirror2_dump_dpt_tiny.py
worldmirror2_dump_reference.py
EOF
)

fail=0
migrated=0
for f in tools/goldens/*.py; do
  base=$(basename "$f")
  [ "$base" = "golden_source.py" ] && continue
  [ "$base" = "onnx_eval.py" ] && continue
  # Only a script that WRITES a manifest is in scope; a converter or a checker
  # that only reads one has no provenance to record.
  grep -qE '(manifest|meta)[^ ]*\.json' "$f" || continue
  if grep -q 'source_block' "$f"; then
    migrated=$((migrated + 1))
    continue
  fi
  if grep -qxF "$base" <<<"$GRANDFATHERED"; then
    continue
  fi
  echo "check-golden-source: $f writes a golden manifest without a \`source\` block."
  fail=1
done

if [ "$fail" -ne 0 ]; then
  cat <<'EOF'

Record which checkpoint the dump came from:

    from golden_source import source_block
    manifest["source"] = source_block(
        checkpoint="<vendor>/<repo>",
        files=[<the weight files actually read>],
        identity={<the config fields that fix the dumped tensors' shapes>},
    )

`identity` is the enforced half - width, depth, head count, vocab: whatever two
tiers of this architecture cannot both have. brain_testutil::golden::Source
compares it against the checkpoint the test is about to run, and a mismatch
becomes a named skip (a hard failure under BRAIN_REQUIRE_FIXTURES=1) instead of
a tensor-shape error deep in the importer, or a wrong number, or a silent pass.
EOF
  exit 1
fi

grandfathered_count=$(grep -c . <<<"$GRANDFATHERED")
echo "check-golden-source: OK ($migrated on the convention, $grandfathered_count grandfathered)"
exit 0
