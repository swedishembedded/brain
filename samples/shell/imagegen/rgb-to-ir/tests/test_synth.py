# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements synthetic thermal-image generation, ingestion
# and label-preservation gating for its clients. If your team needs expertise
# in turning translator outputs into leakage-safe detector training arms, you
# can procure our services by sending an email to info@swedishembedded.com.

"""Spec tests for rir_synth against a FAKE `brain` executable: the generation
driver (resume, bounded retry on memory pressure, abort on any other failure,
the crop that maps outputs back to source pixels, prior instructions, the
adapter identity), and the ingest and gate stages over its outputs."""
import contextlib
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402,F401  (sets the OpenCV log level before cv2 loads)
import cv2  # noqa: E402
import rir_captions as C  # noqa: E402
import rir_synth as S  # noqa: E402

W, H = 70, 50  # not multiples of 16: the crop is 64 x 48 at (3, 1)
CROP = S.Crop(3, 1, 64, 48, W, H)
BOXES = [{"class": "person", "x1": 10.0, "y1": 8.0, "x2": 30.0, "y2": 40.0},
         {"class": "car", "x1": 40.0, "y1": 10.0, "x2": 69.0, "y2": 45.0}]
ADAPTER_BYTES = b"adapter-weights"

# Records how it was called, then consumes the next behaviour of the plan file
# ("ok" writes the reference as the output; "oom" and "fail" write to stderr and fail).
FAKE_BRAIN = """#!/usr/bin/env python3
import json, shutil, sys
argv = sys.argv[1:]
with open(%(log)r, "a") as fh:
    fh.write(json.dumps(argv) + "\\n")
with open(%(plan)r) as fh:
    plan = json.load(fh)
step = plan.pop(0) if plan else "ok"
with open(%(plan)r, "w") as fh:
    json.dump(plan, fh)
if step == "oom":
    print("flux2: no GPU placement fits the model", file=sys.stderr)
    sys.exit(1)
if step == "fail":
    print("flux2: unsupported adapter layout", file=sys.stderr)
    sys.exit(2)
if step == "empty":
    sys.exit(0)
shutil.copy(argv[argv.index("--ref") + 1], argv[argv.index("--out") + 1])
"""


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def write_png(path, img):
    assert cv2.imwrite(path, img)


class World(unittest.TestCase):
    """A split of three frames, a fake brain, a config and a caption report in a temp dir."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        rng = np.random.default_rng(0)
        self.rgb, self.ir, frames = {}, {}, []
        for k in range(3):
            self.rgb[k] = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
            self.ir[k] = rng.integers(0, 256, (H, W), dtype=np.uint8)
            write_png(self.path(f"rgb{k}.png"), self.rgb[k])
            write_png(self.path(f"ir{k}.png"), self.ir[k])
            frames.append({"id": f"f{k}", "dataset": "d", "rgb": self.path(f"rgb{k}.png"), "ir": self.path(f"ir{k}.png"),
                           "boxes": BOXES if k < 2 else [], "sequence_id": f"s{k}", "split": "S", "usable": True,
                           "day_night": "day", "width": W, "height": H})
        self.doc = {"frames": frames}
        self.splits = self.path("splits.json")
        self.write_json("splits.json", self.doc)
        self.brain = self.path("brain")
        self.log, self.plan = self.path("brain.log"), self.path("plan.json")
        with open(self.brain, "w") as fh:
            fh.write(FAKE_BRAIN % {"log": self.log, "plan": self.plan})
        os.chmod(self.brain, os.stat(self.brain).st_mode | stat.S_IXUSR)
        self.set_plan([])
        with open(self.path("adapter.brain"), "wb") as fh:
            fh.write(ADAPTER_BYTES)
        self.write_json("synth.json", {"device": "gpu1", "backend": "cuda", "variant": "klein-4b", "dit": "m/dit.gguf",
                                      "vae": "m/vae.safetensors", "text_encoder": "m/te", "tokenizer": "m/tok.json",
                                      "adapter": "adapter.brain", "strength": 1.0, "seed": 3})
        self.report = {"classes": {
            "person": {"warmer": 442, "cooler": 52, "same": 462, "n": 956},
            "car": {"warmer": 304, "cooler": 45, "same": 827, "n": 1176},
            "bicycle": {"warmer": 4, "cooler": 203, "same": 145, "n": 352}}}
        self.write_json("captions-report.json", self.report)
        self.out = self.path("gen")

    def path(self, name):
        return os.path.join(self.dir, name)

    def write_json(self, name, doc):
        with open(self.path(name), "w") as fh:
            json.dump(doc, fh)

    def set_plan(self, plan):
        self.write_json("plan.json", plan)

    def calls(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log) as fh:
            return [json.loads(line) for line in fh]

    def config(self):
        return S.load_config(self.path("synth.json"))

    def generate(self, modes=("neutral",), config=None, brain=None, retry=None, sleep=lambda s: None, **selection):
        job = S.Job(self.doc, S.Selection(**selection), modes, ("person", "car", "bicycle"),
                    S.prior_polarities(self.report), self.out)
        runner = S.BrainRunner(brain or self.brain, config or self.config(), retry or S.Retry(4, 0.0, 60.0), sleep)
        return S.generate(job, runner)

    def records(self, mode):
        d = os.path.join(self.out, mode)
        return {f: read_json(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".json")}


def opt(argv, name):
    return argv[argv.index(name) + 1]


class Config(World):
    def test_paths_are_relative_to_the_config_file_and_the_adapter_is_optional(self):
        c = self.config()
        self.assertEqual(c.adapter, self.path("adapter.brain"))
        self.assertEqual(c.dit, self.path("m/dit.gguf"))
        self.write_json("zero.json", {k: v for k, v in read_json(self.path("synth.json")).items() if k != "adapter"})
        self.assertIsNone(S.load_config(self.path("zero.json")).adapter)

    def test_unknown_and_missing_keys_are_refused(self):
        self.write_json("bad.json", {**read_json(self.path("synth.json")), "stregnth": 0.5})
        with self.assertRaisesRegex(ValueError, "stregnth"):
            S.load_config(self.path("bad.json"))
        self.write_json("short.json", {"device": "gpu1"})
        with self.assertRaisesRegex(ValueError, "missing"):
            S.load_config(self.path("short.json"))


class CropRecord(World):
    def test_the_crop_is_a_centred_multiple_of_sixteen_and_a_smaller_frame_is_refused(self):
        self.assertEqual(S.crop_for(W, H), CROP)
        self.assertEqual(S.crop_for(640, 512), S.Crop(0, 0, 640, 512, 640, 512))
        with self.assertRaisesRegex(ValueError, "16"):
            S.crop_for(70, 15)

    def test_the_record_maps_the_output_back_to_the_source_pixels_exactly(self):
        self.generate()
        rec = self.records("neutral")["d_f0.json"]
        crop = S.Crop.from_dict(rec["crop"])
        self.assertEqual(crop, CROP)
        out = cv2.imread(os.path.join(self.out, "neutral", rec["output"]))
        self.assertEqual(out.shape, (48, 64, 3))
        # the fake brain returns its reference, which is the cropped source
        np.testing.assert_array_equal(out, self.rgb[0][1:49, 3:67])

    def test_boxes_follow_the_crop_and_a_box_mostly_outside_is_dropped(self):
        boxes = [(0, 10.0, 8.0, 30.0, 40.0), (1, 40.0, 10.0, 69.0, 45.0), (2, 0.0, 0.0, 6.0, 10.0)]
        mapped = S.crop_boxes(boxes, CROP)
        self.assertEqual(mapped[0], (0, 7.0, 7.0, 27.0, 39.0))
        self.assertEqual(mapped[1], (1, 37.0, 9.0, 64.0, 44.0))  # clipped at the right edge
        self.assertEqual(len(mapped), 2)  # the third has 45 percent of its area inside the crop


class Instructions(World):
    def test_prior_polarities_keep_the_commonest_direction_the_data_can_teach(self):
        priors = S.prior_polarities(self.report)
        # person: warmer (442) over cooler (52); car: warmer, as cooler is under a tenth of its statements anyway;
        # bicycle: cooler, and its warmer share (1.1 percent) is below the controllable floor
        self.assertEqual(priors, {"person": "warmer", "car": "warmer", "bicycle": "cooler"})

    def test_a_class_with_no_controllable_direction_has_no_prior(self):
        rare = {"classes": {"truck": {"warmer": 2, "cooler": 3, "same": 95, "n": 100}}}
        self.assertEqual(S.prior_polarities(rare), {})

    def test_a_prior_instruction_states_the_priors_of_the_classes_in_the_frame_with_training_wording_only(self):
        priors = S.prior_polarities(self.report)
        text = S.prior_instruction(["person", "car"], priors, np.random.default_rng(1))
        self.assertTrue(text.startswith(C.NEUTRAL_CAPTION + " "))
        self.assertTrue(C.has_polarity(text, "person", "warmer") and C.has_polarity(text, "car", "warmer"))
        self.assertFalse(C.has_polarity(text, "person", "cooler"))
        for held_out in C.HELD_OUT.values():
            self.assertNotIn(held_out, text)

    def test_a_frame_without_a_class_that_has_a_prior_gets_the_neutral_caption(self):
        self.assertEqual(S.prior_instruction([], S.prior_polarities(self.report), np.random.default_rng(1)), C.NEUTRAL_CAPTION)
        self.assertEqual(S.prior_instruction(["truck"], S.prior_polarities(self.report), np.random.default_rng(1)),
                         C.NEUTRAL_CAPTION)

    def test_no_generated_prior_instruction_uses_a_held_out_template(self):
        self.generate(modes=("neutral", "prior"))
        texts = [r["instruction"] for r in self.records("prior").values()]
        self.assertEqual(texts[2], C.NEUTRAL_CAPTION)  # f2 has no boxes
        self.assertTrue(all(C.NEUTRAL_CAPTION in t for t in texts))
        for t in texts:
            for held_out in C.HELD_OUT.values():
                self.assertNotIn(held_out, t)
        self.assertTrue(all(r["instruction"] == C.NEUTRAL_CAPTION for r in self.records("neutral").values()))

    def test_the_prior_wording_is_the_same_on_a_rerun(self):
        self.generate(modes=("prior",))
        first = {k: v["instruction"] for k, v in self.records("prior").items()}
        self.out = self.path("gen2")
        self.generate(modes=("prior",))
        self.assertEqual(first, {k: v["instruction"] for k, v in self.records("prior").items()})


class Generation(World):
    def test_brain_is_called_with_the_components_the_adapter_the_crop_size_and_the_cropped_reference(self):
        self.generate()
        argv = self.calls()[0]
        self.assertEqual(argv[:5], ["--device", "gpu1", "--backend", "cuda", "flux2"])
        self.assertEqual(argv[5], "generate")
        self.assertEqual(opt(argv, "--variant"), "klein-4b")
        self.assertEqual(opt(argv, "--dit"), self.path("m/dit.gguf"))
        self.assertEqual(opt(argv, "--adapter"), self.path("adapter.brain"))
        self.assertEqual((opt(argv, "--strength"), opt(argv, "--seed")), ("1.0", "3"))
        self.assertEqual((opt(argv, "--width"), opt(argv, "--height")), ("64", "48"))
        self.assertEqual(opt(argv, "--prompt"), C.NEUTRAL_CAPTION)
        self.assertTrue(opt(argv, "--ref").endswith(".ppm"))
        self.assertEqual(len(self.calls()), 3)

    def test_the_record_names_the_frame_mode_instruction_seed_command_and_duration(self):
        self.generate()
        rec = self.records("neutral")["d_f1.json"]
        self.assertEqual((rec["dataset"], rec["id"], rec["mode"], rec["seed"]), ("d", "f1", "neutral", 3))
        self.assertEqual(rec["instruction"], C.NEUTRAL_CAPTION)
        self.assertEqual(rec["command"][0], self.brain)
        self.assertIn("--prompt", rec["command"])
        self.assertGreaterEqual(rec["duration_s"], 0.0)
        self.assertEqual(rec["attempts"], 1)

    def test_the_adapter_identity_is_the_sha256_of_its_bytes(self):
        self.generate()
        self.assertEqual(self.records("neutral")["d_f0.json"]["adapter_sha256"], hashlib.sha256(ADAPTER_BYTES).hexdigest())

    def test_a_zero_shot_config_without_an_adapter_passes_none_and_records_none(self):
        self.write_json("zero.json", {k: v for k, v in read_json(self.path("synth.json")).items() if k != "adapter"})
        self.generate(config=S.load_config(self.path("zero.json")))
        self.assertNotIn("--adapter", self.calls()[0])
        self.assertIsNone(self.records("neutral")["d_f0.json"]["adapter_sha256"])

    def test_a_rerun_skips_finished_frames_and_redoes_a_damaged_output(self):
        first = self.generate()
        self.assertEqual((first["generated"], first["skipped"]), (3, 0))
        with open(os.path.join(self.out, "neutral", "d_f1.ppm"), "ab") as fh:
            fh.write(b"\0")  # trailing bytes: not the size the header promises
        os.remove(os.path.join(self.out, "neutral", "d_f2.ppm"))
        again = self.generate()
        self.assertEqual((again["generated"], again["skipped"]), (2, 1))
        self.assertEqual(len(self.calls()), 5)

    def test_no_source_image_is_left_behind(self):
        self.generate()
        self.assertEqual(sorted(f for f in os.listdir(os.path.join(self.out, "neutral")) if not f.startswith("d_f")), [])
        self.assertEqual(len(os.listdir(os.path.join(self.out, "neutral"))), 6)

    def test_every_frame_and_event_is_appended_to_the_progress_log(self):
        self.generate()
        self.generate()
        with open(os.path.join(self.out, "progress.log")) as fh:
            events = [json.loads(line) for line in fh]
        self.assertEqual([e["event"] for e in events], ["generated"] * 3 + ["skipped"] * 3)
        self.assertEqual(events[0]["id"], "f0")

    def test_a_rerun_with_another_adapter_or_seed_in_the_same_directory_is_refused(self):
        self.generate()
        with open(self.path("adapter.brain"), "ab") as fh:
            fh.write(b"more")
        with self.assertRaisesRegex(S.ConfigMismatch, "adapter_sha256"):
            self.generate()
        self.assertEqual(len(self.calls()), 3)

    def test_shards_partition_the_selected_frames_and_a_limit_selects_before_sharding(self):
        self.generate(shard=(0, 2))
        self.generate(shard=(1, 2))
        ids = [r["id"] for r in self.records("neutral").values()]
        self.assertEqual(sorted(ids), ["f0", "f1", "f2"])
        self.assertEqual(len(self.calls()), 3)
        self.out = self.path("gen-limited")
        done = self.generate(limit=2)
        self.assertEqual(done["generated"], 2)

    def test_a_bad_shard_is_refused(self):
        for bad in ("2/2", "-1/2", "a/b", "1"):
            with self.assertRaises(ValueError, msg=bad):
                S.parse_shard(bad)
        self.assertEqual(S.parse_shard("1/4"), (1, 4))

    def test_an_unknown_mode_is_refused_before_anything_runs(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            self.generate(modes=("bogus",))
        self.assertEqual(self.calls(), [])


class Failures(World):
    def test_memory_pressure_is_retried_after_a_sleep_and_then_succeeds(self):
        self.set_plan(["oom", "oom", "ok"])
        sleeps = []
        done = self.generate(retry=S.Retry(4, 7.0, 60.0), sleep=sleeps.append, limit=1)
        self.assertEqual(done["generated"], 1)
        self.assertEqual(sleeps, [7.0, 7.0])
        self.assertEqual(len(self.calls()), 3)
        self.assertEqual(next(iter(self.records("neutral").values()))["attempts"], 3)

    def test_retries_are_bounded_and_the_error_says_the_memory_never_freed(self):
        self.set_plan(["oom"] * 10)
        with self.assertRaisesRegex(S.GenerationError, "out of memory.*3 attempts"):
            self.generate(retry=S.Retry(2, 0.0, 60.0))
        self.assertEqual(len(self.calls()), 3)
        self.assertEqual([f for f in os.listdir(os.path.join(self.out, "neutral")) if f.endswith(".json")], [])

    def test_any_other_failure_aborts_at_once_with_the_stderr(self):
        self.set_plan(["fail"])
        with self.assertRaisesRegex(S.GenerationError, "unsupported adapter layout"):
            self.generate()
        self.assertEqual(len(self.calls()), 1)

    def test_success_without_a_valid_output_is_a_failure_not_a_frame(self):
        self.set_plan(["empty"])
        with self.assertRaisesRegex(S.GenerationError, "no valid output"):
            self.generate()
        self.assertEqual([f for f in os.listdir(os.path.join(self.out, "neutral")) if f.endswith(".json")], [])

    def test_a_missing_brain_binary_is_a_clear_error(self):
        with self.assertRaisesRegex(S.GenerationError, "cannot run"):
            self.generate(brain=self.path("no-such-brain"))


class Cli(World):
    def args(self, *extra):
        return ["generate", "--splits", self.splits, "--config", self.path("synth.json"), "--brain", self.brain,
                "--out", self.out, "--modes", "neutral,prior", "--captions-report", self.path("captions-report.json"),
                "--retry-sleep", "0", *extra]

    def test_generate_runs_every_mode_for_the_split_and_summarises(self):
        self.assertEqual(S.main(self.args()), 0)
        self.assertEqual(len(self.calls()), 6)
        self.assertEqual(sorted(os.listdir(self.out)), ["neutral", "prior", "progress.log"])

    def test_a_failure_is_a_nonzero_exit_with_the_reason_on_stderr(self):
        self.set_plan(["fail"])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(S.main(self.args()), 1)
        self.assertIn("unsupported adapter layout", err.getvalue())

    def test_the_prior_mode_needs_the_caption_report(self):
        args = self.args()
        i = args.index("--captions-report")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            S.main(args[:i] + args[i + 2:])
        self.assertIn("--captions-report", err.getvalue())

    def test_the_device_flag_overrides_the_config(self):
        S.main(self.args("--device", "gpu0", "--limit", "1", "--modes", "neutral"))
        self.assertEqual(self.calls()[0][1], "gpu0")


if __name__ == "__main__":
    unittest.main()
