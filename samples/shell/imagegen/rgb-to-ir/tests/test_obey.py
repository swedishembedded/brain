# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements instruction-following measurement for
# image-to-image translators for its clients. If your team needs expertise
# in controllable generation or in testing whether a generator does what it
# is told, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Spec tests for rir_obey: direction, controllability, the luminance
shortcut and fidelity, on fake generators that obey, ignore or cheat."""
import json
import os
import secrets
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402,F401  (sets the OpenCV log level before cv2 loads)
import cv2  # noqa: E402
import rir_obey as O  # noqa: E402
import rir_regions as R  # noqa: E402

H = W = 160
SIGN = {"warmer": +1.0, "cooler": -1.0, "same": 0.0}
FLIP = {"warmer": "cooler", "cooler": "warmer"}
STEP = 30.0  # grey levels an obeying generator moves the object by


def box_mask(seed):
    rng = np.random.default_rng(seed)
    x, y = (int(v) for v in rng.integers(10, 40, 2))
    m = np.zeros((H, W), bool)
    m[y:y + 36, x:x + 36] = True
    return m


def source_tile(seed, mask, object_brightness):
    """RGB (BGR) tile: textured background, object of the given grey level."""
    rng = np.random.default_rng(seed)
    gray = np.clip(100 + rng.normal(0, 6, (H, W)), 0, 255)
    gray[mask] = object_brightness
    g = gray.astype(np.uint8)
    return np.dstack([g, g, g])


def background_ir(seed):
    return np.clip(110 + np.random.default_rng(seed).normal(0, 3, (H, W)), 0, 255)


def paint(base, mask, delta):
    out = base.copy()
    out[mask] += delta
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class Generators:
    """Fake translators: (tile seed, rgb, mask, instruction) -> generated IR."""

    @staticmethod
    def obeying(seed, rgb, mask, instruction):
        return paint(background_ir(seed), mask, STEP * SIGN[instruction])

    @staticmethod
    def ignoring(seed, rgb, mask, instruction):
        # decides warm or cool from its own coin, whatever it is told (the instructions cycle with seed % 2)
        return paint(background_ir(seed), mask, STEP * (1 if (seed // 2) % 2 else -1))

    @staticmethod
    def luminance(seed, rgb, mask, instruction):
        # object warm iff it is bright in the RGB: the shortcut a luminance-copying generator takes
        bright = float(rgb[..., 0][mask].mean()) - float(rgb[..., 0][~mask].mean())
        return paint(background_ir(seed), mask, 0.5 * bright)


def make_pairs(generator, n=60, per_sequence=3, source="oracle", instructions=("warmer", "cooler"), counterfactual=True):
    pairs = []
    for i in range(n):
        mask = box_mask(i)
        # brightness varies independently of the instruction, which is cycled
        brightness = 100 + (60 if (i // 2) % 2 else -60) * (0.5 + (i % 5) / 8)
        rgb = source_tile(1000 + i, mask, brightness)
        instruction = instructions[i % len(instructions)]
        cf = None
        if counterfactual and instruction in FLIP:
            cf = generator(i, rgb, mask, FLIP[instruction])
        pairs.append(O.ObeyPair(sequence=f"s{i // per_sequence}", region_type="person", instruction=instruction, source=source,
                                generated=generator(i, rgb, mask, instruction), mask=mask, rgb=rgb, counterfactual=cf,
                                key=f"t{i}"))
    return pairs


def evaluate(pairs, **kw):
    # seed: the interval bounds asserted by these tests were tuned on this resample draw
    return O.evaluate(pairs, n_resamples=200, seed=1, **kw)


def person(report, source="oracle"):
    return report["sources"][source]["region_types"]["person"]


class Obeying(unittest.TestCase):
    def setUp(self):
        self.report = evaluate(make_pairs(Generators.obeying))
        self.person = person(self.report)

    def test_direction_accuracy_is_high(self):
        d = self.person["direction_accuracy"]
        self.assertGreater(d["estimate"], 0.95)
        self.assertGreater(d["ci"][0], 0.85)
        self.assertEqual(d["n"], 60)

    def test_controllability_is_the_signed_gap_between_the_two_generations(self):
        c = self.person["controllability"]
        # warm minus cool is about 2 x STEP grey levels, in units of the frame MAD (about 3 levels at most)
        self.assertGreater(c["estimate"], 5.0)
        self.assertGreater(c["ci"][0], 0.0)
        self.assertEqual(c["n"], 60)

    def test_the_effect_k5_reads_is_the_controllability_and_excludes_zero(self):
        e = self.person["effect"]
        self.assertEqual(e["measure"], "controllability")
        self.assertGreater(e["ci"][0], 0.0)

    def test_the_luminance_shortcut_is_absent(self):
        s = self.person["luminance_shortcut"]
        self.assertLess(abs(s["estimate"]), 0.4)
        self.assertLess(s["ci"][0], 0.4)

    def test_it_is_faithful_when_the_instruction_is_the_truth(self):
        pairs = make_pairs(Generators.obeying)
        for i, p in enumerate(pairs):  # the real IR of the tile: the same physics, other noise
            p.real_ir = paint(background_ir(5000 + i), p.mask, STEP * SIGN[p.instruction])
        f = person(evaluate(pairs))["fidelity"]
        self.assertLess(f["estimate"], 1.5)


class Ignoring(unittest.TestCase):
    def test_a_generator_that_ignores_the_instruction_has_no_controllability_and_chance_direction(self):
        p = person(evaluate(make_pairs(Generators.ignoring)))
        self.assertLessEqual(p["controllability"]["ci"][0], 0.0)
        self.assertGreaterEqual(p["controllability"]["ci"][1], 0.0)
        self.assertAlmostEqual(p["direction_accuracy"]["estimate"], 0.5, delta=0.2)
        self.assertLessEqual(p["effect"]["ci"][0], 0.0)

    def test_without_counterfactuals_the_effect_falls_back_to_direction_over_chance(self):
        p = person(evaluate(make_pairs(Generators.obeying, counterfactual=False)))
        self.assertIsNone(p["controllability"])  # absent, not zero
        self.assertEqual(p["effect"]["measure"], "direction")
        self.assertGreater(p["effect"]["estimate"], 0.4)  # accuracy 1.0 less chance 0.5
        self.assertGreater(p["effect"]["ci"][0], 0.0)


class Cheating(unittest.TestCase):
    def test_a_generator_that_copies_luminance_shows_a_strong_partial_correlation(self):
        p = person(evaluate(make_pairs(Generators.luminance)))
        self.assertGreater(p["luminance_shortcut"]["estimate"], 0.8)
        self.assertGreater(p["luminance_shortcut"]["ci"][0], 0.6)
        self.assertLess(p["direction_accuracy"]["estimate"], 0.8)  # right only when luminance happens to agree

    def test_the_partial_correlation_controls_for_the_instruction(self):
        # c_gen follows the instruction and so does the RGB contrast; once the instruction is held fixed nothing is left.
        # seed: the 0.2 bound is about three standard errors of a null partial correlation on 200 samples and was checked on this draw
        rng = np.random.default_rng(0)
        instruction = np.array(["warmer", "cooler"] * 100)
        sign = np.where(instruction == "warmer", 1.0, -1.0)
        c_gen, c_rgb = sign * 5 + rng.normal(0, 1, 200), sign * 3 + rng.normal(0, 1, 200)
        self.assertGreater(np.corrcoef(c_gen, c_rgb)[0, 1], 0.8)
        self.assertLess(abs(O.partial_correlation(c_gen, c_rgb, instruction, np.ones(200))), 0.2)


class SameBand(unittest.TestCase):
    def test_same_is_obeyed_inside_the_noise_floor_and_broken_outside_it(self):
        flat = make_pairs(Generators.obeying, n=30, instructions=("same",), counterfactual=False)
        # tau is the 75th percentile of the null, so even a perfect "same" is read as same only 3 times in 4
        self.assertGreater(person(evaluate(flat))["direction_accuracy"]["estimate"], 0.5)
        # a generator that warms the object whatever it is told
        pairs = make_pairs(Generators.obeying, n=30, instructions=("same",), counterfactual=False)
        for p in pairs:
            p.generated = paint(background_ir(int(p.key[1:])), p.mask, STEP)
        self.assertLess(person(evaluate(pairs))["direction_accuracy"]["estimate"], 0.1)

    def test_an_explicit_tau_defines_the_band(self):
        pairs = make_pairs(Generators.obeying, n=20, instructions=("warmer",), counterfactual=False)
        wide = [O.ObeyPair(**{**p.__dict__, "tau": 100.0}) for p in pairs]  # any contrast reads as "same"
        self.assertEqual(person(evaluate(wide))["direction_accuracy"]["estimate"], 0.0)


class Sources(unittest.TestCase):
    def test_oracle_and_prior_instructions_are_reported_apart(self):
        oracle = make_pairs(Generators.obeying, n=30, source="oracle")
        prior = make_pairs(Generators.obeying, n=30, source="prior")
        report = evaluate(oracle + prior)
        self.assertEqual(sorted(report["sources"]), ["oracle", "prior"])
        self.assertEqual(report["sources"]["oracle"]["n_pairs"], 30)
        self.assertEqual(report["sources"]["prior"]["n_pairs"], 30)

    def test_a_prior_that_contradicts_the_real_ir_is_obeyed_but_unfaithful(self):
        # The generator obeys the (wrong) prior; the real IR says the opposite.
        pairs = make_pairs(Generators.obeying, n=40, source="prior", counterfactual=False)
        for p in pairs:
            p.c_real = -SIGN[p.instruction] * 8.0
        f = person(evaluate(pairs), "prior")
        self.assertGreater(f["direction_accuracy"]["estimate"], 0.95)
        self.assertGreater(f["fidelity"]["estimate"], 8.0)

    def test_class_priors_are_the_modal_polarity_of_the_caption_report(self):
        report = {"classes": {"person": {"warmer": 442, "cooler": 52, "same": 462}, "car": {"warmer": 304, "cooler": 45, "same": 827},
                              "bicycle": {"warmer": 38, "cooler": 203, "same": 145}}}
        self.assertEqual(O.class_priors(report), {"person": "same", "car": "same", "bicycle": "cooler"})


class ForcedInstructions(unittest.TestCase):
    """The instruction is chosen by the experimenter, not measured: forced wording, training or held-out."""

    def test_forced_instructions_are_a_source_of_their_own_apart_from_held_out_wording(self):
        report = evaluate(make_pairs(Generators.obeying, n=30, source="forced")
                          + make_pairs(Generators.obeying, n=30, source="forced_heldout"))
        self.assertEqual(sorted(report["sources"]), ["forced", "forced_heldout"])
        self.assertGreater(person(report, "forced_heldout")["controllability"]["ci"][0], 0.0)

    def test_the_fraction_of_renders_within_the_noise_floor_separates_a_flat_render_from_an_obeyed_one(self):
        flat = make_pairs(Generators.obeying, n=30, source="forced", counterfactual=False)
        for p in flat:
            p.generated = paint(background_ir(int(p.key[1:])), p.mask, 0.0)
        obeyed = make_pairs(Generators.obeying, n=30, source="forced", counterfactual=False)
        self.assertLess(person(evaluate(obeyed), "forced")["within_noise_floor"]["estimate"], 0.1)
        self.assertGreater(person(evaluate(flat), "forced")["within_noise_floor"]["estimate"], 0.5)
        self.assertEqual(person(evaluate(flat), "forced")["within_noise_floor"]["n"], 30)


class Strata(unittest.TestCase):
    def test_day_and_night_are_reported_apart_and_a_stratum_where_nothing_is_obeyed_shows_it(self):
        day = make_pairs(Generators.obeying, n=40, source="forced")
        night = make_pairs(Generators.ignoring, n=40, source="forced")
        for p in day:
            p.stratum = "day"
        for p in night:
            p.stratum, p.sequence = "night", "n" + p.sequence
        report = person(evaluate(day + night), "forced")
        self.assertEqual(sorted(report["strata"]), ["day", "night"])
        self.assertGreater(report["strata"]["day"]["controllability"]["ci"][0], 0.0)
        self.assertLessEqual(report["strata"]["night"]["controllability"]["ci"][0], 0.0)
        self.assertEqual(report["strata"]["night"]["n"], 40)
        self.assertEqual(report["n"], 80)

    def test_pairs_without_a_stratum_have_no_strata_entry(self):
        self.assertNotIn("strata", person(evaluate(make_pairs(Generators.obeying, n=10))))


class Neutral(unittest.TestCase):
    def test_the_shift_from_the_neutral_render_shows_which_direction_the_instruction_actually_moves(self):
        # warmer is obeyed (the object is warmed above the neutral render); cooler is not (it equals the neutral render)
        pairs = make_pairs(Generators.obeying, n=40, source="forced")
        for i, p in enumerate(pairs):
            p.neutral = paint(background_ir(i), p.mask, 0.0)
            if p.instruction == "cooler":
                p.generated = p.neutral
        r = person(evaluate(pairs), "forced")["neutral_shift"]
        self.assertGreater(r["warmer"]["ci"][0], 1.0)
        self.assertAlmostEqual(r["cooler"]["estimate"], 0.0, delta=0.3)
        self.assertEqual((r["warmer"]["n"], r["cooler"]["n"]), (20, 20))

    def test_without_a_neutral_render_there_is_no_shift(self):
        self.assertIsNone(person(evaluate(make_pairs(Generators.obeying, n=10)))["neutral_shift"])


class FidelityAgainstTheRealIr(unittest.TestCase):
    def test_the_oracle_fidelity_keeps_only_renders_whose_instruction_is_what_the_real_ir_shows(self):
        pairs = make_pairs(Generators.obeying, n=40, source="forced", counterfactual=False)
        for i, p in enumerate(pairs):
            # the real IR shows an object that is warmer, whatever the instruction says
            p.real_ir = paint(background_ir(5000 + i), p.mask, STEP)
        r = person(evaluate(pairs), "forced")
        self.assertEqual(r["fidelity"]["n"], 40)
        self.assertEqual(r["oracle_fidelity"]["n"], 20)  # the warmer half
        self.assertLess(r["oracle_fidelity"]["estimate"], 1.5)
        self.assertGreater(r["fidelity"]["estimate"], r["oracle_fidelity"]["estimate"] + 2.0)

    def test_the_neutral_render_has_its_own_fidelity(self):
        pairs = make_pairs(Generators.obeying, n=30, source="forced", counterfactual=False)
        for i, p in enumerate(pairs):
            p.real_ir = paint(background_ir(5000 + i), p.mask, STEP)
            p.neutral = paint(background_ir(i), p.mask, 0.0)
        r = person(evaluate(pairs), "forced")["neutral_fidelity"]
        self.assertGreater(r["estimate"], 5.0)  # an unprompted render misses a warm object by about STEP / MAD
        self.assertEqual(r["n"], 30)


class Plumbing(unittest.TestCase):
    def test_the_contrast_is_the_one_of_rir_regions(self):
        with mock.patch.object(R, "object_contrast", wraps=R.object_contrast) as spy:
            evaluate(make_pairs(Generators.obeying, n=4))
        self.assertGreater(spy.call_count, 0)

    def test_a_region_too_small_to_measure_is_skipped_and_counted_not_scored(self):
        pairs = make_pairs(Generators.obeying, n=10)
        tiny = np.zeros((H, W), bool)
        tiny[5:7, 5:7] = True
        pairs[0].mask = tiny
        report = evaluate(pairs)
        self.assertEqual(report["sources"]["oracle"]["n_skipped"], {"mask_or_ring_too_small": 1})
        self.assertEqual(person(report)["direction_accuracy"]["n"], 9)

    def test_without_room_for_a_noise_floor_the_direction_is_absent_not_wrong(self):
        big = np.zeros((120, 120), bool)
        big[10:110, 10:110] = True
        ir = paint(np.full((120, 120), 110.0), big, STEP)
        pair = O.ObeyPair("s0", "person", "warmer", "oracle", ir, big, key="t")
        # seed: the resamples of the single pair's interval are read off this stream
        p = person(O.evaluate([pair], n_resamples=50, seed=1))
        self.assertIsNone(p["direction_accuracy"])
        self.assertIsNone(p["effect"])
        self.assertEqual(p["n"], 1)

    def test_resampling_is_by_sequence_so_near_duplicates_do_not_narrow_the_interval(self):
        base = make_pairs(Generators.obeying, n=30, per_sequence=1)
        leaked = base + [O.ObeyPair(**{**p.__dict__}) for p in base]  # a copy of every pair in its own sequence
        w1 = person(evaluate(base))["controllability"]["ci"]
        w2 = person(evaluate(leaked))["controllability"]["ci"]
        self.assertAlmostEqual((w2[1] - w2[0]) / (w1[1] - w1[0]), 1.0, delta=0.15)

    def test_the_report_is_plain_json_and_the_cli_reproduces_it(self):
        tmp = tempfile.mkdtemp()
        seed = secrets.randbits(63)
        pairs = make_pairs(Generators.obeying, n=12)
        rows = []
        for i, p in enumerate(pairs):
            files = {}
            p.neutral = p.generated
            for name, img in (("generated", p.generated), ("rgb", p.rgb), ("counterfactual", p.counterfactual),
                              ("neutral", p.neutral), ("mask", p.mask.astype(np.uint8) * 255)):
                if img is not None:
                    files[name] = f"{name}{i}.png"
                    cv2.imwrite(os.path.join(tmp, files[name]), img)
            rows.append({"sequence": p.sequence, "region_type": p.region_type, "instruction": p.instruction, "source": p.source,
                         "key": p.key, "stratum": "day", **files})
        with open(os.path.join(tmp, "pairs.jsonl"), "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
        out = os.path.join(tmp, "obedience.json")
        self.assertEqual(O.main([os.path.join(tmp, "pairs.jsonl"), "--out", out, "--resamples", "200", "--seed", str(seed)]), 0)
        with open(out) as fh:
            cli = json.load(fh)
        for p in pairs:
            p.stratum = "day"
        direct = json.loads(json.dumps(O.evaluate(pairs, n_resamples=200, seed=seed)))
        self.assertEqual(cli["sources"]["oracle"]["region_types"]["person"]["direction_accuracy"],
                         direct["sources"]["oracle"]["region_types"]["person"]["direction_accuracy"])
        self.assertEqual(cli["sources"]["oracle"]["region_types"]["person"]["strata"]["day"]["neutral_shift"],
                         direct["sources"]["oracle"]["region_types"]["person"]["strata"]["day"]["neutral_shift"])


if __name__ == "__main__":
    unittest.main()
